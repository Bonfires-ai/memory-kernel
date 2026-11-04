"""The whole conversation under G1: one event graph over every captured unit, plus its census.

Ported from a reference-harness module (unreached top-level definitions pruned). Differences: run_whole_picks takes a
ledger.Shard (its on_new_send is the picks meter), `root` and the QuestionBatcher; build_graph takes `root`.

Per unit:

* frames — stock PropBank frames when the unit parsed at the serving budget; clause-split frames
  (``clauses-004``, then ``clauses-002``) when it was cut off; nothing for empty units;
* G1 layers — ``argm-tmp`` time roles, pronoun referents (Jev pick, expletive clause or OPEN; a
  nearest-NP candidate that Jev did not pick is never kept), speaker deixis (I/me/my → the speaker,
  we/us/our → the speaker and others, you/your → the other participant of the session);
* card firing — an event whose roleset is a g1 card's ``discriminant`` carries that card and its meaning
  rows (which question shapes it answers, through which roles).

Nothing here asserts world facts or canonical identity; it is a retrieval-facing evidence graph.
The whole-conversation antecedent picks share the round's USD2 ledger.
"""

import json
from collections import Counter

from memory_kernel.construction.memory_jev_archive import read_jev_archive
from memory_kernel.construction.memory_semantic_jev import seal
from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256
from memory_kernel.grammar.pronouns import of_leaf
from memory_kernel.graph.addressing import addressee_of, is_explicit, is_explicit_at
from memory_kernel.pipeline.ledger import write_new
from memory_kernel.pipeline.stages._antecedent_pick import pick_questions, read_picks
from memory_kernel.pipeline.stages._clause_choice import (
    _clause_index,
    _referent_context,
    capture_context,
    clause_candidates,
    enrich_frames,
)

SCHEMA = "mk-g1-graph/v1"

KEPT_REFERENTS = {"picked", "expletive", "open", "deixis"}


def deixis_referent(word, lex_class, *, speaker, other):
    """Speaker or addressee for a one-token role, when the capture leaf's PTB tag (``lex_class``) says it is a pronoun."""
    pronoun = of_leaf(word, lex_class)
    if pronoun is None:
        return None
    if pronoun.person == 1:
        return {"status": "deixis", "text": f"{speaker} and others" if pronoun.plural else speaker}
    if pronoun.person == 2 and other:
        return {"status": "deixis", "text": other}
    return None


def event_rows(frames, tokens, *, speaker, other, cards, origin="stock", tags=None):
    """``tags``: token index -> the capture leaf's PTB tag; a token with none is never a deixis."""
    tags = tags or {}
    events = []
    for frame in frames:
        roles = {}
        for element in frame["frameElements"]:
            span = list(element["indices"])
            row = {"text": " ".join(tokens[i] for i in span if i in tokens), "span": span}
            referent = element.get("referent")
            if referent is not None:
                row["referent"] = referent if referent["status"] in KEPT_REFERENTS else {"status": "open"}
            elif len(span) == 1 and span[0] in tokens:
                deixis = deixis_referent(tokens[span[0]], tags.get(span[0]), speaker=speaker, other=other)
                if deixis:
                    row["referent"] = deixis
            if "relabelled_from" in element:
                row["relabelled_from"] = element["relabelled_from"]
            roles[element["feRole"]] = row
        card = cards.get(frame["frameName"])
        events.append(
            {
                "roleset": frame["frameName"],
                "origin": origin,
                "anchor": list(frame["frameEvokingElement"]["indices"]),
                "roles": roles,
                "card": card["construct_id"] if card else None,
                "meaning": card["meaning"] if card else [],
            }
        )
    return events


def form_frames(cards, nodes):
    """Fire form-bearing cards: anchor lemma + a dependency path (labels joined by ">") per role; the role
    filler is the path end's dependency subtree. A card fires only when every pattern role is found."""
    leaves = sorted((n for n in nodes if n["node_type"] == "leaf"), key=lambda n: n["start"])
    by_id = {leaf["node_id"]: leaf["start"] for leaf in leaves}
    heads = {leaf["start"]: by_id.get(leaf.get("dependency_head")) for leaf in leaves}
    children = {}
    for leaf in leaves:
        if heads[leaf["start"]] != leaf["start"]:
            children.setdefault(heads[leaf["start"]], []).append(leaf)

    def subtree(i):
        out, stack = set(), [i]
        while stack:
            j = stack.pop()
            if j not in out:
                out.add(j)
                stack.extend(c["start"] for c in children.get(j, []))
        return sorted(out)

    frames = []
    for card in cards:
        form = card.get("form") or {}
        if not form.get("anchor_lemma") or not form.get("pattern"):
            continue
        for anchor in leaves:
            if (anchor.get("lemma") or "").lower() != form["anchor_lemma"].lower():
                continue
            elements = []
            for step in form["pattern"]:
                current = anchor["start"]
                for label in step["dependency_path"].split(">"):
                    current = next(
                        (c["start"] for c in children.get(current, []) if c.get("dependency_label") == label.strip()),
                        None,
                    )
                    if current is None:
                        break
                if current is None:
                    elements = None
                    break
                elements.append({"feRole": step["role"], "feName": step["role"], "indices": subtree(current)})
            if elements:
                frames.append(
                    {
                        "frameName": card["predicate"],
                        "frameEvokingElement": {"indices": [anchor["start"]]},
                        "frameElements": elements,
                    }
                )
    return frames


def _units(capture, source):
    directory = capture / f"{source['order']:04d}"
    for path in sorted(directory.glob("unit-[0-9][0-9].json")):
        yield str(path.relative_to(capture)), json.loads(path.read_bytes())


def _frames(record, clause_index):
    nodes = record["job"]["nodes"]
    leaves = sorted((n for n in nodes if n["node_type"] == "leaf"), key=lambda n: n["start"])
    tokens = {leaf["start"]: leaf["string"] for leaf in leaves}
    if record["capture_status"] == "framed":
        return nodes, tokens, record["response"]["payload"]["frames"], "stock"
    if record["capture_status"] == "cutoff" and record["content_sha256"] in clause_index:
        run, unit = clause_index[record["content_sha256"]]
        pair_unit = {
            "tokens": [{"index": leaf["start"], "text": leaf["string"], "lemma": leaf.get("lemma")} for leaf in leaves]
        }
        return nodes, tokens, clause_candidates(run, unit, pair_unit), "clause"
    return nodes, tokens, [], "none"


def _others(sources):
    by_session = {}
    for source in sources:
        by_session.setdefault(source["session_id"], set()).add(source["speaker"])
    return {s: speakers for s, speakers in by_session.items()}


def collect_units(capture, sources, sources_by_order, clause_index, picks=None):
    """Every captured unit with its (enriched) frames; ``picks`` = {path: {pronoun index: referent}}."""
    for source in sources:
        for path, record in _units(capture, source):
            nodes, tokens, frames, origin = _frames(record, clause_index)
            if frames:
                context = _referent_context(capture, path, sources_by_order)
                frames = enrich_frames(frames, nodes, context, None, picks=(picks or {}).get(path))
            yield source, path, record, nodes, tokens, frames, origin


def run_whole_picks(shard, runs, out, *, root, batcher, live=False, plan=False):
    """Jev antecedent picks for every candidate pronoun role in the conversation.

    ``shard`` is a ``ledger.Shard`` (its ``on_new_send`` is the stage meter); ``root`` is the base a relative
    ``manifest["capture"]["path"]`` is read against (an absolute one wins)."""
    capture, _, sources_by_order = capture_context(runs[0], root)
    sources = [sources_by_order[o] for o in sorted(sources_by_order)]
    clause_index = _clause_index(runs)
    manifest, provenance, _ = shard.load()
    (out / "picks").mkdir(parents=True, exist_ok=True)
    ledger = None if plan else shard.open(manifest, provenance)
    totals = {"messages": 0, "questions": 0, "batches": 0, "picked": 0, "open": 0}
    by_message = {}
    for source, path, record, nodes, _tokens, frames, _origin in collect_units(
        capture, sources, sources_by_order, clause_index
    ):
        if frames:
            by_message.setdefault(source["order"], []).append((path, record, nodes, frames))
    try:
        for order, units in sorted(by_message.items()):
            context_by_unit, pair_units = {}, []
            for number, (path, record, nodes, frames) in enumerate(units):
                leaves = sorted((n for n in nodes if n["node_type"] == "leaf"), key=lambda n: n["start"])
                context = _referent_context(capture, path, sources_by_order)
                for item in context:
                    words = sorted((n for n in item["nodes"] if n["node_type"] == "leaf"), key=lambda n: n["start"])
                    item["text"] = " ".join(w["string"] for w in words)
                context_by_unit[number] = (nodes, context, path)
                pair_units.append(
                    {
                        "unit_id": number,
                        "text": record["job"]["source"]["text"],
                        "record_sha256": record["content_sha256"],
                        "tokens": [{"index": leaf["start"], "text": leaf["string"]} for leaf in leaves],
                        "parsed_evidence": {"native_frame_candidates": frames},
                    }
                )
            pair = {"content_sha256": sources_by_order[order]["content_sha256"], "body": {"units": pair_units}}
            compiled = pick_questions(
                pair, lambda unit, units=context_by_unit: units[unit["unit_id"]][:2], batcher=batcher
            )
            if compiled is None:
                continue
            totals["messages"] += 1
            totals["questions"] += len(compiled["targets"])
            totals["batches"] += len(compiled["batches"])
            if plan:
                continue
            answers = {}
            for wire in compiled["batches"]:
                cached = (ledger.directory / ("jev-" + content_sha256(wire)) / "archive.json").exists()
                if not live and not cached:
                    raise ValueError("missing whole-conversation pick archive; live authorization required")
                answers.update(read_jev_archive(wire, ledger.evaluate(wire), ledger.policy)["answers"])
            picks = read_picks(compiled["targets"], answers)
            by_path = {context_by_unit[u][2]: {str(i): r for i, r in p.items()} for u, p in picks.items()}
            for unit_picks in by_path.values():
                for referent in unit_picks.values():
                    totals["picked" if referent["status"] == "picked" else "open"] += 1
            result = seal(
                {
                    "schema_version": SCHEMA + "#picks",
                    "order": order,
                    "targets": compiled["targets"],
                    "answers": answers,
                    "picks": by_path,
                }
            )
            path = out / "picks" / f"{order:04d}.json"
            if path.exists():
                if path.read_bytes() != canonical_json(result).encode():
                    raise ValueError(f"cached pick result changed: {path}")
            else:
                write_new(path, result)
        return {**totals, "budget": None if ledger is None else ledger.summary()}
    finally:
        if ledger is not None:
            ledger.close()


def _load_whole_picks(out):
    picks = {}
    for path in sorted((out / "picks").glob("*.json")):
        for unit_path, unit_picks in json.loads(path.read_bytes())["picks"].items():
            picks[unit_path] = {int(i): r for i, r in unit_picks.items()}
    return picks


def explicit_flags(sources, *, append=False, explicit_since=None):
    """``{message_id: explicit}``: whether "you" resolves through the message's addressees. A full build: the whole
    conversation's ``is_explicit`` for every message (1.1, unchanged). A forward append (``append``): per message,
    bonfire-wide (``is_explicit_at``) from the chain's ``explicit_since``, else the session's first addressed message."""
    if not append:
        return dict.fromkeys((s["message_id"] for s in sources), is_explicit(sources))
    since = explicit_since
    if since is None:
        since = min((s["order"] for s in sources if s.get("addressees")), default=None)
    return {s["message_id"]: is_explicit_at(s["order"], since) for s in sources}


def build_graph(runs, inventory, out, *, root, append=False, explicit_since=None):
    """``append`` / ``explicit_since``: a forward append decides "you" per message (``explicit_flags``)."""
    capture, _, sources_by_order = capture_context(runs[0], root)
    sources = [sources_by_order[o] for o in sorted(sources_by_order)]
    clause_index = _clause_index(runs)
    all_cards = json.loads(inventory.read_bytes())["cards"]
    cards = {c["discriminant"]: c for c in all_cards if c["discriminant"]}
    form_cards = [c for c in all_cards if c.get("form")]
    cards.update({c["predicate"]: c for c in form_cards})
    speakers = _others(sources)
    explicit_of = explicit_flags(sources, append=append, explicit_since=explicit_since)
    picks = _load_whole_picks(out)
    records = []
    for source, path, record, _nodes, tokens, frames, origin in collect_units(
        capture, sources, sources_by_order, clause_index, picks
    ):
        explicit = explicit_of[source["message_id"]]
        other = addressee_of(
            source.get("addressees", []), source["speaker"], speakers[source["session_id"]], explicit=explicit
        )
        tags = {n["start"]: n.get("lex_class") for n in _nodes if n["node_type"] == "leaf"}
        events = event_rows(
            frames, tokens, speaker=source["speaker"], other=other, cards=cards, origin=origin, tags=tags
        )
        events += event_rows(
            form_frames(form_cards, _nodes),
            tokens,
            speaker=source["speaker"],
            other=other,
            cards=cards,
            origin="g1_form",
            tags=tags,
        )
        records.append(
            {
                "message_id": source["message_id"],
                "session_id": source["session_id"],
                "speaker": source["speaker"],
                **({"addressees": list(source.get("addressees", []))} if explicit else {}),
                "timestamp": source["timestamp"],
                "unit": path,
                "capture_status": record["capture_status"],
                "text": record["job"]["source"]["text"],
                "events": events,
            }
        )
    graph = seal(
        {
            "schema_version": SCHEMA,
            "runs": [r.name for r in runs],
            "inventory": str(inventory),
            "inventory_sha256": json.loads(inventory.read_bytes())["content_sha256"],
            "records": records,
            "census": census(records),
        }
    )
    path = out / "graph.json"
    if path.exists() and path.read_bytes() != canonical_json(graph).encode():
        raise ValueError(f"cached graph changed: {path}; write to a new directory")
    if not path.exists():
        write_new(path, graph)
    return graph["census"]


def census(records):
    c = Counter()
    rolesets, cards, referents, roles, per_session = Counter(), Counter(), Counter(), Counter(), Counter()
    for record in records:
        c["units"] += 1
        c[f"units_{record['capture_status']}"] += 1
        c["units_with_events"] += bool(record["events"])
        for event in record["events"]:
            c["events"] += 1
            c[f"events_{event['origin']}"] += 1
            rolesets[event["roleset"]] += 1
            per_session[record["session_id"]] += 1
            if event["card"]:
                c["card_firings"] += 1
                cards[event["card"]] += 1
            for role, row in event["roles"].items():
                roles[role] += 1
                if "relabelled_from" in row:
                    c["time_relabels"] += 1
                if "referent" in row:
                    referents[row["referent"]["status"]] += 1
    return {
        **dict(c),
        "distinct_rolesets": len(rolesets),
        "top_rolesets": rolesets.most_common(15),
        "card_firings_by_card": cards.most_common(),
        "roles": dict(roles.most_common()),
        "referents": dict(referents),
        "events_per_session": dict(sorted(per_session.items())),
    }
