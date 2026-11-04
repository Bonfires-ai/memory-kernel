"""Clause-split frames of the round's cutoff units, with time roles and pronoun referents, for the G1 graph.

Ported from a reference-harness module (unreached top-level definitions pruned). Differences: capture_context takes
`root` (the base of manifest['capture']['path']; it was the stock capture runner's ROOT).

The reference step asked Jev, per clause-split frame and per role, whether it supports the answer; that step is not
ported. What is left are the helpers ``_g1_graph`` reads:

* ``clause_candidates``: a unit's clause frames as native-candidate rows, every index mapped back to the unit's
  ORIGINAL token positions (placeholder roles expand to the clause they stand for);
* ``capture_context``: the base capture a clause-split run was cut from, its record index and its sources;
* ``enrich_frames``: time roles (``argm-tmp``) and pronoun referents layered onto frames; ``_referent_context``
  gives the units a pronoun's antecedent may come from.
"""

import json
from pathlib import Path

from memory_kernel.pipeline.stages._clauses import _map_frames
from memory_kernel.pipeline.stages._referents import pronoun_roles, resolve_pronoun
from memory_kernel.pipeline.stages._time_role import attach_time, time_spans


def _clause_index(runs):
    """record_sha256 -> (run dir, manifest unit) over the clause runs, later runs filling earlier gaps."""
    index = {}
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_bytes())
        for unit in manifest["units"]:
            if unit["segments"] and unit["record_sha256"] not in index:
                index[unit["record_sha256"]] = (run, unit)
    return index


def clause_candidates(run, unit, pair_unit):
    """Clause frames of one unit as native-candidate rows over the pair unit's original tokens."""
    tokens = {t["index"]: t for t in pair_unit["tokens"]}
    out = []
    for number, segment in enumerate(unit["segments"]):
        row = json.loads((run / unit["path"].replace(".json", f"-s{number:02d}.json")).read_bytes())
        for name, anchor, roles in _map_frames(segment, row):
            if not anchor or not anchor <= set(tokens):
                continue
            elements = []
            for role, span in sorted(roles.items()):
                span = sorted(i for i in span if i in tokens)
                if span:
                    elements.append(
                        {
                            "feName": role,
                            "feRole": role,
                            "indices": span,
                            "feString": " ".join(tokens[i]["text"] for i in span),
                            "feLemmas": [tokens[i].get("lemma") for i in span],
                        }
                    )
            anchor = sorted(anchor)
            out.append(
                {
                    "frameName": name,
                    "id": f"clause{number:02d}-{name}-{anchor[0]}",
                    "frameEvokingElement": {
                        "indices": anchor,
                        "lemma": tokens[anchor[0]].get("lemma"),
                        "felString": " ".join(tokens[i]["text"] for i in anchor),
                    },
                    "frameElements": elements,
                }
            )
    seen, unique = set(), []
    for frame in out:
        key = (frame["frameName"], tuple(frame["frameEvokingElement"]["indices"]))
        if key not in seen:
            seen.add(key)
            unique.append(frame)
    return unique


#: Off: nearest-NP antecedents from the previous message (usually the other speaker's turn)
#: were mostly wrong on the v2 plan ("It -> your family", "it -> summer break"); OPEN beats a guess.
PREVIOUS_MESSAGE_ANTECEDENTS = False


def capture_context(run, root):
    """The base capture a clause-split run was cut from, its record index and its sources by order."""
    capture = root / json.loads((run / "manifest.json").read_bytes())["capture"]["path"]
    index_by_sha = {
        json.loads(p.read_bytes())["content_sha256"]: str(p.relative_to(capture))
        for p in capture.glob("[0-9][0-9][0-9][0-9]*/unit-[0-9][0-9].json")
    }
    sources_by_order = {s["order"]: s for s in json.loads((capture / "sources.json").read_bytes())}
    return capture, index_by_sha, sources_by_order


def _base_nodes(capture, record_sha256, index_by_sha):
    path = index_by_sha[record_sha256]
    return json.loads((capture / path).read_bytes())["job"]["nodes"], path


def _referent_context(capture, path, sources_by_order):
    """Earlier units of the same message (nearest first), then the previous message of the same session."""
    message_dir = (capture / path).parent
    unit_no = int(Path(path).stem.split("-")[1])
    context = [
        {"where": "message", "nodes": json.loads(p.read_bytes())["job"]["nodes"], "text": ""}
        for p in sorted(message_dir.glob("unit-[0-9][0-9].json"), reverse=True)
        if int(p.stem.split("-")[1]) < unit_no
    ]
    order = int(message_dir.name)
    source = sources_by_order[order]
    previous = sources_by_order.get(order - 1)
    if PREVIOUS_MESSAGE_ANTECEDENTS and previous is not None and previous["session_id"] == source["session_id"]:
        for p in sorted((capture / f"{order - 1:04d}").glob("unit-[0-9][0-9].json"), reverse=True):
            context.append(
                {"where": "previous_message", "nodes": json.loads(p.read_bytes())["job"]["nodes"], "text": ""}
            )
    return context


def _referent_label(referent, tokens):
    if referent["status"] in {"candidate", "picked"}:
        return referent["text"]
    if referent["status"] == "expletive":
        return "[" + " ".join(tokens[i] for i in referent["span"] if i in tokens) + "]"
    return "OPEN"


def enrich_frames(frames, nodes, context, segment_tokens, picks=None):
    """Time roles (``argm-tmp``) and pronoun referents layered onto frames over the unit's original tokens."""
    tokens = {n["start"]: n["string"] for n in nodes if n["node_type"] == "leaf"}
    out = attach_time(frames, nodes, time_spans(nodes), segment_tokens=segment_tokens)
    for frame in out:
        for element in frame["frameElements"]:
            element["feString"] = " ".join(tokens[i] for i in element["indices"] if i in tokens)
        for position, index in pronoun_roles(frame, nodes):
            element = frame["frameElements"][position]
            referent = resolve_pronoun(index, nodes, context)
            if picks and index in picks and referent["status"] == "candidate":
                referent = picks[index]  # Jev's antecedent pick (or OPEN) replaces the nearest candidate
            element["referent"] = referent
            element["feString"] = f"{element['feString']} → {_referent_label(referent, tokens)}"
    return out
