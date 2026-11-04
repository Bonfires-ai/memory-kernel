"""Forward resolution in the projection (kernel 1.2): an append resolves its session against the chain.

Fixture: ``project()`` arguments built by hand (the style of ``test_identity_projection``). A "run" is one session of
one chat ``k``; run 0 is the bootstrap (a full build, ``chain=None``), whose ``forward`` seeds the chain; each later
run projects with ``chain=`` and its ``forward`` is folded into the chain the way the 1.8 folder commit folded it
(``fold``: new mints, first route wins, links, the chat's roster and order). Units are named by bonfire-wide order,
as the capture stage names them. Invented names only. Tests that read a lone capitalised name in context (an alias
mention, an identity claim) need WordNet and the spaCy model.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from itertools import count
from pathlib import Path

import pytest

from memory_kernel.grammar.dependencies import wordnet_ready
from memory_kernel.graph.g1_ids import content_uuid, summary_line_id
from memory_kernel.graph.g1_nlp import normalize_exact
from memory_kernel.graph.g1_projection import message_rank, project
from memory_kernel.graph.kg_contract import validate_rows
from memory_kernel.pipeline.chain import Chain, ChatState, SameAs

GROUP = "g-test"
LABEL = {"be.01": "state", "like.01": "occurrence", "meet.01": "occurrence", "own.01": "state"}
NEEDS_WORDNET = pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup)")
_UNIT = count()


def uid(key: str) -> str:
    return content_uuid("ent", GROUP, key)


def me(speaker: str) -> dict:
    return {"text": "I", "span": [0], "referent": {"status": "deixis", "text": speaker}}


def rec(mid: str, speaker: str, text: str, roleset: str, anchor: int, roles: dict, **extra) -> dict:
    """One record with one event; ``roles`` maps a role to (text, span) or to a full role dict."""
    spec = {r: v if isinstance(v, dict) else {"text": v[0], "span": v[1]} for r, v in roles.items()}
    n = next(_UNIT)
    return {
        "unit": f"{n:04d}/unit-00.json",
        "message_id": mid,
        "session_id": extra.pop("session", "s1"),
        "speaker": speaker,
        "text": text,
        "timestamp": f"2026-10-14T{n // 60:02d}:{n % 60:02d}:00Z",
        "capture_status": "captured",
        "events": [{"roleset": roleset, "anchor": [anchor], "roles": spec}],
        **extra,
    }


def likes(mid: str, speaker: str, who: str, what: str = "pears", label: str = "like.01") -> dict:
    return rec(mid, speaker, f"{who} likes {what}", label, len(who.split()), {
        "arg0": (who, list(range(len(who.split())))), "arg1": (what, [len(who.split()) + 1])
    })  # fmt: skip


def my_name(mid: str, speaker: str, name: str) -> dict:
    return rec(mid, speaker, f"my name is {name}", "be.01", 2, {"arg1": ("my name", [0, 1]), "arg2": (name, [3])})


def run(
    records: list[dict], chain: Chain | None = None, number: int = 0, labels: dict | None = None, chat: str = "k"
) -> dict:
    label = LABEL | (labels or {})
    eventuality = {
        f"{r['unit']}#{i}": {"label": label[ev["roleset"]], "p": {label[ev["roleset"]]: 1.0}}
        | ({"anchor_index": r["anchor_index"]} if "anchor_index" in r else {})  # the anchor aligned on the unit's parse
        for r in records
        for i, ev in enumerate(r["events"])
    }
    out = project(
        records,
        eventuality=eventuality,
        episodes={r["message_id"]: f"ep-{r['message_id']}" for r in records},
        group_id=GROUP,
        created_at="2026-10-14T00:00:00Z",
        chain=chain,
        run=number,
        conversation_of={r["message_id"]: chat for r in records} if chain is not None else None,
    )
    assert validate_rows(out["nodes"], out["edges"], previous=out["external_endpoints"]) == []
    return out


def fold(chain: Chain, out: dict, records: list[dict], number: int, chat: str = "k") -> Chain:
    """What the 1.8 folder commit added (Task 4): new mints, routes (the first wins), links, the chat's roster and
    orders (bonfire-wide), the addressing start."""
    f = out["forward"]
    old = chain.chats.get(chat)
    base = max((s.last_order + 1 for s in chain.chats.values()), default=0)
    rank = message_rank(records)
    roster = list(dict.fromkeys([*(old.roster if old else []), *(r["speaker"] for r in records)]))
    addressed = [base + rank[r["message_id"]] for r in records if r.get("addressees")]
    state = ChatState(1, "ep", "", base + len(rank) - 1, roster, [], {})
    return replace(
        chain,
        entities={**chain.entities, **{m.uuid: m for m in f["minted"]}},
        routes={**f["routes"], **chain.routes},
        same_as=[*chain.same_as, *f["same_as"]],
        chats={**chain.chats, chat: state},
        explicit_since=chain.explicit_since if chain.explicit_since is not None else min(addressed, default=None),
        run=number,
    )


def chain_of(records: list[dict]) -> tuple[Chain, dict]:
    """The bootstrap (a full build) and the chain its first mint gives."""
    out = run(records)
    return fold(Chain(), out, records, 0), out


def arg_targets(out: dict, role: str = "arg0") -> dict[str, list[str]]:
    events = {n["props"]["message_id"]: n["props"]["uuid"] for n in out["nodes"] if "Event" in n["labels_neo4j"]}
    return {
        mid: [e["target"] for e in out["edges"] if e["source"] == evt and e["props"].get("role") == role]
        for mid, evt in events.items()
    }


def uuids_of(out: dict) -> set[str]:
    rows = [*out["nodes"], *out["edges"]]
    return (
        {n["props"]["uuid"] for n in out["nodes"]}
        | {u for e in out["edges"] for u in (e["source"], e["target"])}
        | {r["props"]["uuid"] for r in rows}
    )


def entity_names(out: dict) -> dict[str, str]:
    return {
        n["props"]["uuid"]: n["props"]["name"]
        for n in out["nodes"]
        if "Entity" in n["labels_neo4j"] and "Event" not in n["labels_neo4j"]
    }


# -- identity claims across appends ----------------------------------------------------------------------


@NEEDS_WORDNET
def test_the_dee_ana_trace_across_two_appends():
    """The Dee/Ana trace, with a name the parser reads as a noun alone (Mira). The bootstrap mints Mira (a name Bo uses). An append with Ana's "my name is Mira" links Mira -> Ana and routes
    "mira" to Ana, changing no stored row; a third run's "Mira" lands on Ana and the old Mira node is not in it."""
    boot = [
        rec("b1", "Ana", "I like apples", "like.01", 1, {"arg0": me("Ana"), "arg1": ("apples", [2])}),
        likes("b2", "Bo", "Mira"),
    ]
    chain, first = chain_of(boot)
    dee, ana = uid("mira"), uid("ana")
    assert dee in chain.entities and chain.entities[dee].key == "mira" and "mira" not in chain.routes

    claim = [my_name("a1", "Ana", "Mira")]
    second = run(claim, chain, 1)
    f = second["forward"]
    assert [(s.from_uuid, s.to_uuid, s.holder, s.name, s.card, s.run) for s in f["same_as"]] == [
        (dee, ana, "Ana", "Mira", "equative", 1)
    ]
    assert f["routes"]["mira"] == ana and f["conflicts"] == [] and f["minted"] == []
    link = [e for e in second["edges"] if e["type"] == "SAME_AS"]
    assert [(e["source"], e["target"], e["props"]["claim_message"], e["props"]["name"]) for e in link] == [
        (dee, ana, "a1", "Mira")
    ]
    assert dee not in {n["props"]["uuid"] for n in second["nodes"]}  # an external endpoint, not a rewritten row
    assert second["external_endpoints"][dee]["labels_neo4j"] == ["Entity"]
    assert entity_names(second)[ana] == "Ana"  # the stored canonical keeps its name, gains the alias
    assert next(n for n in second["nodes"] if n["props"]["uuid"] == ana)["props"]["identity_aliases"] == ["Mira"]
    chain = fold(chain, second, claim, 1)

    third_in = [likes("c1", "Bo", "Mira", "plums")]
    third = run(third_in, chain, 2)
    assert arg_targets(third)["c1"] == [ana]
    assert dee not in uuids_of(third) and third["forward"]["same_as"] == [] and third["forward"]["minted"] == []
    # the bootstrap's rows of Mira are untouched by both appends: no node or edge of the appends names them
    old_dee_rows = {e["props"]["uuid"] for e in first["edges"] if dee in (e["source"], e["target"])}
    assert old_dee_rows and not old_dee_rows & (uuids_of(second) | uuids_of(third))


@NEEDS_WORDNET
def test_a_claim_and_the_names_first_mention_in_one_session_mint_in_message_order():
    """Review Focus 2. Before the claim, "Mira" mints its own node; the claim links it; after the claim "Mira" lands on
    Ana. A session whose claim comes first mints no Mira node at all."""
    chain, _ = chain_of([likes("b1", "Bo", "Ana")])
    ana = uid("ana")
    session = [likes("a1", "Bo", "Mira"), my_name("a2", "Ana", "Mira"), likes("a3", "Bo", "Mira", "plums")]
    out = run(session, chain, 1)
    dee = uid("mira")
    targets = arg_targets(out)
    assert targets["a1"] == [dee] and targets["a3"] == [ana]
    f = out["forward"]
    assert [(m.key, m.run, m.order) for m in f["minted"]] == [("mira", 1, 0)]
    assert [(s.from_uuid, s.to_uuid, s.message_id) for s in f["same_as"]] == [(dee, ana, "a2")]
    assert f["routes"] == {"mira": ana}

    first = run([my_name("d1", "Ana", "Mira"), likes("d2", "Bo", "Mira")], chain, 1)
    assert arg_targets(first)["d2"] == [ana]
    assert first["forward"]["minted"] == [] and first["forward"]["same_as"] == []
    assert first["forward"]["routes"] == {"mira": ana}
    assert dee not in uuids_of(first)


def i_am(mid: str, speaker: str, name: str) -> dict:
    return rec(mid, speaker, f"I am {name}", "be.01", 1, {"arg1": me(speaker), "arg2": (name, [2])})


def is_(mid: str, speaker: str, holder: str, name: str) -> dict:
    return rec(mid, speaker, f"{holder} is {name}", "be.01", 1, {"arg1": (holder, [0]), "arg2": (name, [2])})


def links(out: dict) -> list[tuple[str, str]]:
    return [(s.from_uuid, s.to_uuid) for s in out["forward"]["same_as"]]


@NEEDS_WORDNET
def test_a_speaker_holding_a_claim_is_canonical_over_an_older_name_node():
    """ "I am Juno" from the speaker Quill links the older "Juno" node to Quill and routes
    "juno" there, though Juno was minted first (in the bootstrap, or earlier in the same session); so does "Quill is
    Juno" said by another speaker before Quill speaks."""
    chain, _ = chain_of([likes("b1", "Bo", "Juno")])
    juno, quill = uid("juno"), uid("quill")
    assert chain.entities[juno].run == 0
    out = run([i_am("a1", "Quill", "Juno"), likes("a2", "Bo", "Juno", "figs")], chain, 1)
    assert links(out) == [(juno, quill)] and out["forward"]["routes"] == {"juno": quill}
    assert arg_targets(out)["a2"] == [quill]

    fresh, _ = chain_of([likes("b1", "Bo", "Ana")])
    same = run([likes("a1", "Bo", "Juno"), i_am("a2", "Quill", "Juno")], fresh, 1)
    assert [m.key for m in same["forward"]["minted"]][:1] == ["juno"]  # Juno minted before Quill
    assert links(same) == [(juno, quill)]

    told = run([is_("a1", "Bo", "Quill", "Juno"), likes("a2", "Quill", "Bo")], chain, 1)
    assert links(told) == [(juno, quill)] and told["forward"]["routes"] == {"juno": quill}


@NEEDS_WORDNET
def test_a_claim_between_two_non_speakers_takes_the_earliest_minted_node():
    """Neither Priya nor Juno speaks: the older node (Juno) is canonical, Priya's links to it and "priya" routes there;
    with the mentions the other way round, Priya's node is the older one and canonical."""
    chain, _ = chain_of([likes("b1", "Bo", "Juno"), likes("b2", "Bo", "Priya", "figs")])
    out = run([is_("c1", "Cy", "Priya", "Juno")], chain, 1)
    assert links(out) == [(uid("priya"), uid("juno"))] and out["forward"]["routes"] == {"priya": uid("juno")}
    chain, _ = chain_of([likes("b1", "Bo", "Priya"), likes("b2", "Bo", "Juno", "figs")])
    out = run([is_("c1", "Cy", "Priya", "Juno")], chain, 1)
    assert links(out) == [(uid("juno"), uid("priya"))] and out["forward"]["routes"] == {"juno": uid("priya")}


@NEEDS_WORDNET
def test_the_possessive_of_a_routed_name_is_the_canonicals_possessive():
    """After the route "mira" -> Ana, a later "Mira's" is the entity "Ana's" is (minted under the canonical
    possessive's key and named "Ana's"), and its key forms route there."""
    chain, _ = chain_of([likes("b1", "Bo", "Ana"), likes("b2", "Ana", "Bo")])
    claim = [my_name("a1", "Ana", "Mira")]
    chain = fold(chain, run(claim, chain, 1), claim, 1)
    visit = [rec("c1", "Bo", "I visited Mira's", "meet.01", 1, {"arg0": me("Bo"), "arg1": ("Mira's", [2])})]
    out = run(visit, chain, 2)
    (target,) = arg_targets(out, "arg1")["c1"]
    assert entity_names(out)[target] == "Ana's"
    assert [(m.uuid, m.display) for m in out["forward"]["minted"]] == [(target, "Ana's")]
    assert set(out["forward"]["routes"].values()) == {target} and "mira's" in out["forward"]["routes"]


@NEEDS_WORDNET
def test_a_conflicting_second_claim_is_reported_with_no_link():
    """Ana claimed "Mira" first; Cy's later "my name is Mira" finds the name routed to Ana: a conflict, no link, and
    its claim event stays content."""
    chain, _ = chain_of([likes("b1", "Bo", "Ana"), likes("b2", "Cy", "Bo")])
    claim = [my_name("a1", "Ana", "Mira")]
    chain = fold(chain, out := run(claim, chain, 1), claim, 1)
    assert out["forward"]["routes"] == {"mira": uid("ana")}
    second = run([my_name("c1", "Cy", "Mira")], chain, 2)
    f = second["forward"]
    assert f["conflicts"] == [{"name": "Mira", "existing": uid("ana"), "claimed": uid("cy"), "message_id": "c1"}]
    assert f["same_as"] == [] and f["routes"] == {}
    assert not [e for e in second["edges"] if e["type"] == "SAME_AS"]
    assert second["dedup"]["identity_conflicts"] and second["dedup"].get("identity_claim_events_dropped") == 0


@NEEDS_WORDNET
def test_a_common_noun_route_needs_a_guarded_mention_in_a_later_append():
    """After Ana's "my name is Rose" (rose is a common noun): in a later append "I asked Rose" lands on Ana; "I like
    rose" and "Rose is lovely." do not."""
    chain, _ = chain_of([likes("b1", "Bo", "Ana")])
    claim = [my_name("a1", "Ana", "Rose")]
    chain = fold(chain, run(claim, chain, 1), claim, 1)
    assert chain.routes["rose"] == uid("ana")
    later = [
        rec("c1", "Bo", "I asked Rose", "meet.01", 1, {"arg0": me("Bo"), "arg1": ("Rose", [2])}),
        rec("c2", "Bo", "I like rose", "like.01", 1, {"arg0": me("Bo"), "arg1": ("rose", [2])}),
        rec("c3", "Bo", "Rose is lovely.", "like.01", 1, {"arg0": ("Rose", [0]), "arg1": ("lovely", [2])}),
    ]
    out = run(later, chain, 2)
    arg0, arg1 = arg_targets(out, "arg0"), arg_targets(out, "arg1")
    assert arg1["c1"] == [uid("ana")]
    assert uid("ana") not in arg1.get("c2", []) and uid("ana") not in arg0.get("c3", [])


# -- fuzzy dedupe ----------------------------------------------------------------------------------------


def test_fuzzy_onto_a_linked_node_lands_on_the_canonical():
    """Review Focus 3: "Delphine Marlowee" is close to the stored "Delphine Marlowe", which is linked to Ana: it lands
    on Ana, not on the old node, and records a route (no link)."""
    chain, _ = chain_of([likes("b1", "Bo", "Delphine Marlowe"), likes("b2", "Ana", "Bo")])
    old, ana = uid("delphine marlowe"), uid("ana")
    chain.same_as.append(SameAs(old, ana, "a1", "Ana", "Delphine Marlowe", "equative", 1))
    chain.routes["delphine marlowe"] = ana
    out = run([likes("c1", "Bo", "Delphine Marlowee")], chain, 2)
    assert arg_targets(out)["c1"] == [ana]
    assert out["forward"]["routes"] == {"delphine marlowee": ana} and out["forward"]["same_as"] == []
    assert old not in uuids_of(out)


def test_a_stored_match_beats_a_session_match_and_the_earliest_mint_wins_a_tie():
    """ "Copperfield Mill" is as close to the stored "Coppperfield Mill" (minted first) as to the stored "Copperfield
    Milll", and also close to the session's earlier "Copperfield Milll"-like key: the earliest stored node wins."""
    chain, _ = chain_of([likes("b1", "Bo", "Coppperfield Mill"), likes("b2", "Bo", "Copperfield Milll")])
    first, second = uid("coppperfield mill"), uid("copperfield milll")
    assert chain.entities[first].order < chain.entities[second].order
    out = run([likes("c1", "Bo", "Copperfield Mill")], chain, 1)
    assert arg_targets(out)["c1"] == [first]
    assert out["forward"]["routes"] == {"copperfield mill": first} and out["forward"]["minted"] == []

    # stored beats session: the session first mints "Copperfield Milll"'s twin, which does not match the stored node
    chain, _ = chain_of([likes("b1", "Bo", "Coppperfield Mill")])
    out = run([likes("c1", "Bo", "Copperfield Milll"), likes("c2", "Bo", "Copperfield Mill")], chain, 1)
    targets = arg_targets(out)
    assert targets["c1"] == [uid("copperfield milll")] and targets["c2"] == [first]

    # with no stored match, the first-occurring session key mints and the later one attaches
    chain, _ = chain_of([likes("b1", "Bo", "Ana")])
    out = run([likes("c1", "Bo", "Copperfield Milll"), likes("c2", "Bo", "Copperfield Mill")], chain, 1)
    targets = arg_targets(out)
    assert targets["c1"] == targets["c2"] == [uid("copperfield milll")]
    assert out["forward"]["routes"] == {"copperfield mill": uid("copperfield milll")}


# -- addressing and the roster ---------------------------------------------------------------------


def test_addressing_is_decided_per_message_and_a_later_session_changes_no_earlier_row():
    """Run 1 (no addressees anywhere yet): Ana's "you" is Bo by the two-party rule. Run 2: its first addressed message
    switches "you" to the addressees from there on; its earlier message keeps the two-party rule; run 1's rows are not
    in run 2."""
    chain, _ = chain_of([likes("b1", "Bo", "Ana"), likes("b2", "Ana", "Bo")])
    first_in = [
        rec("a1", "Ana", "I like your hat", "like.01", 1, {"arg0": me("Ana"), "arg1": ("your hat", [2, 3])}),
        likes("a2", "Bo", "Ana", "figs"),
    ]
    first = run(first_in, chain, 1)
    episodic = {n["props"]["message_id"]: n["props"] for n in first["nodes"] if "Episodic" in n["labels_neo4j"]}
    assert episodic["a1"]["addressee"] == uid("bo")
    assert arg_targets(first, "arg1")["a1"] == [uid("bo's hat")]
    chain = fold(chain, first, first_in, 1)

    second_in = [
        rec("c1", "Ana", "I like your scarf", "like.01", 1, {"arg0": me("Ana"), "arg1": ("your scarf", [2, 3])}),
        rec("c2", "Ana", "I like your coat", "like.01", 1, {"arg0": me("Ana"), "arg1": ("your coat", [2, 3])},
            addressees=["Cy"]),
        rec("c3", "Bo", "I like your boots", "like.01", 1, {"arg0": me("Bo"), "arg1": ("your boots", [2, 3])},
            addressees=[]),
    ]  # fmt: skip
    second = run(second_in, chain, 2)
    episodic = {n["props"]["message_id"]: n["props"] for n in second["nodes"] if "Episodic" in n["labels_neo4j"]}
    assert episodic["c1"]["addressee"] == uid("bo")  # before the first addressed message: two-party
    assert episodic["c2"].get("addressee") is None  # addressed to Cy, who has no node: no addressee uuid
    assert episodic["c3"].get("addressee") is None  # explicit from c2 on: no addressee given, none inferred
    arg1 = arg_targets(second, "arg1")
    assert arg1["c1"] == [uid("bo's scarf")] and arg1["c2"] == [uid("cy's coat")]
    assert not {n["props"]["uuid"] for n in first["nodes"] if "Episodic" in n["labels_neo4j"]} & uuids_of(second)
    assert uid("bo's hat") not in uuids_of(second)


def test_a_new_speaker_joins_the_roster_and_leaves_earlier_keys_unchanged():
    """ "Mo Banks" is a plain name in the bootstrap (its lemma key "mo bank"); when Mo Banks speaks in a later run,
    the run mints the speaker under its exact key; the earlier node keeps its key and gets no row."""
    chain, _ = chain_of([likes("b1", "Bo", "Mo Banks")])
    earlier = [m for m in chain.entities.values() if m.display == "Mo Banks"]
    assert len(earlier) == 1 and earlier[0].key != normalize_exact("Mo Banks")
    out = run([likes("c1", "Mo Banks", "Bo", "plums"), likes("c2", "Bo", "Mo Banks", "figs")], chain, 1)
    speaker = uid("mo banks")
    assert arg_targets(out)["c2"] == [speaker]
    assert [m.key for m in out["forward"]["minted"]] == ["mo banks"]
    assert earlier[0].uuid not in uuids_of(out)


# -- summary lines on stored nodes, and the full build ---------------------------------------------------------------


def test_a_summary_line_about_a_stored_node_is_appended():
    chain, _ = chain_of([likes("b1", "Bo", "Ana")])
    mapmaker = rec("c2", "Bo", "Ana is a mapmaker.", "be.01", 1, {"arg1": ("Ana", [0]), "arg2": ("a mapmaker", [2, 3])})
    out = run([likes("c1", "Bo", "Ana", "pears"), mapmaker], chain, 1, labels={"like.01": "state"})
    ana = next(n for n in out["nodes"] if n["props"]["uuid"] == uid("ana"))["props"]
    assert ana["summary"] == "" and len(ana["summary_append"]) == 2 and "pears" in ana["summary_append"][0]
    episodic = [
        next(n["props"]["uuid"] for n in out["nodes"] if n["props"].get("message_id") == m) for m in ("c1", "c2")
    ]
    assert ana["summary_append_sources"] == episodic
    ids = [summary_line_id(uid("ana"), src, line) for src, line in zip(episodic, ana["summary_append"], strict=True)]
    assert ana["summary_append_line_ids"] == ids
    assert ana["summary_append_identity_line_ids"] == ids[1:]  # "Ana is a mapmaker." is an identity line
    assert out["routing"]["summary_lines_appended_on_reused"] == 2
    assert uid("ana") in out["forward"]["touched"]
    # Ana was minted as a plain entity (named, never speaking): a stored node's labels never change
    assert next(n["labels_neo4j"] for n in out["nodes"] if n["props"]["uuid"] == uid("ana")) == ["Entity"]


def test_summary_lines_carry_their_text_and_identity():
    """A full build writes each line whole (``summary_line_texts``: a line may hold a newline) and the ids of its
    identity lines: an asserted copular clause with a nominal predicate whose subject is the bearer's filler, decided
    by the bearer's referent (Milo's "I"), never by name. An adjective predicate or an asked one is no identity. The
    subject is tested on the unit's parse: a capture shifted off it (indices two tokens on, the anchor aligned) is
    realigned; a filler the parse does not hold is never the subject, and is counted."""
    kira = {"arg1": ("Kira Dace", [0, 1])}
    asked = {"arg1": ("Kira Dace", [1, 2])}
    shifted = {"arg1": ("Kira Dace", [2, 3]), "arg2": ("a pilot", [5, 6])}
    records = [
        rec("d1", "Ana", "Kira Dace is a mapmaker.", "be.01", 2, kira | {"arg2": ("a mapmaker", [3, 4])}),
        rec("d2", "Ana", "Kira Dace is happy.", "be.01", 2, kira | {"arg2": ("happy", [3])}),
        rec("d3", "Ana", "Is Kira Dace a mapmaker?", "be.01", 0, asked | {"arg2": ("a mapmaker", [3, 4])}),
        rec("d4", "Ana", "Is Kira Dace\na mapmaker?", "be.01", 0, asked | {"arg2": ("a mapmaker", [4, 5])}),
        rec("d5", "Milo", "I am a mapmaker.", "be.01", 1, {"arg1": me("Milo"), "arg2": ("a mapmaker", [2, 3])}),
        rec("d6", "Ana", "Kira Dace is a pilot for Brask Labs.", "be.01", 4, shifted, anchor_index=2),
        rec(
            "d7",
            "Ana",
            "Kira Dace is a pilot.",
            "be.01",
            2,
            {"arg1": ("Brask Labs", [0, 1]), "arg2": ("a pilot", [3, 4])},
        ),
    ]
    out = run(records)
    props = {n["props"].get("name"): n["props"] for n in out["nodes"]}
    texts, ids = props["Kira Dace"]["summary_line_texts"], props["Kira Dace"]["summary_line_ids"]
    assert len(texts) == len(ids) == 5 and "\n".join(texts) == props["Kira Dace"]["summary"]
    assert "\n" in texts[3] and len(props["Kira Dace"]["summary"].split("\n")) == 6  # one line, two pieces
    assert props["Kira Dace"]["summary_identity_line_ids"] == [ids[0], ids[4]]
    assert props["Milo"]["summary_identity_line_ids"] == props["Milo"]["summary_line_ids"] != []
    assert props["Brask Labs"]["summary_identity_line_ids"] == [] != props["Brask Labs"]["summary_line_ids"]
    assert out["routing"]["identity_subject_outside_bearers"] == 1


def _fixture(path: str):
    """Another test module's fixtures, by path (test directories are not packages)."""
    spec = importlib.util.spec_from_file_location(f"_fixture_{Path(path).stem}", Path(__file__).parents[1] / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@NEEDS_WORDNET
def test_a_full_builds_forward_is_its_first_mint():
    """Every entity it writes, by first mention (run 0), under its key; routes for the claimed name's key forms."""
    test_identity_projection = _fixture("graph/test_identity_projection.py")

    out = test_identity_projection.run(test_identity_projection._records())
    f = out["forward"]
    ana, bo = uid("ana"), uid("bo")
    assert [(m.uuid, m.key, m.kind, m.display, m.run, m.order) for m in f["minted"]] == [
        (ana, "ana", "person", "Ana", 0, 0),
        (bo, "bo", "person", "Bo", 0, 0),
    ]
    assert f["routes"] == {"dee": ana} and f["same_as"] == [] and f["touched"] == sorted([ana, bo])


# -- the roster is bonfire-wide (fix round 2): a bootstrap in chat k, an append in chat k2 ----------------------------


def k2(*records: dict) -> list[dict]:
    return [{**r, "session_id": "s2"} for r in records]


@NEEDS_WORDNET
def test_a_speaker_of_another_chat_is_a_speaker_for_claims():
    """Bo and Zorvan speak in k; in k2 Bo's "I am Zorvan" would merge two speakers: one conflict, no link, no route.
    Quill spoke in k: Bo's "I am Quill" in k2 is a conflict too."""
    chain, _ = chain_of([likes("b1", "Bo", "Ana"), likes("b2", "Zorvan", "Bo"), likes("b3", "Quill", "Bo")])
    out = run(k2(i_am("a1", "Bo", "Zorvan")), chain, 1, chat="k2")
    f = out["forward"]
    assert len(f["conflicts"]) == 1 and f["same_as"] == [] and f["routes"] == {}
    out = run(k2(i_am("a1", "Bo", "Quill")), chain, 1, chat="k2")
    f = out["forward"]
    assert len(f["conflicts"]) == 1 and f["same_as"] == [] and f["routes"] == {}


def test_a_speaker_of_another_chat_is_a_person_when_named():
    """ "Mo Banks" spoke in k; in k2 Cy's "Mo Banks likes plums" lands on the stored person node, minting nothing new
    but Cy."""
    chain, _ = chain_of([likes("b1", "Mo Banks", "Bo"), likes("b2", "Bo", "Ana")])
    mo = uid("mo banks")
    assert chain.entities[mo].kind == "person"
    out = run(k2(likes("a1", "Cy", "Mo Banks", "plums")), chain, 1, chat="k2")
    assert arg_targets(out)["a1"] == [mo]
    assert [m.key for m in out["forward"]["minted"]] == ["cy"]
    assert {uid("bo"), uid("ana")}.isdisjoint(uuids_of(out))  # the earlier nodes this run does not name stay untouched


@NEEDS_WORDNET
def test_a_claim_naming_a_speaker_of_another_chat_is_extracted():
    """Quill spoke in k; in k2 Bo's "Quill is Juno" is a claim (Quill reads as a name): "juno" routes to Quill."""
    chain, _ = chain_of([likes("b1", "Quill", "Bo"), likes("b2", "Bo", "Ana")])
    out = run(k2(is_("a1", "Bo", "Quill", "Juno")), chain, 1, chat="k2")
    assert out["forward"]["routes"] == {"juno": uid("quill")} and out["forward"]["conflicts"] == []


def test_a_full_builds_first_mint_never_raises_on_an_unmentioned_entity():
    """``emit`` leaves an entity no message names out of a full build's mint (``from_bootstrap`` refuses that mint);
    a forward run (``strict``) raises."""
    from memory_kernel.graph.forward import emit

    out = run([likes("b1", "Bo", "Ana")])
    nodes = {n["props"]["uuid"]: n for n in out["nodes"]}
    lone = {"labels_neo4j": ["Entity"], "props": {"uuid": "z" * 32, "name": "Zed", "group_id": GROUP}}
    args = {"edges": out["edges"], "records": [likes("b1", "Bo", "Ana")], "merged": {}, "stored": set(), "run": 0}
    keys = {u: n["props"].get("name_key", "") for u, n in nodes.items()} | {"z" * 32: "zed"}
    loose = emit(nodes={**nodes, "z" * 32: lone}, rep_key=keys, routes={"zz": "y" * 32}, **args)
    assert "z" * 32 not in {m.uuid for m in loose["minted"]} and loose["routes"] == {}
    with pytest.raises(ValueError, match="never mentioned"):
        emit(nodes={**nodes, "z" * 32: lone}, rep_key=keys, routes={}, strict=True, **args)


def _write_gate(out: dict, chain: Chain, records: list[dict], stored: list[dict]) -> None:
    """The write's forward gate (``stages.write``) on a real forward projection: rule 1 with the run's own derived
    uuids, rule 2 on every row whose uuid an earlier projection (``stored``) wrote, against that row, and the chain
    commit's checks on the run's forward result."""
    from memory_kernel.pipeline.stages import write

    ids = list(dict.fromkeys(r["message_id"] for r in records))
    own = write.own_uuids(out["forward"], records, [f"ep-{m}" for m in ids], GROUP)
    assert {n["props"]["uuid"] for n in out["nodes"] if "Event" in n["labels_neo4j"]} <= own  # Events by derivation
    have = {n["props"]["uuid"]: n for rows in stored for n in rows["nodes"]}
    edges = {e["props"]["uuid"]: e for rows in stored for e in rows["edges"]}
    fold = json.loads(json.dumps(write.forward_record(out["forward"], run=chain.run + 1, episodic=[], own=own)))
    write.forward_gate(
        out,
        [have[n["props"]["uuid"]] for n in out["nodes"] if n["props"]["uuid"] in have],
        [edges[e["props"]["uuid"]] for e in out["edges"] if e["props"]["uuid"] in edges],
        chain,
        own,
        chat="k",
        fold=fold,
    )


@NEEDS_WORDNET
def test_the_write_gate_passes_real_forward_projections():
    """Links, routes, possessives, fuzzy attachments and appended summary lines all pass the write's forward gate."""
    boot = [
        rec("b1", "Ana", "I like apples", "like.01", 1, {"arg0": me("Ana"), "arg1": ("apples", [2])}),
        likes("b2", "Bo", "Mira"),
    ]
    chain, first = chain_of(boot)
    claim = [my_name("a1", "Ana", "Mira")]
    second = run(claim, chain, 1)
    _write_gate(second, chain, claim, [first])
    chain = fold(chain, second, claim, 1)
    visit = [rec("c1", "Bo", "I visited Mira's", "meet.01", 1, {"arg0": me("Bo"), "arg1": ("Mira's", [2])})]
    third = run(visit, chain, 2)
    assert third["forward"]["minted"]  # the canonical's possessive
    _write_gate(third, chain, visit, [first, second])

    chain, boot_rows = chain_of([likes("b1", "Bo", "Delphine Marlowe"), likes("b2", "Ana", "Bo")])
    fuzzy = [likes("c1", "Bo", "Delphine Marlowee"), likes("c2", "Bo", "Ana", "pears")]
    out = run(fuzzy, chain, 1, labels={"like.01": "state"})
    assert out["forward"]["routes"] and any("summary_append" in n["props"] for n in out["nodes"])
    _write_gate(out, chain, fuzzy, [boot_rows])
