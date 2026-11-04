"""The resolver's referee hook (kernel 1.2, referent resolution in appends): a filler with a stored or earlier-session
default is offered to a referee, whose decision (``{"join": uuid}`` / ``{"new": key}`` / None) replaces the default.
Fixtures come from ``test_forward_resolution``. Invented names only."""

from __future__ import annotations

from test_forward_resolution import (
    GROUP,
    LABEL,
    NEEDS_WORDNET,
    arg_targets,
    chain_of,
    fold,
    likes,
    me,
    my_name,
    rec,
    uid,
)

from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.graph.g1_projection import project
from memory_kernel.graph.kg_contract import validate_rows
from memory_kernel.graph.referents import Point, discriminated_key, mention_form


def run(records, chain=None, number=0, referee=None):
    """``test_forward_resolution.run`` with a referee."""
    eventuality = {
        f"{r['unit']}#{i}": {"label": LABEL[ev["roleset"]], "p": {LABEL[ev["roleset"]]: 1.0}}
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
        conversation_of={r["message_id"]: "k" for r in records} if chain is not None else None,
        referee=referee,
    )
    assert validate_rows(out["nodes"], out["edges"], previous=out["external_endpoints"]) == []
    return out


KILN = uid("kiln")


def kiln_chain():
    return chain_of([likes("b1", "Bo", "the kiln")])[0]


def two_kilns():
    return [likes("a1", "Bo", "the kiln", "pears"), likes("a2", "Bo", "the kiln", "pears")]


def test_without_a_referee_resolution_is_unchanged():
    chain = kiln_chain()
    out = run(two_kilns(), chain, 1)
    assert arg_targets(out) == {"a1": [KILN], "a2": [KILN]}
    assert out["forward"]["minted"] == [] and out["forward"]["routes"] == {}


def test_a_referee_sees_a_stored_reuse_as_a_point():
    chain, seen = kiln_chain(), []
    out = run(two_kilns()[:1], chain, 1, referee=lambda p: seen.append(p) or None)
    assert [(p.key, p.default, p.message_id) for p in seen if p.key == "kiln"] == [("kiln", KILN, "a1")]
    assert [p.text for p in seen if p.key == "kiln"] == ["the kiln"]
    assert isinstance(seen[0], Point) and seen[0].point.endswith("|arg0")
    assert arg_targets(out)["a1"] == [KILN]  # None keeps the default


def test_a_join_decision_resolves_to_its_uuid():
    chain, _ = chain_of([likes("b1", "Bo", "the kiln"), likes("b2", "Bo", "the oven", "bread")])
    oven = uid("oven")
    out = run(two_kilns()[:1], chain, 1, referee=lambda p: {"join": oven})
    assert arg_targets(out, "arg0")["a1"] == [oven]


def test_a_new_referent_is_minted_once():
    chain = kiln_chain()
    new = "kiln#0a1b2c3d"
    out = run(two_kilns(), chain, 1, referee=lambda p: {"new": new})
    minted = content_uuid("ent", GROUP, new)
    assert arg_targets(out) == {"a1": [minted], "a2": [minted]}
    assert [m.uuid for m in out["forward"]["minted"]] == [minted]
    assert out["forward"]["routes"] == {}


def test_a_person_filler_is_never_a_point():
    chain, _ = chain_of([likes("b1", "Bo", "Cy"), likes("b2", "Cy", "Bo")])
    seen = []
    session = [
        likes("a1", "Bo", "Cy", "pears"),  # a speaker named: a person
        rec("a2", "Cy", "I like pears", "like.01", 1, {"arg0": me("Cy"), "arg1": ("pears", [2])}),  # deixis
    ]
    run(session, chain, 1, referee=lambda p: seen.append(p) or None)
    assert [p.key for p in seen if p.key in ("bo", "cy")] == []


@NEEDS_WORDNET
def test_a_named_non_person_filler_is_never_a_point():
    """After the claim "my name is Mira" (Ana) the routed name "Mira" is a guarded name mention (``f.named``): it
    resolves by its route and the referee sees no point at all."""
    chain, _ = chain_of([likes("b1", "Bo", "Mira"), likes("b2", "Ana", "pears")])
    claim = [my_name("a1", "Ana", "Mira")]
    chain = fold(chain, run(claim, chain, 1), claim, 1)
    seen = []
    out = run([likes("c1", "Bo", "Mira", "plums")], chain, 2, referee=lambda p: seen.append(p) or None)
    assert seen == []
    assert arg_targets(out)["c1"] == [uid("ana")]


def test_a_referent_resolved_non_person_filler_is_never_a_point():
    chain = kiln_chain()
    seen = []
    it = {"text": "it", "span": [0], "referent": {"status": "picked", "text": "the kiln"}}
    session = [rec("a1", "Bo", "it is a pear", "like.01", 1, {"arg0": it, "arg1": ("pears", [2])})]
    out = run(session, chain, 1, referee=lambda p: seen.append(p) or None)
    assert [p for p in seen if p.key == "kiln"] == []
    assert arg_targets(out)["a1"] == [KILN]


def test_emit_reports_mentioned():
    chain = kiln_chain()
    out = run([likes("a1", "Bo", "the kiln", "clay")], chain, 1)
    mentioned = out["forward"]["mentioned"]
    assert mentioned[KILN] == ["a1"] and mentioned[uid("bo")] == ["a1"]
    episodic = {n["props"]["message_id"] for n in out["nodes"] if "Episodic" in n["labels_neo4j"]}
    assert all(set(ms) <= episodic for ms in mentioned.values())
    assert all(ms == sorted(ms) for ms in mentioned.values())


def test_a_full_build_gets_mentioned_too():
    out = run([likes("b1", "Bo", "the kiln")])
    assert out["forward"]["mentioned"][uid("kiln")] == ["b1"]


# -- the kind/instance split (spec 2b): a first specific mention reaches the referee, the plain key stays the kind ----


def empty_chain():
    return chain_of([likes("b1", "Bo", "Cy")])[0]


def test_first_specific_mention_reaches_referee_with_no_default():
    seen = []
    run([likes("a1", "Bo", "the kiln", "pears")], empty_chain(), 1, referee=lambda p: seen.append(p) or None)
    assert [(p.key, p.default) for p in seen if p.key == "kiln"] == [("kiln", None)]


def test_kind_mention_keeps_plain_key():
    def referee(p):
        return {"new": discriminated_key(p.key, p.message_id)} if mention_form(p.text) == "specific" else None

    session = [likes("a1", "Bo", "Cy", "clay"), likes("a2", "Bo", "Cy", "the clay")]
    out = run(session, empty_chain(), 1, referee=referee)
    kind, instance = arg_targets(out, "arg1")["a1"], arg_targets(out, "arg1")["a2"]
    assert kind == [content_uuid("ent", GROUP, "clay")]
    assert instance == [content_uuid("ent", GROUP, discriminated_key("clay", "a2"))] and instance != kind


def test_specific_first_then_kind_keeps_the_plain_key_for_the_kind():
    def referee(p):
        return {"new": discriminated_key(p.key, p.message_id)} if mention_form(p.text) == "specific" else None

    session = [likes("a1", "Bo", "Cy", "the clay"), likes("a2", "Bo", "Cy", "clay")]
    out = run(session, empty_chain(), 1, referee=referee)
    instance, kind = arg_targets(out, "arg1")["a1"], arg_targets(out, "arg1")["a2"]
    assert instance == [content_uuid("ent", GROUP, discriminated_key("clay", "a1"))]
    assert kind == [content_uuid("ent", GROUP, "clay")] and kind != instance
