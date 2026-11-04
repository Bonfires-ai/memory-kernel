"""Chain ground: built at the bootstrap, folded by each append, absent on an older chain. Invented names."""

from __future__ import annotations

from memory_kernel.graph.referents import Ground
from memory_kernel.pipeline import chain as ch
from unit.pipeline import test_chain as tc


def test_ground_round_trips_through_the_view():
    c = ch.empty()
    c.ground = {"e-k": Ground("e-k", ("tg-a",), ("ilse",), "m1", "m1", "2026-07-01T10:00:00+00:00", 1)}
    c.agent_handles = ["@helper"]
    view = ch.projection_view(c)
    assert view["ground"] == [c.ground["e-k"].to_row()]
    assert view["agent_handles"] == ["@helper"]
    back = ch.from_projection_view(view)
    assert back.ground == c.ground and back.agent_handles == ["@helper"]
    old = {k: v for k, v in view.items() if k not in ("ground", "agent_handles")}
    assert ch.from_projection_view(old).ground is None and ch.from_projection_view(old).agent_handles == []


def test_ground_folds_a_document_message():
    from memory_kernel.graph.referents import fold_ground

    rows = {"doc-kiln:3": {"message_id": "doc-kiln:3", "speaker": "archive", "timestamp": "2026-07-01T00:00:00+00:00",
                           "addressees": []}}  # fmt: skip
    g = fold_ground({}, {"e-k": ["doc-kiln:3"]}, rows, lambda _m: "doc-kiln", set(), legacy=True)
    assert g["e-k"].chats == ("doc-kiln",) and g["e-k"].people == ("archive",)


# -- evidence: the first and last mention of each ground row, kept in the chain (retention prunes runs) ---------------


def kiln_row(mid: str, text: str) -> dict:
    return {"message_id": mid, "speaker": "Ilse", "addressees": ["Oren"], "timestamp": "2026-07-01T10:00:00+00:00",
            "text": text, "order": 3, "session_id": "conv-ember-session_1"}  # fmt: skip


def test_evidence_rows_keep_five_fields_and_stay_out_of_the_view():
    c = ch.empty()
    c.ground = {"e-k": Ground("e-k", ("tg-a",), ("ilse",), "m1", "m1", "2026-07-01T10:00:00+00:00", 1)}
    c.evidence = {"m1": ch.evidence_row(kiln_row("m1", "a kiln arrived"))}
    assert set(c.evidence["m1"]) == set(ch.EVIDENCE_FIELDS)  # only the five fields: no order, no session id
    assert "evidence" not in ch.projection_view(c)


def test_fold_evidence_adds_first_and_last_mentions_only_and_never_overwrites():
    ground = {"e-k": Ground("e-k", ("tg-a",), ("ilse",), "m1", "m3", "2026-07-01T10:00:00+00:00", 3)}
    rows = {m: kiln_row(m, f"{m} new") for m in ("m1", "m2", "m3")}
    stored = {"m1": ch.evidence_row(kiln_row("m1", "m1 as stored"))}
    out = ch.fold_evidence(stored, ground, rows)
    assert set(out) == {"m1", "m3"}  # m2 is no end of a ground row
    assert out["m1"]["text"] == "m1 as stored" and out["m3"]["text"] == "m3 new"
    assert stored == {"m1": ch.evidence_row(kiln_row("m1", "m1 as stored"))}  # the input is not changed


def test_a_handle_and_its_bare_addressee_are_the_same_agent():
    from memory_kernel.graph.referents import _people, fold_ground, person_key

    row = {"message_id": "m1", "speaker": "Ilse", "addressees": ["@helper"], "timestamp": "2026-07-01T10:00:00+00:00"}
    assert _people(row, {person_key("helper")}) == ["ilse"]  # "@helper" addressed, agent given bare
    bare = {**row, "addressees": ["helper"]}
    assert _people(bare, {person_key("@helper")}) == ["ilse"]  # the reverse
    g = fold_ground({}, {"e-k": ["m1"]}, {"m1": bare}, lambda _m: "tg-a", ["@Helper"], legacy=True)
    assert g["e-k"].people == ("ilse",)


def test_from_bootstrap_ground_is_legacy():
    ground = tc.bootstrap_state().ground
    assert ground and all(g.legacy for g in ground.values())
