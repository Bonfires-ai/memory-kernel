"""The forward write gate (``write.forward_gate``, ``write.pending_check``) as pure functions: the stored rows are built
in memory (``test_write_append_mode``'s hand-made bootstrap and session) and each refusal is asserted by name. Kernel 2.0
runs this gate inside the rolled-back dry write, so nothing here needs a Neo4j driver. Invented names only."""

from __future__ import annotations

import copy
import re
from dataclasses import replace

import pytest

from memory_kernel.graph import kg_writer
from memory_kernel.pipeline.chain import ChatState
from memory_kernel.pipeline.stages import write
from unit.pipeline.test_write_append_mode import (
    CHAIN,
    FACT,
    GROUP,
    QUESTION,
    bootstrap,
    edge,
    ent,
    entity,
    ep,
    episodic,
    fact,
    saga,
    saga_row,
    session,
    source,
)

RECORDS = [{"message_id": "m4"}, {"message_id": "m5"}]


def gate(rows: dict, stored: dict | None = None, *, chain=CHAIN, totals=None, fold: bool = False) -> None:
    """``forward_gate`` of the session ``rows`` against the stored rows (default: the bootstrap's)."""
    stored = stored if stored is not None else bootstrap()
    write.keep_contract_aggregates(
        rows, {n["props"]["uuid"] for n in stored["nodes"]}
    )  # as the stage does, before the gate
    own = write.own_uuids(rows["forward"], RECORDS, ["e3"], GROUP)
    record = write.forward_record(rows["forward"], run=chain.run + 1, episodic=[], own=own) if fold else None
    write.forward_gate(rows, stored["nodes"], stored["edges"], chain, own, chat="k1", totals=totals, fold=record)


def named(uuid: str, rows: dict) -> dict:
    return next(n for n in rows["nodes"] if n["props"]["uuid"] == uuid)


def test_the_session_itself_passes_on_the_bootstrap_group():
    gate(session(), fold=True)


def test_rule_1_refuses_a_stored_uuid_the_chain_never_minted():
    """The projection re-keys the kiln onto a stored row the chain does not hold."""
    stored = bootstrap()
    stored["nodes"].append(entity("kilns", "kilns", name_key="kilns"))  # an orphan of some older write
    rows = session()
    named(ent("kiln"), rows)["props"]["uuid"] = ent("kilns")  # the injected wrong uuid
    for e in rows["edges"]:
        e["target"] = ent("kilns") if e["target"] == ent("kiln") else e["target"]
    with pytest.raises(write.Refused, match=rf"forward gate rule 1: Entity {ent('kilns')} .* already in the group"):
        gate(rows, stored)
    # a mint under another key's uuid is a re-key too
    rows = session()
    rows["forward"]["minted"][0] = replace(rows["forward"]["minted"][0], key="glazes")
    with pytest.raises(write.Refused, match="is not the uuid of its key 'glazes'"):
        gate(rows, stored)
    # an edge to a stored row outside the chain
    rows = session()
    rows["edges"].append(edge("MENTIONS", ep("m5"), ent("kilns")))
    with pytest.raises(write.Refused, match=rf"rule 1: MENTIONS .* ends at {ent('kilns')}"):
        gate(rows, stored)
    gate(session(), stored)  # the session itself passes on the same group


def test_rule_1_scopes_to_the_sessions_chat_and_the_contracts_own_uuids():
    stored = bootstrap()
    # another chat's last saga is not this session's to link from
    k2 = ChatState(1, "e7", saga("e7"), 3, [], [source("m7", 1)], {"messages": 1, "episode_start": 0, "cursor": 0})
    other = replace(CHAIN, chats={**CHAIN.chats, "k2": k2}, episodes=CHAIN.episodes + 1)
    stored["nodes"].append(saga_row("e7", ["m7"]))
    rows = session()
    rows["edges"].append(edge("NEXT_SAGA", saga("e7"), saga("e3")))
    with pytest.raises(write.Refused, match=rf"rule 1: NEXT_SAGA .* ends at {saga('e7')}"):
        gate(rows, stored, chain=other)
    # a contract row must carry its group-constant uuid
    rows = session()
    named(QUESTION, rows)["props"]["qid"] = "q9"
    with pytest.raises(write.Refused, match=rf"rule 1: Question {QUESTION} .* already in the group"):
        gate(rows, stored, chain=other)
    # an edge uuid stored between other rows (read by type and uuid)
    mention = edge("MENTIONS", ep("m4"), ent("ana"))
    stored["edges"].append(mention | {"source": ep("m3"), "target": ent("kiln")})
    with pytest.raises(write.Refused, match="rule 1: MENTIONS .* is stored between other rows"):
        gate(session(), stored, chain=other)


@pytest.mark.parametrize(
    ("edit", "fields"),
    [
        (lambda r: named(ent("ana"), r)["props"].update(name="Anna"), "['name']"),
        (lambda r: named(ent("kiln"), r)["labels_neo4j"].append("Event"), "['labels']"),
        (lambda r: r["nodes"].__setitem__(2, saga_row("e2", ["m4"])), "['first_episode_uuid', 'started_at']"),
        (lambda r: named(ent("ana"), r)["props"].update(name_key="anna"), "['name_key']"),
    ],
    ids=["renamed", "new-label", "saga-start-moves", "re-keyed"],
)
def test_rule_2_refuses_a_frozen_field_change(edit, fields):
    rows = session()
    edit(rows)
    with pytest.raises(write.Refused, match=re.escape(f"would change frozen fields {fields}")):
        gate(rows)


def test_rule_2_holds_a_replayed_message_to_what_is_stored():
    stored = bootstrap()
    stored["nodes"] += copy.deepcopy(session()["nodes"][:2])  # the first run landed m4 and m5
    rows = session()
    named(ep("m4"), rows)["props"]["content"] = "Ana: something else"
    with pytest.raises(write.Refused, match=re.escape("would change frozen fields ['content']")):
        gate(rows, stored)
    gate(session(), stored)  # the same replay passes


def test_rule_2_holds_the_writer_to_union_only_edges(monkeypatch):
    """Positive control: a writer that set a stored edge's props on match again is refused."""
    stored = bootstrap()
    stored["edges"].append(fact("m1", "A-old", "plan-old"))
    rows = session()
    rows["edges"].append(fact("m4", "B-new", "plan-new"))
    gate(rows, stored)  # the real writer only grows the union fields: passes
    monkeypatch.setattr(kg_writer, "edge_match_update", lambda update, *, append: update)
    with pytest.raises(write.Refused, match=rf"rule 2: stored RELATES_TO {FACT} would change .*'key_text'"):
        gate(rows, stored)


def test_rule_3_refuses_totals_the_chain_does_not_hold():
    # before the write: a stale message no session of the chain holds
    stored = bootstrap()
    stored["nodes"].append(episodic("m9"))
    with pytest.raises(write.PendingSession, match="the group holds 4 Episodic .the chain 3, this session's stored 0"):
        write.pending_check(
            {"Episodic": 4, "Saga": 2},
            stored["nodes"],
            {"messages": 3, "episodes": 2},
            episodic=[ep("m4"), ep("m5")],
            sagas=[saga("e3")],
        )
    # a crash replay of the same session: its own stored rows are subtracted, so it passes
    replay = bootstrap()
    replay["nodes"] += [episodic("m4"), episodic("m5"), saga_row("e3", ["m5"])]
    write.pending_check(
        {"Episodic": 5, "Saga": 3},
        replay["nodes"],
        {"messages": 3, "episodes": 2},
        episodic=[ep("m4"), ep("m5")],
        sagas=[saga("e3")],
    )
    # after the write: the gate's own total check
    with pytest.raises(write.Refused, match="forward gate rule 3: the group holds 6 Episodic rows, the chain and this"):
        gate(session(), totals={"Episodic": {"got": 6, "expected": 5}})
    gate(session(), totals={"Episodic": {"got": 5, "expected": 5}, "Saga": {"got": 3, "expected": 3}})


def test_a_forward_result_the_chain_cannot_take_is_refused_before_the_write():
    rows = session()
    rows["forward"]["routes"] = {"cy": ent("cy")}  # a route to no entity of the chain
    with pytest.raises(write.Refused, match="the chain cannot take this run's forward result .*lead to no entity"):
        gate(rows, fold=True)


def test_a_stored_questions_aggregates_survive_an_append():
    """The session's codings alone (fold_count 3) never overwrite a stored Question's fold (1); a Question first
    answered in this session keeps its own."""
    rows = session()
    from memory_kernel.graph.g1_ids import content_uuid

    fresh = {"uuid": content_uuid("q", GROUP, "q2"), "name": "Who glazed?", "qid": "q2", "fold_count": 2}
    rows["nodes"].append({"labels_neo4j": ["Question"], "props": fresh | {"group_id": GROUP, "created_at": "t"}})
    stored = {n["props"]["uuid"] for n in bootstrap()["nodes"]}
    assert write.keep_contract_aggregates(rows, stored) == 2  # the stored Purpose and q1
    assert "fold_count" not in named(QUESTION, rows)["props"]
    assert named(fresh["uuid"], rows)["props"]["fold_count"] == 2
