"""Kernel 2.0 state read (``graph.state_read``): a bonfire's chain rebuilt from its graph equals what 1.8 folds from
its folder.

The fixtures (``tests/fixtures/state/``) were frozen once, while 1.8's folder path still existed. On the 1.8 side,
the real projection (structure policy) of a two-chat bootstrap of three sessions and one appended session went
through ``chain.from_bootstrap`` / ``chain.advance``, was committed and loaded back (what the 1.8 chain handed the next
append), then ``_rows.view``. On the graph side, the same rows went through ``graph.state.stamp_state`` with their
Roleset / Route / BonfireMeta rows, were written by the real ``kg_writer.write_rows`` (full build, then append mode)
into a test Neo4j, and were read back with ``read_group``. The append's session adds the agent handle ``@scribe``,
which a bootstrap message had named. Tests read only the frozen files. Invented names only."""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from memory_kernel.graph.kg_writer import _label_clause
from memory_kernel.graph.state_read import (
    StateRows,
    chain_from_rows,
    read_chain,
    read_meta,
    read_state_rows,
    state_view,
)
from memory_kernel.pipeline.chain import ChainError, closed_boundary
from unit.graph._rows import dump, view

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "state"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def rows(name: str) -> tuple[list[dict], list[dict], dict]:
    doc = load(name)
    return doc["nodes"], doc["edges"], doc["meta"]


def chain_of(name: str):
    return chain_from_rows(state_view(*rows(name)))


def uuid_named(state: StateRows, name: str) -> str:
    return next(e["uuid"] for e in state.entities if e["name"] == name)


def test_chain_from_rows_equals_the_bootstrap_chain():
    assert dump(view(chain_of("bootstrap-rows.json"))) == (FIXTURES / "bootstrap-view.json").read_text()


def test_chain_from_rows_equals_the_chain_after_an_append():
    assert dump(view(chain_of("append-rows.json"))) == (FIXTURES / "append-view.json").read_text()


def test_roster_keeps_bootstrap_persons_and_later_speakers_only():
    state = state_view(*rows("append-rows.json"))
    chain = chain_from_rows(state)
    k2 = [e for e in state.episodic if e["chat"] == "k2"]
    said = {
        n
        for e in k2
        for n in (json.loads(e["source_row_json"])["speaker"], *json.loads(e["source_row_json"])["addressees"])
    }
    bo, eli = uuid_named(state, "Bo"), uuid_named(state, "Eli")
    # Bo: a Person of k1 that a k2 message of run 0 mentions joins k2's roster without speaking there
    assert any(bo in e["mentions"] and e["run"] == 0 for e in k2) and "Bo" not in said
    # Eli: a Person (minted at run 0 in k1) that a k2 message of run 1 mentions is NOT added
    assert any(eli in e["mentions"] and e["run"] == 1 for e in k2) and "Eli" not in said
    assert chain.chats["k2"].roster == ["Dov", "scribe", "Bo", "Ana"]


def test_last_at_and_evidence_quote_the_select_timestamp_verbatim():
    nodes, edges, meta = rows("append-rows.json")
    select = {}
    for n in nodes:
        if "Episodic" in n["labels_neo4j"]:
            row = json.loads(n["props"]["source_row_json"])
            select[row["message_id"]] = row["timestamp"]
            n["props"] |= {"valid_at": "2001-01-01T00:00:00Z", "told_at": "2001-01-01T00:00:00Z"}  # never read
    chain = chain_from_rows(state_view(nodes, edges, meta))
    assert chain.ground and all(g.last_at == select[g.last_mid] for g in chain.ground.values())
    assert all(g.last_at.endswith("+00:00") for g in chain.ground.values())
    assert chain.evidence and all(row["timestamp"] == select[mid] for mid, row in chain.evidence.items())
    assert dump(view(chain)) == (FIXTURES / "append-view.json").read_text()


def test_tail_row_is_the_select_row():
    state = state_view(*rows("append-rows.json"))
    chain = chain_from_rows(state)
    for c, chat in chain.chats.items():
        msgs = [e for e in state.episodic if e["chat"] == c]
        last = max(msgs, key=lambda e: e["order"])
        assert chat.tail == [json.loads(last["source_row_json"])]
        assert chat.boundary == closed_boundary(len(msgs)) and chat.last_order == last["order"]


def test_ground_mentions_count_sender_and_addressee():
    nodes, edges, meta = rows("append-rows.json")
    bare = state_view(nodes, [e for e in edges if e["type"] != "MENTIONS"], meta)
    entities = {e["uuid"] for e in bare.entities}
    named = Counter(u for e in bare.episodic for u in {e["sender"], e["addressee"]} & entities)
    assert {u: g.mentions for u, g in chain_from_rows(bare).ground.items()} == dict(named)
    state = state_view(nodes, edges, meta)
    ana = uuid_named(state, "Ana")
    addressed = [e for e in state.episodic if e["addressee"] == ana and ana not in e["mentions"]]
    assert addressed  # Ana is the addressee of a message that does not mention her
    both = {e["message_id"] for e in state.episodic if ana in {e["sender"], e["addressee"], *e["mentions"]}}
    assert chain_from_rows(state).ground[ana].mentions == len(both)


def test_ground_people_leave_out_only_the_handles_known_at_each_run():
    """``@scribe`` joins at run 1 and b4 (Dov -> ["scribe"]) is of run 0. As in 1.8, the people of the entities b4
    names keep "scribe"; a fold against the final handle union would drop it."""
    nodes, edges, meta = rows("append-rows.json")
    state = state_view(nodes, edges, meta)
    dov = uuid_named(state, "Dov")
    chain = chain_from_rows(state)
    assert meta["agent_handles"] == ["@helper", "@scribe"] and meta["agent_handle_runs"] == [0, 1]
    assert chain.ground[dov].people == ("dov", "scribe", "ana")
    assert not any("helper" in g.people for g in chain.ground.values())  # a run-0 handle: never a person
    known_early = chain_from_rows(state_view(nodes, edges, {**meta, "agent_handle_runs": [0, 0]}))
    assert "scribe" not in known_early.ground[dov].people


def test_evidence_keeps_a_superseded_last_mention():
    """Eli's run-0 last mention (b7) is superseded at run 1 (a2); 1.8's evidence is append-only, so b7 stays."""
    chain = chain_of("append-rows.json")
    ends = {m for g in chain.ground.values() for m in (g.first_mid, g.last_mid)}
    assert "b7" in chain.evidence and "b7" not in ends


def test_settled_rolesets_carry_label_and_p_yes():
    state = state_view(*rows("append-rows.json"))
    like, meet, own = (state.rolesets[rs] for rs in ("like.01", "meet.01", "own.01"))
    assert (like["label"], like["p_yes"]["occurrence"], like["compiled_sha256"], like["run"]) == (
        "occurrence",
        0.9,
        "1a" * 32,
        0,
    )
    assert (meet["label"], meet["p_yes"], meet["compiled_sha256"]) == (None, None, None)
    assert (own["label"], own["run"]) == ("state", 1)
    chain = chain_from_rows(state)
    assert chain.rolesets["like.01"] == {"gloss": like["gloss"], "examples": like["examples"], "run": 0}
    assert chain.rolesets["own.01"]["run"] == 1


def test_kinds_follow_the_referent_policy():
    nodes, edges, meta = rows("append-rows.json")
    state = state_view(nodes, edges, meta)
    assert set(chain_from_rows(state).kinds) == {e["uuid"] for e in state.entities}
    config = {**json.loads(meta["build_config_json"]), "referent_policy": "phrase"}
    phrase = {**meta, "build_config_json": json.dumps(config, sort_keys=True)}
    assert chain_from_rows(state_view(nodes, edges, phrase)).kinds is None


# -- the batched read on a live Neo4j ----------------------------------------------------------------------------------


class _Counting:
    """A driver whose sessions count the queries they run (the read must not grow with the group)."""

    def __init__(self, driver) -> None:
        self.driver, self.queries = driver, []

    def session(self, **kwargs):
        return _CountingSession(self, self.driver.session(**kwargs))


class _CountingSession:
    def __init__(self, owner: _Counting, inner) -> None:
        self.owner, self.inner = owner, inner

    async def __aenter__(self):
        self.session = await self.inner.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self.inner.__aexit__(*exc)

    async def run(self, query, **params):
        self.owner.queries.append(query)
        return await self.session.run(query, **params)


def regroup(doc: tuple[list[dict], list[dict], dict], group: str) -> tuple[list[dict], list[dict], dict]:
    nodes, edges, meta = doc
    nodes = [{**n, "props": {**n["props"], "group_id": group}} for n in nodes]
    edges = [{**e, "props": {**e["props"], "group_id": group}} for e in edges]
    return nodes, edges, {**meta, "group_id": group}


async def put(driver, group: str, nodes: list[dict], edges: list[dict]) -> None:
    """The rows as they are (no writer): one CREATE per label set and per edge type."""
    by_labels: defaultdict[tuple[str, ...], list[dict]] = defaultdict(list)
    for n in nodes:
        by_labels[tuple(n["labels_neo4j"])].append(n["props"])
    by_type: defaultdict[str, list[dict]] = defaultdict(list)
    for e in edges:
        by_type[e["type"]].append({"s": e["source"], "t": e["target"], "p": e["props"]})
    async with driver.session() as s:
        await (await s.run("MATCH (n {group_id: $g}) DETACH DELETE n", g=group)).consume()
        for labels, batch in by_labels.items():
            query = f"UNWIND $rows AS p CREATE (n:{_label_clause(labels)}) SET n = p"
            await (await s.run(query, rows=batch)).consume()
        for etype, batch in by_type.items():
            query = (
                "UNWIND $rows AS r MATCH (a {uuid: r.s, group_id: $g}) MATCH (b {uuid: r.t, group_id: $g}) "
                f"CREATE (a)-[e:{_label_clause([etype])}]->(b) SET e = r.p"
            )
            await (await s.run(query, rows=batch, g=group)).consume()


@pytest.mark.needs_neo4j
def test_read_state_rows_equals_state_view(neo4j_target):
    from neo4j import AsyncGraphDatabase

    async def scenario():
        driver = AsyncGraphDatabase.driver(neo4j_target.uri, auth=(neo4j_target.user, neo4j_target.password))
        group = neo4j_target.group
        counts = []
        try:
            assert await read_meta(driver, group) is None
            with pytest.raises(ChainError, match="no BonfireMeta"):
                await read_state_rows(driver, group)
            for name in ("bootstrap-rows.json", "append-rows.json"):
                nodes, edges, meta = regroup(rows(name), group)
                await put(driver, group, nodes, edges)
                counting = _Counting(driver)
                assert await read_state_rows(counting, group) == state_view(nodes, edges, meta)
                counts.append(len(counting.queries))
                assert await read_meta(driver, group) == meta
                frozen = (FIXTURES / name.replace("-rows", "-view")).read_text()
                assert dump(view(await read_chain(driver, group))) == frozen
        finally:
            await driver.close()
        assert counts[0] == counts[1] <= 8  # batched: the same few queries whatever the group's size

    asyncio.run(scenario())
