import asyncio
import re
from collections.abc import Mapping
from typing import Any

import pytest

from memory_kernel.runtime import ops_memory
from memory_kernel.runtime.ops_memory import MEMORY_SPECS, StoreMemory


class FakeReader:
    """Canned rows keyed by a marker found in the Cypher; records every query."""

    def __init__(self, canned: Mapping[str, list[dict]]) -> None:
        self.canned = canned
        self.sent: list[tuple[str, dict]] = []

    async def rows(self, cypher: str, params: Mapping[str, Any]) -> list[dict]:
        self.sent.append((cypher, dict(params)))
        for marker, rows in self.canned.items():
            if marker in cypher:
                return rows
        return []


KIRA = {"uuid": "u1", "name": "Kira Dace", "labels": ["Entity", "Person"]}
ZORVEX = {"uuid": "u2", "name": "Zorvex", "labels": ["Entity", "Organisation"]}
BRASK = {"uuid": "u3", "name": "Brask Labs", "labels": ["Entity", "Place"]}


def run(memory: StoreMemory, op: str, **args: Any) -> Any:
    return asyncio.run(memory.run(op, args))


def test_find_entity_matches_name_exactly_then_case_insensitively():
    reader = FakeReader({"toLower(n.name) = toLower": [KIRA]})
    memory = StoreMemory(reader, "g1")
    assert run(memory, "find-entity", name="kira dace") == [{"uuid": "u1", "name": "Kira Dace", "kinds": ["Person"]}]
    assert len(reader.sent) == 2 and "n.name = $name" in reader.sent[0][0]
    exact = FakeReader({"n.name = $name": [KIRA]})
    run(StoreMemory(exact, "g1"), "find-entity", name="Kira Dace")
    assert len(exact.sent) == 1  # the folded query is not needed
    assert reader.sent[0][1]["g"] == "g1"


def test_slot_of_kind_reads_kind_labels():
    reader = FakeReader({"$kind IN labels": [ZORVEX]})
    memory = StoreMemory(reader, "g1")
    out = run(memory, "slot-of-kind", lines=["[Ana, 2026-01-02] Kira Dace works at Zorvex."], kind="Organisation")
    assert out == [{"uuid": "u2", "name": "Zorvex", "kinds": ["Organisation"]}]
    assert reader.sent[0][1]["kind"] == "Organisation"
    # a candidate the lines do not name as a whole word is dropped
    assert run(memory, "slot-of-kind", lines=["[Ana] Kira Dace likes Zorvexine."], kind="Organisation") == []
    place = FakeReader({"$kind IN labels": [BRASK]})
    run(StoreMemory(place, "g1"), "slot-of-kind", lines=["[Ana] Milo visited Brask Labs."], kind="Place")
    assert place.sent[0][1]["kind"] == "Place"


def test_events_take_a_set_of_frames_and_role_filler_takes_wh_role_pairs():
    """Verb-sense ambiguity travels as a set: events reads any of the frames; role-filler reads the (frame, role)
    pairs wh-role returns, a list of roles, or one role."""
    reader = FakeReader({"$pairs": [dict(ZORVEX, event="e1")], "$frames": [{"uuid": "e1"}]})
    memory = StoreMemory(reader, "g1")
    assert run(memory, "events", frame=["work.01", "work.02"], participant=None, role=None) == [{"uuid": "e1"}]
    assert reader.sent[-1][1]["frames"] == ["work.01", "work.02"]
    run(memory, "events", frame="work.01", participant=None, role=None)
    assert reader.sent[-1][1]["frames"] == ["work.01"]
    sent = len(reader.sent)
    assert run(memory, "events", frame=[], participant=None, role=None) == []  # no frame: nothing to read
    assert len(reader.sent) == sent
    out = run(memory, "role-filler", event=[{"uuid": "e1"}], role=[("work.01", "arg2"), ["work.01", "argm-loc"]])
    assert [e["name"] for e in out] == ["Zorvex"]
    assert reader.sent[-1][1]["pairs"] == [["work.01", "arg2"], ["work.01", "argm-loc"]]
    run(memory, "role-filler", event=[{"uuid": "e1"}], role="arg1")
    assert reader.sent[-1][1]["pairs"] == [[None, "arg1"]]


def test_recall_wraps_context_callable():
    seen: list[str] = []

    async def recall(text: str) -> dict:
        seen.append(text)
        return {"context": "Kira Dace leads Zorvex."}

    memory = StoreMemory(FakeReader({}), "g1", recall=recall)
    assert run(memory, "recall", text="who is Kira Dace") == {"context": "Kira Dace leads Zorvex."}
    assert seen == ["who is Kira Dace"]
    with pytest.raises(RuntimeError, match="recall"):
        run(StoreMemory(FakeReader({}), "g1"), "recall", text="x")


def test_identity_lines_and_state_lines_read_the_line_records(monkeypatch, caplog):
    from memory_kernel.search import summary_lines

    monkeypatch.setattr(summary_lines, "_NOTED", set())

    def boom(*_a, **_k):
        raise AssertionError("a stored line was parsed")

    monkeypatch.setattr("memory_kernel.graph.g1_nlp.query_doc", boom)
    row = {
        "uuid": "u1",
        "name": "Kira Dace",
        "line_ids": ["a", "b"],
        "line_texts": ["[Ana, 2026-01-02] Kira Dace is an engineer.", "[Ana, 2026-01-03] Kira Dace left."],
        "identity_line_ids": ["a"],
    }
    memory = StoreMemory(FakeReader({"n.summary": [row]}), "g1")
    expected = ["[Ana, 2026-01-02] Kira Dace is an engineer."]
    assert run(memory, "identity-lines", entity=[{"uuid": "u1"}]) == expected
    assert run(memory, "state-lines", entity="u1", frame=None) == expected  # no line frames yet
    legacy = StoreMemory(FakeReader({"n.summary": [{"uuid": "u1", "name": "Kira Dace", "summary": "x is a y."}]}), "g2")
    lineless = StoreMemory(FakeReader({"n.summary": [{"uuid": "u2", "name": "Milo", "summary": ""}]}), "g3")
    with caplog.at_level("WARNING"):
        assert run(legacy, "identity-lines", entity=[{"uuid": "u1"}]) == []
        assert run(lineless, "identity-lines", entity=[{"uuid": "u2"}]) == []
    notes = [r.message for r in caplog.records if "identity lines unavailable" in r.message]
    assert any("g2" in m for m in notes) and not any("g3" in m for m in notes)  # an entity with no lines is not legacy


def test_pure_ops_count_and_order():
    memory = StoreMemory(FakeReader({}), "g1")
    items = [{"uuid": "a", "happened_at": "2026-03-01"}, {"uuid": "b", "happened_at": "2026-01-01"}, {"uuid": "c"}]
    assert run(memory, "count", set=items) == 3
    assert run(memory, "order", set=items, end="first")["uuid"] == "b"
    assert run(memory, "order", set=items, end="latest")["uuid"] == "a"
    with pytest.raises(KeyError):
        run(memory, "bogus")


def test_ops_are_read_only():
    row = {"uuid": "u1", "name": "Kira Dace", "labels": ["Person"], "summary": "[Ana] x", "event": "e1"}
    reader = FakeReader({"MATCH": [row]})
    memory = StoreMemory(reader, "g1")
    ev, ent = [{"uuid": "e1"}], [{"uuid": "u1"}]
    calls = {
        "find-entity": {"name": "x"},
        "events": {"frame": "make.01", "participant": ent, "role": "arg0"},
        "role-filler": {"event": ev, "role": "arg1"},
        "identity-lines": {"entity": ent},
        "state-lines": {"entity": ent, "frame": None},
        "slot-of-kind": {"lines": ["Kira Dace"], "kind": "Organisation"},
        "time-of": {"event": ev},
        "source": {"item": ev},
        "mentions": {"entity": ent},
    }
    assert set(calls) | {"count", "order", "recall"} == set(MEMORY_SPECS)
    for op, args in calls.items():
        run(memory, op, **args)
    assert reader.sent and memory.queries
    for cypher, _ in reader.sent:
        assert not re.search(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DETACH)\b", cypher, re.IGNORECASE)


def test_slot_keeps_longest_non_overlapping_matches():
    longer = {"uuid": "u9", "name": "Quillfern team", "labels": ["Organisation"]}
    team = {"uuid": "u5", "name": "team", "labels": ["Organisation"]}
    reader = FakeReader({"$kind IN labels": [team, longer]})
    out = run(
        StoreMemory(reader, "g1"), "slot-of-kind", lines=["[Ana] Milo joined the Quillfern team."], kind="Organisation"
    )
    assert [e["name"] for e in out] == ["Quillfern team"]


def test_missing_input_is_a_clear_error():
    memory = StoreMemory(FakeReader({}), "g1")
    with pytest.raises(ValueError, match="frame"):
        run(memory, "events", frame=None)


def test_order_by_keys_are_projected_aliases():
    queries = {k: v for k, v in vars(ops_memory).items() if k.endswith("_CYPHER") or k.startswith("FIND_ENTITY")}
    assert len(queries) >= 9
    for name, cypher in queries.items():
        ret = cypher.split("RETURN", 1)[1].split("ORDER BY")[0]
        aliases = set(re.findall(r"\bAS\s+(\w+)", ret))
        if "ORDER BY" in cypher:
            keys = cypher.split("ORDER BY", 1)[1].split("LIMIT")[0]
            for key in (k.strip().split()[0] for k in keys.split(",")):
                assert key in aliases, f"{name}: ORDER BY {key} is not a projected alias"


@pytest.mark.needs_neo4j
def test_all_ops_run_read_only_on_live_graph():
    import os

    uri, group = os.environ.get("MK_TEST_NEO4J_URI"), os.environ.get("MK_TEST_NEO4J_GROUP")
    if not (uri and group):
        pytest.skip("MK_TEST_NEO4J_URI / MK_TEST_NEO4J_GROUP not set")
    from neo4j import AsyncGraphDatabase

    user = os.environ.get("MK_TEST_NEO4J_USER", "neo4j")

    async def scenario() -> None:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, os.environ.get("MK_TEST_NEO4J_PASSWORD", "")))

        class LiveReader:
            async def rows(self, cypher, params):
                async with driver.session(default_access_mode="READ") as s:
                    result = await s.run(cypher, **params)
                    return [dict(r) async for r in result]

        reader = LiveReader()

        async def recall(text: str) -> dict:
            return {"text": text}

        try:
            ent = await reader.rows(
                "MATCH (n:Entity {group_id: $g}) WHERE NOT n:Event AND n.summary IS NOT NULL "
                "RETURN n.uuid AS uuid, n.name AS name LIMIT 1",
                {"g": group},
            )
            rs = await reader.rows(
                "MATCH (e:Entity:Event {group_id: $g}) WHERE e.roleset IS NOT NULL "
                "RETURN e.uuid AS uuid, e.roleset AS roleset LIMIT 1",
                {"g": group},
            )
            assert ent, "graph has no entity"
            entity = {"uuid": ent[0]["uuid"], "name": ent[0]["name"]}
            frame = rs[0]["roleset"] if rs else "none.00"
            memory = StoreMemory(reader, group, recall=recall)
            found = await memory.run("find-entity", {"name": entity["name"]})
            evs = await memory.run("events", {"frame": frame, "participant": None, "role": None})
            lines = await memory.run("identity-lines", {"entity": entity})
            calls = {
                "role-filler": {"event": evs, "role": "arg0"},
                "state-lines": {"entity": entity, "frame": None},
                "slot-of-kind": {"lines": lines or [entity["name"]], "kind": "Organisation"},
                "time-of": {"event": evs},
                "source": {"item": evs or entity},
                "mentions": {"entity": entity},
                "count": {"set": found},
                "order": {"set": evs, "end": "latest"},
                "recall": {"text": "x"},
            }
            for op, args in calls.items():
                await memory.run(op, args)
            assert len(calls) + 3 == len(MEMORY_SPECS)
        finally:
            await driver.close()

    asyncio.run(scenario())
