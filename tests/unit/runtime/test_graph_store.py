"""The runtime's state on the graph: the narrow writer's statements, the store over it (live rows persist, drafts do
not, counters are added), the rolesets read, and the writer against a real Neo4j test group."""

from __future__ import annotations

import copy
import dataclasses
import re

import pytest

from memory_kernel.graph.state import meta_props, meta_row
from memory_kernel.runtime import graph_store
from memory_kernel.runtime.constructions import CxnSpec
from memory_kernel.runtime.graph_store import WRITES, NarrowWriter
from memory_kernel.runtime.lexicon import load_rolesets
from memory_kernel.runtime.store import ConstructionStore, StoreError
from unit.runtime.memory_writer import MemoryWriter, memory_store

X = {"slot": "x", "fills": "np"}
BREW = CxnSpec(id="brew.d1.0", kind="taught", unit="root", form=["brew ", X], meaning=[["const", "?x", "tea"]])
LIVE = dataclasses.replace(BREW, status="live")
DRAFT = dataclasses.replace(BREW, id="brew.d2.0", status="staged")


def test_narrow_writer_touches_only_taught_phrase_and_bonfire_meta():
    assert len(WRITES) == 5
    for statement in WRITES:
        labels = set(re.findall(r"\(\w*:(\w+)", statement))
        assert labels <= {"TaughtPhrase", "BonfireMeta"}, statement
        assert not re.search(r"\bCREATE\b(?! SET)", statement.upper())  # MERGE ... ON CREATE SET only
    meta = [w for w in WRITES if "BonfireMeta" in w]
    assert len(meta) == 1
    assert re.findall(r"\bSET\s+(\w+\.\w+)", meta[0]) == ["m.catalogue_json"]
    assert set(re.findall(r"\bm\.(\w+)", meta[0])) <= {"catalogue_json", "group_id"}
    assert not any("Roleset" in w or "Entity" in w or "Episodic" in w for w in WRITES)


def test_confirmed_phrase_persists_and_staged_draft_does_not():
    writer = MemoryWriter()
    store = ConstructionStore(writer)
    store.put(DRAFT)  # a draft: this process's memory only
    assert writer.nodes == {} and [s.id for s in store.staged()] == [DRAFT.id]
    live = dataclasses.replace(LIVE, standing=dataclasses.replace(LIVE.standing, confirmed=1))
    store.update(puts=[live])
    assert sorted(writer.nodes) == [LIVE.id]
    again = ConstructionStore(writer)  # another process, or this one after a restart
    assert [s.id for s in again.all()] == [LIVE.id] and again.get(LIVE.id).standing.confirmed == 1
    assert again.staged() == []  # the draft did not travel
    spec_json = writer.nodes[LIVE.id]["spec_json"]
    assert '"confirmed": 0' in spec_json  # the row holds standing zeroed; the counters sit on the node


def test_a_staged_use_is_counted_in_memory_only_and_a_read_only_store_refuses_writes():
    writer = MemoryWriter()
    store = memory_store(DRAFT, writer=writer)
    assert store.record_uses({DRAFT.id: "ok"})[DRAFT.id].standing.uses_ok == 1
    assert writer.calls == []
    read_only = memory_store(LIVE, read_only=True)
    with pytest.raises(StoreError, match="read-only"):
        read_only.record_uses({LIVE.id: "ok"})
    with pytest.raises(StoreError, match="read-only"):
        read_only.put(LIVE)


def test_record_uses_adds_counters():
    writer = MemoryWriter()
    one = memory_store(LIVE, writer=writer)
    two = ConstructionStore(writer)
    one.record_uses({LIVE.id: "ok"})
    one.record_uses({LIVE.id: "failed"})
    two.record_uses({f"{LIVE.id}#v2": "world"})  # an alternate's name counts for its spec
    assert writer.calls[-1] == ("add_uses", {LIVE.id: (0, 0, 1)})
    node = writer.nodes[LIVE.id]
    assert (node["uses_ok"], node["uses_failed"], node["uses_world"]) == (1, 1, 1)
    standing = two.get(LIVE.id).standing
    assert (standing.uses_ok, standing.uses_failed, standing.uses_world) == (1, 1, 1)  # the counters now stored
    with pytest.raises(KeyError):
        one.record_uses({"nothing-here": "ok"})
    with pytest.raises(StoreError, match="unknown outcome"):
        one.record_uses({LIVE.id: "maybe"})


def test_a_removed_taught_phrase_leaves_the_graph_only_by_a_confirmed_replace():
    writer = MemoryWriter()
    store = memory_store(LIVE, writer=writer)
    with pytest.raises(StoreError, match="confirmed replace"):
        store.update(removes=[LIVE.id])
    store.update(removes=[LIVE.id], replacing_taught=True)
    assert writer.nodes == {}


def test_store_rules_load_order_and_what_a_turn_writes():
    """Live rows persist sorted by id; a re-mine never overwrites a taught row; base rows never enter the store;
    ``load_order`` is the base set and the live rows in ascending precedence; ``record_uses`` writes once for a turn,
    counts an alternate's name for its spec and skips base rows and attached cards; non-ASCII and the line separators
    survive the graph."""
    from memory_kernel.runtime.base import base_specs
    from memory_kernel.runtime.constructions import precedence

    def row(cxn_id, kind, **extra):
        return CxnSpec(id=cxn_id, kind=kind, unit="root", form=["zork", X, "?"], meaning=[["answer", "?x"]], **extra)

    curly = dataclasses.replace(row("m-job", "mined", status="live"), form=["what is", X, "\u2019s job\u2028?"])
    draft = row("a-draft", "taught", status="staged")
    writer = MemoryWriter()
    store = ConstructionStore(writer)
    for spec in (row("zz-taught", "taught", status="live"), curly, draft, row("h-hand", "hand", status="live")):
        store.put(spec)
    assert sorted(writer.nodes) == ["h-hand", "m-job", "zz-taught"]  # the draft stays in memory
    fresh = ConstructionStore(writer)
    assert [s.id for s in fresh.all()] == ["h-hand", "m-job", "zz-taught"] and fresh.get("m-job").form == curly.form
    order = fresh.load_order()
    assert order == sorted(order, key=precedence)
    assert {s.id for s in order} == {s.id for s in base_specs()} | {"h-hand", "m-job", "zz-taught"}
    assert [s.kind for s in order[-3:]] == ["mined", "hand", "taught"]  # base < mined < hand < taught
    with pytest.raises(StoreError, match="taught"):
        fresh.put(row("zz-taught", "mined", status="live"))
    with pytest.raises(StoreError, match="base"):
        fresh.put(row("np-bare", "hand", status="live"))
    with pytest.raises(StoreError, match="base"):
        fresh.put(row("b-row", "base", status="live"))
    fresh.attach_cards([row("card-give", "hand", status="live")])
    writer.calls.clear()
    updated = fresh.record_uses({"m-job#v1": "ok", "zz-taught": "failed", "np-bare": "ok", "card-give": "ok"})
    assert set(updated) == {"m-job", "zz-taught"} and len(writer.calls) == 1  # one write for the turn
    assert fresh.record_use("np-bare", "ok") is None and len(writer.calls) == 1  # base: not stored
    assert ConstructionStore(writer).get("m-job").standing.uses_ok == 1
    with pytest.raises(KeyError):
        fresh.record_uses({"m-job": "ok", "no-such-cxn": "ok"})
    assert len(writer.calls) == 1  # nothing written


def test_a_failing_standing_write_never_fails_the_turn():
    """The turn's outcome is recorded on the graph; a Neo4j error there leaves the computed plan standing."""
    import asyncio

    from memory_kernel.runtime.api import Runtime, _Plan, unread_plan
    from memory_kernel.runtime.executor import Turn
    from memory_kernel.runtime.lexicon import Lexicon, QuestionMap

    class Unavailable(Exception):  # stands in for neo4j.exceptions.ServiceUnavailable
        pass

    writer = MemoryWriter()
    store = memory_store(LIVE, writer=writer)
    writer.fail = Unavailable("down")
    runtime = Runtime("ember", object(), object(), store=store, lexicon=Lexicon([], QuestionMap.load()))  # type: ignore[arg-type]
    turn = Turn("brew tea", "Ana", None, (), "c1", None, "2026-10-30T12:00:00+00:00")
    record = _Plan(unread_plan("ember", turn, []), 1, 0.0, turn, True, uses=frozenset({LIVE.id}))
    asyncio.run(runtime._end(record))
    assert record.recorded and any(d.startswith("standing not recorded") for d in record.plan.diagnostics)
    assert writer.nodes[LIVE.id]["uses_ok"] == 0 and writer.nodes[LIVE.id]["uses_failed"] == 0


def test_puts_are_admitted_against_the_rows_before_the_removals():
    writer = MemoryWriter()
    store = memory_store(LIVE, writer=writer)
    nodes, calls = copy.deepcopy(writer.nodes), list(writer.calls)
    with pytest.raises(StoreError, match="taught"):
        store.update(puts=[dataclasses.replace(LIVE, kind="mined")], removes=[LIVE.id])  # remove, then put over its id
    with pytest.raises(StoreError, match="replace"):
        store.update(removes=[LIVE.id])
    assert writer.nodes == nodes and writer.calls == calls and store.get(LIVE.id).kind == "taught"


def test_line_separators_in_provenance_survive_the_graph():
    seps = ("\u2028", "\u2029", "\x85")
    writer = MemoryWriter()
    store = ConstructionStore(writer)
    for n, sep in enumerate(seps):
        request = {"forms": ["zork {X}?"], "body": [f"give {{X}} to me{sep}"], "defaults": {}}
        provenance = {"teacher": f"Kira{sep}Dace", "chat_id": "c1", "request": request}
        store.put(dataclasses.replace(LIVE, id=f"t-{n}", provenance=provenance))
    again = ConstructionStore(writer)
    assert again.all() == store.all()
    for n, sep in enumerate(seps):
        prov = again.get(f"t-{n}").provenance
        assert prov["teacher"] == f"Kira{sep}Dace" and prov["request"]["body"] == [f"give {{X}} to me{sep}"]


class SyncDriver:
    """A recording sync driver: ``session().run`` returns the rows ``answers`` maps its query to."""

    def __init__(self, answers):
        self.answers, self.queries = answers, []

    def session(self):
        driver = self

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def run(self, cypher, **params):
                driver.queries.append((cypher, params))
                return driver.answers[cypher]

        return Session()


def test_lexicon_reads_roleset_nodes():
    driver = SyncDriver({graph_store.READ_ROLESETS: [{"roleset": "work.01"}, {"roleset": "give.01"}]})
    assert load_rolesets(driver, "ember") == ["give.01", "work.01"]
    assert driver.queries == [(graph_store.READ_ROLESETS, {"g": "ember"})]
    assert load_rolesets(SyncDriver({graph_store.READ_ROLESETS: []}), "ember") == []  # none: every lemma is served


# -- against a real Neo4j test group -----------------------------------------------------------------------------------


def _driver(target):
    from neo4j import GraphDatabase

    return GraphDatabase.driver(target.uri, auth=(target.user, target.password))


@pytest.mark.needs_neo4j
def test_standing_counters_add_up_across_two_writers(neo4j_target):
    with _driver(neo4j_target) as d1, _driver(neo4j_target) as d2:
        first, second = NarrowWriter(d1, neo4j_target.group), NarrowWriter(d2, neo4j_target.group)
        first.put_phrases([LIVE])
        first.add_uses({LIVE.id: (2, 0, 0)})
        second.add_uses({LIVE.id: (1, 3, 1)})
        stored = first.add_uses({LIVE.id: (0, 1, 0)})[LIVE.id]
        assert (stored.uses_ok, stored.uses_failed, stored.uses_world) == (3, 4, 1)
        [phrase] = second.phrases()
        assert phrase.id == LIVE.id and phrase.standing == stored
        second.add_confirmed([LIVE.id, LIVE.id])
        assert first.phrases()[0].standing.confirmed == 2
        first.put_phrases([dataclasses.replace(LIVE, meaning=[["const", "?x", "coffee"]])])  # the row, not the counters
        [after] = first.phrases()
        assert after.meaning == [["const", "?x", "coffee"]] and after.standing.uses_ok == 3
        assert first.add_uses({"gone": (1, 0, 0)}) == {}
        second.remove_phrases([LIVE.id])
        assert first.phrases() == []


@pytest.mark.needs_neo4j
def test_catalogue_is_set_on_the_bonfire_meta_alone(neo4j_target):
    group = neo4j_target.group
    props = meta_row(
        group,
        meta_props(
            bonfire_id=group,
            build_config={"sample_id": group},
            recipe_tag="t",
            extraction_model={"model": "m"},
            agent_handles=[],
            agent_handle_runs=[],
            chat_tagged=False,
            explicit_since=None,
            sessions=1,
            committed_run=3,
            generation="gen-1",
        ),
    )["props"]
    with _driver(neo4j_target) as driver:
        driver.execute_query("CREATE (m:BonfireMeta) SET m = $p", p=props)
        writer = NarrowWriter(driver, group)
        assert writer.catalogue() is None  # absent until the first PUT
        writer.set_catalogue('{"schema_version": "kernel-catalogue/v1", "tools": []}')
        assert writer.catalogue() == '{"schema_version": "kernel-catalogue/v1", "tools": []}'
        [row] = driver.execute_query("MATCH (m:BonfireMeta {group_id: $g}) RETURN properties(m) AS p", g=group).records
        assert {k: v for k, v in row["p"].items() if k != "catalogue_json"} == {
            k: v for k, v in props.items() if v is not None
        }  # nothing else moved
        with pytest.raises(LookupError):
            NarrowWriter(driver, group + "-none").set_catalogue("{}")


@pytest.mark.needs_neo4j
def test_update_is_one_transaction(neo4j_target, monkeypatch):
    """Remove + put + confirm in one ``update``: when the last statement fails, nothing the earlier ones did stays."""
    other = dataclasses.replace(LIVE, id="brew.d9.0")
    with _driver(neo4j_target) as driver:
        writer = NarrowWriter(driver, neo4j_target.group)
        writer.put_phrases([LIVE])
        writer.add_uses({LIVE.id: (1, 0, 0)})

        def state():
            return [(p.id, p.standing, p.meaning) for p in writer.phrases()]

        before = state()
        monkeypatch.setattr(graph_store, "ADD_CONFIRMED", "THIS IS NOT CYPHER")
        changed = dataclasses.replace(LIVE, meaning=[["const", "?x", "coffee"]])
        with pytest.raises(Exception):  # noqa: B017, PT011 - a Neo4j syntax error
            writer.update(puts=[changed, other], removes=[LIVE.id], confirmed=[LIVE.id])
        assert state() == before
