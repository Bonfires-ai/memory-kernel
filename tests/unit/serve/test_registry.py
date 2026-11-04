"""``BonfireRegistry``: a bonfire is a group with a ``BonfireMeta`` node, read from Neo4j alone. The ``needs_neo4j``
tests run against the test container through the ``neo4j_target`` fixture (its own ``mk20-test-<hex>`` group); the
rest use a recording fake driver."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from memory_kernel.graph.state import meta_props, meta_row
from memory_kernel.graph.state_read import META
from memory_kernel.serve import http, registry
from memory_kernel.serve.registry import BonfireRegistry
from unit.serve.fakes import Fakes


class FakeDriver:
    """Records every query; ``answers`` maps a query to the rows it returns."""

    def __init__(self, answers: dict[str, list[dict[str, Any]]]) -> None:
        self.answers = answers
        self.queries: list[tuple[str, dict]] = []

    async def close(self) -> None:
        return None

    def session(self):
        driver = self

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def run(self, cypher, params=None, **kw):
                driver.queries.append((cypher, {**dict(params or {}), **kw}))
                rows = driver.answers[cypher]

                async def gen():
                    for r in rows:
                        yield r

                return gen()

        return Session()


def props(group: str, run: int = 0) -> dict[str, Any]:
    return meta_row(
        group,
        meta_props(
            bonfire_id=group,
            build_config={"sample_id": group},
            recipe_tag="legacy|deixis=on|v4",
            extraction_model={"model": "m"},
            agent_handles=["Rua"],
            agent_handle_runs=[0],
            chat_tagged=True,
            explicit_since=None,
            sessions=1,
            committed_run=run,
            generation="gen-1",
        ),
    )["props"]


# -- offline -----------------------------------------------------------------------------------------------------------


def test_ids_are_cached_for_refresh_s_and_meta_is_always_read_fresh():
    driver = FakeDriver({registry.IDS: [{"id": "b"}, {"id": "a"}, {"id": None}], META: [{"p": {"x": 1}}]})
    now = [100.0]
    reg = BonfireRegistry(driver, refresh_s=30.0, clock=lambda: now[0])

    async def scenario() -> None:
        assert await reg.ids() == ["a", "b"]
        now[0] += 29
        assert await reg.ids() == ["a", "b"]
        assert len([q for q, _ in driver.queries if q == registry.IDS]) == 1
        now[0] += 1
        await reg.ids()
        assert len([q for q, _ in driver.queries if q == registry.IDS]) == 2
        for _ in range(3):
            assert await reg.meta("a") == {"x": 1}
        assert len([q for q, _ in driver.queries if q == META]) == 3

    asyncio.run(scenario())
    assert registry.IDS == "MATCH (m:BonfireMeta) RETURN m.group_id AS id"
    assert "MATCH (m:BonfireMeta {group_id: $g}) RETURN properties(m)" in META  # the one meta read: read_meta


def test_meta_of_a_group_without_one_is_none():
    reg = BonfireRegistry(FakeDriver({META: []}))
    assert asyncio.run(reg.meta("nothing")) is None


def test_watermark_reads_the_select_row_of_the_newest_message():
    row = json.dumps({"message_id": "m-3", "timestamp": "2026-10-01T10:00:00+00:00", "order": 3})
    reg = BonfireRegistry(FakeDriver({registry.WATERMARK: [{"order": 3, "row": row}]}))
    assert asyncio.run(reg.watermark("g", "c")) == {
        "message_id": "m-3",
        "order": 3,
        "timestamp": "2026-10-01T10:00:00+00:00",
    }
    never = BonfireRegistry(FakeDriver({registry.WATERMARK: []}))
    assert asyncio.run(never.watermark("g", "c")) is None


# -- against Neo4j -----------------------------------------------------------------------------------------------------


def _write(target, cypher: str, **params: Any) -> None:
    from neo4j import GraphDatabase

    with GraphDatabase.driver(target.uri, auth=(target.user, target.password)) as driver:
        driver.execute_query(cypher, **params)


def _run(target, make):
    """``make(registry)`` is a coroutine function; it runs against the target's Neo4j through a real driver."""
    from neo4j import AsyncGraphDatabase

    async def scenario():
        driver = AsyncGraphDatabase.driver(target.uri, auth=(target.user, target.password))
        try:
            return await make(driver)
        finally:
            await driver.close()

    return asyncio.run(scenario())


def _episodic(target, uuid: str, chat: str, order: int, message_id: str) -> None:
    record = {"message_id": message_id, "timestamp": f"2026-10-01T10:00:0{order}+00:00", "order": order}
    _write(
        target,
        "CREATE (:Episodic {uuid: $u, group_id: $g, chat: $c, `order`: $o, source_row_json: $row})",
        u=uuid,
        g=target.group,
        c=chat,
        o=order,
        row=json.dumps(record, sort_keys=True),
    )


@pytest.mark.needs_neo4j
def test_group_without_bonfire_meta_is_not_a_bonfire(neo4j_target):
    """A 1.8 group (or a seed still in progress) holds nodes but no BonfireMeta: never listed, no meta."""
    _episodic(neo4j_target, "e1", "chat-a", 1, "m-1")

    async def scenario(driver):
        reg = BonfireRegistry(driver, refresh_s=0.0)
        return neo4j_target.group in await reg.ids(), await reg.meta(neo4j_target.group)

    assert _run(neo4j_target, scenario) == (False, None)


@pytest.mark.needs_neo4j
def test_group_is_invisible_until_bonfire_meta(neo4j_target):
    group = neo4j_target.group
    _episodic(neo4j_target, "e1", "chat-a", 1, "m-1")

    async def seen(driver):
        reg = BonfireRegistry(driver, refresh_s=0.0)
        return group in await reg.ids(), await reg.meta(group)

    assert _run(neo4j_target, seen) == (False, None)
    _write(neo4j_target, "CREATE (m:BonfireMeta) SET m = $props", props=props(group, run=2))
    listed, meta = _run(neo4j_target, seen)
    assert listed is True
    assert meta["bonfire_id"] == group and meta["committed_run"] == 2
    assert meta["query_cache_key"] and meta["schema"] == "mk-bonfire/v3"


@pytest.mark.needs_neo4j
def test_pool_answers_only_groups_with_bonfire_meta(neo4j_target):
    """The pool over the real registry: 404 before the meta lands, 200 after it (``meta`` is read fresh), and the
    run follows the node's ``committed_run``."""
    group = neo4j_target.group
    fakes = Fakes()

    async def scenario(driver):
        pool = http.Pool(BonfireRegistry(driver, refresh_s=3600.0), server_factory=fakes)
        before = await pool.context(group, "q?")
        _write(neo4j_target, "CREATE (m:BonfireMeta) SET m = $props", props=props(group, run=3))
        after = await pool.context(group, "q?")
        _write(neo4j_target, "MATCH (m:BonfireMeta {group_id: $g}) SET m.committed_run = 4", g=group)
        await pool.context(group, "q?")
        await pool.wait_swaps()
        swapped = await pool.context(group, "q?")
        await pool.close()
        return before[0], after[0], after[1]["run"], swapped[1]["run"]

    assert _run(neo4j_target, scenario) == (404, 200, "0003", "0004")
    assert [c.group for c in fakes.opened] == [group, group]


@pytest.mark.needs_neo4j
def test_watermark_is_max_order_of_the_chat(neo4j_target):
    for uuid, chat, order, mid in (
        ("e1", "chat-a", 1, "a-1"),
        ("e3", "chat-a", 3, "a-3"),
        ("e2", "chat-a", 2, "a-2"),
        ("b9", "chat-b", 9, "b-9"),
    ):
        _episodic(neo4j_target, uuid, chat, order, mid)

    async def scenario(driver):
        reg = BonfireRegistry(driver)
        return [await reg.watermark(neo4j_target.group, c) for c in ("chat-a", "chat-b", "chat-never")]

    a, b, never = _run(neo4j_target, scenario)
    assert a == {"message_id": "a-3", "order": 3, "timestamp": "2026-10-01T10:00:03+00:00"}
    assert b["message_id"] == "b-9" and b["order"] == 9
    assert never is None
