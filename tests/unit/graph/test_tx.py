import asyncio

import pytest

from memory_kernel.graph.tx import transaction
from memory_kernel.pipeline.graphdb import label_counts


class _Result:
    def __init__(self, c):
        self._c = c

    async def single(self):
        return {"c": self._c}


class _FakeTx:
    def __init__(self):
        self.queries = []
        self.ended = []

    async def run(self, query, parameters=None, **params):
        self.queries.append((query, params))
        return _Result(len(self.queries))

    async def commit(self):
        self.ended.append("commit")

    async def rollback(self):
        self.ended.append("rollback")


class _FakeSession:
    def __init__(self, tx):
        self.tx = tx
        self.closed = False

    async def begin_transaction(self):
        return self.tx

    async def close(self):
        self.closed = True


class _FakeDriver:
    def __init__(self):
        self.tx = _FakeTx()
        self.sessions = []

    def session(self, **kwargs):
        self.sessions.append(_FakeSession(self.tx))
        return self.sessions[-1]


def test_every_session_runs_on_the_one_transaction():
    driver = _FakeDriver()

    async def scenario():
        async with transaction(driver, commit=True) as scope:
            return await label_counts(scope, "g")

    assert asyncio.run(scenario()) == {"Episodic": 1, "Saga": 2}
    assert [q for q, _ in driver.tx.queries] == [
        "MATCH (n:Episodic {group_id:$g}) RETURN count(n) AS c",
        "MATCH (n:Saga {group_id:$g}) RETURN count(n) AS c",
    ]
    assert len(driver.sessions) == 1 and driver.sessions[0].closed
    assert driver.tx.ended == ["commit"]


def test_dry_transaction_rolls_back_on_success():
    driver = _FakeDriver()

    async def scenario():
        async with transaction(driver, commit=False):
            pass

    asyncio.run(scenario())
    assert driver.tx.ended == ["rollback"]
    assert driver.sessions[0].closed


def test_commit_transaction_rolls_back_on_error_and_reraises():
    driver = _FakeDriver()

    async def scenario():
        async with transaction(driver, commit=True):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(scenario())
    assert driver.tx.ended == ["rollback"]
    assert driver.sessions[0].closed


@pytest.mark.needs_neo4j
def test_reads_in_the_transaction_see_its_writes(neo4j_target):
    from neo4j import AsyncGraphDatabase

    async def scenario():
        driver = AsyncGraphDatabase.driver(neo4j_target.uri, auth=(neo4j_target.user, neo4j_target.password))
        try:
            async with transaction(driver, commit=False) as scope:
                async with scope.session() as s:
                    await (await s.run("CREATE (:Episodic {uuid: 'e1', group_id: $g})", g=neo4j_target.group)).consume()
                return await label_counts(scope, neo4j_target.group)
        finally:
            await driver.close()

    assert asyncio.run(scenario()) == {"Episodic": 1, "Saga": 0}


@pytest.mark.needs_neo4j
def test_rolled_back_write_rows_leave_no_row(neo4j_target):
    from neo4j import AsyncGraphDatabase

    from memory_kernel.graph.kg_writer import ensure_constraints, write_rows

    g = neo4j_target.group
    saga = {
        "labels_neo4j": ["Saga"],
        "props": {
            "uuid": "s1",
            "group_id": g,
            "name": "chat",
            "summary": "",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    }

    async def scenario():
        driver = AsyncGraphDatabase.driver(neo4j_target.uri, auth=(neo4j_target.user, neo4j_target.password))
        try:
            await ensure_constraints(driver)  # DDL stays outside the transaction
            async with transaction(driver, commit=False) as scope:
                await write_rows(
                    scope, {"nodes": [saga], "edges": []}, group_id=g, embed=lambda texts: [[0.0] for _ in texts]
                )
                inside = await label_counts(scope, g)
            return inside, await label_counts(driver, g)
        finally:
            await driver.close()

    assert asyncio.run(scenario()) == ({"Episodic": 0, "Saga": 1}, {"Episodic": 0, "Saga": 0})
