import asyncio
import hashlib

import pytest

from memory_kernel.graph.kg_contract import STRUCTURAL_EDGES
from memory_kernel.graph.kg_writer import (
    CONSTRAINT_NAMES,
    FULLTEXT_INDEX_DDL,
    INDEX_NAMES,
    INDEX_TABLE,
    RANGE_INDEX_DDL,
    ensure_constraints,
    index_statement,
    write_rows,
)

#: sha256 of the 31 Graphiti statements, newline-joined in issue order: the index set a Graphiti-compatible graph
#: expects. The statements are schema facts, so a change to any byte of them is a change of contract, never a refactor.
STATEMENTS_SHA256 = "7ad1a9e75174c1a05a6db3ce78329479ec178b9993fe564da9ef3253f4624468"
#: The edge types of a Graphiti graph; every other type the writer writes is the kernel's own.
GRAPHITI_EDGES = frozenset({"RELATES_TO", "MENTIONS", "HAS_MEMBER", "HAS_EPISODE", "NEXT_EPISODE"})
#: The node labels of a Graphiti graph; the 2.0 type nodes (Roleset, Route, TaughtPhrase, BonfireMeta) are the kernel's own.
GRAPHITI_NODES = frozenset({"Entity", "Episodic", "Community", "Saga"})


def test_kernel_edge_types_have_uuid_indexes():
    """The writer looks every edge it writes up by type and uuid (``_foreign``, ``read_rows``): each type it writes has
    a uuid index. The kernel's own types add theirs; the Graphiti statements stay byte-identical."""
    uuid_indexed = {row[3] for row in INDEX_TABLE if row[0] == "range" and row[2] == "rel" and row[4] == ("uuid",)}
    assert {*STRUCTURAL_EDGES, "RELATES_TO"} <= uuid_indexed
    kernel = [row for row in INDEX_TABLE if row[2] == "rel" and row[3] not in GRAPHITI_EDGES]
    assert sorted(row[3] for row in kernel) == sorted(set(STRUCTURAL_EDGES) - GRAPHITI_EDGES)
    assert {(row[0], row[4]) for row in kernel} == {("range", ("uuid",))}
    graphiti = [index_statement(*row) for row in INDEX_TABLE if row[3] in GRAPHITI_NODES | GRAPHITI_EDGES]
    assert len(graphiti) == 31 and hashlib.sha256("\n".join(graphiti).encode()).hexdigest() == STATEMENTS_SHA256


def test_index_names_follow_the_table_and_the_statements():
    statements = (*RANGE_INDEX_DDL, *FULLTEXT_INDEX_DDL)
    assert tuple(row[1] for row in INDEX_TABLE) == INDEX_NAMES
    assert len(set(INDEX_NAMES)) == len(INDEX_NAMES)
    for name, statement in zip(INDEX_NAMES, statements, strict=True):
        assert f" INDEX {name} IF NOT EXISTS" in statement


def test_index_statements_only_create():
    for statement in (*RANGE_INDEX_DDL, *FULLTEXT_INDEX_DDL):
        assert statement.startswith("CREATE ") and "IF NOT EXISTS" in statement
        assert "DROP" not in statement and "VECTOR" not in statement


T = "2026-01-01T00:00:00Z"


def _fake_embed(texts):
    return [[0.0] for _ in texts]


@pytest.mark.needs_neo4j
def test_type_nodes_keep_their_first_answer(neo4j_target):
    """A second write of a Roleset and a Route changes no non-decision prop: the first gloss, target and union list
    stay (CREATE_ONLY_KINDS empties the on-match update; the union clause is skipped for these kinds too)."""
    from neo4j import AsyncGraphDatabase

    async def scenario():
        driver = AsyncGraphDatabase.driver(neo4j_target.uri, auth=(neo4j_target.user, neo4j_target.password))
        try:
            g = neo4j_target.group
            await ensure_constraints(driver)
            writes = (("first", "state", "e1", ["a1"]), ("second", "occurrence", "e2", ["a2"]))
            for gloss, eventuality, target, aliases in writes:
                nodes = [
                    {
                        "labels_neo4j": ["Roleset"],
                        "props": {
                            "uuid": "rs1",
                            "group_id": g,
                            "created_at": T,
                            "roleset": "go.01",
                            "gloss": gloss,
                            "eventuality": eventuality,
                            "aliases": aliases,
                        },
                    },
                    {
                        "labels_neo4j": ["Route"],
                        "props": {
                            "uuid": "ro1",
                            "group_id": g,
                            "created_at": T,
                            "key": "k1",
                            "target_name": target,
                            "aliases": aliases,
                        },
                    },
                ]
                await write_rows(driver, {"nodes": nodes, "edges": []}, group_id=g, embed=_fake_embed)
            async with driver.session() as s:
                result = await s.run(
                    "MATCH (r:Roleset {uuid: 'rs1', group_id: $g}) MATCH (t:Route {uuid: 'ro1', group_id: $g}) "
                    "RETURN r.gloss AS gloss, r.eventuality AS eventuality, r.aliases AS roleset_aliases, "
                    "t.target_name AS target_name, t.aliases AS route_aliases",
                    g=g,
                )
                return (await result.single()).data()
        finally:
            await driver.close()

    assert asyncio.run(scenario()) == {
        "gloss": "first",
        "eventuality": "state",
        "roleset_aliases": ["a1"],
        "target_name": "e1",
        "route_aliases": ["a1"],
    }


@pytest.mark.needs_neo4j
def test_constraints_refuse_a_second_meta_node(neo4j_target):
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import ConstraintError

    async def scenario():
        driver = AsyncGraphDatabase.driver(neo4j_target.uri, auth=(neo4j_target.user, neo4j_target.password))
        try:
            report = await ensure_constraints(driver)
            assert report["missing_after"] == []
            assert set(CONSTRAINT_NAMES) == set(report["created"]) | set(report["already_present"])
            meta = "CREATE (:BonfireMeta {uuid: $u, group_id: $g, bonfire_id: 'b', schema: 'mk-bonfire/v3'})"
            async with driver.session() as s:
                await (await s.run(meta, u="m1", g=neo4j_target.group)).consume()
                with pytest.raises(ConstraintError):
                    await (await s.run(meta, u="m2", g=neo4j_target.group)).consume()
        finally:
            await driver.close()

    asyncio.run(scenario())
