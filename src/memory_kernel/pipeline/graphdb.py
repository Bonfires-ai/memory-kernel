"""Neo4j access for the write / rewrite / index / sanity stages.

Ported from the reference harness:
  * ``group_counts``                                                  <- the harness's build module
  * ``diff_counts`` / ``other_groups_unchanged`` / ``snapshot`` / ``foreign_edges`` <- the harness's conversation-build module
  * ``cypher_dups``                                                   <- the harness's round build module

Two rebindings differ:
  * ``cypher_dups`` read the module ``GROUP``; here the group is an explicit argument;
  * ``stack_driver()`` (``docker exec`` for the password, ``NEO4J_URI`` env) -> ``driver(cfg)`` from the explicit
    ``cfg.neo4j_uri`` / ``neo4j_user`` / ``neo4j_password``. The async driver binds to its event loop, so a caller builds
    one INSIDE each ``asyncio.run`` and never caches it.

There is no reset or delete path here, nor anywhere in mk: a write only ever MERGEs, into an empty group (``fresh``)
or into the bonfire's own group (``append``, ``stages.write``; kernel 1.2 retired ``mk append --prune-orphans``, the one
delete there was: a forward append leaves no orphans, ``stages.write.forward_gate``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_kernel.pipeline.config import BuildConfig


def driver(cfg: BuildConfig) -> Any:
    """A write-capable async driver (write / rewrite / index: ``write_rows`` MERGEs, ``persist_keys`` SETs)."""
    from neo4j import AsyncGraphDatabase

    if cfg.neo4j_password is None:
        raise ValueError("no Neo4j password configured (NEO4J_PASSWORD, --env-file or --secrets-from-container)")
    return AsyncGraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))


def read_only(cfg: BuildConfig) -> Any:
    """``driver(cfg)`` with every session in READ mode (sanity; the server rejects a write in one)."""
    from memory_kernel.serve.answer import ReadOnlyDriver

    return ReadOnlyDriver(driver(cfg))


# -- verbatim helpers ------------------------------------------------------------------------------------------------


async def group_counts(driver, group: str) -> dict[str, int]:
    """Read-only node and edge counts of one group."""
    async with driver.session() as s:
        n = await (await s.run("MATCH (n {group_id:$g}) RETURN count(n) AS c", g=group)).single()
        e = await (await s.run("MATCH ()-[r {group_id:$g}]->() RETURN count(r) AS c", g=group)).single()
    return {"nodes": int(n["c"]), "edges": int(e["c"])}


async def label_counts(driver, group: str) -> dict[str, int]:
    """Read-only ``:Episodic`` and ``:Saga`` totals of one group (the forward gate's rule 3)."""
    out = {}
    async with driver.session() as s:
        for label in ("Episodic", "Saga"):
            got = await (await s.run(f"MATCH (n:{label} {{group_id:$g}}) RETURN count(n) AS c", g=group)).single()
            out[label] = int(got["c"])
    return out


async def saga_passes(driver, group: str) -> list[list[str]]:
    """Read-only: the distinct stored ``ingest_pass`` lists of the ``:Saga`` rows of one group (``[]`` for a saga
    without one), the identity an append-mode write checks the group against (``stages.write.check_owner``): a few
    rows whatever the group's size."""
    async with driver.session() as s:
        result = await s.run("MATCH (s:Saga {group_id:$g}) RETURN DISTINCT s.ingest_pass AS p", g=group)
        rows = [r["p"] async for r in result]
    return [[p] if isinstance(p, str) else [str(x) for x in p or []] for p in rows]


def diff_counts(got: Mapping[str, int], want: Mapping[str, int]) -> dict[str, dict]:
    return {
        k: {"got": got.get(k, 0), "sealed": want.get(k, 0)}
        for k in sorted(set(got) | set(want))
        if got.get(k, 0) != want.get(k, 0)
    }


def other_groups_unchanged(before: Mapping, after: Mapping, group: str) -> dict:
    """Snapshots are {"nodes": {group: n}, "edges": {group: n}}; every group except ``group`` must be equal."""
    changed = {}
    for kind in ("nodes", "edges"):
        for g in sorted(set(before[kind]) | set(after[kind])):
            if g != group and before[kind].get(g, 0) != after[kind].get(g, 0):
                changed[f"{kind}:{g}"] = {"before": before[kind].get(g, 0), "after": after[kind].get(g, 0)}
    return {"unchanged": not changed, "changed": changed}


async def snapshot(driver) -> dict[str, dict]:
    """Node and edge counts of EVERY group in the database: a scan of every node and relationship, read by a write only
    under ``BuildConfig.strict_db_snapshot`` (``stages.write.write_problems``)."""
    async with driver.session() as s:
        nodes = {r["g"]: r["c"] async for r in await s.run("MATCH (n) RETURN n.group_id AS g, count(n) AS c")}
        edges = {r["g"]: r["c"] async for r in await s.run("MATCH ()-[r]->() RETURN r.group_id AS g, count(r) AS c")}
    return {"nodes": {str(k): v for k, v in nodes.items()}, "edges": {str(k): v for k, v in edges.items()}}


async def foreign_edges(driver, group: str) -> int:
    """Edges of ``group`` whose endpoints sit outside it, plus edges touching group nodes that are not in the group."""
    async with driver.session() as s:
        a = await (
            await s.run(
                "MATCH (a)-[r {group_id:$g}]->(b) WHERE a.group_id <> $g OR b.group_id <> $g RETURN count(r) AS c",
                g=group,
            )
        ).single()
        b = await (
            await s.run(
                "MATCH (a {group_id:$g})-[r]-(b) WHERE r.group_id IS NULL OR r.group_id <> $g RETURN count(r) AS c",
                g=group,
            )
        ).single()
    return int(a["c"]) + int(b["c"])


async def cypher_dups(driver, group: str) -> dict:
    """The operator's duplicate measure, run on the stored graph (``group`` explicit)."""
    queries = {
        "excl_events": "MATCH (n:Entity {group_id:$g}) WHERE NOT n:Event "
        "RETURN count(n) - count(DISTINCT toLower(n.name)) AS d",
        "incl_events": "MATCH (n:Entity {group_id:$g}) RETURN count(n) - count(DISTINCT toLower(n.name)) AS d",
    }
    out = {}
    async with driver.session() as s:
        for name, q in queries.items():
            out[name] = {"cypher": q, "value": (await (await s.run(q, g=group)).single())["d"]}
    return out
