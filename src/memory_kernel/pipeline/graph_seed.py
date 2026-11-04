"""Kernel 2.0: the seed. A new bonfire is built into an empty Neo4j group and sealed with its ``BonfireMeta``; no
directory survives it.

``seed`` runs the 1.8 full build (``run_build`` through seal, which writes the group in batches) and then ``seal_seed``,
which writes the bootstrap's ``Roleset`` / ``Route`` rows and the ``BonfireMeta`` (``committed_run`` 0) in ONE
transaction. Serving finds a bonfire only by its ``BonfireMeta``, so the group stays invisible until that last write.
The ``kernel_ops`` record guards the group: with no record, only an empty group may be seeded; a ``seeding`` record
over a non-empty group is a failed seed, whose rows are deleted (``empty_group``) before it starts again. A seed of a
bonfire that has its ``BonfireMeta`` is ``done`` and touches nothing. A reseed makes a new meta (new ``created_at``,
the commit guard's generation), never a copy of an old one."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from memory_kernel.graph.state import meta_props, meta_row, roleset_rows, route_rows
from memory_kernel.graph.state_read import read_meta
from memory_kernel.graph.tx import transaction
from memory_kernel.ops.base import KernelOps, LeaseKeeper
from memory_kernel.pipeline import graphdb
from memory_kernel.pipeline.bonfire import BUILD_CONFIG, BonfireError
from memory_kernel.pipeline.config import LAST_STAGE, SECRET_FIELDS, BuildConfig
from memory_kernel.pipeline.runner import (
    DIR_KEY,
    ORDER,
    STAGE_DIRS,
    Runner,
    conv_spent,
    conversation_counts,
    marker,
    path,
    run_build,
)

_log = logging.getLogger(__name__)

#: The refusal for a group that holds rows but has no ``kernel_ops`` record: a 1.8 group, or another tenant's. The
#: kernel service matches this text.
NOT_OURS = "group holds data no 2.0 seed started"
#: Build-config fields that name where a service runs: left out of the stored ``build_config_json``.
ADDRESS_FIELDS = ("neo4j_uri", "neo4j_user")
#: The first run of a bonfire: the bootstrap's rows, handles and routes are all run 0.
BOOTSTRAP_RUN = 0
_EMPTY_BATCH = "MATCH (n {group_id: $g}) WITH n LIMIT $batch DETACH DELETE n RETURN count(*) AS n"  # noqa: S608 (parameters)
_HAS_META = "MATCH (m:BonfireMeta {group_id: $g}) RETURN count(m) AS n"
_DELETE_META = "MATCH (m:BonfireMeta {group_id: $g}) DETACH DELETE m RETURN count(*) AS n"  # noqa: S608 (parameters)
_META_COUNT = "MATCH (m:BonfireMeta {group_id: $g}) RETURN count(m) AS n"


def _driver(cfg: BuildConfig) -> Any:
    return graphdb.driver(cfg)


async def _node_count(driver: Any, group: str) -> int:
    return int((await graphdb.group_counts(driver, group))["nodes"])


async def _scalar(driver: Any, query: str, **params: Any) -> int:
    async with driver.session() as s:
        return int([r["n"] async for r in await s.run(query, **params)][0])


async def empty_group(driver: Any, group: str, *, batch: int = 10000, rebuild: bool = False) -> int:
    """Delete every node of ``group`` and its edges, ``batch`` nodes per statement; returns how many nodes went.
    ``BonfireError`` when the group has a ``BonfireMeta``: a sealed bonfire is never emptied, unless ``rebuild`` (the
    recovery path) takes the whole group, its ``BonfireMeta`` included: the meta goes FIRST, so the group is invisible
    at once and a crash mid-delete leaves a group that serving never lists and that no seed heals to ``ready``."""
    if await _scalar(driver, _HAS_META, g=group):
        if not rebuild:
            raise BonfireError(f"group {group!r} has a BonfireMeta: a bonfire is never emptied")
        await _scalar(driver, _DELETE_META, g=group)
    total = 0
    while True:
        gone = await _scalar(driver, _EMPTY_BATCH, g=group, batch=batch)
        total += gone
        if not gone:
            return total


@contextmanager
def rebuild_lease(ops: KernelOps, group: str) -> Iterator[None]:
    """Hold the bonfire's lease for the WHOLE rebuild (owner ``rebuild-<hex>``, renewed by a ``LeaseKeeper``): no append
    can land in a half-deleted group. ``AppendBusy`` while a live lease is held; an expired or uncommitted one is
    cleared first. Inside, the ``kernel_ops`` record is ``seeding`` (``spent_usd`` and ``ceiling_usd`` kept), so a crash
    mid-delete leaves a failed seed, which the next ``mk build --seed`` empties. The lease is released on exit."""
    if ops.get(group) is None:
        raise BonfireError(f"no kernel_ops record for bonfire {group!r}: nothing to rebuild")
    ops.clear_stale_lease(group)  # AppendBusy for a live lease
    owner = f"rebuild-{uuid.uuid4().hex[:12]}"
    ops.take_lease(
        group, owner=owner, session_id="rebuild", run=BOOTSTRAP_RUN, committed_run=BOOTSTRAP_RUN, ttl_s=LEASE_TTL_S
    )
    keeper = LeaseKeeper(ops, group, owner, LEASE_TTL_S).start()
    try:
        ops.set_state(group, "seeding")
        yield
        keeper.check()  # the delete ran under a lease that was still ours
    finally:
        keeper.stop()
        ops.release(group, owner=owner, committed=True)


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise BonfireError(f"seed: {path} does not read ({type(exc).__name__}: {exc})") from None


def seal_rows(root: Path, *, bonfire_id: str, generation: str) -> tuple[str, dict[str, Any]]:
    """``(group, rows)`` of the seal: the bootstrap at ``root``'s ``Roleset`` rows (its ``eventuality.json``), ``Route``
    rows and ``ROUTES_TO`` edges (``bootstrap_chain``) and its ``BonfireMeta`` at ``committed_run`` 0 of
    ``generation``. Pure: reads the root, touches no database."""
    from memory_kernel.pipeline import chain as chain_state
    from memory_kernel.pipeline.commit import driver_start_model
    from memory_kernel.pipeline.stages._conversation import session_keys

    root = Path(root)
    cfg = chain_state._bootstrap_config(root)  # noqa: SLF001 (the chain module's own root reader)
    group = cfg.group
    chain = chain_state.bootstrap_chain(root)
    events = _read(root / STAGE_DIRS["events"] / "eventuality.json")
    tag = _read(root / STAGE_DIRS["index"] / "index.json").get("recipe_tag")
    if not tag:
        raise BonfireError(f"seed: {root}: the index record has no recipe_tag")
    model = driver_start_model(root)
    if model is None:
        raise BonfireError(f"seed: {root}: the build's driver-start records no extraction model")
    conv = _read(Path(cfg.conversation))["conversation"]
    # no secret and no address is stored: an append takes the Neo4j address from its own overrides
    build_config = {k: v for k, v in _read(root / BUILD_CONFIG).items() if k not in SECRET_FIELDS + ADDRESS_FIELDS}
    meta = meta_props(
        bonfire_id=bonfire_id,
        build_config=build_config,
        recipe_tag=tag,
        extraction_model=model,
        agent_handles=chain.agent_handles,
        agent_handle_runs=[BOOTSTRAP_RUN] * len(chain.agent_handles),
        chat_tagged=any(f"{key}_conversation" in conv for key in session_keys(conv)),
        explicit_since=chain.explicit_since,
        sessions=chain.sessions,
        committed_run=BOOTSTRAP_RUN,
        generation=generation,
    )
    route_nodes, route_edges = route_rows(group, chain.routes, run=BOOTSTRAP_RUN)
    rows = {
        "nodes": [*roleset_rows(group, events["rolesets"], run=BOOTSTRAP_RUN), *route_nodes, meta_row(group, meta)],
        "edges": route_edges,
        "external_endpoints": {
            target: {"labels_neo4j": ["Entity"], "props": {"uuid": target}}
            for target in sorted(set(chain.routes.values()))
        },
    }
    return group, rows


async def seal_seed(driver: Any, root: Path, *, bonfire_id: str) -> dict[str, Any]:
    """Write the seal rows (``seal_rows``) in ONE transaction: the bootstrap's ``Roleset`` and ``Route`` rows and the
    ``BonfireMeta``, of a ``generation`` minted for this seal (``uuid4``: a reseed of the group is always another
    generation). From here the group is a bonfire. Returns ``{group, committed_run, generation, rolesets, routes}``."""
    from memory_kernel.graph.kg_writer import ensure_constraints, ensure_indexes, write_rows
    from memory_kernel.pipeline import embed

    generation = uuid.uuid4().hex
    group, rows = seal_rows(Path(root), bonfire_id=bonfire_id, generation=generation)
    await ensure_indexes(driver)  # DDL never runs inside a transaction
    await ensure_constraints(driver)
    async with transaction(driver, commit=True) as tx:
        await write_rows(tx, rows, group_id=group, embed=embed.ReplayEmbed({}))
        if await _scalar(tx, _META_COUNT, g=group) != 1:
            raise BonfireError(f"seed: group {group!r} must hold exactly one BonfireMeta after the seal")
    labels = [n["labels_neo4j"][0] for n in rows["nodes"]]
    return {
        "group": group,
        "committed_run": BOOTSTRAP_RUN,
        "generation": generation,
        "rolesets": labels.count("Roleset"),
        "routes": labels.count("Route"),
    }


def _on_graph(cfg: BuildConfig, fn: Callable[[Any], Awaitable[Any]]) -> Any:
    """``fn(driver)`` on a driver of the build's Neo4j, in its own event loop (the build itself runs stages that use
    theirs, so no driver is held across it)."""

    async def go() -> Any:
        driver = _driver(cfg)
        try:
            return await fn(driver)
        finally:
            await driver.close()

    return asyncio.run(go())


#: The seed's lease: long enough to ride out a slow renew, short enough that a dead seed frees the bonfire.
LEASE_TTL_S = 120.0
#: Stages from the group write on: their done markers and outputs go when the group is (re)written.
WRITES_GROUP = ORDER[ORDER.index("write") :]


def invalidate_group_stages(cfg: BuildConfig) -> list[str]:
    """Remove the done markers and the output dirs of every stage that writes the group (``write`` and after), so the
    build writes it again. The paid upstream stages keep their markers. Returns the stages that had either."""
    cleared = []
    for stage in WRITES_GROUP:
        out = path(cfg, DIR_KEY.get(stage, stage))
        if marker(cfg, stage).exists() or out.exists():
            cleared.append(stage)
        marker(cfg, stage).unlink(missing_ok=True)
        if out.is_dir():
            shutil.rmtree(out)
        elif out.exists():
            out.unlink()
    return cleared


def _spent(cfg: BuildConfig) -> float:
    return conv_spent(cfg)


def _messages(cfg: BuildConfig) -> int:
    return conversation_counts(cfg.conversation, cfg.sample_id).messages


SETTLED_MARKER = "seed-settled.json"


def _settled(cfg: BuildConfig) -> Decimal:
    """What earlier attempts in this workdir already settled: the ledger is cumulative across attempts (the paid
    upstream markers are kept), so each settle charges only the delta."""
    try:
        return Decimal(str(json.loads((cfg.workdir / SETTLED_MARKER).read_text())["settled_usd"]))
    except (OSError, ValueError, KeyError, ArithmeticError):
        return Decimal(0)


def _settle_delta(
    ops: KernelOps, cfg: BuildConfig, group: str, cap: float, *, new_messages: int = 0, total: Decimal | None = None
) -> None:
    """Settle ``ledger total - already settled`` and record the new settled total in the workdir."""
    total = Decimal(str(round(_spent(cfg), 9))) if total is None else total
    before = _settled(cfg)
    ops.settle(
        group,
        run=BOOTSTRAP_RUN,
        spent_usd=max(total - before, Decimal(0)),
        new_messages=new_messages,
        budget_usd=Decimal(str(cap)),
    )
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    (cfg.workdir / SETTLED_MARKER).write_text(json.dumps({"settled_usd": str(max(total, before))}))


def _settle_failed(ops: KernelOps, cfg: BuildConfig, group: str, cap: float) -> None:
    """Settle a failed attempt's hold with its ledger's spend. Never raises: a ledger that does not read, or a settle
    that fails, leaves the hold, which the next hold charges at its estimate."""
    try:
        _settle_delta(ops, cfg, group, cap)
    except Exception:  # noqa: BLE001 - never mask the failure that got us here
        _log.warning("seed of %r: the failed attempt's spend was not settled", group, exc_info=True)


def _decimal(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def seed(
    cfg: BuildConfig,
    *,
    ops: KernelOps,
    ceiling_usd: Decimal | float | str | None = None,
    runner: Runner = subprocess.run,
    secret_sources: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Seed the bonfire ``cfg.group`` (which is also ``cfg.sample_id``): ``{"status": "seeded" | "done"}``.

    1. A ``BonfireMeta`` exists: ``done``. A ``seeding`` record is healed to ``ready`` (a hold still held is settled at
       its estimate first); the graph is not touched.
    2. No ``kernel_ops`` record and a non-empty group: ``BonfireError`` (``NOT_OURS``). No record and an empty group:
       the record is created (``seeding``, with ``ceiling_usd``). A record with no ceiling takes ``ceiling_usd``; an existing
       ceiling is never changed by a seed.
    3. The seed takes the bonfire's lease (``AppendBusy`` when it is held), for the whole seed.
    4. A ``seeding`` record over a non-empty group (a failed seed): ``empty_group``. Any other state with no meta is
       refused. The done markers and outputs of the stages that write the group are removed.
    5. The build's hard cap is the ceiling when that is lower; a run-0 hold for the cap. ``run_build`` through seal:
       the group is written in batches and stays invisible to readers.
    6. ``seal_seed``, the hold settled with the build's spend, the record ``ready``."""
    group = cfg.group
    if group != cfg.sample_id:
        raise BonfireError(f"seed: group {group!r} must be the bonfire id ({cfg.sample_id!r})")
    record = ops.get(group)
    if _on_graph(cfg, lambda d: read_meta(d, group)) is not None:
        if record is not None and record.state == "seeding":
            if record.held is not None:  # sealed, but the process died before it settled: charge the estimate
                run, estimate = record.held
                ops.settle(group, run=run, spent_usd=estimate, new_messages=0, budget_usd=estimate)
            ops.set_state(group, "ready")  # sealed, but the process died before it could say so
        return {"status": "done"}
    nodes = _on_graph(cfg, lambda d: _node_count(d, group))
    wanted = _decimal(ceiling_usd)
    if record is None:
        if nodes:
            raise BonfireError(f"{NOT_OURS}: {group!r} holds {nodes} nodes and has no kernel_ops record")
        ops.create(group, ceiling_usd=wanted, state="seeding")
    elif record.state != "seeding":
        raise BonfireError(f"bonfire {group!r} is {record.state!r} in kernel_ops but its group has no BonfireMeta")
    elif wanted is not None and record.ceiling_usd is None:
        ops.set_ceiling(group, wanted)  # a ceiling is only ever given, never changed, by a seed (`mk bonfire ceiling`)
    owner = f"seed-{uuid.uuid4().hex[:12]}"
    ops.take_lease(
        group, owner=owner, session_id="seed", run=BOOTSTRAP_RUN, committed_run=BOOTSTRAP_RUN, ttl_s=LEASE_TTL_S
    )
    keeper = LeaseKeeper(ops, group, owner, LEASE_TTL_S).start()
    committed = False
    holding = False
    try:
        if record is not None and nodes:
            _on_graph(cfg, lambda d: empty_group(d, group))
        invalidate_group_stages(cfg)
        held = ops.get(group)
        ceiling = held.ceiling_usd if held is not None else wanted
        cap = cfg.cap_usd if ceiling is None else min(cfg.cap_usd, float(ceiling))
        cfg = replace(cfg, through=LAST_STAGE, cap_usd=cap)
        ops.hold(group, run=BOOTSTRAP_RUN, estimate_usd=Decimal(str(cap)))
        holding = True
        out = run_build(cfg, runner=runner, secret_sources=secret_sources)
        if not (out.get("status") == "complete" or (out.get("status") == "through" and out.get("stage") == LAST_STAGE)):
            raise BonfireError(f"the seed build of {group!r} stopped before seal: {json.dumps(out, default=str)}")
        keeper.check()  # nothing is sealed under a lost lease
        _on_graph(cfg, lambda d: seal_seed(d, cfg.conv_root, bonfire_id=group))
        _settle_delta(ops, cfg, group, cap, new_messages=_messages(cfg))
        holding = False
        ops.set_state(group, "ready")
        committed = True
    finally:
        if holding:  # a failed attempt: settle what it really spent, so the retry finds no orphaned hold
            _settle_failed(ops, cfg, group, cap)
        keeper.stop()
        ops.release(group, owner=owner, committed=committed)
    return {"status": "seeded"}
