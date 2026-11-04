"""Kernel 2.0: the append. One session runs into a bonfire whose state lives in Neo4j; no directory holds any of it.

``append``:

1. read the bonfire's ``BonfireMeta`` (no meta: no such bonfire) and take its lease in ``kernel_ops`` for ``run`` =
   ``committed_run`` + 1 (``AppendBusy`` while it is live, ``PendingSession`` behind a stopped session);
2. read its chain (``state_read``: the meta and every row kind in ONE read transaction, so a read cannot mix two runs)
   and plan the session (``bonfire.plan_session``: a duplicate or empty session releases the lease and returns);
3. refuse another extraction model, then hold the run's estimated spend against the bonfire's ceiling (``ops.hold``);
4. lay out the run in its scratch dir (``<scratch_root>/<id>/runs/<run>/``: the session, its conversation, the forward
   input, the evidence rows, the index record and the events Voyage cache, seeded with the stored vectors of the chats'
   tail messages) and run the stages there (``runner.run_build``) in graph mode: they write nothing to the group;
5. when the run sealed, commit it: ``commit.commit_append``, one Neo4j write transaction (an ``already_committed``
   answer, a lost acknowledgement or a duplicate job, is a success);
6. settle the hold with the run's paid total and release the lease; the scratch run dir is always deleted (with dev
   receipts, ``pipeline.receipts``, it is kept, with the shard and the Voyage cache, and a crashed attempt's dir is
   renamed aside).

A run that stops before ``seal`` commits nothing: the graph is as it was, the spend is settled, and the lease keeps
naming the session, so the same session must be sent again before any other (``PendingSession``). The retry takes the
same run number and a FRESH run dir (the stage outputs of a crashed attempt are never re-entered)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import shutil
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from memory_kernel.graph.state_read import StateRows, chain_from_rows, read_meta, read_state_rows
from memory_kernel.graph.tx import transaction
from memory_kernel.ops.base import KernelOps, LeaseKeeper
from memory_kernel.pipeline import embed, graphdb, provenance
from memory_kernel.pipeline.bonfire import (
    EVIDENCE_INPUT,
    FORWARD_INPUT,
    INPUT_COPY,
    RUN_CONVERSATION,
    AppendBusy,
    BonfireError,
    OnBuilt,
    SessionDuplicate,
    SessionEmpty,
    append_budget,
    append_estimate,
    forward_input,
    plan_session,
    require_extraction_model,
    require_ground,
    run_config,
    session_turns,
    shard_spend,
)
from memory_kernel.pipeline.commit import commit_append, commit_input
from memory_kernel.pipeline.config import LAST_STAGE, BuildConfig
from memory_kernel.pipeline.receipts import ATTEMPT_MARK, Receipts
from memory_kernel.pipeline.runner import EVENTS_VOYAGE_CACHE, STAGE_DIRS, Runner, planned_usd, run_build

_log = logging.getLogger(__name__)
#: The run dir's index record (``cfg.bonfire_index_record``): the key recipe tag the group is indexed under.
INDEX_RECORD = "index-record.json"
#: The events stage's Voyage cache, beside the run's other pinned inputs (``forward.json``'s ``voyage_cache``).
VOYAGE_CACHE = f"embeddings/{EVENTS_VOYAGE_CACHE}"
_NEO4J_DEFAULTS = {"neo4j_uri": "bolt://localhost:7687", "neo4j_user": "neo4j", "neo4j_password": None}
_SEED_VECTORS = (
    "MATCH (e:Episodic {group_id: $g}) WHERE e.message_id IN $ids OR e.uuid IN $uuids "
    "RETURN e.source_row_json AS row, e.message_embedding AS vector, e.message_text_sha256 AS sha"
)


def session_key(session: Mapping[str, Any]) -> str:
    """The session's key in the lease: the first 16 hex digits of the sha256 of its ``conversation_id`` and message ids,
    so the same session sent again is the same key (and any other session another)."""
    turns, _ = session_turns(session)
    doc = {"conversation_id": session["conversation_id"], "message_ids": [t["dia_id"] for t in turns]}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _driver(settings: Any) -> Any:
    """A write-capable async driver of the Neo4j ``settings`` (anything with ``neo4j_uri`` / ``neo4j_user`` /
    ``neo4j_password``: the run's config, or ``_settings`` before it exists)."""
    if getattr(settings, "neo4j_password", None) is None:
        raise BonfireError("no Neo4j password configured: the append needs overrides carrying neo4j_password")
    return graphdb.driver(settings)


def _settings(overrides: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**{k: overrides.get(k, default) for k, default in _NEO4J_DEFAULTS.items()})


def _on_graph(settings: Any, fn: Callable[[Any], Awaitable[Any]]) -> Any:
    """``fn(driver)`` on a driver of the Neo4j ``settings``, in its own event loop (no driver is held across the build,
    which runs stages that use theirs)."""

    async def go() -> Any:
        driver = _driver(settings)
        try:
            return await fn(driver)
        finally:
            await driver.close()

    return asyncio.run(go())


@dataclass(frozen=True)
class World:
    """A bonfire as one read transaction saw it: its rows, the 1.8 chain they fold to (its ``rolesets`` carry the
    settled answers) and the stored vectors of the messages the run's embedding steps read again: ``seed`` is
    ``(select row, vector, text sha256)`` of each chat's tail messages and of every message of its last episode."""

    rows: StateRows
    chain: Any
    seed: list[tuple[dict, list[float], str | None]]


async def read_world(driver: Any, group: str) -> World:
    """``group``'s state in ONE read transaction (``state_read.read_state_rows``, then the seed vectors): never
    separate auto-commit queries, so a torn read cannot mix two runs."""
    async with transaction(driver, commit=False) as tx:
        rows = await read_state_rows(tx, group)
        chain = chain_from_rows(rows)
        chain.rolesets = {rs: dict(row) for rs, row in rows.rolesets.items()}  # label / p_yes / compiled_sha256 too
        ids = sorted({r["message_id"] for chat in chain.chats.values() for r in chat.tail})
        last = {chat.last_saga for chat in chain.chats.values()}
        uuids = sorted({u for s in rows.sagas if s["uuid"] in last for u in s["episodes"]})
        seed: list[tuple[dict, list[float], str | None]] = []
        async with tx.session() as s:
            async for rec in await s.run(_SEED_VECTORS, g=group, ids=ids, uuids=uuids):
                if rec["vector"] is not None and rec["row"] is not None:
                    seed.append((json.loads(rec["row"]), list(rec["vector"]), rec["sha"]))
    return World(rows, chain, seed)


def seeded_cache(seed: list[tuple[dict, list[float], str | None]]) -> dict[str, list[float]]:
    """The events stage's Voyage cache (``{text: vector}``) seeded from the stored vectors (``World.seed``): the text is
    the string the events stage embeds (``embed.message_texts``), kept only when the stored text sha256 matches it."""
    cache: dict[str, list[float]] = {}
    for row, vector, sha in seed:
        text = embed.message_texts([row])[row["message_id"]]
        if sha == hashlib.sha256(text.encode()).hexdigest():
            cache[text] = vector
    return cache


def _write(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True))


def _history(ops: KernelOps, bonfire_id: str) -> list[dict[str, Any]]:
    """``ops.history`` as ``append_estimate`` reads it (plain numbers: the stores hold decimals)."""
    out = []
    for row in ops.history(bonfire_id):
        try:
            out.append(
                {
                    "run": int(row["run"]),
                    "budget_usd": float(row["budget_usd"]),
                    "new_messages": int(row["new_messages"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _sealed(out: Mapping[str, Any]) -> bool:
    return out.get("status") == "complete" or (out.get("status") == "through" and out.get("stage") == LAST_STAGE)


def _clean(run_dir: Path) -> None:
    """Delete the run dir, then the scratch dirs that held only it."""
    shutil.rmtree(run_dir, ignore_errors=True)
    for parent in (run_dir.parent, run_dir.parent.parent):
        try:
            parent.rmdir()
        except OSError:
            return


def append(
    bonfire_id: str,
    session: Mapping[str, Any],
    *,
    ops: KernelOps,
    scratch_root: Path | str,
    owner: str,
    lease_ttl_s: float = 3600,
    on_built: OnBuilt = "refuse",
    overrides: Mapping[str, Any] | None = None,
    secret_sources: Mapping[str, str | None] | None = None,
    runner: Runner = subprocess.run,
    receipts: Receipts | None = None,
    allow_parser_change: bool = False,
) -> dict[str, Any]:
    """Run ``session`` (``SessionInput``'s shape: ``conversation_id``, ``messages``, ``agent_handles?``) into the
    bonfire ``bonfire_id`` (the module docstring's steps). ``overrides`` are ``BuildConfig`` fields over the bonfire's
    inherited build config: the secrets, the Neo4j address, the host's install paths.

    Returns ``{status, run, sessions, new_messages, skipped_ids, result}`` (1.8's ``api.append_session``):

    * ``"duplicate"`` / ``"empty"``: nothing ran (``run`` and ``result`` are None);
    * ``"appended"``: the run sealed and its commit landed (``run`` is the 4-digit run number);
    * ``"stopped"``: the build stopped before ``seal``; nothing was committed, the spend is settled and the same
      session must be sent again before any other.

    Refused (``BonfireError``): no such bonfire, a malformed session, ``AppendBusy`` (retriable), ``PendingSession``,
    another extraction model, a chain without ground, the ceiling (``kernel_ops``), and every refusal of the plan, the
    run and the commit (``StaleCommit``: another commit landed first; the run is not retried)."""
    overrides = dict(overrides or {})
    scratch_root = Path(scratch_root)
    from memory_kernel.serve.http import valid_bonfire_id

    if not valid_bonfire_id(bonfire_id):
        raise BonfireError(f"not a plain bonfire id: {bonfire_id!r}")
    session_turns(session)  # a malformed session is refused before anything is taken
    key = session_key(session)
    meta = _on_graph(_settings(overrides), lambda d: read_meta(d, bonfire_id))
    if meta is None:
        raise BonfireError(f"no such bonfire: {bonfire_id!r} has no BonfireMeta")
    committed_run = int(meta["committed_run"])
    run = committed_run + 1
    record = ops.get(bonfire_id)
    # a lease still naming an uncommitted run is a stopped session: only a landed session may clear it
    pending = record is not None and record.lease is not None and record.lease.run > committed_run
    ops.take_lease(
        bonfire_id, owner=owner, session_id=key, run=run, committed_run=committed_run, ttl_s=lease_ttl_s
    )  # before anything else: nothing is read for a session that cannot run
    keeper = LeaseKeeper(ops, bonfire_id, owner, lease_ttl_s).start()
    run_dir = scratch_root / bonfire_id / "runs" / f"{run:04d}"
    if receipts is not None:
        run_dir = receipts.run_dir(bonfire_id, run)
    clear = not pending  # released as cleared unless a pending session's lease was taken over, or the build began
    held = False
    estimate = Decimal(0)
    cfg: BuildConfig | None = None
    plan = None
    try:
        settle_leftovers(ops, bonfire_id, run_dir.parent, keep_after=committed_run if receipts else None)
        world = _on_graph(_settings(overrides), lambda d: read_world(d, bonfire_id))
        meta, chain = world.rows.meta, world.chain
        if int(meta["committed_run"]) != committed_run:  # another commit landed between the read and the lease
            raise AppendBusy(f"busy: bonfire {bonfire_id!r} moved to run {meta['committed_run']} while it was leased")
        base = BuildConfig.from_json(json.loads(meta["build_config_json"]), secrets={})
        base = replace(base, boundary_policy=meta.get("boundary_policy") or base.boundary_policy)
        try:
            plan = plan_session(
                session, chain, base, sample_id=base.sample_id, chat_tagged=bool(meta["chat_tagged"]), on_built=on_built
            )
        except (SessionDuplicate, SessionEmpty) as exc:
            clear = True  # nothing is left to run: the session has landed
            return {
                "status": "duplicate" if isinstance(exc, SessionDuplicate) else "empty",
                "run": None,
                "sessions": chain.sessions,
                "new_messages": 0,
                "result": None,
                "skipped_ids": [],
            }
        require_extraction_model(meta, allow_parser_change=allow_parser_change)
        require_ground(chain)
        kept = {} if receipts is None else _persisted(receipts.voyage_cache(bonfire_id))  # refused before the hold
        # the seed (run 0) settled its whole cap over its messages: scaled per message that would block every first append
        history = [row for row in _history(ops, bonfire_id) if row["run"] > 0]
        estimate = append_estimate(plan.new_messages, history)
        room = ops.hold(bonfire_id, run=run, estimate_usd=estimate)  # refused over the ceiling (kernel_ops)
        held, clear = True, False
        # a fresh run dir: a crashed attempt's stage outputs are never re-entered
        shutil.rmtree(run_dir, ignore_errors=True)
        run_root = run_dir / base.sample_id
        run_dir.mkdir(parents=True)
        cache = run_dir / VOYAGE_CACHE if receipts is None else receipts.voyage_cache(bonfire_id)
        # a receipts shard that exists is reused; a fresh bonfire's run makes its own in the run dir (adopted after)
        shard = None if receipts is None else receipts.shard(bonfire_id)
        shard = shard if shard is not None and (shard / "manifest.json").is_file() else None
        base_usd = Decimal(0) if shard is None else shard_spend(shard)
        cfg = run_config(
            base,
            run_root,
            conversation=run_dir / RUN_CONVERSATION,
            forward_input=run_dir / FORWARD_INPUT,
            new_messages=plan.new_messages,
            limit_base_usd=base_usd,
            ceiling_usd=None
            if room is None
            else base_usd + estimate + room,  # absolute: the limit counts from the shard
            index_record=run_dir / INDEX_RECORD,
            shard_dir=shard,
            capture_cache=None,
            **({"through": LAST_STAGE} | overrides),
        )
        if shard is not None:  # as 1.8's seed_run did: the shard at its usual stage dir (the seal hashes it there)
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / STAGE_DIRS["shard"]).symlink_to(shard.resolve(), target_is_directory=True)
        _write(run_dir / INPUT_COPY, dict(session))
        _write(run_dir / RUN_CONVERSATION, plan.conversation)
        _write(
            run_dir / FORWARD_INPUT, forward_input(chain, plan, voyage_cache=cache, boundary_policy=cfg.boundary_policy)
        )
        _write(run_dir / EVIDENCE_INPUT, chain.evidence or {})
        _write(run_dir / INDEX_RECORD, {"recipe_tag": meta["recipe_tag"]})
        _write(cache, kept | seeded_cache(world.seed) if receipts else seeded_cache(world.seed))
        out = run_build(cfg, runner=runner, secret_sources=secret_sources)
        sealed = _sealed(out)
        landed: dict[str, Any] | None = None
        if sealed:
            keeper.check()  # nothing is committed under a lost lease
            ci = commit_input(run_root, group=bonfire_id, run=run, chain=chain, generation=str(meta["generation"]))
            landed = _on_graph(cfg, lambda d: commit_append(d, ci))
            clear = True
        return {
            "status": "appended" if sealed else "stopped",
            "run": f"{run:04d}",
            "sessions": plan.number,
            "new_messages": plan.new_messages,
            "skipped_ids": list(plan.skipped_ids),
            "result": {
                **out,
                "bonfire": bonfire_id,
                "session": plan.session_id,
                "session_number": plan.number,
                "chat": plan.chat,
                "messages": chain.messages + plan.new_messages,
                "new_messages": plan.new_messages,
                "skipped_ids": list(plan.skipped_ids),
                "recorded": sealed,
                "committed": landed,
            },
        }
    finally:
        try:
            if held:
                _settle(ops, bonfire_id, run, cfg, 0 if plan is None else plan.new_messages, estimate)
        finally:
            if receipts is not None and cfg is not None:
                try:  # paid Jev answers survive a run that stopped; a failure here never masks the run's own error
                    adopt_shard(cfg, receipts.shard(bonfire_id))
                except Exception:
                    _log.exception("adopting the run's shard into %s failed", receipts.shard(bonfire_id))
            keeper.stop()
            try:
                ops.release(bonfire_id, owner=owner, committed=clear)
            finally:
                if receipts is None:
                    _clean(run_dir)


def ledger_paid(ledger: Path, estimate: Decimal) -> Decimal:
    """What the cost ledger at ``ledger`` paid: $0 when there is none (the build never wrote one); the hold's
    ``estimate`` (conservative) when it is unreadable or corrupt, logged as an error."""
    if not ledger.is_file():
        return Decimal(0)
    try:
        return Decimal(str(round(float(provenance.ledger_summary(ledger)["paid_usd"]), 9)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _log.error(
            "cost ledger %s does not read (%s): settling the hold's estimate $%s", ledger, type(exc).__name__, estimate
        )
        return estimate


def _settle(
    ops: KernelOps, bonfire_id: str, run: int, cfg: BuildConfig | None, new_messages: int, estimate: Decimal
) -> None:
    """Settle the run's hold with what its ledger paid (0 when the build never wrote one or never began; the hold's
    estimate when the ledger is corrupt)."""
    spent, planned = Decimal(0), Decimal(0)
    if cfg is not None:
        spent = ledger_paid(cfg.ledger, estimate)
        with contextlib.suppress(OSError, ValueError):
            planned = Decimal(str(planned_usd(cfg)))
    budget = append_budget(new_messages, planned_jev_usd=planned)
    ops.settle(bonfire_id, run=run, spent_usd=spent, new_messages=new_messages, budget_usd=budget)


def adopt_shard(cfg: BuildConfig, shard: Path) -> None:
    """Move the shard a fresh receipts bonfire's run made in its own run root (and its ``-pairs`` sibling) to ``shard``,
    so the next append replays it. No-op when the run used ``shard`` itself or made none."""
    if cfg.shard_dir is not None or (shard / "manifest.json").is_file():
        return
    made = cfg.conv_root / STAGE_DIRS["shard"]
    if not made.is_dir():
        return
    shard.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(shard, ignore_errors=True)  # a leftover without a manifest is not a shard
    pairs = made.parent / (made.name + "-pairs")
    if pairs.is_dir():
        kept_pairs = shard.parent / (shard.name + "-pairs")
        shutil.rmtree(kept_pairs, ignore_errors=True)
        shutil.move(str(pairs), str(kept_pairs))
    shutil.move(str(made), str(shard))


def _persisted(cache: Path) -> dict[str, list[float]]:
    """The Voyage cache a receipts root kept from earlier appends (empty when there is none)."""
    if not cache.is_file():
        return {}
    try:
        doc = json.loads(cache.read_bytes())
    except (OSError, ValueError) as exc:
        raise BonfireError(f"the kept Voyage cache {cache} does not read ({type(exc).__name__}): remove it") from None
    if not isinstance(doc, dict):
        raise BonfireError(f"the kept Voyage cache {cache} is not a {{text: vector}} object: remove it")
    return doc


def settle_leftovers(ops: KernelOps, bonfire_id: str, runs_dir: Path, *, keep_after: int | None = None) -> None:
    """Before any leftover run dir of a crashed attempt is cleared, settle what it spent: for each ``runs/NNNN`` that
    is the run the ops hold slot is held for, settle the hold with its ledger (``ledger_paid``). Then delete it.

    With receipts (``keep_after`` = the bonfire's committed run) nothing is deleted: a run dir past ``keep_after`` is a
    crashed or stopped attempt and is renamed ``NNNN.attempt-<n>``; a run dir at or before it landed and stays; the
    renamed dirs of earlier attempts stay. The spend is settled the same way in both modes."""
    if not runs_dir.is_dir():
        return
    for left in sorted(runs_dir.iterdir()):
        rec = ops.get(bonfire_id)
        if left.name.isdigit() and rec is not None and rec.held is not None and rec.held[0] == int(left.name):
            estimate = rec.held[1]
            ops.settle(
                bonfire_id,
                run=int(left.name),
                spent_usd=ledger_paid(left / "cost-ledger.jsonl", estimate),
                new_messages=0,
                budget_usd=estimate,
            )
        if keep_after is not None:
            if left.is_dir() and left.name.isdigit() and int(left.name) > keep_after:
                n = 1
                while (aside := left.with_name(f"{left.name}{ATTEMPT_MARK}{n}")).exists():
                    n += 1
                left.rename(aside)
            continue
        shutil.rmtree(left, ignore_errors=True) if left.is_dir() else left.unlink(missing_ok=True)
