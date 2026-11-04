"""Kernel 2.0: an append's commit is ONE Neo4j write transaction.

The run's write, rewrite and index stages ran in graph mode (``stages.write.graph_mode``): each in a rolled-back
transaction that recorded what it embedded, so the group is still as the last commit left it. ``commit_input`` reads
what they left in the run dir; ``commit_append`` replays it in one transaction, which commits as a whole or not at all:

1. **Guard:** the bonfire's ``BonfireMeta`` of the run's generation (its ``generation`` when the run started: each
   seal mints one, so a re-seeded bonfire is another) moves from ``committed_run`` ``run - 1`` to ``run``. It is locked first, so a commit
   that overlaps another commit of its run waits, then finds the run moved. Not there -> ``StaleCommit``, unless this
   very run already landed (same generation, ``committed_run`` = ``run``, every message of its session held: a lost
   acknowledgement or a duplicate job), which returns ``already_committed`` and writes nothing.
2. ``write_rows`` of the dry write's rows (``append=True``), with its recorded vectors (``ReplayEmbed``).
3. The same rows again: they must create and embed nothing, else ``CommitRefused``.
4. ``persist_keys`` of the dry index's keys, with its recorded vectors.
5. The session's keys computed again from the graph (``stages.index.scoped_keys``) must be the dry index's, else
   ``CommitRefused("key drift")``.
6. The group's Episodic and Saga totals are the dry pass's, every written row is held, and every Episodic carries a
   key stamped with the recipe.
7. ``BonfireMeta`` takes the run's ``sessions``, ``explicit_since``, both handle lists (a stored handle keeps its run,
   a new one is stamped with this run), ``extraction_model_json`` and ``committed_at``.

Every embedder here is a replay: the transaction makes no network call. A failure anywhere rolls everything back, so
a retry starts again from the committed state and can neither double a count nor change a decision."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memory_kernel.errors import BonfireError
from memory_kernel.graph.tx import transaction
from memory_kernel.pipeline import embed, graphdb
from memory_kernel.search import kg_index

#: The second write's counts that must all be 0 (step 3).
IDEMPOTENT = ("nodes_created", "edges_created", "node_texts_embedded", "edge_texts_embedded")
#: Locks the meta node before ``committed_run`` is read, so two overlapping commits of one run serialise.
GUARD = (
    "MATCH (m:BonfireMeta {group_id: $g}) SET m._commit_lock = true REMOVE m._commit_lock "
    "WITH m WHERE m.committed_run = $run - 1 AND m.generation = $generation SET m.committed_run = $run "
    "RETURN m.agent_handles AS handles, m.agent_handle_runs AS runs"
)
_META = "MATCH (m:BonfireMeta {group_id: $g}) RETURN m.committed_run AS run, m.generation AS generation"
_HELD = "MATCH (e:Episodic {group_id: $g}) WHERE e.uuid IN $ids RETURN count(e) AS n"
_STAMPS = "MATCH (m:Episodic {group_id: $g}) RETURN m.search_key_recipe AS recipe, count(*) AS n"
_SET_META = "MATCH (m:BonfireMeta {group_id: $g}) SET m += $p RETURN count(m) AS n"


class StaleCommit(BonfireError):
    """The bonfire's ``committed_run`` is not the run before this one (another commit landed first, or there is no
    ``BonfireMeta``); nothing was written."""


class CommitRefused(BonfireError):
    """The commit's own checks failed inside its transaction (or its inputs do not read); nothing was written."""


class _Landed(Exception):  # noqa: N818 (a control-flow signal, not an error)
    """The guard found this very run already committed: the transaction rolls back and the commit reports it."""


@dataclass(frozen=True)
class CommitInput:
    """What ``commit_append`` writes, read from a graph-mode run dir by ``commit_input``.

    ``rows``: the dry write's rows (``nodes`` / ``edges`` / ``external_endpoints``); ``write_vectors``: text -> vector
    of what it embedded; ``keys``: the session's Episodic uuid -> search key; ``key_vectors``: key -> vector;
    ``scope``: the session's Episodic uuids; ``meta_update``: ``sessions``, ``explicit_since``, ``agent_handles``
    (the union) and ``extraction_model_json``; ``expected``: the group's ``{"Episodic", "Saga"}`` totals after it;
    ``generation``: the ``BonfireMeta`` ``generation`` the run started from."""

    group: str
    run: int
    rows: dict
    write_vectors: dict
    keys: dict
    key_vectors: dict
    key_render: str
    scope: list
    meta_update: dict
    expected: dict
    generation: str


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise CommitRefused(f"commit: {path} does not read ({type(exc).__name__})") from None


def driver_start_model(run_root: Path) -> Mapping[str, Any] | None:
    """The extraction model the run's newest ``driver-start`` event records (``runner.recipe_record``), or None."""
    try:
        lines = (run_root / "driver-events.jsonl").read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("event") == "driver-start":
            return (row.get("recipe") or {}).get("extraction_model") or None
    return None


def commit_input(run_root: Path, *, group: str, run: int, chain: Any, generation: str) -> CommitInput:
    """The commit of the graph-mode run at ``run_root`` (``<run dir>/<sample>``) onto ``chain`` (the committed state
    the run started from, at ``run - 1``) of the bonfire generation ``generation`` (its ``BonfireMeta`` ``generation``
    then, ``state_read.read_meta``). Reads the write's ``summary.json`` (a dry one, passing ``write_problems``) and
    ``rerun.json`` (idempotent, its readback whole), ``commit-rows.json`` and ``write-vectors.json``, the index's
    ``index.json`` (passed) and ``keys.json``, the select rows, the run dir's ``conversation.json`` and the run's
    ``driver-start`` model. ``CommitRefused`` when one is missing, failed, or does not belong to this run."""
    from memory_kernel.pipeline.bonfire import RUN_CONVERSATION
    from memory_kernel.pipeline.chain import ChainError, session_number
    from memory_kernel.pipeline.runner import STAGE_DIRS
    from memory_kernel.pipeline.stages.index import KEYS, STAGE_FILES
    from memory_kernel.pipeline.stages.write import COMMIT_ROWS, RERUN, SUMMARY, WRITE_VECTORS, write_problems

    def stage(key: str, name: str) -> Any:
        return _read(run_root / STAGE_DIRS[key] / name)

    if chain.run != run - 1:
        raise CommitRefused(f"commit of run {run}: the chain it started from is at run {chain.run}, not {run - 1}")
    summary = stage("write", SUMMARY)
    gate = summary.get("append") or {}
    forward = gate.get("forward") or {}
    if summary.get("dry") is not True or not forward:
        raise CommitRefused(f"commit: {run_root}: the write was not a graph-mode dry write (nothing to replay)")
    if problems := write_problems(summary, prefix=None, strict=summary.get("strict_db_snapshot") is True):
        raise CommitRefused(f"commit: {run_root}: the dry write failed: {'; '.join(problems)}")
    rerun = stage("write", RERUN)
    readback = rerun.get("readback")
    if (
        rerun.get("dry") is not True
        or (rerun.get("idempotent") or {}).get("passed") is not True
        or not isinstance(readback, Mapping)
        or readback.get("missing_nodes")
        or readback.get("missing_edges")
    ):
        raise CommitRefused(f"commit: {run_root}: the dry rewrite did not pass (idempotent, readback whole)")
    if stage("index", STAGE_FILES["index"]).get("passed") is not True:
        raise CommitRefused(f"commit: {run_root}: the dry index did not pass its gate")
    if int(forward.get("run", -1)) != run:
        raise CommitRefused(f"commit: {run_root}: the write minted at run {forward.get('run')}, not {run}")
    keyed = stage("index", KEYS)
    scope = list(forward["episodic"])
    if not scope:
        raise CommitRefused(f"commit: {run_root}: the write's session holds no messages (nothing to commit)")
    if set(keyed["keys"]) != set(scope):
        raise CommitRefused(f"commit: {run_root}: the index's keys are not the session's {len(scope)} messages")
    if keyed["recipe_tag"] != kg_index.recipe_tag(keyed["key_render"]):
        raise CommitRefused(
            f"commit: the index keyed under {keyed['recipe_tag']!r}, this process under "
            f"{kg_index.recipe_tag(keyed['key_render'])!r} (check the deixis env)"
        )
    records = stage("select", "sources.json")
    try:
        sessions = {session_number(r["session_id"]) for r in records}
    except ChainError as exc:
        raise CommitRefused(f"commit: {exc}") from None
    if sessions != {chain.sessions + 1}:
        raise CommitRefused(f"commit: the run's select rows are not session_{chain.sessions + 1} ({sorted(sessions)})")
    since = chain.explicit_since
    if since is None:
        since = next((r["order"] for r in records if r.get("addressees")), None)
    conversation = _read(run_root.parent / RUN_CONVERSATION)["conversation"]
    model = driver_start_model(run_root)
    if model is None:
        raise CommitRefused(f"commit: {run_root}: the run's driver-start records no extraction model")
    return CommitInput(
        group=group,
        run=run,
        rows=stage("write", COMMIT_ROWS),
        write_vectors=stage("write", WRITE_VECTORS),
        keys=dict(keyed["keys"]),
        key_vectors=dict(keyed["vectors"]),
        key_render=str(keyed["key_render"]),
        scope=scope,
        meta_update={
            "sessions": chain.sessions + 1,
            "explicit_since": since,
            "agent_handles": list(dict.fromkeys([*chain.agent_handles, *(conversation.get("agent_handles") or ())])),
            "extraction_model_json": json.dumps(model, sort_keys=True),
        },
        expected={"Episodic": int(gate["episodic"]["expected"]), "Saga": int(gate["saga"]["expected"])},
        generation=str(generation),
    )


async def guard(tx: Any, ci: CommitInput) -> dict[str, list]:
    """Step 1 (``GUARD``): ``committed_run`` ``run - 1`` -> ``run`` on the locked meta node of ``ci.generation``.
    Returns its stored handle lists. When it does not match: ``_Landed`` if this run already landed (same generation,
    ``committed_run`` = ``run``, every Episodic of ``ci.scope`` in the group), else ``StaleCommit`` naming what it
    found."""
    async with tx.session() as s:
        found = [dict(r) async for r in await s.run(GUARD, g=ci.group, run=ci.run, generation=ci.generation)]
        if found:
            return {"handles": list(found[0]["handles"] or ()), "runs": list(found[0]["runs"] or ())}
        meta = [dict(r) async for r in await s.run(_META, g=ci.group)]
        held = int((await (await s.run(_HELD, g=ci.group, ids=sorted(set(ci.scope)))).single())["n"])
    what = f"commit of run {ci.run} into {ci.group!r} refused"
    if not meta:
        raise StaleCommit(f"{what}: the group has no BonfireMeta")
    run, generation = meta[0]["run"], meta[0]["generation"]
    if generation != ci.generation:
        raise StaleCommit(
            f"{what}: the bonfire was re-created (generation {generation!r}, the run's {ci.generation!r}); its "
            f"committed run is {run}"
        )
    if run == ci.run and held == len(set(ci.scope)):
        raise _Landed
    raise StaleCommit(f"{what}: the bonfire's committed run is {run}, not {ci.run - 1}")


def handle_lists(stored: Mapping[str, Sequence], wanted: Sequence[str], run: int) -> tuple[list[str], list[int]]:
    """``(agent_handles, agent_handle_runs)`` after the commit of ``run``: the stored handles with their runs, then
    each handle of ``wanted`` the bonfire did not hold, stamped ``run``. ``CommitRefused`` when the stored lists differ
    in length or ``wanted`` drops a stored handle."""
    handles, runs = list(stored["handles"]), [int(r) for r in stored["runs"]]
    if len(handles) != len(runs):
        raise CommitRefused(f"commit: BonfireMeta holds {len(handles)} agent handles but {len(runs)} handle runs")
    if dropped := [h for h in handles if h not in wanted]:
        raise CommitRefused(f"commit: the run's agent handles drop the bonfire's {dropped}")
    new = [h for h in dict.fromkeys(wanted) if h not in handles]
    return handles + new, runs + [run] * len(new)


async def commit_append(driver: Any, ci: CommitInput) -> dict[str, Any]:
    """Commit ``ci`` in ONE transaction (the module docstring's steps 1-7); returns ``{created, keys, guard,
    already_committed}``. ``StaleCommit`` / ``CommitRefused`` / any other error rolls the whole transaction back. A
    run that already landed writes nothing and returns ``already_committed: true`` with zero counts."""
    node_ids = [n["props"]["uuid"] for n in ci.rows["nodes"]]
    edge_keys = [(e["type"], e["props"]["uuid"]) for e in ci.rows["edges"]]
    tag = kg_index.recipe_tag(ci.key_render)
    try:
        return await _commit(driver, ci, node_ids, edge_keys, tag)
    except _Landed:
        return {
            "created": {"nodes": 0, "edges": 0},
            "keys": {"keys": len(ci.keys), "embedded": 0, "written": 0, "stamped": 0},
            "guard": {"previous": ci.run - 1, "committed_run": ci.run},
            "already_committed": True,
        }


async def _commit(driver: Any, ci: CommitInput, node_ids: list, edge_keys: list, tag: str) -> dict[str, Any]:
    """``commit_append``'s transaction (steps 1-7); ``_Landed`` from the guard leaves it rolled back."""
    from memory_kernel.graph.kg_writer import read_rows, write_rows
    from memory_kernel.pipeline.stages.index import scoped_keys
    from memory_kernel.pipeline.stages.write import Refused, readback_missing

    async with transaction(driver, commit=True) as tx:
        stored = await guard(tx, ci)
        first = await write_rows(tx, ci.rows, group_id=ci.group, embed=embed.ReplayEmbed(ci.write_vectors), append=True)
        again = await write_rows(tx, ci.rows, group_id=ci.group, embed=embed.ReplayEmbed(ci.write_vectors), append=True)
        if moved := {k: again[k] for k in IDEMPOTENT if again[k]}:
            raise CommitRefused(f"commit: the rows written a second time were not a no-op: {moved}")
        keyed = await kg_index.persist_keys(
            tx, ci.group, ci.keys, embed.ReplayEmbed(ci.key_vectors), key_render=ci.key_render
        )
        try:
            keys = await scoped_keys(tx, ci.group, list(ci.scope), ci.key_render)
        except Refused as exc:
            raise CommitRefused(f"commit: {exc}") from None
        if keys != ci.keys:
            drift = sorted(u for u in set(keys) | set(ci.keys) if keys.get(u) != ci.keys.get(u))
            raise CommitRefused(f"commit: key drift on {len(drift)} messages: {drift[:3]}")
        counts = await graphdb.label_counts(tx, ci.group)
        if counts != ci.expected:
            raise CommitRefused(f"commit: the group would hold {counts}, the dry pass {ci.expected}")
        missing = readback_missing(node_ids, edge_keys, *await read_rows(tx, ci.group, node_ids, edge_keys))
        if missing["missing_nodes"] or missing["missing_edges"]:
            raise CommitRefused(f"commit: the readback misses written rows: {missing}")
        async with tx.session() as s:
            stamps = {r["recipe"]: int(r["n"]) async for r in await s.run(_STAMPS, g=ci.group)}
        if stamps.get(tag, 0) != counts["Episodic"]:
            raise CommitRefused(f"commit: {stamps.get(tag, 0)} of {counts['Episodic']} Episodic keys carry {tag!r}")
        handles, runs = handle_lists(stored, ci.meta_update["agent_handles"], ci.run)
        props = {
            **ci.meta_update,
            "agent_handles": handles,
            "agent_handle_runs": runs,
            "committed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        async with tx.session() as s:
            await (await s.run(_SET_META, g=ci.group, p=props)).consume()
    return {
        "created": {"nodes": int(first["nodes_created"]), "edges": int(first["edges_created"])},
        "keys": keyed,
        "guard": {"previous": ci.run - 1, "committed_run": ci.run},
        "already_committed": False,
    }
