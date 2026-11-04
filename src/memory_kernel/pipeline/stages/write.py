"""build_dry, write and rewrite: the C2 projection of one conversation, contract-validated and invariant-checked
($0, no Neo4j), then written to ONE empty Neo4j group and re-written idempotently.

Ported from the reference harness:
  * ``build_dry`` / ``invariants`` / ``Refused`` / ``UNGATED``,
    ``_check_counts`` / ``_preflight`` / ``_write`` / ``write`` (here ``run_write``) /
    ``foreign_changes`` / ``write_problems`` / ``adopt_first_write`` / ``index_projection_usd``   <- the harness's build module
  * ``rewrite`` and the ``call_stage`` write branch (adopt, else write)                         <- the harness's run module
  * ``plan_expectations`` / ``episodic_checks``                                                            <- the harness's conversation-build module
  * ``expected_episodes``                                                                                 <- the harness's conversation module

The frozen projection is ``_g1_build_u2`` (+ ``_g1_build`` helpers, ``_invariants_gate.run_invariants``,
``_invariants_rows.frozen_sha``). The write never resets a group: a first write refuses a group with any node or edge.

Two write modes (``BuildConfig.write_mode``):
  * ``fresh`` (every ordinary build): the first write MERGEs into an EMPTY group.
  * ``append`` (``mk append``; always graph mode, below): the run projects ONE session against the chain state its
    forward input pins (``project``: the chain view, the mint run = the chain's run + 1, its chat's last saga as
    ``existing_sagas``) and MERGEs the new rows, the new edges onto stored nodes and the ``SAME_AS`` links into the
    bonfire's existing group. The group must be non-empty and its own (``check_owner``: every stored Saga carries
    this build's ``ingest_pass``). Every newly minted entity is stamped with its frozen ``minted_order``. Before
    anything is written, ``forward_gate`` refuses a row that is in the group but neither the chain's nor
    this session's (rule 1; ``own_uuids`` makes a crash replay pass), a stored node whose frozen fields would
    change or a stored edge whose props would (rule 2: a stored edge takes its union fields only, so a fact a later
    session restates keeps the ``key_text`` an older message's key was built from) and a forward result the chain
    commit would refuse; before the write, the group's Episodic and Saga totals less this session's own stored rows
    must equal the chain's (rule 3 before: ``pending_check``, so a session written but never committed is named
    before another session writes anything, while a crash replay of the same session passes), and after the write
    they must equal the chain's plus the session's (rule 3, enforced by ``write_problems``). Nothing is deleted (rule 4): an append produces no orphans and no drift
    by construction, so kernel 1.1's orphan and drift reports and ``--prune-orphans`` are gone. Only rows without an
    embedding are embedded (``kg_writer._already_embedded``). The projection's ``forward`` is kept in the write's
    ``summary.json`` (``forward_record``) for the commit after sanity (``pipeline.commit``). Its
    readback is the written rows only (``kg_writer.read_rows``; ``group_rows``, the shape, envelope, Episodic check,
    graph report, readback hashes and readback validation are over them). Every row read of an append is looked up by
    this session's uuids; what still spans the group is count-only (``group_counts``, ``label_counts``) or one
    ``DISTINCT`` saga-pass read (the owner check, once). No write reads other groups: the database-wide ``snapshot``
    (every group's counts, ``write_problems``) is taken only under ``BuildConfig.strict_db_snapshot``, so another
    tenant of a shared Neo4j writing at the same time never fails a write, and the writer refuses a row uuid another
    group holds (``kg_writer.ForeignUuids``). The group-wide
    ``nodes_outside_group_via_edges`` and ``cypher_dups`` are a full build's only (the writer's own
    ``_foreign`` check and rule 1 cover the written rows). A stored C2 contract node's aggregates (Question / Plan
    ``fold_*``, ``status``) are create-only in an append (``keep_contract_aggregates``); parked follow-up: a fold over
    the stored ``ANSWERS`` edges.
Rebindings replaced:
  * the harness's build binding (``INPUTS``, ``GROUP``, ``EXPECTED_MESSAGES``,
    ``EXPECTED_EPISODES`` rebound on the build modules) -> the arguments of
    ``load_inputs`` / ``project_u2``; ``EXPECTED_EPISODES`` is read from the boundary stage's output when the stage
    runs (never cached); ``assert_group`` is dropped: no module state is left to check;
  * the harness gate module's ``PRE_C1`` -> ``cfg.invariants_reference``;
  * the cost row is ``ctx.cost("build", ...)``;
  * the harness's group and build bindings + ``assert_group`` -> ``cfg.group`` and ``project(ctx)``;
    ``cypher_dups`` -> ``graphdb.cypher_dups(driver, group)``; ``_driver()`` -> ``graphdb.driver(cfg)`` (built
    inside each ``asyncio.run``); ``_voyage()`` -> ``CappedEmbed(embed.embedder(cfg, "document"))``; ``_cap_check``
    -> ``_common.cap_check(ctx, ...)``; ``spec.messages`` -> ``ctx.counts.messages``; ``CostRecorder`` ->
    ``ctx.cost``; ``ROOT / p`` -> ``cfg.root / p``.
  * Declared additions: ``summary.json`` / ``rerun.json`` carry ``"embedder"`` (``embed.record``); ``rewrite`` returns
    ``{**out, "gate": ...}`` (mk's child wraps the result once; the harness returned it pre-wrapped).
Graph mode (kernel 2.0, ``graph_mode``: every append): the group stays as committed
until ``pipeline.commit`` replays the run in ONE transaction. The write adds this run's type rows (``with_type_rows``:
the events stage's ``new_rolesets`` as ``Roleset`` rows, the forward routes the chain does not hold as ``Route`` rows
and ``ROUTES_TO`` edges), runs ``_forward_write`` in a rolled-back transaction with a ``RecordingEmbed`` and writes
``commit-rows.json`` (the rows as sent; the writer's own embeddings are not in them) and ``write-vectors.json`` (what
it embedded) before its ``summary.json`` (``"dry": true``). The rewrite replays those rows twice in one rolled-back
transaction (``_dry_rewrite``): the second pass must create and embed nothing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memory_kernel.pipeline import embed, graphdb, provenance, saga_vectors
from memory_kernel.pipeline.config import load_forward
from memory_kernel.pipeline.stages import _g1_build as base
from memory_kernel.pipeline.stages import _g1_build_u2 as u2
from memory_kernel.pipeline.stages._common import cap_check
from memory_kernel.pipeline.stages._conversation import conversations_of
from memory_kernel.pipeline.stages._g1_cards import write_once
from memory_kernel.pipeline.stages._invariants_gate import load_rows, run_invariants
from memory_kernel.pipeline.stages._invariants_rows import frozen_sha

if TYPE_CHECKING:
    from memory_kernel.pipeline.runner import Ctx

_log = logging.getLogger(__name__)
NO_EPISODES = -1  # EXPECTED_EPISODES sentinel before the boundary stage exists: the frozen episode check refuses
#: Invariant checks that gate a build; ``budget`` is reported only (its reference is the development conversation's earlier baseline).
UNGATED = frozenset({"budget"})
SUMMARY = "summary.json"
RERUN = "rerun.json"
#: Graph mode: the rows the dry write sent (``write_rows``'s input) and the vectors it embedded, ``{text: vector}``.
COMMIT_ROWS = "commit-rows.json"
WRITE_VECTORS = "write-vectors.json"


class Refused(RuntimeError):
    """A build precondition failed; nothing was written."""


def graph_mode(cfg: Any) -> bool:
    """Kernel 2.0's graph mode: an append (``write_mode`` append) is always one, since a bonfire's state lives in its
    graph and ``pipeline.commit`` is the only writer of an append's rows. (``BuildConfig.graph_commit`` is kept for the
    build-config.json files that carry it.)"""
    return cfg.write_mode == "append"


class PendingSession(Refused):
    """A forward append found rows of ANOTHER session in the group that the chain does not hold (a run that wrote and
    then failed before its commit); nothing was written (``pending_check``)."""


# -- pure helpers (the harness's build module) -----------------------------------------------------------------------


def episodic_checks(nodes: list[dict]) -> dict:
    """Every :Episodic must carry sender, told_at, act and mood."""
    eps = [n["props"] for n in nodes if "Episodic" in n["labels_neo4j"]]
    missing = {f: sum(1 for p in eps if not p.get(f)) for f in ("sender", "told_at", "act", "mood")}
    return {"episodic": len(eps), "missing": missing, "passed": bool(eps) and not any(missing.values())}


def plan_expectations(rows_shape: Mapping, occurrence_predicates: int, episodes: int, messages: int) -> dict:
    """Expectations: N Saga = boundary-stage episodes, one Episodic per message, one Event per occurrence."""
    by = rows_shape["nodes_by_label"]
    sagas = int(by.get("Saga", 0))
    episodic = int(by.get("Episodic", 0))
    events = sum(int(v) for k, v in by.items() if "Event" in k.split("+"))
    return {
        "saga": {"got": sagas, "expected": episodes, "passed": sagas == episodes},
        "episodic": {"got": episodic, "expected": messages, "passed": episodic == messages},
        "event": {
            "got": events,
            "occurrence_predicates": occurrence_predicates,
            "diff": events - occurrence_predicates,
            "note": "development conversation: 698 Event nodes for 695 occurrence predicates",
        },
    }


# -- inputs ----------------------------------------------------------------------------------------------------------


def expected_episodes(ctx: Ctx) -> int:
    """Distinct episodes of the boundary stage's applied map, or ``NO_EPISODES`` before it exists."""
    applied = ctx.path("boundary") / "applied-boundaries.json"
    if not applied.is_file():
        return NO_EPISODES
    return len(set(json.loads(applied.read_bytes())["episode_by_message"].values()))


def projection_inputs(ctx: Ctx) -> dict[str, str]:
    """The projection's inputs as root-relative path strings: graph_a's records (1.8: the walk boundaries + picks
    graph; no Jev1 labels, alignments or card inventory), the ctx stage's eventuality and the boundary map."""
    rel = lambda p: provenance.rel(p, ctx.cfg.root)  # noqa: E731
    return {
        "graph": rel(ctx.path("graph_a") / "graph.json"),
        "eventuality": rel(ctx.path("ctx") / "eventuality.json"),
        "episode_map": rel(ctx.path("boundary") / "applied-boundaries.json"),
    }


def _stored_keys(data: Mapping[str, Any]) -> dict[str, str] | None:
    """A forward append's chain entities, uuid -> the key each was minted under (None on a full build)."""
    chain = data.get("chain")
    return None if chain is None else {uuid: m.key for uuid, m in chain.entities.items()}


def invariants(rows: dict[str, Any], reference: Path | None) -> dict[str, Any]:
    """``run_invariants`` against the budget reference (None: the budget is skipped and recorded as such, the seven row
    checks still run); refuses on any gated violation."""
    report = run_invariants(rows, load_rows(reference) if reference is not None else None)
    gated = {k: v for k, v in report["violations"].items() if k not in UNGATED and v}
    report["gated_violations"] = gated
    if gated:
        raise Refused(f"C2 invariants violated: {gated}; samples {report['violation_samples']}")
    return report


def project(
    ctx: Ctx, *, referee: Any = "auto", centroids: Callable[[dict], Mapping[str, dict]] | None = None
) -> tuple[dict, dict, dict]:
    """``(data, {"ingest_pass": cfg.ingest_pass}, rows)``: load the stage outputs and project them through the frozen
    projection (the middle element is what the summaries record).

    ``centroids`` (the write: ``saga_centroids``) is called with the loaded inputs and gives every Saga its centroid
    properties (``g1_projection.project(saga_vectors=...)``); None (``build_dry``, the referents stage, the chain's
    bootstrap) projects Sagas without them.

    ``referee`` decides an append's specific mentions (``graph.forward.resolve``): ``"auto"`` (build_dry, the write)
    reads the referents stage's ``decisions.json`` in an append (refused when it is missing) and is None in a full
    build; None projects without one (the backfill); a callable is used as given (the referents stage's collecting
    pass)."""
    cfg = ctx.cfg
    inputs = projection_inputs(ctx)
    data = u2.load_inputs(
        cfg.root,
        inputs,
        frame_dirs=cfg.frame_dirs,
        expected_messages=ctx.counts.messages,
        expected_episodes=expected_episodes(ctx),
        session_conversations=conversations_of(ctx),
    )
    forward = load_forward(cfg)
    if forward is not None:  # a forward append resolves against the chain its run pinned (kernel 1.2)
        from memory_kernel.pipeline.chain import from_projection_view

        if "chain" not in forward or "chat" not in forward:
            raise Refused(f"{cfg.forward_input}: the forward input carries no chain view or names no chat")
        # The mint run is one past the pinned chain's last committed run: a pure function of the chain, so a replay
        # of the same session (a new runs/NNNN after "sealed but not committed") mints exactly the same rows.
        data["chain"] = from_projection_view(forward["chain"])
        data["run"] = data["chain"].run + 1
        data["existing_sagas"] = chain_sagas(data["chain"], forward["chat"], cfg.group, cfg.ingest_pass)
        if referee == "auto":
            referee = decided_referee(ctx)
    elif referee == "auto":
        referee = None
    kw = {"ingest_pass": cfg.ingest_pass}
    vectors = centroids(data) if centroids is not None else None
    rows = u2.project_u2(
        data,
        group=cfg.group,
        frame_dirs=cfg.frame_dirs,
        mode="b",
        referee=referee,
        saga_vectors=vectors,
        referent_policy=getattr(cfg, "referent_policy", "phrase"),
        **kw,
    )
    return data, kw, rows


def run_vectors(ctx: Ctx, texts: Iterable[str], cost: provenance.CostRecorder) -> Mapping[str, list[float]]:
    """``{text: vector}`` for every one of ``texts``, from the events stage's cache (``runner.events_voyage_cache``).

    A text the cache lacks (the default segmenter never embeds a short time segment, so a ``walk`` build's cache
    misses some) is embedded here through the same cache, once: its Voyage spend goes on this stage's cost row and
    ``--cap-usd`` stops it (``llm.StopSpend``) before the call. Under ``segmenter`` the boundary already embedded every
    text, so nothing is sent. With nothing missing, nothing is added to the cost row."""
    from memory_kernel.pipeline import llm
    from memory_kernel.pipeline.runner import events_voyage_cache

    cfg = ctx.cfg
    wanted = list(dict.fromkeys(t for t in texts if t is not None))
    path = events_voyage_cache(cfg)
    cache = embed.TextVoyageCache(path, cfg.voyage_api_key)
    missing = [t for t in wanted if t not in cache.vectors]
    if missing:
        if not cfg.voyage_api_key:
            raise Refused(
                f"saga centroids: {len(missing)} message texts are not in the events cache {path} and there is no "
                "Voyage key to embed them"
            )
        spend = embed.voyage_usd(sum(embed.token_est(t) for t in missing))
        paid = provenance.ledger_summary(cfg.ledger)["paid_usd"]
        if paid + spend > cfg.cap_usd:
            raise llm.StopSpend(f"unit cap {cfg.cap_usd} crossed by the saga-centroid voyage spend {spend:.4f}")
        try:
            cache.create_batch(missing)
        finally:  # the live client's count, as the boundary stage records it
            inner = cache.inner
            if inner is not None:
                tokens = int(inner.tokens)
                requests = int(getattr(inner, "requests", 0))
                cost.add("voyage", requests=requests, input_tokens=tokens, usd=embed.voyage_usd(tokens))
                cost.note += f"; saga centroids embedded {len(missing)} message texts the events cache lacked"
    return {t: cache.vectors[t] for t in wanted if t in cache.vectors}


def saga_centroids(ctx: Ctx, data: Mapping[str, Any], cost: provenance.CostRecorder) -> dict[str, dict]:
    """Every Saga's centroid (``saga_vectors.saga_centroids``) from the events stage's cache: its members are the
    projection's (``saga_members``: a grown Saga's ``prior`` messages included, whose texts come from the chat's chain
    tail rows), their texts ``embed.message_texts``; the vectors are ``run_vectors``' (what the cache lacks is
    embedded there)."""
    from memory_kernel.pipeline.stages.ingest import load_sources

    cfg = ctx.cfg
    members = saga_vectors.saga_members(data)
    rows = list(load_sources(ctx)[1])
    if data.get("chain") is not None:  # a forward append: a continued Saga's prior messages are the chat's tail rows
        state = data["chain"].chats.get(str((load_forward(cfg) or {}).get("chat")))
        rows = [*(state.tail if state is not None else ()), *rows]
    texts = embed.message_texts(rows)
    vectors = run_vectors(ctx, (texts.get(m) for ids in members.values() for m in ids), cost)
    return saga_vectors.saga_centroids(members, texts, vectors)


def stamp_run(
    ctx: Ctx, data: Mapping[str, Any], rows: Mapping[str, Any], cost: provenance.CostRecorder, *, append: bool
) -> dict[str, int]:
    """The 2.0 state properties on the projected rows (``graph.state.stamp_state``): the run is the append's mint run
    (``data["run"]``, one past the pinned chain's) or 0 for a full build; the records are this run's select rows."""
    from memory_kernel.graph.state import stamp_state
    from memory_kernel.pipeline.stages.ingest import load_sources

    records = list(load_sources(ctx)[1])
    vectors = run_vectors(ctx, embed.message_texts(records).values(), cost)
    try:
        return stamp_state(
            rows,
            run=data["run"] if append else 0,
            records=records,
            chat_of=data["conversation_of"],
            vectors=vectors,
        )
    except ValueError as exc:
        raise Refused(str(exc)) from exc


def decided_referee(ctx: Ctx) -> Any:
    """The referee of an append's projection: the referents stage's decision per point (None: today's resolution)."""
    target = ctx.path("referents") / "decisions.json"
    if not target.is_file():
        raise Refused(f"{target}: missing (the referents stage runs before build_dry in an append)")
    decided = json.loads(target.read_bytes())["decisions"]
    return lambda p: decided.get(p.point)


# -- the stage ---------------------------------------------------------------------------------------------------------


def build_dry(ctx: Ctx, live: bool) -> dict:  # noqa: ARG001 ($0, local)
    """Projection, contract validation, C2 invariants and plan counts (Saga == episodes, Episodic == messages)."""
    from memory_kernel.graph.kg_contract import validate_rows

    cfg = ctx.cfg
    out_dir = ctx.path("build_dry")
    target = out_dir / SUMMARY
    if target.exists():
        raise FileExistsError(f"{target} exists; write-once (bump the stage dir)")
    inputs = projection_inputs(ctx)
    with ctx.cost("build", run=out_dir.name, note=f"C2 dry projection {cfg.sample_id}, $0"):
        data, kw, rows = project(ctx)
        inv = invariants(rows, cfg.invariants_reference)
        errors = validate_rows(rows["nodes"], rows["edges"], previous=rows["external_endpoints"])
        shape = base.shape(rows["nodes"], rows["edges"])
        ev = json.loads((cfg.root / inputs["eventuality"]).read_bytes())
        result = {
            "sample_id": cfg.sample_id,
            "group": cfg.group,
            "ingest_pass": kw["ingest_pass"],
            **shape,
            "validation_errors": len(errors),
            "validation_error_sample": errors[:5],
            "expectations": plan_expectations(
                shape,
                int(ev["distribution_predicates"].get("occurrence", 0)),
                expected_episodes(ctx),
                ctx.counts.messages,
            ),
            "invariants": inv,
            "envelope": base.envelope_census(rows["nodes"], rows["edges"]),
            "routing": rows["routing"],
            "dedup": rows["dedup"],
            **base.graph_report(rows["nodes"], rows["edges"], stored_keys=_stored_keys(data)),
            "episodic_check": episodic_checks(rows["nodes"]),
            "row_set_sha256": base.row_set_hash(rows["nodes"], rows["edges"]),
            "frozen_sha": frozen_sha(rows),
            "inputs": {n: {"path": p, "sha256": base.sha256_file(cfg.root / p)} for n, p in inputs.items()},
        }
        result["passed"] = (
            not errors
            and result["expectations"]["saga"]["passed"]
            and result["expectations"]["episodic"]["passed"]
            and result["entity_name_dups_excl_events"] == 0
        )
    write_once(target, result)
    return result


# -- write / rewrite ------------------------------------------------------------------------


def _out(ctx: Ctx, key: str, name: str) -> Path:
    """The write-once output file; refused up front so no stage spends and then fails on its own output."""
    target = ctx.path(key) / name
    if target.exists():
        raise FileExistsError(f"{target} exists; write-once (bump the stage dir)")
    return target


def index_projection_usd(ctx: Ctx) -> float:
    """1.5 x what write, rewrite and index cost on the reference conversation (sanity is a $0 receipt), scaled by
    message count. (It was the whole-build rate, ``INDEX_USD_PER_MESSAGE``: ~2,000x the write's own cost, so a $1.44
    chat bootstrap projected $4.37 for a ~$0.01 write and needed a $6 cap.)"""
    from memory_kernel.pipeline.runner import INDEX_USD, REF_MESSAGES, STOP_FACTOR

    reference = sum(INDEX_USD[stage] for stage in ("write", "rewrite", "index", "sanity"))
    return STOP_FACTOR * reference * ctx.counts.messages / REF_MESSAGES


def _check_counts(group: str, counts: Mapping[str, Any], *, rewrite: bool, append: bool = False) -> None:
    """The group-state rule of a write. ``fresh`` (``append=False``): a first write needs an EMPTY group, a rewrite a
    non-empty one. ``append``: the write merges into the bonfire's existing group, so the group must be NON-empty
    (first write and rewrite alike); ``check_owner`` then checks it is the bonfire's."""
    if append:
        if not counts["nodes"]:
            raise Refused(
                f"append to {group!r} but the group is empty: a bonfire's group holds its bootstrap's write "
                "(build the bootstrap through seal, or use a fresh build)"
            )
        return
    if not rewrite and (counts["nodes"] or counts["edges"]):
        raise Refused(f"group {group!r} is not empty ({counts}); first write refused (never reset)")
    if rewrite and not counts["nodes"]:
        raise Refused(f"rewrite of {group!r} but the group is empty; the first write is missing")


def check_owner(group: str, saga_passes: list[list[str]], ingest_pass: str | None) -> None:
    """An append-mode write merges only into the bonfire's OWN group: the group has at least one ``:Saga`` and every
    one carries this build's ``ingest_pass`` (the first write stamps it on every saga; ``mk append`` inherits it from
    the bootstrap's config). ``saga_passes`` = ``graphdb.saga_passes``."""
    if not saga_passes:
        raise Refused(
            f"append to {group!r}: the group has no Saga, so no kernel build wrote it (no ingest pass to check that it "
            "belongs to this bonfire)"
        )
    foreign = [p for p in saga_passes if ingest_pass not in p]
    if foreign:
        seen = sorted({x for p in foreign for x in p} or {"<none>"})
        raise Refused(
            f"append to {group!r}: the group does not belong to this bonfire: {len(foreign)} of the "
            f"{len(saga_passes)} distinct ingest-pass lists of its sagas "
            f"sagas lack its ingest pass {ingest_pass!r} (they carry {seen[:5]})"
        )


async def _preflight(ctx: Ctx, *, rewrite: bool) -> dict[str, int]:
    """Read-only group check before any ledger row, projection or embedder (emptiness; in append mode non-empty and
    owned, ``check_owner``)."""
    cfg = ctx.cfg
    append = cfg.write_mode == "append"
    driver = graphdb.driver(cfg)
    try:
        counts = await graphdb.group_counts(driver, cfg.group)
        passes = await graphdb.saga_passes(driver, cfg.group) if append else None
    finally:
        await driver.close()
    _check_counts(cfg.group, counts, rewrite=rewrite, append=append)
    if passes is not None:
        check_owner(cfg.group, passes, cfg.ingest_pass)
    return counts


#: Saga extension fields a forward append moves on its chat's continued saga: the episode grows (its start is
#: create-only, ``kg_writer.CREATE_ONLY``) and its centroid is recomputed whole (a Saga stored before centroids gets
#: its first one).
SAGA_GROWS = frozenset({"ended_at", "message_count"}) | saga_vectors.CENTROID_FIELDS
#: The C2 coding vocabulary: group-level rows (one Purpose, a Question per qid, a Plan per card) every run sends again
#: under the same uuid. Their extension fields are aggregates over the codings (``fold_*``, ``status``); in a forward
#: append they are create-only (``keep_contract_aggregates``).
CONTRACT_KINDS = frozenset({"Purpose", "Question", "Plan", "RevisionPass"})
#: Props the writer owns (``write_rows`` embeds them for rows that have none) or the index stamps: never compared.
WRITER_OWNED = frozenset(
    {
        "name_embedding",
        "fact_embedding",
        "search_key",
        "search_key_sha",
        "search_key_model",
        "search_key_recipe",
        "search_key_embedding",
    }
)
#: A stored node's fields a forward append never changes, compared with the projection whatever the writer would do:
#: the identity of an entity (its resolution key and mint order) and of a saga (its first message and start).
FROZEN_IDENTITY = {
    "Entity": ("name_key", "minted_order"),
    "Saga": ("first_episode_uuid", "started_at"),
}


def _plain(value: Any) -> Any:
    """Neo4j returns lists: compare a projected tuple as a list."""
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def chain_sagas(chain: Any, chat: str, group: str, ingest_pass: str | None) -> list[dict[str, Any]]:
    """``project(existing_sagas=...)`` of a forward append (Review Focus 4): its chat's last saga, so the session's
    first NEW saga links from it by ``NEXT_SAGA`` (``chain.chats[chat].last_saga``, never another chat's), and a
    session that continues the chat's open episode reuses it (``NEXT_EPISODE`` from its stored last message; its
    ``prior`` messages, the open episode's rows of the chat's tail, keep its start and count whole). ``[]`` for a new
    chat."""
    from memory_kernel.graph.g1_ids import content_uuid

    state = chain.chats.get(chat)
    if state is None or not state.tail:
        return []
    first_held = int(state.boundary["messages"]) - len(state.tail)
    start = int(state.boundary["episode_start"])
    if start < first_held:
        raise Refused(f"chat {chat!r}: the chain's tail does not reach back to its open episode (start {start})")

    def ep(message_id: str) -> str:
        return content_uuid("ep", group, message_id)

    prior = [
        (r["message_id"], ep(r["message_id"]), r["speaker"], r["timestamp"]) for r in state.tail[start - first_held :]
    ]
    return [
        {
            "uuid": state.last_saga,
            "name": state.last_episode,
            "episodes": [ep(state.tail[-1]["message_id"])],
            "ingest_pass": [ingest_pass] if ingest_pass else [],
            "prior": prior,
        }
    ]


def own_uuids(forward: Mapping, records: list[Mapping], opened: list[str], group: str) -> set[str]:
    """The uuids this projection mints for its session, DERIVED from the session's identity alone (so a replay after a
    crash names the same ones, Review Focus 1): its messages' Episodic, its new episodes' Saga, its Events (``evt`` of
    each record's unit, roleset and anchor) and ``forward["minted"]``, each of which must be ``content_uuid("ent",
    group, key)`` of its own key (a mint under another key's uuid is a re-key: ``Refused``). ``records``: the session's
    graph records."""
    from memory_kernel.graph.g1_ids import content_uuid

    own = {content_uuid("ep", group, r["message_id"]) for r in records}
    own |= {content_uuid("saga", group, e) for e in opened}
    own |= {
        content_uuid("evt", group, r["unit"], ev["roleset"], *ev["anchor"])
        for r in records
        for ev in r.get("events") or ()
    }
    for m in forward["minted"]:
        if m.uuid != content_uuid("ent", group, m.key):
            raise Refused(f"forward gate: minted entity {m.uuid} is not the uuid of its key {m.key!r} (a re-key)")
        own.add(m.uuid)
    return own


def _frozen_changes(row: Mapping, have: Mapping) -> list[str]:
    """The fields a stored node ``have`` would change, or that differ where they are frozen, if ``row`` were written
    on it (rule 2). Labels: no NEW label (``SET n:extra`` only adds; a stored Place reused as a plain entity is fine),
    except a kind label (``KIND_LABELS``) on a row with ``kinds_replace`` (structure: its kind was decided again over
    the stored evidence and replaces the stored kind, ``kg_writer.kind_label_clause``).
    An Episodic: every prop. Otherwise the ``FROZEN_IDENTITY`` fields and ``name``, then what the writer would SET on
    match (``kg_writer._split``): only ``EVOLVING`` fields and a continued saga's ``SAGA_GROWS`` may differ (a stored
    contract node's aggregates are dropped before the gate, ``keep_contract_aggregates``). Union fields only grow and
    ``summary_append`` lines only append (``append_summary``)."""
    from memory_kernel.graph.g1_names import KIND_LABELS
    from memory_kernel.graph.kg_contract import EVOLVING, NODE_BASE
    from memory_kernel.graph.kg_writer import SUMMARY_APPEND, _clean, _primary, _split

    kind = _primary(row["labels_neo4j"])
    stored = have["props"]
    props = {k: v for k, v in row["props"].items() if k not in SUMMARY_APPEND and k not in WRITER_OWNED}
    new = set(row["labels_neo4j"]) - set(have["labels_neo4j"])
    if row.get("kinds_replace"):
        new -= set(KIND_LABELS)
    out = ["labels"] if new else []

    def differs(key: str) -> bool:
        return props.get(key) is not None and _plain(stored.get(key)) != _plain(props[key])

    if kind == "Episodic":
        return out + sorted(k for k in props if differs(k))
    out += [k for k in ("name", *FROZEN_IDENTITY.get(kind, ())) if differs(k)]
    _, update = _split(_clean(props), NODE_BASE[kind])
    grows = EVOLVING | (SAGA_GROWS if kind == "Saga" else frozenset())
    # kind evidence (``kind_person``, ``kind_head_<K>``, ...) is set to the projection's merged row on every write
    return out + sorted(
        k for k in update if k not in grows and k not in out and not k.startswith("kind_") and differs(k)
    )


def _edge_changes(row: Mapping, have: Mapping) -> list[str]:
    """The props a stored edge ``have`` would change if ``row`` were written on it (rule 2 for edges): what the writer
    SETs on a stored edge in append mode (``kg_writer.edge_match_update``: its union fields grow, its embedding is the
    writer's) besides those. Empty by construction; the check holds the writer to it."""
    from memory_kernel.graph.kg_contract import BASE_RELATES_TO
    from memory_kernel.graph.kg_writer import STRUCTURAL_EDGES, _clean, _split, edge_match_update

    structural = STRUCTURAL_EDGES.get(str(row["type"]))
    _, update = _split(_clean(dict(row["props"])), structural[0] if structural else BASE_RELATES_TO)
    update = edge_match_update(update, append=True)
    stored = have["props"]
    return sorted(k for k in update if k not in WRITER_OWNED and _plain(stored.get(k)) != _plain(update[k]))


def contract_uuid(row: Mapping, group: str) -> str | None:
    """The group-constant uuid a C2 contract row must carry (the 1.7 codings module, archived: Purpose per group,
    Question per qid, Plan per plan id = its name); None for another kind."""
    from memory_kernel.graph.g1_ids import content_uuid
    from memory_kernel.graph.kg_writer import _primary

    props, kind = row["props"], _primary(row["labels_neo4j"])
    if kind == "Purpose":
        return content_uuid("purpose", group)
    if kind == "Question" and props.get("qid") is not None:
        return content_uuid("q", group, props["qid"])
    if kind == "Plan":
        return content_uuid("plan", group, props.get("name"))
    return None


def forward_gate(
    rows: Mapping,
    stored_nodes: list[dict],
    stored_edges: list[dict],
    chain: Any,
    own_uuids: set[str],
    *,
    chat: str,
    totals: Mapping[str, Mapping[str, int]] | None = None,
    fold: Mapping[str, Any] | None = None,
) -> None:
    """The forward write gate; ``Refused`` naming the first violation.

    1. Every row is the chain's or this session's: a node already in the group must be a chain entity, one of
       ``own_uuids`` (its own derived uuids: a crash replay passes), a stored row of an earlier run the chain accounts
       for (the session's ``chat``: its last Saga and its last message's Episodic) or a C2 coding row carrying its
       group-constant uuid (``contract_uuid``); a projected entity must be a chain entity or minted here; every edge
       ends at such rows, and an edge already in the group joins the same two rows.
    2. A stored node changes no frozen field (``_frozen_changes``), a stored edge no prop (``_edge_changes``).
    3. With ``totals`` (``{"Episodic"|"Saga": {"got", "expected"}}``, read after the write): each total equals the
       chain's plus this session's.
    4. (By construction: the append write path issues no DELETE, ``kg_writer.write_rows`` only MERGEs; a structure
       row's ``kinds_replace`` removes only the kind labels it does not carry, ``kg_writer.kind_label_clause``.)

    ``fold`` (the write's ``forward_record``): the commit's own checks (``chain._fold_forward``: mints new to the
    chain at its run + 1, routes and links that stay inside it) run here, so a run the chain cannot take is refused
    before anything is written.

    ``stored_nodes`` / ``stored_edges``: the group's rows among the projected uuids (``kg_writer.read_rows``), read
    before the write."""
    from memory_kernel.graph.g1_ids import content_uuid
    from memory_kernel.graph.kg_writer import _primary

    groups = {n["props"]["group_id"] for n in rows["nodes"]}
    known = set(chain.entities) | set(own_uuids)
    state = chain.chats.get(chat)
    if state is not None:
        known.add(state.last_saga)
        if state.tail:
            known |= {content_uuid("ep", g, state.tail[-1]["message_id"]) for g in groups}
    known |= {
        n["props"]["uuid"]
        for n in rows["nodes"]
        if _primary(n["labels_neo4j"]) in CONTRACT_KINDS
        and n["props"]["uuid"] == contract_uuid(n, n["props"]["group_id"])
    }
    have_nodes = {n["props"].get("uuid"): n for n in stored_nodes}
    have_edges = {e["props"].get("uuid"): e for e in stored_edges}
    for n in rows["nodes"]:
        uuid, labels = n["props"]["uuid"], n["labels_neo4j"]
        if uuid in known:
            continue
        what = f"{'+'.join(labels)} {uuid} ({n['props'].get('name')!r})"
        if uuid in have_nodes:
            raise Refused(f"forward gate rule 1: {what} is already in the group but neither the chain's nor this run's")
        raise Refused(f"forward gate rule 1: {what} is neither in the chain nor minted by this run")
    for e in rows["edges"]:
        uuid = e["props"]["uuid"]
        outside = [end for end in (e["source"], e["target"]) if end not in known]
        if outside:
            raise Refused(
                f"forward gate rule 1: {e['type']} {uuid} ends at {outside[0]}, outside the chain and the run"
            )
        have = have_edges.get(uuid)  # read by type and uuid: the same type
        if have is not None and (have["source"], have["target"]) != (e["source"], e["target"]):
            raise Refused(f"forward gate rule 1: {e['type']} {uuid} is stored between other rows")
    for n in rows["nodes"]:
        have = have_nodes.get(n["props"]["uuid"])
        changed = _frozen_changes(n, have) if have is not None else []
        if changed:
            raise Refused(
                f"forward gate rule 2: stored {'+'.join(have['labels_neo4j'])} {n['props']['uuid']} would change "
                f"frozen fields {changed}"
            )
    for e in rows["edges"]:
        have = have_edges.get(e["props"]["uuid"])
        changed = _edge_changes(e, have) if have is not None else []
        if changed:
            raise Refused(f"forward gate rule 2: stored {e['type']} {e['props']['uuid']} would change {changed}")
    if fold is not None:
        from memory_kernel.pipeline.chain import ChainError, _fold_forward

        try:
            _fold_forward(chain, fold)
        except ChainError as exc:
            raise Refused(f"forward gate: the chain cannot take this run's forward result ({exc})") from None
    for label, count in (totals or {}).items():
        if count["got"] != count["expected"]:
            raise Refused(
                f"forward gate rule 3: the group holds {count['got']} {label} rows, the chain and this session "
                f"{count['expected']}"
            )


def pending_check(
    counts: Mapping[str, int],
    stored_nodes: list[dict],
    totals: Mapping[str, Any],
    *,
    episodic: list[str],
    sagas: list[str],
    chain_run: Any = None,
) -> dict[str, dict[str, int]]:
    """Rule 3 BEFORE the write (``PendingSession``, nothing written): the group's Episodic total (``counts``, read
    before the write) minus this session's own Episodic rows already stored (``episodic`` among ``stored_nodes``: a
    crash replay of the SAME session wrote them) must equal the chain's message total, and the same for the Saga
    total against the session's opened sagas (``sagas``) and the chain's episode total. More rows: another session
    was written and never committed (rerun it first); fewer: the group is not the chain's projection."""
    stored = {n["props"].get("uuid") for n in stored_nodes}
    own = {"Episodic": sum(u in stored for u in episodic), "Saga": sum(u in stored for u in sagas)}
    chain = {"Episodic": int(totals["messages"]), "Saga": int(totals["episodes"])}
    extra = {k: int(counts[k]) - own[k] - chain[k] for k in chain}
    shown = ", ".join(f"{counts[k]} {k} (the chain {chain[k]}, this session's stored {own[k]})" for k in chain)
    if any(v > 0 for v in extra.values()):
        after = f"after chain run {chain_run} " if chain_run is not None else ""
        raise PendingSession(
            f"forward gate: the group holds {shown}: a run {after}wrote another session but was not committed: rerun "
            "that session first (mk append names its run); nothing was written"
        )
    if any(v < 0 for v in extra.values()):
        raise Refused(f"forward gate: the group holds {shown}: fewer rows than the chain (not its projection)")
    return {k: {"group": int(counts[k]), "chain": chain[k], "own_stored": own[k]} for k in chain}


def forward_record(forward: Mapping, *, run: int, episodic: list[str], own: set[str]) -> dict[str, Any]:
    """What the write's ``summary.json`` keeps of the projection's ``forward`` (JSON): the commit folds
    ``minted`` / ``routes`` / ``same_as`` / ``conflicts`` (``pipeline.commit``), the index keys ``episodic``
    (the session's new Episodic uuids); ``touched`` is reported. ``kinds`` (``referent_policy`` structure only: each
    written entity's merged kind evidence) is folded into the chain's ``kinds``."""
    from dataclasses import asdict

    kinds = {"kinds": dict(forward["kinds"])} if "kinds" in forward else {}
    return {
        "run": run,
        "minted": [asdict(m) for m in forward["minted"]],
        "routes": dict(forward["routes"]),
        "same_as": [asdict(x) for x in forward["same_as"]],
        "conflicts": list(forward["conflicts"]),
        "touched": list(forward["touched"]),
        "mentioned": {u: sorted(ms) for u, ms in sorted(forward.get("mentioned", {}).items())},
        "episodic": sorted(episodic),
        "own": len(own),
        **kinds,
    }


def session_continuation(ctx: Ctx) -> dict[str, Any]:
    """The boundary's ``continuation``: the chat's walk after the session (its opened episodes, its ``inversion``)."""
    applied = json.loads((ctx.path("boundary") / "applied-boundaries.json").read_bytes())
    end = applied.get("continuation")
    if end is None:
        raise Refused("forward append: the boundary output carries no continuation (the chat's walk after the session)")
    return end


def session_opened(ctx: Ctx) -> list[str]:
    """The episodes this session opened (the boundary's ``continuation``): its new Sagas."""
    return list(session_continuation(ctx)["opened"])


def keep_contract_aggregates(rows: Mapping, stored: set[str]) -> int:
    """A forward append sends a STORED C2 contract node (Purpose / Question / Plan) with its base fields only, so its
    aggregates (``fold_*``, ``status``, computed from this session's codings alone) never overwrite the stored ones:
    create-only in append mode. A contract node first written by this session keeps them. Returns how many rows.

    Parked follow-up: recompute a stored Question's fold over its stored ``ANSWERS`` edges after the write."""
    from memory_kernel.graph.kg_contract import NODE_BASE
    from memory_kernel.graph.kg_writer import _primary

    kept = 0
    for n in rows["nodes"]:
        kind = _primary(n["labels_neo4j"])
        if kind in CONTRACT_KINDS and n["props"]["uuid"] in stored:
            keep = NODE_BASE[kind] | {"qid"}  # a Question's qid names its group-constant uuid (``contract_uuid``)
            n["props"] = {k: v for k, v in n["props"].items() if k in keep}
            kept += 1
    return kept


def readback_missing(
    node_ids: list[str], edge_keys: list[tuple[str, str]], nodes: list[dict], edges: list[dict]
) -> dict[str, list]:
    """The rows a write sent that its readback (``nodes`` / ``edges``, read from the group after the write) does not
    hold: ``missing_nodes`` (uuids) and ``missing_edges`` (``[type, uuid]``). Compared as uuid SETS only: a read row
    carries what the MERGE kept and the writer added (stored fields, embeddings), never the projection's props."""
    held_nodes = {n["props"].get("uuid") for n in nodes}
    held_edges = {(e["type"], e["props"].get("uuid")) for e in edges}
    return {
        "missing_nodes": sorted(set(node_ids) - held_nodes),
        "missing_edges": [list(k) for k in sorted(set(edge_keys) - held_edges)],
    }


async def _forward_write(
    ctx: Ctx, driver, data: Mapping, rows: Mapping, embedder, *, extra_own: Iterable[str] = ()
) -> tuple[dict, dict, tuple[list[dict], list[dict]]]:
    """A forward append's write: rule 3 on the group's totals before the write (``pending_check``: an uncommitted
    other session is refused before anything is written), rules 1-2 of ``forward_gate`` on the group's stored rows,
    the MERGE, then rule 3 on the group's totals after it (``label_counts``). The readback is the written rows only
    (``read_rows``): nothing reads the whole group. Every row the write sent must be in it (``readback_missing``), or
    the gate fails. ``extra_own``: more uuids of this run's own (graph mode: its type rows, ``with_type_rows``). Returns
    ``(written, mode, (nodes, edges))``."""
    from memory_kernel.graph.g1_ids import content_uuid
    from memory_kernel.graph.kg_writer import read_rows, write_rows

    cfg = ctx.cfg
    totals = (load_forward(cfg) or {}).get("totals")
    if not totals:
        raise Refused(f"{cfg.forward_input}: the forward input carries no chain totals")
    fwd = rows["forward"]
    continuation = session_continuation(ctx)
    opened = list(continuation["opened"])
    ids = list(dict.fromkeys(r["message_id"] for r in data["records"]))
    own = own_uuids(fwd, data["records"], opened, cfg.group) | set(extra_own)
    episodic = [content_uuid("ep", cfg.group, m) for m in ids]
    record = forward_record(fwd, run=data["run"], episodic=episodic, own=own)
    node_ids = [n["props"]["uuid"] for n in rows["nodes"]]
    edge_keys = [(e["type"], e["props"]["uuid"]) for e in rows["edges"]]
    stored_nodes, stored_edges = await read_rows(driver, cfg.group, node_ids, edge_keys)
    pinned = load_forward(cfg) or {}
    before = pending_check(
        await graphdb.label_counts(driver, cfg.group),
        stored_nodes,
        totals,
        episodic=episodic,
        sagas=[content_uuid("saga", cfg.group, e) for e in opened],
        chain_run=pinned.get("chain_run"),
    )
    kept = keep_contract_aggregates(rows, {n["props"].get("uuid") for n in stored_nodes})
    chat = str(pinned["chat"])
    fold = json.loads(json.dumps(record))  # the JSON the commit reads back
    forward_gate(rows, stored_nodes, stored_edges, data["chain"], own, chat=chat, fold=fold)
    written = await write_rows(driver, rows, group_id=cfg.group, embed=embedder, append=True)
    readback = await read_rows(driver, cfg.group, node_ids, edge_keys)
    got = await graphdb.label_counts(driver, cfg.group)
    expected = {
        "Episodic": int(totals["messages"]) + ctx.counts.messages,
        "Saga": int(totals["episodes"]) + len(opened),
    }
    counts = {k: {"got": got[k], "expected": expected[k], "passed": got[k] == expected[k]} for k in expected}
    missing = readback_missing(node_ids, edge_keys, *readback)
    held = {**missing, "passed": not missing["missing_nodes"] and not missing["missing_edges"]}
    gate = {
        "rules": "forward",
        "stored_before": {"nodes": len(stored_nodes), "edges": len(stored_edges)},
        "totals_before": before,
        "contract_aggregates_kept": kept,
        "created": {"nodes": int(written["nodes_created"]), "edges": int(written["edges_created"])},
        "episodic": counts["Episodic"],
        "saga": counts["Saga"],
        "readback": held,
        "passed": all(c["passed"] for c in counts.values()) and held["passed"],
    }
    # a session earlier than its chat's last stored message is appended in arrival order; the run reports it
    block = {**gate, "inversion": continuation.get("inversion"), "forward": record}
    return written, {"write_mode": "append", "append": block}, readback


def with_type_rows(ctx: Ctx, data: Mapping, rows: Mapping) -> tuple[dict, set[str]]:
    """Graph mode: ``(rows plus this run's type rows, their uuids)``. The type rows are a ``Roleset`` row of each of
    the events stage's ``new_rolesets`` (asked this run, so new to the bonfire), and a ``Route`` row and ``ROUTES_TO``
    edge of each forward route whose key the chain does not route yet (the first route for a key wins, so a routed key
    never gets a second ``ROUTES_TO``). A ``ROUTES_TO`` target the rows do not carry (a stored entity) joins
    ``external_endpoints``. The uuids join ``own_uuids``, so forward-gate rule 1 accepts the rows. ``rows`` is not
    changed: the projection's own hashes stay over the projection."""
    from memory_kernel.graph.state import roleset_rows, route_rows

    group, run = ctx.cfg.group, data["run"]
    events = json.loads((ctx.path("events") / "eventuality.json").read_bytes())
    rolesets = roleset_rows(group, events.get("new_rolesets") or {}, run=run)
    routes = {str(k): str(u) for k, u in rows["forward"]["routes"].items() if k not in data["chain"].routes}
    route_nodes, route_edges = route_rows(group, routes, run=run)
    held = {n["props"]["uuid"] for n in rows["nodes"]}
    external = dict(rows.get("external_endpoints") or {})
    for target in routes.values():
        if target not in held:
            external.setdefault(target, {"labels_neo4j": ["Entity"], "props": {"uuid": target}})
    sent = {
        **rows,
        "nodes": [*rows["nodes"], *rolesets, *route_nodes],
        "edges": [*rows["edges"], *route_edges],
        "external_endpoints": external,
    }
    return sent, {n["props"]["uuid"] for n in (*rolesets, *route_nodes)}


async def _dry_forward_write(
    ctx: Ctx, driver, data: Mapping, rows: Mapping, embedder: embed.RecordingEmbed
) -> tuple[dict, dict, tuple[list[dict], list[dict]], dict]:
    """Graph mode's write: ``_forward_write`` of the rows and this run's type rows (``with_type_rows``) in ONE
    transaction that is rolled back, so the group stays as committed. Then ``COMMIT_ROWS`` (the rows exactly as sent:
    ``keep_contract_aggregates`` trimmed them in place; ``write_rows`` embeds into its own copies, so they carry no
    writer embedding) and ``WRITE_VECTORS`` (``embedder``'s record). Returns ``_forward_write``'s result and the sent
    rows."""
    from memory_kernel.graph.tx import transaction

    sent, types = with_type_rows(ctx, data, rows)
    async with transaction(driver, commit=False) as scope:
        written, mode, readback = await _forward_write(ctx, scope, data, sent, embedder, extra_own=types)
    out = ctx.path("write")
    write_once(out / COMMIT_ROWS, {k: sent[k] for k in ("nodes", "edges", "external_endpoints")})
    embedder.save(out / WRITE_VECTORS)
    return written, mode, readback, sent


def commit_rows(ctx: Ctx) -> tuple[dict, dict[str, list[float]]]:
    """Graph mode: the dry write's ``(COMMIT_ROWS, WRITE_VECTORS)``; ``Refused`` when either is missing."""
    out = ctx.path("write")
    try:
        return json.loads((out / COMMIT_ROWS).read_bytes()), json.loads((out / WRITE_VECTORS).read_bytes())
    except (OSError, ValueError) as exc:
        raise Refused(f"{out}: the dry write's rows or vectors do not read ({type(exc).__name__})") from None


async def _dry_rewrite(ctx: Ctx) -> dict:
    """Graph mode's rewrite: the dry write's rows replayed twice (``write_rows`` with its recorded vectors,
    ``ReplayEmbed``) in ONE rolled-back transaction. ``write`` is the second pass, which must create and embed
    nothing (``run_write``'s idempotent gate); ``first`` is the first's, which creates what the dry write did."""
    from memory_kernel.graph.kg_contract import validate_rows
    from memory_kernel.graph.kg_writer import read_rows, write_rows
    from memory_kernel.graph.tx import transaction

    cfg = ctx.cfg
    rows, vectors = commit_rows(ctx)
    replay = embed.ReplayEmbed(vectors)
    node_ids = [n["props"]["uuid"] for n in rows["nodes"]]
    edge_keys = [(e["type"], e["props"]["uuid"]) for e in rows["edges"]]
    driver = graphdb.driver(cfg)
    try:
        async with transaction(driver, commit=False) as scope:
            first = await write_rows(scope, rows, group_id=cfg.group, embed=replay, append=True)
            again = await write_rows(scope, rows, group_id=cfg.group, embed=replay, append=True)
            nodes, edges = await read_rows(scope, cfg.group, node_ids, edge_keys)
    finally:
        await driver.close()
    errors = validate_rows(nodes, edges, previous=rows["external_endpoints"])
    return {
        "sample_id": cfg.sample_id,
        "group_id": cfg.group,
        "rewrite": True,
        "dry": True,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "first": first,
        "write": again,
        "embedding_tokens": replay.tokens,
        "embedding_requests": replay.requests,
        "embedder": embed.record(replay),
        "readback": readback_missing(node_ids, edge_keys, nodes, edges),
        "validation_errors": len(errors),
        "validation_error_sample": errors[:5],
    }


async def _write(ctx: Ctx, *, rewrite: bool, cost: provenance.CostRecorder) -> dict:
    """Sibling of the harness build module's ``_write`` with the C2 projection; group = ``cfg.group``; no reset
    path at all. Graph mode (``graph_mode``) also ensures the type-node constraints, and writes in a rolled-back
    transaction (``_dry_forward_write``) with a ``RecordingEmbed``; its result says ``"dry": true``."""
    from memory_kernel.graph.kg_contract import validate_rows
    from memory_kernel.graph.kg_writer import ensure_constraints, ensure_indexes, read_group, write_rows

    cfg = ctx.cfg
    group = cfg.group
    append = cfg.write_mode == "append"
    graph = graph_mode(cfg)
    driver = graphdb.driver(cfg)
    embedder: Any = None
    try:
        # The database-wide snapshot (every group's counts) is the strict mode only: on a shared Neo4j other tenants
        # move it, so by default a write checks its own group alone (the writer's foreign-uuid refusal, the readback).
        strict = cfg.strict_db_snapshot
        snapshots: dict[str, Any] = {"snapshot_before": await graphdb.snapshot(driver)} if strict else {}
        counts = await graphdb.group_counts(driver, group)
        _check_counts(group, counts, rewrite=rewrite, append=append)  # again, under the same driver as the write
        # an append's owner was checked by ``_preflight`` (one saga-pass read per run)
        if append and load_forward(cfg) is None:
            raise Refused("an append write needs its forward input (kernel 1.2: mk append runs one session)")
        data, kw, rows = project(ctx, centroids=lambda loaded: saga_centroids(ctx, loaded, cost))
        stamp_run(ctx, data, rows, cost, append=append)
        inv = invariants(rows, cfg.invariants_reference)
        errors = validate_rows(rows["nodes"], rows["edges"], previous=rows["external_endpoints"])
        if errors:
            raise Refused(f"projection violates the contract: {errors[:5]}")
        cap = cap_check(ctx, index_projection_usd(ctx))
        embedder = embed.CappedEmbed(embed.embedder(cfg, "document"), cfg=cfg)
        indexes = await ensure_indexes(driver)
        graph_only: dict[str, Any] = {}
        if graph:  # DDL never runs inside a transaction: before the dry write's
            embedder = embed.RecordingEmbed(embedder)
            graph_only = {"dry": True, "constraints": await ensure_constraints(driver)}
        if strict:
            snapshots["snapshot_after_indexes"] = await graphdb.snapshot(driver)
        mode: dict[str, Any] = {}
        if graph:
            written, mode, (nodes, edges), sent = await _dry_forward_write(ctx, driver, data, rows, embedder)
            readback_errors = validate_rows(nodes, edges, previous=sent["external_endpoints"])
            group_wide: dict[str, Any] = {"readback": "written rows, rolled back"}
        else:
            written = await write_rows(driver, rows, group_id=group, embed=embedder)
            nodes, edges = await read_group(driver, group)
            readback_errors = validate_rows(nodes, edges)
            group_wide = {
                "nodes_outside_group_via_edges": await graphdb.foreign_edges(driver, group),
                "cypher_dups": await graphdb.cypher_dups(driver, group),
            }
        if strict:
            after = snapshots["snapshot_after"] = await graphdb.snapshot(driver)
            snapshots["other_groups"] = graphdb.other_groups_unchanged(snapshots["snapshot_before"], after, group)
        report = base.graph_report(nodes, edges)
        texts = {r["text"].strip() for r in data["records"]}
        report["event_names"]["equal_to_message_text"] = sum(
            n["props"]["name"].strip() in texts for n in nodes if "Event" in n["labels_neo4j"]
        )
        inputs = projection_inputs(ctx)
        return {
            "sample_id": cfg.sample_id,
            "group_id": group,
            "rewrite": rewrite,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "created_at": base.CREATED_AT,
            "render_mode": "b",
            "ingest_pass": kw["ingest_pass"],
            "invariants": inv,
            "programme_cap": cap,
            "embedding_model": embedder.model,
            "group_counts_before": counts,
            "strict_db_snapshot": strict,
            **snapshots,
            "group_rows": {"nodes": len(nodes), "edges": len(edges)},
            **group_wide,
            **base.shape(nodes, edges),
            "envelope": base.envelope_census(nodes, edges),
            "episodic_check": episodic_checks(nodes),
            "write": written,
            "embedding_tokens": embedder.tokens,
            "embedding_requests": embedder.requests,
            "embedder": embed.record(embedder),
            **report,
            "readback_row_set_sha256": base.row_set_hash(nodes, edges),
            "projection_row_set_sha256": base.row_set_hash(rows["nodes"], rows["edges"]),
            "projection_frozen_sha": frozen_sha(rows, extra_drop=saga_vectors.CENTROID_FIELDS),
            "readback_content_sha256": base.content_hash(nodes, edges),
            "indexes": indexes,
            "validation_errors": len(readback_errors),
            "validation_error_sample": readback_errors[:5],
            "inputs": {n: {"path": p, "sha256": base.sha256_file(cfg.root / p)} for n, p in inputs.items()},
            **mode,
            **graph_only,
        }
    finally:
        await driver.close()
        if embedder is not None:  # the spend is recorded even when the write fails after embedding
            tokens = embedder.tokens
            cost.add("voyage", requests=embedder.requests, input_tokens=tokens, usd=embed.voyage_usd(tokens))
            cost.note += f"; voyage usd = tokens x ${embed.VOYAGE_USD_PER_M}/M (ASSUMED list rate)"


def run_write(ctx: Ctx) -> dict:
    """The harness's build write. First call: write the empty group, or in append mode merge into the bonfire's
    group (``summary.json``). Second call: the rewrite (``rerun.json``; graph mode: ``_dry_rewrite``). Third: refused
    (write-once)."""
    cfg = ctx.cfg
    out = ctx.path("write")
    rewrite = (out / SUMMARY).is_file()
    target = _out(ctx, "write", RERUN if rewrite else SUMMARY)
    asyncio.run(_preflight(ctx, rewrite=rewrite))
    note = f"C2 {'rewrite' if rewrite else 'write'} {cfg.sample_id} to {cfg.group}"
    with ctx.cost("build", run=out.name, note=note) as cost:
        if rewrite and graph_mode(cfg):  # the first write rolled back: replay its rows, never re-project
            result = asyncio.run(_dry_rewrite(ctx))
        else:
            result = asyncio.run(_write(ctx, rewrite=rewrite, cost=cost))
    w = result["write"]
    if rewrite:
        created = {k: w[k] for k in ("nodes_created", "edges_created", "node_texts_embedded", "edge_texts_embedded")}
        result["idempotent"] = {
            **created,
            "embedding_tokens": result["embedding_tokens"],
            "passed": not any((*created.values(), result["embedding_tokens"])),
        }
    write_once(target, result)  # the evidence persists before any failure below
    problems = write_problems(result, prefix=cfg.concurrent_group_prefix, strict=cfg.strict_db_snapshot)
    if rewrite and not result["idempotent"]["passed"]:
        problems.append(f"rewrite not idempotent: {result['idempotent']}")
    if problems:
        raise RuntimeError(f"{target}: " + "; ".join(problems))
    return result


#: The ``graphdb.snapshot`` key of the nodes and edges that carry no ``group_id`` (``str(None)``).
UNGROUPED = "None"


def foreign_changes(other_groups: Mapping, *, prefix: str | None) -> dict:
    """The other-group changes that fail a strict write (``BuildConfig.strict_db_snapshot``). With ``prefix=None``
    ANY change of another group counts, growth included, so the write fails closed. ``prefix``
    (``--concurrent-group-prefix``) tolerates exactly one thing: GROWTH (``after >= before``) of the other groups whose
    name starts with it, a concurrent build's own first write. Never tolerated, whatever the prefix: a shrink, and the
    ungrouped bucket ``UNGROUPED`` (nodes and edges without a ``group_id``; a group literally named ``None`` cannot be
    told apart from it, so it is that bucket too). Keys are ``nodes:<group>`` / ``edges:<group>``
    (``graphdb.other_groups_unchanged``)."""

    def tolerated(key: str, change: Mapping) -> bool:
        group = key.split(":", 1)[1]
        return (
            prefix is not None
            and group != UNGROUPED
            and group.startswith(prefix)
            and change["after"] >= change["before"]
        )

    return {key: v for key, v in other_groups["changed"].items() if not tolerated(key, v)}


def write_problems(result: Mapping, *, prefix: str | None, strict: bool) -> list[str]:
    """What fails a write result. ``prefix`` = ``BuildConfig.concurrent_group_prefix`` and ``strict`` =
    ``BuildConfig.strict_db_snapshot``: required (no defaults), so the live write, the adoption of a landed first write
    and the rewrite re-check all apply the same rule. Other groups count only under ``strict``, and then a result
    written without the database snapshot fails too."""
    problems = []
    if strict:
        other = result.get("other_groups")
        if other is None:
            problems.append("strict_db_snapshot: the write took no database snapshot")
        elif foreign := foreign_changes(other, prefix=prefix):
            problems.append(f"other groups changed: {foreign}")
    if result["validation_errors"]:
        problems.append(f"readback validation errors: {result['validation_error_sample']}")
    gate = result.get("append")  # a forward append's write (rule 3, the readback); a fresh result has none
    if gate is not None and not gate["passed"]:
        shown = {k: gate.get(k) for k in ("created", "episodic", "saga", "readback")}  # an older gate has no readback
        failed = [
            *(["forward gate rule 3"] if not all(gate[k]["passed"] for k in ("episodic", "saga")) else []),
            *(["the readback misses written rows"] if (gate.get("readback") or {}).get("passed") is False else []),
        ]
        problems.append(f"append gate failed ({'; '.join(failed) or 'forward gate'}): {shown}")
    return problems


def adopt_first_write(ctx: Ctx) -> dict | None:
    """The first write's ``summary.json`` when it landed but its stage was not marked done (the write raised on a
    gate this module has since corrected): re-check it with today's gate and return it. None = no first write yet.
    A write that still fails the gate raises, as the write itself would."""
    target = ctx.path("write") / SUMMARY
    if not target.is_file():
        return None
    result = json.loads(target.read_bytes())
    problems = write_problems(result, prefix=ctx.cfg.concurrent_group_prefix, strict=ctx.cfg.strict_db_snapshot)
    if problems:
        raise RuntimeError(f"{target} (adopted): " + "; ".join(problems))
    return {**result, "adopted": str(target)}


def write(ctx: Ctx, live: bool) -> dict:  # noqa: ARG001 (single live call, as the harness)
    """The ``write`` stage = the harness's ``call_stage`` write branch: adopt a landed first write, else write."""
    adopted = adopt_first_write(ctx)
    if adopted is not None:
        return adopted  # the first write landed but its stage was not marked done: never a second first write
    return run_write(ctx)


def rewrite(ctx: Ctx, live: bool) -> dict:  # noqa: ARG001
    """The harness's rewrite: ``run_write`` a second time (it detects ``summary.json``); gate = 0 created, 0 embedded.

    Refused without the first write's ``summary.json``: the call would otherwise be a FIRST write."""
    from memory_kernel.pipeline.runner import Stop

    if not (ctx.path("write") / SUMMARY).is_file():
        raise Stop(f"rewrite: {ctx.path('write') / SUMMARY} missing (the first write has not run)")
    rerun = ctx.path("write") / RERUN
    if rerun.is_file():  # the rewrite landed but its stage was not marked done: re-check it with today's gate
        out = json.loads(rerun.read_bytes())
        problems = write_problems(out, prefix=ctx.cfg.concurrent_group_prefix, strict=ctx.cfg.strict_db_snapshot)
        if problems:
            raise RuntimeError(f"{rerun} (adopted): " + "; ".join(problems))
        out["adopted"] = str(rerun)
    else:
        out = run_write(ctx)
    idem = out.get("idempotent") or {}
    created = {k: v for k, v in idem.items() if k != "passed"}
    passed = bool(out.get("rewrite")) and bool(idem.get("passed")) and bool(created) and not any(created.values())
    gate = {"name": "write-idempotent", "pass": passed, "values": created}
    return {**out, "gate": gate}
