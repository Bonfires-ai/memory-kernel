"""select and boundary: the question-blind ingest stages.

Ported from the reference harness:
  * ``select`` / ``build``                    <- the harness's select module
  * ``boundary`` / ``load_sources`` / ``boundary_replay_gate``
                                              <- the harness's stage module

The frozen bodies are in ``_conversation`` (conversation split, prepared batches, sources) and ``_boundary_walk`` +
``_boundary_frozen`` (the UTC-day boundary walk and the frozen boundary compiler). Kernel 1.8 archived the taxonomy
stage. Rebindings replaced:
  * the harness's conversation binding (conversation, sources and output paths / a conversation episode prefix /
    the cost ledger) -> explicit ``ctx.path(...)`` / ``cfg.episode_prefix`` / ``cfg.ledger`` arguments;
  * ``_load_key()`` and the ``os.environ`` key loaders -> ``cfg.openrouter_api_key`` (``Shard.api_key``);
  * the pilot-tree imports -> copies (``_boundary_frozen``, ``_conversation``).
``select`` reads the conversation JSON of the config (a sessions-JSON sample ``{sample_id, conversation, qa?}``) instead of
the benchmark JSON + its sha pin; ``selection.json`` records that file's sha as ``dataset_sha256``.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from memory_kernel.construction.memory_semantic_jev import sha
from memory_kernel.pipeline import llm
from memory_kernel.pipeline.config import grammar_image_sha, load_conversation, load_forward, load_replay_scope
from memory_kernel.pipeline.ledger import MK_CODE, make_execution, round_cap
from memory_kernel.pipeline.stages import _boundary_walk as ub
from memory_kernel.pipeline.stages._common import cap_check, gate, result
from memory_kernel.pipeline.stages._conversation import (
    build_prepared,
    build_sources,
    conversations_of,
    split_conversation,
)
from memory_kernel.pipeline.stages._g1_cards import write_once

if TYPE_CHECKING:
    from memory_kernel.pipeline.runner import Ctx


# -- select ---------------------------------------------------------------------------------------------------------


def build(item: dict, scope: dict, first_order: int = 0) -> tuple[dict, dict, list, list]:
    """(conv_only, eval_qa, prepared, sources) for ``item`` under ``scope``; ``first_order`` is the first message's
    order (a forward append's session continues the bonfire's). Writes nothing."""
    conv_only, eval_qa = split_conversation(item)
    prepared = build_prepared(conv_only)
    return conv_only, eval_qa, prepared, build_sources(prepared, scope, first_order)


def select(ctx: Ctx, live: bool = True) -> dict:  # noqa: ARG001 -- select has no plan mode
    """Write-once stage output for the conversation; a second call into the same dir raises. A forward append's run
    (``cfg.forward_input``) selects its one session, its orders continuing the bonfire's."""
    cfg = ctx.cfg
    out = ctx.path("select")
    out.mkdir(parents=True, exist_ok=False)
    stem = cfg.stem
    scope = {"bonfire_id": cfg.bonfire_id, "conversation_id": cfg.sample_id}
    item, raw = load_conversation(cfg.conversation, cfg.sample_id)  # a sample object, or one item of a list of samples
    forward = load_forward(cfg)
    conv_only, eval_qa, prepared, sources = build(item, scope, forward["first_order"] if forward else 0)
    if forward and {s["session_id"] for s in sources} != {forward["session_id"]}:
        raise ValueError(f"{cfg.conversation}: the forward append's session is {forward['session_id']}")
    flat = [r for b in prepared for r in b]
    write_once(out / f"{stem}-conversation-only.json", conv_only)
    write_once(out / f"{stem}-eval-qa.json", eval_qa)
    write_once(out / "prepared_batches.json", prepared)
    write_once(out / "sources.json", sources)
    selection = {
        "schema_version": "mk-selection/v1",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "rule": "explicit sample id",
        "selected": cfg.sample_id,
        "eval_qa_sha256": sha((out / f"{stem}-eval-qa.json").read_bytes()),
        "eval_qa_questions": len(eval_qa["qa"]),
        "counts": {
            "sessions": len(prepared),
            "messages": len(sources),
            "captions": sum(len(s["attachments"]) for s in sources),
            "prepared_rows": len(flat),
            "non_adversarial": sum(1 for q in eval_qa["qa"] if int(q["category"]) != 5),
        },
        "scope": scope,
    }
    write_once(out / "selection.json", selection)
    return selection


# -- boundary -------------------------------------------------------------------------------------------------------


def load_sources(ctx: Ctx) -> tuple[Path, list[dict]]:
    """This conversation's select output, checked to be this conversation."""
    sources_path = ctx.path("select") / "sources.json"
    sources = json.loads(sources_path.read_bytes())
    convs = {s["scope"]["conversation_id"] for s in sources}
    if convs != {ctx.cfg.sample_id}:
        raise ValueError(f"{sources_path}: conversation ids {sorted(convs)} != {ctx.cfg.sample_id}")
    return sources_path, sources


def boundary_replay_gate(ctx: Ctx) -> dict:
    """$0 offline equivalence of an archived boundary run: recompile every same-day wire from the pair's sources and
    compare it to the archived decision row (400/400 on the development-conversation pair).

    Opt-in: it runs only when ``--boundary-replay-archive`` (and its ``--boundary-replay-scope`` file, the constants the
    archive was made with) are given; without the pair the gate is recorded as skipped."""
    cfg = ctx.cfg
    pair = cfg.boundary_replay_archive
    if pair is None:
        return gate("boundary-replay", True, {"skipped": "no boundary replay pair"})
    if cfg.boundary_replay_scope is None:
        raise ValueError("boundary replay gate: --boundary-replay-archive needs --boundary-replay-scope")
    archive, sources = pair
    r = ub.replay_equivalence(archive, json.loads(sources.read_bytes()), **load_replay_scope(cfg.boundary_replay_scope))
    return gate("boundary-replay", r["match"] == r["total"] and r["total"] and r["episodes_match"], r)


def ensure_round_shard(ctx: Ctx, sources_path: Path) -> None:
    """The conversation's round-ledger shard (``03-boundary-001-ledger``), made once by the boundary stage under
    either policy: the later Jev stages open it (``ctx.shard()``)."""
    cfg = ctx.cfg
    shard_path = ctx.path("shard")
    if not shard_path.exists():
        make_execution(
            shard_path,
            round_id=f"{cfg.bonfire_id}-boundary",
            capture_sha=hashlib.sha256(sources_path.read_bytes()).hexdigest(),
            g0_sha=grammar_image_sha(cfg),
            code=MK_CODE,
            cap_usd=round_cap(ctx.messages),
        )


def segmenter_boundary(ctx: Ctx, live: bool) -> dict:
    """The boundary stage under ``boundary_policy == "segmenter"``: each session is a hard boundary, cut inside by
    the embedding segmenter (``_boundary_segment``). No Jev call and no reservation: the Voyage tokens are the
    stage's cost row, and ``--cap-usd`` stops it (``llm.StopSpend``). The vectors go through the events stage's cache.

    The conversation's round-ledger shard is still made here (sealed files only, nothing reserved or sent through
    it): the later Jev stages open it (``ctx.shard()``), and the boundary stage is the one that makes it.

    A forward append segments its one session on its own (a session is a hard boundary): its episodes are numbered
    from the bonfire's next episode (the forward input's ``boundary.next_episode``), and the applied file carries the
    chat's closed state after it (``continuation``), what the commit stores. A chat the chain walked is
    refused."""
    from memory_kernel.pipeline import embed, provenance
    from memory_kernel.pipeline.runner import events_voyage_cache
    from memory_kernel.pipeline.stages import _boundary_segment as us

    cfg = ctx.cfg
    forward = load_forward(cfg)
    stored = forward["boundary"] if forward else None  # an append: the chat's stored state and the next episode
    if stored is not None and int(stored["messages"]) and stored.get("policy", "walk") != us.POLICY:
        raise ValueError(
            f"boundary: chat {stored['chat']!r} was built by the walk policy; a segmenter append cannot continue it"
        )
    if cfg.boundary_replay_archive is not None:
        raise ValueError(
            "boundary: --boundary-replay-archive replays a walk run's Jev choices; the segmenter policy makes none"
        )
    sources_path, sources = load_sources(ctx)
    conversations = conversations_of(ctx)
    plan = us.plan(sources, conversations)
    ctx.save_plan(f"boundary-run-{ctx.path('boundary').name}", plan)
    replay = gate("boundary-replay", True, {"not_applicable": True, "policy": us.POLICY})
    if not live:
        return result("boundary", plan, gate=replay)
    out = ctx.path("boundary")
    ensure_round_shard(ctx, sources_path)
    cache_path = events_voyage_cache(cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = embed.TextVoyageCache(cache_path, cfg.voyage_api_key)
    paid: list[float] = []

    def unit_guard(spend: float) -> None:
        if spend <= 0:
            return
        if not paid:  # the build's paid total at stage start: this stage's row is written when it ends
            paid.append(provenance.ledger_summary(cfg.ledger)["paid_usd"])
        if paid[0] + spend > cfg.cap_usd:
            raise llm.StopSpend(f"unit cap {cfg.cap_usd} crossed by the boundary voyage spend {spend:.4f}")

    with ctx.cost("boundaries", run=out.name, note="segmenter boundary: Voyage embeddings, no Jev") as cost:
        try:
            summary = us.run_segment_boundary(
                sources,
                out,
                cache=cache,
                episode_prefix=cfg.episode_prefix,
                conversations=conversations,
                first_episode=int(stored["next_episode"]) if stored else 0,  # a full build: 0, as the walk's
                unit_guard=unit_guard,
                continuation=stored,
            )
        finally:
            # the live client's count: ``cache.tokens`` is refreshed only after a whole batch of chunks succeeds
            tokens = int(cache.inner.tokens) if cache.inner is not None else int(cache.tokens)
            requests = int(getattr(cache.inner, "requests", 0)) if cache.inner is not None else 0
            cost.add("voyage", requests=requests, input_tokens=tokens, usd=embed.voyage_usd(tokens))
    verdict = gate("boundary", True, {"replay": replay["values"], "stop_rules": summary["stop_rules"]})
    return result("boundary", plan, summary, {**verdict, "advisory": True})


def boundary(ctx: Ctx, live: bool) -> dict:
    if getattr(ctx.cfg, "boundary_policy", "walk") == "segmenter":  # absent = walk (a config made before the field)
        return segmenter_boundary(ctx, live)
    cfg = ctx.cfg
    sources_path, sources = load_sources(ctx)
    cap = ub.stage_cap(ctx.messages)
    conversations = conversations_of(ctx)  # one walk per conversation (chat); one conversation = the 1.0.0 walk
    forward = load_forward(cfg)
    continuation = forward["boundary"] if forward else None  # a forward append: the chat's stored walk goes on
    plan = ub.plan(
        sources,
        episode_prefix=cfg.episode_prefix,
        window_prefix=cfg.boundary_window_prefix,
        cap=cap,
        conversations=conversations,
        continuation=continuation,
    )
    ctx.save_plan(f"boundary-run-{ctx.path('boundary').name}", plan)
    replay = boundary_replay_gate(ctx)
    if not live or not replay["pass"]:
        return result("boundary", plan, gate=replay)
    cap_check(ctx, float(cap))
    out = ctx.path("boundary")
    ensure_round_shard(ctx, sources_path)
    usage: dict = {}
    summary: dict | None = None
    with ctx.cost("boundaries", run=out.name, note="boundary-only Jev choices") as cost:
        try:
            summary = ub.run_boundary(
                sources,
                out,
                shard=ctx.shard(reserve_guard=llm.unit_guard(cfg, usage, float(cap))),
                episode_prefix=cfg.episode_prefix,
                window_prefix=cfg.boundary_window_prefix,
                live=True,
                plan_only=False,
                usage=usage,
                cap=cap,
                max_shift_share=cfg.max_shift_share,
                strict=cfg.strict_gates,
                conversations=conversations,
                continuation=continuation,
            )
        finally:
            cost.add(
                "jev",
                requests=usage.get("requests", 0),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                usd=usage.get("usd", 0.0),
            )
    assert summary is not None  # CostRecorder.__exit__ returns False: an error in run_boundary propagates
    if summary.get("stage_cap_exceeded"):  # recorded in summary.json; the stage cap is never a stop
        spend = max(Decimal(str(usage["usd"])), Decimal(summary["plan"]["reservation_usd"]))
        llm.stage_cap_record(cfg, "boundary", cap, spend, basis="max(plan reservation, spend)")
    stops = summary["stop_rules"]
    values = {"replay": replay["values"], "stop_rules": stops}
    # the stop rules are an advisory gate; in an append they never fail it (an inherited --strict-gates would stop)
    fired = stops["fired"] and continuation is None
    verdict = {**gate("boundary", replay["pass"] and not fired, values), "advisory": True}
    return result("boundary", plan, summary, verdict)
