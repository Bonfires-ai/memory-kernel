"""events and ctx: eventuality per roleset, then per predicate in context.

Ported from the reference harness:
  * ``events`` / ``ctx`` / ``adopt_events`` / ``EVENTS_FILES`` / ``CTX_SPEND_CAP_USD``  <- the harness's stage module
  * ``run_events`` / ``run_ctx`` / ``gate_fallback`` / ``events_inputs`` / ``ctx_inputs`` / ``rel``  <- the harness's event module

The frozen bodies are ``_events_core`` / ``_eventuality_ctx``. Rebindings replaced:
  * ``rebind(ev, INPUTS=, EXECUTION=, subprocess=GitShim)`` -> the ``inputs`` / ``shard`` /
    ``head`` arguments of ``run_events`` (``provenance.code_head()``, the imported package's checkout: no ``GitShim``);
  * ``rebind(ctx, SPEND_CAP_USD=0.35)`` -> ``spend_cap_usd`` (the plan's; the live run passes ``UNCAPPED``: the
    stage cap is recorded, never a stop);
  * ``armed("events")`` / ``armed("ctx")`` -> ``ctx.prefetch.jev`` called by the frozen loop;
  * ``_run_main`` (argv swap + stdout capture) -> ``run_events`` / ``run_ctx`` return their result;
  * Voyage (segment embeddings on a cache miss) -> ``cfg.voyage_api_key``, no ``docker exec``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from memory_kernel.pipeline import frames, llm, provenance
from memory_kernel.pipeline.config import load_forward
from memory_kernel.pipeline.stages import _events_core as ev
from memory_kernel.pipeline.stages import _eventuality_ctx as cx
from memory_kernel.pipeline.stages._common import gate, result

if TYPE_CHECKING:
    from memory_kernel.pipeline.runner import Ctx

#: voyage-3 list price per million tokens. The repo carries no price constant; recorded as an ASSUMED rate.
VOYAGE_USD_PER_M = 0.06
GATE_FALLBACK_MAX = 0.40  # development conversation: 748/2946 = 25 percent
CTX_SPEND_CAP_USD = 0.35  # The frozen ctx SPEND_CAP_USD (0.20) is rebound to this cap

#: The frozen events runner's five sealed outputs, in its write order (``summary.json`` last).
EVENTS_FILES = ("segments.json", "eventuality.json", "occurrences.json", "thread-support-group.json", "summary.json")


def gate_fallback(counts: Mapping, maximum: float = GATE_FALLBACK_MAX) -> dict:
    """Share of routed predicates that fell back to the roleset prior (pure)."""
    routed = int(counts["routed"])
    fell = int(counts["fallback_prior"])
    share = fell / routed if routed else 1.0
    return {
        "routed": routed,
        "fallback_prior": fell,
        "excluded_nominal_modifier": int(counts.get("excluded_nominal_modifier", 0)),
        "share": round(share, 4),
        "max": maximum,
        "passed": routed > 0 and share <= maximum,
    }


def adopt_events(out: Path) -> dict | None:
    """A COMPLETE earlier live write of ``events`` (all five sealed files) is adopted as this attempt's result.

    The frozen write is write-once with an identical-rewrite no-op, but a rerun is never identical: ``segments.json``
    records ``voyage_tokens_this_run`` (0 once the embedding cache is warm) and ``summary.json`` the kernel head. A
    run whose child finished after its driver died (full-run restart 10-09, another conversation) would otherwise be refused
    for ever. A partial dir (no ``summary.json``) is not adopted: the rerun refuses it as before."""
    if not all((out / name).is_file() for name in EVENTS_FILES):
        return None
    summary = json.loads((out / "summary.json").read_bytes())
    return {"adopted": str(out), "summary": summary}


def _rel(ctx: Ctx, path: Path) -> str:
    """INPUTS entries are root-relative strings (a path outside the root stays as given)."""
    return provenance.rel(path, ctx.cfg.root)


def events_inputs(ctx: Ctx) -> dict[str, str]:
    return {
        "graph": _rel(ctx, ctx.path("graph_a") / "graph.json"),
        "inventory": _rel(ctx, ctx.path("graph_a") / "empty-inventory.json"),
        "archived_boundaries": _rel(ctx, ctx.path("boundary") / "applied-boundaries.json"),
    }


def ctx_inputs(ctx: Ctx) -> dict[str, str]:
    return {
        "graph": _rel(ctx, ctx.path("graph_a") / "graph.json"),
        "inventory": _rel(ctx, ctx.path("graph_a") / "empty-inventory.json"),
        "eventuality_m3b": _rel(ctx, ctx.path("events") / "eventuality.json"),
    }


def _run_events(ctx: Ctx, shard, out: Path, *, live: bool, plan: bool, prefetch=None) -> dict:
    from memory_kernel.pipeline.runner import events_voyage_cache

    cfg = ctx.cfg
    forward = load_forward(cfg)  # a forward append: the chain's frozen roleset wires
    return ev.run_events(
        shard=shard,
        root=cfg.root,
        frame_label=lambda directory: frames.frame_label(directory, vault=cfg.vault, data_dir=cfg.data_dir),
        frame_dirs=cfg.frame_dirs,
        inputs=events_inputs(ctx),
        out=out,
        voyage_cache=events_voyage_cache(cfg),
        head=provenance.code_head(),
        live=live,
        plan=plan,
        batcher=ctx.batcher,
        prefetch=prefetch,
        voyage_api_key=cfg.voyage_api_key,
        frozen_rolesets=forward["rolesets"] if forward else None,
    )


def events(ctx: Ctx, live: bool) -> dict:
    out = ctx.path("events")
    ctx.freeze_check()
    plan = _run_events(ctx, ctx.shard(), out, live=False, plan=True)
    ctx.save_plan(f"eventuality-events-{out.name}", plan)
    if not live:
        return result("events", plan)
    adopted = adopt_events(out)
    if adopted is not None:
        return result("events", plan, adopted)
    ctx.freeze_check()
    from memory_kernel.pipeline.runner import events_voyage_cache

    events_voyage_cache(ctx.cfg).parent.mkdir(parents=True, exist_ok=True)
    shard = ctx.shard()
    with llm.metered(
        ctx,
        "eventuality",
        out.name,
        shard,
        note="roleset eventuality",
        stage_cap_override=llm.stage_cap("events", ctx.messages),
    ) as meter:
        _run_events(
            ctx,
            shard.with_hooks(reserve_guard=llm.guard(meter)),
            out,
            live=True,
            plan=False,
            prefetch=lambda ledger, wires: ctx.prefetch.jev(ledger, wires, stage="events", meter=meter),
        )
        summary = json.loads((out / "summary.json").read_bytes())
        tokens = int(summary["spend"]["voyage_tokens_this_run"])
        if tokens and meter.cost is not None:
            meter.cost.add("voyage", requests=0, input_tokens=tokens, usd=tokens * VOYAGE_USD_PER_M / 1e6)
            meter.cost.note += f"; voyage usd = tokens x ${VOYAGE_USD_PER_M}/M (ASSUMED list rate)"
    doc = json.loads((out / "eventuality.json").read_bytes())
    return result(
        "events",
        plan,
        {
            "occurrences": summary["occurrences"],
            "segments": summary["segments"],
            "distribution_rolesets": doc["distribution_rolesets"],
            "distribution_predicates": doc["distribution_predicates"],
            "predicates": doc["predicates"],
            "thread_messages": len(summary["thread_messages"]),
            "voyage_tokens": tokens,
            "stage_spend_usd": meter.spent(),
            **meter.cap_record(),
            "eventuality_file_sha256": provenance.sha_file(out / "eventuality.json"),
        },
    )


#: The spend authorisation the live ctx stage hands the frozen ``run_ctx``: none. Its two stage-cap stops (the plan
#: estimate over the cap, a running receipt sum at the cap) never fire; the stage records them instead (``ctx``) and
#: ``--cap-usd`` (the stage meter's unit cap) is the spend stop. The sealed ``eventuality.json`` does not record the
#: cap, so a run within it writes the same bytes.
UNCAPPED = Decimal("Infinity")


def _run_ctx(ctx: Ctx, shard, out: Path, *, live: bool, plan: bool, prefetch=None, spend_cap_usd=None) -> dict:
    cfg = ctx.cfg
    return cx.run_ctx(
        shard=shard,
        root=cfg.root,
        inputs=ctx_inputs(ctx),
        out=out,
        live=live,
        plan_only=plan,
        batcher=ctx.batcher,
        spend_cap_usd=Decimal(str(llm.stage_cap("ctx", ctx.messages))) if spend_cap_usd is None else spend_cap_usd,
        code_key=provenance.code_key(cx),
        prefetch=prefetch,
    )


def ctx(ctx: Ctx, live: bool) -> dict:
    """``run_ctx`` rebinds the frozen ``SPEND_CAP_USD`` to ``STAGE_CAPS['ctx']``; pinned here at 0.35."""
    if Decimal(str(llm.STAGE_CAPS["ctx"])) != Decimal(str(CTX_SPEND_CAP_USD)):
        raise ValueError(f"STAGE_CAPS['ctx'] {llm.STAGE_CAPS['ctx']} != {CTX_SPEND_CAP_USD}")
    out = ctx.path("ctx")
    ctx.freeze_check()
    plan = _run_ctx(ctx, ctx.shard(), out, live=False, plan=True)
    ctx.save_plan(f"eventuality-ctx-{out.name}", plan)
    if not live:
        return result("ctx", plan)
    ctx.freeze_check()
    shard = ctx.shard()
    cap = Decimal(str(llm.stage_cap("ctx", ctx.messages)))
    with llm.metered(
        ctx,
        "eventuality",
        out.name,
        shard,
        note="per-predicate context eventuality",
        stage_cap_override=llm.stage_cap("ctx", ctx.messages),
    ) as meter:
        _run_ctx(
            ctx,
            shard.with_hooks(reserve_guard=llm.guard(meter)),
            out,
            live=True,
            plan=False,
            prefetch=lambda ledger, wires: ctx.prefetch.jev(ledger, wires, stage="ctx", meter=meter, gate_cap=cap),
            spend_cap_usd=UNCAPPED,
        )
    doc = json.loads((out / "eventuality.json").read_bytes())
    estimate = Decimal(plan["estimate_usd_m3b_receipt_rate"])
    over = meter.cap_record() or (
        llm.stage_cap_record(ctx.cfg, "ctx", cap, estimate, basis="plan estimate") if estimate > cap else {}
    )
    ran = {
        "counts": doc["counts"],
        "gate": gate_fallback(doc["counts"]),
        "distribution_predicates": doc["distribution_predicates"],
        "spend": doc["spend"],
        "stage_spend_usd": meter.spent(),
        **over,
        "eventuality_file_sha256": provenance.sha_file(out / "eventuality.json"),
    }
    return result("ctx", plan, ran, ctx_gate(ran["gate"], forward=load_forward(ctx.cfg) is not None))


def ctx_gate(verdict: Mapping, *, forward: bool) -> dict:
    """The ``ctx-prior-fallback`` gate: advisory, a recorded metric that stops the build only under
    ``--strict-gates``. A full build passes as ``gate_fallback`` decides; a forward append (one session, often a
    handful of predicates, or none) also passes when no predicate was routed (there is nothing to fall back).

    It was hard in a full build until 2026-10-15 (operator): a document corpus (technical prose, mostly states and
    general statements) missed it at 0.41 against 0.40. The share says how much of the context step fell back to the
    roleset prior -- a fit diagnostic, not a broken build."""
    if not forward:
        return {**gate("ctx-prior-fallback", verdict["passed"], verdict), "advisory": True}
    ok = verdict["passed"] or int(verdict["routed"]) == 0
    return {**gate("ctx-prior-fallback", ok, verdict), "advisory": True}
