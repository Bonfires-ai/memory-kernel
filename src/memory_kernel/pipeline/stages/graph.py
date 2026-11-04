"""graph_picks and graph_a: the whole-conversation G1 event graph.

Ported from the reference harness:
  * ``graph_picks`` / ``check_picks`` / ``graph_a`` / ``picks_plan_path``  <- the harness's stage module
  * ``picks`` (the stage meter) / ``graph_build``                        <- the harness's run module
  * ``empty_inventory``                                                  <- the harness's stage-parameter module

Kernel 1.8 archived graph_b (graph_a's build again over the remined inventory):
every reader takes graph_a's ``graph.json``. The frozen graph bodies are in ``_g1_graph`` and the modules it reads.
Rebindings replaced:
  * ``RoundBudget.evaluate = metered`` -> ``Shard.on_new_send`` (the picks meter), on this stage's shard only;
  * ``partition_question_wire`` -> ``ctx.batcher``;
  * the harness capture runner's ``ROOT`` (the base of ``manifest["capture"]["path"]``) -> ``cfg.root``;
  * ``_load_key()`` (reads ``<vault>/.env``) is gone: the key is ``cfg.openrouter_api_key``.
Graph_picks is NOT prefetched (the harness's ``ARMS`` never covered it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from memory_kernel.construction.memory_semantic_jev import seal
from memory_kernel.pipeline import llm
from memory_kernel.pipeline.config import genesis_zero_ok, load_forward
from memory_kernel.pipeline.stages._common import cap_check, result
from memory_kernel.pipeline.stages._g1_cards import GRAMMAR_ID, census, write_once
from memory_kernel.pipeline.stages._g1_graph import build_graph, run_whole_picks

if TYPE_CHECKING:
    from memory_kernel.pipeline.runner import Ctx

PICKS_CAP_USD = 0.05  # the harness run module's inline graph_picks cap (projection for the programme cap check)


def empty_inventory() -> dict:
    """A sealed g1 inventory with no cards (the harness's empty card census)."""
    return seal(
        {
            "schema_version": "mk-g1-cards/v1",
            "grammar_id": GRAMMAR_ID,
            "base_grammar": "memory-kernel-propbank-grammar-1",
            "cards": [],
            "census": census([]),
            "start": "empty (no cards)",
        }
    )


def picks_plan_path(ctx: Ctx) -> Path:
    """Where ``graph_picks`` saved its plan (``Ctx.save_plan`` -> ``<conv root>/logs``)."""
    return ctx.path("logs") / f"graph-picks--{ctx.path('graph_a').name}-plan.json"


def _picks_live(ctx: Ctx, out: Path, runs: Path) -> dict:
    """The harness run module's ``graph_picks``, live: the frozen picks under a stage meter that counts new sends
    only. Its stage cap is recorded, never a stop; past it, a hold that would cross ``--cap-usd`` is refused
    (``llm.unit_guard``)."""
    ctx.freeze_check()
    stage_cap = 0.05  # stage cap for the Jev antecedent picks
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}

    def on_new_send(archive: dict) -> None:
        receipt = json.loads(archive["receipt"])["usage"]
        usage["requests"] += 1
        usage["input_tokens"] += receipt["input_tokens"]
        usage["output_tokens"] += receipt["output_tokens"]
        usage["usd"] += float(receipt["cost_usd"])

    shard = ctx.shard(on_new_send=on_new_send, reserve_guard=llm.unit_guard(ctx.cfg, usage, stage_cap))
    with ctx.cost("picks", run=out.name, note="antecedent picks (shared shard)") as cost:
        try:
            ran = run_whole_picks(shard, [runs], out, root=ctx.cfg.root, batcher=ctx.batcher, live=True)
        finally:
            cost.add("jev", **usage)
    over = llm.stage_cap_record(ctx.cfg, "picks", stage_cap, usage["usd"]) if usage["usd"] > stage_cap else {}
    return {**ran, "usage": usage, **over}


def graph_picks(ctx: Ctx, live: bool) -> dict:
    out, runs = ctx.path("graph_a"), ctx.path("runs")
    ctx.freeze_check()
    plan = run_whole_picks(ctx.shard(), [runs], out, root=ctx.cfg.root, batcher=ctx.batcher, plan=True)
    ctx.save_plan(f"graph-picks--{out.name}", plan)
    if not live:
        return result("graph_picks", plan)
    cap_check(ctx, PICKS_CAP_USD)
    return result("graph_picks", plan, _picks_live(ctx, out, runs))


def _no_picks_needed(ctx: Ctx) -> bool:
    """A forward append whose picks plan counted no message needing a pick (a one-message session with no pronoun,
    say), or a genesis build whose plan counted none (``genesis_zero_ok``: a seed of a few Setup sentences): zero pick
    files is then complete. Any other full build, or a run with no recorded plan, never takes zero."""
    recorded = picks_plan_path(ctx)
    if (load_forward(ctx.cfg) is None and not genesis_zero_ok(ctx.cfg)) or not recorded.is_file():
        return False
    return json.loads(recorded.read_bytes())["messages"] == 0


def check_picks(ctx: Ctx) -> dict:
    """graph-A must not build without the picks: ``build_graph`` writes a write-once graph.json with picks=0 and a
    later build with real picks fails. One pick file per message the picks plan counted; in a
    forward append or a genesis build, none when the plan counted none (``_no_picks_needed``)."""
    picks = ctx.path("graph_a") / "picks"
    files = len([p for p in picks.glob("*.json") if p.is_file()]) if picks.is_dir() else 0
    if not files:
        if _no_picks_needed(ctx):
            return {"pick_files": 0, "plan_messages": 0}
        raise FileNotFoundError(f"{picks}: no picks (run graph_picks live first)")
    recorded = picks_plan_path(ctx)
    expected = json.loads(recorded.read_bytes())["messages"] if recorded.is_file() else None
    if expected is not None and files != expected:
        raise ValueError(f"{picks}: {files} pick files != {expected} messages in the picks plan {recorded}")
    return {"pick_files": files, "plan_messages": expected}


def _graph_build(ctx: Ctx, out: Path, runs: Path) -> dict:
    """The harness run module's ``graph_build``: the whole-conversation graph (frozen ``build_graph``), $0, one
    cost row."""
    ctx.freeze_check()
    inventory = out / "empty-inventory.json"
    out.mkdir(parents=True, exist_ok=True)
    if not inventory.exists():
        write_once(inventory, empty_inventory())
    forward = load_forward(ctx.cfg)  # a forward append: "you" per message, from the chain's addressing start
    with ctx.cost("graph", run=out.name, note="graph build, local, $0"):
        return build_graph(
            [runs],
            inventory,
            out,
            root=ctx.cfg.root,
            append=forward is not None,
            explicit_since=forward["explicit_since"] if forward else None,
        )


def graph_a(ctx: Ctx, live: bool) -> dict:
    """$0 local build, but write-once: only with ``live=True`` and only over a complete picks dir."""
    picks = check_picks(ctx)
    if not live:
        return result("graph_a", picks)
    built = _graph_build(ctx, ctx.path("graph_a"), ctx.path("runs"))
    return result("graph_a", picks, built)
