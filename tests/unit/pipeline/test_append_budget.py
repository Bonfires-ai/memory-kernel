"""Per-append budgets (kernel 2.0): an append spends against its own limit, counted from the shard's total when the run
was planned (``run_config``'s ``limit_base_usd``) and held under the bonfire's ceiling; ``kernel_ops`` refuses the hold
past the ceiling (``test_kernel_ops::test_hold_past_the_ceiling_is_refused``, ``test_graph_append::test_ceiling_refusal_
comes_from_kernel_ops``). Pure: no shard, no network."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from memory_kernel.pipeline import bonfire, runner
from memory_kernel.pipeline.config import BuildConfig


def test_three_message_preflight_covers_planned_jev():
    stop = Decimal(str(runner.STOP_FACTOR))
    assert bonfire.append_budget(3, planned_jev_usd=Decimal("0.0125")) >= Decimal("0.0125") * stop
    assert bonfire.append_budget(3, planned_jev_usd=Decimal("1")) == stop  # a plan past the floor sets the budget
    per_message = stop * Decimal(str(runner.INDEX_USD_PER_MESSAGE)) * 400
    assert bonfire.append_budget(400, planned_jev_usd=Decimal(0)) == per_message  # a long session: per message


def run_cfg(tmp_path: Path, *, base_usd, ceiling) -> BuildConfig:
    base = BuildConfig(
        sample_id="conv-ember", conversation=tmp_path / "c.json", group="g", workdir=tmp_path / "w", root=tmp_path
    )
    return bonfire.run_config(
        base,
        tmp_path / "w" / "runs" / "0001" / "conv-ember",
        conversation=tmp_path / "c.json",
        forward_input=None,
        new_messages=3,
        limit_base_usd=base_usd,
        ceiling_usd=ceiling,
        index_record=tmp_path / "index.json",
        shard_dir=None,
        capture_cache=None,
    )


def test_the_run_limit_is_the_base_plus_the_budget_under_the_ceiling(tmp_path):
    budget = bonfire.append_budget(3, planned_jev_usd=Decimal(0))
    cfg = run_cfg(tmp_path, base_usd=Decimal("26.0"), ceiling=None)
    assert cfg.graph_commit is True and cfg.write_mode == "append" and cfg.append_messages == 3
    assert (
        bonfire.append_limit(cfg) == Decimal("26.0") + budget
    )  # past the old sealed caps: they bound the bootstrap only
    capped = run_cfg(tmp_path, base_usd=Decimal("26.0"), ceiling=Decimal("26.05"))
    assert bonfire.append_limit(capped) == Decimal("26.05")  # never past the bonfire's ceiling


def test_a_run_that_is_not_an_append_has_no_limit(tmp_path):
    base = BuildConfig(sample_id="s", conversation=tmp_path / "c.json", group="g", workdir=tmp_path, root=tmp_path)
    assert bonfire.append_limit(base) is None


def test_the_estimate_scales_the_last_run_and_never_drops_below_the_floor():
    floor = bonfire.append_budget(2, planned_jev_usd=Decimal(0))
    assert bonfire.append_estimate(2, []) == floor
    runs = [{"budget_usd": 1.0, "new_messages": 10}]
    assert bonfire.append_estimate(20, runs) == max(bonfire.append_budget(20, planned_jev_usd=Decimal(0)), Decimal("2"))
    assert bonfire.append_estimate(2, [{"new_messages": 0, "budget_usd": 5}]) == floor  # no messages: skipped


def test_the_floor_is_the_least_an_append_may_add():
    assert bonfire.append_budget(1, planned_jev_usd=Decimal(0)) >= bonfire.APPEND_FLOOR_USD
