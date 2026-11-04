from decimal import Decimal
from pathlib import Path

from memory_kernel.pipeline import ledger, llm


def test_stage_cap_is_unchanged_up_to_the_reference_and_scales_above():
    assert llm.stage_cap("events", 300) == llm.STAGE_CAPS["events"]
    assert llm.stage_cap("events", 369) == llm.STAGE_CAPS["events"]
    assert abs(llm.stage_cap("events", 1717) - llm.STAGE_CAPS["events"] * 1717 / 369) < 1e-9


def test_round_cap_is_sealed_and_reported(tmp_path: Path):
    out = tmp_path / "shard"
    ledger.make_execution(out, round_id="r1", capture_sha="c" * 64, g0_sha="d" * 64, cap_usd=Decimal("5"))
    shard = ledger.Shard(path=out)
    manifest, provenance, _ = shard.load()
    budget = shard.open(manifest, provenance)
    try:
        assert budget.summary()["cap_usd"] == "5"
    finally:
        budget.close()


def test_default_round_cap_is_two_dollars(tmp_path: Path):
    out = tmp_path / "shard"
    ledger.make_execution(out, round_id="r1", capture_sha="c" * 64, g0_sha="d" * 64)
    manifest, _, _ = ledger.Shard(path=out).load()
    assert manifest["budget_cap_usd"] == "2"


def test_round_cap_above_the_integrated_cap_is_refused():
    import pytest

    assert Decimal("25") == ledger.INTEGRATED_CAP
    most = 4612  # 25 / 2 * 369, rounded down
    assert ledger.round_cap(most) <= ledger.INTEGRATED_CAP
    with pytest.raises(ValueError, match=r"4613 messages.*\$25.*at most 4612 messages"):
        ledger.round_cap(most + 1)


def test_a_new_shard_records_the_integrated_cap(tmp_path: Path):
    out = tmp_path / "shard"
    ledger.make_execution(out, round_id="r1", capture_sha="c" * 64, g0_sha="d" * 64)
    manifest, _, _ = ledger.Shard(path=out).load()
    assert manifest["integrated_cap_usd"] == "25"
    assert ledger.integrated_cap(manifest) == Decimal("25")


def _hold_then_summary(shard: ledger.Shard, usd: str) -> dict:
    """Reserve ``usd`` under a round cap of $25, then read the ledger's summary (where the integrated cap is enforced)."""
    manifest, provenance, _ = shard.load()
    budget = shard.open(manifest, provenance)
    try:
        budget.reserve(
            wire={"w": usd},
            stage="jev",
            model=ledger.MODEL,
            input_cap=1_000_000,
            output_cap=0,
            input_rate=Decimal(usd),
            output_rate=Decimal(0),
        )
        return budget.summary()
    finally:
        budget.close()


def test_a_manifest_without_the_field_enforces_ten_and_a_new_one_twenty_five(tmp_path: Path):
    import json

    import pytest

    from memory_kernel.construction.memory_semantic_jev import seal

    new, old = tmp_path / "new", tmp_path / "old"
    for out in (new, old):
        ledger.make_execution(out, round_id="r1", capture_sha="c" * 64, g0_sha="d" * 64, cap_usd=Decimal("25"))
    body = {k: v for k, v in json.loads((old / "manifest.json").read_bytes()).items() if k != "content_sha256"}
    del body["integrated_cap_usd"]
    (old / "manifest.json").write_bytes(ledger.canonical_json(seal(body)).encode())
    with pytest.raises(ValueError, match="shared integrated budget exceeded"):
        _hold_then_summary(ledger.Shard(path=old), "10.5")
    summary = _hold_then_summary(ledger.Shard(path=new), "10.5")
    assert (summary["integrated_cap_usd"], summary["held_or_spent_usd"]) == ("25", "10.5")
    # the legacy shard reports its own $10; LedgerView takes the shard's value, falling back to $10
    assert llm.LedgerView({**summary, "integrated_cap_usd": "10", "cap_usd": "25"}).integrated_cap == Decimal("10")
    legacy = {k: v for k, v in summary.items() if k != "integrated_cap_usd"}
    assert llm.LedgerView(legacy).integrated_cap == Decimal("10")
    assert llm.LedgerView(summary).integrated_cap == Decimal("25")


def test_a_stage_cap_is_recorded_and_the_unit_cap_still_refuses_a_hold(tmp_path: Path):
    """Past its stage cap a metered stage holds and sends on (``cap_record`` says so); ``--cap-usd`` (the Meter's
    unit cap) still refuses a hold that would cross it, through the ledger's ``reserve_guard``."""
    import json

    import pytest

    out = tmp_path / "shard"
    ledger.make_execution(out, round_id="r1", capture_sha="c" * 64, g0_sha="d" * 64, cap_usd=Decimal("5"))
    shard = ledger.Shard(path=out)
    meter = llm.Meter(shard.budget, 0.01, unit_spent=0.5, unit_cap=1.0)
    assert meter.cap_record() == {}
    (shard.budget / "jev-sent").mkdir(parents=True)
    (shard.budget / "jev-sent" / "completed.json").write_text(json.dumps({"receipt": {"cost_usd": 0.03}}))
    assert meter.cap_record() == {"stage_cap_usd": "0.01", "stage_cap_spend_usd": "0.03", "stage_cap_exceeded": True}
    assert meter.would_cross(0.2).startswith("stage cap 0.0100 would be crossed")  # the prefetch planner's view

    def hold(usd: str) -> None:
        manifest, provenance, _ = shard.load()
        budget = shard.with_hooks(reserve_guard=llm.guard(meter)).open(manifest, provenance)
        try:
            budget.reserve(
                wire={"w": usd},
                stage="jev",
                model=ledger.MODEL,
                input_cap=1_000_000,
                output_cap=0,
                input_rate=Decimal(usd),
                output_rate=Decimal(0),
            )
        finally:
            budget.close()

    hold("0.2")  # 0.03 spent + 0.2 > the 0.01 stage cap: held
    with pytest.raises(llm.StopSpend, match=r"unit cap 1.0 would be crossed"):
        hold("0.5")  # 0.5 paid + 0.03 + 0.5 > the 1.0 unit cap: refused
