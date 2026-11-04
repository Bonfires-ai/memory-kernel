"""A model family check accepts the pinned alias or a dated snapshot of it; another family is refused and releases its
hold (the Jev archive reader)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from memory_kernel.model_family import model_family_ok
from memory_kernel.pipeline import ledger


def test_dated_snapshot_of_the_alias_is_accepted():
    assert model_family_ok("openai/gpt-6-luna", "openai/gpt-6-luna-20261201")
    assert model_family_ok("openai/gpt-6-luna", "openai/gpt-6-luna")


def test_other_family_is_refused():
    assert not model_family_ok("openai/gpt-6-luna", "anthropic/claude-x")
    assert not model_family_ok("openai/gpt-6-luna", "openai/gpt-6-luna-mini")
    assert not model_family_ok("openai/gpt-6-luna", "openai/gpt-6-luna-")


def _jev_budget(tmp_path: Path) -> ledger.RoundBudget:
    ledger.make_execution(tmp_path / "j", round_id="x1-jev", capture_sha="c" * 64, g0_sha="d" * 64)
    shard = ledger.Shard(path=tmp_path / "j", sender=None, api_key="k")
    manifest, provenance, _pairs = shard.load()
    return shard.open(manifest, provenance)


_JEV_WIRE = {"model": ledger.MODEL, "questions": {"q1": {"type": "choice", "criteria": ["yes", "no"], "text": "t"}}}


def _jev_sender(model: str):
    def send(_wire, *, on_response, **_kwargs):
        raw = {
            "model": model,
            "usage": {"input_tokens": 12, "output_tokens": 3, "cost": 0},
            "answers": {"q1": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9, "no": 0.1}}},
        }
        on_response(json.dumps(raw).encode(), raw, 1)
        return raw, None

    return send


def test_a_dated_jev_snapshot_is_accepted_by_the_archive_reader(tmp_path: Path):
    budget = _jev_budget(tmp_path)
    try:
        archive = budget.evaluate(_JEV_WIRE, sender=_jev_sender(ledger.MODEL + "-20261201"))
        assert json.loads(archive["raw"])["model"] == ledger.MODEL + "-20261201"
    finally:
        budget.close()


def test_a_refused_jev_model_is_settled_and_re_askable(tmp_path: Path):
    budget = _jev_budget(tmp_path)
    try:
        with pytest.raises(ledger.JevModelRefused):
            budget.evaluate(_JEV_WIRE, sender=_jev_sender("anthropic/claude-x"))
        summary = budget.summary()
        assert Decimal(summary["unresolved_reservations_usd"]) == 0 and summary["completed"] == 1
        kept = list(budget.directory.glob("jev-*-refused-1"))
        assert len(kept) == 1 and (kept[0] / "raw.json").exists()
        assert json.loads((kept[0] / "failure.json").read_bytes())["reservation_retained"] is False
        archive = budget.evaluate(_JEV_WIRE, sender=_jev_sender(ledger.MODEL + "-20261201"))
        assert archive["raw"]
    finally:
        budget.close()
