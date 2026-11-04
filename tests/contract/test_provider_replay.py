"""Provider replay: a shard pins the code that spends, an archived Jev response is served by its wire sha, and a miss
is refused instead of sent.

Everything is in ``tmp_path``, offline. ``socket.connect`` is patched to fail, so an accidental send would raise
instead of leaving the machine, and the ledger is asserted to hold no completed (paid) call at the end.
"""

from __future__ import annotations

import json
import socket
from decimal import Decimal
from pathlib import Path

import pytest

from memory_kernel.construction.memory_semantic_jev import seal, sha
from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256
from memory_kernel.pipeline import ledger
from memory_kernel.pipeline.llm import RefusingSender, ReplayMiss

CAPTURE_SHA, G0_SHA = "c" * 64, "d" * 64


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("replay tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def make_shard(out: Path, sender=None, api_key=None) -> ledger.Shard:
    ledger.make_execution(out, round_id="x1-replay", capture_sha=CAPTURE_SHA, g0_sha=G0_SHA)
    return ledger.Shard(path=out, sender=sender, api_key=api_key)


def open_budget(shard: ledger.Shard) -> ledger.RoundBudget:
    manifest, provenance, _pairs = shard.load()
    return shard.open(manifest, provenance)


def wire(label: str) -> dict:
    """A Jev request: one finite-choice question. ``label`` makes the wire (and so its sha) unique."""
    return {"model": ledger.MODEL, "questions": {"q1": {"type": "choice", "criteria": ["yes", "no"], "text": label}}}


def write_archive(budget: ledger.RoundBudget, request: dict) -> str:
    """Hand-write ``budget/jev-<sha>/archive.json`` in the shape ``read_jev_archive`` accepts for a receipted call."""
    request_sha = content_sha256(request)
    usage = {"input_tokens": 12, "output_tokens": 3, "cost": 0}
    raw = canonical_json(
        {
            "model": ledger.MODEL,
            "usage": usage,
            "answers": {"q1": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9, "no": 0.1}}},
        }
    )
    pins = budget.policy["provider_run"]
    ledger_record = canonical_json(
        {
            "status": "completed",
            "stage": "jev",
            "model": ledger.MODEL,
            "request_sha256": request_sha,
            "request_id": "jev-" + request_sha,
            "reservation_id": "hold-" + request_sha,
            "manifest_sha256": pins["manifest"]["content_sha256"],
            "provenance_sha256": pins["provenance"]["content_sha256"],
            "created_at_unix": 1.0,
            "completed_at_unix": 2.0,
            "input_token_cap": 4096,
            "output_token_cap": 4096,
            "reserved_usd": 0.001,
            "charged_usd": 0.001,
            "receipt": {"input_tokens": 12, "output_tokens": 3, "cost_usd": 0.0},
        }
    )
    receipt = seal(
        {
            "schema_version": ledger.RECEIPT_SCHEMA,
            "evidence_mode": "provider_receipted",
            "request_sha256": request_sha,
            "raw_response_sha256": sha(raw.encode()),
            "ledger_record_sha256": sha(ledger_record.encode()),
            "usage": {"input_tokens": 12, "output_tokens": 3, "cost_usd": 0.0},
        }
    )
    archive = {
        "wire": canonical_json(request),
        "raw": raw,
        "receipt": canonical_json(receipt),
        "ledger_record": ledger_record,
    }
    attempt = budget.directory / ("jev-" + request_sha)
    attempt.mkdir()
    (attempt / "archive.json").write_text(json.dumps(archive))
    return request_sha


def test_make_execution_pins_the_mk_code_and_reopens(tmp_path):
    shard = make_shard(tmp_path / "shard")
    manifest, provenance, pairs = shard.load()
    assert set(provenance["code_pins"]) == set(ledger.MK_CODE)
    assert manifest["budget_cap_usd"] == "2"
    assert manifest["rows"] == [] and pairs == []  # a build's shard is question-blind


def test_an_archived_response_is_served_by_wire_sha_without_a_network(tmp_path, no_network):
    sender = RefusingSender()
    shard = make_shard(tmp_path / "shard", sender=sender, api_key="replay-invalid")
    budget = open_budget(shard)
    try:
        request = wire("archived")
        request_sha = write_archive(budget, request)

        archive = budget.evaluate(request)

        assert archive["wire"] == canonical_json(request)
        assert json.loads(archive["receipt"])["request_sha256"] == request_sha
        assert sender.misses == []  # served from the archive; the sender was never called
        summary = budget.summary()
        assert summary["calls"] == 0 and summary["completed"] == 0 and Decimal(summary["actual_usd"]) == 0
    finally:
        budget.close()


def test_a_miss_is_refused_and_nothing_is_paid(tmp_path, no_network):
    sender = RefusingSender()
    shard = make_shard(tmp_path / "shard", sender=sender, api_key="replay-invalid")
    budget = open_budget(shard)
    try:
        write_archive(budget, wire("archived"))
        with pytest.raises(ReplayMiss, match="replay mode never sends"):
            budget.evaluate(wire("not archived"))
        assert len(sender.misses) == 1
        summary = budget.summary()
        assert summary["completed"] == 0 and Decimal(summary["actual_usd"]) == 0
    finally:
        budget.close()


def test_a_ledger_with_no_sender_or_key_refuses_a_miss_before_it_reserves_anything(tmp_path, no_network):
    shard = make_shard(tmp_path / "shard")
    budget = open_budget(shard)
    try:
        with pytest.raises(ValueError, match="replay-only"):
            budget.evaluate(wire("not archived"))
        assert list(budget.directory.glob("*/reservation.json")) == []
    finally:
        budget.close()
