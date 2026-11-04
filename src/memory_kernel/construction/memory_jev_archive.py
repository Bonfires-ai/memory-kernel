"""Receipt-bound finite Jev choices shared by integrated ingestion phases.

The host owns sending and the budget ledger. This reader accepts original UTF-8
artifacts, never an answer dictionary supplied independently of those artifacts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from memory_kernel.construction.memory_jev_probabilities import normalize_integrated_probabilities
from memory_kernel.construction.memory_mention_jev import MODEL
from memory_kernel.construction.memory_semantic_jev import check_seal, run_pins, sha
from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256
from memory_kernel.model_family import model_family_ok

RECEIPT_SCHEMA = "memory-integrated-jev-send-receipt/v1"
MIN_PROBABILITY = .75
MIN_MARGIN = .20


class JevArchiveError(ValueError):
    """An archived finite choice is not bound to its authorized request."""


def read_jev_archive(wire: Mapping[str, Any], archive: Mapping[str, str],
                     policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact transport, accounting, model identity and finite options."""
    mode = policy.get("evidence_mode")
    if (mode not in {"synthetic", "provider_receipted"} or policy.get("model") != MODEL
            or wire.get("model") != MODEL):
        raise JevArchiveError("integrated Jev model or evidence policy differs")
    expected = {"wire", "raw", "receipt"}
    if mode == "provider_receipted":
        expected.add("ledger_record")
    if not isinstance(archive, Mapping) or set(archive) != expected or any(
        not isinstance(value, str) for value in archive.values()
    ):
        raise JevArchiveError("integrated Jev archive fields differ")
    if archive["wire"] != canonical_json(wire):
        raise JevArchiveError("archived wire bytes differ")
    raw = json.loads(archive["raw"])
    decimal_raw = json.loads(archive["raw"], parse_float=Decimal)
    receipt = json.loads(archive["receipt"])
    check_seal(receipt, RECEIPT_SCHEMA)
    if not isinstance(raw, Mapping) or not model_family_ok(MODEL, str(raw.get("model"))):
        raise JevArchiveError("provider response model differs")
    usage = raw.get("usage")
    if not isinstance(usage, Mapping) or set(usage) != {"input_tokens", "output_tokens", "cost"}:
        raise JevArchiveError("provider usage differs")
    if (any(type(usage[key]) is not int or usage[key] < 0 for key in ("input_tokens", "output_tokens"))
            or type(usage["cost"]) not in (int, float) or not math.isfinite(usage["cost"])
            or usage["cost"] < 0):
        raise JevArchiveError("provider usage values differ")
    expected_usage = {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                      "cost_usd": float(usage["cost"])}
    if (set(receipt) != {"schema_version", "evidence_mode", "request_sha256", "raw_response_sha256",
                        "ledger_record_sha256", "usage", "content_sha256"}
            or receipt["evidence_mode"] != mode or receipt["usage"] != expected_usage
            or receipt["request_sha256"] != content_sha256(wire)
            or receipt["raw_response_sha256"] != sha(archive["raw"].encode())):
        raise JevArchiveError("receipt differs from original wire or response")
    if mode == "provider_receipted":
        pins = run_pins({key: {"path": pin["path"], "file_sha256": pin["file_sha256"]}
                         for key, pin in policy["provider_run"].items()})
        if pins != policy["provider_run"]:
            raise JevArchiveError("physical provider run pins differ")
        ledger = json.loads(archive["ledger_record"])
        fields = {"charged_usd", "completed_at_unix", "created_at_unix", "input_token_cap",
                  "manifest_sha256", "model", "output_token_cap", "provenance_sha256", "receipt",
                  "request_id", "request_sha256", "reservation_id", "reserved_usd", "stage", "status"}
        if (not isinstance(ledger, Mapping) or set(ledger) != fields
                or sha(archive["ledger_record"].encode()) != receipt["ledger_record_sha256"]
                or ledger["status"] != "completed" or ledger["stage"] != "jev"
                or ledger["model"] != MODEL or ledger["request_sha256"] != content_sha256(wire)
                or ledger["manifest_sha256"] != pins["manifest"]["content_sha256"]
                or ledger["provenance_sha256"] != pins["provenance"]["content_sha256"]
                or ledger["receipt"] != expected_usage
                or any(type(ledger[key]) not in (int, float) or not math.isfinite(ledger[key]) or ledger[key] < 0
                       for key in ("charged_usd", "reserved_usd", "created_at_unix", "completed_at_unix"))
                or ledger["completed_at_unix"] < ledger["created_at_unix"]
                or ledger["charged_usd"] != max(ledger["reserved_usd"], usage["cost"])
                or usage["cost"] > ledger["reserved_usd"]
                or any(not isinstance(ledger[key], str) or not ledger[key]
                       for key in ("request_id", "reservation_id"))
                or any(type(ledger[key]) is not int or ledger[key] < 0
                       for key in ("input_token_cap", "output_token_cap"))
                or usage["input_tokens"] > ledger["input_token_cap"]
                or usage["output_tokens"] > ledger["output_token_cap"]):
            raise JevArchiveError("completed provider ledger differs from authorized run")
    elif receipt["ledger_record_sha256"] is not None or policy.get("provider_run") is not None:
        raise JevArchiveError("synthetic evidence cannot carry provider authority")
    questions, answers = wire.get("questions"), raw.get("answers")
    if (not isinstance(questions, Mapping) or not questions or not isinstance(answers, Mapping)
            or set(questions) != set(answers)):
        raise JevArchiveError("provider answer question IDs differ")
    result = {}
    for qid, question in questions.items():
        answer = answers[qid]
        if (not isinstance(question, Mapping) or question.get("type") != "choice"
                or not isinstance(answer, Mapping) or answer.get("type") != "choice"
                or not isinstance(answer.get("probabilities"), Mapping)
                or set(answer["probabilities"]) != set(question["criteria"])
                or answer.get("choice") not in question["criteria"]):
            raise JevArchiveError("provider finite choice differs")
        if len(answer["probabilities"]) < 2:
            raise JevArchiveError("provider distribution differs")
        try:
            probabilities, normalization = normalize_integrated_probabilities(
                decimal_raw["answers"][qid]["probabilities"], answer["choice"])
        except ValueError as exc:
            raise JevArchiveError(str(exc)) from exc
        confidence = answer.get("confidence")
        if confidence is not None and (type(confidence) not in (int, float)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise JevArchiveError("provider confidence telemetry differs")
        ordered = sorted(probabilities.values(), reverse=True)
        selected = probabilities[answer["choice"]]
        result[qid] = {"choice": answer["choice"], "probabilities": dict(probabilities),
                       "raw_probabilities": answer["probabilities"], "normalization": normalization,
                       "confidence_telemetry": confidence,
                       "accepted": selected == ordered[0] and selected >= MIN_PROBABILITY
                       and ordered[0] - ordered[1] >= MIN_MARGIN}
    return {"answers": result, "usage": expected_usage, "raw_sha256": receipt["raw_response_sha256"],
            "receipt_sha256": receipt["content_sha256"], "evidence_mode": mode}
