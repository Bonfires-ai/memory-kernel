"""The provider's ``usage`` must carry the three metered keys; extra keys are accepted, bad values still refused."""

from __future__ import annotations

import pytest

from memory_kernel.construction.memory_jev_archive import RECEIPT_SCHEMA, JevArchiveError, read_jev_archive
from memory_kernel.construction.memory_mention_jev import MODEL
from memory_kernel.construction.memory_semantic_jev import seal, sha
from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256

WIRE = {"model": MODEL, "questions": {"q1": {"type": "choice", "criteria": ["yes", "no"]}}}
METERED = {"input_tokens": 12, "output_tokens": 3, "cost": 0.0003}


def archive_for(usage: dict) -> dict:
    raw = canonical_json(
        {
            "model": MODEL,
            "usage": usage,
            "answers": {"q1": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9, "no": 0.1}}},
        }
    )
    receipt = seal(
        {
            "schema_version": RECEIPT_SCHEMA,
            "evidence_mode": "synthetic",
            "request_sha256": content_sha256(WIRE),
            "raw_response_sha256": sha(raw.encode()),
            "ledger_record_sha256": None,
            "usage": {"input_tokens": 12, "output_tokens": 3, "cost_usd": 0.0003},
        }
    )
    return {"wire": canonical_json(WIRE), "raw": raw, "receipt": canonical_json(receipt)}


def read(usage: dict) -> dict:
    policy = {"evidence_mode": "synthetic", "model": MODEL, "provider_run": None}
    return read_jev_archive(WIRE, archive_for(usage), policy)


def test_extra_provider_usage_keys_are_accepted():
    extra = {**METERED, "is_byok": False, "cost_details": {"upstream_inference_cost": 0.0003}}
    assert read(extra)["usage"] == {"input_tokens": 12, "output_tokens": 3, "cost_usd": 0.0003}


@pytest.mark.parametrize("usage", [{"input_tokens": 12, "output_tokens": 3}, {**METERED, "cost": -1}])
def test_a_missing_or_negative_metered_value_is_still_refused(usage):
    with pytest.raises(JevArchiveError, match="provider usage"):
        read(usage)
