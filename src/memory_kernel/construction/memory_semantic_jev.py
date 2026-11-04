"""Finite existing-instance reference, assertion and slot-fit questions.

No model calls occur here. Exact transport artifacts are required for replay;
the explicitly synthetic mode never claims provider or native acceptance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from memory_kernel.construction.mined_affordance_contract import content_sha256

MIN_PROBABILITY = 0.75
MIN_MARGIN = 0.20


class SemanticSlotError(ValueError):
    """Semantic evidence is unlicensed, stale or outside its finite scope."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**deepcopy(dict(body)), "content_sha256": content_sha256(body)}


def check_seal(row: Mapping[str, Any], schema: str) -> None:
    if (
        not isinstance(row, Mapping)
        or row.get("schema_version") != schema
        or row.get("content_sha256") != content_sha256({k: v for k, v in row.items() if k != "content_sha256"})
    ):
        raise SemanticSlotError("semantic schema or digest differs")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_pins(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {"manifest", "provenance"}:
        raise SemanticSlotError("provider run descriptors differ")
    result = {}
    for key, pin in raw.items():
        if not isinstance(pin, Mapping) or set(pin) != {"path", "file_sha256"}:
            raise SemanticSlotError("provider run physical descriptor differs")
        pin = cast(Mapping[str, Any], pin)
        if not isinstance(pin["path"], str):
            raise SemanticSlotError("provider run path must be text")
        path = Path(pin["path"]).resolve(strict=True)
        data = path.read_bytes()
        value = json.loads(data)
        if sha(data) != pin["file_sha256"] or not isinstance(value, Mapping):
            raise SemanticSlotError("provider run physical bytes differ")
        result[key] = {"path": str(path), "file_sha256": sha(data), "content_sha256": content_sha256(value)}
    return result


def semantic_policy(
    *, model: str, evidence_mode: str = "synthetic", provider_run: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if evidence_mode not in {"synthetic", "provider_receipted"} or not isinstance(model, str) or not model:
        raise SemanticSlotError("semantic provider policy differs")
    if (evidence_mode == "provider_receipted") != (provider_run is not None):
        raise SemanticSlotError("provider mode requires immutable physical run descriptors")
    return {
        "model": model,
        "evidence_mode": evidence_mode,
        "provider_run": run_pins(provider_run) if provider_run is not None else None,
        "minimum_probability": MIN_PROBABILITY,
        "minimum_margin": MIN_MARGIN,
    }
