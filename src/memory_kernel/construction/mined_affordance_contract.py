"""Receipt-bound, production-inert adapter from mined cards to question DAGs."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


__all__ = [
    "canonical_json",
    "content_sha256",
    "source_sha256",
]
