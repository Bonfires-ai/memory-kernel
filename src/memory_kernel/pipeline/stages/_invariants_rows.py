"""The frozen-field sha of a projection: its canonical rows with the embedding fields dropped, hashed.

Ported from a reference-harness module (unreached top-level definitions pruned).

``canonical_rows`` writes one sorted-key JSON line per node and edge (``props`` minus ``EMBEDDING_FIELDS`` and
any extra fields), sorted; ``frozen_sha`` is the sha256 of those lines.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

EMBEDDING_FIELDS = frozenset({"name_embedding", "fact_embedding", "search_key_embedding"})


def _row(row: dict[str, Any], drop: frozenset[str]) -> str:
    out = {k: v for k, v in row.items() if k != "props"}
    out["props"] = {k: v for k, v in row["props"].items() if k not in drop}
    return json.dumps(out, sort_keys=True, ensure_ascii=True, default=str)


def canonical_rows(rows: dict[str, Any], *, drop: frozenset[str] = EMBEDDING_FIELDS) -> list[str]:
    """One sorted-key JSON line per node and edge, sorted."""
    return sorted(_row(r, drop) for r in [*rows["nodes"], *rows["edges"]])


def frozen_sha(rows: dict[str, Any], *, extra_drop: frozenset[str] = frozenset()) -> str:
    text = "\n".join(canonical_rows(rows, drop=EMBEDDING_FIELDS | extra_drop))
    return hashlib.sha256(text.encode()).hexdigest()
