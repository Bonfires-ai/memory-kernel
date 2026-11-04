"""The one content-derived uuid formula of the G1 projection and its codings.

A leaf module (no graph imports), so ``g1_projection`` and the archived 1.7 codings module shared one definition: a
drift between copies would silently mint different uuids for the same row.
"""

from __future__ import annotations

import hashlib


def content_uuid(*parts: object) -> str:
    """``sha256("|".join(parts))[:32]``: the uuid of a row from its identifying parts."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:32]


def summary_line_id(entity: str, source: str, line: str) -> str:
    """Id of one attributed summary line: its entity, its source Episodic and its text (the line's dedup key), so
    adding, removing or reordering other lines never changes it."""
    return content_uuid("line", entity, source, line)
