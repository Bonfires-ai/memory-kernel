"""An entity's summary lines as stored records (1.4.1): serving reads them, it never parses a stored line.

The writer stores, parallel to ``summary_line_ids``, ``summary_line_texts`` and ``summary_identity_line_ids`` (the
identity lines among them, decided at write). A row without them was built before 1.4.1 and not yet backfilled.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_LOG = logging.getLogger(__name__)
_NOTED: set[str] = set()


@dataclass(frozen=True)
class SheetLine:
    id: str
    text: str
    identity: bool


def lines_of(row: Mapping[str, Any]) -> list[SheetLine] | None:
    """The row's line records, in line order: ``[]`` for an entity with no line (no line id, a blank summary); None
    when the row's lines have no whole per-line data (``kg_contract.line_data``: a legacy row, built before 1.4.1 or
    Graphiti-written, or one desynced since its backfill)."""
    from memory_kernel.graph.kg_contract import line_data

    ids, texts, ident = row.get("line_ids"), row.get("line_texts"), row.get("identity_line_ids")
    if ids is None:
        ids = row.get("summary_line_ids")
    if texts is None:
        texts = row.get("summary_line_texts")
    if ident is None:
        ident = row.get("summary_identity_line_ids")
    state = line_data(row.get("summary"), ids, texts, ident)
    if state != "whole":
        return [] if state == "none" else None
    marked = set(ident or ())
    return [SheetLine(str(i), str(t), str(i) in marked) for i, t in zip(ids or (), texts or (), strict=True)]


def note_missing(group_id: str) -> None:
    """Log once per group that identity lines cannot be read from a legacy row."""
    if group_id in _NOTED:
        return
    _NOTED.add(group_id)
    _LOG.warning(
        "identity lines unavailable (graph built before 1.4.1): run `mk bonfire backfill-lines <bonfire>` (group %s)",
        group_id,
    )


def recover_lines(entity: str, summary: str, sources: Sequence[str], ids: Sequence[str]) -> list[str] | None:
    """The exact line texts of a legacy row, by the content hash that identifies each line.

    A line may contain a newline, so ``summary.split("\\n")`` is not the line list: pieces are joined with a newline
    until the candidate's ``summary_line_id`` is the stored id (a text repeated under two sources is told apart by
    its source). None when an id matches no run of pieces, or pieces remain after the last id."""
    from memory_kernel.graph.g1_ids import summary_line_id

    if len(sources) != len(ids):
        return None
    pieces = summary.split("\n")
    out: list[str] = []
    at = 0
    for source, want in zip(sources, ids, strict=True):
        for end in range(at + 1, len(pieces) + 1):
            candidate = "\n".join(pieces[at:end])
            if summary_line_id(entity, source, candidate) == want:
                out.append(candidate)
                at = end
                break
        else:
            return None
    return out if at == len(pieces) else None
