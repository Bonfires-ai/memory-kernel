"""Bounded source units, not independent statements or semantic scope licenses."""

from __future__ import annotations

from collections.abc import Mapping

from memory_kernel.construction.memory_discourse_contract import (
    check_discourse_syntax,
    discourse_id,
)


def compile_source_units(source: Mapping, syntax: Mapping) -> list[dict]:
    """Preserve the message partition and every proposed embedded-clause dependency.

    Sentence boundaries bound computation, not truth or operator independence.
    An incomplete syntactic partition remains one opaque source unit, never a
    hard word cut. No native full-message parse is required to retain its context.
    """
    check_discourse_syntax(syntax, source)
    root = next(row for row in syntax["segments"] if row["parent_id"] is None)
    sentences = sorted(
        (row for row in syntax["segments"] if row["kind"] == "sentence"), key=lambda row: (row["start"], row["end"])
    )
    cursor = 0
    complete = bool(sentences)
    for row in sentences:
        complete = complete and row["start"] == cursor
        cursor = row["end"]
    complete = complete and cursor == len(source["text"])
    selected = sentences if complete else [root]
    units = []
    for ordinal, row in enumerate(selected):
        units.append(
            {
                "unit_id": discourse_id("source-unit", source["content_sha256"], row["segment_id"]),
                "source_sha256": source["content_sha256"],
                "message_id": source["message_id"],
                "ordinal": ordinal,
                "segment_id": row["segment_id"],
                "kind": row["kind"],
                "start": row["start"],
                "end": row["end"],
                "parent_segment_id": row["parent_id"],
                "retained_clause_ids": [
                    child["segment_id"]
                    for child in syntax["segments"]
                    if child["kind"] == "clause" and row["start"] <= child["start"] < child["end"] <= row["end"]
                ],
            }
        )
    return units
