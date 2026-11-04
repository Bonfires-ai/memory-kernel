"""Portable text and statement normalization helpers.

The tokenize/dedupe/scalar/identity primitives this module used to OWN now live in
:mod:`memory_kernel.primitives`, the single home for them (six competing homes,
three of which had diverged — see that module's docstring). They are re-exported
here unchanged so every existing importer, in this tree and outside it, keeps
working; what stays defined here is the statement-normalization layer built ON
those primitives.
"""

from __future__ import annotations

from typing import Any

from memory_kernel.primitives import (
    as_float,
    bounded,
    clean,
    dedupe,
    english_stopwords,
    retrieval_tokenize,
    stable_id,
    string_list,
)


def _timestamp_sort_key(value: Any) -> str:
    """Normalize a timestamp (datetime or ISO-str) to a comparable ISO string.

    Statements carry heterogeneous timestamp types — datetime objects from some
    sources, ISO strings from others — so a bare ``max`` raises ``TypeError`` on
    mixed types (and on naive-vs-aware datetimes). Keying by the ISO-string form
    makes the comparison total without changing the value that is returned.
    """
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def latest_timestamp(statements: list[Any]) -> Any:
    values = [
        getattr(statement, "timestamp", None) for statement in statements if getattr(statement, "timestamp", None)
    ]
    return max(values, key=_timestamp_sort_key) if values else None


__all__ = [
    "as_float",
    "bounded",
    "clean",
    "dedupe",
    "english_stopwords",
    "latest_timestamp",
    "stable_id",
    "string_list",
    "retrieval_tokenize",
]
