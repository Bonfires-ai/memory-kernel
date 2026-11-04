"""Scope-preserving native task scheduling on one attested grammar image.

Type predictions order source tasks; they do not reorder FCG constructions or
certify frames. Child parses are contextual evidence, never independent effects.
"""

from __future__ import annotations

from memory_kernel.construction.memory_discourse_contract import (
    DiscourseContractError,
)


class DiscourseNativeError(DiscourseContractError):
    """The native request or retained evidence cannot establish exact context."""


def _native_token_char_spans(text: str, tokens: tuple[str, ...], origin: int) -> list[list[int]]:
    """Align the native basis in order; token coordinates are never char offsets."""
    cursor = 0
    spans = []
    for token in tokens:
        if type(token) is not str or not token or any(char.isspace() for char in token):
            raise DiscourseNativeError("native token basis differs")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if not text.startswith(token, cursor):
            raise DiscourseNativeError("native token does not align exactly with original source")
        spans.append([origin + cursor, origin + cursor + len(token)])
        cursor += len(token)
    if text[cursor:].strip():
        raise DiscourseNativeError("native tokens leave original source uncovered")
    return spans
