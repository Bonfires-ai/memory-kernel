"""Native PropBank frames and their token basis from one parser request."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class NativeFrameTransportError(ValueError):
    """The native response cannot establish exact frame coordinates."""


def source_tokens_from_nodes(nodes: Any, text: str) -> tuple[str, ...]:
    """Read the native serializer's leaf coordinates, never retokenize text."""
    if not isinstance(nodes, list) or not nodes:
        raise NativeFrameTransportError("missing native syntax nodes")
    leaves: dict[int, str] = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            raise NativeFrameTransportError("malformed native syntax node")
        if node.get("node--type") != "leaf":
            continue
        start, end, token = node.get("start"), node.get("end"), node.get("string")
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end != start + 1
            or start in leaves
            or not isinstance(token, str)
            or not token
            or any(char.isspace() for char in token)
        ):
            raise NativeFrameTransportError("invalid or duplicate native leaf coordinate")
        leaves[start] = token
    if not leaves or sorted(leaves) != list(range(len(leaves))):
        raise NativeFrameTransportError("native leaf coordinates are incomplete")
    tokens = tuple(leaves[index] for index in range(len(leaves)))
    if "".join(tokens) != "".join(text.split()):
        raise NativeFrameTransportError("native token basis does not cover the utterance")
    return tokens
