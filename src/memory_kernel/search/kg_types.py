"""Shared result types and rank fusion for the graph message search (``kg_recipe``, ``kg_reader``, ``kg_context``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple


class MessageHit(NamedTuple):
    """A ranked message and the item uuids (edges, nodes, episodes) that support it, best first."""

    message: str
    score: float
    support: tuple[str, ...]


@dataclass
class SearchResult:
    hits: list[MessageHit]
    sagas: list[tuple[str, float]]
    trace: dict[str, dict[str, Any]] = field(default_factory=dict)  # item uuid -> per-stage contributions
    items: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # item uuid -> rendered-fact props (edges and events)
    messages: dict[str, dict[str, Any]] = field(default_factory=dict)
    lists: dict[str, list[str]] = field(default_factory=dict)
