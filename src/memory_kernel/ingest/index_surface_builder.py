"""Searchable text surfaces for hypergraph nodes."""

from __future__ import annotations

from typing import Protocol


class TaxonomyTermsProvider(Protocol):
    """The one capability this module needs from a WordNet backend, kept as a Protocol so it can be substituted."""


class IndexSurfaceBuilder:
    """Build weighted topic, episode, and fact document surfaces."""

    def __init__(self, *, wordnet_semantics: TaxonomyTermsProvider | None = None) -> None:
        """Hold an optional WordNet provider used to expand fact text.

        Passing ``None`` does not disable expansion — it defers to the process-wide provider, resolved lazily at
        first use so constructing a builder never loads a corpus.
        """
        self._wordnet_semantics = wordnet_semantics


__all__ = [
    "IndexSurfaceBuilder",
    "TaxonomyTermsProvider",
]
