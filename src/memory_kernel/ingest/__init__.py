"""Ingest-side helpers owned by the memory kernel.

LAZY BY CONSTRUCTION (PEP 562), same mechanism as :mod:`memory_kernel.graph` and the
package root. This file used to import every exported name eagerly, off 30-odd submodules,
and Python must execute it to reach ANY of them — so ``from memory_kernel.ingest.X import Y``
loaded the whole ingest stack, whatever ``X`` was.

That is not merely a cost: it was the **structural cause of the import cycles** the lazy
package root unmasked. Every one of them ran through this file, and none was a genuine
leaf-to-leaf cycle.

Making this ``__init__`` lazy removes the edge that was never intended — the one from "I want
one ingest leaf" to "load every ingest module" — instead of deferring the *legitimate*
leaf imports one at a time behind function bodies, which would have hidden the same dependency
structure from every static tool (call graphs, ``lint-imports``, this package's own import checks).

``from memory_kernel.ingest import X`` and ``import memory_kernel.ingest`` + attribute access
both still work for every name: ``__getattr__`` imports the owning submodule on demand and
caches the value in module globals. ``__all__`` is a literal list of the same names as ``_LAZY_EXPORTS``,
and the import-smoke test resolves every name in both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_kernel.knowledge.temporal_deixis import (
        content_with_relative_time,
        temporal_text,
    )
    from memory_kernel.normalization import (
        latest_timestamp,
    )

    from .episode_boundary import (
        MemoryKernelCentroidTimestampBoundaryService,
        emit_boundary_progress,
        parse_timestamp,
    )
    from .hypergraph_builder import (
        HypergraphBuilder,
        validate_hypergraph,
    )
    from .index_surface_builder import (
        IndexSurfaceBuilder,
    )
    from .nlp_identity_anchor_service import (
        IdentityAnchorDependencies,
        IdentityAnchorService,
    )
    from .nlp_models import (
        Config,
        Episode,
        EpisodeHyperedge,
        Fact,
        FactHyperedge,
        Hypergraph,
        IndexDocument,
        NodeType,
        RetrievalIndex,
        Topic,
    )
    from .retrieval_index import (
        RetrievalIndexBuilder,
        dense_index_by_type,
        documents_grouped_by_type,
        embed_texts,
        embed_texts_adaptive,
        embed_texts_batched,
        normalize,
        propagate_embeddings,
        softmax,
        tokenize,
        vector_sum,
        weighted_edge_embedding,
    )

#: Public name -> the submodule that defines it: the one table ``__getattr__`` resolves names from.
#: ``__all__`` is a literal list of the same names.
_LAZY_EXPORTS: dict[str, str] = {
    "EpisodeHyperedge": ".nlp_models",
    "FactHyperedge": ".nlp_models",
    "Config": ".nlp_models",
    "Episode": ".nlp_models",
    "Fact": ".nlp_models",
    "Hypergraph": ".nlp_models",
    "HypergraphBuilder": ".hypergraph_builder",
    "IndexDocument": ".nlp_models",
    "IndexSurfaceBuilder": ".index_surface_builder",
    "RetrievalIndex": ".nlp_models",
    "RetrievalIndexBuilder": ".retrieval_index",
    "Topic": ".nlp_models",
    "IdentityAnchorDependencies": ".nlp_identity_anchor_service",
    "IdentityAnchorService": ".nlp_identity_anchor_service",
    "MemoryKernelCentroidTimestampBoundaryService": ".episode_boundary",
    "NodeType": ".nlp_models",
    "content_with_relative_time": "memory_kernel.knowledge.temporal_deixis",
    "dense_index_by_type": ".retrieval_index",
    "documents_grouped_by_type": ".retrieval_index",
    "embed_texts": ".retrieval_index",
    "embed_texts_adaptive": ".retrieval_index",
    "embed_texts_batched": ".retrieval_index",
    "emit_boundary_progress": ".episode_boundary",
    "latest_timestamp": "memory_kernel.normalization",
    "normalize": ".retrieval_index",
    "parse_timestamp": ".episode_boundary",
    "propagate_embeddings": ".retrieval_index",
    "softmax": ".retrieval_index",
    "temporal_text": "memory_kernel.knowledge.temporal_deixis",
    "tokenize": ".retrieval_index",
    "validate_hypergraph": ".hypergraph_builder",
    "vector_sum": ".retrieval_index",
    "weighted_edge_embedding": ".retrieval_index",
}

__all__ = [
    "EpisodeHyperedge",
    "FactHyperedge",
    "Config",
    "Episode",
    "Fact",
    "Hypergraph",
    "HypergraphBuilder",
    "IndexDocument",
    "IndexSurfaceBuilder",
    "RetrievalIndex",
    "RetrievalIndexBuilder",
    "Topic",
    "IdentityAnchorDependencies",
    "IdentityAnchorService",
    "MemoryKernelCentroidTimestampBoundaryService",
    "NodeType",
    "content_with_relative_time",
    "dense_index_by_type",
    "documents_grouped_by_type",
    "embed_texts",
    "embed_texts_adaptive",
    "embed_texts_batched",
    "emit_boundary_progress",
    "latest_timestamp",
    "normalize",
    "parse_timestamp",
    "propagate_embeddings",
    "softmax",
    "temporal_text",
    "tokenize",
    "validate_hypergraph",
    "vector_sum",
    "weighted_edge_embedding",
]


def __getattr__(name: str) -> Any:
    """Import the owning submodule on first access (PEP 562), then cache it in globals."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module  # noqa: PLC0415 — deferred with the rest of the cost

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Every name this package can serve, sorted: those already resolved plus those still lazy.

    Listing a name does not import it — the union is read from the export table, so ``dir()`` stays as cheap
    as a cold ``import memory_kernel.ingest``, which this package's import budget pins. Nothing here may grow
    an import.
    """
    return sorted({*globals(), *_LAZY_EXPORTS})
