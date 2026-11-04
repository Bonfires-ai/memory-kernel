"""Host-neutral NLP ingest graph/index models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from memory_kernel.messages import Message as Message

NodeType = Literal["topic", "episode", "fact"]


@dataclass(frozen=True)
class Config:
    """Retrieval tuning knobs for one search.

    ``output_type`` is a three-character on/off mask over the topic, episode and fact tiers in that order, carried
    through to the retrieval backend rather than interpreted here. The ``*_top_k`` fields cap how many hits of
    each tier survive, and ``initial_candidates`` bounds the pre-fusion candidate pool, so it must be at least as
    large as a tier's ``top_k`` for that tier to be filled. Frozen: one config may be shared across searches.
    """

    initial_candidates: int = 100
    topic_top_k: int = 4
    episode_top_k: int = 6
    fact_top_k: int = 30
    output_type: str = "111"
    retrieval_type: Literal["keyword", "vector", "rrf"] = "rrf"
    rrf_k: int = 60
    use_reranker: bool = False
    use_late_reranker: bool = False
    node_emb_update_weight: float = 0.5
    hyperedge_emb_aggregate_type: Literal["mean", "sum"] = "sum"


@dataclass
class Episode:
    """One contiguous stretch of conversation, the middle tier of the hypergraph.

    ``original_data`` holds the raw messages the episode was cut from and is the heaviest field on the record;
    ``summary``, ``keywords`` and ``episode_description`` are the derived text that indexing embeds. ``timestamp``
    may be a ``datetime``, an already-formatted string, or ``None`` when no message in the span carried one.
    """

    id: str
    original_data: list[dict[str, Any]]
    timestamp: datetime | str | None
    summary: str
    user_ids: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    subject: str = ""
    episode_description: str = ""


@dataclass
class Topic:
    """The coarsest tier of the hypergraph: a named theme grouping several episodes.

    ``episode_ids`` are the members and are the only link back down the hierarchy; the topic itself owns no
    messages. ``timestamp`` is the span's representative time and is ``None`` when none of its episodes had one.
    """

    id: str
    title: str
    summary: str
    episode_ids: list[str]
    timestamp: datetime | str | None = None
    user_ids: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class Fact:
    """The finest tier of the hypergraph: one extracted claim, with the episodes and topic it came from.

    ``content`` is the claim on its own; ``temporal`` and ``spatial`` are optional qualifiers kept apart from it
    so callers can index the bare claim and render the qualifiers separately (see :meth:`to_text`).
    ``query_patterns`` are retrieval-time aliases — phrasings that should reach this fact — and are not part of
    the claim. ``episode_ids`` may list more than one episode when the same claim was attested repeatedly.
    """

    id: str
    content: str
    episode_ids: list[str]
    topic_id: str
    confidence: float = 0.8
    temporal: str | None = None
    spatial: str | None = None
    keywords: list[str] = field(default_factory=list)
    query_patterns: list[str] = field(default_factory=list)
    timestamp: datetime | str | None = None
    source_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeHyperedge:
    """A weighted membership edge binding episodes into one topic.

    ``relation`` maps each participating episode id to the role it plays in the topic, and ``weights`` carries a
    strength per the same keys; the two are expected to share a key set.
    """

    id: str
    relation: dict[str, str]
    weights: dict[str, float]
    topic_id: str
    coherence_score: float = 0.8


@dataclass
class FactHyperedge:
    """A weighted membership edge binding facts to the episode they were extracted from.

    ``relation`` maps each participating fact id to its role and ``weights`` carries a strength per the same keys;
    ``extraction_confidence`` is confidence in the extraction as a whole, not in any single fact.
    """

    id: str
    relation: dict[str, str]
    weights: dict[str, float]
    episode_id: str
    extraction_confidence: float = 0.8


@dataclass
class Hypergraph:
    """The whole three-tier store: topics over episodes over facts, plus the hyperedges that bind them.

    Every mapping is keyed by node id, and cross-tier references (``Fact.topic_id``,
    ``Topic.episode_ids``, …) are ids into these dicts rather than object references, so a node can be
    replaced in place without rewriting its neighbours. ``primitive_graph`` is an optional handle to a lower-level
    graph representation and is ``None`` for a hypergraph built by NLP indexing alone.
    """

    topics: dict[str, Topic] = field(default_factory=dict)
    episodes: dict[str, Episode] = field(default_factory=dict)
    facts: dict[str, Fact] = field(default_factory=dict)
    episode_hyperedges: dict[str, EpisodeHyperedge] = field(default_factory=dict)
    fact_hyperedges: dict[str, FactHyperedge] = field(default_factory=dict)
    primitive_graph: Any | None = None

    def stats(self) -> dict[str, int]:
        """Node and edge counts per tier, for diagnostics and index-build reporting."""
        return {
            "topics": len(self.topics),
            "episodes": len(self.episodes),
            "facts": len(self.facts),
            "episode_hyperedges": len(self.episode_hyperedges),
            "fact_hyperedges": len(self.fact_hyperedges),
        }


@dataclass(frozen=True)
class IndexDocument:
    """One retrievable unit: the text that is actually indexed, plus the payload returned on a hit.

    ``text`` is the only field searched — it is what BM25 tokenizes and what the embedder sees. ``data`` is
    carried alongside untouched and is what a hit hands back to the caller; nothing in it is matched against.
    ``id`` is the source node's id, so a hit can be resolved back into the hypergraph tier named by ``node_type``.
    """

    id: str
    node_type: NodeType
    text: str
    data: dict[str, Any]


@dataclass
class RetrievalIndex:
    """Built search structures over a hypergraph: a sparse index, a dense index, and per-tier slices of both.

    ``documents`` is the corpus in build order and ``tokenized_corpus`` is positionally aligned with it, as are
    the ``*_by_type`` slices with each other — code that zips a BM25 score vector against documents depends on
    that alignment. ``embeddings`` is keyed by document id and ``normalized_embeddings`` holds the unit-length
    copies used for cosine scoring; both are empty when the index was built without an embedder, in which case
    only the sparse path can answer. ``dense_documents_by_type`` pre-pairs each document with its vector so the
    dense path never has to re-look-up by id.
    """

    documents: list[IndexDocument]
    bm25: Any | None = None
    tokenized_corpus: list[list[str]] = field(default_factory=list)
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    documents_by_type: dict[NodeType, list[IndexDocument]] = field(default_factory=dict)
    bm25_by_type: dict[NodeType, Any] = field(default_factory=dict)
    tokenized_corpus_by_type: dict[NodeType, list[list[str]]] = field(default_factory=dict)
    normalized_embeddings: dict[str, list[float]] = field(default_factory=dict)
    dense_documents_by_type: dict[NodeType, list[tuple[IndexDocument, list[float]]]] = field(default_factory=dict)


__all__ = [
    "EpisodeHyperedge",
    "FactHyperedge",
    "Config",
    "Episode",
    "Fact",
    "Hypergraph",
    "IndexDocument",
    "Message",
    "RetrievalIndex",
    "Topic",
    "NodeType",
]
