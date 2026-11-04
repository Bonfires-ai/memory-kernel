"""BM25, dense, RRF, and embedding propagation."""

from __future__ import annotations

import logging
import math
import os
import re
import time
import unicodedata
from collections import defaultdict
from typing import Any

from rank_bm25 import BM25Okapi

import memory_kernel.stopwords as stopwords_module

from .nlp_models import (
    Config,
    Hypergraph,
    IndexDocument,
    NodeType,
    RetrievalIndex,
)

# Digit-initial tokens (years, counts, prices, flight numbers, "1980s", "Q4")
# are kept: the leading class admits digits, matching the search side's
# ``normalization.retrieval_tokenize``. The old
# letter-only leading class silently deleted every numeral token across the 15
# ingest/recall modules that import this tokenizer. Only the leading class
# changed; casing, splitting, ``'s`` stripping, stopwording and the length
# floor are unchanged.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_'-]*")
_EMBED_BATCH_TOKEN_BUDGET = 35_000
_EMBED_BATCH_COUNT_LIMIT = 1_000
logger = logging.getLogger(__name__)


class RetrievalIndexBuilder:
    """Build an in-memory retrieval index."""

    def __init__(self, *, embedder: Any | None = None, config: Config | None = None) -> None:
        """Hold the live embedder handle and the tuning used while building.

        ``embedder`` is a caller-owned client that will be awaited during :meth:`build`; passing ``None`` builds a
        sparse-only index — no vectors are produced and the dense lane cannot answer against it. ``config``
        defaults to :class:`Config`; only its embedding-propagation fields matter at build time.
        """
        self._embedder = embedder
        self._config = config or Config()

    async def build(
        self,
        *,
        graph: Hypergraph,
        documents: list[IndexDocument],
        known_text_embeddings: dict[str, list[float]] | None = None,
    ) -> RetrievalIndex:
        """Build the index; optionally reuse known embeddings by exact text.

        ``known_text_embeddings`` maps document
        text -> embedding vector (the statement embedding map keyed by
        normalized statement text). Documents whose text matches reuse the
        vector; only the remaining (composite topic/episode/fact) documents
        are embedded. ``None`` (v1) keeps the historical single embed pass
        over every document byte-identical.
        """

        total_start = time.perf_counter()
        tokenize_start = time.perf_counter()
        corpus = [tokenize(doc.text) for doc in documents]
        tokenize_ms = _elapsed_ms(tokenize_start)
        bm25_start = time.perf_counter()
        bm25 = BM25Okapi(corpus) if corpus else None
        documents_by_type = documents_grouped_by_type(documents)
        grouped_tokens: dict[NodeType, list[list[str]]] = defaultdict(list)
        for doc, tokens in zip(documents, corpus, strict=False):
            grouped_tokens[doc.node_type].append(tokens)
        tokenized_corpus_by_type = {node_type: grouped_tokens[node_type] for node_type in documents_by_type}
        bm25_by_type = {
            node_type: BM25Okapi(typed_corpus)
            for node_type, typed_corpus in tokenized_corpus_by_type.items()
            if typed_corpus
        }
        bm25_ms = _elapsed_ms(bm25_start)
        embeddings: dict[str, list[float]] = {}
        embedding_ms = 0
        propagation_ms = 0
        reused_embeddings = 0
        if self._embedder is not None and documents:
            embedding_start = time.perf_counter()
            if known_text_embeddings:
                reused_vectors: dict[int, list[float]] = {}
                pending: list[tuple[int, str]] = []
                for index, doc in enumerate(documents):
                    known_vector = known_text_embeddings.get(doc.text)
                    if known_vector:
                        reused_vectors[index] = [float(value) for value in known_vector]
                    else:
                        pending.append((index, doc.text))
                reused_embeddings = len(reused_vectors)
                if pending:
                    raw_pending = await embed_texts_batched(
                        self._embedder,
                        [text for _index, text in pending],
                        token_budget=_EMBED_BATCH_TOKEN_BUDGET,
                        count_limit=_embedding_batch_count_limit(),
                    )
                    for (index, _text), vec in zip(pending, raw_pending, strict=False):
                        reused_vectors[index] = list(map(float, vec))
                embeddings = {
                    doc.id: reused_vectors[index] for index, doc in enumerate(documents) if index in reused_vectors
                }
            else:
                raw_vectors = await embed_texts_batched(
                    self._embedder,
                    [doc.text for doc in documents],
                    token_budget=_EMBED_BATCH_TOKEN_BUDGET,
                    count_limit=_embedding_batch_count_limit(),
                )
                embeddings = {doc.id: list(map(float, vec)) for doc, vec in zip(documents, raw_vectors, strict=False)}
            embedding_ms = _elapsed_ms(embedding_start)
            propagation_start = time.perf_counter()
            embeddings = propagate_embeddings(graph=graph, embeddings=embeddings, config=self._config)
            propagation_ms = _elapsed_ms(propagation_start)
        dense_start = time.perf_counter()
        normalized_embeddings, dense_documents_by_type = dense_index_by_type(
            documents_by_type=documents_by_type,
            embeddings=embeddings,
        )
        dense_ms = _elapsed_ms(dense_start)
        logger.warning(
            "Retrieval index build docs=%d embeddings=%d reused_statement_embeddings=%d timings_ms=%s",
            len(documents),
            len(embeddings),
            reused_embeddings,
            {
                "tokenize": tokenize_ms,
                "bm25": bm25_ms,
                "embedding": embedding_ms,
                "propagation": propagation_ms,
                "dense": dense_ms,
                "total": _elapsed_ms(total_start),
            },
        )
        return RetrievalIndex(
            documents=documents,
            bm25=bm25,
            tokenized_corpus=corpus,
            embeddings=embeddings,
            documents_by_type=documents_by_type,
            bm25_by_type=bm25_by_type,
            tokenized_corpus_by_type=tokenized_corpus_by_type,
            normalized_embeddings=normalized_embeddings,
            dense_documents_by_type=dense_documents_by_type,
        )


def documents_grouped_by_type(documents: list[IndexDocument]) -> dict[NodeType, list[IndexDocument]]:
    """Split documents into per-tier lists, preserving input order within each tier.

    All three tier keys are always present, empty where nothing matched, so callers can index without a guard.
    """
    grouped: dict[NodeType, list[IndexDocument]] = {"topic": [], "episode": [], "fact": []}
    for doc in documents:
        grouped[doc.node_type].append(doc)
    return grouped


def dense_index_by_type(
    *,
    documents_by_type: dict[NodeType, list[IndexDocument]],
    embeddings: dict[str, list[float]],
) -> tuple[
    dict[str, list[float]],
    dict[NodeType, list[tuple[IndexDocument, list[float]]]],
]:
    """Build the dense side of the index: unit-length vectors by document id, and per-tier document/vector pairs.

    Pre-normalizing here is what lets the dense lane score with a plain dot product at query time. Documents
    without an embedding are skipped rather than zero-filled, so a tier's dense list may be shorter than its
    document list. With no embeddings at all, both structures come back empty with the three tier keys present.
    """
    normalized_embeddings: dict[str, list[float]] = {}
    dense_documents_by_type: dict[NodeType, list[tuple[IndexDocument, list[float]]]] = {
        "topic": [],
        "episode": [],
        "fact": [],
    }
    if not embeddings:
        return normalized_embeddings, dense_documents_by_type
    for node_type, docs in documents_by_type.items():
        typed_dense = dense_documents_by_type.setdefault(node_type, [])
        for doc in docs:
            vec = embeddings.get(doc.id)
            if not vec:
                continue
            normalized = normalize(vec)
            normalized_embeddings[doc.id] = normalized
            typed_dense.append((doc, normalized))
    return normalized_embeddings, dense_documents_by_type


def tokenize(text: str) -> list[str]:
    """The sparse lane's tokenizer: ASCII-fold, split on word characters, lowercase, strip ``'s``, drop stopwords
    and single characters.

    This is the index side of a matched pair — it must stay in step with the tokenizer the search side applies to
    queries, or terms indexed one way become unreachable by queries tokenized the other. Digit-initial tokens
    (years, counts, prices) are kept. Non-ASCII characters are folded away, so a term that folds to nothing
    disappears entirely.
    """
    tokens = [_normalize_token(m.group(0)) for m in _TOKEN_RE.finditer(_fold_ascii(text or ""))]
    stopwords = _english_stopwords()
    return [token for token in tokens if len(token) >= 2 and token not in stopwords]


def _normalize_token(token: str) -> str:
    """Lowercase a token and drop a possessive ``'s``, so "Ben's" and "Ben" index as the same term."""
    lowered = token.lower()
    if lowered.endswith("'s") and len(lowered) > 2:
        return lowered[:-2]
    return lowered


def _fold_ascii(text: str) -> str:
    """Decompose and strip to ASCII, so accented spellings tokenize to the same terms as unaccented ones.

    Characters with no ASCII decomposition are dropped rather than substituted.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _elapsed_ms(start: float) -> int:
    """Whole milliseconds since ``start``, which must be a ``time.perf_counter()`` reading, truncated not rounded."""
    return int((time.perf_counter() - start) * 1000)


#: This used to be its own nltk-first/scikit-learn-fallback implementation, one of four in the
#: tree. Now a re-export of the one kernel-owned accessor (`memory_kernel.stopwords`), which
#: resolves a byte-snapshotted data artifact instead of any live nltk/scikit-learn
#: call -- see that module's docstring. Still private (`_english_stopwords`): the
#: index-side/fact-side tokenizer split this backs is a deliberate variant
#: (see :func:`tokenize`'s module-level neighbors), not something
#: this re-export touches.
_english_stopwords = stopwords_module.english_stopwords


async def embed_texts(embedder: Any, texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` through whichever interface the embedder happens to expose, returning vectors in input
    order.

    Duck-typed by design so any client with ``create_batch``, ``embed`` or ``create`` works: a single text is sent
    through ``create`` when available, otherwise a batch method is preferred and per-text ``create`` is the last
    resort. Sync and async embedders are both accepted — an awaitable return is awaited.

    Raises ``TypeError`` when the embedder exposes none of the three. Provider errors are not caught here.
    """
    if len(texts) == 1 and hasattr(embedder, "create"):
        result = embedder.create(texts[0])
        if hasattr(result, "__await__"):
            result = await result
        return [result]
    if hasattr(embedder, "create_batch"):
        result = embedder.create_batch(texts)
        if hasattr(result, "__await__"):
            result = await result
        return result
    if hasattr(embedder, "embed"):
        result = embedder.embed(texts)
        if hasattr(result, "__await__"):
            result = await result
        return result
    if hasattr(embedder, "create"):
        out = []
        for text in texts:
            result = embedder.create(text)
            if hasattr(result, "__await__"):
                result = await result
            out.append(result)
        return out
    raise TypeError("Embedder must provide create_batch, embed, or create")


async def embed_texts_batched(
    embedder: Any,
    texts: list[str],
    *,
    token_budget: int,
    count_limit: int | None = None,
) -> list[list[float]]:
    """Embed ``texts`` in batches bounded by both an estimated token budget and a count limit.

    The returned vectors are in input order and one per input — the batching is invisible to the caller. Token
    counts are estimated from :func:`tokenize` and are approximate, so ``token_budget`` should leave headroom
    against the provider's true limit; a single text over budget is still sent on its own rather than dropped.
    ``count_limit`` falls back to the module default and is floored at 1. Each batch is logged.
    """
    out: list[list[float]] = []
    batch: list[str] = []
    batch_tokens = 0
    batch_index = 0
    max_count = max(1, count_limit or _EMBED_BATCH_COUNT_LIMIT)
    for text in texts:
        estimated_tokens = max(1, len(tokenize(text)))
        if batch and (len(batch) >= max_count or batch_tokens + estimated_tokens > token_budget):
            batch_index += 1
            logger.warning(
                "Embedding batch %d: texts=%d estimated_tokens=%d",
                batch_index,
                len(batch),
                batch_tokens,
            )
            out.extend(await embed_texts_adaptive(embedder, batch))
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += estimated_tokens
    if batch:
        batch_index += 1
        logger.warning(
            "Embedding batch %d: texts=%d estimated_tokens=%d",
            batch_index,
            len(batch),
            batch_tokens,
        )
        out.extend(await embed_texts_adaptive(embedder, batch))
    return out


async def embed_texts_adaptive(embedder: Any, texts: list[str]) -> list[list[float]]:
    """Embed a batch, halving and retrying recursively when the provider rejects it as too large.

    Only size-rejection errors are retried, and only while more than one text remains — every other failure, and
    a single text the provider still refuses, propagates unchanged. Splits preserve input order, so the result is
    positionally aligned with ``texts`` however many times it was divided. Each split is logged.
    """
    try:
        return await embed_texts(embedder, texts)
    except Exception as exc:
        if len(texts) <= 1 or not _embedding_batch_too_large(exc):
            raise
        midpoint = max(1, len(texts) // 2)
        logger.warning(
            "Embedding provider rejected batch as too large; splitting texts=%d into %d/%d: %s",
            len(texts),
            midpoint,
            len(texts) - midpoint,
            exc,
        )
        left = await embed_texts_adaptive(embedder, texts[:midpoint])
        right = await embed_texts_adaptive(embedder, texts[midpoint:])
        return [*left, *right]


def _embedding_batch_too_large(exc: Exception) -> bool:
    """Whether an embedder error looks like a size rejection worth retrying with a smaller batch.

    Decided by substring match on the exception's type name and message, since providers signal this differently
    and rarely with a distinct exception class. A provider whose wording is not listed is treated as a hard
    failure, which is the safe direction: the error surfaces instead of being retried forever.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "max allowed tokens" in text
        or "too many tokens" in text
        or "lower the number of tokens" in text
        or "batch" in text
        and "too large" in text
    )


def _embedding_batch_count_limit() -> int:
    """Maximum texts per embedding request, from ``MK_RETRIEVAL_EMBED_BATCH_COUNT``, floored at 1.

    Read on every build rather than cached. An unparseable value raises rather than falling back to the default.
    """
    return max(1, int(os.getenv("MK_RETRIEVAL_EMBED_BATCH_COUNT", str(_EMBED_BATCH_COUNT_LIMIT))))


def propagate_embeddings(
    *,
    graph: Hypergraph,
    embeddings: dict[str, list[float]],
    config: Config,
) -> dict[str, list[float]]:
    """Blend hyperedge context into each node's embedding, returning a new mapping.

    Every hyperedge gets a softmax-weighted vector over its member nodes, and each node that participates in at
    least one edge is shifted toward the aggregate of its edges, scaled by ``config.node_emb_update_weight``.
    That is what lets a node be retrieved by wording that only appeared in a sibling it shares an edge with.

    The aggregate is a sum or a mean per ``config.hyperedge_emb_aggregate_type``; with a sum, a node in many
    edges moves further than one in few. The input mapping is not mutated, nodes with no edges are copied
    through unchanged, and the returned vectors are NOT re-normalized — callers that need unit length must
    normalize downstream. Returns the input unchanged when there is nothing to propagate.
    """
    if not embeddings:
        return embeddings
    hyperedge_embeddings: dict[str, list[float]] = {}
    node_to_edges: dict[str, list[str]] = defaultdict(list)

    for edge in graph.fact_hyperedges.values():
        edge_vec = weighted_edge_embedding(edge.weights, embeddings)
        if edge_vec:
            hyperedge_embeddings[edge.id] = edge_vec
            for node_id in edge.relation:
                node_to_edges[node_id].append(edge.id)

    for edge in graph.episode_hyperedges.values():
        edge_vec = weighted_edge_embedding(edge.weights, embeddings)
        if edge_vec:
            hyperedge_embeddings[edge.id] = edge_vec
            for node_id in edge.relation:
                node_to_edges[node_id].append(edge.id)

    out = dict(embeddings)
    alpha = config.node_emb_update_weight
    for node_id, edge_ids in node_to_edges.items():
        original = embeddings.get(node_id)
        if not original:
            continue
        edge_vecs = [hyperedge_embeddings[eid] for eid in edge_ids if eid in hyperedge_embeddings]
        if not edge_vecs:
            continue
        aggregate = vector_sum(edge_vecs)
        if config.hyperedge_emb_aggregate_type == "mean":
            aggregate = [x / float(len(edge_vecs)) for x in aggregate]
        out[node_id] = [a + alpha * b for a, b in zip(original, aggregate, strict=False)]
    return out


def weighted_edge_embedding(weights: dict[str, float], embeddings: dict[str, list[float]]) -> list[float]:
    """A hyperedge's vector: its members' embeddings combined under softmax-normalised weights.

    Members without an embedding are skipped, and the dimension is taken from the first member that survives —
    any later member of a different dimension is skipped rather than raising, so a mixed-dimension corpus
    silently loses contributions instead of failing. Returns an empty list when no member has an embedding.
    """
    items = [
        (node_id, embeddings[node_id], float(weight)) for node_id, weight in weights.items() if node_id in embeddings
    ]
    if not items:
        return []
    soft = softmax([weight for _node_id, _vec, weight in items])
    dim = len(items[0][1])
    out = [0.0] * dim
    for (_node_id, vec, _weight), coeff in zip(items, soft, strict=False):
        if len(vec) != dim:
            continue
        for idx, value in enumerate(vec):
            out[idx] += coeff * value
    return out


def softmax(values: list[float]) -> list[float]:
    """Softmax over ``values``, returning weights that sum to 1 in the input's order.

    The maximum is subtracted before exponentiating, so large inputs do not overflow. An empty input gives an
    empty list.
    """
    if not values:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def normalize(vec: list[float]) -> list[float]:
    """Scale ``vec`` to unit length; a zero vector comes back as zeros of the same length rather than raising."""
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 0:
        return [0.0 for _ in vec]
    return [float(x) / norm for x in vec]


def vector_sum(vectors: list[list[float]]) -> list[float]:
    """Element-wise sum of ``vectors``, sized by the first one.

    Vectors of a different length are skipped rather than raising, so a mismatch loses a contribution silently.
    Returns an empty list for an empty input.
    """
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            continue
        for idx, value in enumerate(vec):
            out[idx] += value
    return out
