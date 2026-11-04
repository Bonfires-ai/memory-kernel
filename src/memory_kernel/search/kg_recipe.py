"""Message retrieval over the Graphiti-shaped graph: dense cosine over expanded search keys.

The unit is the message. The index lives in the graph (``kg_index``: ``search_key*`` properties on ``:Episodic``); ``search_messages_v2``
reads it and returns a ``SearchResult``. The recipe is the measured winner on the development conversation (R@10 .737, 119/152): a message's key is its content plus the
plain facts and Event names it supports plus the names of the entities it mentions; the key is embedded (voyage-3
document vectors) and a question vector ranks the keys by cosine. BM25, the label prior, personalized PageRank and a
time boost all measured worse or equal and are gone.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from memory_kernel.graph.kg_writer import KEY_EMBED_DIM, KEY_EMBED_MODEL
from memory_kernel.search.kg_types import MessageHit, SearchResult
from memory_kernel.search.recipe_primitives import StatementEntry, blend_scores, bm25_scores, build_bm25, cosine

if TYPE_CHECKING:
    from memory_kernel.search.kg_index import KgIndex

_ROLE_SUFFIX = re.compile(r"\s*\((?:arg\w*|argm-\w+):[^)]*\)\s*$")


@dataclass(frozen=True)
class KgRecipeConfig:
    """The measured development-conversation recipe: dense cosine over legacy-fragment search keys (R@10 .7420; envelope-rendered
    message line .6959 and the 0.7/0.3 BM25 blend .7369 both measured worse, so ``key_render`` stays "legacy" and
    ``bm25_weight`` 0)."""

    key: str = "expanded"  # "plain" | "expanded"
    list_limit: int = 50
    max_support: int = 3
    key_render: str = (
        "legacy"  # a ``kg_index.KEY_RENDERS`` name (the key text recipe; the index must have been built with it)
    )
    bm25_weight: float = 0.0  # 0 = dense only; w > 0 = (1-w)*minmax(cosine) + w*minmax(BM25 over the same keys)


SEALED_RECIPE = KgRecipeConfig()


def strip_role(fact: str) -> str:
    """A fact as plain text: the trailing ``(argN: filler)`` role tag removed."""
    return _ROLE_SUFFIX.sub("", fact).strip()


def expanded_key(content: str, facts: Sequence[str], events: Sequence[str], entities: Sequence[str]) -> str:
    """content, then plain facts, Event names and entity names; each group deduped and sorted (deterministic)."""
    parts = [content.strip()]
    seen = {parts[0]}
    for group in (facts, events, entities):
        for text in sorted({strip_role(t) for t in group if t}):
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
    return "\n".join(parts)


def _fact_cosine(qvec: list[float], emb: Sequence[float] | None) -> float:
    """Cosine to a fact vector; 0.0 when the fact has none or was embedded at another dimension (graphiti's own)."""
    return cosine(qvec, [float(x) for x in emb]) if emb and len(emb) == len(qvec) else 0.0


def dense_top(qvec: Sequence[float], docs: Mapping[str, Sequence[float]], k: int) -> list[tuple[str, float]]:
    q = list(qvec)
    scored = sorted(((d, cosine(q, list(v))) for d, v in docs.items()), key=lambda kv: (-kv[1], kv[0]))
    return scored[:k]


def blend(dense: Mapping[str, float], bm25: Mapping[str, float], *, bm25_weight: float) -> dict[str, float]:
    """Production's linear fusion: ``(1-w)*minmax(dense) + w*minmax(bm25)``; a doc missing from one lane scores 0 there."""
    return blend_scores(dict(dense), dict(bm25), 1.0 - bm25_weight, bm25_weight)


def _bm25_over_keys(keys: Mapping[str, str], query: str) -> dict[str, float]:
    """Okapi BM25 (``recipe_primitives``) of ``query`` over the search keys, keyed by message uuid."""
    entries = [StatementEntry(u, t, "", "", "", 0) for u, t in sorted(keys.items())]
    return bm25_scores(build_bm25(entries), query)


def rank_recipe(
    idx: KgIndex, *, qvec: Sequence[float], cfg: KgRecipeConfig, query: str = ""
) -> list[tuple[str, float]]:
    """Ranked ``(message uuid, score)``: the top ``cfg.list_limit`` message keys by cosine to the question vector.

    With ``cfg.bm25_weight > 0`` the score is the linear blend of min-max cosine and min-max BM25 of ``query`` over
    the same keys.

    The query dimension is checked against the index first, so a model mismatch is an ``IndexModelMismatch``, never
    a bare ``ValueError`` from the cosine.
    """
    from memory_kernel.search.kg_index import check_query_dim

    check_query_dim(qvec, idx.dim)
    if cfg.bm25_weight <= 0.0:
        return dense_top(qvec, idx.vectors, cfg.list_limit)
    dense = dict(dense_top(qvec, idx.vectors, len(idx.vectors)))
    sparse = _bm25_over_keys({u: idx.keys[u] for u in idx.vectors}, query)
    scores = blend(dense, sparse, bm25_weight=cfg.bm25_weight)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: cfg.list_limit]


async def search_messages_v2(
    driver: Any,
    *,
    group_id: str,
    query: str,
    qvec: Sequence[float],
    shape: str | None = None,
    label: str | None = None,
    weights: Mapping[str, float] | None = None,
    config: KgRecipeConfig | None = None,
    index: KgIndex | None = None,
    k: int | None = None,
    allow_stale: bool = False,
    query_model: str | None = None,
) -> SearchResult:
    """Messages ranked by ``rank_recipe`` over the group's persisted index, as a ``SearchResult``.

    The index is read from the graph with ``kg_index.load_index`` unless a preloaded ``index`` is passed (a caller
    that runs many queries loads once). Only the top ``config.list_limit`` hits are returned (``k`` can only narrow it, never widen past ``list_limit``). A stale or partly unindexed index raises
    ``IndexStale`` unless ``allow_stale`` (then stored vectors score and unindexed messages are absent).
    ``query_model`` is the id of the model that embedded ``qvec``: it is loaded against (and, for a preloaded index,
    compared with) the model the keys were embedded with, and a mismatch raises ``IndexModelMismatch`` even when both
    are 1024-d. ``None`` (callers that predate the argument) leaves only the dimension check.
    ``query``, ``shape``, ``label`` and ``weights`` are accepted for the ``search_fn`` call shape of
    ``kg_reader.build_answer_context`` and unused: every ranking constant lives in ``KgRecipeConfig``.
    The search itself is ``search_index`` (synchronous: an async caller with a preloaded index runs that in a thread).
    """
    from memory_kernel.search import kg_index

    cfg = config or SEALED_RECIPE
    idx = index or await kg_index.load_index(
        driver,
        group_id,
        model=query_model or KEY_EMBED_MODEL,
        dim=KEY_EMBED_DIM,
        key_render=cfg.key_render,
        purpose="query",
    )
    return search_index(idx, qvec=qvec, config=cfg, query=query, k=k, allow_stale=allow_stale, query_model=query_model)


def search_index(
    idx: KgIndex,
    *,
    qvec: Sequence[float],
    config: KgRecipeConfig | None = None,
    query: str = "",
    k: int | None = None,
    allow_stale: bool = False,
    query_model: str | None = None,
) -> SearchResult:
    """``search_messages_v2`` over a loaded index: no I/O, all CPU (a cosine over every key), so an async caller runs
    it in a worker thread (``serve.answer.Server.context``). Same checks, same ranking, same result."""
    from memory_kernel.search import kg_index

    cfg = config or SEALED_RECIPE
    qlist = list(qvec)
    if k is not None:
        cfg = replace(cfg, list_limit=min(k, cfg.list_limit))
    if query_model is not None:
        kg_index.check_query_model(query_model, idx.model)
    if (idx.stale or idx.missing) and not allow_stale:
        raise kg_index.IndexStale(idx.stale, idx.missing)
    ranked = rank_recipe(idx, qvec=qvec, cfg=cfg, query=query)
    items: dict[str, dict[str, Any]] = {}
    hits = []
    for m, score in ranked:
        support = []
        facts = sorted(idx.msg_facts[m], key=lambda f: (-_fact_cosine(qlist, f.get("emb")), f["uuid"]))
        for f in facts[: cfg.max_support]:
            items[f["uuid"]] = {"fact": f["fact"], "asserted": f.get("asserted"), "modality": f.get("modality")}
            support.append(f["uuid"])
        for ev in idx.msg_events[m][: cfg.max_support]:
            items[ev["uuid"]] = {
                "name": ev.get("source_text") or ev["name"],  # structure Events: the sentence, not the frame name
                "asserted": ev.get("asserted"),
                "modality": ev.get("modality"),
                "p_by_role": {},
            }
            support.append(ev["uuid"])
        hits.append(MessageHit(m, score, tuple(support)))
    msgs = {
        m: {
            "uuid": m,
            **{k_: idx.messages[m].get(k_) for k_ in ("message_id", "content", "sender_name", "told_at", "saga")},
            "valid_at": idx.told_at.get(m),
        }
        for m, _ in ranked
    }
    freq: dict[str, int] = defaultdict(int)
    for m, _ in ranked[:10]:
        for e in idx.msg_entities[m]:
            freq[e["uuid"]] += 1
            items[e["uuid"]] = {"name": e["name"]}
    cand = sorted(freq, key=lambda e: (-freq[e], e))
    trace = {e: {"score": float(freq[e])} for e in cand}
    sagas: dict[str, float] = {}
    for h in hits:
        s = msgs[h.message]["saga"]
        if s and (s not in sagas or h.score > sagas[s]):
            sagas[s] = h.score
    return SearchResult(
        hits=hits,
        sagas=sorted(sagas.items(), key=lambda kv: (-kv[1], kv[0])),
        trace=trace,
        items=items,
        messages=msgs,
        lists={"bm25_nodes": cand, "semantic_nodes": []},
    )
