"""Python transcription of the run-004 retrieval primitives.

Direct, function-for-function port of
the benchmark harness's TypeScript retrieval provider (and its `cosine` from
the embedding client) — the kernel-native search recipe.
Skips `lanePBoost`, `Aggregates`, and everything caption-prompt related
(`buildAnswerPromptV2/V3`).

Golden parity was checked against fixtures that were exported from the live TS run (run-004).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from memory_kernel.primitives import bm25_tokenize as tokenize

K1 = 1.2
B = 0.75


@dataclass
class StatementEntry:
    """One retrievable statement: the unit the recipe scores, ranks and returns.

    `hash` is the statement's stable identity — every score map, pool and result list in this module is
    keyed by it. `session` plus `session_index` place the statement in its conversation, which is what
    reply expansion walks to reach the following turn. `construct_ids` are the construction labels the
    grammar attached to the utterance, and may be empty for statements no construction matched.
    """

    hash: str
    utterance: str
    ts: str
    actor_id: str
    session: str
    session_index: int
    construct_ids: list[str] = field(default_factory=list)
    #: Every correlation id whose utterance hashed to this statement. The statement unit
    #: is the UTTERANCE, so one row can carry several. Edge scoring inverts
    #: `stable_uuid(correlation_id, construct_id)` over the cross product of this list and
    #: `construct_ids` to map a graph edge back to the statement that produced it.
    #: EMPTY on corpora published before 2026-08-15 — readers must degrade, not fail.
    correlation_ids: list[str] = field(default_factory=list)
    statement_provenance: list[dict[str, Any]] = field(default_factory=list)
    #: The envelope-rendered form of `utterance` ("I went there last week" ->
    #: "Ben went there the week of 2023-06-26"), read from the corpus row's
    #: `utterance_rendered` field when
    #: the host serves rendered text. NEVER computed here: the render is
    #: produced once, upstream, by `memory_kernel.envelope.render`. Empty means
    #: "no render was published for this row", and every reader falls back to
    #: `utterance` through :func:`served_text` — which is why an already-rendered
    #: legacy corpus (the July golden fixture) can never be double-rendered.
    rendered: str = ""


def served_text(entry: object) -> str:
    """The text a statement is SERVED as: `rendered` when the corpus published one, else the raw `utterance`.

    The ONE accessor for the raw/rendered choice. The BM25 index (:func:`build_bm25`)
    reads it, and no surface re-derives the fallback, so a rendered answer body is never
    scored by a lexical lane that still tokenizes the raw first-person surface.

    `entry` is `object`, not `StatementEntry`: both fields are read with `getattr`, so any
    statement-shaped row works, and a row missing either degrades to `""` rather than raising.
    """

    return str(getattr(entry, "rendered", "") or "") or str(getattr(entry, "utterance", "") or "")


# ---------- BM25 (Okapi, k1=1.2 b=0.75; plain lowercase tokens, no stopword strip) ----------
#
# `tokenize` is `primitives.bm25_tokenize`, imported under its historical local name
# so every call site here and in `search/recipe` is untouched. The body moved
# verbatim; the golden-parity fixture below is what pins that.


@dataclass
class Bm25Index:
    """Precomputed Okapi BM25 statistics over a statement corpus.

    `doc_tokens` maps statement hash to term frequencies, `doc_length` to token count, `df` maps term to
    document frequency, and `avg_length` / `count` are the corpus-wide normalizers. The index is a pure
    snapshot: it is built once per corpus and never mutated by scoring.
    """

    doc_tokens: dict[str, dict[str, int]]
    doc_length: dict[str, int]
    df: dict[str, int]
    avg_length: float
    count: int


def build_bm25(statements: list[StatementEntry]) -> Bm25Index:
    """Build the BM25 index over the statements' SERVED text (:func:`served_text`).

    Tokenization is plain lowercase with no stopword removal, so common words stay in the vocabulary and
    are discounted by IDF rather than dropped. `avg_length` divides by at least 1, so an empty corpus
    yields a usable index that simply scores nothing.

    Indexing `served_text` rather than `utterance` is what keeps the sparse lane in the same register as
    the answer body: with the render armed, the answerer reads "Ben went ..." while a raw index would
    still be scoring "I went ...". With no render published the two are the same string, so this is
    byte-identical to the pre-render index.
    """
    doc_tokens: dict[str, dict[str, int]] = {}
    doc_length: dict[str, int] = {}
    df: dict[str, int] = {}
    total = 0
    for statement in statements:
        tokens = tokenize(served_text(statement))
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        doc_tokens[statement.hash] = tf
        doc_length[statement.hash] = len(tokens)
        total += len(tokens)
        for term in tf:
            df[term] = df.get(term, 0) + 1
    return Bm25Index(
        doc_tokens=doc_tokens,
        doc_length=doc_length,
        df=df,
        avg_length=total / max(len(statements), 1),
        count=len(statements),
    )


def bm25_scores(index: Bm25Index, query: str) -> dict[str, float]:
    """Score the corpus against a raw query string, every distinct query term weighted equally.

    Duplicate query terms are collapsed, so repeating a word in the question does not amplify it. Only
    statements matching at least one term appear in the result; the rest are absent rather than zero.
    """
    terms = list(dict.fromkeys(tokenize(query)))
    return bm25_scores_weighted(index, {term: 1.0 for term in terms})


def bm25_scores_weighted(index: Bm25Index, term_weights: dict[str, float]) -> dict[str, float]:
    """Score the corpus against per-term weights, summing each term's weighted BM25 contribution.

    Terms with a non-positive weight and terms absent from the corpus vocabulary contribute nothing.
    Statements matching no weighted term are omitted from the returned map entirely.
    """
    scores: dict[str, float] = {}
    for term, weight in term_weights.items():
        if weight <= 0:
            continue
        df = index.df.get(term)
        if not df:
            continue
        idf = math.log(1 + (index.count - df + 0.5) / (df + 0.5))
        for hash_, tf in index.doc_tokens.items():
            freq = tf.get(term)
            if not freq:
                continue
            length = index.doc_length.get(hash_, 0)
            denom = freq + K1 * (1 - B + (B * length) / index.avg_length)
            scores[hash_] = scores.get(hash_, 0.0) + weight * idf * ((freq * (K1 + 1)) / denom)
    return scores


# ---------- dense + blend ----------


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 when either has zero magnitude.

    Raises ValueError if the two vectors differ in length — dimensions are validated when embeddings are
    loaded, so a mismatch here is a caller bug rather than a condition to absorb.
    """
    dot = 0.0
    na = 0.0
    nb = 0.0
    # dims are validated at artifact load (load_recipe_artifacts); a mismatch here
    # is a bug, not a runtime condition to tolerate, so fail loud instead of the
    # TS reference's truncate-on-shorter-length behavior.
    for av, bv in zip(a, b, strict=True):
        dot += av * bv
        na += av * av
        nb += bv * bv
    denom = math.sqrt(na) * math.sqrt(nb)
    return 0.0 if denom == 0 else dot / denom


def min_max(scores: dict[str, float]) -> dict[str, float]:
    """Rescale a score map onto [0, 1] so lanes with different natural ranges can be blended.

    An empty input yields an empty map. When every score is identical the range is zero and all keys are
    mapped to 1.0, which keeps a degenerate lane fully weighted instead of erasing it.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    rng = hi - lo
    return {key: (1.0 if rng == 0 else (value - lo) / rng) for key, value in scores.items()}


def blend_scores(
    dense: dict[str, float], sparse: dict[str, float], w_dense: float, w_sparse: float
) -> dict[str, float]:
    """Combine the dense and sparse lanes into one score map after normalizing each onto [0, 1].

    The result is the union of both key sets; a statement missing from one lane contributes 0.0 for that
    lane rather than being dropped. Key order follows dense keys first, then sparse-only keys.
    """
    d = min_max(dense)
    s = min_max(sparse)
    out: dict[str, float] = {}
    for key in dict.fromkeys([*d.keys(), *s.keys()]):
        out[key] = w_dense * d.get(key, 0.0) + w_sparse * s.get(key, 0.0)
    return out


# ---------- temporal gate (structural: explicit month/year tokens only) ----------


# ---------- reply expansion ----------
