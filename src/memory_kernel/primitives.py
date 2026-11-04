"""The kernel's lowest layer: text, scalar and identity primitives — ONE home.

WHY THIS MODULE EXISTS. Tokenize/dedupe/scalar-coercion/identity were spread across **six
competing homes** (`normalization`, `ingest/nlp_runtime_utils`, an ontology helper module,
`ingest/retrieval_index`, `search/recipe_primitives`, `grammar/dependencies`) — and
the copies were **not equivalent**. Several tokenizers under one name, `dedupe`
emitting the literal string ``"None"`` in two of three homes, thirteen copies of
the same scalar coercion under nine names. Every later file split would otherwise
have inherited a partial, uncoordinated merge of the same six homes.

This module is a LEAF: it imports stdlib only, plus :mod:`memory_kernel.stopwords`
(itself stdlib-only — see its docstring) for :func:`english_stopwords`, so anything
may import it and nothing here may import back. It used to import the optional
nltk/scikit-learn stopword
corpora directly; it now delegates to the one kernel-owned accessor instead, so
the kernel's stopword set has ZERO runtime dependency on any optional NLP model
or corpus.

## The tokenizers are NAMED VARIANTS, not one function

They are genuinely different tools and each keeps the callers it has:

* :func:`retrieval_tokenize` — the one most modules use: no stemming,
  strips ``'s``, stopword filter, ``len >= 2`` floor, **keeps digit-initial
  tokens** (years, counts, prices, flight numbers).
* :func:`bm25_tokenize` — the documented BM25 contract from
  `search/recipe_primitives`: strip non-``[a-z0-9\\s]``, split. No ASCII fold, no
  stopwords, no stemming. Changing it changes BM25 df/idf.

**`ingest/retrieval_index.tokenize` is deliberately NOT here.** It is a
third variant whose regex requires a letter first, so it deletes every
digit-initial token that :func:`retrieval_tokenize` keeps; unifying it re-keys stored keyword/ngram/
anchor fields across the ingest/recall modules. That is a decision with
a pinned eval attached, and it stays where it is until then.

## `dedupe` is the UNION OF FIXES — no prior copy had it

The three former bodies diverged on **three** axes, and each home had one of the
two fixes:

===================  ==============================  ==============================
axis                 `normalization`                 `nlp_runtime_utils`
===================  ==============================  ==============================
``None`` / falsy     ``str(value or "")`` — DROPS    ``str(value)`` — LEAKS the
                     ``None``/``0``/``False``        literals ``"None"``/``"0"``/
                                                     ``"False"`` into the output
internal whitespace  NOT collapsed — ``"New   York"``  collapsed — the two fold
                     and ``"New York"`` are two rows   into one row
===================  ==============================  ==============================

Latest-wins selects the WRONG copy here: the newest body (`nlp_runtime_utils`)
is the one carrying the leak. Canonical is the
union — the falsy guard **and** the whitespace collapse, unconditionally. Both
directions are behaviour changes at some call site, and both are the intended
ones; see the module's tests for the pinned axes.

## Everything else here is an EXACT clone consolidation

:func:`as_float` (6 byte-identical copies under 4 names), :func:`as_int` (2),
:func:`bounded` (2), :func:`clean` (3), :func:`string_list` (3 — two orderings,
executed equal), :func:`stable_id` (2 spellings, executed equal). Zero
behavioural delta by construction, pinned by exact-output tests.

`as_float` is LOUD, and deliberately: an absent field (``None``) degrades to the
default, a **present but malformed** value raises. That is the kernel's rule for
deployment values and this is the canonical copy of it — there is no softer variant to fork
toward.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

import memory_kernel.stopwords as _stopwords_module

#: The token universe of both stemming-free and retrieval variants: alphanumeric-initial, then alphanumerics and ``'_-``.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
#: The BM25 variant's: everything that is not a lowercase alphanumeric or whitespace becomes a space.
_BM25_TOKEN_STRIP = re.compile(r"[^a-z0-9\s]")
#: Edge punctuation the retrieval variant strips off a matched token.
_EDGE_PUNCTUATION_RE = re.compile(r"(^['_-]+|['_-]+$)")
#: Any whitespace run — :func:`clean` and :func:`dedupe` collapse it to one space.
_WHITESPACE_RUN_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------


def fold_ascii(text: str) -> str:
    """NFKD-fold to bare ASCII: ``Zoë`` -> ``Zoe``, ``Café`` -> ``Cafe``."""

    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_retrieval_token(token: str) -> str:
    """The retrieval per-token shape: lowercase, strip edge ``'_-``, drop a trailing ``'s``."""

    lowered = _EDGE_PUNCTUATION_RE.sub("", token.lower())
    if lowered.endswith("'s") and len(lowered) > 2:
        return lowered[:-2]
    return lowered


def retrieval_tokenize(text: str) -> list[str]:
    """Retrieval — no stemming, stopword-filtered, ``len >= 2``, keeps digit-initial tokens."""

    tokens = [normalize_retrieval_token(match.group(0)) for match in _TOKEN_RE.finditer(fold_ascii(text or ""))]
    stopwords = english_stopwords()
    return [token for token in tokens if len(token) >= 2 and token not in stopwords]


def bm25_tokenize(text: str) -> list[str]:
    """The BM25 contract: plain lowercase tokens, no fold, no stopword strip.

    Kept byte-for-byte from `search/recipe_primitives`, whose golden parity fixture
    was exported from the live TS run.
    """

    return [token for token in _BM25_TOKEN_STRIP.sub(" ", text.lower()).split() if token]


#: This used to be its own nltk-first/scikit-learn-fallback implementation, one of FOUR in the
#: tree (with `grammar/dependencies.py`, a fact-utility module and
#: `ingest/retrieval_index.py`, the latter two falling back to spaCy/scikit-learn respectively)
#: behind a comment claiming a "3-way parity" that never accounted for the fourth site. All four
#: now import the same cached accessor from :mod:`memory_kernel.stopwords`, which resolves a
#: byte-snapshotted data artifact (equal to every site's live nltk output —
#: 198 terms) instead of any live nltk/spaCy/scikit-learn call. Re-exported here, not
#: reimplemented, so the identity `normalization.english_stopwords is primitives.english_stopwords`
#: still holds.
english_stopwords = _stopwords_module.english_stopwords


def dedupe(values: Iterable[Any]) -> list[str]:
    """Case-insensitive, order-preserving string dedup — the UNION OF FIXES.

    ``None``, ``0``, ``False`` and ``[]`` are DROPPED (they were leaking as the
    literals ``"None"``/``"0"``/``"False"`` out of two of the three former homes),
    and internal whitespace is collapsed (``"New   York"`` and ``"New York"`` are
    one entry, not two). See the module docstring for the axis
    table and which home had which half.
    """

    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _WHITESPACE_RUN_RE.sub(" ", str(value or "").strip())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def string_list(value: Any) -> list[str]:
    """A ``str`` or a ``list`` to a list of non-empty stripped strings; anything else ``[]``."""

    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def clean(value: Any) -> str:
    """One trimmed line: whitespace runs collapsed, ends stripped, ``None`` -> ``""``."""

    return _WHITESPACE_RUN_RE.sub(" ", str(value or "")).strip()


# ---------------------------------------------------------------------------
# scalar
# ---------------------------------------------------------------------------


def as_float(value: Any, default: float) -> float:
    """Coerce ``value`` to ``float``; ``None`` (the absent-field shape) uses ``default``.

    A present-but-malformed value (unlike an absent one) raises: this
    canonical coercion helper matches every other ``as_float``/``_float``-shaped
    converter across the kernel, all of them loud.
    """

    if value is None:
        return default
    return float(value)


def as_int(value: Any, default: int) -> int:
    """Same contract as :func:`as_float`: absent degrades, malformed raises."""

    if value is None:
        return default
    return int(value)


def bounded(value: float, default: float) -> float:
    """Clamp-to-default over the unit interval: outside ``[0, 1]`` yields ``default``."""

    if value < 0.0 or value > 1.0:
        return default
    return value


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def stable_id(*parts: Any) -> str:
    """A deterministic UUID-shaped id over ``|``-joined parts (sha256, first 32 hex).

    The two former spellings — this hand-formatted ``{8}-{4}-{4}-{4}-{12}`` and
    `nlp_runtime_utils`' ``str(uuid.UUID(digest[:32]))`` — were executed against
    each other and are equal for every input; pinned in the tests.

    **The argument list of any caller is frozen.** These ids are persisted
    identity: reordering, adding a part, or changing a part's normalization
    silently re-keys every stored row and breaks aggregation against rows already
    written.
    """

    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


__all__ = [
    "as_float",
    "as_int",
    "bm25_tokenize",
    "bounded",
    "clean",
    "dedupe",
    "english_stopwords",
    "fold_ascii",
    "normalize_retrieval_token",
    "retrieval_tokenize",
    "stable_id",
    "string_list",
]
