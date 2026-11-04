"""Accessor for the kernel's English stopword list -- THE owner.

WHY THIS MODULE EXISTS. The same ``english_stopwords`` set used to be implemented FOUR times, with TWO
different optional-dependency fallback chains:

* :mod:`memory_kernel.grammar.dependencies` -- nltk, falling back to spaCy's ``STOP_WORDS``.
* a fact-utility module -- nltk, falling back to spaCy's ``STOP_WORDS``. It has been retired; its
  ``english_stopwords`` re-export had already been pointed at this module.
* :mod:`memory_kernel.ingest.retrieval_index` (private ``_english_stopwords``) -- nltk,
  falling back to scikit-learn's ``ENGLISH_STOP_WORDS``.
* :mod:`memory_kernel.primitives` -- nltk, falling back to scikit-learn's ``ENGLISH_STOP_WORDS``,
  carrying a comment claiming a "3-way parity" check that never accounted for the fourth site
  and only ever executed the nltk arm.

Removing the ``nltk`` dependency BEFORE this unification would have
silently promoted two DIFFERENT fallback chains (spaCy's list at two sites, scikit-learn's at
the other two) into production, forking index-side and fact-side tokenization apart with no
test able to catch it -- the classic "widening one site is a no-op with a green suite" failure
this module exists to close off.

THE FIX. Vocabulary lists are standardized DATA, never hardcoded literals, and
never re-derived from an optional model/corpus at runtime: the live nltk arm's output was
snapshotted byte-for-byte into ``src/memory_kernel/data/english_stopwords.json`` (verified to
already equal all four sites' *current* live output -- 198 terms; see the artifact's own header).
This module is the ONE place that reads it.
All four sites above now import :func:`english_stopwords` from here instead of computing their
own; their nltk/spaCy/scikit-learn fallback chains are deleted, not merely unreachable.

WHY A DATA FILE, NOT A PYTHON LITERAL OR A LIVE nltk/spaCy/scikit-learn CALL. Supply-side
vocabulary ships as a DATA ARTIFACT so it has one auditable, versioned, diffable
home, and this module's only job is to load and shape-check it. It also means the kernel's
stopword set no longer depends on ANY optional NLP model or downloaded corpus being present at
runtime -- zero model/corpus dependency, exactly the ownership axis this consolidation moves
onto.

WHY PACKAGE-DATA + ``importlib.resources``, NOT REPO-ROOT-RELATIVE. The artifact
ships under ``src/memory_kernel/data/`` and is
declared in ``[tool.setuptools.package-data]``, resolved via
``importlib.resources.files("memory_kernel")`` so it is found correctly whether
``memory_kernel`` is an editable install or unpacked from a wheel into site-packages with no
repo checkout above it. :data:`STOPWORDS_PATH_ENV` is an OVERRIDE for a regenerated artifact or
a test fixture, never the only way in.

LOUD FAILURE, ALWAYS. A missing, unreadable, non-JSON, shape-malformed, or hash-mismatched
artifact raises :class:`StopwordsError` naming the resolved path -- never ``None``, never an
empty set. A silently empty stopword set would make every downstream filter a no-op rather than
surfacing "the artifact failed to load", which is precisely the confusion the vocabulary-as-data rule exists to rule
out for supply data.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import cast

#: The committed artifact's location inside the installed ``memory_kernel`` package.
#: ``importlib.resources.files`` returns
#: a concrete ``pathlib.Path`` for every topology this package ships in.
DEFAULT_STOPWORDS_PATH: Path = cast(Path, resources.files("memory_kernel") / "data" / "english_stopwords.json")

#: Optional override -- a caller-supplied path always wins over this, and this always wins over
#: :data:`DEFAULT_STOPWORDS_PATH`. Never the only way in (see module docstring).
STOPWORDS_PATH_ENV = "MK_STOPWORDS_FILE"

#: Every top-level key the snapshot job stamps. A missing one means the artifact predates this
#: accessor's contract or was hand-edited -- either way, loud, not a ``.get(..., default)`` silence.
_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {"artifact", "version", "extracted_at", "extracted_from", "count", "sha256", "terms"}
)


class StopwordsError(RuntimeError):
    """Raised when the English stopword artifact cannot be loaded or does not shape up."""


def stopwords_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the stopword artifact path.

    Precedence: explicit ``path``, then :data:`STOPWORDS_PATH_ENV`, then
    :data:`DEFAULT_STOPWORDS_PATH` -- the packaged committed artifact, which is what resolves
    with NO environment configured. Never raises; :func:`english_stopwords` is what checks the
    result exists.
    """

    raw = str(path or os.getenv(STOPWORDS_PATH_ENV) or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_STOPWORDS_PATH


def _load_terms(path: str | os.PathLike[str] | None = None) -> frozenset[str]:
    resolved = stopwords_path(path)
    if not resolved.is_file():
        override_hint = f"set {STOPWORDS_PATH_ENV} to override the packaged default"
        raise StopwordsError(f"english stopwords artifact not found: {resolved} ({override_hint})")
    try:
        raw_text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise StopwordsError(f"english stopwords artifact unreadable: {resolved}") from exc
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise StopwordsError(f"english stopwords artifact is not valid JSON: {resolved}") from exc
    if not isinstance(data, dict):
        raise StopwordsError(f"english stopwords artifact root must be a JSON object: {resolved}")

    missing_keys = sorted(_REQUIRED_TOP_LEVEL_KEYS - set(data))
    if missing_keys:
        raise StopwordsError(f"english stopwords artifact missing keys {missing_keys}: {resolved}")

    terms = data.get("terms")
    if not isinstance(terms, list) or not terms:
        raise StopwordsError(f"english stopwords artifact carries no 'terms' list: {resolved}")
    if not all(isinstance(term, str) and term.strip() for term in terms):
        raise StopwordsError(f"english stopwords artifact 'terms' must all be non-blank strings: {resolved}")

    lowered = sorted(str(term).lower() for term in terms)

    expected_count = data.get("count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count != len(lowered):
        raise StopwordsError(
            f"english stopwords artifact 'count' {expected_count!r} does not match 'terms' length {len(lowered)}: {resolved}"
        )

    expected_sha = data.get("sha256")
    computed_sha = hashlib.sha256("\n".join(lowered).encode("utf-8")).hexdigest()
    if not isinstance(expected_sha, str) or expected_sha != computed_sha:
        raise StopwordsError(
            f"english stopwords artifact 'sha256' mismatch (expected {expected_sha!r}, computed {computed_sha!r}): {resolved}"
        )

    return frozenset(lowered)


@lru_cache(maxsize=1)
def english_stopwords() -> frozenset[str]:
    """The kernel's ONE English stopword set: lowercased, resolved once per process, cached.

    Loads and shape-checks ``src/memory_kernel/data/english_stopwords.json`` (a byte-for-byte
    snapshot of nltk's ``stopwords.words("english")``, verified 2026-08-26 to equal what all
    four former call sites returned live) -- no optional NLP model or corpus is consulted at
    call time. Raises :class:`StopwordsError` loudly on any defect; never returns an empty set
    as a silent degradation.
    """

    return _load_terms()


__all__ = [
    "DEFAULT_STOPWORDS_PATH",
    "STOPWORDS_PATH_ENV",
    "StopwordsError",
    "english_stopwords",
    "stopwords_path",
]
