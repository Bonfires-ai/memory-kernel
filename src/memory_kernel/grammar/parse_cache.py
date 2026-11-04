"""Shared bounded LRU cache for spaCy parses across grammar/ingest consumers.

Every grammar and ingest call site historically ran a per-statement
``nlp(text)`` call, and the same statement text is parsed at multiple sites
during one extraction run (statement-shape gates, keyword/noun-chunk
extraction, WH-domain features, relation triples, event frames, ...).
spaCy parsing is deterministic for a fixed model, and all consumers in this
codebase treat ``Doc`` objects as read-only (audited: no ``set_extension``
writes, no ``doc._.``/``token._.`` assignment, no ``doc.ents`` assignment,
no retokenization), so one parse can safely serve all call sites.

Knobs:

- ``MK_GRAMMAR_PARSE_CACHE`` (default on): share parses through a
  process-wide LRU keyed by (model identity, exact text). Set to ``0`` to
  restore the legacy parse-per-call behavior.
- ``MK_GRAMMAR_PARSE_CACHE_SIZE`` (default 4096): LRU entry bound.

Thread-safety: the cache dictionary is guarded by a plain lock. The
``nlp(...)`` invocations themselves happen exactly where
the call sites historically ran them (inside whatever stage lock the caller
already holds — e.g. ``grammar_stage_cpu_lock`` under concurrent grammar
extraction), so the existing serialization discipline for the shared spaCy
pipeline instance is preserved.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any

_CACHE_ENV = "MK_GRAMMAR_PARSE_CACHE"
_CACHE_SIZE_ENV = "MK_GRAMMAR_PARSE_CACHE_SIZE"
_DEFAULT_CACHE_SIZE = 4096
_FALSY = {"0", "false", "no", "off"}

_CACHE_LOCK = threading.Lock()
_CACHE: OrderedDict[tuple[str, str], Any] = OrderedDict()
_STATS = {"hits": 0, "misses": 0, "evictions": 0}


def parse_cache_enabled() -> bool:
    """Whether the shared spaCy parse cache is active (default on)."""

    value = str(os.getenv(_CACHE_ENV) or "").strip().lower()
    if not value:
        return True
    return value not in _FALSY


def parse_cache_size() -> int:
    """LRU entry bound for the shared parse cache (default 4096).

    A malformed value raises instead of silently falling back to the default: a mistyped
    deployment knob should fail fast at startup.
    """

    raw = str(os.getenv(_CACHE_SIZE_ENV) or "").strip()
    if not raw:
        return _DEFAULT_CACHE_SIZE
    return max(1, int(raw))


def unwrap_nlp(nlp: Any) -> Any:
    """Return the underlying spaCy pipeline for a possibly wrapped instance."""

    if isinstance(nlp, CachingNlp):
        return nlp._nlp
    return nlp


def _model_key(nlp: Any) -> str:
    """Stable identity for a pipeline so equal models share cached parses."""

    meta = getattr(nlp, "meta", None)
    if isinstance(meta, dict):
        name = str(meta.get("name") or "").strip()
        if name:
            lang = str(meta.get("lang") or "").strip()
            version = str(meta.get("version") or "").strip()
            return f"{lang}_{name}@{version}"
    return f"id:{id(nlp)}"


def _lookup(key: tuple[str, str]) -> Any | None:
    with _CACHE_LOCK:
        doc = _CACHE.get(key)
        if doc is not None:
            _CACHE.move_to_end(key)
            _STATS["hits"] += 1
        return doc


def _store(key: tuple[str, str], doc: Any) -> Any:
    with _CACHE_LOCK:
        _STATS["misses"] += 1
        existing = _CACHE.get(key)
        if existing is not None:
            _CACHE.move_to_end(key)
            return existing
        _CACHE[key] = doc
        limit = parse_cache_size()
        while len(_CACHE) > limit:
            _CACHE.popitem(last=False)
            _STATS["evictions"] += 1
        return doc


def cached_parse(nlp: Any, text: Any) -> Any:
    """Parse ``text`` with ``nlp``, reusing a shared Doc for repeated texts."""

    real = unwrap_nlp(nlp)
    if not isinstance(text, str) or not parse_cache_enabled():
        return real(text)
    key = (_model_key(real), text)
    doc = _lookup(key)
    if doc is not None:
        return doc
    return _store(key, real(text))


class CachingNlp:
    """Transparent pipeline wrapper routing ``nlp(text)`` through the cache.

    Attribute access (``vocab``, ``meta``, ``pipe``, ...) and calls with
    keyword arguments delegate untouched to the wrapped pipeline, so the
    wrapper behaves like the underlying spaCy ``Language`` everywhere except
    that plain single-text calls reuse cached Docs.
    """

    __slots__ = ("_nlp",)

    def __init__(self, nlp: Any) -> None:
        self._nlp = nlp

    def __call__(self, text: Any, **kwargs: Any) -> Any:
        if kwargs:
            return self._nlp(text, **kwargs)
        return cached_parse(self._nlp, text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._nlp, name)

    def __repr__(self) -> str:
        return f"CachingNlp({self._nlp!r})"


def wrap_with_parse_cache(nlp: Any) -> Any:
    """Wrap a pipeline so its plain calls share the parse cache (idempotent)."""

    if nlp is None or isinstance(nlp, CachingNlp):
        return nlp
    return CachingNlp(nlp)


__all__ = [
    "CachingNlp",
    "cached_parse",
    "parse_cache_enabled",
    "parse_cache_size",
    "unwrap_nlp",
    "wrap_with_parse_cache",
]
