# Derived from an internal parser service, file cache.py, source sha256
# 9f1731bb3e2681afee6044d27170779b897cff1787c6fc7797a1f55aee713b21.
# Copied from the origin file; edits: this header, the package-relative imports and a prose-only scrub of docstrings and comments.
"""In-process bounded LRU cache for spaCy parses.

Ported from memory_kernel's ``grammar/parse_cache.py`` (reimplemented here,
not imported — this service must not depend on memory_kernel). spaCy parsing
is deterministic for a fixed model, so a single Doc can safely serve every
caller that asks for the same (model identity, exact text) pair within one
process.

Knobs (see :mod:`.config`):

- ``MK_PARSER_PARSE_CACHE`` (default on): set to a falsy value to disable
  and parse every call fresh.
- ``MK_PARSER_PARSE_CACHE_SIZE`` (default 4096): LRU entry bound.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from .config import get_settings

_LOCK = threading.Lock()
_CACHE: OrderedDict[tuple[str, str], Any] = OrderedDict()
_STATS = {"hits": 0, "misses": 0, "evictions": 0}


def model_key(nlp: Any) -> str:
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
    with _LOCK:
        doc = _CACHE.get(key)
        if doc is not None:
            _CACHE.move_to_end(key)
            _STATS["hits"] += 1
        return doc


def _store(key: tuple[str, str], doc: Any) -> Any:
    with _LOCK:
        _STATS["misses"] += 1
        existing = _CACHE.get(key)
        if existing is not None:
            _CACHE.move_to_end(key)
            return existing
        _CACHE[key] = doc
        limit = get_settings().parse_cache_size
        while len(_CACHE) > limit:
            _CACHE.popitem(last=False)
            _STATS["evictions"] += 1
        return doc


def cached_parse(nlp: Any, text: str) -> Any:
    """Parse ``text`` with ``nlp``, reusing a shared Doc for repeated texts."""

    if not isinstance(text, str) or not get_settings().parse_cache_enabled:
        return nlp(text)
    key = (model_key(nlp), text)
    doc = _lookup(key)
    if doc is not None:
        return doc
    return _store(key, nlp(text))


def cache_stats() -> dict[str, int]:
    with _LOCK:
        return {**_STATS, "size": len(_CACHE)}


def clear_cache() -> None:
    """Drop all cached parses and reset counters (test hygiene)."""

    with _LOCK:
        _CACHE.clear()
        for key in _STATS:
            _STATS[key] = 0


__all__ = ["cache_stats", "cached_parse", "clear_cache", "model_key"]
