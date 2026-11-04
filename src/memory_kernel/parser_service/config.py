# Derived from an internal parser service, file config.py, source sha256
# 55d9820bd0a1fb97e5f6d0d2a9113e3527861da27906054be7059cc1cd4f66f9.
# Copied from the origin file; edits: this header, the package-relative imports, a prose-only scrub of docstrings and
# comments, the environment names renamed to the MK_PARSER_ prefix (the origin prefix is not repeated here), and the
# host/port settings removed (the managed service takes its address from the command line).
"""Environment-driven configuration for the parser service."""

from __future__ import annotations

import os
from dataclasses import dataclass

_FALSY = {"0", "false", "no", "off"}


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in _FALSY


def _int_env(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    return max(1, int(raw))


def _extraction_model() -> str:
    from memory_kernel.grammar.dependencies import spacy_model

    return spacy_model("extraction")


@dataclass(frozen=True)
class Settings:
    """Resolved once at import time; re-read via :func:`get_settings` in tests."""

    spacy_model: str
    benepar_model: str
    parse_cache_enabled: bool
    parse_cache_size: int


def get_settings() -> Settings:
    return Settings(
        spacy_model=os.getenv("MK_PARSER_SPACY_MODEL", "").strip() or _extraction_model(),
        # Matches penelope-interface.lisp's run-penelope-syntactic-parser
        # default (:model "en_benepar") in NAME only — the model this
        # resolves to is benepar's own "benepar_en3", attached to this
        # service's one spaCy pipeline (see registry.get_benepar_pipe).
        benepar_model=os.getenv("MK_PARSER_BENEPAR_MODEL", "benepar_en3").strip() or "benepar_en3",
        parse_cache_enabled=_bool_env("MK_PARSER_PARSE_CACHE", True),
        parse_cache_size=_int_env("MK_PARSER_PARSE_CACHE_SIZE", 4096),
    )
