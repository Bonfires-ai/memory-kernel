# Derived from an internal parser service, file registry.py, source sha256
# 78ca7e1839eafe53bdb5ca1a6d128c0270b17fea98d4bd30a42c6cf0dc8186a9.
# Copied from the origin file; edits: this header, the package-relative imports and a prose-only scrub of docstrings and comments.
"""spaCy model registry — loads the pipeline exactly once per process.

The model name is configurable via ``MK_PARSER_SPACY_MODEL`` (default
``en_core_web_sm``), but this service only ever holds ONE loaded pipeline at
a time: the constructor is memoized on the resolved model name so repeated
calls (across requests, across the two wires) return the identical
``Language`` instance and the identical version stamp.

Benepar (``MK_PARSER_BENEPAR_MODEL``, default ``benepar_en3``) is attached
to that SAME pipeline, not a second one — this service's whole reason to
exist is ONE spaCy authority, not a VUB-style split between an "en" pipeline
and a separate "en_benepar" one. It is attached **disabled** and lazily (on
first ``/syntactic-parser`` call, not at startup): added to ``nlp.pipeline``
but immediately disabled so ``nlp(text)`` — used by every other route via
``cached_parse`` — never pays its ~40ms cost or depends on the model being
present. ``get_benepar_pipe()`` returns the component directly; calling it
on a ``Doc`` (``pipe(doc)``) works regardless of the enabled/disabled flag,
since disabling only affects ``Language.__call__``'s own orchestration.
"""

from __future__ import annotations

import functools
import threading
from typing import Any

import spacy

from . import PIPELINE_VERSION
from .config import get_settings


@functools.lru_cache(maxsize=1)
def _load(model_name: str) -> Any:
    return spacy.load(model_name)


def get_nlp() -> Any:
    """Return the process-wide spaCy pipeline, loading it on first use."""

    return _load(get_settings().spacy_model)


_benepar_lock = threading.Lock()


def get_benepar_pipe() -> Any:
    """Return the benepar pipe, attaching it (disabled, once) on first call."""

    nlp = get_nlp()
    # NOTE: nlp.pipe_names excludes DISABLED components — only
    # nlp.component_names lists everything (spaCy's own distinction). Using
    # pipe_names here would re-attempt add_pipe on every call after the
    # first (since we deliberately disable benepar right after adding it)
    # and blow up with "already exists in pipeline".
    if "benepar" not in nlp.component_names:
        with _benepar_lock:
            if "benepar" not in nlp.component_names:
                import benepar  # noqa: F401 — side effect: registers the "benepar" spaCy factory

                nlp.add_pipe("benepar", config={"model": get_settings().benepar_model})
                nlp.disable_pipe("benepar")
    return nlp.get_pipe("benepar")


def version_stamp() -> dict[str, str]:
    """{model, model_version, pipeline_version} — carried on every response.

    Non-negotiable: without a stamp nothing identifies the analysis, which is why
    lemma disagreement across consumers would be silent. Every
    native and facade response includes this.
    """

    nlp = get_nlp()
    meta = getattr(nlp, "meta", {}) or {}
    # NOTE: nlp.meta["name"] is spaCy's own internal short name (e.g.
    # "core_web_sm" for the "en_core_web_sm" package — lang and name are
    # split apart there). The configured model identifier is the one
    # deployments actually reason about, so it wins here; model_version
    # still comes from the loaded pipeline's own metadata.
    return {
        "model": get_settings().spacy_model,
        "model_version": str(meta.get("version") or "unknown"),
        "pipeline_version": PIPELINE_VERSION,
    }


def reset_registry_for_tests() -> None:
    """Clear the memoized pipeline (test hygiene only — reloading is slow)."""

    _load.cache_clear()


__all__ = ["get_benepar_pipe", "get_nlp", "reset_registry_for_tests", "version_stamp"]
