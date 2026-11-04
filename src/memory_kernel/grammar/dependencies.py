"""Lazy accessors for kernel-owned grammar dependencies."""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Any, Literal

logger = logging.getLogger(__name__)


SpacyPurpose = Literal["turn", "extraction", "query"]

#: ``purpose -> (the env var that selects its spaCy model, the default)``. The one place a model is chosen.
_SPACY_SETTINGS: dict[str, tuple[str, str]] = {
    "turn": ("MK_TURN_SPACY_MODEL", "en_core_web_trf"),
    "extraction": ("MK_EXTRACTION_SPACY_MODEL", "en_core_web_trf"),  # =en_core_web_sm opts out
    "query": ("MK_EXTRACTION_SPACY_MODEL", "en_core_web_trf"),  # serve reads graph text with the graph's own parser
}
#: What turn parsing falls back to when its configured model is not installed.
SPACY_FALLBACK_MODEL = "en_core_web_sm"


def spacy_model(purpose: SpacyPurpose) -> str:
    """The spaCy model for ``purpose``: ``turn`` is the runtime's one parser for everything said in a turn (questions,
    commands, taught phrases), ``extraction`` is everything that reads text into the graph (and the parser service),
    ``query`` is serve reading graph text back (same model as ``extraction``, but a model that will not load falls back
    to ``SPACY_FALLBACK_MODEL`` instead of failing; see ``graph.g1_nlp.nlp``). An empty setting means the default."""

    env, default = _SPACY_SETTINGS[purpose]
    return (os.getenv(env) or "").strip() or default


#: The model every bonfire built before 1.4 was parsed with (no model was recorded then).
PRE_1_4_EXTRACTION_MODEL = "en_core_web_sm"


def extraction_model_record() -> dict[str, str | None]:
    """The extraction parser a capture or build seals: the configured model's name and its installed version (None when
    the package is not installed; the version is read from its package metadata, nothing is loaded)."""
    from spacy.util import get_package_version

    name = spacy_model("extraction")
    return {"name": name, "version": get_package_version(name)}


def parser_change_problem(recorded: dict[str, Any] | None, *, what: str, flag: bool = True) -> str | None:
    """The refusal text when the current extraction model's NAME differs from the ``recorded`` one, else None (a version
    difference alone is a note, not a refusal). No record = a ``what`` made before 1.4, parsed with
    ``en_core_web_sm``. Names both models and the ways out (``flag`` False: the env var only, for a step with no flag)."""
    current = extraction_model_record()
    recorded_name = (recorded or {}).get("name") or PRE_1_4_EXTRACTION_MODEL
    if recorded_name == current["name"]:
        return None
    origin = "" if recorded else f" (no model is recorded: this {what} was made before 1.4, with {recorded_name})"
    way_out = f"set MK_EXTRACTION_SPACY_MODEL={recorded_name} to continue with the recorded parser"
    if flag:
        way_out += ", or pass --allow-parser-change to mix parsers knowingly"
    return (
        f"the {what} was parsed with {recorded_name}{origin}, the extraction model now is {current['name']}: {way_out}"
    )


def load_spacy_model(model_name: str | None = None, *, policy: Literal["build", "query"] = "build") -> Any:
    """The spaCy pipeline for grammar operations: ``model_name`` or ``MK_GRAMMAR_SPACY_MODEL`` if given (loaded as
    asked, raising if it cannot be), else the ``extraction`` model under ``policy`` (``build`` raises when it will not load;
    ``query`` falls back to the small model). Always the process's one shared copy of that model
    (``graph.g1_nlp.pipeline``); a failure is raised, never remembered."""

    from memory_kernel.graph.g1_nlp import nlp, pipeline

    name = model_name or os.getenv("MK_GRAMMAR_SPACY_MODEL")
    if name:
        return pipeline(name)
    return nlp("query" if policy == "query" else "extraction")


@lru_cache(maxsize=1)
def _wordnet_module() -> Any:
    """Lazy wordnet corpus accessor -- always succeeds.

    ``from nltk.corpus import wordnet`` returns an
    ``nltk.corpus.util.LazyCorpusLoader`` and cannot fail here even when the
    ``wordnet``/``omw-1.4`` corpus data is missing on disk -- nltk only
    raises (typically ``LookupError``) on first *real* use (a
    ``.synsets()``/``.morphy()`` call), not on import. The real availability
    decision lives in ``wordnet_ready()`` below, the single seam every
    wordnet-backed caller in the kernel checks (or degrades through) instead
    of each inventing its own load guard.
    """
    from nltk.corpus import wordnet as wn  # noqa: PLC0415

    return wn


#: `None` = not yet probed; `True`/`False` after `wordnet_ready()`'s first
#: call. A plain module global (not another `lru_cache`) so tests can reset
#: it directly without reaching into a decorator's cache internals.
_WORDNET_READY: bool | None = None
#: Every WordNet lookup in the process takes this one lock: NLTK reads an uncached synset by seek + readline on one
#: shared data file per part of speech, unlocked, so two threads could read each other's line
#: (``grammar.wordnet_semantics``, ``graph.identity.is_common_noun``, and ``wordnet_ready``'s one probe, which also
#: loads the corpus: NLTK's lazy loader is not thread-safe on its first load). Reentrant: lookups nest, and a lookup
#: may probe.
WORDNET_LOOKUP_LOCK = threading.RLock()


def wordnet_ready() -> bool:
    """One-time, process-wide probe: is the wordnet corpus actually usable?

    The single shared seam for every wordnet-backed module in
    the kernel (`grammar/wordnet_semantics.py`, for example)
    rather than inventing a
    second, independent probe. Probes once with a
    trivial `synsets()` call, logs ONCE at error level naming the missing
    corpus on failure, and caches the verdict for the rest of the process. A
    `LookupError` that somehow occurs *after* a successful probe is a real
    bug and is allowed to raise loud, matching this module's default.
    """
    global _WORDNET_READY
    if _WORDNET_READY is not None:
        return _WORDNET_READY
    with WORDNET_LOOKUP_LOCK:  # the probe is a lookup (and the corpus load): never beside another
        if _WORDNET_READY is not None:
            return _WORDNET_READY
        try:
            _wordnet_module().synsets("test")
        except Exception as exc:
            _WORDNET_READY = False
            logger.error(
                "nltk wordnet corpus unavailable (run nltk.download('wordnet') and "
                "nltk.download('omw-1.4')) -- every wordnet-backed grammar/construction "
                "feature in this process degrades to its safe default for the rest of "
                "this process: %s",
                exc,
            )
            return False
        _WORDNET_READY = True
        return True


def wordnet_module() -> Any:
    """Return the wordnet corpus module if ready, else ``None``.

    The convenience accessor for the common ``wn = wordnet_module(); if wn is
    None: return <default>`` shape already used throughout
    `grammar/wordnet_semantics.py`; equivalent to (and built on)
    ``wordnet_ready()`` above.
    """
    if not wordnet_ready():
        return None
    return _wordnet_module()


__all__ = [
    "SPACY_FALLBACK_MODEL",
    "load_spacy_model",
    "spacy_model",
    "wordnet_module",
    "wordnet_ready",
]
