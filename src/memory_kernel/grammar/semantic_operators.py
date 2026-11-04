"""The tense and aspect of a text, read from its parse (``knowledge.temporal_deixis`` asks whether it is future)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

from .dependencies import load_spacy_model
from .nlp_client import doc_to_parsed
from .rows import ParsedDoc

logger = logging.getLogger(__name__)

TENSE_FILE = ("data", "runtime-grammar", "en", "tense.json")


@dataclass(frozen=True)
class SemanticOperatorProfile:
    """``tense_aspects`` holds ``past``, ``present``, ``future``, ``progressive`` and ``perfect``; it is empty when the
    text has no parse (spaCy missing or the parse failed): nothing is guessed from words."""

    tense_aspects: tuple[str, ...] = ()


class SemanticOperatorService:
    def profile(self, text: str) -> SemanticOperatorProfile:
        return _operator_profile_cached(text)


@lru_cache(maxsize=1)
def get_semantic_operator_service() -> SemanticOperatorService:
    return SemanticOperatorService()


@lru_cache(maxsize=1)
def future_auxiliaries() -> frozenset[str]:
    """The English auxiliaries that mark the future (it is not in the morphology): ``data/runtime-grammar/en/tense.json``."""
    text = resources.files("memory_kernel").joinpath(*TENSE_FILE).read_text(encoding="utf-8")
    return frozenset(str(word).lower() for word in json.loads(text)["future_auxiliaries"])


@lru_cache(maxsize=8192)
def _operator_profile_cached(text: str) -> SemanticOperatorProfile:
    parsed = _doc_cached(str(text or ""))
    if parsed is None:
        return SemanticOperatorProfile()
    by_i = {tok.i: tok for tok in parsed.tokens}
    markers: list[str] = []
    for token in parsed.tokens:
        markers.extend(_tense_aspect_markers(token, head=by_i.get(token.head_i)))
    return SemanticOperatorProfile(tense_aspects=_dedupe_tuple(markers))


@lru_cache(maxsize=8192)
def _doc_cached(text: str) -> ParsedDoc | None:
    """Row-shaped parse of ``text``: the shared, lru-cached spaCy pipeline (:func:`_load_nlp`) converted to row shape.
    ANY failure (model load or parse) degrades to ``None``."""
    text = str(text or "")
    try:
        nlp = _load_nlp()
        if nlp is None:
            return None
        return doc_to_parsed(nlp(text), text)
    except Exception as exc:
        # KEEP -- `text` is arbitrary statement text; `_load_nlp() is None` is already an ordinary, non-fatal degrade
        # signal every caller handles.
        logger.warning("spaCy failed to parse text for tense reading, degrading to None doc: %s", exc)
        return None


@lru_cache(maxsize=1)
def _load_nlp() -> Any | None:
    # KEEP -- spaCy MODEL LOAD, not a parse call; the model is a separately-downloaded resource, genuinely absent in
    # some deployments.
    try:
        return load_spacy_model(policy="query")  # a missing extraction model falls back to sm: tense still reads
    except Exception as exc:
        logger.warning("spaCy model unavailable, tense reading degrades to empty: %s", exc)
        return None


def _dedupe_tuple(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _tense_aspect_markers(token: Any, *, head: Any | None = None) -> list[str]:
    text = str(getattr(token, "text", "") or "").lower()
    lemma = str(getattr(token, "lemma", "") or text).lower()
    pos = str(getattr(token, "pos", "") or "")
    dep = str(getattr(token, "dep", "") or "")
    tag = str(getattr(token, "tag", "") or "")
    head_tag = str(getattr(head, "tag", "") or "")
    out: list[str] = []
    if tag == "MD" and (text in future_auxiliaries() or lemma in future_auxiliaries()):
        out.append("future")
    if tag == "VBD":
        out.append("past")
    if tag in {"VBP", "VBZ"}:
        out.append("present")
    if tag == "VBG":
        out.append("progressive")
    if lemma == "have" and dep.startswith("aux") and head_tag == "VBN":
        out.append("perfect")
    if pos == "AUX" and lemma == "be" and head_tag == "VBG":
        out.append("progressive")
    return out


__all__ = [
    "SemanticOperatorProfile",
    "SemanticOperatorService",
    "get_semantic_operator_service",
]
