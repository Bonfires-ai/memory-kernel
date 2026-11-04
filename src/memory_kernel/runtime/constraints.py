"""Slot constraints from tool parameters, checked at plan time (before a tool step is issued).

``check_constraint(value, param)`` answers ``pass`` / ``fail`` / ``unknown`` for one bound tool argument:

1. ``param.kind`` (a store kind label: Person, Organisation, Artifact, Place) against an entity: it passes when the
   kind is among the entity's labels and fails when the entity carries labels and the kind is not among them. An entity
   with no labels, or text (a name the game has not resolved yet), is ``unknown``.
2. ``param.constraint`` (a common-noun concept lemma) against text or a noun phrase: the head lemma of the phrase
   (spaCy: the root of its first noun chunk) passes when it equals the constraint, or when ANY WordNet noun sense of
   the head lies under ANY noun sense of the constraint (hypernym and instance-hypernym closure,
   ``WordNetSemantics.noun_senses_under``); it fails when the head has noun senses and none qualifies. A head with no
   WordNet noun sense (a coinage), a proper-noun head (a name) or an entity is ``unknown``.

Only the kind check (a resolved entity of the wrong store kind) blocks a step; the executor attaches the lexical result
to the issued step (``ToolStep.checks``) and never blocks on it: WordNet is too coarse to veto the world's own objects (a
keyboard is a device, not an instrument), so the game's Resolve decides. Several values (a ``many`` argument) pass when all pass,
fail when one fails, else are unknown. The diagnostic of a ``fail`` reads ``"<text> is not a <constraint>"`` (or
``"... is not a <kind>"``). No word lists: the head is a dependency / POS test, the concept test is WordNet's.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal

from memory_kernel.grammar.wordnet_semantics import get_wordnet_semantics
from memory_kernel.runtime.program import ToolParam

logger = logging.getLogger(__name__)

ConstraintCheck = Literal["pass", "fail", "unknown"]

#: the POS of a proper noun: a name, whose head word says nothing about the concept it falls under
_PROPER_NOUN_POS = "PROPN"


def check_constraint(value: Any, param: ToolParam) -> tuple[ConstraintCheck, str | None]:
    """Check one bound argument against its parameter's kind and constraint (see the module docstring)."""
    if param.kind is None and param.constraint is None:
        return "pass", None
    if isinstance(value, (list, tuple, set, frozenset)):
        results = [check_constraint(item, param) for item in value]
        failed = next((r for r in results if r[0] == "fail"), None)
        if failed is not None:
            return failed
        if results and all(r[0] == "pass" for r in results):
            return "pass", None
        return "unknown", None
    if isinstance(value, Mapping):
        return _check_entity(value, param)
    if isinstance(value, str) and value.strip() and param.constraint is not None:
        return _check_text(value.strip(), param.constraint)
    return "unknown", None


def _check_entity(entity: Mapping[str, Any], param: ToolParam) -> tuple[ConstraintCheck, str | None]:
    if param.kind is None:  # a constraint is a concept of common nouns: an entity is identified, not described
        return "unknown", None
    kinds = entity.get("kinds")
    labels = {str(k).casefold() for k in kinds} if isinstance(kinds, (list, tuple, set, frozenset)) else set()
    if not labels:
        return "unknown", None
    if param.kind.casefold() in labels:
        return "pass", None
    return "fail", f"{entity.get('name') or 'it'} is not a {param.kind}"


def _check_text(text: str, constraint: str) -> tuple[ConstraintCheck, str | None]:
    try:
        head = _head_lemma(text)
    except Exception as exc:  # noqa: BLE001 - any spaCy failure (load, parse): the check is advisory, so it is unknown
        logger.warning("spaCy could not read %r for a constraint check; the result is unknown: %s", text, exc)
        return "unknown", None
    if head is None:
        return "unknown", None
    if head == constraint.casefold():
        return "pass", None
    under = get_wordnet_semantics().noun_senses_under(head, constraint)
    if under is None:
        return "unknown", None
    return ("pass", None) if under else ("fail", f"{text} is not a {constraint}")


def _head_lemma(text: str) -> str | None:
    """The lowercased lemma of the head of the noun phrase ``text``; ``None`` for a name (a proper-noun head) or text
    with no word. spaCy tags a lone lower-case noun ("kettle") a proper noun, so a lower-case phrase tagged that way is
    read again after a determiner, which fixes its reading as a common noun phrase."""
    head = _head(text)
    if head is not None and head.pos_ == _PROPER_NOUN_POS and text[:1].islower():
        head = _head(f"the {text}")
    if head is None or head.pos_ == _PROPER_NOUN_POS or not head.is_alpha:
        return None
    return (head.lemma_ or head.text).casefold()


def _head(text: str) -> Any:
    from memory_kernel.graph.g1_nlp import nlp

    doc = nlp("turn")(text)
    chunk = next(iter(doc.noun_chunks), None)
    return chunk.root if chunk is not None else next((t for t in doc if t.dep_ == "ROOT"), None)


__all__ = ["ConstraintCheck", "check_constraint"]
