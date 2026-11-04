"""Referring filter v2 and owner-qualified possessives for G1 fillers.

A filler is non-referring when g1's ``non_identifying`` holds (imported, not copied) or when it fills a predicative
role: the PropBank function tag ``PRD`` (e.g. arg2 of be.01 in "that is a good way to ..."). A predicate nominal
ascribes a property and introduces no referent, whatever its form. Specific indefinites in argument roles stay
referring. No word list and no WordNet facet: the tag is supplied by the caller (``g1_roles.function_tag``).
"""

from __future__ import annotations

from memory_kernel.construction.referents import classify_mention
from memory_kernel.graph import g1_nlp
from memory_kernel.graph.g1_constructions import NONE, PREDICATIVE, role_construction

__all__ = ["is_referring", "owner_key"]


def is_referring(text: str, *, function_tag: str | None = None) -> bool:
    """g1 ``analyse`` referring flag, plus the role-based rules. Same parse, same head and W-tag tests.

    ``function_tag`` is the PropBank function tag of the filler's role (``g1_roles.function_tag``):
    - ``TMP``: a time-role filler is never an entity referent, so it is non-referring.
    - ``PRD``: a predicate nominal is non-referring only when it is a noun phrase (no leading preposition), has no
      identifying determiner, and its head is not name-shaped ("I'm Ana" still refers to a person).
    ``None`` (no frame known) leaves the decision to the form-based tests.
    The tag is read through the role-construction table (``g1_constructions.role_construction``).
    """
    referent = role_construction(function_tag).referent
    if referent == NONE:
        return False
    doc = g1_nlp.nlp()(text)
    leading_adp = bool(len(doc)) and doc[0].pos_ == "ADP"
    start = 0
    while start < len(doc) and doc[start].pos_ == "ADP":
        start += 1
    phrase = doc[start:]
    while start < len(doc) and doc[start].pos_ in g1_nlp.STRIP_LEAD:
        start += 1
    kept = doc[start:]
    if not len(kept):
        return False
    if referent == PREDICATIVE and _predicate_nominal(phrase, kept, leading_adp):
        return False
    return (
        kept.root.pos_ in g1_nlp.REFERRING_HEAD
        and not kept[0].tag_.startswith("W")
        and not g1_nlp.non_identifying(phrase)
    )


def _predicate_nominal(phrase, kept, leading_adp: bool) -> bool:
    """A PRD filler that introduces no referent: a bare or indefinite noun phrase, not a PP, not name-shaped."""
    if leading_adp:
        return False
    if kept.root.pos_ == "PROPN" or kept.root.text[:1].isupper():  # as the projection's short-name test: a name's shape
        return False
    return not any(t.dep_ in ("det", "poss") and g1_nlp.identifying(t) for t in phrase)


def _without_leading_adposition(surface: str):
    """The parse of the surface after its leading preposition(s) ("with my family" -> "my family"), by part of speech."""
    doc = g1_nlp.nlp()(surface)
    start = 0
    while start < len(doc) and doc[start].pos_ == "ADP":
        start += 1
    return doc[start:] if start < len(doc) else doc


def owner_key(surface: str, *, speaker: str, addressee: str | None) -> tuple[str, str] | None:
    """(key, display) for a first- or second-person possessive, qualified by its owner; None otherwise.

    "my family" said by Ana -> ("ana's family", "Ana's family"); a leading preposition is stripped
    first, so "with my family" is the same entity. Third-person and ownerless possessives return None: they stay
    unqualified, and so does a second-person possessive when there is no single addressee (``addressee`` None).
    """
    analysis = classify_mention(_without_leading_adposition(surface))
    if analysis.mention != "possessive":
        return None
    if analysis.owner_hint == "speaker":
        owner = speaker.strip()
    elif analysis.owner_hint == "addressee":
        owner = (addressee or "").strip()
    else:
        return None
    rest = analysis.head_surface.strip()
    if not owner or not rest:
        return None
    return f"{owner.lower()}'s {g1_nlp.lemma_key(rest)}", f"{owner}'s {rest}"
