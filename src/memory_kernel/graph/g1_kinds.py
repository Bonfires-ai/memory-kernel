"""Kinds from evidence (``referent_policy`` structure): one kind per referent, decided over all its mentions.

Evidence about a referent accumulates over every filler that resolves to it (``KindEvidence``), and one kind is decided
in this order (``decide``):

1. **Person**: the speaker, a roster/deixis person or an identity-joined name (today's rule).
2. **Kind nouns the text supplies about it**: the head of a predicative (``PRD``) filler of its frame, for the
   frame's proto-patient (``PPT``) role ("Brask is a **company**", ``frame_evidence``; only a NOUN head: an adjective
   predicate, "Kira is **rich**", supplies nothing), an appositive head, or a common noun before a name head ("the
   **journalist** Milo Dace", ``Mention.kind_nouns``). A name compound of a common head supplies nothing to the name:
   "Zorvex **spyware**" is its own referent and the head is its own class (step 3).
3. **Its own head noun class**, for a common-noun referent with a NOUN head ("the **device**", "the Zorvex
   **spyware**"); never for a name.
4. **Place** from the role profile: a location-tagged (``LOC``) role filled **and** a mention compatible with a
   location (``located``). A single LOC filler of an artifact ("on the device") is decided by step 3.

A noun's kind is the kind its WordNet class (lexicographer file, ``g1_names.KIND_OF_LEXNAME``) gives: that of its
first noun sense, unless the first two noun senses give different kinds (then the noun gives none). A group class
(``noun.group``) gives Organisation only when the noun's first noun sense is ``organization.n.01`` or a hyponym of it
(``g1_names.ORGANISATION_SYNSET``: "team", "company"); another group noun ("meeting", "audience") gives no kind. Kinds
stay language-layer, never card-derived.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from memory_kernel.grammar.wordnet_semantics import get_wordnet_semantics
from memory_kernel.graph.g1_mentions import Mention
from memory_kernel.graph.g1_names import (
    KIND_LABELS,
    KIND_OF_LEXNAME,
    LOCATION_TAG,
    ORGANISATION,
    ORGANISATION_SYNSET,
    PATIENT_TAG,
    PERSON,
    PLACE,
    PREDICATIVE_TAG,
    is_core_role,
)

#: The WordNet class (lexicographer file) of a location: the one ``KIND_OF_LEXNAME`` maps to ``PLACE``.
_LOCATION = next(lexname for lexname, kind in KIND_OF_LEXNAME.items() if kind == PLACE)


@dataclass
class KindEvidence:
    person: bool = False
    supplied: Counter[str] = field(default_factory=Counter)  # kind -> count, from kind nouns
    head: Counter[str] = field(default_factory=Counter)  # kind -> count, from the referent's own head class
    loc_roles: int = 0  # LOC-tagged roles filled
    located: bool = False  # a mention compatible with a location

    def add(self, other: KindEvidence) -> None:
        self.person = self.person or other.person
        self.supplied.update(other.supplied)
        self.head.update(other.head)
        self.loc_roles += other.loc_roles
        self.located = self.located or other.located

    def to_row(self) -> dict[str, Any]:
        """A JSON row (sorted keys, plain dicts)."""
        return {
            "head": dict(sorted(self.head.items())),
            "loc_roles": self.loc_roles,
            "located": self.located,
            "person": self.person,
            "supplied": dict(sorted(self.supplied.items())),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> KindEvidence:
        return cls(
            person=bool(row.get("person")),
            supplied=Counter(row.get("supplied") or {}),
            head=Counter(row.get("head") or {}),
            loc_roles=int(row.get("loc_roles") or 0),
            located=bool(row.get("located")),
        )


def _most(counts: Counter[str]) -> str | None:
    """The most frequent kind; a tie goes to the earlier kind in ``KIND_LABELS``."""
    present = [k for k in KIND_LABELS if counts.get(k, 0) > 0]
    return min(present, key=lambda k: (-counts[k], KIND_LABELS.index(k))) if present else None


def decide(ev: KindEvidence) -> str | None:
    """At most one kind: Person, else the most supplied kind, else the most frequent head kind, else Place when a
    location role meets a location-compatible mention, else none."""
    if ev.person:
        return PERSON
    for counts in (ev.supplied, ev.head):
        if (kind := _most(counts)) is not None:
            return kind
    return PLACE if ev.loc_roles > 0 and ev.located else None


def _senses(term: str) -> tuple[str, ...]:
    return get_wordnet_semantics().noun_sense_lexnames(term.lower())


#: The WordNet class (lexicographer file) ``KIND_OF_LEXNAME`` maps to ``ORGANISATION``: a group noun.
_GROUP = next(lexname for lexname, kind in KIND_OF_LEXNAME.items() if kind == ORGANISATION)


@cache
def noun_kind(lemma: str) -> str | None:
    """The kind a common noun gives: its first noun sense's, unless the first two noun senses give different kinds. A
    group sense gives Organisation only when the noun's first noun sense is or lies below ``ORGANISATION_SYNSET``;
    otherwise it gives no kind."""
    if not lemma:
        return None
    organisation = ORGANISATION_SYNSET in get_wordnet_semantics().first_noun_sense_closure(lemma.lower())
    kinds = {None if s == _GROUP and not organisation else KIND_OF_LEXNAME.get(s) for s in _senses(lemma)[:2]}
    return kinds.pop() if len(kinds) == 1 else None


def located(mention: Mention | None) -> bool:
    """A mention compatible with a location: a name whose first noun sense is a location (its key with spaces as
    underscores, "new_york"), or a common head with a location among its first two noun senses."""
    if mention is None:
        return False
    if mention.name:
        senses = _senses(mention.key.replace(" ", "_")) if mention.key else ()
        return bool(senses) and senses[0] == _LOCATION
    return bool(mention.head) and _LOCATION in _senses(mention.head)[:2]


def filler_evidence(person: bool, mention: Mention | None, tag: str | None) -> KindEvidence:
    """One filler's evidence about its referent: a person, its kind nouns, its own head class (a common noun only),
    and whether it fills a location role with a location-compatible mention."""
    ev = KindEvidence(person=person, loc_roles=int(tag == LOCATION_TAG), located=located(mention))
    if mention is not None:
        if not mention.name and (kind := noun_kind(mention.head)) is not None:
            ev.head[kind] += 1
        for noun in mention.kind_nouns:
            if (kind := noun_kind(noun)) is not None:
                ev.supplied[kind] += 1
    return ev


def frame_evidence(tags: Mapping[str, str], mentions: Mapping[str, Mention | None]) -> dict[str, KindEvidence]:
    """One event's predicative kind nouns: when a role is tagged ``PRD`` and its mention's (common) head has a kind,
    the other core role(s) tagged ``PPT`` (the proto-patient the predicate is said of) are supplied that kind ("Brask
    is a company": be.01 arg1 -> Organisation; "The board appointed Zelda director": appoint.01 arg1 Zelda -> Person,
    never the agent arg0). ``tags`` and ``mentions`` map each role of the event to its function tag and to its
    filler's mention."""
    out: dict[str, KindEvidence] = {}
    for role, tag in tags.items():
        if tag != PREDICATIVE_TAG:
            continue
        mention = mentions.get(role)
        if mention is None or mention.name or (kind := noun_kind(mention.head)) is None:
            continue
        for other in tags:
            if other != role and is_core_role(other) and tags[other] == PATIENT_TAG:
                out.setdefault(other, KindEvidence()).supplied[kind] += 1
    return out
