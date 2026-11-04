"""Person, number and gender of an English pronoun, read from ``data/pronouns.json`` first and from the parse second.

A spaCy token carries them in its morphology (``Person``, ``Number``, ``Gender``, ``Poss``, ``Reflex``), but a model's
morphology on a pronoun is arbitrary ("yours" Plur) and a capture leaf has none (a PTB tag and a lemma). So the person,
number and gender come from the lowercase form's row in the data file whenever it has one: a meaning table, not a
detector; the morphology serves only a form the table lacks. Whether a token IS a pronoun is always the parse's call (the PTB tags ``PRP`` and ``PRP$``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

__all__ = ["Pronoun", "clitic_form", "of_leaf", "of_token", "of_word"]

PRONOUNS_FILE = ("data", "pronouns.json")
_PRONOUN_TAGS = ("PRP", "PRP$")
_NOUN_MODIFIERS = frozenset(
    {"det", "predet", "amod", "compound", "nummod", "poss", "nmod"}
)  # a noun takes them, a pronoun does not


@dataclass(frozen=True)
class Pronoun:
    person: int
    number: str | None = None  # "sing" | "plur" | None (you)
    gender: str | None = None  # "masc" | "fem" | "neut" | None (they, you, I, we)
    possessive: bool = False
    reflexive: bool = False
    independent: bool = False  # "mine", "yours": a possessive that stands without a noun

    @property
    def plural(self) -> bool:
        return self.number == "plur"

    @property
    def ungendered_third(self) -> bool:
        """Third person with no gender (they) or an inanimate one (it): the forms a coreference pass may resolve."""
        return self.person == 3 and self.gender in (None, "neut")


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    return json.loads(resources.files("memory_kernel").joinpath(*PRONOUNS_FILE).read_text(encoding="utf-8"))


def of_word(form: str) -> Pronoun | None:
    """The row for a lowercase pronoun form, or None."""
    row = _data()["pronouns"].get(str(form or "").lower())
    return Pronoun(**row) if row else None


def of_leaf(string: str, lex_class: str | None) -> Pronoun | None:
    """A capture leaf's pronoun, when its PTB tag says pronoun (``PRP``, ``PRP$``); the row supplies the features."""
    return of_word(string) if str(lex_class or "").startswith("PRP") else None


def of_token(token: Any) -> Pronoun | None:
    """A spaCy token's pronoun: the parse says whether it is one (PTB ``PRP``/``PRP$``); person, number and gender come
    from the form's row in the table, and from the morphology only for a form the table lacks."""
    if token.tag_ not in _PRONOUN_TAGS:
        return _independent_possessive(token)
    morph = token.morph
    row = of_word(token.lower_)
    if row is not None:  # the table decides person, number and gender: a model's morph on a pronoun is arbitrary
        return Pronoun(
            person=row.person,
            number=row.number,
            gender=row.gender,
            possessive=bool(morph.get("Poss")) or token.tag_ == "PRP$" or row.possessive,
            reflexive=bool(morph.get("Reflex")) or row.reflexive,
            independent=row.independent,
        )
    person = morph.get("Person")
    if not person:
        return None
    number = (morph.get("Number") or [None])[0]
    gender = (morph.get("Gender") or [None])[0]
    return Pronoun(
        person=int(person[0]),
        number={"Sing": "sing", "Plur": "plur"}.get(number or ""),
        gender={"Masc": "masc", "Fem": "fem", "Neut": "neut"}.get(gender or ""),
        possessive=bool(morph.get("Poss")) or token.tag_ == "PRP$",
        reflexive=bool(morph.get("Reflex")),
    )


def _independent_possessive(token: Any) -> Pronoun | None:
    """ "mine", "yours" and "ours" the tagger reads as a noun or adjective ("a friend of mine"): a pronoun when its form is
    an independent possessive in the table and it stands bare, with no determiner or modifier (a "gold mine" has both)."""
    row = of_word(token.lower_)
    if (
        row is None
        or not row.independent
        or token.tag_ not in ("NN", "NNS", "JJ")
        or any(child.dep_ in _NOUN_MODIFIERS for child in token.children)
    ):
        return None
    return row


def clitic_form(lemma: str, *, plural: bool) -> str | None:
    """The form of the auxiliary a clitic token ('m, 're, 've, 'll, 'd) stands for, by its lemma, for a singular or plural subject."""
    forms = _data()["clitic_forms"].get(str(lemma or "").lower())
    return forms["plural" if plural else "singular"] if forms else None
