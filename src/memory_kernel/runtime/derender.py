"""The de-renderer: a turn's text becomes the units the unit grammar consumes (``RuntimeEngine.comprehend_units``).

``derender`` parses the turn with the kernel's spaCy pipeline and emits one ``Unit`` per word, in turn order:

* ``string`` is the word's lowercased token texts joined by single spaces (so a name run is "jean - paul vorn");
  it is the word lowercased (as the grammar's literals match it); ``start``/``end`` are character offsets into
  the ORIGINAL turn (code points). Whitespace-only tokens are dropped;
* a word whose lowercased string is in ``literals`` (the grammar's closed-class registry) is a ``lit`` unit and never
  gets an open-class alternative. The registry is a parameter: it is collected from the construction specs, not a
  list here (rule 18);
* a single token written as a capitalised part, one glued hyphen or apostrophe, a capitalised part ("O'Neil") is a
  ``name`` whatever its POS. A literal wins over this and over every class: a one-token name equal to a literal
  ("Will") is ``lit``;
* a run of proper nouns is ONE ``name`` unit (CLEVR's multiword merge); a hyphen or apostrophe written between two
  proper nouns with no space ("Jean-Paul") is part of the run. The possessive "'s" is its own unit;
* otherwise the class comes from the spaCy POS: NOUN, VERB, ADJ, NUM, ADP (``prep``), DET, PRON, and an AUX or a
  non-AUX token governing an ``aux`` dependent (``is_marked_verb``) is a ``verb``;
* a NOUN/VERB word whose WordNet morphy gives BOTH a noun and a verb reading is ``acat``, ``lemmas`` the
  ``(class, lemma)`` pairs. Any other open-class word carries the spaCy lemma, lowercased;
* a word that has no class (punctuation, adverbs, particles, symbols) or no lemma is a ``lit`` unit of its own string:
  a lexical unit never has an empty ``lemmas``. ``lemmas`` is always a tuple.

``Units.spans`` implements ``SpanText`` over the units' own text (``normalised`` is the unit-token strings joined by
single spaces): ``cut``/``np`` are ``TokenSpans``', so a name span resolves against the original text, a leading
determiner is dropped by POS, and the ``quantified`` flag is kept.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from memory_kernel.runtime.normalise import ASTRAL_STANDIN, TokenSpans, _sendable, is_marked_verb

#: spaCy POS -> the unit word class (a POS test, no word list)
_CLASS_OF_POS = {
    "NOUN": "noun",
    "VERB": "verb",
    "ADJ": "adj",
    "NUM": "num",
    "ADP": "prep",
    "DET": "det",
    "PRON": "pron",
}
#: the characters that may be written glued inside a name ("Jean-Paul", "O'Neil")
GLUE = frozenset("-'’")
#: a capitalised alphabetic part, one glue character, a capitalised alphabetic part (an orthographic test)
_GLUED_NAME = re.compile(r"([^\W\d_]+)([-'’])([^\W\d_]+)")
#: classes whose unit must carry at least one lemma
_LEMMATISED = frozenset({"noun", "verb", "adj", "num", "prep"})


@dataclass(frozen=True)
class Unit:
    id: str
    string: str
    cat: str
    lemmas: tuple = ()
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class Units:
    """The turn's units, in order, and the span text their offsets cut back through."""

    units: tuple[Unit, ...]
    spans: TokenSpans

    def __iter__(self) -> Iterator[Unit]:
        return iter(self.units)

    def __len__(self) -> int:
        return len(self.units)


def _morphy(word: str, pos: str) -> str:
    from nltk.corpus import wordnet as wn

    return wn.morphy(word, getattr(wn, pos)) or ""


def _glued_name(text: str) -> bool:
    m = _GLUED_NAME.fullmatch(text)
    return bool(m) and m.group(1)[0].isupper() and m.group(3)[0].isupper()


def _unit_for(token, lowered: str, literals: frozenset[str]) -> tuple[str, tuple]:
    """The (cat, lemmas) of one single-token word. A literal wins over every class, so a single-token name equal to a
    literal ("Will") is ``lit``."""
    if lowered in literals:
        return "lit", ()
    if _glued_name(token.text):  # "O'Neil" is a name whatever spaCy tags it
        return "name", ()
    cat = "verb" if is_marked_verb(token) or token.pos_ == "AUX" else _CLASS_OF_POS.get(token.pos_, "lit")
    if token.pos_ == "PROPN":
        cat = "name"
    if cat in ("noun", "verb") and token.pos_ != "AUX":  # an auxiliary is never read as a noun ("does": doe)
        noun, verb = _morphy(lowered, "NOUN"), _morphy(lowered, "VERB")
        if noun and verb:
            return "acat", (("noun", noun), ("verb", verb))
    if cat in _LEMMATISED:
        lemma = token.lemma_.lower()
        return (cat, (lemma,)) if lemma else ("lit", ())
    return cat, ()


def derender(text: str, *, literals: frozenset[str]) -> Units:
    from memory_kernel.graph.g1_nlp import turn_doc

    # spaCy cannot hash a lone surrogate: parse a 1:1 stand-in copy, so every offset is an offset into ``text``
    parsed = "".join(ASTRAL_STANDIN if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in text)
    tokens = [t for t in turn_doc(parsed) if not t.is_space]
    units: list[Unit] = []
    cuts: list[tuple[int, int, int, int]] = []
    pos: list[str] = []
    morph: list[str] = []
    pieces: list[str] = []
    at = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        j = i
        if token.pos_ == "PROPN":
            while j + 1 < len(tokens):
                nxt = tokens[j + 1]
                glued = nxt.idx == tokens[j].idx + len(tokens[j].text)
                if nxt.pos_ == "PROPN":
                    j += 1
                elif (
                    glued
                    and nxt.pos_ == "PUNCT"
                    and nxt.text in GLUE
                    and j + 2 < len(tokens)
                    and tokens[j + 2].pos_ == "PROPN"
                    and tokens[j + 2].idx == nxt.idx + len(nxt.text)
                ):
                    j += 2
                else:
                    break
        first, last = tokens[i], tokens[j]
        start, end = first.idx, last.idx + len(last.text)
        lowered = " ".join(t.text.lower() for t in tokens[i : j + 1])
        if j > i:
            cat, lemmas = "name", ()
        else:
            cat, lemmas = _unit_for(first, lowered, literals)
        units.append(Unit(f"w{len(units) + 1}", lowered, cat, lemmas, start, end))
        for t in tokens[i : j + 1]:  # token alignment for the cut: one entry per spaCy token
            written = _sendable(t.text.lower())
            if pieces:
                at += 1
            pieces.append(written)
            cuts.append((at, at + len(written), t.idx, t.idx + len(t.text)))
            pos.append(t.pos_)
            morph.append(str(t.morph))
            at += len(written)
        i = j + 1
    spans = TokenSpans(text, " ".join(pieces), tuple(cuts), tuple(pos), tuple(morph))
    return Units(tuple(units), spans)


__all__ = ["Unit", "Units", "derender"]
