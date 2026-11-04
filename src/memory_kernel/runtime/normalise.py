"""The text the runtime grammar reads, and the way back to the surface as written.

``normalise`` parses an utterance with the kernel's spaCy pipeline (``graph.g1_nlp.turn_doc``) and writes one string the
runtime grammar's constructions match against:

* tokens joined by single spaces; whitespace-only tokens (runs of spaces, tabs, newlines) are dropped;
* a verb is written as its lemma between the verb marks, ``VERB_OPEN`` ("⟪", U+27EA) and ``VERB_CLOSE`` ("⟫",
  U+27EB): "published" is "⟪publish⟫". One base verb filler consumes exactly one marked token and the lexicon resolves
  the lemma to the bonfire's rolesets, so the grammar holds no construction per roleset and no inflection list exists
  anywhere. A token is a verb (``is_marked_verb``) when its POS is VERB, or when it is not an AUX and governs an
  ``aux`` dependent: spaCy tags "work" in "Where does Kira Dace work?" as a NOUN heading the auxiliary. An AUX token
  is written as its unmarked lemma ("does" is "do", "is" is "be");
* every token is lowercased (lemmas too). Construction forms are literals plus ``[?]`` only (no character classes in
  authored forms), so lowercase text lets one plain literal match a word whether or not it opens the sentence. Case
  never reaches a slot value: slot strings are cut from the ORIGINAL text, not this one;
* the text is wrapped in two sentinel tokens, ``SENTINEL_START`` ("⟦", U+27E6) first and ``SENTINEL_END`` ("⟧", U+27E7)
  last, so a construction can anchor a slot at either edge of the utterance ("brew {X}") and a root construction spans
  sentinel to sentinel (a turn with no final punctuation still has a right anchor). The sentinels are not tokens of
  the original: a span covering only a sentinel cuts to nothing (``ValueError``). A sentinel or verb-mark character in
  the input is written as U+FFFD, so a turn can never forge an edge or a verb;
* a character outside the Basic Multilingual Plane (an emoji, say), and a lone surrogate (U+D800-U+DFFF), is written
  as U+FFFD. FCG Go refuses the surrogate escape JSON would carry it as, and its offsets would otherwise count UTF-16
  units rather than code points.

``Normalised.spans`` implements the executor's ``SpanText``: it carries the normalised text (``normalised``) and each
token's POS, and ``cut(left, right)`` takes the normalised offsets FCG bound for a slot and returns the original surface
of every token overlapping ``[left, right)``, from the first such token's first character to the last one's last
character, with the whitespace and punctuation as written ("Jean-Paul Vorn's report", "O'Neil"). A cut covering a
marked verb returns the verb as written ("published"). The cut is the NP cut (``np``): leading determiner tokens (POS
``DET``: "the team" is "team") are dropped, unless the span holds nothing else, or the determiner is capitalised as
written and every other token of the span is a proper noun (POS ``PROPN``: "The Brask Group" is a name); the
determiner's meaning is the filler's (``definite``, ``indefinite``), never part of the name. A dropped determiner that
is a quantifier (morph ``PronType=Tot``, or no ``PronType`` at all: neither an article nor a demonstrative; the
pinned English model writes no PronType for "every", "all", "each", "no") marks the cut ``quantified``: a constraint
no keyword applies yet, which the caller must not answer as unconstrained. ``marked_lemma`` reads a marked verb's lemma back out of the
normalised text (the executor's ``lemma-span``): the normaliser is the only lemmatiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

#: written in place of a character outside the Basic Multilingual Plane, or a lone surrogate (see the module docstring)
ASTRAL_STANDIN = "�"
#: the first and last token of every normalised text (see the module docstring)
SENTINEL_START = "⟦"
SENTINEL_END = "⟧"
#: the marks around a verb's lemma (see the module docstring)
VERB_OPEN = "⟪"
VERB_CLOSE = "⟫"


#: the POS of a determiner token, dropped at the head of an NP cut (a POS test, no word list)
DETERMINER_POS = "DET"
#: the POS of a proper noun: a capitalised determiner before nothing else is part of a name
PROPER_NOUN_POS = "PROPN"
#: the PronType values of a determiner that only marks (in)definiteness or points; any other is a quantifier
_MARKING_PRONTYPES = frozenset({"Art", "Dem"})


class NPCut(NamedTuple):
    """An NP cut: the surface text, and whether a quantifying determiner was dropped from its head."""

    text: str
    quantified: bool


@dataclass(frozen=True)
class TokenSpans:
    """Token alignment between the original text and its normalised form: per token, ``(normalised start, normalised
    end, original start, original end)``, in order, and (``pos``, parallel to ``tokens``) its spaCy POS."""

    original: str
    normalised: str
    tokens: tuple[tuple[int, int, int, int], ...]
    pos: tuple[str, ...] = ()
    #: each token's spaCy morph features (``str(token.morph)``), parallel to ``tokens``
    morph: tuple[str, ...] = ()

    @property
    def normalised_length(self) -> int:
        return len(self.normalised)

    def cut(self, left: int, right: int) -> str:
        """The surface text of the NP cut over normalised offsets ``[left, right)`` (``np``)."""
        return self.np(left, right).text

    def np(self, left: int, right: int) -> NPCut:
        """The original surface of the tokens covering normalised offsets ``[left, right)``, leading determiners
        dropped (see the module docstring), and whether a dropped one quantifies. ``ValueError`` for an empty or
        out-of-range span, or one that covers no token (only the single spaces between tokens)."""
        if not (0 <= left < right <= self.normalised_length):
            raise ValueError(
                f"span [{left}, {right}) is empty or outside the normalised text (0..{self.normalised_length})"
            )
        covered = [i for i, t in enumerate(self.tokens) if t[0] < right and t[1] > left]
        if not covered:
            raise ValueError(f"span [{left}, {right}) covers no token")
        head = next((n for n, i in enumerate(covered) if self._pos(i) != DETERMINER_POS), 0)
        dropped, rest = covered[:head], covered[head:]
        if dropped and self._names_a_proper_noun(dropped[0], rest):
            dropped, rest = [], covered
        quantified = any(self._quantifies(i) for i in dropped)
        return NPCut(self.original[self.tokens[rest[0]][2] : self.tokens[rest[-1]][3]], quantified)

    def _names_a_proper_noun(self, determiner: int, rest: list[int]) -> bool:
        """A determiner capitalised as written, before nothing but proper nouns ("The Brask Group")."""
        written = self.original[self.tokens[determiner][2] : self.tokens[determiner][3]]
        return written[:1].isupper() and all(self._pos(i) == PROPER_NOUN_POS for i in rest)

    def _quantifies(self, index: int) -> bool:
        features = dict(f.split("=", 1) for f in self._morph(index).split("|") if "=" in f)
        return features.get("PronType") not in _MARKING_PRONTYPES  # Tot, or none: a quantifier

    def _pos(self, index: int) -> str:
        return self.pos[index] if index < len(self.pos) else ""

    def _morph(self, index: int) -> str:
        return self.morph[index] if index < len(self.morph) else ""


def marked_lemma(normalised: str, left: int, right: int) -> str:
    """The lemma of the one marked verb token ``[left, right)`` falls in (the span may be the lemma alone, as a verb
    slot between the marks binds it, or the whole ``⟪lemma⟫`` token). ``ValueError`` when the span is empty or out of
    range, covers more than one token, or its token is not a marked verb."""
    if not (0 <= left < right <= len(normalised)):
        raise ValueError(f"span [{left}, {right}) is empty or outside the normalised text (0..{len(normalised)})")
    if " " in normalised[left:right]:
        raise ValueError(f"span [{left}, {right}) covers more than one token: {normalised[left:right]!r}")
    start = normalised.rfind(" ", 0, left) + 1
    end = normalised.find(" ", right)
    token = normalised[start : len(normalised) if end < 0 else end]
    if len(token) < 3 or token[0] != VERB_OPEN or token[-1] != VERB_CLOSE:
        raise ValueError(f"span [{left}, {right}) is not a marked verb: {token!r}")
    return token[1:-1]


@dataclass(frozen=True)
class Normalised:
    text: str
    spans: TokenSpans


def is_marked_verb(token) -> bool:
    """A verb the normaliser marks: POS VERB, or a non-AUX token governing an ``aux`` dependent (a POS and dependency
    test, no word list)."""
    if token.pos_ == "AUX" or not token.lemma_:
        return False
    return token.pos_ == "VERB" or any(child.dep_ == "aux" for child in token.children)


def _written(token) -> str:
    if is_marked_verb(token):
        return VERB_OPEN + _sendable(token.lemma_.lower()) + VERB_CLOSE
    form = token.lemma_ if token.pos_ == "AUX" and token.lemma_ else token.text
    return _sendable(form.lower())


def _sendable(text: str) -> str:
    return "".join(ASTRAL_STANDIN if _unsendable(ch) else ch for ch in text)


def _unsendable(ch: str) -> bool:
    code = ord(ch)
    return code > 0xFFFF or 0xD800 <= code <= 0xDFFF or ch in _RESERVED


_RESERVED = frozenset((SENTINEL_START, SENTINEL_END, VERB_OPEN, VERB_CLOSE))


def normalise(text: str) -> Normalised:
    from memory_kernel.graph.g1_nlp import turn_doc

    pieces: list[str] = [SENTINEL_START]
    tokens: list[tuple[int, int, int, int]] = []
    pos: list[str] = []
    morph: list[str] = []
    at = len(SENTINEL_START)
    # spaCy cannot hash a lone surrogate: parse a 1:1 stand-in copy, so every token offset is an offset into ``text``
    parsed = "".join(ASTRAL_STANDIN if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in text)
    for token in turn_doc(parsed):
        if token.is_space:
            continue
        written = _written(token)
        at += 1  # the single joining space
        pieces.append(written)
        tokens.append((at, at + len(written), token.idx, token.idx + len(token.text)))
        pos.append(token.pos_)
        morph.append(str(token.morph))
        at += len(written)
    pieces.append(SENTINEL_END)
    joined = " ".join(pieces)
    return Normalised(joined, TokenSpans(text, joined, tuple(tokens), tuple(pos), tuple(morph)))


__all__ = [
    "ASTRAL_STANDIN",
    "DETERMINER_POS",
    "PROPER_NOUN_POS",
    "NPCut",
    "SENTINEL_END",
    "SENTINEL_START",
    "VERB_CLOSE",
    "VERB_OPEN",
    "Normalised",
    "TokenSpans",
    "is_marked_verb",
    "marked_lemma",
    "normalise",
]
