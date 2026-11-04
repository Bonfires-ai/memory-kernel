"""Row-shaped parse representation — the Doc-independent half of the parse seam.

``ParsedDoc`` and its row dataclasses are the shape that ``memory_kernel.grammar.nlp_client``
builds from a live spaCy ``Doc``. Everything here is **pure stdlib** — no spaCy import, no
model dependency — so any consumer that walks a ``ParsedDoc`` via these helpers never needs
a live ``Doc`` object.

Row shapes: the ``TokenRow`` baseline {i, text, lemma, pos, tag, dep, head_i, char_start,
char_end, sent_idx} plus four additions — ``whitespace`` (``text_with_ws`` reconstruction
need), ``is_alpha``/``is_punct``/``is_space``/``is_stop``, ``morph`` restricted to
{PronType, Number, Tense} (the subset of the parser's features copied onto the row), and ``ent_type``
(the per-token BIO-stripped entity label). ``NounChunkRow`` carries ``lemma``
(wordnet_semantics.py reads ``chunk.lemma_``). Sents/Ents are unchanged.
Vectors are not carried.

``children``/``subtree``/``ancestors``/slicing are DERIVED from ``head_i``
here, never stored on the row — the same discipline as for a
live spaCy ``Doc`` (dependency edges are the only structural truth).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

#: The morphological features copied from the parser onto each row:
#: PronType (article vs. quantifier/demonstrative determiners),
#: Number (singular/plural) and Tense (past/present/future).
MORPH_FEATURES: tuple[str, ...] = ("PronType", "Number", "Tense")


@dataclass(frozen=True)
class TokenRow:
    """One token's row-shaped features. Immutable, stdlib-only."""

    i: int
    text: str
    lemma: str
    pos: str
    tag: str
    dep: str
    head_i: int
    char_start: int
    char_end: int
    sent_idx: int
    whitespace: str
    is_alpha: bool
    is_punct: bool
    is_space: bool
    is_stop: bool
    morph: Mapping[str, tuple[str, ...]]
    ent_type: str


@dataclass(frozen=True)
class SentRow:
    """One sentence's token/char span."""

    start_token: int
    end_token: int
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class EntRow:
    """One named-entity span (spaCy ``.ents`` — ``ner_source="spacy"``)."""

    text: str
    label: str
    char_start: int
    char_end: int
    start_token: int
    end_token: int


@dataclass(frozen=True)
class NounChunkRow:
    """One noun-chunk span, plus its chunk-level lemma."""

    text: str
    start_token: int
    end_token: int
    lemma: str


@dataclass(frozen=True)
class ParsedDoc:
    """A full parse over one text — the Doc-independent unit everything downstream walks.

    ``stamp`` carries provider/model provenance (e.g. ``"inproc:en_core_web_sm@3.7.0"``)
    so a consumer can assert which model produced the rows; this module only carries it.
    """

    text: str
    tokens: tuple[TokenRow, ...] = field(default_factory=tuple)
    sents: tuple[SentRow, ...] = field(default_factory=tuple)
    ents: tuple[EntRow, ...] = field(default_factory=tuple)
    noun_chunks: tuple[NounChunkRow, ...] = field(default_factory=tuple)
    stamp: str = ""


__all__ = [
    "MORPH_FEATURES",
    "EntRow",
    "NounChunkRow",
    "ParsedDoc",
    "SentRow",
    "TokenRow",
]
