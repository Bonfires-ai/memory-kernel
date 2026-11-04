"""One fact per proposition: a G1 event rendered as one English sentence.

Policy (b), mode "b": the assembled clause (predicate + fillers in span order, referent names, resolved time) is used
only when a structural check passes: every non-clausal dependent of the predicate is covered by the predicate or a role
span, and no word repeats. The assembled text then goes through the envelope renderer, which resolves person deixis
(I/me/my/you), inflection and any date still written as a relative phrase. Every other case (interrogative mood, no
anchor, a non-finite or verbless anchor, a relabelled agent, an unclean assembly) is the envelope-rendered sentence.
Mode "c" is the sentence everywhere, for comparison.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from spacy.tokens import Doc, Token

from memory_kernel.envelope.render import render as envelope_render
from memory_kernel.graph import g1_nlp
from memory_kernel.graph.g1_names import TIME_ROLE

#: Dependency labels of the children that belong to the predicate (auxiliary, passive, negation, copula).
_PREDICATE_DEPS = frozenset({"aux", "auxpass", "neg", "cop"})
#: Predicate heads: a verb anchor, or a copula heading a complement (acomp/attr).
_VERB_POS = frozenset({"VERB", "AUX"})
_COMPLEMENT_DEPS = frozenset({"acomp", "attr"})
#: A nominal predicate (an identity statement's complement): a noun or a name.
_NOMINAL_POS = frozenset({"NOUN", "PROPN"})
#: Finite tags: a predicate without one is non-finite (gerund, infinitive, participle) and is no assembled proposition.
_FINITE_TAGS = frozenset({"VBD", "VBZ", "VBP", "MD"})
#: Dependents of the predicate that need no coverage: clausal, coordinating and discourse material.
_UNCOVERED_DEPS = frozenset(
    {"conj", "cc", "advcl", "parataxis", "punct", "discourse", "intj", "mark", "dep", "npadvmod", "advmod"}
)
_DATE = re.compile(r"\d{4}-W\d{2}|\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|\d{4}")
_REPEATED_WORD = re.compile(r"\b(\w+) \1\b", re.IGNORECASE)
MODES = ("b", "c")


def _rejoin(tokens: Sequence[str]) -> str:
    """The unit text rebuilt from its tokens: a clitic ("'m", "n't") attaches to the word before it."""
    out = ""
    for tok in tokens:
        attach = tok[:1] in ("'", "\u2019") or tok == "n't"
        out += tok if (attach or not out) else " " + tok
    return out


@lru_cache(maxsize=4096)
def _parse_tokens(tokens: tuple[str, ...]) -> Doc:
    """The spaCy doc of a pre-tokenized unit, for callers that hold no parse (the projection passes its own ``doc``)."""
    nlp = g1_nlp.nlp()
    doc = nlp(_rejoin(tokens))
    if [t.text for t in doc] == list(tokens):
        return doc
    with g1_nlp.PARSE_LOCK:  # a parse, as ``nlp(text)`` is (``g1_nlp.PARSE_LOCK``): the doc and its components
        doc = Doc(nlp.vocab, words=list(tokens))
        for _, component in nlp.pipeline:
            doc = component(doc)
    return doc


def _is_clitic(tok: Token) -> bool:
    return tok.text[:1] in ("'", "\u2019") or tok.text.lower() == "n't"


def _is_first_or_second_person(tok: Token) -> bool:
    """A personal pronoun of the first or second person, by spaCy morphology."""
    morph = tok.morph.to_dict()
    return morph.get("PronType") == "Prs" and morph.get("Person") in ("1", "2")


def _surface(tok: Token) -> str:
    """The surface form of one token: a clitic takes spaCy's norm where it has one ("n't" -> "not") and stays as
    written otherwise ("'s", "'d", "'m")."""
    if _is_clitic(tok) and tok.norm_ != tok.text:
        return tok.norm_
    return tok.text


def _join(words: Sequence[str]) -> str:
    """Words joined with spaces, except that a clitic ("'s", "n't") attaches to the word before it."""
    out = ""
    for word in words:
        attach = word[:1] in ("'", "\u2019")
        out += word if (attach or not out) else " " + word
    return out


def _predicate_root(anchor: Token) -> Token | None:
    """The verb the predicate hangs on: the anchor when it is a verb, else the copula heading an adjective or noun
    complement ("It's important": the copula is the head). None when there is no verb to render (a noun anchor)."""
    if anchor.pos_ in _VERB_POS:
        return anchor
    if anchor.dep_ in _COMPLEMENT_DEPS and anchor.head.lemma_ == "be":
        return anchor.head
    return None


def nominal_copula_subject(anchor: Token) -> Token | None:
    """The ``nsubj`` of a copular clause with a nominal predicate ("Kira Dace is a mapmaker": "Dace"), else None.

    The predicate is a copula heading a NOUN/PROPN ``attr`` complement, the anchor being that copula (a be.01 anchor)
    or the complement itself; the copula is the one ``_predicate_root`` finds for the complement. A parse that heads
    the clause on the noun instead (a NOUN/PROPN with a ``cop`` child) counts the same."""
    if anchor.pos_ in _NOMINAL_POS and any(c.dep_ == "cop" for c in anchor.children):
        head = anchor
    else:
        head = _predicate_root(anchor)
        if head is None or not any(
            c.dep_ == "attr" and c.pos_ in _NOMINAL_POS and _predicate_root(c) == head for c in head.children
        ):
            return None
    return next((c for c in head.children if c.dep_ == "nsubj"), None)


def _predicate_nodes(anchor: Token) -> list[Token] | None:
    """The predicate verb with its auxiliary, negation and copula children, the anchor included, in sentence order."""
    root = _predicate_root(anchor)
    if root is None:
        return None
    nodes = {anchor, root}
    for head in (anchor, root):
        nodes |= {c for c in head.children if c.dep_ in _PREDICATE_DEPS}
    return sorted(nodes, key=lambda t: t.i)


def _anchor_index(event: Mapping[str, Any], doc: Doc) -> int | None:
    anchors = [int(a) for a in event.get("anchor") or []]
    if not anchors or not 0 <= min(anchors) < len(doc):
        return None
    return min(anchors)


def _time_present(surface: str, phrase: str, others: Sequence[str]) -> bool:
    """Whether the time already appears in another filler: its surface words, or a date the phrase names."""
    dates = _DATE.findall(phrase)
    return any((surface and surface.lower() in o.lower()) or any(d in o for d in dates) for o in others)


def _cleanup(text: str) -> str:
    text = re.sub(r"\s+(?=['’]\w)", "", re.sub(r"\s+", " ", text))  # "you 've" -> "you've"
    text = re.sub(r"(\w) ' (?=\w)", r"\1' ", text)  # "kids ' smiles" -> "kids' smiles"
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip().rstrip(".!?").rstrip()


def _reason(event: Mapping[str, Any], doc: Doc) -> str | None:
    """Why the event is not an assembled proposition, or None when it is."""
    roles: Mapping[str, Any] = event.get("roles") or {}
    argm = roles.get(TIME_ROLE)
    if argm is not None and argm.get("relabelled_from") == "arg0":
        return "relabelled_arg0"
    index = _anchor_index(event, doc)
    if index is None:
        return "no_anchor"
    parts = _predicate_nodes(doc[index])
    if parts is None:
        return "no_verb"
    if not any(t.tag_ in _FINITE_TAGS for t in parts):
        return "non_finite"
    return None


def _filler_text(role: str, val: Mapping[str, Any], shown: Mapping[str, str], doc: Doc) -> str:
    """The text of one filler. A lone first- or second-person pronoun stays as written, so the envelope pass resolves
    it together with the predicate that agrees with it ("I'm" -> "Ana is"); any other filler shows its name."""
    span = sorted(val.get("span") or [])
    if len(span) == 1 and span[0] < len(doc) and _is_first_or_second_person(doc[span[0]]):
        return doc[span[0]].text
    return shown.get(role) or val.get("text") or ""


def _covers_predicate(anchor: Token, covered: set[int]) -> bool:
    """Whether every non-clausal dependent of the predicate (anchor or its copula head) lies in the predicate or a
    role span, so the assembled clause drops no complement."""
    for head in {anchor, _predicate_root(anchor)}:
        if head is None:
            continue
        for child in head.children:
            if child.i in covered or child.dep_ in _UNCOVERED_DEPS:
                continue
            if not {t.i for t in child.subtree} & covered:
                return False
    return True


def _assemble(
    event: Mapping[str, Any], doc: Doc, shown: Mapping[str, str], time_phrase: str | None
) -> tuple[str, bool] | None:
    """The assembled one-clause proposition and whether it passes the structural check, or None (no assembly)."""
    roles: Mapping[str, Any] = event.get("roles") or {}
    argm = roles.get(TIME_ROLE)
    index = _anchor_index(event, doc)
    if index is None or _reason(event, doc) is not None:
        return None
    anchor = doc[index]
    nodes = _predicate_nodes(anchor) or []
    parts: list[tuple[int, str]] = [(index, _join([_surface(t) for t in nodes]))]
    fillers: list[str] = []
    covered = {t.i for t in nodes}
    for role, val in roles.items():
        if role == TIME_ROLE:
            continue
        text = _filler_text(role, val, shown, doc)
        if text:
            fillers.append(text)
            parts.append((min(val.get("span") or [len(doc)]), text))
        covered |= set(val.get("span") or [])
    if argm is not None:
        surface = argm.get("text") or ""
        text = time_phrase or surface
        covered |= set(argm.get("span") or [])
        if text and not _time_present(surface, time_phrase or "", fillers):
            parts.append((min(argm.get("span") or [len(doc)]), text))

    text = _cleanup(" ".join(t for _, t in sorted(parts, key=lambda p: p[0])))
    if not text:
        return None
    out = text[:1].upper() + text[1:] + "."
    return out, _covers_predicate(anchor, covered) and not _REPEATED_WORD.search(out)


def render_proposition(
    event: Mapping[str, Any],
    unit_tokens: Sequence[str],
    shown: Mapping[str, str],
    time_phrase: str | None,
    *,
    sentence: str,
    speaker: str,
    addressee: str | None,
    told_at: str,
    mood: str,
    doc: Doc | None = None,
    mode: str = "b",
) -> str:
    """One G1 event as one English proposition.

    Args:
        event: the G1 record event (``anchor``, ``roles`` with ``span``/``text``/``relabelled_from``).
        unit_tokens: the token texts of the unit; parsed only when ``doc`` is None.
        shown: role -> display text for the fillers (referent names, else the filler text).
        time_phrase: the resolved phrase for the argm-tmp filler, or None.
        sentence, speaker, addressee, told_at: the envelope inputs.
        mood: the sentence mood; an interrogative sentence always uses the envelope-rendered sentence.
        doc: the spaCy parse of the unit the event spans index (the projection's own); reparsed when absent.
        mode: "b" (assembled text when the structural check passes, else the sentence) or "c" (sentence everywhere).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    def deictic(text: str) -> str:
        return envelope_render(text, speaker=speaker, addressee=addressee, ts=told_at)

    if mode == "c" or mood == "interrogative":
        return deictic(sentence)
    unit = doc if doc is not None else _parse_tokens(tuple(unit_tokens))
    assembled = _assemble(event, unit, shown, time_phrase)
    if assembled is not None and assembled[1]:
        return deictic(assembled[0])
    return deictic(sentence)


def legacy_fragment(event: Mapping[str, Any], shown: Mapping[str, str]) -> str:
    """The earlier search-key fragment of one event.

    The roleset lemma stands at the anchor and each filler's display text at the start of its span, in utterance
    order, joined by spaces: no inflection, no date resolution, raw time text ("yesterday"). Fillers the projection
    dropped (the referring filter) are absent from ``shown``. A filler without a span goes last, in role order.
    """
    anchor = event.get("anchor") or [0]
    items: list[tuple[int, int, str]] = [(int(min(anchor)), 0, str(event["roleset"]).split(".")[0])]
    for n, role in enumerate(sorted(shown), start=1):
        span = (event.get("roles") or {}).get(role, {}).get("span") or []
        items.append((int(min(span)) if span else 10**6, n, shown[role]))
    return " ".join(text for _, _, text in sorted(items) if text)
