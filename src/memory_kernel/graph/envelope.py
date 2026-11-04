"""Message envelope: mood and force from the parse, the act ladder, attributed rendering.

A message is an envelope (sender, addressee, act, mood, told_at, reply_to: synthesized from metadata and the parse
of its form, never stored in content) plus a payload (the plain PropBank structure). Everything here is pure: the
functions take spaCy ``Span``/``Token`` objects or plain dicts and decide from tags, dependencies and morphology,
never from word lists. The two closed-class lemmas read below (the modal ``will`` and the ``go`` of the periphrastic
future, with its infinitival ``to``) are English's grammatical future markers, not a vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from memory_kernel.grammar.semantic_operators import future_auxiliaries

#: Act of a clause or message when no context-labelled speech-act predicate names it (the roleset lemma does).
ASK, DIRECT, COMMIT, EVALUATE, ASSERT = "ask", "direct", "commit", "evaluate", "assert"
#: Payload modality by force: commissive -> intended, directive -> requested, else actual.
MODALITY = {COMMIT: "intended", DIRECT: "requested"}
_SUBJECTS = frozenset({"nsubj", "nsubjpass", "expl", "csubj"})
_COPULAR_COMPLEMENTS = frozenset({"acomp", "attr", "oprd"})


def _words(sent) -> list[Any]:
    return [t for t in sent if not t.is_space]


def sentence_mood(sent) -> str:
    """declarative / interrogative / imperative of one spaCy sentence.

    Interrogative: the sentence ends in "?"; or, with no terminal punctuation at all, it is wh-fronted (first word
    tagged W*) or subject-auxiliary inverted (first word an AUX whose head has a subject to its right). A terminal
    "." or "!" is the writer's explicit signal and is never overridden by structure ("What a great day!").
    Imperative: the root is a base-form verb (``VerbForm=Inf``, tag VB) with no subject dependent and no auxiliary
    ("Take a look!", "Call me later"). Else declarative.
    """
    words = _words(sent)
    if not words:
        return "declarative"
    last = words[-1]
    if "?" in last.text:
        return "interrogative"
    if not last.is_punct:
        first = next((t for t in words if not t.is_punct and t.pos_ != "INTJ"), None)
        if first is not None and first.tag_.startswith("W"):
            return "interrogative"
        if (
            first is not None
            and first.pos_ == "AUX"
            and first.dep_ == "aux"
            and any(c.dep_ in _SUBJECTS and c.i > first.i for c in first.head.children)
        ):
            return "interrogative"
    root = sent.root
    if (
        root.tag_ == "VB"
        and "Inf" in root.morph.get("VerbForm", [])
        and not any(c.dep_ in _SUBJECTS or c.dep_ in ("aux", "auxpass") for c in root.children)
    ):
        return "imperative"
    return "declarative"


def message_mood(moods: Iterable[str]) -> str:
    """The message's mood over its sentences: interrogative if any is, else imperative if any is, else declarative."""
    found = set(moods)
    for mood in ("interrogative", "imperative"):
        if mood in found:
            return mood
    return "declarative"


def clause_force(mood: str, commit: bool) -> dict[str, Any]:
    """The payload force of a clause: ``asserted`` (not a question) and ``modality`` (by force: direct/commit/assert)."""
    force = DIRECT if mood == "imperative" else COMMIT if commit else ASSERT
    return {"asserted": mood != "interrogative", "modality": MODALITY.get(force, "actual")}


def episodic_envelope(
    *,
    who: Any,
    who_name: Any,
    addressee: Any,
    addressee_name: Any,
    moods: Iterable[str],
    act: str,
    clause_acts: list[str],
    told_at: Any,
    reply_to: Any,
) -> dict[str, Any]:
    """The flat envelope properties of a message's ``:Episodic``."""
    return {
        "sender": who,
        "sender_name": who_name,
        "addressee": addressee,
        "addressee_name": addressee_name,
        "mood": message_mood(moods),
        "act": act,
        "clause_acts": clause_acts,
        "told_at": told_at,
        "reply_to": reply_to,
    }


def _first_person(tok) -> bool:
    return any(c.dep_ in _SUBJECTS and "1" in c.morph.get("Person", []) for c in tok.children)


def _going_to_head(tok):
    """The progressive ``go`` of a be-going-to future governing ``tok`` (tok = xcomp infinitive with aux ``to``)."""
    head = tok.head
    if tok.dep_ != "xcomp" or head.i == tok.i or head.lemma_.lower() != "go" or head.tag_ != "VBG":
        return None
    if not any(c.dep_ == "aux" and c.tag_ == "TO" for c in tok.children):
        return None
    if not any(c.dep_ in ("aux", "auxpass") and c.pos_ == "AUX" for c in head.children):
        return None
    return head


def commissive(tok) -> bool:
    """The predicate at ``tok`` is a first-person future: the speaker commits to it.

    Future = an auxiliary dependent tagged MD with lemma ``will`` (incl. "'ll"), or the periphrastic be-going-to
    (``tok`` an xcomp infinitive with aux ``to`` under a progressive ``go`` that has its own ``be`` auxiliary;
    "gonna" tokenizes as go + to), read through a copula for adjectival/nominal predicates ("I'll be happy"). The
    anchor may also be the ``go`` of the construction itself. Commissive = that future with a first-person subject
    (``Person=1`` on the subject): "Ben will paint" said by Ana is a prediction, an assertion. English has
    no Tense=Fut morphology, so the auxiliary chain is the only form-level signal; intention verbs ("plan to",
    "want to") are lexical and are NOT detected (they would need a vocabulary list).
    """
    verbs = [tok]
    if tok.dep_ in _COPULAR_COMPLEMENTS and tok.head.i != tok.i:
        verbs.append(tok.head)
    for v in verbs:
        if any(c.dep_ == "aux" and c.tag_ == "MD" and c.lemma_.lower() in future_auxiliaries() for c in v.children):
            return _first_person(v)
        going = _going_to_head(v)
        if going is not None:
            return _first_person(going)
        if (
            v.lemma_.lower() == "go"
            and v.tag_ == "VBG"
            and any((g := _going_to_head(child)) is not None and g.i == v.i for child in v.children)
        ):
            return _first_person(v)
    return False


def act_ladder(speech_acts: Iterable[str], moods: Iterable[str], commits: bool, holderless_evaluation: bool) -> str:
    """The act of a clause or message: a context-labelled speech-act predicate names it (its roleset lemma,
    first in utterance order); else interrogative -> ask; imperative -> direct; a commissive future -> commit; a
    holderless evaluation -> evaluate; else assert."""
    named = next(iter(speech_acts), None)
    if named:
        return named
    found = set(moods)
    if "interrogative" in found:
        return ASK
    if "imperative" in found:
        return DIRECT
    if commits:
        return COMMIT
    if holderless_evaluation:
        return EVALUATE
    return ASSERT


def render_attributed(edge_props: Mapping[str, object], envelope_props: Mapping[str, object]) -> str:
    """``[sender, told_at date] fact`` plus a force marker when the payload is not an actual assertion.

    ``edge_props``: a content edge's (``fact``) or an Event's (``name``) properties, with ``asserted`` and
    ``modality``; ``envelope_props``: the telling ``:Episodic``'s properties (``sender_name``, ``told_at`` or
    ``valid_at``). Markers: "(asked)" when ``asserted`` is false, "(intended)" / "(requested)" by modality.
    """
    fact = str(edge_props.get("fact") or edge_props.get("name") or "")
    sender = str(envelope_props.get("sender_name") or "?")
    told = str(envelope_props.get("told_at") or envelope_props.get("valid_at") or "")[:10]
    head = f"[{sender}, {told}] {fact}" if told else f"[{sender}] {fact}"
    return " ".join([head, *force_markers(edge_props)])


def force_markers(edge_props: Mapping[str, object]) -> list[str]:
    """The force markers of a payload: "(asked)" when ``asserted`` is false, "(intended)" / "(requested)" by modality."""
    markers = []
    if edge_props.get("asserted") is False:
        markers.append("(asked)")
    modality = edge_props.get("modality")
    if modality in ("intended", "requested"):
        markers.append(f"({modality})")
    return markers
