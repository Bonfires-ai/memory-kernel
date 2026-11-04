"""What a question asks for, read from its parse: the answer shape, the answer role and kind.

``read_shape`` takes the parse of a question (``graph.g1_nlp.turn_doc``, the parse ``normalise`` reads, so a question is
parsed once and the same way everywhere) and reads, from parse features only:

* the wh token (its lemma, looked up in ``shapes.json``) and its dependency path up to the predicate that governs it,
  with the predicate's voice;
* for "how", the governed adjective or adverb (its POS; its lemma picks ``how_many`` / ``how_long`` in the data);
* the ``Number`` morphology of the wh phrase's head, or of the subject of a copula the wh word is the complement of,
  and a quantifying determiner (``PronType=Tot``): a plural wh phrase of an enumerating shape asks for a list;
* the auxiliary opener of a question with no wh word: a polar question, or a hypothetical modal (an inference).

The closed-class vocabulary (wh lemmas, the modal openers, the shape labels they map to, the kinds) is data in
``data/runtime-grammar/en/shapes.json``. The answer role comes from patterns keyed by (wh lemma, dependency path, voice)
(``mine_shape_patterns`` mines them from question rows; kernel 1.8's build writes no such rows and nothing calls it);
the expected kind follows from the shape and the role through the data. Deterministic.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

RULES_FILE = ("data", "runtime-grammar", "en", "shapes.json")
#: POS of the adjective or adverb a "how" governs (spaCy's tag names, not a word list)
DEGREE_POS = frozenset({"ADJ", "ADV"})
NOUN_POS = frozenset({"NOUN", "PROPN"})
PREDICATE_POS = frozenset({"VERB", "AUX"})
#: dependency labels of an auxiliary child, and of the passive subject and auxiliary that make a voice
AUX_DEPS = frozenset({"aux", "auxpass"})
PASSIVE_DEPS = frozenset({"nsubjpass", "auxpass"})
#: the dependency of a wh word that is the complement of a copula, and of a subject
COMPLEMENT_DEPS = frozenset({"attr", "acomp"})
SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass"})
#: POS of a token that only quantifies (a pronoun or determiner head): its number is its "of" object's
PARTITIVE_POS = frozenset({"PRON", "DET"})
#: a proper noun's Number says nothing about the question (a title: "Little Women" is one thing)
NAME_POS = "PROPN"
PLURAL = "Plur"
QUANTIFIER = "PronType=Tot"
_MAX_CLIMB = 12


@dataclass(frozen=True)
class QuestionShape:
    shape: str | None
    answer_role: str | None
    expected_kind: str | None
    plural: bool
    polar: bool
    #: the lemma of the question's wh word ("what", "whose", "how"), or None when it has none
    wh: str | None = None


@lru_cache(maxsize=1)
def load_rules() -> dict[str, Any]:
    text = resources.files("memory_kernel").joinpath(*RULES_FILE).read_text(encoding="utf-8")
    return json.loads(text)


def _tokens(doc: Any) -> list[Any]:
    return [t for t in doc if not t.is_space]


def _lemma(token: Any) -> str:
    return token.lemma_.lower()


def _is_predicate(token: Any) -> bool:
    return token.pos_ in PREDICATE_POS or any(c.dep_ in AUX_DEPS for c in token.children)


def _wh_token(tokens: list[Any], rules: Mapping[str, Any]) -> Any | None:
    return next((t for t in tokens if _lemma(t) in rules["wh_shapes"]), None)


def _climb(wh: Any) -> tuple[Any | None, str]:
    """The predicate that governs ``wh`` and the dependency path down to the wh word, e.g. ``prep>pobj``."""
    deps: list[str] = []
    token = wh
    for _ in range(_MAX_CLIMB):
        deps.append(token.dep_)
        head = token.head
        if head.i == token.i:
            return (token if _is_predicate(token) else None), ">".join(reversed(deps[:-1]))
        token = head
        if _is_predicate(token):
            return token, ">".join(reversed(deps))
    return None, ""


def _voice(predicate: Any | None) -> str | None:
    if predicate is None:
        return None
    return "pass" if any(c.dep_ in PASSIVE_DEPS for c in predicate.children) else "act"


def _plural(token: Any) -> bool:
    if token.pos_ == NAME_POS:
        return False
    if token.pos_ in PARTITIVE_POS:
        return any(_plural(o) for c in token.children if c.dep_ == "prep" for o in c.children if o.dep_ == "pobj")
    return PLURAL in token.morph.get("Number")


def _how_shape(wh: Any, rules: Mapping[str, Any]) -> str:
    governed = wh.head
    if wh.dep_ == "advmod" and governed.i != wh.i and governed.pos_ in DEGREE_POS:
        return rules["how_degree_shapes"].get(_lemma(governed), rules["how_other_degree"])
    return rules["how_manner"]


def _plural_wh_phrase(wh: Any, predicate: Any | None, tokens: list[Any]) -> bool:
    if wh.dep_ == "det" and wh.head.pos_ in NOUN_POS and _plural(wh.head):
        return True
    if (
        wh.dep_ in COMPLEMENT_DEPS | SUBJECT_DEPS
        and predicate is not None
        and any(c.dep_ in COMPLEMENT_DEPS | SUBJECT_DEPS and c.i != wh.i and _plural(c) for c in predicate.children)
    ):
        return True
    return any(QUANTIFIER in str(t.morph) and t.i > wh.i for t in tokens)


def pattern_key(wh: Any, path: str, voice: str | None) -> tuple[str, str, str | None]:
    return (_lemma(wh), path, voice)


def _kind(shape: str | None, role: str | None, rules: Mapping[str, Any]) -> str | None:
    return rules["wh_kinds"].get(shape) or rules["role_kinds"].get(role)


def _collapsed(doc: Any) -> Any:
    """The parse of the question with runs of whitespace collapsed (a double space misparses); the doc itself when
    it has none."""
    text = " ".join(doc.text.split())
    if text == doc.text:
        return doc
    from memory_kernel.graph.g1_nlp import turn_doc

    return turn_doc(text)


def _addressed(predicate: Any, rules: Mapping[str, Any]) -> bool:
    """An indirect request: the root's subject is second person and the wh clause is its complement ("Do you know
    where Milo lives?" asks where; "Did Ana say who left?" asks yes or no)."""
    root = predicate.sent.root
    return predicate.dep_ in rules["addressed_complement_deps"] and any(
        c.dep_ in SUBJECT_DEPS and "2" in c.morph.get("Person") for c in root.children
    )


def _embedded(tokens: list[Any], predicate: Any | None) -> bool:
    """A question that opens on an auxiliary before its subject (inversion at the root) and whose wh word sits in a
    clause below the root asks yes or no."""
    return predicate is not None and bool(tokens) and tokens[0].pos_ == "AUX" and predicate.sent.root.i != predicate.i


def read_shape(doc: Any, patterns: Iterable[Mapping[str, Any]]) -> QuestionShape:
    """The shape of a question; it assumes its input is a question (callers gate on that)."""
    rules = load_rules()
    doc = _collapsed(doc)
    tokens = _tokens(doc)
    wh = _wh_token(tokens, rules)
    if wh is None:
        first = tokens[0] if tokens else None
        if first is not None and first.pos_ == "AUX":
            if _lemma(first) in rules["inference_openers"]:
                return QuestionShape(rules["inference_shape"], None, None, False, False)
            return QuestionShape(rules["polar_shape"], None, None, False, True)
        return QuestionShape(None, None, None, False, False)
    wh_lemma = _lemma(wh)
    shape = (
        _how_shape(wh, rules)
        if rules["wh_shapes"][_lemma(wh)] == rules["degree_shape"]
        else rules["wh_shapes"][_lemma(wh)]
    )
    predicate, path = _climb(wh)
    if _embedded(tokens, predicate) and not _addressed(predicate, rules):
        return QuestionShape(rules["polar_shape"], None, None, False, True, wh_lemma)
    plural = _plural_wh_phrase(wh, predicate, tokens)
    if plural and shape in rules["enumerating_shapes"]:
        shape = rules["list_shape"]
    index = {(p["wh"], p["path"], p["voice"]): p for p in patterns}
    hit = index.get(pattern_key(wh, path, _voice(predicate)))
    role = hit["answer_role"] if hit else None
    return QuestionShape(shape, role, _kind(shape, role, rules), plural, False, wh_lemma)


def wh_path(doc: Any) -> dict[str, Any] | None:
    """The wh token, its path to the governing predicate and the predicate's voice (for the agreement sheet)."""
    tokens = _tokens(doc)
    wh = _wh_token(tokens, load_rules())
    if wh is None:
        return None
    predicate, path = _climb(wh)
    return {
        "wh": _lemma(wh),
        "path": path,
        "voice": _voice(predicate),
        "predicate": _lemma(predicate) if predicate else None,
    }


def mine_shape_patterns(taxq_rows: Iterable[Mapping[str, Any]], nlp: Callable[[str], Any]) -> list[dict[str, Any]]:
    """(wh lemma, dependency path, voice) -> the majority answer role of a bonfire's taxonomy questions, with the row
    count (``rows``) and the majority's share of them (``agree``). A pattern seen on one row only is dropped, and so is
    one whose top role ties with the runner-up (a role must win by a strict plurality). Questions are parsed as ``read_shape``
    parses them, with whitespace collapsed."""
    roles: dict[tuple[str, str, str | None], Counter[str]] = defaultdict(Counter)
    for row in taxq_rows:
        role = row.get("answer_role")
        found = wh_path(nlp(" ".join(row["question"].split()))) if role else None
        if found is not None and found["predicate"] is not None:
            roles[(found["wh"], found["path"], found["voice"])][role] += 1
    out = []
    for (wh, path, voice), counts in roles.items():
        total = sum(counts.values())
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        role, agree = ranked[0]
        if total < 2 or (len(ranked) > 1 and ranked[1][1] == agree):
            continue
        out.append({"wh": wh, "path": path, "voice": voice, "answer_role": role, "rows": total, "agree": agree})
    return sorted(out, key=lambda p: (-p["rows"], p["wh"], p["path"], p["voice"] or ""))
