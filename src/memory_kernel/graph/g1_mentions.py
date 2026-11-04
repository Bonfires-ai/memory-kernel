"""Mention decomposition: a role filler (a phrase) -> its referents (``referent_policy`` structure).

A pure module over the unit's spaCy doc, the capture's token spans and WordNet noun classes (lexicographer files, a
published inventory). For one filler span:

1. **Nested frames are cut.** A frame of the unit whose anchor and role spans all lie inside the span (a relative or
   reduced clause, a nominal predicate) is cut out, except the filler's root and its left nominal dependents. The root's
   ``appos`` / ``relcl`` / ``acl`` clauses are dropped too (on the filler's own parse this is the whole of the step).
2. **Coordination splits:** one mention per conjunct (the root and its ``conj`` chain).
3. **Partitive and light heads:** "this kind of X" refers to X (a measure head, or a category: "kind" or a WordNet
   hyponym of it); a bare measure head ("a lot") refers to nothing. Both need a NOUN head.
4. **Keys:** a name core keys by the name; otherwise possessor + compounds + head ("data set", "Zorvex spyware": a
   name compound is part of the key, never the key). A 3rd-person pronoun possessor is dropped ("their phone" ->
   "phone"); a nominal one keeps its own key without determiners ("the Brask Labs' device" -> "brask labs ' device"). A lowercased known name ("the brask labs") is a name. Only a
   NOUN head has a head class (the predicate "rich" in "Kira is rich" has none).
5. **Names inside** a mention that are not its key become ``names`` entries ("Zorvex" in "Zorvex spyware"); names
   inside a cut frame are that frame's.

A capture span indexes the projection's own doc of the unit. A misaligned or span-less filler whose text is a
contiguous token run of the unit's parse is realigned onto that run (``realign``: the run nearest the given span),
with the dependency cuts only (the unit's frames come from the same suspect capture indices); any
other filler is decomposed on its own text (``decompose_text``) and counts as a fallback. On that fallback a mention
refers only with a nominal head (NOUN, PROPN, PRON) or as a name: a verb or adverb root ("force" in "until the
conditions force a choice") is dropped.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from typing import TYPE_CHECKING

from memory_kernel.grammar.wordnet_semantics import get_wordnet_semantics
from memory_kernel.graph.g1_nlp import REFERRING_HEAD, STRIP_LEAD, lemma_key, nlp, normalize_exact
from memory_kernel.graph.identity import capitalised, is_common_noun

if TYPE_CHECKING:
    from spacy.tokens import Token

#: Left dependents of the filler's root that a nested-frame cut never removes (spaCy labels).
_PROTECTED_DEPS = frozenset({"det", "poss", "amod", "compound", "nummod"})
#: Clausal dependents of a conjunct head dropped from its mention (spaCy labels).
_CLAUSE_DEPS = frozenset({"appos", "relcl", "acl"})
#: Dependents of a conjunct head that separate it from the next conjunct (spaCy labels).
_CONJUNCT_DEPS = frozenset({"conj", "cc", "punct"})
#: The only dependents a bare light head may have and still refer to nothing: none is a complement (spaCy labels).
_LIGHT_DEPS = frozenset({"det", "relcl", "acl", "punct", "cc", "conj"})
#: Dependency labels that join the tokens of one multi-token name.
_NAME_DEPS = frozenset({"compound", "flat"})
#: WordNet lexicographer file of measures: a partitive or bare light head ("a lot of tools", "a lot").
_QUANTITY = "noun.quantity"
#: WordNet synset of categories: a partitive head is it or a hyponym of it ("kind", "sort", "type").
_CATEGORY = "kind.n.01"
#: Parts of speech of a fallback mention's head that refer (spaCy POS): the referring heads and pronouns.
_NOMINAL = REFERRING_HEAD | {"PRON"}


@dataclass(frozen=True)
class Mention:
    tokens: tuple[int, ...]  # doc indices of the referring core (determiners included, cut parts excluded)
    surface: str  # display text: the key tokens in their written case ("data set", "Zorvex")
    key: str  # lemma_key of the key tokens
    head: str  # head token lemma, lowercased (WordNet lookups; "" for a name or a head that is not a NOUN)
    name: bool  # a name core (PROPN chain, a known name in name position, or a known name lowercased)
    kind_nouns: tuple[str, ...]  # lemmas the text supplies about it: appositive head, a name head's common compound


@dataclass(frozen=True)
class Decomposition:
    mentions: tuple[Mention, ...]  # the role's referents, one per conjunct; () = nothing referring
    names: tuple[Mention, ...]  # names inside the core(s) that are not a mention themselves
    fallback: bool  # decided on the filler text's own parse (no usable span)
    #: the unit-doc run a misaligned span was moved to; () when the span was used as given
    realigned: tuple[int, ...] = ()
    #: fallback mentions dropped for a head that is not nominal
    dropped: int = 0
    #: conjuncts whose bare light head refers to nothing ("a lot")
    light: int = 0


_EMPTY = Decomposition((), (), False)


# --- WordNet classes -------------------------------------------------------------------------------------------------
def _senses(lemma: str) -> tuple[str, ...]:
    return get_wordnet_semantics().noun_sense_lexnames(lemma.lower())


@cache
def bare_light(lemma: str) -> bool:
    """A head that refers to nothing without a complement: its first noun sense is a measure, ``noun.quantity``
    ("lot", "bit")."""
    senses = _senses(lemma)
    return bool(senses) and senses[0] == _QUANTITY


def _pobj(prep):
    return next((c for c in prep.children if c.dep_ == "pobj"), None)


def partitive_head(token) -> bool:
    """A partitive / measure head ("this kind of attacks", "a lot of tools"): a NOUN whose only dependents are
    determiners, compounds ("a Zorvex kind of …") and one preposition whose object is a NOUN without a determiner,
    and whose first noun sense is a
    measure (``noun.quantity``) or a category (``kind.n.01`` or a hyponym of it). Another cognition noun is not
    ("the investigation of attacks")."""
    if token.pos_ != "NOUN":
        return False
    kids = list(token.children)
    preps = [c for c in kids if c.dep_ == "prep"]
    if len(preps) != 1 or preps[0].pos_ != "ADP" or any(c.dep_ not in ("det", "compound", "prep") for c in kids):
        return False
    obj = _pobj(preps[0])
    if obj is None or obj.pos_ != "NOUN" or any(c.dep_ == "det" for c in obj.children):
        return False
    senses = _senses(token.lemma_)
    if senses and senses[0] == _QUANTITY:
        return True
    return _CATEGORY in get_wordnet_semantics().first_noun_sense_closure(token.lemma_.lower())


# --- Names -----------------------------------------------------------------------------------------------------------
def _text(doc, indices: Iterable[int]) -> str:
    """The tokens in their written form: adjacent tokens keep their own spacing ("Brask's"), a gap is one space."""
    order = sorted(indices)
    parts = []
    for n, i in enumerate(order):
        tok = doc[i]
        last = n + 1 == len(order)
        parts.append(tok.text + ("" if last else tok.whitespace_ if order[n + 1] == i + 1 else " "))
    return "".join(parts).strip()


@lru_cache(maxsize=4096)
def known_names(doc) -> frozenset[str]:
    """``normalize_exact`` of every PROPN chain (PROPN tokens joined by ``compound`` / ``flat``) in the doc. Cached
    per doc object, bounded (a fresh doc per call cannot grow it without limit)."""
    propn = {t.i for t in doc if t.pos_ == "PROPN"}
    seen: set[int] = set()
    out: set[str] = set()
    for i in sorted(propn):
        if i not in seen:
            chain = _chain(doc[i], propn)
            seen |= set(chain)
            out.add(normalize_exact(_text(doc, chain)))
    return frozenset(out)


def _chain(tok, names: Collection[int]) -> tuple[int, ...]:
    """The maximal set of name tokens (``names``) joined to ``tok`` by ``compound`` / ``flat``, in doc order."""
    out = {tok.i}
    stack = [tok]
    while stack:
        t = stack.pop()
        near = [c for c in t.children if c.dep_ in _NAME_DEPS]
        if t.dep_ in _NAME_DEPS:
            near.append(t.head)
        for n in near:
            if n.i in names and n.i not in out:
                out.add(n.i)
                stack.append(n)
    return tuple(sorted(out))


def _name_tokens(doc, scope: frozenset[int], known: Collection[str]) -> frozenset[int]:
    """The name tokens of a conjunct: every PROPN, and a known name (``known``, not a WordNet common noun) in name
    position: a ``compound`` dependent, or the conjunct's sole token. The part of speech of a lowercased unknown word
    is no evidence (spaCy tags "zorvex" ADJ), so position and the known-name set decide."""

    def known_name(tok) -> bool:
        in_position = tok.dep_ == "compound" or len(scope) == 1
        return in_position and normalize_exact(tok.text) in known and not is_common_noun(tok.text)

    return frozenset(i for i in scope if doc[i].pos_ == "PROPN" or known_name(doc[i]))


def _third_person_possessive(tok) -> bool:
    """A 3rd-person possessive pronoun ("their", "its"): spaCy morph ``PronType=Prs`` + ``Poss=Yes`` + ``Person=3``,
    or Penn tag ``PRP$`` with no 1st/2nd person."""
    morph = tok.morph
    persons = set(morph.get("Person"))
    if "Prs" in morph.get("PronType") and "Yes" in morph.get("Poss"):
        return "3" in persons
    return tok.tag_ == "PRP$" and not persons & {"1", "2"}


def _key_tokens(head, scope: frozenset[int], names: frozenset[int]) -> set[int]:
    """A common head's key tokens: the head, its ``compound`` subtrees and its possessor's owner tokens."""
    key = {head.i}
    for c in head.children:
        if c.i not in scope:
            continue
        if c.dep_ == "compound":
            key |= _subtree(c) & scope
        elif c.dep_ == "poss":
            key |= _owner_tokens(c, scope, names)
    return key


def _owner_tokens(poss, scope: frozenset[int], names: frozenset[int]) -> set[int]:
    """A possessor's part of the key. A 3rd-person pronoun is dropped like a determiner ("their phone" -> phone); a
    1st/2nd-person one stays as written ("my phone": ``owner_key`` qualifies it); a nominal owner keeps its own key and
    case marker, without its determiners ("the Brask Labs' device" -> "Brask Labs' device")."""
    if poss.pos_ == "PRON" or poss.tag_ == "PRP$":
        return set() if _third_person_possessive(poss) else _subtree(poss) & scope
    own = set(_chain(poss, names)) if poss.i in names else _key_tokens(poss, scope, names)
    return own | {c.i for c in poss.children if c.dep_ == "case" and c.i in scope}


def _name_mention(doc, chain: Sequence[int]) -> Mention:
    surface = _text(doc, chain)
    return Mention(tuple(chain), surface, lemma_key(surface), "", True, ())


# --- Decomposition ---------------------------------------------------------------------------------------------------
def _subtree(tok) -> set[int]:
    return {t.i for t in tok.subtree}


def _reach(head, core: Collection[int]) -> set[int]:
    """The conjunct's tokens: those reachable from its head through dependents inside the core, except the head's
    own conjunct separators and clauses. A token whose governor was cut is unreachable."""
    out = {head.i}
    stack = [c for c in head.children if c.dep_ not in _CONJUNCT_DEPS | _CLAUSE_DEPS and c.i in core]
    while stack:
        t = stack.pop()
        if t.i in out:
            continue
        out.add(t.i)
        stack.extend(c for c in t.children if c.i in core and c.i not in out)
    return out


def _conjunct_heads(head: Token, within: Collection[int]) -> list[Token]:
    """``head`` and its ``conj`` chain inside ``within``, in doc order."""
    heads = [head]
    for h in heads:
        heads.extend(c for c in h.children if c.dep_ == "conj" and c.i in within and c not in heads)
    return sorted(heads, key=lambda t: t.i)


def _left_nominal(tok) -> set[int]:
    """A head and its left nominal dependents (det, poss, amod, compound, nummod), with their subtrees."""
    out = {tok.i}
    for c in tok.children:
        if c.i < tok.i and c.dep_ in _PROTECTED_DEPS:
            out |= _subtree(c)
    return out


def _bare_light_head(head) -> bool:
    """A NOUN head with a measure first sense (``bare_light``) whose dependents are all ``_LIGHT_DEPS``: no complement."""
    return head.pos_ == "NOUN" and bare_light(head.lemma_) and all(c.dep_ in _LIGHT_DEPS for c in head.children)


def _appositive_name(doc, head, named, core: Collection[int], known: Collection[str]):
    """``_conjunct`` for a common head with a PROPN appositive ("the journalist Milo Dace", read as ``journalist`` +
    ``appos`` Milo): the appositive's name chain is the mention (a name), the common noun a kind it supplies, never a
    second entity. Other names in the conjunct stay inner names."""
    toks = _reach(head, core) | (_subtree(named) & set(core))
    scope = frozenset(toks)
    names = _name_tokens(doc, scope, known)
    chains = sorted({_chain(doc[i], names) for i in names})
    key_toks = _chain(named, names)
    surface = _text(doc, key_toks)
    mention = Mention(tuple(sorted(toks)), surface, lemma_key(surface), "", True, (head.lemma_.lower(),))
    return [(mention, toks, [c for c in chains if c != key_toks])]


def _conjunct(
    doc, head, core: Collection[int], known: Collection[str], *, narrowed: bool = False
) -> list[tuple[Mention | None, set[int], list[tuple[int, ...]]]]:
    """One conjunct -> [(its mention or None, its tokens, the name chains inside it that are not its key tokens)].
    A partitive head gives way to its object's conjuncts ("this kind of cats and dogs" -> cats, dogs). A name
    compound of a common head is part of its key and an inner name ("Zorvex spyware": Zorvex), never its key."""
    toks = _reach(head, core)
    if partitive_head(head):
        obj = _pobj(next(c for c in head.children if c.dep_ == "prep"))
        if obj is not None and obj.i in toks:
            parts = [p for h in _conjunct_heads(obj, core) for p in _conjunct(doc, h, core, known, narrowed=True)]
            rest = frozenset(toks - set().union(*(p[1] for p in parts)))
            inner = _name_tokens(doc, rest, known)  # a name compound of the partitive head is an inner name
            return [(None, set(rest), sorted({_chain(doc[i], inner) for i in inner})), *parts]
    appositives = tuple(c.lemma_.lower() for c in head.children if c.dep_ == "appos" and c.pos_ == "NOUN")
    named = next((c for c in head.children if c.dep_ == "appos" and c.pos_ == "PROPN" and c.i in core), None)
    restrictive = named is not None and not any(t.tag_ == "," for t in doc[min(head.i, named.i) : max(head.i, named.i)])
    if named is not None and restrictive and head.pos_ == "NOUN" and not any(c.dep_ == "conj" for c in named.children):
        # "the journalist Milo Dace" on a parse that heads it by the noun: the appositive is the name, the noun a kind
        # (a listed appositive, "my friends Ana and Milo", keeps the group reading: the noun is the mention; a
        # comma-set one, "my graph, Kira", is no restrictive name: the common noun stays the mention; a bracketed
        # one, "a wiki (Zorvex)", is: only a COMMA between head and appositive refuses the name)
        return _appositive_name(doc, head, named, core, known)
    scope = frozenset(toks)
    names = _name_tokens(doc, scope, known)
    chains = sorted({_chain(doc[i], names) for i in names})
    if head.i in names:
        key_toks = _chain(head, names)
        partners = tuple(
            c.lemma_.lower()
            for c in head.children
            if c.dep_ == "compound" and c.i in scope - names and c.pos_ == "NOUN"
        )
        lemma, name, kind_nouns = "", True, partners + appositives
    elif not narrowed and _bare_light_head(head):
        return [(None, toks, chains)]
    else:
        key_toks = tuple(sorted(_key_tokens(head, scope, names)))
        written = _text(doc, key_toks)
        if normalize_exact(written) in known and not is_common_noun(written):  # a known name lowercased
            lemma, name = "", True  # ("the brask labs"; "the rose" stays a common noun, as in ``_name_tokens``)
        else:  # only a NOUN head has a head class
            lemma, name = (head.lemma_.lower() if head.pos_ == "NOUN" else ""), False
        kind_nouns = appositives
    surface = _text(doc, key_toks)
    key = lemma_key(surface)
    mention = Mention(tuple(sorted(toks)), surface, key, lemma, name, kind_nouns)
    return [(mention, toks, [c for c in chains if c != key_toks])]


def _decompose(doc, span: Sequence[int], frames, known: Collection[str]) -> Decomposition:
    idx = sorted(set(span))
    if not idx:
        return _EMPTY
    inside = set(idx)
    root = next(doc[i] for i in idx if doc[i].head.i not in inside or doc[i].head.i == i)
    if root.pos_ == "ADP" and (obj := _pobj(root)) is not None and obj.i in inside:
        root = obj
    # the root, every conjunct head and their left nominal dependents are never cut ("Kira and targets selected by …")
    protected = set().union(*(_left_nominal(h) for h in _conjunct_heads(root, inside)))
    cut: set[int] = set()
    for anchor, roles in frames:
        a = set(anchor)
        spans = [set(r) for r in roles]
        if a and a <= inside and root.i not in a and all(r <= inside for r in spans):
            cut |= a.union(*spans)
    core = inside - (cut - protected)
    mentions: list[Mention] = []
    inner: dict[tuple[int, ...], Mention] = {}
    covered: set[int] = set()
    light = 0
    for h in _conjunct_heads(root, core):
        parts = _conjunct(doc, h, core, known)
        light += _bare_light_head(h) and all(p[0] is None for p in parts)
        for mention, toks, chains in parts:
            covered |= toks
            if mention is not None:
                mentions.append(mention)
            for chain in chains:
                inner.setdefault(chain, _name_mention(doc, chain))
    # never lose a name: one in a dependency-dropped clause or appositive (not a frame's) is an inner name too
    rest = frozenset(core - covered)
    names = _name_tokens(doc, rest, known)
    for chain in {_chain(doc[i], names) for i in names}:
        inner.setdefault(chain, _name_mention(doc, chain))
    return Decomposition(tuple(mentions), tuple(inner[c] for c in sorted(inner)), False, light=light)


def decompose(
    doc,
    span: Sequence[int],
    text: str,
    frames: Sequence[tuple[Sequence[int], Sequence[Sequence[int]]]],
    known: Collection[str] = (),
) -> Decomposition:
    """The referents of one role filler. ``span`` indexes ``doc`` (the projection's parse of the unit); ``frames`` is
    every other non-excluded event of the unit as (anchor indices, [role spans]); ``known`` the conversation's names
    (``known_names``, normalized). A misaligned or span-less filler is realigned (``realign``; ``realigned`` holds the
    run used; no frame cuts, the unit's other capture indices being as suspect as its own), else decided on its own
    text (``fallback`` True), where a mention whose head is not nominal and that is not a name refers to nothing
    (``dropped`` counts them)."""
    if _aligned(doc, span, text):
        return _decompose(doc, span, frames, known)
    if (run := realign(doc, span, text)) is not None:
        return replace(_decompose(doc, run, (), known), realigned=run)  # the capture's other indices are suspect
    dec = decompose_text(text, known)
    own = _own_doc(text)
    kept = tuple(m for m in dec.mentions if m.name or mention_head(own, m).pos_ in _NOMINAL)
    return replace(dec, mentions=kept, fallback=True, dropped=len(dec.mentions) - len(kept))


def _aligned(doc, span: Sequence[int], text: str) -> bool:
    return bool(span) and all(0 <= i < len(doc) for i in span) and " ".join(doc[i].text for i in sorted(span)) == text


def doc_span(doc, span: Sequence[int], text: str) -> tuple[int, ...] | None:
    """A capture span on the unit's parse ``doc``, as ``decompose`` places it: the span itself when it is aligned
    (its tokens are ``text``), else its realigned run (``realign``); None when the parse does not hold ``text``."""
    if _aligned(doc, span, text):
        return tuple(sorted(span))
    return realign(doc, span, text)


def realign(doc, span: Sequence[int], text: str) -> tuple[int, ...] | None:
    """The contiguous token run of ``doc`` whose characters are exactly ``text``'s, all whitespace removed from both
    (blind to tokenisation: "re-exploit" is spaCy's "re - exploit", "Ana's" its "Ana 's"; whitespace tokens are
    skipped and left out of the run): the one nearest the span's first index (the earlier on a tie; the first without
    a span); None when no run matches."""
    target = "".join(text.split())
    words = [t.i for t in doc if not t.is_space]
    runs = []
    for n in range(len(words)):
        joined, end = "", n
        while len(joined) < len(target) and end < len(words):
            joined += doc[words[end]].text
            end += 1
        if target and joined == target:
            runs.append(tuple(words[n:end]))
    if not runs:
        return None
    first = min(span) if span else 0
    return min(runs, key=lambda run: (abs(run[0] - first), run[0]))


def mention_head(doc, mention: Mention):
    """A mention's head token: the one whose governor lies outside its tokens."""
    toks = set(mention.tokens)
    return next(doc[i] for i in mention.tokens if doc[i].head.i not in toks or doc[i].head.i == i)


@lru_cache(maxsize=4096)
def _own_doc(text: str):
    return nlp()(text)


@cache
def _decompose_text(text: str, known: frozenset[str]) -> Decomposition:
    doc = _own_doc(text)
    return _decompose(doc, range(len(doc)), (), known)


def decompose_text(text: str, known: Collection[str] = ()) -> Decomposition:
    """The referents of a phrase on its own parse (no frames: the dependency cuts are the whole of step 1).
    ``fallback`` is False here; ``decompose`` sets it when it falls back to this."""
    words = normalize_exact(text)
    relevant = frozenset(f for t in _own_doc(text) if (f := normalize_exact(t.text)) in known) | frozenset(
        k for k in known if " " in k and re.search(rf"(?<!\w){re.escape(k)}(?!\w)", words)
    )
    return _decompose_text(text, relevant)


def _stripped(doc):
    """The doc without its leading prepositions, then determiners (``g1_projection.analyse``'s surface)."""
    start = 0
    while start < len(doc) and doc[start].pos_ == "ADP":
        start += 1
    while start < len(doc) and doc[start].pos_ in STRIP_LEAD:
        start += 1
    return doc[start:]


def structure_key(text: str) -> str:
    """The first mention's key of the phrase on its own parse, else ``lemma_key`` of the phrase.

    Orthographic guard: a short phrase parsed alone can lose a name ("Zorvex targets" tags "Zorvex" ADJ). When a
    capitalised word of the stripped phrase is not among the first mention's key words, the key is ``lemma_key``
    of the stripped surface, as ``analyse``'s."""
    dec = decompose_text(text)
    if not dec.mentions:
        return lemma_key(text)
    first = dec.mentions[0]
    kept = _stripped(_own_doc(text))
    words = [t.text for t in kept if capitalised(t.text)]
    if all(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", first.surface) for w in words):
        return first.key
    return lemma_key(kept.text) if len(kept) else normalize_exact(text)
