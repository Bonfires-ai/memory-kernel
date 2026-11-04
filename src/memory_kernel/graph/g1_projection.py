"""Pure projection of the G1 event graph onto the Graphiti-compatible contract rows.

An episode is a ``:Saga`` (``HAS_EPISODE`` to its messages, ``NEXT_EPISODE`` in message order); a message is an
immutable ``:Episodic``; each predicate is routed by its eventuality (occurrence -> ``:Entity:Event``; state -> fact
edge or attributed entity summary line; evaluation with a holder -> as a state; holderless evaluation and speech act
-> the message envelope's act, no content). The rows carry no topic labels, codings or card data (kernel 1.8).
Attribution lives in the envelope (flat properties of the ``:Episodic``).

Node economy: only referring fillers (resolved people/names, identifying content noun phrases) become
``:Entity``; adjectival, wh-, unresolved-pronoun, time, generic and other non-nominal fillers stay text. Entities are
deduplicated by a deterministic ladder: referent identity, exact lemma key (batch and existing nodes of the group),
fuzzy over high-entropy names (character-trigram Jaccard, found by an exact similarity join; never across differing
possessive determiners), else separate.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from functools import lru_cache
from itertools import product
from typing import TYPE_CHECKING, Any

from memory_kernel.graph.addressing import addressee_of, is_explicit
from memory_kernel.graph.envelope import (
    act_ladder,
    clause_force,
    commissive,
    episodic_envelope,
    render_attributed,
    sentence_mood,
)
from memory_kernel.graph.g1_constructions import (
    LANGUAGE_CONSTRUCTIONS,
    NONE,
    construction_for,
    language_version,
    role_construction,
)
from memory_kernel.graph.g1_ids import content_uuid as _sha
from memory_kernel.graph.g1_ids import summary_line_id
from memory_kernel.graph.g1_kinds import KindEvidence, decide, filler_evidence, frame_evidence
from memory_kernel.graph.g1_mentions import (
    Decomposition,
    Mention,
    decompose,
    decompose_text,
    doc_span,
    known_names,
    structure_key,
)
from memory_kernel.graph.g1_mentions import _own_doc as _own_parse
from memory_kernel.graph.g1_mentions import _text as _tokens_text
from memory_kernel.graph.g1_mentions import mention_head as _mention_head
from memory_kernel.graph.g1_names import (
    AGENT_ROLE,
    CAUSE_TAG,
    CONTEXT,
    EVENTUALITIES,
    PLACE,
    PRIOR,
    TIME_ROLE,
    is_core_role,
)
from memory_kernel.graph.g1_nlp import (
    REFERRING_HEAD as _REFERRING_HEAD,
)
from memory_kernel.graph.g1_nlp import (
    STRIP_LEAD as _STRIP_LEAD,
)
from memory_kernel.graph.g1_nlp import (
    lemma_key,
    non_identifying,
    normalize_exact,
)
from memory_kernel.graph.g1_nlp import (
    nlp as _nlp,
)
from memory_kernel.graph.g1_referring import is_referring, owner_key
from memory_kernel.graph.g1_render import MODES, legacy_fragment, nominal_copula_subject, render_proposition
from memory_kernel.graph.g1_roles import function_tag, holder_role
from memory_kernel.graph.g1_time import event_time
from memory_kernel.graph.identity import (
    Claim,
    Joins,
    alias_occurs,
    capitalised,
    identity_claims,
    is_common_noun,
    resolve_joins,
)
from memory_kernel.graph.identity_cards import load_identity_cards
from memory_kernel.knowledge.propbank_framesets import PropBankRolesetDefinition

if TYPE_CHECKING:
    from memory_kernel.pipeline.chain import Chain

log = logging.getLogger(__name__)

#: The time fields shared by Events, role edges and state fact edges (``g1_time.event_time``).
TIME_FIELDS = ("told_at", "happened_at", "happened_at_grain", "time_source", "valid_at")
_SUPPLETIVE_3SG = {"be": "is", "have": "has"}
#: Existing nodes that are not candidates for entity resolution (kernel-reified events, taxonomy labels).
NOT_CANDIDATES = frozenset({"Event", "TaxonomyLabel", "Episodic"})


def _person_name(text: str) -> str:
    return text.removesuffix(" and others").strip()


# --- Entity dedup ladder: the fuzzy step (deterministic; no model call) --------------------------------------------
#: A name is fuzzy-matchable only when it is long enough (characters, or words) and not repetitive (character entropy).
FUZZY_MIN_CHARS = 6
FUZZY_MIN_WORDS = 2
FUZZY_MIN_ENTROPY_BITS = 1.5
#: Two names join when the Jaccard similarity of their character trigram sets is at least NUM/DEN (0.9), compared in
#: integers so the threshold is exact.
FUZZY_JACCARD_NUM = 9
FUZZY_JACCARD_DEN = 10


def fuzzy_form(name: str) -> str:
    """The comparison form of a name: exact normalisation, every character outside ``[a-z0-9' ]`` read as a space,
    runs of spaces collapsed to one, no leading or trailing space."""
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", normalize_exact(name)).split())


def fuzzy_eligible(form: str) -> bool:
    """Long enough (``FUZZY_MIN_CHARS`` characters or ``FUZZY_MIN_WORDS`` words) and varied enough: the Shannon entropy
    of its characters, spaces left out, is at least ``FUZZY_MIN_ENTROPY_BITS`` bits."""
    if len(form) < FUZZY_MIN_CHARS and len(form.split()) < FUZZY_MIN_WORDS:
        return False
    letters = Counter(form.replace(" ", ""))
    n = sum(letters.values())
    if n == 0:
        return False
    entropy = -sum(p * math.log2(p) for p in (count / n for count in letters.values()))
    return entropy >= FUZZY_MIN_ENTROPY_BITS


def trigram_set(form: str) -> frozenset[str]:
    """Character trigrams of the form with its spaces removed; a string shorter than two characters is its own item."""
    chars = form.replace(" ", "")
    grams = frozenset(map("".join, zip(chars, chars[1:], chars[2:], strict=False)))
    return grams if len(chars) >= 2 else frozenset(filter(None, [chars]))


def _min_overlap(size: int) -> int:
    """The fewest shared trigrams a set of ``size`` can have with a set it joins: ceil(size * NUM / DEN)."""
    return (FUZZY_JACCARD_NUM * size + FUZZY_JACCARD_DEN - 1) // FUZZY_JACCARD_DEN


def _joins(shared: int, size_a: int, size_b: int) -> bool:
    """Jaccard ``shared / union`` at or above the threshold, in integers."""
    return FUZZY_JACCARD_DEN * shared >= FUZZY_JACCARD_NUM * (size_a + size_b - shared)


def similar_pairs(sets: Sequence[frozenset[str]]) -> list[tuple[int, int]]:
    """Every index pair ``(i, j)``, i < j, whose sets have Jaccard >= NUM/DEN: an exact set-similarity join.

    Candidates come from prefix filtering. Tokens are ranked by ascending document frequency over ``sets`` (ties by
    the token), each set is read in that rank, and its prefix is its first ``size - _min_overlap(size) + 1`` tokens.
    Two sets reaching the threshold share at least ``_min_overlap`` of the larger size, so their prefixes share a
    token; no qualifying pair is missed. A size filter drops pairs whose sizes alone rule the threshold out
    (``DEN * small >= NUM * large``), and every candidate is checked by its exact intersection. Empty sets join nothing.
    """
    frequency = Counter(token for tokens in sets for token in tokens)
    ranked = {i: sorted(tokens, key=lambda t: (frequency[t], t)) for i, tokens in enumerate(sets) if tokens}
    postings: defaultdict[str, list[int]] = defaultdict(list)
    found: list[tuple[int, int]] = []
    for i in sorted(ranked, key=lambda k: (len(ranked[k]), k)):
        size = len(ranked[i])
        prefix = ranked[i][: size - _min_overlap(size) + 1]
        seen: set[int] = set()
        for token in prefix:
            for j in postings[token]:
                if j in seen or FUZZY_JACCARD_DEN * len(ranked[j]) < FUZZY_JACCARD_NUM * size:
                    continue
                seen.add(j)
                if _joins(len(sets[i] & sets[j]), size, len(ranked[j])):
                    found.append((min(i, j), max(i, j)))
        for token in prefix:
            postings[token].append(i)
    return sorted(found)


def fuzzy_clusters(
    unresolved: Iterable[str], existing: dict[str, list[str]], weight: dict[str, int] | None = None
) -> dict[str, tuple[str, str]]:
    """Fuzzy step of the ladder.

    ``unresolved`` are batch keys with no exact hit; ``existing`` maps an existing node's key to its uuids. Pairs
    with Jaccard >= 0.9 over 3-gram shingles, both high-entropy, at least one side unresolved, are joined (candidates
    from ``similar_pairs``, then exact Jaccard). A cluster resolves to its single existing node; with several existing
    nodes it is ambiguous and its batch keys stay separate; with none, the batch keys merge onto one representative
    (highest weight, then smallest key), so the outcome never depends on input order. Returns only keys that moved or
    are ambiguous: key -> ("existing", uuid) | ("batch", representative key) | ("ambiguous", "").

    A pair whose possessive determiners differ never joins ("our friendship" vs "your friendship" name different
    possessors' things however close the strings are). Equality is required per pair, so every cluster carries one
    possessor signature.
    """
    weights = weight or {}
    # One record per distinct batch key, then one per stored node (a key listed with several uuids is several nodes).
    records: list[tuple[bool, str, str]] = [(False, key, "") for key in sorted(set(unresolved))]
    records.extend((True, key, str(uuid)) for key in sorted(existing) for uuid in sorted(existing[key]))
    matchable = [index for index, (_, key, _) in enumerate(records) if fuzzy_eligible(fuzzy_form(key))]
    shapes = [trigram_set(fuzzy_form(records[index][1])) for index in matchable]

    linked: defaultdict[int, set[int]] = defaultdict(set)
    for left, right in similar_pairs(shapes):
        a, b = matchable[left], matchable[right]
        both_stored = records[a][0] and records[b][0]
        if not both_stored and possessors(records[a][1]) == possessors(records[b][1]):
            linked[a].add(b)
            linked[b].add(a)

    out: dict[str, tuple[str, str]] = {}
    placed: set[int] = set()
    for seed in sorted(linked):
        if seed in placed:
            continue
        placed.add(seed)
        frontier, component = [seed], []
        while frontier:
            here = frontier.pop()
            component.append(records[here])
            for there in sorted(linked[here] - placed):
                placed.add(there)
                frontier.append(there)
        out.update(_settle_component(component, weights))
    return out


def _settle_component(component: list[tuple[bool, str, str]], weights: dict[str, int]) -> dict[str, tuple[str, str]]:
    """Where a joined component's batch keys go: its one stored node; nowhere when it holds several (ambiguous); else
    onto the batch key with the highest weight, ties to the smallest key."""
    stored = {uuid for is_node, _, uuid in component if is_node}
    keys = [key for is_node, key, _ in component if not is_node]
    if len(stored) > 1:
        return dict.fromkeys(keys, ("ambiguous", ""))
    if stored:
        return dict.fromkeys(keys, ("existing", stored.pop()))
    keeper = sorted(keys, key=lambda key: (-weights.get(key, 0), key))[0]
    return {key: ("batch", keeper) for key in keys if key != keeper}


# --- Filler analysis (POS from spaCy; tag classes, never word lists) -----------------------------------------------
_NAME_POS = frozenset({"PROPN", "NOUN"})


@lru_cache(maxsize=4096)
def _parse(text: str):
    """The spaCy doc of a unit text (mood and force are read from it)."""
    return _nlp()(text)


@lru_cache(maxsize=4096)
def possessors(key: str) -> tuple[str, ...]:
    """The possessive determiners of a name (spaCy morph ``Poss=Yes``), lowercased, in order."""
    return tuple(t.lower_ for t in _nlp()(key) if "Yes" in t.morph.get("Poss", []))


@lru_cache(maxsize=4096)
def analyse(text: str) -> tuple[str, str, bool]:
    """(surface, key, referring) for a filler span.

    Leading ADP/DET are stripped; key = ``lemma_key`` of the stripped surface. Referring = the span's syntactic head
    is a noun or proper noun, the phrase is not a wh-phrase (Penn tag W*), and it is not ``non_identifying`` (judged
    on the phrase with its determiners, after leading prepositions).
    """
    doc = _nlp()(text)
    start = 0
    while start < len(doc) and doc[start].pos_ == "ADP":
        start += 1
    phrase = doc[start:]
    while start < len(doc) and doc[start].pos_ in _STRIP_LEAD:
        start += 1
    kept = doc[start:]
    if not len(kept):
        return text, normalize_exact(text), False
    referring = kept.root.pos_ in _REFERRING_HEAD and not kept[0].tag_.startswith("W") and not non_identifying(phrase)
    return kept.text, lemma_key(kept.text), referring


@lru_cache(maxsize=4096)
def _short_name(text: str) -> bool:
    """One capitalised nominal token: the shape of a vocative or short name ("Mel", "Caro")."""
    doc = _nlp()(text)
    return len(doc) == 1 and doc[0].pos_ in _NAME_POS and text[:1].isupper()


def role_key(role: str) -> str:
    """Neo4j-safe property stem for a role (``argm-tmp`` -> ``argm_tmp``)."""
    return re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_")


class _Filler:
    """One classified role filler: an entity candidate (key) or an Event property (text only).

    Under ``referent_policy`` structure a filler that splits holds its referents in ``parts`` (no key of its own),
    the names lifted from inside it in ``inner``, and a filler built from a decomposition its ``mention``
    (``g1_mentions.Mention``) and its conjunct's own text (``conjunct``); a referent whose mention indexes the unit's
    own parse (a decomposition on it, as captured or realigned) holds those indices in ``doc_tokens``. Under phrase
    all five stay empty."""

    __slots__ = (
        "alias",
        "conjunct",
        "doc_tokens",
        "inner",
        "joined_from",
        "key",
        "mention",
        "named",
        "parts",
        "person",
        "resolved",
        "surface",
        "text",
    )

    def __init__(
        self,
        text: str,
        key: str | None = None,
        surface: str = "",
        person: bool = False,
        alias: str | None = None,
        resolved: bool = False,
        *,
        parts: tuple[_Filler, ...] = (),
        inner: tuple[_Filler, ...] = (),
        mention: Mention | None = None,
        conjunct: str = "",
    ):
        self.text, self.key, self.surface, self.person, self.alias = text, key, surface, person, alias
        self.parts, self.inner, self.mention, self.conjunct = parts, inner, mention, conjunct
        #: The unit-doc indices of this referent's mention (``_classify``); () when unknown (no mention, or one on
        #: another parse: a fallback decomposition, a picked antecedent).
        self.doc_tokens: tuple[int, ...] = ()
        #: The filler is a pronoun bound to an antecedent: rendered by the entity's name, not by its surface.
        self.resolved = resolved
        #: Exact key of the joined name this filler mentions (``_JoinNames.match``): who it was before the join.
        self.joined_from: str | None = None
        #: A guarded name mention (``_JoinNames.filler``): a forward run follows even a common-noun route for it.
        self.named = False


class _People:
    """The person roster and who "you" is. A full build: every speaker and deixis name of the conversation, and the
    conversation-wide ``is_explicit``. A forward run (``roster`` given): the stored roster, grown message by message
    (``grow``), and "you" per message (``explicit_of``: message id -> explicit); session speakers stay whole-session."""

    def __init__(
        self, records: list[dict], roster: Sequence[str] | None = None, explicit_of: Mapping[str, bool] | None = None
    ) -> None:
        self.explicit = is_explicit(records)
        self.explicit_of = explicit_of
        self.display: dict[str, str] = {}
        self.sessions: defaultdict[str, set[str]] = defaultdict(set)
        if roster is not None:
            for name in roster:
                self.add(name)
            for rec in records:
                self.sessions[rec["session_id"]].add(rec["speaker"])
            return
        for rec in records:
            self.add(rec["speaker"])
            self.sessions[rec["session_id"]].add(rec["speaker"])
        for rec in records:
            self._deixis(rec)

    def _deixis(self, rec: Mapping) -> None:
        for ev in rec["events"]:
            for spec in ev["roles"].values():
                ref = spec.get("referent") or {}
                if ref.get("status") == "deixis":
                    self.add(_person_name(ref["text"]))

    def grow(self, rec: Mapping) -> None:
        """A forward run reaches ``rec``: its speaker and deixis names join the roster from here on."""
        self.add(rec["speaker"])
        self._deixis(rec)

    def add(self, name: str) -> None:
        self.display.setdefault(normalize_exact(name), name)

    def addressee(self, rec: Mapping) -> str | None:
        explicit = self.explicit if self.explicit_of is None else self.explicit_of[rec["message_id"]]
        return addressee_of(
            rec.get("addressees", []),
            rec["speaker"],
            self.sessions.get(rec["session_id"], set()),
            explicit=explicit,
        )

    def bind_short(self, text: str, rec: Mapping) -> str | None:
        """A short name that is a strict prefix of exactly one of {addressee, speaker} binds to that person."""
        low = normalize_exact(text)
        if len(low) < 3 or not _short_name(text):
            return None
        hits = {p for p in (self.addressee(rec), rec["speaker"]) if p and normalize_exact(p).startswith(low)}
        hits = {p for p in hits if normalize_exact(p) != low}
        return next(iter(hits)) if len(hits) == 1 else None


def _referents(f: _Filler) -> list[_Filler]:
    """The role's referents: the filler itself when it has a key, else its parts (structure; none under phrase)."""
    return [f] if f.key is not None else list(f.parts)


def _all_keyed(f: _Filler) -> list[_Filler]:
    """Every keyed filler a role filler yields: its referents, then the names lifted from inside it."""
    return _referents(f) + list(f.inner)


def unit_frames(rec: Mapping[str, Any], excluded: Collection[str]) -> dict[int, tuple[list[int], list[list[int]]]]:
    """Event index -> (anchor, role spans) of every non-excluded event of the unit (``g1_mentions.decompose``'s
    nested-frame input; an event's own entry is never passed to its own fillers)."""
    return {
        i: ([int(a) for a in ev["anchor"]], [list(spec.get("span") or []) for spec in ev["roles"].values()])
        for i, ev in enumerate(rec["events"])
        if f"{rec['unit']}#{i}" not in excluded
    }


def conversation_names(records: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """The names of the conversation (``g1_mentions.known_names`` over the projection's own docs of the units)."""
    return frozenset().union(*(known_names(_parse(r["text"])) for r in records))


def _deictic(head, rec: Mapping[str, Any], people: _People) -> str | None:
    """The person a first- or second-person personal pronoun conjunct names (spaCy morph ``PronType=Prs`` with
    ``Person=1`` / ``Person=2``, not possessive): the speaker, or the message's addressee when there is one. A
    whole-filler pronoun is bound by the capture's ``deixis`` status instead."""
    morph = head.morph
    if "Prs" not in morph.get("PronType", []) or "Yes" in morph.get("Poss", []):
        return None
    person = morph.get("Person", [])
    if "1" in person:
        return rec["speaker"]
    if "2" in person:
        return people.addressee(rec)
    return None


def _conjunct_text(doc, span: Iterable[int], mention: Mention) -> str:
    """A conjunct's own text before the nested-frame cut: its head's dependents inside the span, without the head's
    conjunct separators (``conj``, ``cc``, ``punct``)."""
    toks = set(mention.tokens)
    head = _mention_head(doc, mention)
    inside = set(span) & {t.i for t in head.subtree}
    for c in head.children:
        if c.dep_ in ("conj", "cc", "punct"):
            inside -= {t.i for t in c.subtree}
    return _tokens_text(doc, inside | toks)


def _inner_filler(m: Mention, shown: str, rec: Mapping[str, Any], people: _People, names: _JoinNames | None) -> _Filler:
    """A name lifted from inside a filler: a roster person, a joined name, else its own key."""
    if normalize_exact(m.surface) in people.display:
        return _Filler(shown, normalize_exact(m.surface), m.surface, person=True, mention=m, conjunct=m.surface)
    if names is not None and (hit := names.filler(m.surface, shown, unit=str(rec.get("text") or ""))) is not None:
        hit.mention, hit.conjunct = m, m.surface
        return hit
    return _Filler(shown, m.key, m.surface, mention=m, conjunct=m.surface)


def _mention_filler(
    m: Mention,
    head,
    conj: str,
    shown: str,
    rec: Mapping[str, Any],
    people: _People,
    tag: str | None,
    names: _JoinNames | None,
    *,
    judge: bool,
    picked: bool,
) -> _Filler | None:
    """Today's per-filler ladder on one mention (``head``: its head token): a deictic pronoun conjunct (``_deictic``,
    not for a picked antecedent), roster person, joined name, short name (not for a picked antecedent), referring (a
    common-noun conjunct on its own text, when ``judge``; one conjunct was judged whole by the caller), owner."""
    surface = m.surface
    if judge and not picked and (deixis := _deictic(head, rec, people)):  # "Kira and I": I is the speaker
        name = _person_name(deixis)
        return _Filler(shown, normalize_exact(name), name, person=True, mention=m, conjunct=conj)
    if normalize_exact(surface) in people.display:
        return _Filler(shown, normalize_exact(surface), surface, person=True, resolved=picked, mention=m, conjunct=conj)
    unit = None if picked else str(rec.get("text") or "")
    if names is not None and (hit := names.filler(surface, shown, resolved=picked, unit=unit)) is not None:
        hit.mention, hit.conjunct = m, conj
        return hit
    if not picked and (bound := people.bind_short(surface, rec)):
        return _Filler(shown, normalize_exact(bound), bound, person=True, alias=surface, mention=m, conjunct=conj)
    # a name conjunct refers (its name-hood was read in the unit's parse; a lone "Milo" parsed alone tags INTJ)
    if judge and not m.name and not is_referring(conj, function_tag=tag):
        return None
    if not picked and not (m.name and m.kind_nouns):  # a name with a kind noun ("my friend Milo") shows the name chain
        speaker, addressee = rec["speaker"], people.addressee(rec)
        owned = owner_key(analyse(conj)[0], speaker=speaker, addressee=addressee)
        if owned is not None:  # the conjunct is the speaker's / addressee's: its core, owner-qualified
            core = owner_key(surface, speaker=speaker, addressee=addressee) or owned
            return _Filler(shown, core[0], core[1], resolved=True, mention=m, conjunct=conj)
    return _Filler(shown, m.key, surface, person=m.key in people.display, resolved=picked, mention=m, conjunct=conj)


def _by_mention(
    dec: Decomposition,
    doc,
    span: Iterable[int],
    judged: str,
    shown: str,
    rec: Mapping[str, Any],
    people: _People,
    tag: str | None,
    names: _JoinNames | None,
    counts: Counter[str] | None,
    *,
    picked: bool = False,
) -> _Filler:
    """Structure: a role filler -> its referents. The whole filler (``judged``) is judged referring first, as today
    (a clausal or predicative filler keys nothing); then each mention runs the per-filler ladder, judged on its
    conjunct's own text when the filler splits. One referent: that filler; several: a keyless filler holding them
    (``parts``); none: a text filler keeping its first mention. Names lifted from inside a referring filler (or one
    whose only head was light) are ``inner``."""
    span = list(span)
    inner = () if picked else tuple(_inner_filler(m, shown, rec, people, names) for m in dec.names)
    first = dec.mentions[0] if dec.mentions else None
    if not is_referring(judged, function_tag=tag):  # a clausal or predicative filler keys nothing, inner names too
        return _Filler(shown, mention=first)
    if counts is not None:  # fallback mentions whose head is not nominal ("force" in "until … force a choice")
        counts["referents_nonnominal_dropped"] += dec.dropped
    if first is None:  # a light head with no complement ("a lot") refers to nothing
        if counts is not None and dec.light:
            counts["referents_light_dropped"] += 1
        return _Filler(shown, inner=inner)
    one = len(dec.mentions) == 1
    parts: list[_Filler] = []
    for m in dec.mentions:
        conj = judged if one else _conjunct_text(doc, span, m)
        f = _mention_filler(
            m, _mention_head(doc, m), conj, shown, rec, people, tag, names, judge=not one, picked=picked
        )
        if f is not None:
            parts.append(f)
        elif counts is not None:  # a conjunct that refers to nothing on its own ("a vendor" in "Kira and a vendor")
            counts["referents_conjunct_dropped"] += 1
    if len(parts) == 1:
        parts[0].inner = inner
        return parts[0]
    if not parts:
        return _Filler(shown, inner=inner, mention=first)
    return _Filler(shown, parts=tuple(parts), inner=inner)


def _classify(
    spec: dict,
    role: str,
    rec: dict,
    people: _People,
    tag: str | None = None,
    names: _JoinNames | None = None,
    *,
    policy: str = "phrase",
    frames: Sequence[tuple[Sequence[int], Sequence[Sequence[int]]]] = (),
    known: Collection[str] = (),
    counts: Counter[str] | None = None,
) -> _Filler | None:
    """Classify one role filler. ``tag`` is the PropBank function tag of its role (``g1_roles.function_tag``): a
    predicative (PRD) noun phrase introduces no referent (``g1_referring.is_referring``).

    ``names`` are the identity joins' names (``_JoinNames``; None without joins): a filler naming one is an entity
    under the canonical name's key (its possessive under the canonical possessive's key) whatever its parse (a lone
    "Dee" can tag as ADJ), keeping its own surface.

    ``policy`` structure (``referent_policy``): after the same status, time-role and whole-filler person / joined-name
    / short-name checks, the filler is decomposed into its mentions (``g1_mentions.decompose`` on the unit's doc, with
    the unit's other frames ``frames`` and the conversation's names ``known``) and keyed per mention (``_by_mention``).
    ``counts`` takes the decomposition's ``referents_fallback`` / ``referents_realigned`` /
    ``referents_nonnominal_dropped`` / ``referents_light_dropped`` counts."""
    structure = policy == "structure"
    ref = spec.get("referent") or {}
    status = ref.get("status")
    text = spec["text"].strip()
    if status == "expletive":
        return None
    if status == "open":
        return _Filler(text) if text else None
    if status == "deixis":
        name = _person_name(ref["text"])
        return _Filler(text, normalize_exact(name), name, person=True)
    if status == "picked":
        if names is not None and (hit := names.filler(ref["text"].strip(), text, resolved=True)) is not None:
            return hit
        surface, key, _ = analyse(ref["text"].strip())
        referring = is_referring(ref["text"].strip(), function_tag=tag)
        if not referring and key not in people.display:
            return _Filler(text)
        if structure and key not in people.display:
            antecedent = ref["text"].strip()
            dec = decompose_text(antecedent, known)
            own = _own_parse(antecedent)
            return _by_mention(
                dec, own, range(len(own)), antecedent, text, rec, people, tag, names, counts, picked=True
            )
        return _Filler(text, key, surface, person=key in people.display, resolved=True)
    if not text:
        return None
    # Time-role fillers are never entities (g1_time owns them). Keyed by the function tag, not the role name: this
    # generalises argm-tmp to NUMBERED roles the frames tag TMP (e.g. reschedule.01 arg3), even a person name there.
    if role_construction(tag).referent == NONE:
        return _Filler(text)
    if normalize_exact(text) in people.display:
        return _Filler(text, normalize_exact(text), text, person=True)
    if names is not None and (hit := names.filler(text, text, unit=str(rec.get("text") or ""))) is not None:
        return hit
    bound = people.bind_short(text, rec)
    if bound:
        return _Filler(text, normalize_exact(bound), bound, person=True, alias=text)
    if structure:
        doc, span = _parse(rec["text"]), list(spec.get("span") or [])
        dec = decompose(doc, span, text, frames, known)
        if dec.fallback:  # the mention tokens index the filler's own parse
            doc = _own_parse(text)
            span = list(range(len(doc)))
        elif dec.realigned:  # the filler's text found at other indices of the unit's parse
            span = list(dec.realigned)
        if counts is not None:
            counts["referents_fallback"] += dec.fallback
            counts["referents_realigned"] += bool(dec.realigned)
        out = _by_mention(dec, doc, span, text, text, rec, people, tag, names, counts)
        if not dec.fallback:  # the mentions index the unit's parse: the identity test reads them (``_bearer_tokens``)
            for x in _referents(out):
                if x.mention is not None:
                    x.doc_tokens = x.mention.tokens
        return out
    surface, key, _ = analyse(text)
    if not is_referring(text, function_tag=tag):
        return _Filler(text)
    owned = owner_key(surface, speaker=rec["speaker"], addressee=people.addressee(rec))
    if owned is not None:  # "my family" is the speaker's family, not one shared node
        return _Filler(text, owned[0], owned[1], resolved=True)
    return _Filler(text, key, surface, person=key in people.display)


def _kinds(person: bool, tag: str | None) -> set[str]:
    """Entity kinds are language-layer, never card-derived: Person from the referent; any other kind is the
    one the filled role's PropBank function tag confers (``g1_constructions.role_construction``; LOC -> Place, for an
    ``argm-loc`` adjunct and a numbered location argument alike)."""
    kinds = {"Person"} if person else set()
    kind = role_construction(tag).kind
    if kind and not person:  # Person and Place are mutually exclusive kinds
        kinds.add(kind)
    return kinds


class _Candidate:
    def __init__(self) -> None:
        self.surfaces: Counter[str] = Counter()
        self.kinds: set[str] = set()
        #: ``referent_policy`` structure: what the key's fillers say about its kind (``g1_kinds``); unread under phrase.
        self.evidence = KindEvidence()
        self.aliases: set[str] = set()
        self.person: str | None = None


def _existing_index(existing: list[dict], policy: str = "phrase") -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Stored nodes by key: the stored ``name_key`` wherever there is one, else the key of the stored name
    (``analyse``; ``g1_mentions.structure_key`` under ``referent_policy`` structure)."""
    by_key: defaultdict[str, list[str]] = defaultdict(list)
    by_uuid: dict[str, dict] = {}
    for row in existing:
        labels = set(row.get("labels") or [])
        if labels & NOT_CANDIDATES or not row.get("name"):
            continue
        key = row.get("name_key") or (structure_key(row["name"]) if policy == "structure" else analyse(row["name"])[1])
        by_key[key].append(row["uuid"])
        by_uuid[row["uuid"]] = row
    return dict(by_key), by_uuid


class _Builder:
    def __init__(self, group_id: str, created_at: str) -> None:
        self.group_id, self.created_at = group_id, created_at
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}

    def node(self, kinds: Iterable[str], uuid: str, name: str, **extra: object) -> str:
        if uuid not in self.nodes:
            kinds = list(kinds)
            props = {"uuid": uuid, "name": name, "group_id": self.group_id, "created_at": self.created_at}
            if "Entity" in kinds:
                props |= {"summary": "", "name_embedding": None}
            props |= {k: v for k, v in extra.items() if v is not None}
            self.nodes[uuid] = {"labels_neo4j": kinds, "props": props}
        return uuid

    def edge(self, kind: str, source: str, target: str, name: str, seed: tuple, **props: object) -> dict:
        uuid = _sha("edge", self.group_id, kind, source, target, name, *seed)
        base = {"uuid": uuid, "group_id": self.group_id, "created_at": self.created_at}
        if kind == "RELATES_TO":
            base |= {"name": name, "source_uuid": source, "target_uuid": target, "fact_embedding": None}
        merged = base | {k: v for k, v in props.items() if v is not None}
        row = {"type": kind, "source": source, "target": target, "props": merged}
        self.edges[uuid] = row
        return row


def _resolve_entities(
    cands: dict[str, _Candidate], existing: list[dict], group_id: str, policy: str = "phrase"
) -> tuple[dict[str, str], dict[str, dict], dict]:
    """Ladder steps 2-4 over candidate keys -> uuid. Step 1 (referent identity) already chose the keys."""
    ex_by_key, ex_by_uuid = _existing_index(existing, policy)
    own = {key: _sha("ent", group_id, key) for key in cands}
    resolved: dict[str, str] = {}
    stats: dict[str, Any] = {
        "merged_exact": 0,
        "merged_fuzzy": 0,
        "ambiguous": 0,
        "ambiguous_keys": [],
        "fuzzy_merges": [],
    }
    unresolved: list[str] = []
    for key in sorted(cands):
        hits = ex_by_key.get(key, [])
        if own[key] in hits:
            resolved[key] = own[key]
        elif len(hits) == 1:
            resolved[key] = hits[0]
            stats["merged_exact"] += 1
        elif hits:
            resolved[key] = own[key]
            stats["ambiguous"] += 1
            stats["ambiguous_keys"].append(key)
            log.info("ambiguous exact match for %r: %d existing nodes, kept separate", key, len(hits))
        else:
            unresolved.append(key)
    weight = {k: sum(c.surfaces.values()) for k, c in cands.items()}
    moves = fuzzy_clusters(unresolved, ex_by_key, weight)
    for key in unresolved:
        how, target = moves.get(key, ("own", ""))
        if how == "existing":
            resolved[key] = target
        elif how == "batch":
            resolved[key] = own[target]
        else:
            resolved[key] = own[key]
        if how in ("existing", "batch"):
            stats["merged_fuzzy"] += 1
            stats["fuzzy_merges"].append([key, resolved[key]])
        elif how == "ambiguous":
            stats["ambiguous"] += 1
            stats["ambiguous_keys"].append(key)
            log.info("ambiguous fuzzy match for %r, kept separate", key)
    rep_key = {own[k]: k for k in cands if resolved[k] == own[k]}
    # A hit on the uuid this projection derives itself is its own earlier write, not a foreign node to reuse.
    reused = {u: ex_by_uuid[u] for u in set(resolved.values()) if u in ex_by_uuid and u not in rep_key}
    return resolved, reused, stats | {"rep_key": rep_key}


def _name_forms(surface: str) -> list[str]:
    """The keys a mention of a name can take in this projection: its exact key and its lemma key (``analyse``)."""
    return [normalize_exact(surface), analyse(surface)[1]]


def _possessive_forms(surface: str) -> list[str]:
    """Both keys of a name's possessive ("Dee's" -> ``dee's`` and ``dee 's``)."""
    owned = f"{surface}'s"
    return [normalize_exact(owned), analyse(owned)[1]]


def _region(text: str, unit: str) -> tuple[int, int] | None:
    """Where the filler ``text`` sits in its unit's text (first occurrence, exact case first), else None."""
    at = unit.find(text)
    if at < 0:
        at = unit.lower().find(text.lower())
    return (at, at + len(text)) if at >= 0 and text else None


class _JoinNames:
    """The names of the identity joins, as the projection matches mentions of them.

    A mention whose exact form (or, after a leading preposition/determiner strip, whose surface) is a joined name
    resolves to the canonical name's key; ``<name>'s`` resolves to the canonical name's possessive key ("Dee's" is the
    entity "Ana's" is). Whether the mention names the person is ``identity.alias_occurs`` on the filler's own span in
    its unit (the predicate retrieval shares): a name that is not a common noun (``identity.is_common_noun``) matches
    in any case and position; a common-noun name only capitalised and not sentence-initial. After "my name is Rose",
    "I asked Rose" joins; "the rose", "I like roses" and "Rose is lovely." do not; after "I am Mary", "mary likes
    mangoes" joins. A filler not found in its unit (a picked antecedent) is judged by its own form: capitalised, or
    not a common noun."""

    def __init__(self, joins: Joins, speakers: Collection[str]) -> None:
        spoken = {normalize_exact(sp) for sp in speakers}
        #: exact key of a joined name -> (canonical key, possessive key of the canonical, common noun?)
        self.by_name: dict[str, tuple[str, str, bool]] = {}
        for root in sorted(joins.aliases):
            owned = _possessive_forms(joins.display[root])[1]
            for surface in (joins.display[root], *joins.aliases[root]):
                key = normalize_exact(surface)
                if key not in spoken:  # a speaker's name already resolves to the speaker
                    self.by_name.setdefault(key, (root, owned, is_common_noun(key)))

    def common(self, key: str) -> bool:
        return self.by_name.get(key, ("", "", True))[2]

    def filler(self, text: str, shown: str, resolved: bool = False, unit: str | None = None) -> _Filler | None:
        region = _region(text, unit) if unit else None
        for form in dict.fromkeys((text, analyse(text)[0])):
            key = normalize_exact(form)
            possessive = key.endswith("'s") and key[:-2] in self.by_name
            name = key[:-2] if possessive else key
            if name not in self.by_name:
                continue
            if region is not None and unit is not None:
                if not alias_occurs(name, unit, is_common=self.common, within=region):
                    continue
            elif not (capitalised(form) or not self.common(name)):
                continue
            root, owned, _ = self.by_name[name]
            hit = _Filler(shown, owned if possessive else root, form, resolved=resolved)
            hit.joined_from = None if possessive else name
            hit.named = True
            return hit
        return None


def _join_entities(
    joins: Joins,
    names: _JoinNames,
    resolved: dict[str, str],
    fixed: set[str],
    group_id: str,
    rep_key: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Identity joins applied to the resolved keys (in place). Mentions matched by ``_JoinNames`` already carry the
    canonical keys; other key forms of a joined name that is not a common noun (a lemma key, a possessive after a
    preposition) move, with every key sharing their uuid, onto the canonical name's uuid, and its possessive forms onto
    the canonical possessive's entity (its resolved uuid, else the uuid "Ana's" would get, added to ``rep_key``).
    Speakers' uuids (``fixed``) never move. Returns (canonical uuid -> canonical key, possessive uuid -> canonical
    display name)."""

    def move(forms: Iterable[str], target: str) -> None:
        moved = {resolved[f] for f in forms if f in resolved} - {target} - fixed
        for key, uuid in resolved.items():
            if uuid in moved:
                resolved[key] = target

    joined: dict[str, str] = {}
    owners: dict[str, str] = {}
    for root in sorted(joins.aliases):
        display = joins.display[root]
        plain = [x for x in (display, *joins.aliases[root]) if not names.common(normalize_exact(x))]
        forms = [root, *(f for x in plain for f in _name_forms(x))]
        target = next((resolved[f] for f in forms if f in resolved), None)
        if target is not None:
            move(forms, target)
            joined[target] = root
        canon_owned = _possessive_forms(display)
        owned = [canon_owned[1], *(f for x in plain if x != display for f in _possessive_forms(x))]
        if any(f in resolved for f in owned):
            owner = next((resolved[f] for f in reversed(canon_owned) if f in resolved), None)
            if owner is None:  # the canonical possessive is never mentioned: the uuid "Ana's" would get
                owner = _sha("ent", group_id, canon_owned[1])
                rep_key.setdefault(owner, canon_owned[1])
            move(owned, owner)
            owners[owner] = display
    return joined, owners


def _resolve_full(
    records: list[dict],
    people: _People,
    group_id: str,
    existing: list[dict] | None,
    excluded: frozenset[str],
    tag_of: Callable[[str, str], str],
    identity_cards,
    policy: str = "phrase",
) -> tuple:
    """A full build's pass 1 and resolution (the first mint): identity joins over every claim of the input, every
    filler classified, keys resolved by the ladder (``_resolve_entities``) and the joins applied (``_join_entities``).
    Under ``policy`` structure every filler is decomposed (``_classify``) and each of its keyed fillers
    (``_all_keyed``) is a candidate; the ``referents_*`` counts join the stats."""
    speakers = list(people.display.values())
    #: event key -> the identity claims its frame makes (claims are per event, so their concatenation in record and
    #: event order is ``identity_claims(records, ...)``).
    claims_of = {
        f"{rec['unit']}#{i}": identity_claims([{**rec, "events": [ev]}], identity_cards, speakers)
        for rec in records
        for i, ev in enumerate(rec["events"])
    }
    joins = resolve_joins([c for claims in claims_of.values() for c in claims], speakers)
    #: events whose identity claim was accepted (its two names are one entity now): they write no fact or line
    claim_events = (
        {k for k, claims in claims_of.items() if any(_accepted(c, joins) for c in claims)} if joins.canonical else set()
    )
    names = _JoinNames(joins, speakers) if joins.canonical else None

    # Pass 1: classify every filler; collect entity candidates by key (ladder step 1 = referent identity).
    cands: defaultdict[str, _Candidate] = defaultdict(_Candidate)
    classified: dict[tuple[str, int, str], _Filler | None] = {}
    structure = policy == "structure"
    known = conversation_names(records) if structure else frozenset()
    counts: Counter[str] = Counter()
    for rec in records:
        speaker = cands[normalize_exact(rec["speaker"])]
        speaker.surfaces[rec["speaker"]] += 1
        speaker.kinds.add("Person")
        speaker.evidence.person = True
        speaker.person = people.display[normalize_exact(rec["speaker"])]
        frames = unit_frames(rec, excluded) if structure else {}
        for index, ev in enumerate(rec["events"]):
            if f"{rec['unit']}#{index}" in excluded:
                continue
            extra: dict[str, Any] = (
                {
                    "policy": policy,
                    "frames": [v for i, v in frames.items() if i != index],
                    "known": known,
                    "counts": counts,
                }
                if structure
                else {}
            )
            tags: dict[str, str] = {}
            fillers: dict[str, _Filler] = {}
            for role, spec in ev["roles"].items():
                tag = tag_of(ev["roleset"], role)
                f = _classify(spec, role, rec, people, tag, names, **extra)
                classified[(rec["unit"], index, role)] = f
                if f is None:
                    continue
                if structure:
                    counts["referents_split"] += len(f.parts) >= 2
                    counts["referents_inner_names"] += len(f.inner)
                    tags[role], fillers[role] = tag, f
                    for x in _referents(f):
                        _candidate(cands, x, set(), people, filler_evidence(x.person, x.mention, tag))
                    for x in f.inner:  # a name inside the filler does not fill the role: no role evidence
                        _candidate(cands, x, set(), people, filler_evidence(x.person, x.mention, None))
                    continue
                for x in _referents(f):
                    _candidate(cands, x, _kinds(x.person, tag), people)
                for x in f.inner:  # a name inside the filler does not fill the role: no role-conferred kind
                    _candidate(cands, x, _kinds(x.person, None), people)
            if structure:
                for role, ev_of in frame_evidence(tags, {r: f.mention for r, f in fillers.items()}).items():
                    for x in _referents(fillers[role]):  # "Brask is a company": the kind noun is about Brask
                        if x.key is not None:
                            cands[x.key].evidence.add(ev_of)

    resolved, reused, stats = _resolve_entities(cands, existing or [], group_id, policy)
    if structure:
        stats |= {k: counts[k] for k in REFERENT_STATS}
    rep_key = stats.pop("rep_key")
    #: canonical uuid -> canonical key of an identity join; ``before`` = key -> uuid as resolved without joins.
    joined: dict[str, str] = {}
    owners: dict[str, str] = {}  # possessive entity uuid -> canonical display name ("Ana" for "Dee's")
    before: dict[str, str] = {}
    if names is not None:
        before = dict(resolved)
        speaker_uuids = {resolved[normalize_exact(r["speaker"])] for r in records}
        joined, owners = _join_entities(joins, names, resolved, speaker_uuids, group_id, rep_key)
        for uuid in set(before.values()) - set(resolved.values()):
            reused.pop(uuid, None)

    return classified, cands, resolved, reused, stats, rep_key, joins, joined, owners, before, claim_events, names


#: The ``dedup`` counts a structure projection adds (``referent_policy`` structure only).
REFERENT_STATS = (
    "referents_fallback",
    "referents_split",
    "referents_light_dropped",
    "referents_inner_names",
    "referents_conjunct_dropped",
    "referents_realigned",
    "referents_nonnominal_dropped",
)


def _candidate(
    cands: Mapping[str, _Candidate],
    f: _Filler,
    kinds: set[str],
    people: _People,
    evidence: KindEvidence | None = None,
) -> None:
    """One keyed filler joins its key's candidate (pass 1 of a full build): its role-conferred ``kinds`` (phrase) or
    its kind ``evidence`` (structure)."""
    if f.key is None:
        return
    cand = cands[f.key]
    cand.surfaces[f.surface] += 1
    cand.kinds |= kinds
    if evidence is not None:
        cand.evidence.add(evidence)
    if f.person:
        cand.person = people.display.get(f.key, f.surface)
    if f.alias:
        cand.aliases.add(f.alias)


def _accepted(claim: Claim, joins: Joins) -> bool:
    """The claim's two names ended up joined (``joins.canonical`` puts them in one group)."""
    holder, name = normalize_exact(claim.holder), normalize_exact(claim.name)
    return holder != name and joins.canonical.get(holder, holder) == joins.canonical.get(name, name)


def _unwritten_by_join(
    fillers: Mapping[str, _Filler],
    refer_all: Mapping[str, list[str]],
    before: Mapping[str, str],
    resolved: Mapping[str, str],
) -> int:
    """How many state edges the identity joins left unwritten: the edges between this predicate's core participants
    (every referent of each core role, ``refer_all``) as they were before the joins (a joined-name mention is its own
    name then) minus the edges written now."""
    core = [role for role in refer_all if is_core_role(role)]
    was = {x.joined_from or before.get(str(x.key), resolved[str(x.key)]) for r in core for x in _referents(fillers[r])}
    now = {u for r in core for u in refer_all[r]}
    return max(len(was) - 1, 0) - max(len(now) - 1, 0)


def relation_name(roleset: str, pos: str = "verb", card: dict | None = None) -> str:
    """Relation name of a roleset from its PropBank predicate POS (fix 1c); a card ``relation`` wins when present.

    verb -> the third-person present of its lemma, upper snake case (thank.01 -> THANKS): regular English morphology
    (-ies after consonant+y; -es after a sibilant or o; else -s), plus English's two suppletive 3sg forms, ``is`` and
    ``has`` (inflection, not a vocabulary list). adjective -> ``IS_<LEMMA>`` (happy.01 -> IS_HAPPY); noun ->
    ``HAS_<LEMMA>``. Adjectives and nouns are never inflected, so "HAPPIES" cannot be produced.
    """
    if card and card.get("relation"):
        return str(card["relation"])
    lemma = roleset.split(".")[0].upper()
    if pos == "adjective":
        return f"IS_{lemma}"
    if pos == "noun":
        return f"HAS_{lemma}"
    head, *rest = roleset.split(".")[0].split("_")
    if head in _SUPPLETIVE_3SG:
        verb = _SUPPLETIVE_3SG[head]
    elif re.search(r"[^aeiou]y$", head):
        verb = head[:-1] + "ies"
    elif re.search(r"(s|sh|ch|x|z|o)$", head):
        verb = head + "es"
    else:
        verb = head + "s"
    return "_".join([verb, *rest]).upper()


def message_order(message_id: str) -> tuple:
    """Numeric order of a message id ("D1:9" before "D1:10"), the id itself breaking ties."""
    return tuple(int(x) for x in re.findall(r"\d+", message_id)), message_id


def message_rank(records: Sequence[Mapping]) -> dict[str, int]:
    """Source order of each message: the position of its first record (records come in source order)."""
    rank: dict[str, int] = {}
    for rec in records:
        rank.setdefault(rec["message_id"], len(rank))
    return rank


def order_key(rank: Mapping[str, int]) -> Callable[[str], tuple]:
    """Sort key by source order; an id outside ``rank`` sorts after them by ``message_order``."""
    return lambda message_id: (0, rank[message_id]) if message_id in rank else (1, message_order(message_id))


def _role_order(role: str) -> tuple[int, str]:
    return (0 if re.fullmatch(r"arg\d", role) else 1, role)


def frame_name(roleset: str, roles: Sequence[tuple[str, str]]) -> str:
    """An Event's name under ``referent_policy`` structure: its frame and its roles, ``own.01: arg0 Kira and Milo;
    arg1 Brask Labs``. ``roles`` are (role, text) in role order; the caller passes the core roles (all when none is
    filled), a referring role's text being its referents' display names joined by " and "."""
    return f"{roleset}: " + "; ".join(f"{role} {text}" for role, text in roles)


#: PropBank role function tag of a cause/stimulus argument. A CAU arg0 is not the bearer of the state.
_CAUSE = CAUSE_TAG


def _direction(refer: dict[str, str], functions: dict[str, str], holder: str | None = None) -> list[tuple[str, str]]:
    """Referring CORE participants (role, uuid) in edge order: the holder (``g1_roles.holder_role``: the
    frame's experiencer, else a non-CAU arg0) is the source when it is referring; then the other core args in
    numeric order, CAU last (cause/stimulus: happy.01 arg0 "for the encouragement" never heads the edge, so the bearer
    happy.01 arg1 f=PPT does). The next referring core arg is the target. Each uuid at its first role."""
    roles = sorted(
        (r for r in refer if is_core_role(r)),
        key=lambda r: (r != holder, functions.get(r) == _CAUSE, _role_order(r)),
    )
    out: list[tuple[str, str]] = []
    for role in roles:
        if refer[role] not in {u for _, u in out}:
            out.append((role, refer[role]))
    return out


def _bearer_tokens(doc, f: _Filler, spec: Mapping[str, Any], is_bearer: Callable[[_Filler], bool]) -> set[int]:
    """The unit-doc indices of the bearer's own words in its role filler ``f`` (the identity test of a summary line):
    the mention tokens of each referent of ``f`` that is the bearer (``doc_tokens``, structure), else the role's
    capture span on the unit's parse (``g1_mentions.doc_span``: as captured when aligned, else realigned; phrase, or a
    referent with no mention on the unit's parse); empty when the parse does not hold the filler."""
    own = [x for x in _referents(f) if is_bearer(x)]
    if own and all(x.doc_tokens for x in own):
        return {i for x in own for i in x.doc_tokens}
    return set(doc_span(doc, [int(i) for i in spec.get("span") or ()], str(spec.get("text") or "").strip()) or ())


def _anchor(doc, entry: dict[str, Any], ev: dict[str, Any]):
    """The predicate's anchor token on the spaCy doc of its unit: the aligned index of the eventuality artifact
    when given, else the recorded G1 anchor; clipped into the doc. None for an empty doc."""
    if not len(doc):
        return None
    index = entry.get("anchor_index")
    if index is None:
        index = min(ev["anchor"] or [0])
    return doc[max(0, min(int(index), len(doc) - 1))]


def _saga_summary(msgs: list[tuple[str, str, str, str]]) -> str:
    """Deterministic saga summary: time span + message count + speakers."""
    speakers = ", ".join(sorted({m[2] for m in msgs}))
    return f"{msgs[0][3]} to {msgs[-1][3]}; {len(msgs)} messages; speakers: {speakers}"


def _saga_vector(saga_vectors: Mapping[str, Mapping[str, object]] | None, episode: str, count: int) -> dict:
    """The centroid properties of one Saga (``project(saga_vectors=...)``); ``{}`` without ``saga_vectors``."""
    if saga_vectors is None:
        return {}
    got = saga_vectors.get(episode)
    if got is None:
        raise ValueError(f"saga vectors: episode {episode!r} has no centroid")
    if got.get("centroid_messages") != count:
        raise ValueError(
            f"saga vectors: episode {episode!r} centroid spans {got.get('centroid_messages')} of {count} messages"
        )
    return dict(got)


def project(
    records: list[dict],
    *,
    eventuality: dict[str, dict],
    episodes: dict[str, str],
    group_id: str,
    created_at: str,
    existing: list[dict] | None = None,
    existing_sagas: list[dict] | None = None,
    excluded: Iterable[str] = (),
    predicate_pos: dict[str, str] | None = None,
    role_functions: dict[str, dict[str, str]] | None = None,
    roleset_index: Mapping[str, PropBankRolesetDefinition] | None = None,
    role_themes: Callable[[str], Mapping[str, Sequence[str]]] | None = None,
    verbnet: Callable[[str], Sequence[str]] | None = None,
    render_mode: str = "b",
    ingest_pass: str | None = None,
    conversation_of: Mapping[str, str] | None = None,
    chain: Chain | None = None,
    run: int = 0,
    referee: Callable | None = None,
    saga_vectors: Mapping[str, Mapping[str, object]] | None = None,
    referent_policy: str = "phrase",
) -> dict:
    """Rows for ``kg_contract.validate_rows`` plus ``dedup``, ``routing`` and ``forward``. Pure: same inputs -> same rows.

    ``chain`` (kernel 1.2): None is a full build, the first mint, resolved over the whole input (unchanged). A chain is
    a forward append of run ``run`` (``graph.forward``): keys resolve against the chain's entities and routes in
    message order, identity claims add routes and ``SAME_AS`` links (edge rows; a stored endpoint is an
    ``external_endpoints`` row), "you" is decided per message, a stored node keeps its name and labels (under
    ``referent_policy`` structure its kind label is decided again over the chain's ``kinds`` evidence and replaces
    the stored one: ``kinds_replace``) and gets its new summary lines as ``summary_append`` / ``summary_append_sources`` / ``summary_append_line_ids`` (parallel lists,
    appended by the writer), with ``summary_append_identity_line_ids`` (the identity
    lines among them). ``existing`` is not read then. Either way ``forward`` says
    what the projection adds to the chain: ``minted`` (``chain.Minted``, by first mention), ``routes``, ``same_as``
    (``chain.SameAs``), ``conflicts`` (``{name, existing, claimed, message_id}``) and ``touched`` (entity uuids written). A forward run's ``referee`` (``stages.referents``) decides specific mentions.

    One ``:Saga`` per episode (``episodes`` maps message id -> external episode id), one immutable
    ``:Episodic`` per message, ``HAS_EPISODE`` saga -> message and ``NEXT_EPISODE`` in message order within the saga.
    Each predicate is routed by its eventuality (``eventuality`` maps ``unit#index`` -> {"label", "p"}):
    occurrence -> ``:Entity:Event`` with role edges; state -> a fact edge (2+ referring participants) or a line of
    the participant's summary (1); evaluation and speech act: see the envelope note below (the speaker-sourced EVALUATES and
    speaker-as-agent speech-act edges are superseded). No event coreference.

    A summary line is rendered by ``render_attributed`` ("[sender, date] clause", plus "(asked)" / "(intended)" /
    "(requested)" when not actual), and the entity's flat ``summary_sources`` lists each line's source Episodic uuid,
    in line order (a line never loses its provenance); the parallel ``summary_line_ids`` holds
    each line's ``g1_ids.summary_line_id`` (entity, source, text: position-free), so a line can be named. The
    parallel ``summary_line_texts`` holds each line whole (a line may contain a newline, so ``summary`` is never split
    to recover lines) and ``summary_identity_line_ids`` the ids of the identity lines, in line order: an asserted,
    actual clause whose predicate is a copula heading a nominal complement (``g1_render.nominal_copula_subject``) and
    whose subject lies in the bearer's filler (by referent, never by name). All five are recomputed from the full
    input, so a rerun reproduces them exactly.

    ``existing`` lists the group's stored ``:Entity`` nodes as ``{"uuid", "name", "labels"}`` (optionally
    ``name_key``); an exact or fuzzy hit reuses that uuid, which is how the kernel continues a Graphiti graph. A reused
    node keeps its stored summary: state lines are only written onto nodes this projection owns.

    ``existing_sagas`` lists the group's stored ``:Saga`` nodes as ``{"uuid", "name", "episodes"}`` (``episodes``:
    its Episodic uuids, most recent first). A Saga whose ``name`` equals an episode id is reused (Graphiti
    ``_get_or_create_saga``: lookup by name + group; the first listed wins), and its most recent episode that this
    projection does not write is chained to the first appended message by ``NEXT_EPISODE`` (Graphiti's append).
    A row may carry ``ingest_pass`` (its stored ordered list) and ``prior`` (its stored messages as
    ``(message_id, Episodic uuid, speaker, timestamp)``, oldest first: the reused saga's summary, start and count then
    span them too; a forward append feeds its chat's open episode so, ``stages.write.chain_sagas``).

    ``saga_vectors`` (episode id -> ``pipeline.saga_vectors.saga_centroids`` properties; the write stage computes them)
    adds ``centroid_embedding`` / ``centroid_model`` / ``centroid_messages`` to every Saga written; each must cover
    the Saga's whole span (its ``prior`` messages included), else ``ValueError``. None: no centroid (``build_dry``).

    Saga provenance: with ``ingest_pass`` set, every saga written records it (``ingest_pass`` list: a
    reused saga appends the pass, never duplicated), ``language_version`` and ``inventory_sha`` (None since 1.8: no
    card inventory; the key stays so a stored Saga's value passes the forward gate), and ``NEXT_SAGA``
    chains the sagas in source order of their first message. A new first saga links from the latest stored saga
    outside this batch with the max ``ended_at`` (then ``started_at``, then row order); a reused first saga is already in the chain and is not linked. With ``ingest_pass=None``
    none of this is written. ``conversation_of`` (message id -> conversation (chat) id) runs one chain per
    conversation: a saga links only from the previous saga of its own chat (the chat of its first message), so
    interleaved chats never share a chain. None (or every message in one chat) is the single chain above.

    No codings (kernel 1.8): an ``:Episodic`` carries no ``topic_labels`` / ``topic_p`` (absent keys, not empty
    lists), no Question / Plan / Purpose node or ANSWERS / ANSWERS_VIA / DERIVED_FROM edge is emitted, and a Saga's
    summary has no labels clause. Observations carry no card data: no ``*_answers`` / ``*_p_role`` / ``answers`` /
    ``p_role`` and no Event ``card``.

    Role names: a role edge is named by the PropBank function tag of its argument (``g1_roles.function_tag`` over
    ``roleset_index``, the single tag source; an explicit ``role_functions`` entry wins for its role) and keeps its
    ``role`` (argN); Events carry ``verbnet_classes`` when ``verbnet`` is given. Entity minting uses
    ``g1_referring.is_referring`` with the role's tag, and a first/second-person possessive is owner-qualified
    (``owner_key``: "my family" said by Ana is "Ana's family"). Every predicate, on every eventuality
    branch, carries ``g1_time.event_time`` fields (``told_at``, ``happened_at``, ``happened_at_grain``,
    ``time_source``, ``valid_at``). Event names and fact texts come from ``g1_render.render_proposition``
    (``render_mode`` "b" by default, "c" = the envelope sentence everywhere), parsed once with the projection's doc.
    State fact-edge names are unchanged in this unit.

    ``excluded`` predicate keys (fix 1b: nominal modifiers inside a noun phrase) are not predicates: they mint no
    node, edge, summary line or mix entry and need no eventuality label. ``predicate_pos`` maps a roleset to its
    PropBank predicate POS for relation names (fix 1c; absent = verb).

    Attribution lives in the envelope: each ``:Episodic`` carries the envelope as flat properties set at
    creation (``sender``/``sender_name``, ``addressee``/``addressee_name`` = the other participant of a 2-party
    session, ``mood``, ``act``, ``clause_acts``, ``told_at``, ``reply_to``). The payload is plain PropBank structure:
    no speaker substitution, no attribution in ``fact``. A speech-act predicate whose agent is absent or the sender
    is the envelope's act (no content; it names the act unless its eventuality label is a roleset-prior fallback or its
    sentence is a question, where the act is ``ask``); a reported one (another agent) is content, routed as an occurrence. An
    evaluation with no holder (its holder role not referring and no person among its core args) is the envelope's act
    ``evaluate`` (no content); with a holder it is routed as a state. State/evaluation edges run from the holder
    (else the lowest-numbered referring core arg; never a CAU arg unless it is the holder, ``role_functions`` =
    roleset -> {role: PropBank f}) to the next referring core arg. Holder and direction from frame data:
    the holder is ``g1_roles.holder_role`` over ``role_themes(roleset)`` (role -> VerbNet thematic roles,
    ``propbank_framesets.verbnet_role_themes``): the lowest-numbered core role whose theme is ``experiencer``
    (amaze.01 arg1, look.02 arg2), else arg0 unless it is tagged CAU. ``role_themes=None`` means no theme for any
    roleset (the same single rule then reduces to arg0-unless-CAU); a caller that passes ``roleset_index`` must pass
    ``role_themes`` too (ValueError otherwise), so a build that knows its frames never falls back silently. A state
    with no referring participant is dropped and stays in the message text (``state_dropped``). Direction of
    fit is read per predicate from the sentence that holds its anchor: interrogative -> ``asserted: false``;
    imperative -> ``modality: requested``; first-person future -> ``modality: intended``; else asserted/actual. Every routed row carries ``eventuality_source`` (``prior`` when
    the eventuality entry has ``fallback``, else ``context``) and the four ``eventuality_p_*``.
    """
    if referent_policy not in ("phrase", "structure"):
        raise ValueError(f"referent_policy {referent_policy!r}: must be 'phrase' or 'structure'")
    if referent_policy == "structure":
        from memory_kernel.grammar.dependencies import wordnet_ready

        if not wordnet_ready():
            raise ValueError("referent_policy structure needs the WordNet corpus (mk setup)")
    if render_mode not in MODES:
        raise ValueError(f"render_mode must be one of {MODES}, got {render_mode!r}")
    excluded = frozenset(excluded)
    predicate_pos = predicate_pos or {}
    role_functions = role_functions or {}
    frames: Mapping[str, PropBankRolesetDefinition] = roleset_index if roleset_index is not None else {}
    if roleset_index is not None and role_themes is None:
        raise ValueError("role_themes is required with roleset_index (R7c: holder and direction from frame data)")
    themes_of = role_themes if role_themes is not None else (lambda _roleset: {})

    def tag_of(roleset: str, role: str) -> str:
        """The role's PropBank function tag: ONE source, ``g1_roles.function_tag`` over ``roleset_index``. An explicit
        ``role_functions`` entry (tests, callers with a bare tag map) takes precedence for its role."""
        given = role_functions.get(roleset, {}).get(role.lower())
        return given.upper() if given else function_tag(roleset, role, index=frames)

    by_order = order_key(message_rank(records))
    unassigned = sorted({r["message_id"] for r in records} - set(episodes), key=by_order)
    if unassigned:
        raise ValueError(f"{len(unassigned)} messages have no episode: {unassigned[:5]}")
    unlabelled = [f"{r['unit']}#{i}" for r in records for i in range(len(r["events"]))]
    unlabelled = [k for k in unlabelled if k not in eventuality and k not in excluded]
    if unlabelled:
        raise ValueError(f"{len(unlabelled)} predicates have no eventuality label: {unlabelled[:5]}")

    identity_cards = load_identity_cards()
    fwd = None
    if chain is not None:
        from memory_kernel.graph import forward

        people = _People(
            records,
            roster=forward.roster(chain),
            explicit_of=forward.explicit_flags(records, chain),
        )
        fwd = forward.resolve(
            records,
            chain=chain,
            run=run,
            group_id=group_id,
            people=people,
            tag_of=tag_of,
            excluded=excluded,
            identity_cards=identity_cards,
            by_order=by_order,
            referee=referee,
            policy=referent_policy,
        )
        classified, cands, resolved, reused, stats = fwd.classified, fwd.cands, fwd.resolved, fwd.reused, fwd.stats
        rep_key, joins, joined, owners, claim_events = fwd.rep_key, fwd.joins, fwd.joined, fwd.owners, fwd.claim_events
        names, before = None, dict(resolved)
    else:
        people = _People(records)
        classified, cands, resolved, reused, stats, rep_key, joins, joined, owners, before, claim_events, names = (
            _resolve_full(records, people, group_id, existing, excluded, tag_of, identity_cards, referent_policy)
        )
    structure = referent_policy == "structure"
    # Canonical node per uuid: union of kinds/aliases over the keys that resolved to it. Under structure the keys'
    # kind evidence is summed instead and one kind decided from it (``g1_kinds.decide``).
    merged: defaultdict[str, _Candidate] = defaultdict(_Candidate)
    for key, uuid in resolved.items():
        m, c = merged[uuid], cands[key]
        m.surfaces.update(c.surfaces)
        if structure:
            m.evidence.add(c.evidence)
        else:
            m.kinds |= c.kinds
        m.aliases |= c.aliases
        m.person = m.person or c.person
    for uuid in joined:  # a joined name is a person (speaker-declared or equated with one)
        if structure:
            merged[uuid].evidence.person = True
        else:
            merged[uuid].kinds.add("Person")
    if structure:
        # a forward run: a stored entity's evidence is the chain's (its every earlier mention) plus this session's
        stored_kinds = chain.kinds if chain is not None and chain.kinds is not None else {}
        for uuid in sorted(set(resolved.values()) & set(stored_kinds)):
            merged[uuid].evidence.add(KindEvidence.from_row(stored_kinds[uuid]))
        for m in merged.values():
            kind = decide(m.evidence)
            m.kinds = {kind} if kind is not None else set()
    else:
        for m in merged.values():  # a key that resolves to a Person is never also a Place
            if "Person" in m.kinds:
                m.kinds.discard(PLACE)
    display: dict[str, str] = {}
    b = _Builder(group_id, created_at)

    def entity(uuid: str) -> str:
        if uuid in b.nodes:
            return uuid
        m = merged[uuid]
        if uuid in reused:
            row = reused[uuid]
            name = row["name"]
            if structure:  # the kind decided over all its evidence replaces the stored kind label (``kinds_replace``)
                kinds = m.kinds
            else:  # labels frozen
                kinds = (set(row.get("labels") or []) | (m.kinds if fwd is None else set())) - {"Entity"}
            extra: dict[str, object] = {}
        else:
            name = m.person or min(m.surfaces.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            kinds = m.kinds
            extra = {"name_key": rep_key[uuid]}
        if uuid in joined:
            if uuid not in reused:
                name = joins.display[joined[uuid]]
            extra["identity_aliases"] = sorted(joins.aliases[joined[uuid]])
        elif uuid in owners and uuid not in reused:
            name = f"{owners[uuid]}'s"
        if m.aliases:
            extra["aliases"] = sorted(a for a in m.aliases if a != name)
        display[uuid] = name
        b.node(["Entity", *sorted(kinds)], uuid, name, **extra)
        if structure:  # a row key, not a prop: the writer removes the kind labels the row does not carry
            b.nodes[uuid]["kinds_replace"] = True
        return uuid

    def fact_edge(
        source: str,
        target: str,
        name: str,
        fact: str,
        ts: str,
        ep: str,
        key: str,
        time: dict[str, str | None],
        **props: object,
    ) -> None:
        """A fact edge; a repeat (same source, target, name, normalized fact, asserted, modality) only grows episodes.

        The force is part of the identity: an asked fact and an asserted one are never one edge."""
        seed = (normalize_exact(fact), props.get("asserted"), props.get("modality"))
        row = b.edges.get(_sha("edge", group_id, "RELATES_TO", source, target, name, *seed))
        if row is None:
            b.edge(
                "RELATES_TO",
                source,
                target,
                name,
                seed,
                fact=fact,
                episodes=[ep],
                reference_time=ts,
                event_keys=[key],
                **{k: time[k] for k in TIME_FIELDS},
                **props,
            )
            return
        p = row["props"]
        p["episodes"] = sorted({*p["episodes"], ep})
        p["event_keys"] = sorted({*p["event_keys"], key})
        p["reference_time"] = min(p["reference_time"], ts)
        p["valid_at"] = min(p["valid_at"], str(time["valid_at"]))
        p["told_at"] = min(p["told_at"], str(time["told_at"]))
        # One edge folding several messages keeps the earliest happened_at, with the time_source of the kept one.
        kept, new = p.get("happened_at"), time["happened_at"]
        if new is not None and (kept is None or new < kept):
            p["happened_at"], p["happened_at_grain"], p["time_source"] = (
                new,
                time["happened_at_grain"],
                time["time_source"],
            )

    stats["aliases"] = sum(len(m.aliases) for m in merged.values())
    if joins.canonical or joins.conflicts:
        stats["identity_joins"] = len(joins.canonical)
        stats["identity_conflicts"] = list(joins.conflicts)
        stats["identity_self_facts_dropped"] = 0
        stats["identity_claim_events_dropped"] = 0
    routing: Counter[str] = Counter()
    #: entity uuid -> [(attributed line, source Episodic uuid)] in message order (lines keep provenance).
    summary_lines: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    #: entity uuid -> its identity lines among ``summary_lines`` (``summary_identity_line_ids``), decided at write.
    identity_lines: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    saga_msgs: defaultdict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    by_message: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_message[rec["message_id"]].append(rec)
    #: session -> (speaker, Episodic uuid) of its latest message, for ``reply_to`` (adjacency pairs).
    last_turn: dict[str, tuple[str, str]] = {}

    for message_id in sorted(by_message, key=by_order):
        recs = sorted(by_message[message_id], key=lambda r: message_order(r["unit"]))
        first = recs[0]
        speaker, ts = first["speaker"], first["timestamp"]
        ep = _sha("ep", group_id, message_id)
        who = entity(resolved[normalize_exact(speaker)])
        addressee_name = people.addressee(first)
        mentioned = {who}
        mix: list[str] = []
        #: Per sentence (unit, sentence start): mood, named speech acts, commissive?, holderless evaluation?
        clauses: dict[tuple[str, int], dict[str, Any]] = {}
        for rec in recs:
            for sent in _parse(rec["text"]).sents:
                clauses[(rec["unit"], sent.start)] = {
                    "mood": sentence_mood(sent),
                    "speech_acts": [],
                    "commit": False,
                    "holderless": False,
                }

        for rec in recs:
            for index, ev in enumerate(rec["events"]):
                key = f"{rec['unit']}#{index}"
                if key in excluded:
                    routing["excluded_nominal_modifier"] += 1
                    continue
                label = eventuality[key]["label"]
                c = construction_for(label)
                probs = eventuality[key].get("p") or {}
                route = c.route
                mix.append(label)
                routing[label] += 1
                if key in claim_events:  # the identity statement itself is the join, not content
                    stats["identity_claim_events_dropped"] += 1
                    routing["identity_claim_event_dropped"] += 1
                    # ... but the joined person it names exists and this message mentions it
                    for role in ev["roles"]:
                        f = classified[(rec["unit"], index, role)]
                        for x in _referents(f) if f is not None else ():
                            if resolved[str(x.key)] in joined:
                                mentioned.add(entity(resolved[str(x.key)]))
                    if fwd is not None:
                        mentioned |= {entity(u) for u in fwd.claim_canonicals[key]}
                    continue
                lemma = ev["roleset"].split(".")[0]
                fillers = {role: f for role in ev["roles"] if (f := classified[(rec["unit"], index, role)]) is not None}
                refer = {role: entity(resolved[f.key]) for role, f in fillers.items() if f.key is not None}
                #: role -> its distinct entity uuids, first mention first (structure: a coordinated role fans out)
                refer_all = {role: [uuid] for role, uuid in refer.items()}
                if structure:
                    refer_all = {
                        role: uu
                        for role, f in fillers.items()
                        if (uu := list(dict.fromkeys(entity(resolved[str(x.key)]) for x in _referents(f))))
                    }
                    refer = {role: uu[0] for role, uu in refer_all.items()}
                    mentioned |= {u for uu in refer_all.values() for u in uu}
                    mentioned |= {entity(resolved[str(x.key)]) for f in fillers.values() for x in f.inner}
                mentioned |= set(refer.values())
                shown = {
                    r: display[refer[r]] if (f.person or f.resolved) and r in refer else f.text
                    for r, f in fillers.items()
                }
                doc = _parse(rec["text"])
                tok = _anchor(doc, eventuality[key], ev)
                clause = clauses.get((rec["unit"], tok.sent.start)) if tok is not None else None
                if clause is None:
                    clause = clauses.setdefault(
                        (rec["unit"], -1),
                        {"mood": "declarative", "speech_acts": [], "commit": False, "holderless": False},
                    )
                tmp = (ev["roles"].get(TIME_ROLE) or {}).get("text")
                when = event_time(tmp, ts)  # every eventuality branch carries the time fields (occurrence or state)
                time_props = {k: when[k] for k in TIME_FIELDS}
                rendered = render_proposition(
                    ev,
                    [t.text for t in doc],
                    shown,
                    when["phrase"],
                    sentence=tok.sent.text if tok is not None else rec["text"],
                    speaker=speaker,
                    addressee=addressee_name,
                    told_at=ts,
                    mood=clause["mood"],
                    doc=doc,
                    mode=render_mode,
                )
                key_text = legacy_fragment(ev, shown)
                commit = tok is not None and commissive(tok)
                clause["commit"] = clause["commit"] or commit
                force = clause_force(clause["mood"], commit)
                evp: dict[str, object] = {
                    "roleset": ev["roleset"],
                    "eventuality": label,
                    "eventuality_source": PRIOR if eventuality[key].get("fallback") else CONTEXT,
                    **force,
                }
                evp |= {f"eventuality_p_{e}": probs.get(e) for e in EVENTUALITIES}
                functions = {r: tag_of(ev["roleset"], r) for r in ev["roles"] if is_core_role(r)}
                holder = holder_role(themes_of(ev["roleset"]), lambda r, rs=ev["roleset"]: tag_of(rs, r))

                if route == "speech":
                    if AGENT_ROLE in fillers and refer.get(AGENT_ROLE) != who:
                        # Reported speech is a communication Activity. It is written through the
                        # event writer under the speech construction's own plan; its eventuality stays speech_act.
                        route = "event"
                        routing["speech_act_reported"] += 1
                    else:
                        routing["speech_act_envelope"] += 1
                        if eventuality[key].get("fallback"):
                            routing["speech_act_envelope_fallback_unnamed"] += 1
                        elif clause["mood"] == "interrogative":
                            routing["speech_act_in_question_unnamed"] += 1
                        else:
                            clause["speech_acts"].append(lemma)
                        continue
                if route == "holder_or_envelope":
                    has_holder = any(
                        role == holder or "Person" in b.nodes[uuid]["labels_neo4j"]
                        for role, uuid in refer.items()
                        if is_core_role(role)
                    )
                    if not has_holder:
                        routing["evaluation_holderless"] += 1
                        clause["holderless"] = True
                        continue
                    routing["evaluation_with_holder"] += 1

                if route == "event":
                    props: dict[str, object] = {}
                    for role, f in sorted(fillers.items()):
                        if role in refer:
                            continue
                        stem = role_key(role)
                        props[f"{stem}_text"] = f.text
                    name = rendered
                    if structure:  # named by its frame and roles; the sentence moves to ``source_text``
                        named = [r for r in fillers if is_core_role(r)] or list(fillers)
                        name = frame_name(
                            ev["roleset"],
                            [
                                (
                                    r,
                                    " and ".join(display[u] for u in refer_all[r])
                                    if r in refer_all
                                    else fillers[r].text,
                                )
                                for r in sorted(named, key=_role_order)
                            ],
                        )
                        props["source_text"] = rendered
                    evt = b.node(
                        ["Entity", "Event"],
                        _sha("evt", group_id, rec["unit"], ev["roleset"], *ev["anchor"]),
                        name,
                        anchor=[int(i) for i in ev["anchor"]],
                        origin=ev.get("origin"),
                        message_id=message_id,
                        session_id=rec["session_id"],
                        episode_uuid=ep,
                        event_key=key,
                        key_text=key_text,
                        plan=c.plan,
                        verbnet_classes=list(verbnet(ev["roleset"])) if verbnet is not None else None,
                        **time_props,
                        **evp,
                        **props,
                    )
                    b.nodes[evt]["props"]["summary"] = f"{ev['roleset']}: " + "; ".join(
                        f"{r} {fillers[r].text}" for r in sorted(fillers)
                    )
                    mentioned.add(evt)
                    routing["events_with_participant"] += bool(refer)
                    for role, uuids in sorted(refer_all.items()):
                        for uuid in uuids:  # one edge per (role, referent); the target is in the edge's uuid hash
                            b.edge(
                                "RELATES_TO",
                                evt,
                                uuid,
                                tag_of(ev["roleset"], role),
                                (role,),
                                fact=rendered,
                                episodes=[ep],
                                reference_time=ts,
                                **time_props,
                                role=role,
                                event_uuid=evt,
                                key_text=key_text,
                                plan=c.plan,
                                **evp,
                            )
                else:  # state, or an evaluation with a holder (ordinary content); one fact per referent combination
                    routing["state_argm_participants_unlinked"] += sum(1 for r in refer if not is_core_role(r))
                    if joined:  # an edge between two names of one joined person would be a self-loop: unwritten
                        stats["identity_self_facts_dropped"] += _unwritten_by_join(fillers, refer_all, before, resolved)
                    core_roles = sorted(r for r in refer_all if is_core_role(r))
                    combos = [
                        _direction(dict(zip(core_roles, combo, strict=True)), functions, holder)
                        for combo in product(*(refer_all[r] for r in core_roles))
                    ]
                    edges = [parts for parts in combos if len(parts) >= 2]
                    if edges:
                        for parts in edges:
                            (src_role, src), others = parts[0], parts[1:]
                            for tgt_role, tgt in others:
                                fact_edge(
                                    src,
                                    tgt,
                                    # Invariant O: the name comes from the roleset and its POS, never a card
                                    relation_name(ev["roleset"], predicate_pos.get(ev["roleset"], "verb")),
                                    rendered,
                                    ts,
                                    ep,
                                    key,
                                    time_props,
                                    source_role=src_role,
                                    target_role=tgt_role,
                                    key_text=key_text,
                                    plan=c.plan,
                                    **evp,
                                )
                        routing[f"{label}_to_edge"] += 1
                        routing["state_edges_with_3plus_participants"] += any(len(p) > 2 for p in edges)
                    elif any(combos):
                        line = render_attributed(
                            {"fact": rendered, "asserted": evp["asserted"], "modality": evp["modality"]},
                            {"sender_name": display[who], "told_at": ts},
                        )
                        # An identity line: an asserted, actual copular clause with a nominal predicate whose subject
                        # (or one of its conjuncts: "Kira and Milo are pilots") lies in the bearer's own words on the
                        # unit's parse (the bearer IS the entity: decided by referent, not by name).
                        plain = tok is not None and evp["asserted"] and evp["modality"] == "actual"
                        subject = nominal_copula_subject(tok) if plain else None
                        subjects = {subject.i, *(c.i for c in subject.conjuncts)} if subject is not None else set()
                        bearers: dict[str, str] = {}  # bearer uuid -> its role, first combo first
                        for parts in combos:
                            if parts:
                                bearers.setdefault(parts[0][1], parts[0][0])
                        found = False
                        for bearer, role in bearers.items():
                            summary_lines[bearer].append((line, ep))
                            if subjects and subjects & _bearer_tokens(
                                doc,
                                fillers[role],
                                ev["roles"][role],
                                lambda x, b=bearer: x.key is not None and entity(resolved[str(x.key)]) == b,
                            ):
                                identity_lines[bearer].add((line, ep))
                                found = True
                        if subjects and not found:  # a copula subject outside every bearer's words: no identity line
                            routing["identity_subject_outside_bearers"] += 1
                        routing[f"{label}_to_summary"] += 1
                        routing["summary_lines_nonactual"] += not evp["asserted"] or evp["modality"] != "actual"
                    else:
                        routing[f"{label}_dropped"] += 1

        ordered = [clauses[k] for k in sorted(clauses)]
        moods = [c["mood"] for c in ordered]
        clause_acts = [act_ladder(c["speech_acts"], [c["mood"]], c["commit"], c["holderless"]) for c in ordered]
        act = act_ladder(
            [a for c in ordered for a in c["speech_acts"]],
            moods,
            any(c["commit"] for c in ordered),
            any(c["holderless"] for c in ordered),
        )
        addressee_key = normalize_exact(addressee_name) if addressee_name else None
        if fwd is not None:  # a forward run resolved it against the chain (an earlier session's speaker, too)
            found = fwd.addressee[message_id]
            addressee = entity(found) if found is not None else None
        else:
            addressee = entity(resolved[addressee_key]) if addressee_key in resolved else None
        previous_turn = last_turn.get(first["session_id"])
        reply_to = previous_turn[1] if previous_turn and previous_turn[0] != speaker else None
        last_turn[first["session_id"]] = (speaker, ep)
        b.node(
            ["Episodic"],
            ep,
            message_id,
            source="message",
            source_description=first["session_id"],
            content=f"{speaker}: {' '.join(r['text'].strip() for r in recs)}",
            valid_at=ts,
            entity_edges=[],
            message_id=message_id,
            eventuality_mix=mix or None,
            **episodic_envelope(
                who=who,
                who_name=display[who],
                addressee=addressee,
                addressee_name=display[addressee] if addressee is not None else None,
                moods=moods,
                act=act,
                clause_acts=clause_acts,
                told_at=ts,
                reply_to=reply_to,
            ),
        )
        for target in sorted(mentioned):
            b.edge("MENTIONS", ep, target, "MENTIONS", ())
        saga_msgs[episodes[message_id]].append((message_id, ep, speaker, ts))

    for uuid, lines in summary_lines.items():
        identity = identity_lines.get(uuid, set())
        if uuid in reused and fwd is not None:  # a stored node: its new lines are appended by the writer
            kept = list(dict.fromkeys(lines))
            ids = [summary_line_id(uuid, source, line) for line, source in kept]
            props = b.nodes[uuid]["props"]
            props["summary_append"] = [line for line, _ in kept]
            props["summary_append_sources"] = [source for _, source in kept]
            props["summary_append_line_ids"] = ids
            props["summary_append_identity_line_ids"] = [i for i, kl in zip(ids, kept, strict=True) if kl in identity]
            routing["summary_lines_appended_on_reused"] += len(kept)
            continue
        if uuid in reused:
            routing["summary_lines_skipped_on_reused"] += len(lines)
            continue
        kept = list(dict.fromkeys(lines))
        ids = [summary_line_id(uuid, source, line) for line, source in kept]
        props = b.nodes[uuid]["props"]
        props["summary"] = "\n".join(line for line, _ in kept)
        props["summary_sources"] = [source for _, source in kept]
        props["summary_line_ids"] = ids
        props["summary_line_texts"] = [line for line, _ in kept]
        props["summary_identity_line_ids"] = [i for i, kl in zip(ids, kept, strict=True) if kl in identity]

    found_sagas: dict[str, dict] = {}
    for row in existing_sagas or []:
        found_sagas.setdefault(row["name"], row)
    own_eps = {ep for msgs in saga_msgs.values() for _, ep, _, _ in msgs}
    #: Stored endpoints outside this batch (a reused saga's earlier episode), for ``validate_rows(previous=...)``.
    external: dict[str, dict] = {}
    stats["sagas_reused"] = 0
    ordered_sagas = sorted(saga_msgs.items(), key=lambda kv: by_order(kv[1][0][0]))
    # The chain tail: max (ended_at, started_at); max() keeps the first of equal keys, so reader row order breaks ties.
    latest_stored = max(
        (r for r in existing_sagas or [] if r["name"] not in saga_msgs),
        key=lambda r: (r.get("ended_at") or "", r.get("started_at") or ""),
        default=None,
    )
    #: The last saga of each conversation (chat) so far: ``NEXT_SAGA`` runs within a chat, never across chats.
    prev_saga_of: dict[str | None, str] = {}
    for position, (episode_id, msgs) in enumerate(ordered_sagas):
        found = found_sagas.get(episode_id)
        # a reused saga's earlier messages (a forward append continuing its chat's open episode): the saga spans them
        whole = [*(tuple(m) for m in (found or {}).get("prior") or () if m[1] not in own_eps), *msgs]
        summary = _saga_summary(whole)
        if found is not None:
            stats["sagas_reused"] += 1
            previous = next((u for u in found.get("episodes") or [] if u not in own_eps), None)
            if previous is not None:
                b.edge("NEXT_EPISODE", previous, msgs[0][1], "NEXT_EPISODE", ())
                external[previous] = {"labels_neo4j": ["Episodic"], "props": {"uuid": previous}}
        provenance: dict[str, object] = {}
        if ingest_pass is not None:
            stored_passes = list((found or {}).get("ingest_pass") or [])
            provenance = {
                "ingest_pass": stored_passes if ingest_pass in stored_passes else [*stored_passes, ingest_pass],
                "language_version": language_version(LANGUAGE_CONSTRUCTIONS),  # follows the psych-holder switch
            }
        saga = b.node(
            ["Saga"],
            found["uuid"] if found is not None else _sha("saga", group_id, episode_id),
            episode_id,
            summary=summary,
            first_episode_uuid=whole[0][1],
            last_episode_uuid=msgs[-1][1],
            started_at=whole[0][3],
            ended_at=msgs[-1][3],
            message_count=len(whole),
            **provenance,
            **_saga_vector(saga_vectors, episode_id, len(whole)),
        )
        if ingest_pass is not None:
            # 1.8: no card inventory. The key stays, None (``b.node`` drops None props, so it is set here); the writer
            # never SETs a None, so a stored Saga keeps its hex value and the forward gate passes it.
            b.nodes[saga]["props"]["inventory_sha"] = None
            chat = conversation_of[msgs[0][0]] if conversation_of is not None else None
            prev_saga = prev_saga_of.get(chat)
            if prev_saga is not None:
                b.edge("NEXT_SAGA", prev_saga, saga, "NEXT_SAGA", ())
            elif position == 0 and found is None and latest_stored is not None:
                b.edge("NEXT_SAGA", latest_stored["uuid"], saga, "NEXT_SAGA", ())
                external[latest_stored["uuid"]] = {"labels_neo4j": ["Saga"], "props": {"uuid": latest_stored["uuid"]}}
            prev_saga_of[chat] = saga
        for _, ep, _, _ in msgs:
            b.edge("HAS_EPISODE", saga, ep, "HAS_EPISODE", ())
        for (_, prev, _, _), (_, nxt, _, _) in zip(msgs, msgs[1:], strict=False):
            b.edge("NEXT_EPISODE", prev, nxt, "NEXT_EPISODE", ())

    if fwd is not None:  # identity links: a stored endpoint is an external endpoint of the batch
        for link in fwd.links:
            row = b.edge(
                "SAME_AS",
                link["from_uuid"],
                link["to_uuid"],
                "SAME_AS",
                (link["message_id"], link["name"]),
                claim_message=link["message_id"],
                holder=link["holder"],
                card=link["card"],
            )
            row["props"]["name"] = link["name"]
            for end in (link["from_uuid"], link["to_uuid"]):
                if end not in b.nodes:
                    external[end] = {"labels_neo4j": ["Entity"], "props": {"uuid": end}}

    membership: defaultdict[str, list[str]] = defaultdict(list)
    for uuid, row in b.edges.items():
        if row["type"] == "RELATES_TO":
            for episode in row["props"]["episodes"]:
                membership[episode].append(uuid)
    for ep, uuids in membership.items():
        b.nodes[ep]["props"]["entity_edges"] = sorted(uuids)
    stats["reused_existing"] = len(reused)
    rows_with_force = [n["props"] for n in b.nodes.values()] + [e["props"] for e in b.edges.values()]
    routing["unasserted_rows"] = sum(1 for p in rows_with_force if p.get("asserted") is False)
    for modality in ("intended", "requested"):
        routing[f"modality_{modality}_rows"] = sum(1 for p in rows_with_force if p.get("modality") == modality)
    out = {
        "nodes": list(b.nodes.values()),
        "edges": list(b.edges.values()),
        "external_endpoints": external,
        "dedup": stats,
        "routing": dict(routing),
    }
    out["forward"] = _forward_of(
        b, records, merged, rep_key, reused, run, fwd, resolved, joins, names, owners, chain, structure
    )
    return out


def _forward_of(
    b, records, merged, rep_key, reused, run, fwd, resolved, joins, names, owners, chain, structure=False
) -> dict:
    """``project``'s ``forward`` (``graph.forward.emit``): a full build's first mint, or a forward run's additions.

    Under ``referent_policy`` structure a minted entity is a ``person`` only when its evidence says so (a speaker,
    roster, deixis or join person; never a WordNet ``noun.person``), and ``kinds`` holds every written entity's merged
    kind evidence (``KindEvidence.to_row``; a forward run's includes the chain's evidence of a stored entity)."""
    from memory_kernel.graph import forward

    persons = {u for u, m in merged.items() if m.evidence.person} if structure else None
    written = {u for u, n in b.nodes.items() if "Entity" in n["labels_neo4j"] and "Event" not in n["labels_neo4j"]}
    if fwd is None:
        routes = forward.full_build_routes(
            resolved=resolved, rep_key=rep_key, written=written - set(reused), joins=joins, names=names, owners=owners
        )
        out = forward.emit(
            nodes=b.nodes, edges=list(b.edges.values()), records=records, merged=merged, rep_key=rep_key,
            stored=set(reused), run=run, routes=routes, persons=persons,
        )  # fmt: skip
        if structure:
            out["kinds"] = {u: merged[u].evidence.to_row() for u in sorted(written)}
        return out
    appended = {u for u, n in b.nodes.items() if "summary_append" in n["props"]}
    out = forward.emit(
        nodes=b.nodes, edges=list(b.edges.values()), records=records, merged=merged, rep_key=rep_key,
        stored=set(chain.entities), run=run, routes=fwd.routes, links=fwd.links, conflicts=fwd.conflicts,
        summary_touched=appended, strict=True, persons=persons,
    )  # fmt: skip
    if structure:
        out["kinds"] = {u: merged[u].evidence.to_row() for u in sorted(written)}
    return out
