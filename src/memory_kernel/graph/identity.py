"""Identity claims and joins: the naming constructions (identity_cards) applied to captured frames."""

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from memory_kernel.grammar.dependencies import WORDNET_LOOKUP_LOCK, wordnet_module
from memory_kernel.graph.g1_nlp import normalize_exact
from memory_kernel.graph.identity_cards import IdentityCards

_SPLIT = re.compile(r"\s*(?:,|\band\b|&)\s*", re.IGNORECASE)
_POSSESSIVE = re.compile(r"\s?['\u2019]s$", re.IGNORECASE)
_SENTENCE_START = re.compile(r"(?:.*[.!?])?\W*", re.DOTALL)


def is_common_noun(name: str) -> bool:
    """WordNet has a noun lemma spelled exactly as the lowercase name: "rose", "egg", "hope", "grace" are; "mary"
    is not (WordNet knows it only as the proper noun "Mary"). Without WordNet
    every name counts as a common noun, so a lowercase mention never joins."""
    wn = wordnet_module()
    if wn is None:
        return True
    key = normalize_exact(name).replace(" ", "_")
    with WORDNET_LOOKUP_LOCK:  # NLTK's lookups share one file handle: one thread at a time
        return any(lemma.name() == key for s in wn.synsets(key, "n") for lemma in s.lemmas())


def capitalised(text: str) -> bool:
    return bool(text.split()) and all(tok[:1].isupper() for tok in text.split())


def alias_occurs(
    alias: str, text: str, *, is_common: Callable[[str], bool], within: tuple[int, int] | None = None
) -> bool:
    """An occurrence of ``alias`` in ``text`` (case-insensitive, whole word) counts as naming the person when the alias
    is not a common noun (``is_common`` of its lowercase form; any case, any position), or when the occurrence is
    capitalised and not sentence-initial (only non-word characters between the text start or the last . ! ? and it).
    So for the alias "Egg": "I asked Egg" counts; "the egg" and "Egg is life." do not. ``within`` = (start, end): only
    occurrences starting there count (the projection's filler span in its unit). The one predicate the projection
    (``g1_projection._JoinNames``) and retrieval (``kg_reader.aliased_uuids``) share."""
    words = alias.split()
    if not words:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in words) + r"(?!\w)"
    common: bool | None = None
    for m in re.finditer(pattern, text, re.IGNORECASE):
        if within is not None and not within[0] <= m.start() < within[1]:
            continue
        if common is None:
            common = is_common(normalize_exact(alias))
        if not common:
            return True
        if capitalised(m.group()) and not _SENTENCE_START.fullmatch(text[: m.start()]):
            return True
    return False


@dataclass(frozen=True)
class Claim:
    holder: str
    name: str
    kind: str
    first_person: bool
    message_id: str
    speaker: str
    card: str = ""  # the identity construction's id (``IdentityConstruction.id``)


def _clean(text: str) -> str:
    return text.strip().lstrip("@").strip()


def _strip_prefix(text: str, cards: IdentityCards) -> str:
    """Drop one leading prefix word ("to me", "as Eli") named by the card."""
    head, _, rest = text.strip().partition(" ")
    return rest.strip() if rest and head.lower() in cards.strip_prefixes else text.strip()


def _locate(name: str, unit: str):
    """The span of the unit's own parse where the name's tokens occur in order (case-insensitive, first occurrence)."""
    from memory_kernel.graph.g1_projection import _parse

    if not unit.strip():
        return None
    want = [t.text.lower() for t in _parse(name)]
    doc = _parse(unit)
    have = [t.text.lower() for t in doc]
    for i in range(len(have) - len(want) + 1):
        if want and have[i : i + len(want)] == want:
            return doc[i : i + len(want)]
    return None


def _name_site(text: str, cards: IdentityCards, speakers: Collection[str], unit: str):
    """The unit-text span of ``text`` when it is a name there, else None."""
    t = _clean(text)
    if not t or _POSSESSIVE.search(t):  # "I'm Bo's": a possessive is not a name
        return None
    speaker = t.lower() in {s.lower() for s in speakers}
    if not speaker:  # the cheap checks first: the unit's parse only for a filler shaped like a name
        tokens = t.split()
        if t.lower() in cards.not_names or any(tok.lower() in cards.not_names for tok in tokens):
            return None
        if not 1 <= len(tokens) <= cards.max_name_tokens or not capitalised(t):
            return None
    span = _locate(t, unit)
    if span is None:
        return None
    return span if speaker or span.root.pos_ == "PROPN" else None


def is_name(text: str, cards: IdentityCards, speakers: Collection[str], context: str = "") -> bool:
    """A speaker name, or 1..max_name_tokens capitalised tokens that are proper nouns, as read in ``context`` (the
    unit's own text): a filler that does not occur there is never a name, and there is no standalone-parse fallback.
    The head must be PROPN in the unit's parse. Calendar words (the card's ``not_names``) and possessives ("Bo's") are
    never names."""
    return _name_site(text, cards, speakers, context) is not None


def _negated(span, unit: str, cards: IdentityCards) -> bool:
    """A negation or irrealis cue ("not", "never"; "if", "wish", "pretend") in the same clause (after the last
    , ; . ! ? before the name) and before the name."""
    before = unit[: span.start_char].replace("\u2019", "'").lower()
    clause = re.split(r"[,;.!?]", before)[-1]
    return _has_cue(clause, cards.negation_cues) or _has_cue(clause, cards.irrealis_cues)


def _negated_marker(sites: Sequence, unit: str, cards: IdentityCards) -> bool:
    """A negation cue between the names and the same-person marker ("Dee and Ana are not the same person")."""
    low = unit.replace("\u2019", "'").lower()
    first, last = min(s.start_char for s in sites), max(s.end_char for s in sites)
    for marker in cards.same_person_markers:
        at = low.find(marker, last)
        if at >= 0 and _has_cue(low[last:at], cards.negation_cues):
            return True
        at = low.rfind(marker, 0, first)
        if at >= 0 and _has_cue(low[at + len(marker) : first], cards.negation_cues):
            return True
    return False


def _has_cue(clause: str, cues: Sequence[str]) -> bool:
    """``clause`` (lowercased) holds one of the card's ``cues``."""
    for cue in cues:
        if cue.startswith("n'"):  # a clitic: "isn't", "do n't"
            if cue in clause:
                return True
        elif re.search(rf"(?<![a-z']){re.escape(cue)}(?![a-z])", clause):
            return True
    return False


def _holder(spec: Mapping, rec: Mapping, cards: IdentityCards, speakers: Collection[str]) -> tuple[str, bool] | None:
    text = _strip_prefix(str(spec.get("text") or ""), cards)
    ref = spec.get("referent") or {}
    if text.lower() in cards.self_holders or (ref.get("status") == "deixis" and ref.get("text") == rec["speaker"]):
        return rec["speaker"], True
    return (_clean(text), False) if is_name(text, cards, speakers, str(rec.get("text") or "")) else None


def identity_claims(records: Sequence[Mapping], cards: IdentityCards, speakers: Collection[str]) -> list[Claim]:
    out: list[Claim] = []
    for rec in records:
        unit = str(rec.get("text") or "")
        if unit.rstrip().endswith("?"):  # a question ("Am I Dee?") claims nothing
            continue
        for ev in rec.get("events", []):
            roles = {k.lower(): v for k, v in (ev.get("roles") or {}).items()}
            for c in cards.constructions:
                if ev.get("roleset") not in c.rolesets:
                    continue
                if c.kind == "same_person":
                    out.extend(_same_person(c, roles, rec, cards, speakers))
                    continue
                h, n = roles.get(c.holder_role or ""), roles.get(c.name_role or "")
                if not h or not n:
                    continue
                name = _clean(_strip_prefix(str(n.get("text") or ""), cards))
                site = _name_site(name, cards, speakers, unit)
                if site is None or _negated(site, unit, cards):
                    continue
                held = _holder(h, rec, cards, speakers)
                if held and held[0].lower() != name.lower():
                    out.append(Claim(held[0], name, c.kind, held[1], rec["message_id"], rec["speaker"], c.id))
    return out


def _same_person(card, roles, rec, cards, speakers) -> list[Claim]:
    names_role = card.names_role or ""
    spec = roles.get(names_role)
    if not spec:
        return []
    others = (
        " ".join(str(v.get("text") or "") for k, v in roles.items() if k != names_role)
        + " "
        + str(rec.get("text") or "")
    )
    if not any(m in others.lower() for m in cards.same_person_markers):
        return []
    parts = [_clean(p) for p in _SPLIT.split(str(spec.get("text") or "")) if _clean(p)]
    unit = str(rec.get("text") or "")
    sites = [_name_site(p, cards, speakers, unit) for p in parts]
    if len(parts) < 2 or any(site is None or _negated(site, unit, cards) for site in sites):
        return []
    if _negated_marker(sites, unit, cards):
        return []
    return [
        Claim(parts[0], p, card.kind, False, rec["message_id"], rec["speaker"], card.id)
        for p in parts[1:]
        if p.lower() != parts[0].lower()
    ]


@dataclass(frozen=True)
class Joins:
    canonical: dict[str, str]
    aliases: dict[str, tuple[str, ...]]
    display: dict[str, str]
    conflicts: tuple[str, ...]


def resolve_joins(claims: Sequence[Claim], speakers: Collection[str]) -> Joins:
    spk = {normalize_exact(s): s for s in speakers}
    parent: dict[str, str] = {}
    surface: dict[str, str] = {}
    first_owner: dict[str, str] = {}
    first_held: dict[str, int] = {}  # group root -> index (in application order) of its earliest claim's holder
    conflicts: list[str] = []

    def find(k: str) -> str:
        while parent.get(k, k) != k:
            k = parent[k]
        return k

    def speakers_in(root: str) -> set[str]:
        return {k for k in list(parent) + [root] if find(k) == root and k in spk}

    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        if speakers_in(ra) and speakers_in(rb):
            return False
        if ra in spk or rb in spk:
            keep, drop = (ra, rb) if ra in spk else (rb, ra)
        else:  # no speaker: the group of the earliest claim's holder keeps its key
            keep, drop = (ra, rb) if first_held.get(ra, len(claims)) <= first_held.get(rb, len(claims)) else (rb, ra)
        first_held[keep] = min(first_held.get(keep, len(claims)), first_held.get(drop, len(claims)))
        parent[drop] = keep
        parent.setdefault(keep, keep)
        return True

    ordered = [x for x in claims if x.first_person] + [x for x in claims if not x.first_person]
    owners: dict[str, set[str]] = {}
    for x in claims:
        if x.first_person:
            owners.setdefault(normalize_exact(x.name), set()).add(normalize_exact(x.holder))
    # A name several speakers each claim as their own is blocked for every claim, reported once.
    contested = {n for n, who in owners.items() if len(who) > 1}
    for n in sorted(contested):
        first = next(x for x in claims if x.first_person and normalize_exact(x.name) == n)
        conflicts.append(f"{first.message_id}: {first.name!r} claimed by several speakers")
    for idx, x in enumerate(ordered):
        h, n = normalize_exact(x.holder), normalize_exact(x.name)
        surface.setdefault(h, x.holder)
        surface.setdefault(n, x.name)
        if h in contested or n in contested:
            continue
        if not x.first_person and (
            (n in first_owner and first_owner[n] != find(h)) or (h in first_owner and first_owner[h] != find(n))
        ):
            conflicts.append(f"{x.message_id}: {x.holder!r} = {x.name!r} contradicts a self-declaration")
            continue
        parent.setdefault(h, h)
        parent.setdefault(n, n)
        ranked = h not in first_held  # only applied claims rank their holder
        first_held.setdefault(h, idx)
        if not union(h, n):
            if ranked:
                del first_held[h]
            conflicts.append(f"{x.message_id}: {x.holder!r} = {x.name!r} would merge two speakers")
            continue
        if x.first_person:
            first_owner[n] = find(h)
    canonical = {k: find(k) for k in sorted(parent) if find(k) != k}
    aliases: dict[str, tuple[str, ...]] = {}
    for k, root in canonical.items():
        aliases[root] = tuple(sorted({*aliases.get(root, ()), surface[k]}))
    display = {root: spk.get(root, surface.get(root, root)) for root in aliases}
    return Joins(canonical, aliases, display, tuple(conflicts))
