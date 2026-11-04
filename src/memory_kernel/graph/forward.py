"""Forward resolution (kernel 1.2): an append resolves its session's entity keys against the chain.

A full build (``project(chain=None)``) resolves globally, as the first mint; it then describes what it minted
(``first_mint``). An append (``project(chain=...)``) walks its session message by message (``resolve``):

1. the person roster grows (the chat's stored roster, then each message's speaker and deixis names);
2. the message's identity claims apply (first-person claims first): routes, ``SAME_AS`` links, conflicts;
3. its fillers are classified and each key resolves, in this order: a session speaker's own key (``normalize_exact``,
   never routed, never moved); a route (a common-noun route key only for a guarded name mention, ``_JoinNames``);
   the possessive of a routed name (onto the canonical's possessive entity); a stored ``Minted.key``; a key this run
   already resolved; fuzzy (a stored match beats an earlier key of the session; highest Jaccard, then the earliest
   ``(run, order)``; then along ``SAME_AS`` links), which records a route; else a new mint, ``content_uuid`` of the key.

Every filler's key is then rewritten to the key its uuid was minted under, so the row pass of ``project`` keeps one
key -> uuid map. Nothing stored is re-keyed: a stored node keeps its key, name and labels (under ``referent_policy``
structure its kind label is decided again: ``g1_projection.project``); new evidence only adds mentions, facts, summary
lines (``summary_append``), routes and links.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from typing import TYPE_CHECKING, Any

from memory_kernel.errors import BonfireError
from memory_kernel.graph import g1_projection as gp
from memory_kernel.graph.addressing import is_explicit_at
from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.graph.g1_kinds import KindEvidence, filler_evidence, frame_evidence
from memory_kernel.graph.g1_nlp import normalize_exact
from memory_kernel.graph.identity import Claim, Joins, capitalised, identity_claims, is_common_noun
from memory_kernel.graph.referents import Point

if TYPE_CHECKING:
    from memory_kernel.pipeline.chain import Chain, Minted


class Refused(BonfireError):
    """A bonfire's chain cannot be appended to or relocated as it is (``check_chain_group``); nothing was changed."""


def check_chain_group(chain: Chain, group: str) -> None:
    """Refuse a chain minted under another group than ``group`` (the bonfire's). Every chain entity's uuid is
    ``content_uuid("ent", group, key)`` of the key it was minted under (``_Resolver.mint``; a full build's first mint
    derives it the same way), so an entity whose uuid is not was minted under another group. A uuid is a global
    identity: a copy of a bonfire that kept its source's uuids under a new group (a clone) collides with the source in
    Neo4j, and the writer refuses both from the first append that reuses an entity (``kg_writer.ForeignUuids``).
    ``graph_append.append`` calls it before anything else."""
    wrong = [m for m in chain.entities.values() if m.uuid != content_uuid("ent", group, m.key)]
    if wrong:
        first = wrong[0]
        raise Refused(
            f"chain minted under another group (a clone): {len(wrong)} of the chain's {len(chain.entities)} entities "
            f"do not carry the uuid group {group!r} gives their key (first: {first.uuid}, key {first.key!r}). A copy "
            "that keeps its source's uuids collides with the source in Neo4j: re-key the copy or build it fresh"
        )


@cache
def _common(key: str) -> bool:
    return is_common_noun(key)


def explicit_flags(records: Sequence[Mapping], chain: Chain) -> dict[str, bool]:
    """Message id -> "you" resolves through its addressees (``addressing.is_explicit_at``): the bonfire's
    first addressed message (``chain.explicit_since``, else the session's own first addressed message) is at or before
    it. Orders continue the chain (``next_order``) in source order."""
    rank = gp.message_rank(records)
    base = max((s.last_order + 1 for s in chain.chats.values()), default=0)
    since = chain.explicit_since
    if since is None:
        addressed = [rank[r["message_id"]] for r in records if r.get("addressees")]
        since = base + min(addressed) if addressed else None
    return {mid: is_explicit_at(base + k, since) for mid, k in rank.items()}


def roster(chain: Chain) -> list[str]:
    """The stored person roster a session starts from (bonfire-wide, like a full build's): every
    chat's roster (chats in name order), then every stored person entity (mint order). The session's own speakers
    and deixis names join it message by message (``_People.grow``); the two-party "you" rule stays per session."""
    names = [name for chat in sorted(chain.chats) for name in chain.chats[chat].roster]
    persons = sorted((m for m in chain.entities.values() if m.kind == "person"), key=lambda m: (m.run, m.order, m.uuid))
    return list(dict.fromkeys([*names, *(m.display for m in persons)]))


class _RouteNames(gp._JoinNames):
    """The guarded name matcher of a forward run (``_JoinNames.filler``): the chain's route keys and the names this
    run's claims route. A hit keeps the name's own key (resolution follows the route, even a common-noun one)."""

    def __init__(self) -> None:
        self.by_name: dict[str, tuple[str, str, bool]] = {}

    def add(self, key: str) -> None:
        if key and not key.endswith("'s"):
            self.by_name.setdefault(key, (key, f"{key}'s", _common(key)))


@dataclass
class Forward:
    """What ``project``'s row pass needs from a forward run, and what the run adds to the chain."""

    classified: dict
    cands: defaultdict
    resolved: dict[str, str]
    reused: dict[str, dict]
    rep_key: dict[str, str]
    stats: dict
    joins: Joins
    joined: dict[str, str]
    owners: dict[str, str]
    claim_events: set[str]
    claim_canonicals: dict[str, list[str]]  # claim event key -> canonical uuids it names (MENTIONS)
    addressee: dict[str, str | None]  # message id -> addressee uuid
    links: list[dict] = field(default_factory=list)  # SAME_AS rows of this run
    routes: dict[str, str] = field(default_factory=dict)  # routes this run adds
    conflicts: list[dict] = field(default_factory=list)


class _Resolver:
    def __init__(self, chain: Chain, *, group_id: str, run: int, records: Sequence[Mapping], people) -> None:
        self.chain, self.group, self.run, self.people = chain, group_id, run, people
        self.stored: Mapping[str, Minted] = chain.entities
        self.by_key = {m.key: u for u, m in self.stored.items()}
        self.routes: dict[str, str] = dict(chain.routes)
        self.new_routes: dict[str, str] = {}
        self.links: dict[str, str] = {s.from_uuid: s.to_uuid for s in chain.same_as}
        self.minted: dict[str, str] = {}  # new uuid -> key
        self.rank_of: dict[str, int] = {}  # new uuid -> rank of the message that minted it
        self.hint: dict[str, str] = {}  # new uuid -> display (speaker name, claim holder, possessive owner)
        self.known: dict[str, str] = {}  # key -> uuid, this run
        self.session: list[tuple[str, frozenset[str], str]] = []  # earlier keys of the session (fuzzy candidates)
        self.speakers = {normalize_exact(r["speaker"]) for r in records}
        self.owners: dict[str, str] = {}
        self.link_rows: list[dict] = []  # SAME_AS rows of this run
        self.stats: dict[str, Any] = {
            "merged_exact": 0,
            "merged_fuzzy": 0,
            "ambiguous": 0,
            "ambiguous_keys": [],
            "fuzzy_merges": [],
        }
        self._index: tuple[list, dict[str, list[int]]] | None = None
        self.names = _RouteNames()
        for key in sorted(self.routes):  # a claimed name and a fuzzy key alike (the chain does not tell them apart)
            if key not in self.speakers:
                self.names.add(key)

    # -- what a uuid is --------------------------------------------------------------------------------------------

    def key_of(self, uuid: str) -> str:
        return self.stored[uuid].key if uuid in self.stored else self.minted[uuid]

    def kind(self, uuid: str) -> str:
        return self.stored[uuid].kind if uuid in self.stored else "entity"

    def display_of(self, uuid: str) -> str:
        return self.stored[uuid].display if uuid in self.stored else self.hint.get(uuid, self.minted.get(uuid, ""))

    def minted_at(self, uuid: str) -> tuple[int, int]:
        m = self.stored.get(uuid)
        return (m.run, m.order) if m is not None else (self.run, self.rank_of[uuid])

    def follow(self, uuid: str) -> str:
        """Along ``SAME_AS`` links to the canonical node (a cycle stops where it closes)."""
        seen = {uuid}
        while (nxt := self.links.get(uuid)) is not None and nxt not in seen:
            seen.add(nxt)
            uuid = nxt
        return uuid

    # -- resolution ------------------------------------------------------------------------------------------------

    def add_route(self, key: str, uuid: str) -> None:
        """First route wins; a key never routes to the node minted under it."""
        if key and key not in self.routes and self.key_of(uuid) != key:
            self.routes[key] = self.new_routes[key] = uuid

    def mint(self, key: str, rank: int, hint: str | None = None) -> str:
        uuid = content_uuid("ent", self.group, key)
        if uuid in self.stored or uuid in self.minted:
            return uuid
        self.minted[uuid], self.rank_of[uuid], self.known[key] = key, rank, uuid
        if hint:
            self.hint[uuid] = hint
        self._remember(key, uuid)
        return uuid

    def _remember(self, key: str, uuid: str) -> None:
        form = gp.fuzzy_form(key)
        if gp.fuzzy_eligible(form):
            self.session.append((key, gp.trigram_set(form), uuid))

    def route(self, key: str, named: bool) -> str | None:
        target = self.routes.get(key)
        if target is None or (not named and _common(key)):
            return None
        return self.follow(target)

    def existing(self, key: str, *, named: bool = True, plain: bool = True) -> str | None:
        """The node ``key`` names now, minting nothing (``plain``: a stored or this-run key counts, not only a route)."""
        if key in self.speakers:
            uuid = content_uuid("ent", self.group, key)
            return uuid if uuid in self.stored or uuid in self.minted else None
        if (hit := self.route(key, named)) is not None:
            return hit
        if plain and key in self.by_key:
            return self.follow(self.by_key[key])
        return self.known.get(key) if plain else None

    def resolve(self, key: str, rank: int, *, named: bool = False, depth: int = 0) -> str:
        if key in self.speakers:
            return self.mint(key, rank)
        if (hit := self.route(key, named)) is not None:
            return hit
        if depth < 3 and (owner := self._possessive(key, rank, named, depth)) is not None:
            return owner
        if key in self.by_key:
            self.stats["merged_exact"] += 1
            return self.follow(self.by_key[key])
        if key in self.known:
            return self.known[key]
        if (hit := self._fuzzy(key)) is not None:
            self.stats["merged_fuzzy"] += 1
            self.stats["fuzzy_merges"].append([key, hit])
            self.add_route(key, hit)
            self.known[key] = hit
            self._remember(key, hit)
            return hit
        return self.mint(key, rank)

    def _possessive(self, key: str, rank: int, named: bool, depth: int) -> str | None:
        """``<routed name>'s`` (both key forms) is the canonical's possessive entity (``_join_entities``'s rule)."""
        for suffix in (" 's", "'s"):
            if key.endswith(suffix) and len(key) > len(suffix):
                target = self.route(key[: -len(suffix)], named)
                if target is None:
                    return None
                display = self.display_of(target)
                owner_key = gp._possessive_forms(display)[1]
                if owner_key == key:
                    return None
                owner = self.resolve(owner_key, rank, depth=depth + 1)
                if owner in self.minted:
                    self.hint.setdefault(owner, f"{display}'s")
                    self.owners.setdefault(owner, display)
                self.add_route(key, owner)
                return owner
        return None

    def _stored_index(self) -> tuple[list, dict[str, list[int]]]:
        if self._index is None:
            entries, postings = [], defaultdict(list)
            for uuid, m in sorted(self.stored.items(), key=lambda kv: (kv[1].run, kv[1].order, kv[0])):
                form = gp.fuzzy_form(m.key)
                if gp.fuzzy_eligible(form):
                    grams = gp.trigram_set(form)
                    for gram in grams:
                        postings[gram].append(len(entries))
                    entries.append((uuid, m.key, grams))
            self._index = (entries, postings)
        return self._index

    def _fuzzy(self, key: str) -> str | None:
        """A stored match (highest Jaccard, then the earliest minted) beats an earlier key of the session
        (the first-occurring); the result follows ``SAME_AS`` links. Same similarity rule as ``fuzzy_clusters``."""
        form = gp.fuzzy_form(key)
        if not gp.fuzzy_eligible(form):
            return None
        grams = gp.trigram_set(form)

        def similar(other_key: str, other: frozenset[str]) -> Fraction | None:
            shared = len(grams & other)
            if not gp._joins(shared, len(grams), len(other)):
                return None
            if gp.possessors(key) != gp.possessors(other_key):
                return None
            return Fraction(shared, len(grams | other))

        entries, postings = self._stored_index()
        best: tuple | None = None
        for i in sorted({i for gram in grams for i in postings.get(gram, ())}):
            uuid, other_key, other = entries[i]
            score = similar(other_key, other)
            if score is not None and (best is None or (-score, *self.minted_at(uuid), uuid) < best[0]):
                best = ((-score, *self.minted_at(uuid), uuid), uuid)
        if best is not None:
            return self.follow(best[1])
        for other_key, other, uuid in self.session:
            if other_key != key and similar(other_key, other) is not None:
                return self.follow(uuid)
        return None

    # -- claims ----------------------------------------------------------------------------------------

    def person(self, key: str, rank: int) -> str:
        """A roster person's node: a session speaker's own; else whatever the name names now, minted if new."""
        if key in self.speakers:
            return self.mint(key, rank)
        return self.existing(key) or self.mint(key, rank)

    def claim(self, c: Claim, rank: int) -> tuple[str, str] | dict:
        """Apply one claim: ``(canonical uuid, routed name)``, or a conflict row. The canonical is the
        holder's node when the holder is a speaker; else the name's when the name is one; else the earliest minted of
        the two (a tie, both minted by the same message, goes to the holder). Two speakers are a conflict. Speakers are
        the bonfire-wide roster (every chat's, stored persons) and this session's speakers."""
        h, n = normalize_exact(c.holder), normalize_exact(c.name)
        # a speaker (of this session, or a roster person) never moves: a speaker holder is ALWAYS canonical, even over
        # an older node of the name; the earliest-minted rule is only for two non-speakers
        persons = set(self.people.display) | self.speakers
        if h in persons and n in persons:
            return self._conflict(c, c.name, content_uuid("ent", self.group, n), content_uuid("ent", self.group, h))
        node_h = self.existing(h)
        node_n = self.existing(n, plain=not _common(n))  # a common-noun name never takes over a plain node
        if h in persons:
            canonical, other = self.person(h, rank), (n, node_n, c.name)
        elif n in persons:
            canonical, other = self.person(n, rank), (h, node_h, c.holder)
        else:
            present = [u for u in (node_h, node_n) if u is not None]
            # ``min`` keeps the first of equal mints: a holder and name minted by the same message -> the holder
            canonical = min(present, key=self.minted_at) if present else self.resolve(h, rank, named=True)
            other = (h, node_h, c.holder) if node_n is not None and canonical == node_n else (n, node_n, c.name)
        canonical = self.follow(canonical)
        key, node, name = other
        routed = self.routes.get(key)
        if routed is not None and self.follow(routed) != canonical:
            return self._conflict(c, name, self.follow(routed), canonical)
        if node is not None and self.follow(node) != canonical:
            if node in self.links:  # already linked to another canonical
                return self._conflict(c, name, self.follow(node), canonical)
            self.links[node] = canonical
            self.link_rows.append(
                {
                    "from_uuid": node,
                    "to_uuid": canonical,
                    "message_id": c.message_id,
                    "holder": c.holder,
                    "name": name,
                    "card": c.card,
                }  # fmt: skip
            )
        self.add_route(key, canonical)
        if not _common(key):
            self.add_route(gp.analyse(name)[1], canonical)  # the lemma form ``_join_entities`` moves too
        if key not in self.speakers:
            self.names.add(key)
        return canonical, name

    def _conflict(self, c: Claim, name: str, existing: str, claimed: str) -> dict:
        return {"name": name, "existing": existing, "claimed": claimed, "message_id": c.message_id}


def resolve(
    records: Sequence[dict],
    *,
    chain: Chain,
    run: int,
    group_id: str,
    people,
    tag_of: Callable[[str, str], str],
    excluded: Iterable[str],
    identity_cards,
    by_order: Callable[[str], tuple],
    referee: Callable[[Point], dict | None] | None = None,
    policy: str = "phrase",
) -> Forward:
    """The forward run's pass 1 and resolution, message by message in source order.

    ``referee`` (None: today's resolution, unchanged) decides a specific mention: a filler with a key that is not a
    person, a guarded name or a deixis referent, whose default is a stored entity or an earlier key of the session, is
    offered as a ``Point``; ``{"join": uuid}`` binds it there, ``{"new": key}`` mints (once per key, no route), None
    keeps the default.

    ``policy`` structure (``referent_policy``): fillers are decomposed as in a full build (``g1_projection._classify``),
    with the session's names plus every stored capitalised name as the known names. Each referent of a role resolves
    on its own (a split role's later referents are offered as ``"<event_key>|<role>|<n>"``, with their conjunct's
    text); a name lifted from inside a filler resolves too, never offered."""
    excluded = frozenset(excluded)
    structure = policy == "structure"
    r = _Resolver(chain, group_id=group_id, run=run, records=records, people=people)
    known: frozenset[str] = frozenset()
    if structure:
        stored_names = {normalize_exact(m.display) for m in chain.entities.values() if capitalised(m.display)}
        known = gp.conversation_names(records) | stored_names
    counts: Counter[str] = Counter()
    rank = gp.message_rank(records)
    session_speakers = list(dict.fromkeys(rec["speaker"] for rec in records))
    by_message: defaultdict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_message[rec["message_id"]].append(rec)
    cands: defaultdict[str, gp._Candidate] = defaultdict(gp._Candidate)
    classified: dict[tuple[str, int, str], gp._Filler | None] = {}
    resolved: dict[str, str] = {}
    canonical_root: dict[str, str] = {}  # canonical uuid -> root key (its key)
    display: dict[str, str] = {}
    aliases: defaultdict[str, set[str]] = defaultdict(set)
    claim_canonicals: dict[str, list[str]] = {}
    conflicts: list[dict] = []
    addressee: dict[str, str | None] = {}

    def bind(uuid: str) -> str:
        key = r.key_of(uuid)
        resolved[key] = uuid
        return key

    for message_id in sorted(by_message, key=by_order):
        recs = sorted(by_message[message_id], key=lambda x: gp.message_order(x["unit"]))
        at = rank[message_id]
        for rec in recs:
            people.grow(rec)
        speaker = recs[0]["speaker"]
        who = r.mint(normalize_exact(speaker), at, hint=speaker)
        cand = cands[bind(who)]
        cand.surfaces[speaker] += 1
        cand.kinds.add("Person")
        cand.evidence.person = True
        cand.person = people.display[normalize_exact(speaker)]

        # the message's claims first (first-person claims first), then its mentions (Review Focus 2)
        # the roster so far plus the session's speakers (known whole-session): a speaker's name is a name here
        named = list(dict.fromkeys([*people.display.values(), *session_speakers]))
        claims = [
            (f"{rec['unit']}#{i}", c)
            for rec in recs
            for i, ev in enumerate(rec["events"])
            if f"{rec['unit']}#{i}" not in excluded
            for c in identity_claims([{**rec, "events": [ev]}], identity_cards, named)
        ]
        for event_key, c in sorted(claims, key=lambda kc: not kc[1].first_person):
            out = r.claim(c, at)
            if isinstance(out, dict):
                conflicts.append(out)
                continue
            canonical, name = out
            root = bind(canonical)
            canonical_root[canonical] = root
            display.setdefault(root, r.display_of(canonical) or c.holder)
            aliases[root].add(name)
            cands[root].person = cands[root].person or display[root]
            if canonical not in claim_canonicals.setdefault(event_key, []):
                claim_canonicals[event_key].append(canonical)

        for rec in recs:
            frames = gp.unit_frames(rec, excluded) if structure else {}
            for index, ev in enumerate(rec["events"]):
                event_key = f"{rec['unit']}#{index}"
                if event_key in excluded:
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
                fillers: dict[str, gp._Filler] = {}
                for role, spec in ev["roles"].items():
                    tag = tag_of(ev["roleset"], role)
                    f = gp._classify(spec, role, rec, people, tag, r.names, **extra)
                    classified[(rec["unit"], index, role)] = f
                    if structure and f is not None:  # a predicative filler keys nothing but supplies a kind noun
                        tags[role], fillers[role] = tag, f
                    if f is None or not gp._all_keyed(f):
                        continue
                    if event_key in claim_canonicals:  # the claim itself: it names its canonical, nothing else
                        f.key, f.parts, f.inner = None, (), ()
                        continue
                    if structure:
                        counts["referents_split"] += len(f.parts) >= 2
                        counts["referents_inner_names"] += len(f.inner)
                    text = str(spec.get("text") or "")
                    for n, x in enumerate(gp._referents(f)):
                        decided = None
                        if referee is not None and not x.person and not x.named and not spec.get("referent"):
                            point = f"{event_key}|{role}" if n == 0 else f"{event_key}|{role}|{n}"
                            default = r.existing(str(x.key), named=x.named)
                            decided = referee(Point(point, rec["message_id"], str(x.key), x.surface,
                                                    x.conjunct or text, default))  # fmt: skip
                        if structure:
                            evidence = filler_evidence(x.person, x.mention, tag)
                            _resolve_filler(r, x, at, decided, set(), people, cands, bind, evidence)
                        else:
                            _resolve_filler(r, x, at, decided, gp._kinds(x.person, tag), people, cands, bind)
                    for x in f.inner:  # a name inside the filler does not fill the role: no role-conferred kind
                        if structure:
                            evidence = filler_evidence(x.person, x.mention, None)
                            _resolve_filler(r, x, at, None, set(), people, cands, bind, evidence)
                        else:
                            _resolve_filler(r, x, at, None, gp._kinds(x.person, None), people, cands, bind)
                if structure:
                    for role, ev_of in frame_evidence(tags, {k: f.mention for k, f in fillers.items()}).items():
                        for x in gp._referents(fillers[role]):  # resolved: each key is its uuid's key now
                            if x.key is not None:
                                cands[x.key].evidence.add(ev_of)
        shown = people.addressee(recs[0])
        found = r.existing(normalize_exact(shown)) if shown else None
        addressee[message_id] = found
        if found is not None:
            bind(found)

    stats = r.stats
    if structure:
        stats |= {k: counts[k] for k in gp.REFERENT_STATS}
    reused = {
        u: {
            "uuid": u,
            "name": r.stored[u].display,
            "labels": ["Entity", "Person"] if r.stored[u].kind == "person" else ["Entity"],
        }
        for u in set(resolved.values())
        if u in r.stored
    }
    joins = Joins(
        canonical={name: root for root, names in aliases.items() for name in names},
        aliases={root: tuple(sorted(names)) for root, names in aliases.items()},
        display=display,
        conflicts=tuple(f"{x['message_id']}: {x['name']!r} is routed elsewhere" for x in conflicts),
    )
    return Forward(
        classified=classified,
        cands=cands,
        resolved=resolved,
        reused=reused,
        rep_key={u: k for u, k in r.minted.items()},
        stats=stats,
        joins=joins,
        joined=canonical_root,
        owners=r.owners,
        claim_events=set(claim_canonicals),
        claim_canonicals=claim_canonicals,
        addressee=addressee,
        links=r.link_rows,
        routes=r.new_routes,
        conflicts=conflicts,
    )


def _resolve_filler(
    r: _Resolver,
    f: gp._Filler,
    at: int,
    decided: Mapping[str, str] | None,
    kinds: set[str],
    people,
    cands: defaultdict[str, gp._Candidate],
    bind: Callable[[str], str],
    evidence: KindEvidence | None = None,
) -> None:
    """One keyed filler resolves (a referee's decision first) and is rebound to the key its uuid was minted under;
    its candidate takes its role-conferred ``kinds`` (phrase) or its kind ``evidence`` (structure)."""
    key = str(f.key)
    if decided is None:
        uuid = r.resolve(key, at, named=f.named)
    elif "join" in decided:
        uuid = r.follow(decided["join"])
    else:
        uuid = r.mint(decided["new"], at, hint=f.surface)
    person = people.display.get(key, f.surface) if f.person else None
    if uuid in r.minted:
        r.hint.setdefault(uuid, person or f.surface)
    f.key = bind(uuid)
    cand = cands[f.key]
    cand.surfaces[f.surface] += 1
    cand.kinds |= kinds
    if evidence is not None:
        cand.evidence.add(evidence)
    if person is not None:
        cand.person = person
    if f.alias:
        cand.aliases.add(f.alias)


# -- what a projection adds to the chain -----------------------------------------------------------------------------


def _is_entity(node: Mapping) -> bool:
    return "Entity" in node["labels_neo4j"] and "Event" not in node["labels_neo4j"]


def _mentions(nodes: Mapping[str, Mapping], edges: Iterable[Mapping], touch: Callable[[str, str | None], None]) -> None:
    """Calls ``touch(message_id, entity uuid)`` for each sender, addressee and MENTIONS target of an Episodic node."""

    episodic = {u: n["props"] for u, n in nodes.items() if "Episodic" in n["labels_neo4j"]}
    for ep in episodic.values():
        touch(ep["message_id"], ep.get("sender"))
        touch(ep["message_id"], ep.get("addressee"))
    for edge in edges:
        if edge["type"] == "MENTIONS" and edge["source"] in episodic:
            touch(episodic[edge["source"]]["message_id"], edge["target"])


def first_mentions(nodes: Mapping[str, Mapping], edges: Iterable[Mapping], rank: Mapping[str, int]) -> dict[str, int]:
    """Entity uuid -> rank of the first message that names it: its sender, its addressee or a MENTIONS edge."""
    first: dict[str, int] = {}

    def touch(message_id: str, uuid: str | None) -> None:
        if uuid is not None and uuid in nodes and _is_entity(nodes[uuid]):
            first[uuid] = min(first.get(uuid, rank[message_id]), rank[message_id])

    _mentions(nodes, edges, touch)
    return first


def mentioned_by(nodes: Mapping[str, Mapping], edges: Iterable[Mapping]) -> dict[str, list[str]]:
    """Entity uuid -> the Episodic message ids (sorted) that name it: as sender, addressee or MENTIONS target."""
    found: defaultdict[str, set[str]] = defaultdict(set)

    def touch(message_id: str, uuid: str | None) -> None:
        if uuid is not None and uuid in nodes and _is_entity(nodes[uuid]):
            found[uuid].add(message_id)

    _mentions(nodes, edges, touch)
    return {u: sorted(ms) for u, ms in sorted(found.items())}


def emit(
    *,
    nodes: Mapping[str, Mapping],
    edges: Sequence[Mapping],
    records: Sequence[Mapping],
    merged: Mapping[str, Any],
    rep_key: Mapping[str, str],
    stored: Collection[str],
    run: int,
    routes: Mapping[str, str],
    links: Sequence[Mapping] = (),
    conflicts: Sequence[Mapping] = (),
    summary_touched: Collection[str] = (),
    strict: bool = False,
    persons: Collection[str] | None = None,
) -> dict:
    """The run's ``forward``: ``minted`` (each entity this projection creates, ordered by its first mention: run,
    message rank), ``routes`` (key -> uuid, each to a stored or minted entity), ``same_as``, ``conflicts`` and
    ``touched`` (the entities this projection writes rows for: new mints, and stored nodes with a new mention, fact,
    or summary line) and ``mentioned`` (entity uuid -> the session's Episodic message ids that name it).

    A full build never raises here: an entity no message names is left out of ``minted`` (``chain.from_bootstrap``
    refuses such a first mint) and a route to an unwritten entity is dropped. ``strict`` (a forward run) raises
    ``ValueError`` on either instead.

    ``persons`` (``referent_policy`` structure): a mint's ``kind`` is ``person`` iff its uuid is listed (a speaker,
    roster, deixis or join person), so a ``Person`` label from a WordNet class never makes a roster person. None:
    ``person`` iff the node carries the ``Person`` label (today's rule)."""
    from memory_kernel.pipeline.chain import Minted, SameAs

    rank = gp.message_rank(records)
    first = first_mentions(nodes, edges, rank)
    minted: list[Minted] = []
    for uuid, node in nodes.items():
        if not _is_entity(node) or uuid in stored:
            continue
        props = node["props"]
        if uuid not in first:
            if strict:
                raise ValueError(f"forward: entity {uuid} ({props['name']!r}) is never mentioned: no mint order")
            continue
        cand = merged.get(uuid)
        surfaces = {props["name"], *(props.get("aliases") or ()), *(props.get("identity_aliases") or ())}
        surfaces |= set(cand.surfaces) if cand is not None else set()
        person = uuid in persons if persons is not None else "Person" in node["labels_neo4j"]
        minted.append(
            Minted(
                uuid=uuid,
                key=rep_key[uuid],
                kind="person" if person else "entity",
                display=props["name"],
                surfaces=tuple(sorted(surfaces)),
                run=run,
                order=first[uuid],
            )
        )
    minted.sort(key=lambda m: (m.order, m.uuid))
    known = {m.uuid for m in minted} | set(stored)
    stray = sorted(k for k, u in routes.items() if u not in known)
    if stray and strict:
        raise ValueError(f"forward: routes to no stored or minted entity: {stray[:5]}")
    routes = {k: u for k, u in routes.items() if u in known}
    touched = {m.uuid for m in minted} | set(summary_touched)
    for edge in edges:
        if edge["type"] == "MENTIONS" or edge["type"] == "RELATES_TO":
            touched |= {u for u in (edge["source"], edge["target"]) if u in nodes and _is_entity(nodes[u])}
    same_as = [
        SameAs(
            from_uuid=x["from_uuid"],
            to_uuid=x["to_uuid"],
            message_id=x["message_id"],
            holder=x["holder"],
            name=x["name"],
            card=x["card"],
            run=run,
        )
        for x in sorted(links, key=lambda x: (rank.get(x["message_id"], 0), x["from_uuid"], x["to_uuid"]))
    ]
    return {
        "minted": minted,
        "routes": dict(sorted(routes.items())),
        "same_as": same_as,
        "conflicts": list(conflicts),
        "touched": sorted(touched),
        "mentioned": mentioned_by(nodes, edges),
    }


def full_build_routes(
    *,
    resolved: Mapping[str, str],
    rep_key: Mapping[str, str],
    written: Collection[str],
    joins: Joins,
    names,
    owners: Mapping[str, str],
) -> dict[str, str]:
    """The routes of a full build's first mint: every key that resolved onto another key's node (a fuzzy merge, a
    key form ``_join_entities`` moved), every claimed name (its exact form; the lemma form of one that is not a common
    noun) onto its canonical, and the possessive forms of the claimed names onto the canonical's possessive entity.
    Only routes onto entities the rows write."""
    routes: dict[str, str] = {}

    def add(key: str, uuid: str | None) -> None:
        if uuid is not None and uuid in written and rep_key.get(uuid) != key:
            routes.setdefault(key, uuid)

    for key in sorted(resolved):
        add(key, resolved[key])
    canonical_of = {root: resolved.get(root) for root in joins.aliases}
    owner_of = {display: uuid for uuid, display in owners.items()}
    for root in sorted(joins.aliases):
        target = canonical_of[root]
        for alias in joins.aliases[root]:
            key = normalize_exact(alias)
            add(key, target)
            common = names.common(key) if names is not None else True
            if not common:
                add(gp.analyse(alias)[1], target)
                for form in gp._possessive_forms(alias):
                    add(form, owner_of.get(joins.display[root]))
    return routes
