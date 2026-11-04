"""Referent resolution of a forward append: one entity store per bonfire, resolution of a specific
mention scoped to the conversation.

A decision point (``Point``) is a filler the resolver would resolve to a stored entity or to an earlier key of the
session. Its form decides who resolves it (``mention_form``):
- named, kind or pronoun: today's resolution (global, or the antecedent picker's);
- indefinite in a statement: a new referent of the session;
- specific: the session's earlier referent of the key; else the stored candidates in its common ground (``Ground``:
  same chat, or one of its people took part), an easy join when one is in this chat and salient, a hard case for Jev
  otherwise, a new referent when none is shared.

Pure: no I/O, no model call. The stage (``pipeline.stages.referents``) asks Jev and calls ``settle``."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime

from memory_kernel.graph import g1_nlp
from memory_kernel.graph.envelope import message_mood, sentence_mood

SALIENCE_S = 3 * 86400
TAU = 0.5
MAX_CANDIDATES = 3


@dataclass(frozen=True)
class Ground:
    """Where and with whom a stored entity was mentioned (one row of ``chain/ground.jsonl``)."""

    uuid: str
    chats: tuple[str, ...]
    people: tuple[str, ...]
    first_mid: str
    last_mid: str
    last_at: str
    mentions: int
    legacy: bool = True  # existed before the kind/instance split: a plain key may still name an instance

    def to_row(self) -> dict:
        return {**asdict(self), "chats": list(self.chats), "people": list(self.people)}

    @classmethod
    def from_row(cls, row: Mapping) -> Ground:
        return cls(
            **{**row, "chats": tuple(row["chats"]), "people": tuple(row["people"]), "legacy": row.get("legacy", True)}
        )


def person_key(name: str) -> str:
    """How a person or an agent handle is compared: a leading "@" stripped, then ``normalize_exact`` (an addressee
    naming a handle is its bare name, ``_conversation.build_prepared``; a handle may be given with or without "@")."""
    return g1_nlp.normalize_exact(name.strip().lstrip("@"))


def _people(row: Mapping, agents: Collection[str]) -> list[str]:
    """The row's speaker and addressees (``person_key``), less ``agents`` (already ``person_key``-normalized)."""
    names = [row.get("speaker") or "", *(row.get("addressees") or ())]
    return [n for n in (person_key(x) for x in names if x) if n and n not in agents]


def _at(row: Mapping) -> str:
    return str(row.get("timestamp") or "")


def fold_ground(
    ground: Mapping[str, Ground],
    mentioned: Mapping[str, Iterable[str]],
    rows: Mapping[str, Mapping],
    chat_of: Callable[[str], str],
    agents: Collection[str],
    *,
    legacy: bool,
) -> dict[str, Ground]:
    """``ground`` with each entity's new mentions folded in (chats and people only grow, ``last_*`` only advances).
    ``agents`` are agent handles (with or without "@"): never people of a ground. A new row takes ``legacy`` (whether
    its entity predates the kind/instance split); an existing row keeps its own."""
    agents = {person_key(a) for a in agents}
    out = dict(ground)
    for uuid in sorted(mentioned):
        mids = sorted(set(mentioned[uuid]), key=lambda m: (_at(rows[m]), m))
        if not mids:
            continue
        old = out.get(uuid)
        chats = list(old.chats) if old else []
        people = list(old.people) if old else []
        for m in mids:
            c = chat_of(m)
            if c not in chats:
                chats.append(c)
            for p in _people(rows[m], agents):
                if p not in people:
                    people.append(p)
        last = mids[-1]
        newer = old is None or (_at(rows[last]), last) >= (old.last_at, old.last_mid)
        out[uuid] = Ground(
            uuid=uuid,
            chats=tuple(chats),
            people=tuple(people),
            first_mid=old.first_mid if old else mids[0],
            last_mid=last if newer else old.last_mid,
            last_at=_at(rows[last]) if newer else old.last_at,
            mentions=(old.mentions if old else 0) + len(mids),
            legacy=old.legacy if old else legacy,
        )
    return out


@dataclass(frozen=True)
class Point:
    """A decision point of the resolver (``forward.resolve``'s referee hook): ``point`` is ``"<event_key>|<role>"``,
    ``default`` the uuid today's resolution gives, or ``None`` when the key resolves to nothing yet."""

    point: str
    message_id: str
    key: str
    surface: str
    text: str
    default: str | None


def mention_form(text: str, context: str | None = None) -> str:
    """named | kind | pronoun | indefinite | specific, from the parse of the filler (UD features; leading ADP
    stripped): a PROPN or entity-typed head, an @handle or a possessor is named; a personal pronoun head is a
    pronoun; an indefinite determiner (``Definite=Ind``) is indefinite; a bare common noun (no determiner, possessor or
    number) is a kind; anything else is specific. With ``context`` (the message), the filler's first occurrence is
    parsed inside it, where a lone name tags PROPN; else the text is parsed alone."""
    if text.lstrip().startswith("@"):
        return "named"
    doc = None
    span = None
    if context and text.strip():
        at = context.lower().find(text.strip().lower())
        if at >= 0:
            doc = g1_nlp.nlp()(context)
            span = doc.char_span(at, at + len(text.strip()), alignment_mode="expand")
    if span is None:
        doc = g1_nlp.nlp()(text)
        span = doc[:]
    start = span.start
    while start < span.end and doc[start].pos_ == "ADP":
        start += 1
    span = doc[start : span.end]
    if not len(span):
        return "specific"
    head = span.root
    deps = [c for c in head.children if span.start <= c.i < span.end]
    if head.pos_ == "PROPN" or head.ent_type_ or any(c.dep_ == "poss" for c in deps):
        return "named"
    if head.pos_ == "PRON":
        return "pronoun"
    dets = [c for c in deps if c.dep_ in ("det", "predet")]
    if any("Ind" in c.morph.get("Definite", []) for c in dets):
        return "indefinite"
    if head.pos_ == "NOUN" and not dets and not any(c.dep_ == "nummod" for c in deps):
        return "kind"
    return "specific"


def is_question(text: str) -> bool:
    return message_mood([sentence_mood(s) for s in g1_nlp.nlp()(text).sents]) == "interrogative"


_DISCRIMINATED = re.compile(r"(.+)#[0-9a-f]{8}")


def split_discriminated(key: str) -> tuple[str, bool]:
    """``(base key, is discriminated)``: only a trailing ``#`` + 8 lowercase hex marks an instance; a plain
    key may itself hold a "#" (``c#``, ``#general channel``)."""
    m = _DISCRIMINATED.fullmatch(key)
    return (m.group(1), True) if m else (key, False)


def discriminated_key(key: str, message_id: str) -> str:
    return f"{key}#{hashlib.sha256(message_id.encode()).hexdigest()[:8]}"


@dataclass(frozen=True)
class HardCase:
    index: int
    point: Point
    candidates: tuple[str, ...]


def _ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def plan_session(
    points: Sequence[Point],
    *,
    chat: str,
    rows: Mapping[str, Mapping],
    ground: Mapping[str, Ground],
    stored_keys: Mapping[str, str],
    agents: Collection[str],
) -> tuple[dict[str, dict], list[HardCase], dict]:
    """Provisional decisions, in point order (the session's message order): ``{"join": uuid}`` (an easy join),
    ``{"session": n}`` (session referent n: an earlier one of the key, or a new one) or ``{"hard": i, "session": n}``
    (a new session referent n that Jev may join to a stored candidate). Points named, kind or pronoun get no decision.
    ``stored_keys``: stored uuid -> its key (``Minted.key``). A stored entity is an instance candidate when its key is
    discriminated (``k#<8 hex>``) or its ground row is ``legacy``: the plain key names the kind."""
    agents = {person_key(a) for a in agents}
    family: dict[str, list[str]] = {}
    for uuid, key in stored_keys.items():
        family.setdefault(split_discriminated(key)[0], []).append(uuid)
    provisional: dict[str, dict] = {}
    hard: list[HardCase] = []
    session_of_key: dict[str, int] = {}  # key -> its latest session referent
    session_join: dict[int, str] = {}  # session referent -> the stored uuid it joined (easy)
    counts = {"points": len(points), "named_kind_pronoun": 0, "new_intro": 0, "session_join": 0,
              "easy_join": 0, "new_no_ground": 0, "hard": 0}  # fmt: skip
    referents = 0
    for p in points:
        form = mention_form(p.text, rows[p.message_id].get("text"))
        if form in ("named", "kind", "pronoun"):
            counts["named_kind_pronoun"] += 1
            continue
        message = rows[p.message_id]
        if form == "indefinite" and not is_question(str(message.get("text") or "")):
            session_of_key[p.key] = referents
            provisional[p.point] = {"session": referents}
            referents += 1
            counts["new_intro"] += 1
            continue
        if p.key in session_of_key:
            n = session_of_key[p.key]
            provisional[p.point] = {"join": session_join[n]} if n in session_join else {"session": n}
            counts["session_join"] += 1
            continue
        now = _ts(str(message["timestamp"]))
        people = set(_people(message, agents))
        shared = [
            u for u in {*family.get(split_discriminated(p.key)[0], []), p.default} - {None}
            if u in ground and (split_discriminated(stored_keys.get(u, ""))[1] or ground[u].legacy)
            and (chat in ground[u].chats or people & set(ground[u].people))
        ]  # fmt: skip
        shared.sort(key=lambda u: (ground[u].last_at, u), reverse=True)
        easy = [u for u in shared if chat in ground[u].chats and now - _ts(ground[u].last_at) <= SALIENCE_S]
        n = referents
        referents += 1
        session_of_key[p.key] = n
        if easy:
            session_join[n] = easy[0]
            provisional[p.point] = {"join": easy[0]}
            counts["easy_join"] += 1
        elif shared:
            hard.append(HardCase(len(hard), p, tuple(shared[:MAX_CANDIDATES])))
            provisional[p.point] = {"hard": len(hard) - 1, "session": n}
            counts["hard"] += 1
        else:
            provisional[p.point] = {"session": n}
            counts["new_no_ground"] += 1
    return provisional, hard, counts


def session_keys(points: Sequence[Point], provisional: Mapping[str, dict]) -> dict[int, tuple[str, str]]:
    """Session referent -> (its key, the message that introduced it): the first point that names it."""
    out: dict[int, tuple[str, str]] = {}
    for p in points:
        n = (provisional.get(p.point) or {}).get("session")
        if n is not None and n not in out:
            out[n] = (p.key, p.message_id)
    return out


def settle(
    provisional: Mapping[str, dict],
    hard: Sequence[HardCase],
    p: Mapping[tuple[int, str], float],
    *,
    session_keys: Mapping[int, tuple[str, str]],
) -> dict[str, dict]:
    """Final decisions: ``{"join": uuid}`` or ``{"new": key}``. A hard case joins its candidate with the highest
    p(yes) when it is >= TAU (``KeyError`` when an answer is missing: never read as "no"). A new referent is always
    ``discriminated_key``: the plain key names the kind. Two referents of one key introduced in one message
    would share it: the later takes ``discriminated_key(key, f"{message_id}|{n}")``."""
    joined: dict[int, str] = {}
    for h in hard:
        n = provisional[h.point.point]["session"]
        best = max(h.candidates, key=lambda u: (p[(h.index, u)], u))
        if p[(h.index, best)] >= TAU:
            joined[n] = best
    final_key: dict[int, str] = {}
    for n in sorted(session_keys):
        if n in joined:
            continue
        key, mid = session_keys[n]
        k = discriminated_key(key, mid)
        if k in final_key.values():  # same key, same message: a distinct, deterministic key per referent
            k = discriminated_key(key, f"{mid}|{n}")
        final_key[n] = k
    out: dict[str, dict] = {}
    for point, d in provisional.items():
        if "join" in d:
            out[point] = {"join": d["join"]}
        elif d["session"] in joined:
            out[point] = {"join": joined[d["session"]]}
        else:
            out[point] = {"new": final_key[d["session"]]}
    return out
