"""Read-only memory operations of the runtime grammar, run against the bonfire's graph.

``StoreMemory`` implements the executor's ``MemoryOps`` protocol. Every query goes through a :class:`Reader` (a READ
session) scoped by ``$g`` (the group id), and is recorded in ``StoreMemory.queries`` for the trace. Entities travel
between ops as plain dicts ``{"uuid", "name", "kinds"}`` and events as ``{"uuid", "name", "source_text", ...}``; an
op reading an entity or event input also accepts a list of them, a bare uuid string, or ``None`` (reads nothing).
``find-entity``'s name may instead be the turn window's ``Referent`` (``window``'s output: an already-resolved
entity), which it returns as it is, with no query; no other value passes through.
Optional inputs (``participant``, ``role``, ``frame``) are absent when ``None`` or empty. Undated items are dropped by
``order`` and come back as ``None`` from ``time-of``.

Verb-sense ambiguity travels as a set: ``events`` takes one frame or a list of frames (the ``frames-of`` output) and
reads events of any of them. Its ``participant`` is one participant (an entity, a uuid, or a flat list of entities:
the candidates ``find-entity`` returns for one name, any of them) or a list of such lists, which the event must have
ALL of (an intersection; ``role`` is then one role or a list parallel to them); a participant that names nobody (an
empty list) reads no event. A role is one of the closed constants ``arg0``..``arg5`` or ``obj`` (any core role but
arg0: the participant of a question whose subject is asked, "who told Milo?", may be any non-subject role), or absent
(any role). ``role-filler`` takes one role, a list of roles, or the ``(frame, role)`` pairs ``wh-role``
returns (a pair reads its role only on events of its frame), and never returns an entity of its ``except`` input (the
participants the question gives: "who met Kira Dace?" never answers Kira Dace). ``slot-of-kind(value ← lines, kind)`` reads
the graph entities of one store kind (``g1_names.KIND_LABELS``: Person, Organisation, Artifact, Place), or of any kind
of a set (``role-kinds``' output; an empty set reads nothing), that the lines name; a kind is a constant of the
construction's meaning or a label from the lexicon, matched to the store's label whatever its case (FCG hands meaning
constants back lowercased).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from memory_kernel.graph.g1_names import KIND_LABELS
from memory_kernel.runtime.executor import MemoryOps, Referent, item_date
from memory_kernel.runtime.program import OpSpec
from memory_kernel.search.summary_lines import lines_of, note_missing

MEMORY_SPECS: Mapping[str, OpSpec] = {
    spec.name: spec
    for spec in (
        OpSpec("find-entity", ("name",), ("entity",), "memory"),
        OpSpec("events", ("frame", "participant", "role"), ("events",), "memory"),
        OpSpec("role-filler", ("event", "role", "except"), ("value",), "memory"),
        OpSpec("identity-lines", ("entity",), ("lines",), "memory"),
        OpSpec("state-lines", ("entity", "frame"), ("lines",), "memory"),
        OpSpec("slot-of-kind", ("lines", "kind"), ("value",), "memory"),
        OpSpec("time-of", ("event",), ("time",), "memory"),
        OpSpec("source", ("item",), ("source",), "memory"),
        OpSpec("mentions", ("entity",), ("mentions",), "memory"),
        OpSpec("count", ("set",), ("n",), "memory"),
        OpSpec("order", ("set", "end"), ("item",), "memory"),
        OpSpec("recall", ("text",), ("context",), "memory"),
    )
}

#: Most events one ``events`` op returns.
EVENTS_LIMIT = 200


FIND_ENTITY_EXACT = """
MATCH (n:Entity {group_id: $g}) WHERE n.name = $name AND NOT n:Event
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels ORDER BY uuid"""
FIND_ENTITY_FOLDED = """
MATCH (n:Entity {group_id: $g}) WHERE toLower(n.name) = toLower($name) AND NOT n:Event
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels ORDER BY uuid"""
EVENTS_CYPHER = """
MATCH (ev:Entity:Event {group_id: $g})
WHERE (ev.roleset IN $frames
       OR EXISTS { MATCH (ev)-[f:RELATES_TO {group_id: $g}]->(:Entity) WHERE f.roleset IN $frames })
  AND all(grp IN $groups WHERE EXISTS {
        MATCH (ev)-[p:RELATES_TO {group_id: $g}]->(x:Entity {group_id: $g})
        WHERE x.uuid IN grp.uuids AND (grp.roles IS NULL OR p.role IN grp.roles) })
RETURN ev.uuid AS uuid, ev.name AS name, ev.source_text AS source_text,
       toString(ev.happened_at) AS happened_at, toString(ev.told_at) AS told_at
ORDER BY happened_at, uuid
LIMIT $limit"""
ROLE_FILLER_CYPHER = """
MATCH (ev:Entity:Event {group_id: $g})-[r:RELATES_TO {group_id: $g}]->(n:Entity {group_id: $g})
WHERE ev.uuid IN $uuids
  AND any(p IN $pairs WHERE r.role = p[1] AND (p[0] IS NULL OR ev.roleset = p[0] OR r.roleset = p[0]))
RETURN ev.uuid AS event, n.uuid AS uuid, n.name AS name, labels(n) AS labels ORDER BY event, uuid"""
SUMMARY_CYPHER = """
MATCH (n:Entity {group_id: $g}) WHERE n.uuid IN $uuids AND NOT n:Event
RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, n.summary_line_ids AS line_ids,
       n.summary_line_texts AS line_texts, n.summary_identity_line_ids AS identity_line_ids ORDER BY uuid"""
KIND_MENTIONS_CYPHER = """
MATCH (n:Entity {group_id: $g}) WHERE $kind IN labels(n) AND NOT n:Event AND toLower($text) CONTAINS toLower(n.name)
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels ORDER BY uuid"""
TIME_CYPHER = """
MATCH (ev:Entity:Event {group_id: $g}) WHERE ev.uuid IN $uuids
RETURN ev.uuid AS uuid, toString(ev.happened_at) AS happened_at, toString(ev.told_at) AS told_at ORDER BY uuid"""
SOURCE_CYPHER = """
MATCH (x {group_id: $g}) WHERE x.uuid IN $uuids
OPTIONAL MATCH (ep:Episodic {group_id: $g})-[:MENTIONS]->(x)
WITH x, collect(DISTINCT ep) + CASE WHEN x:Episodic THEN [x] ELSE [] END AS eps
UNWIND eps AS e
OPTIONAL MATCH (s:Saga {group_id: $g})-[:HAS_EPISODE]->(e)
RETURN x.uuid AS uuid, e.uuid AS message, s.uuid AS saga, s.name AS session ORDER BY uuid, message"""
MENTIONS_CYPHER = """
MATCH (ep:Episodic {group_id: $g})-[:MENTIONS]->(n:Entity {group_id: $g}) WHERE n.uuid IN $uuids
RETURN DISTINCT ep.uuid AS uuid, ep.content AS content, ep.sender_name AS sender, toString(ep.valid_at) AS valid_at
ORDER BY valid_at, uuid"""


class Reader(Protocol):
    async def rows(self, cypher: str, params: Mapping[str, Any]) -> list[dict]:
        """Run one READ query and return its rows."""
        ...


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _uuids(value: Any) -> list[str]:
    out: list[str] = []
    for item in _items(value):
        uuid = item.get("uuid") if isinstance(item, Mapping) else item
        if isinstance(uuid, str) and uuid and uuid not in out:
            out.append(uuid)
    return out


def _entity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "uuid": row["uuid"],
        "name": row.get("name"),
        "kinds": [label for label in row.get("labels") or () if label in KIND_LABELS],
    }


def _resolved(value: Any) -> list[dict] | None:
    """A turn-window ``Referent`` as ``find-entity``'s output (one already-resolved entity), else None: any other value
    is a name to look up (a dict that merely looks resolved is never trusted)."""
    if not isinstance(value, Referent):
        return None
    return [{"uuid": value.uuid, "name": value.name, "kinds": list(value.kinds)}]


def _need(args: Mapping[str, Any], op: str, key: str) -> Any:
    if args.get(key) is None:
        raise ValueError(f"{op} needs its {key!r} input")
    return args[key]


def _strings(value: Any) -> list[str]:
    """One string or a list of them (a frame, the ``frames-of`` set), de-duplicated in order."""
    out: list[str] = []
    for item in _items(value):
        if isinstance(item, str) and item and item not in out:
            out.append(item)
    return out


def _participant_groups(participant: Any, role: Any) -> list[dict[str, Any]] | None:
    """``events``' participants as groups the event must ALL have: a list of lists is one group per inner list (one
    participant's candidates, any of them); a flat list of entities (what ``find-entity`` returns: the candidates for
    one name) or one entity / uuid is one group. ``role`` is one role for every group, or a list parallel to the groups.
    No participant (None) is no group; a participant that names nobody (an empty list or group) is None."""
    if participant is None:
        return []
    items = _items(participant)
    nested = bool(items) and all(isinstance(i, (list, tuple)) for i in items)
    candidates = [_uuids(i) for i in items] if nested else [_uuids(items)]
    if not all(candidates):
        return None
    roles = _items(role) if isinstance(role, (list, tuple)) else [_present(role)] * len(candidates)
    if len(roles) != len(candidates):
        raise ValueError(f"events: {len(roles)} roles for {len(candidates)} participants")
    return [{"uuids": uuids, "roles": _event_roles(r)} for uuids, r in zip(candidates, roles, strict=True)]


#: ``events``' role constants: a core role, or ``obj`` (any core role but the subject's arg0)
CORE_ROLES = tuple(f"arg{n}" for n in range(6))
NON_SUBJECT = "obj"


def _event_roles(role: Any) -> list[str] | None:
    """The PropBank roles a participant may hold: None for any role; ``arg0``..``arg5`` for that one; ``obj`` for
    arg1..arg5. Any other constant is refused (the set is closed; FCG hands constants back lowercased)."""
    role = _present(role)
    if role is None:
        return None
    key = str(role).casefold()
    if key == NON_SUBJECT:
        return list(CORE_ROLES[1:])
    if key in CORE_ROLES:
        return [key]
    raise ValueError(f"events: role {role!r} is not one of {list(CORE_ROLES)} or {NON_SUBJECT!r}")


def _role_pairs(value: Any) -> list[list[str | None]]:
    """``[frame | None, role]`` pairs from one role, a list of roles, or ``wh-role``'s ``(frame, role)`` pairs."""
    out: list[list[str | None]] = []
    for item in _items(value):
        if isinstance(item, str):
            pair: list[str | None] = [None, item]
        elif isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, str) for x in item):
            pair = [item[0], item[1]]
        else:
            raise ValueError(f"role-filler: a role is a string or a (frame, role) pair, got {item!r}")
        if pair not in out:
            out.append(pair)
    return out


#: a store kind label by its casefold (FCG hands a meaning constant such as "Place" back as "place")
_KIND_BY_FOLD: Mapping[str, str] = {label.casefold(): label for label in KIND_LABELS}


def _present(value: Any) -> Any:
    return None if value is None or value == "" else value


_date = item_date


class StoreMemory(MemoryOps):
    """Memory operations over one bonfire's graph; ``recall`` is the fallback ``/context`` callable."""

    def __init__(
        self,
        reader: Reader,
        group_id: str,
        *,
        recall: Callable[[str], Awaitable[dict]] | None = None,
    ) -> None:
        self._reader = reader
        self._group = group_id
        self._recall = recall
        self.queries: list[dict[str, Any]] = []  # {"op", "cypher", "params"} in run order, for the trace

    async def _q(self, op: str, cypher: str, **params: Any) -> list[dict]:
        full = {"g": self._group, **params}
        self.queries.append({"op": op, "cypher": cypher, "params": full})
        return await self._reader.rows(cypher, full)

    async def run(self, op: str, args: Mapping[str, Any]) -> Any:
        handler = getattr(self, "_op_" + op.replace("-", "_"), None)
        if op not in MEMORY_SPECS or handler is None:
            raise KeyError(f"unknown memory operation {op!r}")
        return await handler(args)

    async def _op_find_entity(self, args: Mapping[str, Any]) -> list[dict]:
        given = _need(args, "find-entity", "name")
        resolved = _resolved(given)
        if resolved is not None:
            return resolved  # a referent the turn window already resolved: never looked up again by name
        name = str(given).strip()
        rows = await self._q("find-entity", FIND_ENTITY_EXACT, name=name)
        if not rows:
            rows = await self._q("find-entity", FIND_ENTITY_FOLDED, name=name)
        return [_entity(r) for r in rows]

    async def _op_events(self, args: Mapping[str, Any]) -> list[dict]:
        frames = _strings(_need(args, "events", "frame"))
        if not frames:
            return []
        groups = _participant_groups(args.get("participant"), args.get("role"))
        if groups is None:
            return []  # a participant was given and names nobody: no event has it
        rows = await self._q("events", EVENTS_CYPHER, frames=frames, limit=EVENTS_LIMIT, groups=groups)
        return [dict(r) for r in rows]

    async def _op_role_filler(self, args: Mapping[str, Any]) -> list[dict]:
        uuids = _uuids(args.get("event"))
        if not uuids:
            return []
        pairs = _role_pairs(_need(args, "role-filler", "role"))
        if not pairs:
            return []
        rows = await self._q("role-filler", ROLE_FILLER_CYPHER, uuids=uuids, pairs=pairs)
        given = set(_uuids(args.get("except")))
        seen: dict[str, dict] = {}
        for r in rows:
            if r["uuid"] not in given:
                seen.setdefault(r["uuid"], _entity(r))
        return list(seen.values())

    async def _summaries(self, op: str, entity: Any) -> list[dict]:
        uuids = _uuids(entity)
        return await self._q(op, SUMMARY_CYPHER, uuids=uuids) if uuids else []

    async def _op_identity_lines(self, args: Mapping[str, Any]) -> list[str]:
        return await self._identity("identity-lines", args.get("entity"))

    async def _identity(self, op: str, entity: Any) -> list[str]:
        return self._identity_of(await self._summaries(op, entity), self._group)

    @staticmethod
    def _identity_of(rows: list[dict], group: str = "") -> list[str]:
        """The identity lines among the rows' stored line records; a row without them (built before 1.4.1) has none."""
        lines: list[str] = []
        for row in rows:
            recs = lines_of(row)
            if recs is None:
                note_missing(group)
                continue
            lines += [r.text for r in recs if r.identity and r.text.strip() and r.text not in lines]
        return lines

    async def _op_state_lines(self, args: Mapping[str, Any]) -> list[str]:
        entity, frame = args.get("entity"), _present(args.get("frame"))
        rows = await self._summaries("state-lines", entity)
        filtered = self._frame_filter(rows, frame)
        if filtered is not None:
            return filtered
        return self._identity_of(rows, self._group)  # the summaries already read: one query

    def _frame_filter(self, rows: list[dict], frame: Any) -> list[str] | None:
        """Seam for Plan B's line frames: return the summary lines tagged with ``frame``, or None while lines carry
        no frames (today), which makes ``state-lines`` fall back to the identity lines."""
        return None

    async def _kind_slot(self, op: str, kinds: list[str], lines: Any) -> list[dict]:
        texts = [str(x) for x in _items(lines)]
        text = "\n".join(texts)
        if not text.strip() or not kinds:
            return []
        rows: list[dict] = []
        for kind in kinds:  # one query per kind, in the given order (one kind: the single query it always was)
            rows += await self._q(op, KIND_MENTIONS_CYPHER, kind=kind, text=text)
        # Longest non-overlapping matches only: "FriedApple team" must not also yield the entity "team".
        spans: list[tuple[int, int, dict]] = []
        for r in rows:
            name = r.get("name") or ""
            if name:
                pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
                spans += [(m.start(), m.end(), r) for m in re.finditer(pattern, text, re.IGNORECASE)]
        taken: list[tuple[int, int]] = []
        found: dict[str, dict] = {}
        for start, end, r in sorted(spans, key=lambda t: (t[0] - t[1], t[0])):
            if all(end <= a or start >= b for a, b in taken):
                taken.append((start, end))
                found.setdefault(r["uuid"], _entity(r))
        return list(found.values())

    async def _op_slot_of_kind(self, args: Mapping[str, Any]) -> list[dict]:
        labels: list[str] = []
        for kind in _items(_need(args, "slot-of-kind", "kind")):
            label = _KIND_BY_FOLD.get(str(kind).casefold())
            if label is None:
                raise ValueError(f"slot-of-kind: kind {kind!r} is not a store kind label {list(KIND_LABELS)}")
            if label not in labels:
                labels.append(label)
        return await self._kind_slot("slot-of-kind", labels, args.get("lines"))

    async def _op_time_of(self, args: Mapping[str, Any]) -> Any:
        event = args.get("event")
        uuids = _uuids(event)
        if not uuids:
            return None
        by_uuid = {r["uuid"]: _date(r) for r in await self._q("time-of", TIME_CYPHER, uuids=uuids)}
        times = [by_uuid.get(u) for u in uuids]
        return times if isinstance(event, (list, tuple)) else times[0]

    async def _op_source(self, args: Mapping[str, Any]) -> list[dict]:
        uuids = _uuids(args.get("item"))
        if not uuids:
            return []
        rows = await self._q("source", SOURCE_CYPHER, uuids=uuids)
        out: list[dict] = []
        for r in rows:
            entry = {"item": r["uuid"], "message": r.get("message"), "saga": r.get("saga"), "session": r.get("session")}
            if entry not in out:
                out.append(entry)
        return out

    async def _op_mentions(self, args: Mapping[str, Any]) -> list[dict]:
        uuids = _uuids(args.get("entity"))
        return [dict(r) for r in await self._q("mentions", MENTIONS_CYPHER, uuids=uuids)] if uuids else []

    async def _op_count(self, args: Mapping[str, Any]) -> int:
        return len(_items(args.get("set")))

    async def _op_order(self, args: Mapping[str, Any]) -> Any:
        end = _need(args, "order", "end")
        if end not in ("first", "latest"):
            raise ValueError(f"order takes 'first' or 'latest', got {end!r}")
        dated = [(d, x) for x in _items(args.get("set")) if (d := _date(x)) is not None]
        if not dated:
            return None
        dated.sort(key=lambda p: p[0])  # ISO dates sort lexically; the sort is stable
        return (dated[0] if end == "first" else dated[-1])[1]

    async def _op_recall(self, args: Mapping[str, Any]) -> dict:
        if self._recall is None:
            raise RuntimeError("recall needs a context callable: StoreMemory(reader, group_id, recall=...)")
        return await self._recall(str(_need(args, "recall", "text")))
