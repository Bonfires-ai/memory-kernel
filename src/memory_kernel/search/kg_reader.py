"""Neo4j reader for the answer-context rows over the Graphiti-shaped graph: windows, entity sheets, timeline.

Message search itself is ``kg_recipe.search_messages_v2`` (dense cosine over expanded keys, preloaded index); this
module reads the rows around the hits and assembles the answer context. Every query is scoped to one ``group_id``.
Reading only: nothing here writes. No Graphiti library.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from memory_kernel.graph.identity import alias_occurs, is_common_noun
from memory_kernel.graph.kg_contract import RESOLVED_TIME_SOURCES, STRUCTURAL_EDGES, ext
from memory_kernel.search.kg_context import ContextConfig, assemble_context
from memory_kernel.search.kg_recipe import KgRecipeConfig
from memory_kernel.search.kg_types import MessageHit, SearchResult
from memory_kernel.search.summary_lines import SheetLine, lines_of, note_missing

_STRUCTURAL = tuple(STRUCTURAL_EDGES)


async def _rows(driver, cypher: str, **params: Any) -> list[dict[str, Any]]:
    async with driver.session() as s:
        result = await s.run(cypher, **params)
        return [dict(r) async for r in result]


# --- answer-context readers (windows, entity sheets, timeline) and the orchestrator -------------------------------


def _msg_projection(v: str) -> str:
    return (
        f"{{uuid: {v}.uuid, message_id: {v}.message_id, content: {v}.content, sender_name: {v}.sender_name, "
        f"addressee_name: {v}.addressee_name, act: {v}.act, mood: {v}.mood, "
        f"valid_at: toString({v}.valid_at), told_at: toString({v}.told_at), "
        f"source_description: {v}.source_description, attributes: {v}.attributes}}"
    )


WINDOW_ANCHORS_CYPHER = f"""
MATCH (m:Episodic {{group_id: $g}}) WHERE m.uuid IN $uuids
OPTIONAL MATCH (s:Saga {{group_id: $g}})-[:HAS_EPISODE]->(m)
RETURN m.uuid AS anchor, s.uuid AS saga, s.name AS session, {_msg_projection("m")} AS msg"""
#: Neighbours along NEXT_EPISODE: every node on the path belongs to the anchor's saga (an episode with no saga, as
#: Graphiti's default ``add_episode`` writes, walks the group's NEXT_EPISODE chain) and every edge to the group, so
#: a window never leaves its saga. ``%(pat)s`` is the hop pattern (direction and a literal bound).
_WINDOW_HOPS = """
MATCH (m:Episodic {group_id: $g}) WHERE m.uuid IN $uuids
OPTIONAL MATCH (s:Saga {group_id: $g})-[:HAS_EPISODE]->(m)
MATCH p = (m)%(pat)s(b:Episodic {group_id: $g})
WHERE all(n IN nodes(p) WHERE n.group_id = $g AND (s IS NULL OR (s)-[:HAS_EPISODE]->(n)))
  AND all(r IN relationships(p) WHERE r.group_id = $g)
RETURN m.uuid AS anchor, length(p) AS d, %(msg)s AS msg"""
ENTITY_SHEET_CYPHER = """
MATCH (n:Entity {group_id: $g}) WHERE n.uuid IN $uuids AND NOT n:Event
RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary,
       n.summary_line_ids AS line_ids, n.summary_line_texts AS line_texts,
       n.summary_identity_line_ids AS identity_line_ids"""
#: Joined people (``identity_aliases``: the surfaces joined to one Person); matched against the query in Python.
IDENTITY_ALIASES_CYPHER = """
MATCH (n:Entity {group_id: $g}) WHERE n.identity_aliases IS NOT NULL AND NOT n:Event
RETURN n.uuid AS uuid, n.name AS name, n.identity_aliases AS identity_aliases"""
#: ``SAME_AS`` neighbourhood of the candidate entities: undirected, at most 3 hops, every node and edge in the group,
#: one query per answer. One row per distinct (start, target); ``edges`` are the distinct ``[from, to]`` uuid
#: pairs along the paths between them (the direction is kept, the traversal is not), so a dense cluster cannot multiply rows.
SAME_AS_COMPONENT_CYPHER = """
MATCH p = (s:Entity {group_id: $g})-[:SAME_AS*1..3]-(t:Entity {group_id: $g})
WHERE s.uuid IN $uuids AND NOT s:Event AND NOT t:Event
  AND all(r IN relationships(p) WHERE r.group_id = $g)
UNWIND relationships(p) AS r
WITH s, t, collect(DISTINCT [startNode(r).uuid, endNode(r).uuid]) AS edges
RETURN s.uuid AS start, s.name AS start_name, s.minted_order AS start_minted_order,
       t.uuid AS uuid, t.name AS name, t.minted_order AS minted_order, edges"""
TIMELINE_CYPHER = """
MATCH (e:Entity:Event {group_id: $g})-[r]-(n:Entity {group_id: $g})
WHERE n.uuid IN $uuids AND r.group_id = $g AND NOT type(r) IN $structural AND e <> n
WITH e, collect(DISTINCT n.uuid) AS entities
RETURN e.uuid AS uuid, e.name AS name, e.source_text AS source_text, toString(e.happened_at) AS happened_at,
       toString(e.told_at) AS told_at, e.time_source AS time_source, e.argm_tmp_text AS time_text, e.asserted AS asserted, e.modality AS modality,
       e.attributes AS attributes, entities"""
EVENT_DATES_CYPHER = """
MATCH (e:Entity:Event {group_id: $g}) WHERE e.uuid IN $uuids
RETURN e.uuid AS uuid, toString(e.happened_at) AS happened_at, e.time_source AS time_source,
       e.attributes AS attributes"""


#: The entity-sheet route (``entity_sheet_view``): a name's exact matches, one sheet row with its per-line sources,
#: and the message ids of those source Episodics.
SHEET_BY_NAME_CYPHER = """
MATCH (n:Entity {group_id: $g}) WHERE NOT n:Event AND toLower(n.name) = toLower($name)
RETURN n.uuid AS uuid ORDER BY uuid"""
SHEET_VIEW_CYPHER = """
MATCH (n:Entity {group_id: $g}) WHERE n.uuid IN $uuids AND NOT n:Event
RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, n.summary AS summary,
       n.summary_line_ids AS line_ids, n.summary_line_texts AS line_texts,
       n.summary_identity_line_ids AS identity_line_ids, n.summary_sources AS sources,
       n.identity_aliases AS identity_aliases"""
SOURCE_MESSAGE_IDS_CYPHER = """
MATCH (m:Episodic {group_id: $g}) WHERE m.uuid IN $uuids
RETURN m.uuid AS uuid, m.message_id AS message_id"""
SHEET_MAX_ENTITIES = 3
SHEET_TIMELINE_CAP = 20


def _resolved_date(row: Mapping[str, Any]) -> str | None:
    """``happened_at`` of an Event row only when its ``time_source`` resolved it (never an unresolved/told-default day)."""
    return ext(row, "happened_at") if ext(row, "time_source") in RESOLVED_TIME_SOURCES else None


@dataclass(frozen=True)
class SameAsComponent:
    """Entities joined by ``SAME_AS`` (within 3 hops of a candidate), read as one person. ``id`` is the canonical
    node's uuid: the node the ``SAME_AS`` edges point to (the speaker, when the component has one), else, among
    several, the earliest minted (a node with no ``minted_order`` is a bootstrap entity: run 0), then the uuid."""

    id: str
    name: str
    members: tuple[str, ...]  # canonical first, then the rest by uuid
    also: tuple[str, ...]  # the other members' names, sorted, without repeats or the canonical name

    @property
    def display(self) -> str:
        return f"{self.name} (also: {', '.join(self.also)})" if self.also else self.name


def _minted_key(order: Any) -> tuple[int, int]:
    try:
        run, n = order
        return int(run), int(n)
    except (TypeError, ValueError):
        return (0, 0)


async def expand_same_as(driver, uuids: Sequence[str], group_id: str) -> dict[str, SameAsComponent]:
    """Candidate uuid -> its ``SameAsComponent``, for the candidates that have a ``SAME_AS`` neighbour (the rest are
    absent). One bounded query (``SAME_AS_COMPONENT_CYPHER``): undirected, at most 3 hops, scoped to the group and to
    the candidates. Candidates whose neighbourhoods overlap share one component."""
    starts = list(dict.fromkeys(uuids))
    if not starts:
        return {}
    rows = await _rows(driver, SAME_AS_COMPONENT_CYPHER, g=group_id, uuids=starts)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    info: dict[str, tuple[Any, Any]] = {}
    out_edges: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        parent[find(r["uuid"])] = find(r["start"])
        info[r["start"]] = (r.get("start_name"), r.get("start_minted_order"))
        info[r["uuid"]] = (r.get("name"), r.get("minted_order"))
        for a, b in r.get("edges") or ():
            out_edges[a].add(b)
            find(a), find(b)
            parent[find(a)] = find(r["start"])
            parent[find(b)] = find(r["start"])
    groups: dict[str, set[str]] = defaultdict(set)
    for u in list(parent):
        groups[find(u)].add(u)
    comps: dict[str, SameAsComponent] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        sinks = [m for m in members if not (out_edges.get(m, set()) & members)] or list(members)
        canon = min(sinks, key=lambda m: (_minted_key(info.get(m, (None, None))[1]), m))
        names = {m: str(info.get(m, (None, None))[0] or "") for m in members}
        also = tuple(sorted({n for m, n in names.items() if m != canon and n and n != names[canon]}))
        comp = SameAsComponent(canon, names[canon], (canon, *sorted(members - {canon})), also)
        comps.update({m: comp for m in members})
    return {u: comps[u] for u in starts if u in comps}


async def fetch_windows(
    driver, message_uuids: Sequence[str], group_id: str, before: int, after: int
) -> dict[str, dict[str, Any]]:
    """Anchor message uuid -> ``{"saga", "session", "messages"}``: the anchor and up to ``before`` / ``after``
    neighbours along ``NEXT_EPISODE``, oldest first, all inside the anchor's saga and group. Unknown anchors are absent.
    """
    uuids = list(dict.fromkeys(message_uuids))
    if not uuids:
        return {}
    anchors = {r["anchor"]: r for r in await _rows(driver, WINDOW_ANCHORS_CYPHER, g=group_id, uuids=uuids)}
    side: dict[str, list[tuple[int, int, dict[str, Any]]]] = {a: [] for a in anchors}
    for sign, bound, pat in ((-1, before, "<-[:NEXT_EPISODE*1..%d]-"), (1, after, "-[:NEXT_EPISODE*1..%d]->")):
        if int(bound) < 1:
            continue
        cypher = _WINDOW_HOPS % {"pat": pat % int(bound), "msg": _msg_projection("b")}
        for r in await _rows(driver, cypher, g=group_id, uuids=uuids):
            side[r["anchor"]].append((sign, r["d"], r["msg"]))
    out: dict[str, dict[str, Any]] = {}
    for anchor, row in anchors.items():
        ordered = sorted(side[anchor], key=lambda t: (t[0], -t[1] if t[0] < 0 else t[1]))
        before_msgs = [m for sg, _, m in ordered if sg < 0]
        after_msgs = [m for sg, _, m in ordered if sg > 0]
        msgs = [*before_msgs, row["msg"], *after_msgs]
        out[anchor] = {
            "saga": row["saga"],
            "session": row["session"],
            "messages": [{**m, "session": row["session"], "saga": row["saga"], "support": []} for m in msgs],
        }
    return out


async def fetch_entity_sheets(driver, entity_uuids: Sequence[str], group_id: str) -> list[dict[str, Any]]:
    """``[{uuid, name, summary, lines}]`` for the Entity nodes (Events excluded) in the order asked."""
    uuids = list(dict.fromkeys(entity_uuids))
    if not uuids:
        return []
    rows = {r["uuid"]: r for r in await _rows(driver, ENTITY_SHEET_CYPHER, g=group_id, uuids=uuids)}
    return [_with_lines(rows[u], group_id) for u in uuids if u in rows]


def _with_lines(row: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    """The sheet row with its stored line records (``lines``: None for a row built before 1.4.1) and its group."""
    return {**row, "lines": lines_of(row), "group_id": group_id}


async def fetch_timeline(driver, entity_uuids: Sequence[str], group_id: str, cap: int) -> list[dict[str, Any]]:
    """Dated Events linked to the entities by a role edge, earliest first (happened_at, else told_at), at most ``cap``.
    Rows carry ``time_source`` so the context never shows an unresolved time as a happened-on date."""
    uuids = list(dict.fromkeys(entity_uuids))
    if not uuids or cap < 1:
        return []
    rows = await _rows(driver, TIMELINE_CYPHER, g=group_id, uuids=uuids, structural=list(_STRUCTURAL))
    keyed = [(str(ext(r, "happened_at") or ext(r, "told_at") or ""), str(r["uuid"]), r) for r in rows]
    # Extension properties may live in ``attributes``, so the dated filter, order and cap are applied here.
    return [r for when, _, r in sorted(keyed, key=lambda t: t[:2]) if when][: int(cap)]


def _row_lines(row: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """``[(text, [source Episodic uuid])]`` in line order. A row without whole per-line data (built before 1.4.1) gives
    its summary's pieces; sources are paired only when they are one per line."""
    recs = lines_of(row)
    texts = [r.text for r in recs] if recs is not None else str(row.get("summary") or "").split("\n")
    sources = list(row.get("sources") or ())
    paired = len(sources) == len(texts)
    return [(t, [str(sources[i])] if paired and sources[i] else []) for i, t in enumerate(texts) if t.strip()]


async def entity_sheet_view(
    driver, group_id: str, *, name: str | None = None, uuid: str | None = None
) -> list[dict[str, Any]]:
    """The "who is X" view: at most ``SHEET_MAX_ENTITIES`` entities named ``name`` (``n.name``, then the joined people
    whose ``identity_aliases`` ``aliased_uuids`` finds in it) or the one with ``uuid``, each read as its ``SAME_AS``
    component (one entry, the canonical node first). Each entry is ``{uuid, name, kinds, summary, lines: [{text,
    message_ids}], aliases, identities, timeline}``. Reading only; no model call."""
    if uuid is not None:
        found = [uuid]
    else:
        exact = [r["uuid"] for r in await _rows(driver, SHEET_BY_NAME_CYPHER, g=group_id, name=name or "")]
        alias_rows = await _rows(driver, IDENTITY_ALIASES_CYPHER, g=group_id)
        found = list(dict.fromkeys([*exact, *aliased_uuids(alias_rows, name or "")]))
    found = found[:SHEET_MAX_ENTITIES]
    if not found:
        return []
    comps = await expand_same_as(driver, found, group_id)
    members = list(dict.fromkeys([*found, *(m for u in found if u in comps for m in comps[u].members)]))
    rows = {r["uuid"]: r for r in await _rows(driver, SHEET_VIEW_CYPHER, g=group_id, uuids=members)}
    ordered: list[tuple[str, tuple[str, ...], str | None]] = []  # (id, members, canonical name)
    seen: set[str] = set()
    for u in found:
        comp = comps.get(u)
        key, group = (comp.id, comp.members) if comp else (u, (u,))
        if key not in seen:
            seen.add(key)
            ordered.append((key, group, comp.name if comp else None))
    per_entity = [
        (key, group, cname, [(t, s) for m in group for t, s in _row_lines(rows.get(m) or {})])
        for key, group, cname in ordered
        if any(m in rows for m in group)
    ]
    src = list(dict.fromkeys(s for *_, lines in per_entity for _, ss in lines for s in ss))
    ids = (
        {r["uuid"]: r["message_id"] for r in await _rows(driver, SOURCE_MESSAGE_IDS_CYPHER, g=group_id, uuids=src)}
        if src
        else {}
    )
    timelines = {
        key: await fetch_timeline(driver, group, group_id, SHEET_TIMELINE_CAP) for key, group, _, _ in per_entity
    }
    out: list[dict[str, Any]] = []
    for key, group, cname, raw in per_entity:
        merged: dict[str, list[str]] = {}
        episodes: dict[str, list[str]] = {}
        for text, sources in raw:
            slot = merged.setdefault(text.strip(), [])
            slot.extend(str(ids[s]) for s in sources if ids.get(s) and str(ids[s]) not in slot)
            eps = episodes.setdefault(text.strip(), [])
            eps.extend(s for s in sources if s not in eps)
        canon = rows.get(key) or rows[next(m for m in group if m in rows)]
        display = cname or str(canon.get("name") or "")
        aliases = dict.fromkeys(
            a
            for m in group
            for a in [
                *(() if m == key else (str((rows.get(m) or {}).get("name") or ""),)),
                *((rows.get(m) or {}).get("identity_aliases") or ()),
            ]
            if a and a != display
        )
        kinds = dict.fromkeys(k for m in group for k in ((rows.get(m) or {}).get("labels") or ()) if k != "Entity")
        out.append(
            {
                "uuid": key,
                "name": display,
                "kinds": list(kinds),
                "summary": "\n".join(merged),
                "lines": [{"text": t, "message_ids": mids, "episode_ids": episodes[t]} for t, mids in merged.items()],
                "aliases": list(aliases),
                "identities": list(group),
                "timeline": [
                    {
                        "name": r.get("name"),
                        "happened_at": _resolved_date(r),
                        "told_at": ext(r, "told_at"),
                        "time_source": ext(r, "time_source"),
                        "asserted": ext(r, "asserted"),
                        "modality": ext(r, "modality"),
                    }
                    for r in timelines[key]
                ],
            }
        )
    return out


def _support_rows(
    result: SearchResult, hit: MessageHit, event_dates: Mapping[str, str | None], limit: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for uuid in hit.support:
        item = result.items.get(uuid) or {}
        if item.get("fact"):
            rows.append({"kind": "fact", **{k: item.get(k) for k in ("fact", "asserted", "modality")}})
        elif "p_by_role" in item:
            rows.append(
                {"kind": "event", "name": item.get("name"), "happened_at": event_dates.get(uuid)}
                | {k: item.get(k) for k in ("asserted", "modality")}
            )
        if len(rows) >= limit:
            break
    return rows


def aliased_uuids(rows: Sequence[Mapping[str, Any]], query: str) -> list[str]:
    """Uuids of the joined people whose name or any alias occurs in ``query`` by ``identity.alias_occurs`` (whole word;
    a common-noun name only capitalised and not sentence-initial), the predicate the projection's join uses."""
    out: list[str] = []
    for r in rows:
        surfaces = [r.get("name"), *(r.get("identity_aliases") or ())]
        if any(s and alias_occurs(str(s), query, is_common=is_common_noun) for s in surfaces):
            out.append(str(r["uuid"]))
    return out


def _top_entities(result: SearchResult, query: str, limit: int, alias_named: Sequence[str] = ()) -> list[str]:
    """Entity uuids from the node candidate lists: query-named first, then by fused score; ``limit`` of them.
    ``alias_named`` are joined people the query names by an alias: named too, even when outside the pool."""
    ql = query.lower()
    pool = dict.fromkeys(u for name in ("bm25_nodes", "semantic_nodes") for u in result.lists.get(name, ()))
    pool.update(dict.fromkeys(alias_named))
    aliased = set(alias_named)
    named = [
        u
        for u in pool
        if u in aliased
        or (
            str((result.items.get(u) or {}).get("name") or "").lower() in ql and (result.items.get(u) or {}).get("name")
        )
    ]
    rest = sorted((u for u in pool if u not in named), key=lambda u: (-result.trace.get(u, {}).get("score", 0.0), u))
    named.sort(key=lambda u: (-result.trace.get(u, {}).get("score", 0.0), u))
    return [*named, *rest][:limit]


def _ranked_entities(
    result: SearchResult, query: str, limit: int, alias_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """``_top_entities`` with the joined people the query names by an alias (``aliased_uuids``): one thread hop."""
    return _top_entities(result, query, limit, aliased_uuids(alias_rows, query))


def _person_sheet(
    uuid: str, rows: Mapping[str, Mapping[str, Any]], comps: Mapping[str, SameAsComponent]
) -> dict[str, Any] | None:
    """The sheet of a candidate: its own row, or, for a ``SAME_AS`` component, one sheet named once
    (``"Ana (also: Dee)"``) whose summary is the members' lines, the canonical node's first, de-duplicated."""
    comp = comps.get(uuid)
    if comp is None:
        return dict(rows[uuid]) if uuid in rows else None
    merged: dict[str, SheetLine] = {}  # de-duplicated by stripped text, the canonical member's lines first
    group = ""
    for m in comp.members:
        row = rows.get(m) or {}
        group = group or str(row.get("group_id") or "")
        recs = row.get("lines")
        if recs is None:  # built before 1.4.1: its summary pieces, none an identity line
            recs = [SheetLine("", ln, False) for ln in str(row.get("summary") or "").split("\n")]
        for rec in recs:
            if rec.text.strip():
                merged.setdefault(rec.text.strip(), rec)
    lines = [SheetLine(r.id, k, r.identity) for k, r in merged.items()]
    legacy = [(rows.get(m) or {}).get("lines") is None for m in comp.members]
    if any(legacy):  # any member without per-line data: one note per group, naming the backfill
        note_missing(group)
    every_legacy = all(legacy)
    return {
        "uuid": comp.id,
        "name": comp.display,
        "summary": "\n".join(r.text for r in lines),
        "lines": None if every_legacy else lines,
        "group_id": group,
        "members": comp.members,
    }


async def build_answer_context(
    driver,
    query: str,
    qvec: Sequence[float],
    *,
    group_id: str,
    shape: str,
    label: str | None,
    search_config: KgRecipeConfig,
    context_config: ContextConfig,
    search_fn: Callable[..., Awaitable[SearchResult]],
    weights: Mapping[str, float] | None = None,
    window_anchors: int = 20,
    query_model: str | None = None,
    relative: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """``search_fn`` then ``assemble_context``: windows around the top messages, the sheets of the top entities
    (named in the query first), and the timeline of those entities when the shape asks for one. ``search_fn`` is
    ``kg_recipe.search_messages_v2`` bound to an index (``functools.partial(search_messages_v2, index=idx)``).
    The Neo4j reads are awaited on the caller's loop; the synchronous work runs in worker threads, so the loop never
    waits on it: the entity ranking with its alias matching (WordNet, under its one lookup lock), and the assembly
    (token counts, relative times). ``relative``: the messages' relative-time texts computed ahead
    (``kg_context.relative_texts``, serve's at open), so the assembly parses no message.
    """
    extra = {"query_model": query_model} if query_model is not None else {}
    result = await search_fn(
        driver,
        group_id=group_id,
        query=query,
        qvec=qvec,
        shape=shape,
        label=label,
        weights=weights or {"role": 1.0, "label": 1.0},
        config=search_config,
        **extra,
    )
    hits = result.hits[:window_anchors]
    windows = await fetch_windows(
        driver, [h.message for h in hits], group_id, context_config.window_before, context_config.window_after
    )
    event_ids = [u for h in hits for u in h.support if "p_by_role" in (result.items.get(u) or {})]
    dates = {r["uuid"]: _resolved_date(r) for r in await _rows(driver, EVENT_DATES_CYPHER, g=group_id, uuids=event_ids)}
    for h in hits:
        win = windows.get(h.message)
        if win:
            for m in win["messages"]:
                if m["uuid"] == h.message:
                    m["support"] = _support_rows(result, h, dates, search_config.max_support)
    alias_rows = await _rows(driver, IDENTITY_ALIASES_CYPHER, g=group_id)
    top = await asyncio.to_thread(_ranked_entities, result, query, context_config.entity_sheet_k * 4, alias_rows)
    comps = await expand_same_as(driver, top, group_id)
    if comps:  # a component is one person: rank by its best candidate, read every member's sheet and timeline
        order = list(dict.fromkeys(comps[u].id if u in comps else u for u in top))
        members = list(dict.fromkeys(m for u in top if u in comps for m in comps[u].members))
        rows = {r["uuid"]: r for r in await fetch_entity_sheets(driver, [*top, *members], group_id)}
        by_id = {c.id: c for c in comps.values()}
        sheets = [_person_sheet(u, rows, by_id) for u in order if u in rows or u in by_id]
        sheets = [s for s in sheets if s][: context_config.entity_sheet_k]
        scope = [m for s in sheets for m in s.get("members", (s["uuid"],))]
    else:
        sheets = (await fetch_entity_sheets(driver, top, group_id))[: context_config.entity_sheet_k]
        scope = [e["uuid"] for e in sheets]
    timeline: list[dict[str, Any]] = []
    if shape in context_config.timeline_shapes:
        timeline = await fetch_timeline(driver, scope, group_id, context_config.timeline_cap)
    return await asyncio.to_thread(
        assemble_context,
        hits,
        windows=windows,
        entities=sheets,
        timeline=timeline,
        shape=shape,
        config=context_config,
        relative=relative,
    )


__all__ = [
    "SameAsComponent",
    "build_answer_context",
    "expand_same_as",
    "fetch_entity_sheets",
    "fetch_timeline",
    "fetch_windows",
]
