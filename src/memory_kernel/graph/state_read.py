"""Kernel 2.0: a bonfire's chain read from its graph, in place of the 1.8 folder chain.

``read_state_rows`` reads the group with a fixed set of batched queries (one per row kind, whatever the group's size)
into ``StateRows``. ``state_view`` gives the same ``StateRows`` from ``read_group``-shaped rows (pure). ``chain_from_rows``
folds them into the 1.8 ``Chain``, equal to what 1.8 folded from its folder:

- **entities:** every non-Event ``Entity``, with ``key`` = ``name_key``, ``kind`` = ``mint_kind``, ``display`` =
  ``name``, ``(run, order)`` = ``minted_order``, and no ``surfaces`` (2.0 drops them: nothing reads them).
- **routes** from ``Route`` -> ``ROUTES_TO``, **same_as** from ``SAME_AS`` (in the 1.8 chain order),
  **rolesets** from ``Roleset`` (``first_run`` is the run).
- **chats:** each chat's Episodic in ``order``. Its last message's Saga gives ``last_episode`` / ``last_saga``, its
  Sagas are counted for ``episodes``, its ``boundary`` is ``closed_boundary(n)`` and its ``tail`` is its last ``TAIL``
  select rows (``source_row_json``). The roster follows 1.8: at run 0, each message's speaker, addressees and the
  ``person`` entities it names (``from_bootstrap``); later, speakers and addressees only (``advance``).
- **ground** and **evidence** are replayed run by run, as 1.8 folded them: ``fold_ground`` with the run's mentions
  (sender, addressee and ``MENTIONS`` targets), its select rows and the agent handles known by then
  (``BonfireMeta.agent_handle_runs``), then ``fold_evidence`` (append-only: a superseded last mention stays).
- **kinds** (``referent_policy`` structure, read from ``build_config_json``): ``kind_row`` of each entity's
  ``kind_*`` properties; None under phrase.
- **message_ids**, **messages**, **episodes** and **explicit_since** come from the Episodic and Saga rows;
  **sessions**, **agent_handles** and **run** (``committed_run``) from ``BonfireMeta``."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from memory_kernel.graph.g1_nlp import normalize_exact
from memory_kernel.graph.g1_projection import message_order
from memory_kernel.graph.referents import fold_ground
from memory_kernel.graph.state import kind_row
from memory_kernel.pipeline.chain import (
    TAIL,
    Chain,
    ChainError,
    ChatState,
    Minted,
    SameAs,
    closed_boundary,
    fold_evidence,
)

KIND = "kind_"
EPISODIC_FIELDS = ("uuid", "message_id", "order", "chat", "session", "run", "source_row_json", "sender", "addressee")
SAME_AS_FIELDS = ("from_uuid", "to_uuid", "message_id", "holder", "name", "card", "run")


@dataclass(frozen=True)
class StateRows:
    """A bonfire's state as its graph holds it (one group), each list in a canonical order.

    - ``meta``: the ``BonfireMeta`` node's properties.
    - ``episodic``, by order: ``{uuid, message_id, order, chat, session, run, source_row_json, sender, addressee,
      mentions}`` (``mentions``: its ``MENTIONS`` targets that are non-Event entities).
    - ``sagas``, by uuid: ``{uuid, name, chat, run, episodes}`` (``episodes``: its ``HAS_EPISODE`` Episodic uuids).
    - ``entities``, by uuid, the non-Event ones: ``{uuid, name, name_key, mint_kind, minted_order, kinds}``
      (``kinds``: its ``kind_*`` properties).
    - ``same_as``: the ``SameAs`` fields, in the 1.8 chain's order.
    - ``routes``: ``Route.key`` -> its ``ROUTES_TO`` entity (the earliest run's).
    - ``rolesets``: roleset -> ``{gloss, examples, run, label, p_yes, compiled_sha256}`` (``run``: ``first_run``)."""

    meta: dict
    episodic: list[dict]
    sagas: list[dict]
    entities: list[dict]
    same_as: list[dict]
    routes: dict[str, str]
    rolesets: dict[str, dict]


# -- one row of each kind: shared by the pure view and the batched read, so both give equal rows ----------------------


def _episodic(props: Mapping[str, Any], mentions: Iterable[str]) -> dict[str, Any]:
    return {**{k: props.get(k) for k in EPISODIC_FIELDS}, "mentions": sorted(set(mentions))}


def _saga(props: Mapping[str, Any], episodes: Iterable[str]) -> dict[str, Any]:
    keep = ("uuid", "name", "chat", "run")
    return {**{k: props.get(k) for k in keep}, "episodes": sorted(set(episodes))}


def _entity(props: Mapping[str, Any]) -> dict[str, Any]:
    order = props.get("minted_order")
    return {
        **{k: props.get(k) for k in ("uuid", "name", "name_key", "mint_kind")},
        "minted_order": None if order is None else [int(x) for x in order],
        "kinds": {k: v for k, v in sorted(props.items()) if k.startswith(KIND)},
    }


def _same_as(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in SAME_AS_FIELDS}


def _same_as_order(row: Mapping[str, Any]) -> tuple:
    """The 1.8 chain's order of ``same_as.jsonl`` (what the next append reads)."""
    return (
        row["run"] if row["run"] is not None else -1,
        message_order(str(row["message_id"] or "")),
        *(str(row[k] or "") for k in ("from_uuid", "to_uuid", "holder", "name", "card")),
    )


def _routes(found: Iterable[tuple[str, str, Any]]) -> dict[str, str]:
    """``(key, target, run)`` -> key -> target; were a key routed twice, its earliest run wins (1.8: first wins)."""
    routes: dict[str, str] = {}
    for key, target, _run in sorted(found, key=lambda r: (r[0], r[2] if r[2] is not None else -1, r[1])):
        routes.setdefault(key, target)
    return dict(sorted(routes.items()))


def _roleset(props: Mapping[str, Any]) -> dict[str, Any]:
    p_yes = props.get("p_yes_json")
    return {
        "gloss": props.get("gloss"),
        "examples": list(props.get("examples") or ()),
        "run": props.get("first_run"),
        "label": props.get("label"),
        "p_yes": None if p_yes is None else json.loads(p_yes),
        "compiled_sha256": props.get("compiled_sha256"),
    }


def _rows(
    meta: Mapping[str, Any],
    episodic: list[dict],
    sagas: list[dict],
    entities: list[dict],
    same_as: list[dict],
    routes: dict[str, str],
    rolesets: Mapping[str, dict],
) -> StateRows:
    return StateRows(
        meta=dict(meta),
        episodic=sorted(episodic, key=lambda e: (e["order"] is None, e["order"] or 0, e["uuid"])),
        sagas=sorted(sagas, key=lambda s: s["uuid"]),
        entities=sorted(entities, key=lambda e: e["uuid"]),
        same_as=sorted(same_as, key=_same_as_order),
        routes=routes,
        rolesets=dict(sorted(rolesets.items())),
    )


# -- the pure view over read_group rows --------------------------------------------------------------------------------


def state_view(nodes: Sequence[Mapping], edges: Sequence[Mapping], meta: Mapping[str, Any]) -> StateRows:
    """``StateRows`` of one group's ``read_group`` rows (``nodes`` / ``edges``) and its ``BonfireMeta`` properties."""
    labels = {n["props"]["uuid"]: set(n["labels_neo4j"]) for n in nodes}

    def kind(uuid: str, label: str) -> bool:
        return label in labels.get(uuid, ()) and (label != "Entity" or "Event" not in labels[uuid])

    mentions: defaultdict[str, list[str]] = defaultdict(list)
    members: defaultdict[str, list[str]] = defaultdict(list)
    same_as, routed = [], []
    keys = {n["props"]["uuid"]: n["props"].get("key") for n in nodes if "Route" in n["labels_neo4j"]}
    for e in edges:
        source, target, props = e["source"], e["target"], e["props"]
        if e["type"] == "MENTIONS" and kind(source, "Episodic") and kind(target, "Entity"):
            mentions[source].append(target)
        elif e["type"] == "HAS_EPISODE" and kind(source, "Saga") and kind(target, "Episodic"):
            members[source].append(target)
        elif e["type"] == "SAME_AS" and kind(source, "Entity") and kind(target, "Entity"):
            same_as.append(
                _same_as({**props, "from_uuid": source, "to_uuid": target, "message_id": props.get("claim_message")})
            )
        elif e["type"] == "ROUTES_TO" and source in keys and kind(target, "Entity"):
            routed.append((keys[source], target, props.get("run")))
    by_kind: defaultdict[str, list[Mapping]] = defaultdict(list)
    for n in nodes:
        for label in ("Episodic", "Saga", "Entity", "Roleset"):
            if kind(n["props"]["uuid"], label):
                by_kind[label].append(n["props"])
    return _rows(
        meta,
        [_episodic(p, mentions[p["uuid"]]) for p in by_kind["Episodic"]],
        [_saga(p, members[p["uuid"]]) for p in by_kind["Saga"]],
        [_entity(p) for p in by_kind["Entity"]],
        same_as,
        _routes(routed),
        {p["roleset"]: _roleset(p) for p in by_kind["Roleset"]},
    )


# -- the 1.8 chain from the rows ---------------------------------------------------------------------------------------


def _need(row: Mapping[str, Any], fields: Sequence[str], what: str) -> None:
    if missing := [f for f in fields if row.get(f) is None]:
        raise ChainError(f"state read: {what} {row.get('uuid')} lacks {missing} (not a 2.0 bonfire's graph)")


def _policy(meta: Mapping[str, Any]) -> str:
    return str(json.loads(meta.get("build_config_json") or "{}").get("referent_policy") or "phrase")


def _handles_by_run(meta: Mapping[str, Any]) -> list[tuple[str, int]]:
    handles = list(meta.get("agent_handles") or ())
    runs = list(meta.get("agent_handle_runs") or ())
    if len(runs) != len(handles):
        raise ChainError(f"state read: BonfireMeta has {len(handles)} agent handles but {len(runs)} handle runs")
    return [(str(h), int(r)) for h, r in zip(handles, runs, strict=True)]


def chain_from_rows(rows: StateRows) -> Chain:
    """The 1.8 ``Chain`` the rows describe, every field filled (see the module docstring). ``ChainError`` when a row
    lacks a 2.0 property the fold needs."""
    meta = rows.meta
    _need(meta, ("sessions", "committed_run"), "BonfireMeta")
    for e in rows.episodic:
        _need(e, ("message_id", "order", "chat", "run", "source_row_json"), "Episodic")
    for m in rows.entities:
        _need(m, ("name", "name_key", "mint_kind", "minted_order"), "Entity")
    source = {e["message_id"]: json.loads(e["source_row_json"]) for e in rows.episodic}

    entities = {
        m["uuid"]: Minted(
            uuid=m["uuid"],
            key=m["name_key"],
            kind=m["mint_kind"],
            display=m["name"],
            surfaces=(),
            run=m["minted_order"][0],
            order=m["minted_order"][1],
        )
        for m in rows.entities
    }
    named = {
        e["message_id"]: [u for u in (e["sender"], e["addressee"]) if u in entities] + list(e["mentions"])
        for e in rows.episodic
    }
    persons = {u: m.display for u, m in entities.items() if m.kind == "person"}

    saga_of = {ep: s for s in rows.sagas for ep in s["episodes"]}
    by_chat: defaultdict[str, list[dict]] = defaultdict(list)
    for e in rows.episodic:
        by_chat[e["chat"]].append(e)
    chats: dict[str, ChatState] = {}
    for chat, msgs in sorted(by_chat.items()):
        last = msgs[-1]
        if last["uuid"] not in saga_of:
            raise ChainError(f"state read: chat {chat!r}: its last message {last['message_id']!r} is in no Saga")
        roster: dict[str, str] = {}
        for e in msgs:
            row = source[e["message_id"]]
            names = [row["speaker"], *(row.get("addressees") or ())]
            if e["run"] == 0:  # the bootstrap's roster also takes the Person entities each message names
                names += sorted(persons[u] for u in named[e["message_id"]] if u in persons)
            for name in names:
                roster.setdefault(normalize_exact(name), name)
        chats[chat] = ChatState(
            episodes=len({saga_of[e["uuid"]]["uuid"] for e in msgs if e["uuid"] in saga_of}),
            last_episode=saga_of[last["uuid"]]["name"],
            last_saga=saga_of[last["uuid"]]["uuid"],
            last_order=int(last["order"]),
            roster=list(roster.values()),
            tail=[source[e["message_id"]] for e in msgs[max(0, len(msgs) - TAIL) :]],
            boundary=closed_boundary(len(msgs)),
        )

    handles = _handles_by_run(meta)
    by_run: defaultdict[int, list[dict]] = defaultdict(list)
    for e in rows.episodic:
        by_run[int(e["run"])].append(e)
    ground: dict = {}
    evidence: dict[str, dict] = {}
    for run, msgs in sorted(by_run.items()):  # 1.8 folded each run once, in run order: the bootstrap, then appends
        run_rows = {e["message_id"]: source[e["message_id"]] for e in msgs}
        mentioned: defaultdict[str, list[str]] = defaultdict(list)
        for e in msgs:
            for u in named[e["message_id"]]:
                mentioned[u].append(e["message_id"])
        chat_of = {e["message_id"]: e["chat"] for e in msgs}
        agents = [h for h, joined in handles if joined <= run]
        ground = fold_ground(ground, mentioned, run_rows, chat_of.__getitem__, agents, legacy=run == 0)
        evidence = fold_evidence(evidence, ground, run_rows)

    kinds = None
    if _policy(meta) == "structure":
        kinds = {m["uuid"]: kind_row(m["kinds"]) for m in rows.entities if m["kinds"]}
    return Chain(
        entities=entities,
        routes=dict(rows.routes),
        same_as=[SameAs(**x) for x in rows.same_as],
        chats=chats,
        rolesets={
            rs: {"gloss": r["gloss"], "examples": list(r["examples"]), "run": r["run"]}
            for rs, r in rows.rolesets.items()
        },
        message_ids={e["message_id"]: int(e["run"]) for e in rows.episodic},
        messages=len(rows.episodic),
        episodes=len(rows.sagas),
        sessions=int(meta["sessions"]),
        explicit_since=min(
            (int(e["order"]) for e in rows.episodic if source[e["message_id"]].get("addressees")), default=None
        ),
        agent_handles=[h for h, _ in handles],
        run=int(meta["committed_run"]),
        ground=ground,
        evidence=evidence,
        kinds=kinds,
    )


# -- the batched read --------------------------------------------------------------------------------------------------

_EPISODIC = (
    "MATCH (e:Episodic {group_id: $g}) "
    "OPTIONAL MATCH (e)-[:MENTIONS]->(t:Entity {group_id: $g}) WHERE NOT t:Event "
    "RETURN e {" + ", ".join(f".`{f}`" for f in EPISODIC_FIELDS) + "} AS e, collect(DISTINCT t.uuid) AS mentions"
)
_SAGAS = (
    "MATCH (s:Saga {group_id: $g}) "
    "OPTIONAL MATCH (s)-[:HAS_EPISODE]->(e:Episodic {group_id: $g}) "
    "RETURN s {.uuid, .name, .chat, .run} AS s, collect(DISTINCT e.uuid) AS episodes"
)
_ENTITIES = (
    "MATCH (n:Entity {group_id: $g}) WHERE NOT n:Event "
    "RETURN n {.uuid, .name, .name_key, .mint_kind, .minted_order} AS n, "
    f"[k IN keys(n) WHERE k STARTS WITH '{KIND}' | [k, n[k]]] AS kinds"
)
_SAME_AS = (
    "MATCH (a:Entity {group_id: $g})-[r:SAME_AS {group_id: $g}]->(b:Entity {group_id: $g}) "
    "RETURN a.uuid AS from_uuid, b.uuid AS to_uuid, r.claim_message AS message_id, r.holder AS holder, "
    "r.name AS name, r.card AS card, r.run AS run"
)
_ROUTES = (
    "MATCH (r:Route {group_id: $g})-[x:ROUTES_TO {group_id: $g}]->(t:Entity {group_id: $g}) "
    "RETURN r.key AS key, t.uuid AS target, x.run AS run"
)
_ROLESETS = (
    "MATCH (r:Roleset {group_id: $g}) "
    "RETURN r {.roleset, .gloss, .examples, .first_run, .label, .p_yes_json, .compiled_sha256} AS r"
)
#: The one read of a bonfire's ``BonfireMeta`` (``read_meta``; the serving registry reads it through it too).
META = "MATCH (m:BonfireMeta {group_id: $g}) RETURN properties(m) AS p"


async def _records(session, query: str, group: str) -> list:
    result = await session.run(query, g=group)
    return [r async for r in result]


async def read_meta(driver, group: str) -> dict | None:
    """The group's ``BonfireMeta`` properties, or None when it has none (not a 2.0 bonfire, or one still seeding)."""
    async with driver.session() as s:
        found = await _records(s, META, group)
    return dict(found[0]["p"]) if found else None


async def read_state_rows(driver, group: str) -> StateRows:
    """``StateRows`` of ``group``, read with one query per row kind (a fixed number, whatever the group's size;
    ``driver`` may be a ``tx.TxScope``). ``ChainError`` when the group has no ``BonfireMeta``."""
    async with driver.session() as s:
        meta = await _records(s, META, group)
        if not meta:
            raise ChainError(f"state read: group {group!r} has no BonfireMeta (not a 2.0 bonfire)")
        episodic = [_episodic(r["e"], r["mentions"]) for r in await _records(s, _EPISODIC, group)]
        sagas = [_saga(r["s"], r["episodes"]) for r in await _records(s, _SAGAS, group)]
        entities = [_entity({**r["n"], **dict(r["kinds"])}) for r in await _records(s, _ENTITIES, group)]
        same_as = [_same_as(dict(r)) for r in await _records(s, _SAME_AS, group)]
        routes = _routes((r["key"], r["target"], r["run"]) for r in await _records(s, _ROUTES, group))
        rolesets = {r["r"]["roleset"]: _roleset(r["r"]) for r in await _records(s, _ROLESETS, group)}
    return _rows(meta[0]["p"], episodic, sagas, entities, same_as, routes, rolesets)


async def read_chain(driver, group: str) -> Chain:
    """The 1.8 ``Chain`` of ``group``, read from its graph (``read_state_rows`` then ``chain_from_rows``)."""
    return chain_from_rows(await read_state_rows(driver, group))
