"""Writer: contract rows -> stack Neo4j. Embeds, MERGEs by {uuid, group_id}, never
deletes. A uuid is a global identity: a batch with a row uuid another group holds is refused (``ForeignUuids``).

Append rule: immutable base fields are set ON CREATE only; on match only EVOLVING base fields change (and only
to a non-empty value), ``episodes``/``aliases``/``event_uuids``/``event_keys`` grow as set unions, extension
properties are set except the create-only ones (``CREATE_ONLY``: an entity's ``minted_order``, a saga's
``started_at``, a row's times). An ``:Episodic`` (one message) is immutable once written: nothing is set on match.

Summary lines (kernel 1.2): a forward append sends a stored entity's NEW summary lines as ``summary_append`` /
``summary_append_sources`` / ``summary_append_line_ids`` (``SUMMARY_APPEND``, parallel lists) with
``summary_append_identity_line_ids`` (1.4.1: the identity lines among them). They never become node properties: the
writer strips them, appends the lines whose line id the node does not hold yet to its stored ``summary`` and grows
``summary_sources`` / ``summary_line_ids`` with them, and ``summary_line_texts`` / ``summary_identity_line_ids`` when
the node holds that per-line data (``append_summary``), so a replay appends nothing twice. In a forward append
(``write_rows(append=True)``) a stored edge takes its union fields only (``edge_match_update``): a restated fact
never moves the ``key_text`` an older message's key was built from. Every
read is a typed lookup of the batch's uuids. Nothing is ever deleted.
Embeddings follow the Graphiti graph shape: ``name_embedding`` on every ``:Entity``, ``fact_embedding`` on every fact
edge; the writer embeds nothing on ``:Saga`` or ``:Episodic`` (a Saga's ``centroid_embedding`` comes with its row,
``pipeline.saga_vectors``) and nothing on MENTIONS/HAS_EPISODE/NEXT_EPISODE.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from memory_kernel.graph.g1_names import EVENTUALITIES, KIND_LABELS
from memory_kernel.graph.g1_projection import NOT_CANDIDATES
from memory_kernel.graph.kg_contract import (
    BASE_RELATES_TO,
    EVOLVING,
    NODE_BASE,
    NODE_KINDS,
    STRUCTURAL_EDGES,
    line_data,
    validate_rows,
)

Embed = Callable[[list[str]], list[list[float]]]

BATCH = 1000
EMBED_BATCH = 128
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: List properties that only ever grow (set union on match).
UNION_FIELDS = ("episodes", "aliases", "identity_aliases", "event_uuids", "event_keys")
#: Extension properties set ON CREATE only (frozen once written): an entity's mint order (``[run, order]``, written
#: at mint by a forward append), a saga's start, and a row's times (a fact edge a later session states again keeps
#: the times it was first written with, as ``valid_at`` / ``reference_time`` already do; a full build folds the
#: earliest into the row itself).
#: 2.0 decision fields: a node's or edge's eventuality, its source and per-class probabilities, its mint kind and
#: legacy flag, and the run, chat, session and order that wrote it. Frozen at create: a later run never rewrites a
#: decision it was not part of.
DECISION_FIELDS = frozenset(
    {
        "eventuality",
        "eventuality_source",
        *(f"eventuality_p_{e}" for e in EVENTUALITIES),
        "mint_kind",
        "ground_legacy",
        "run",
        "chat",
        "session",
        "order",
    }
)
CREATE_ONLY = frozenset(
    {"minted_order", "started_at", "told_at", "happened_at", "happened_at_grain", "time_source"}
) | (DECISION_FIELDS)
#: Kinds whose on-match update is empty, as ``Episodic``'s is: a roleset or route, once written, is kept as it was
#: first decided.
CREATE_ONLY_KINDS = frozenset({"Roleset", "Route"})
#: A stored entity's new summary lines in a forward append's row (parallel lists): consumed by the writer
#: (``append_summary``), never stored.
SUMMARY_APPEND = (
    "summary_append",
    "summary_append_sources",
    "summary_append_line_ids",
    "summary_append_identity_line_ids",
)
#: The stored summary fields ``append_summary`` reads (``_stored_summaries``).
_SUMMARY_FIELDS = ("summary", "summary_sources", "summary_line_ids", "summary_line_texts", "summary_identity_line_ids")
#: Pinned embedder for message and fact search keys (``VoyageEmbedder`` defaults).
KEY_EMBED_MODEL = "voyage-3"
KEY_EMBED_DIM = 1024

#: The index set a Graphiti-compatible graph expects, plus a uuid index on each of the kernel's own edge types (the
#: writer looks every edge up by type and uuid: ``_foreign``, ``_already_embedded``, ``read_rows``); no vector index
#: DDL is issued. The names, labels, relationship types, properties and fulltext field lists are schema facts of that
#: graph shape, so they never change. One row per index, in the order the statements are issued and reported: (kind,
#: name, element, label or type, properties, variable), where kind is "range" (one property) or "fulltext" (a field
#: list), element is "node" or "rel", and variable is the pattern variable the statement uses.
INDEX_TABLE: tuple[tuple[str, str, str, str, tuple[str, ...], str], ...] = (
    ("range", "entity_uuid", "node", "Entity", ("uuid",), "n"),
    ("range", "episode_uuid", "node", "Episodic", ("uuid",), "n"),
    ("range", "community_uuid", "node", "Community", ("uuid",), "n"),
    ("range", "saga_uuid", "node", "Saga", ("uuid",), "n"),
    ("range", "relation_uuid", "rel", "RELATES_TO", ("uuid",), "e"),
    ("range", "mention_uuid", "rel", "MENTIONS", ("uuid",), "e"),
    ("range", "has_member_uuid", "rel", "HAS_MEMBER", ("uuid",), "e"),
    ("range", "has_episode_uuid", "rel", "HAS_EPISODE", ("uuid",), "e"),
    ("range", "next_episode_uuid", "rel", "NEXT_EPISODE", ("uuid",), "e"),
    ("range", "next_saga_uuid", "rel", "NEXT_SAGA", ("uuid",), "e"),
    ("range", "answers_uuid", "rel", "ANSWERS", ("uuid",), "e"),
    ("range", "answers_via_uuid", "rel", "ANSWERS_VIA", ("uuid",), "e"),
    ("range", "derived_from_uuid", "rel", "DERIVED_FROM", ("uuid",), "e"),
    ("range", "invalidated_by_uuid", "rel", "INVALIDATED_BY", ("uuid",), "e"),
    ("range", "same_as_uuid", "rel", "SAME_AS", ("uuid",), "e"),
    ("range", "entity_group_id", "node", "Entity", ("group_id",), "n"),
    ("range", "episode_group_id", "node", "Episodic", ("group_id",), "n"),
    ("range", "community_group_id", "node", "Community", ("group_id",), "n"),
    ("range", "saga_group_id", "node", "Saga", ("group_id",), "n"),
    ("range", "relation_group_id", "rel", "RELATES_TO", ("group_id",), "e"),
    ("range", "mention_group_id", "rel", "MENTIONS", ("group_id",), "e"),
    ("range", "has_episode_group_id", "rel", "HAS_EPISODE", ("group_id",), "e"),
    ("range", "next_episode_group_id", "rel", "NEXT_EPISODE", ("group_id",), "e"),
    ("range", "name_entity_index", "node", "Entity", ("name",), "n"),
    ("range", "saga_name", "node", "Saga", ("name",), "n"),
    ("range", "created_at_entity_index", "node", "Entity", ("created_at",), "n"),
    ("range", "created_at_episodic_index", "node", "Episodic", ("created_at",), "n"),
    ("range", "valid_at_episodic_index", "node", "Episodic", ("valid_at",), "n"),
    ("range", "name_edge_index", "rel", "RELATES_TO", ("name",), "e"),
    ("range", "created_at_edge_index", "rel", "RELATES_TO", ("created_at",), "e"),
    ("range", "expired_at_edge_index", "rel", "RELATES_TO", ("expired_at",), "e"),
    ("range", "valid_at_edge_index", "rel", "RELATES_TO", ("valid_at",), "e"),
    ("range", "invalid_at_edge_index", "rel", "RELATES_TO", ("invalid_at",), "e"),
    ("range", "roleset_uuid", "node", "Roleset", ("uuid",), "n"),
    ("range", "route_uuid", "node", "Route", ("uuid",), "n"),
    ("range", "taught_phrase_uuid", "node", "TaughtPhrase", ("uuid",), "n"),
    ("range", "bonfire_meta_uuid", "node", "BonfireMeta", ("uuid",), "n"),
    ("range", "routes_to_uuid", "rel", "ROUTES_TO", ("uuid",), "e"),
    ("fulltext", "episode_content", "node", "Episodic", ("content", "source", "source_description", "group_id"), "e"),
    ("fulltext", "node_name_and_summary", "node", "Entity", ("name", "summary", "group_id"), "n"),
    ("fulltext", "community_name", "node", "Community", ("name", "group_id"), "n"),
    ("fulltext", "edge_name_and_fact", "rel", "RELATES_TO", ("name", "fact", "group_id"), "e"),
)


def index_statement(kind: str, name: str, element: str, label: str, props: tuple[str, ...], var: str) -> str:
    """The ``CREATE ... IF NOT EXISTS`` statement of one ``INDEX_TABLE`` row (never a DROP)."""
    pattern = f"({var}:{label})" if element == "node" else f"()-[{var}:{label}]-()"
    if kind == "range":
        (prop,) = props
        return f"CREATE INDEX {name} IF NOT EXISTS FOR {pattern} ON ({var}.{prop})"
    fields = ", ".join(f"{var}.{prop}" for prop in props)
    return f"CREATE FULLTEXT INDEX {name} IF NOT EXISTS\n        FOR {pattern} ON EACH [{fields}]"


RANGE_INDEX_DDL: tuple[str, ...] = tuple(index_statement(*row) for row in INDEX_TABLE if row[0] == "range")
FULLTEXT_INDEX_DDL: tuple[str, ...] = tuple(index_statement(*row) for row in INDEX_TABLE if row[0] == "fulltext")
#: The table lists every range index before the fulltext ones, so this is also the order of the statements above.
INDEX_NAMES: tuple[str, ...] = tuple(row[1] for row in INDEX_TABLE)


async def _index_names(driver) -> set[str]:
    async with driver.session() as s:
        result = await s.run("SHOW INDEXES YIELD name RETURN name")
        return {r["name"] async for r in result}


async def ensure_indexes(driver) -> dict[str, list[str]]:
    """Issue the ``INDEX_TABLE`` statements (IF NOT EXISTS, never DROP); report which names it created vs found."""
    before = await _index_names(driver)
    async with driver.session() as s:
        for ddl in (*RANGE_INDEX_DDL, *FULLTEXT_INDEX_DDL):
            await (await s.run(ddl)).consume()
    after = await _index_names(driver)
    return {
        "created": [n for n in INDEX_NAMES if n not in before and n in after],
        "already_present": [n for n in INDEX_NAMES if n in before],
        "missing_after": [n for n in INDEX_NAMES if n not in after],
    }


#: Uniqueness per bonfire (``group_id``) for the 2.0 type nodes; each constraint carries its own backing index.
CONSTRAINT_DDL: tuple[str, ...] = (
    "CREATE CONSTRAINT roleset_group_roleset IF NOT EXISTS FOR (n:Roleset) REQUIRE (n.group_id, n.roleset) IS UNIQUE",
    "CREATE CONSTRAINT route_group_key IF NOT EXISTS FOR (n:Route) REQUIRE (n.group_id, n.key) IS UNIQUE",
    "CREATE CONSTRAINT taught_phrase_group_id IF NOT EXISTS FOR (n:TaughtPhrase) REQUIRE (n.group_id, n.id) IS UNIQUE",
    "CREATE CONSTRAINT bonfire_meta_group_id IF NOT EXISTS FOR (n:BonfireMeta) REQUIRE n.group_id IS UNIQUE",
)
#: The name each ``CONSTRAINT_DDL`` statement creates (the third word of the statement).
CONSTRAINT_NAMES: tuple[str, ...] = tuple(ddl.split()[2] for ddl in CONSTRAINT_DDL)


async def _constraint_names(driver) -> set[str]:
    async with driver.session() as s:
        result = await s.run("SHOW CONSTRAINTS YIELD name RETURN name")
        return {r["name"] async for r in result}


async def ensure_constraints(driver) -> dict[str, list[str]]:
    """Issue the ``CONSTRAINT_DDL`` statements (IF NOT EXISTS, never DROP); report which names it created vs found."""
    before = await _constraint_names(driver)
    async with driver.session() as s:
        for ddl in CONSTRAINT_DDL:
            await (await s.run(ddl)).consume()
    after = await _constraint_names(driver)
    return {
        "created": [n for n in CONSTRAINT_NAMES if n not in before and n in after],
        "already_present": [n for n in CONSTRAINT_NAMES if n in before],
        "missing_after": [n for n in CONSTRAINT_NAMES if n not in after],
    }


def _label_clause(labels: Sequence[str]) -> str:
    for label in labels:
        if not _IDENT.match(label):
            raise ValueError(f"unsafe label {label!r}")
    return ":".join(labels)


def kind_label_clause(labels: Sequence[str], replace: bool) -> str:
    """What a node's MERGE removes after its labels are set: with ``replace`` (a ``referent_policy`` structure row's
    ``kinds_replace``: its kind was decided again over the stored evidence), every kind label (``KIND_LABELS``) the row
    does not carry, so a stale stored kind goes; else nothing (a phrase row: labels only ever added)."""
    stale = [label for label in KIND_LABELS if label not in labels] if replace else []
    return f" REMOVE n:{_label_clause(stale)}" if stale else ""


def _check_group(nodes: list[dict], edges: list[dict], group_id: str) -> None:
    if not group_id:
        raise ValueError("group_id required")
    for row in [*nodes, *edges]:
        got = row["props"].get("group_id")
        if got != group_id:
            raise ValueError(f"row {row['props'].get('uuid')} group_id {got!r} != writer group_id {group_id!r}")


def _clean(props: dict) -> dict:
    """Neo4j drops null properties; omit them so a re-write never nulls an embedding already stored."""
    return {k: v for k, v in props.items() if v is not None}


async def _already_embedded(driver, group_id: str, entities: list[str], facts: list[tuple[str, str]]):
    """``(entity uuids, fact edge uuids)`` among this batch's (``entities``; ``facts`` as ``(type, uuid)``) that the
    group already stores with an embedding: typed lookups by uuid, never a read of the whole group."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for etype, uuid in facts:
        by_type[etype].append(uuid)
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    async with driver.session() as s:
        if entities:
            nodes = await s.run(
                "MATCH (n:Entity {group_id:$g}) WHERE n.uuid IN $ids AND n.name_embedding IS NOT NULL "
                "RETURN n.uuid AS u",
                g=group_id,
                ids=sorted(set(entities)),
            )
            node_ids = {r["u"] async for r in nodes}
        for etype, uuids in sorted(by_type.items()):
            edges = await s.run(
                f"MATCH ()-[e:{_label_clause([etype])} {{group_id:$g}}]->() WHERE e.uuid IN $ids "
                "AND e.fact_embedding IS NOT NULL RETURN e.uuid AS u",
                g=group_id,
                ids=sorted(set(uuids)),
            )
            edge_ids |= {r["u"] async for r in edges}
    return node_ids, edge_ids


async def _embed_missing(rows: list[dict], text_field: str, vec_field: str, done: set[str], embed: Embed) -> int:
    todo = [r for r in rows if r["props"]["uuid"] not in done and r["props"].get(vec_field) is None]
    texts = sorted({r["props"][text_field] for r in todo})
    vectors: dict[str, list[float]] = {}
    for i in range(0, len(texts), EMBED_BATCH):
        chunk = texts[i : i + EMBED_BATCH]
        out = await asyncio.to_thread(embed, chunk)
        if len(out) != len(chunk):
            raise ValueError("embed returned a different number of vectors than texts")
        vectors.update({t: [float(x) for x in v] for t, v in zip(chunk, out, strict=True)})
    for r in todo:
        r["props"][vec_field] = vectors[r["props"][text_field]]
    return len(texts)


def _primary(labels: Iterable[str]) -> str:
    labels = set(labels)
    return next((k for k in reversed(NODE_KINDS) if k != "Entity" and k in labels), "Entity")


def _split(props: dict, base: frozenset) -> tuple[dict, dict]:
    """(on-create props, on-match props). On match: evolving base fields with a non-empty value + extensions."""
    update = {}
    for key, value in props.items():
        if key in UNION_FIELDS or key in CREATE_ONLY or value is None:
            continue
        if key in base and (key not in EVOLVING or key == "name" or value in ("", [])):
            continue
        update[key] = value
    return props, update


def strip_appends(props: dict) -> tuple[list[str], list[str], list[str], list[str]] | None:
    """Pop a row's ``SUMMARY_APPEND`` lists (line, source Episodic, line id; the identity line ids among them), None
    when it carries none."""
    if not any(k in props for k in SUMMARY_APPEND):
        return None
    lines, sources, ids, identity = (list(props.pop(k, None) or []) for k in SUMMARY_APPEND)
    if not len(lines) == len(sources) == len(ids):
        raise ValueError(f"row {props.get('uuid')}: summary_append lists differ in length")
    if not set(identity) <= set(ids):
        raise ValueError(f"row {props.get('uuid')}: summary_append_identity_line_ids names a line it does not append")
    return lines, sources, ids, identity


def append_summary(
    stored: Mapping[str, Any],
    lines: Sequence[str],
    sources: Sequence[str],
    line_ids: Sequence[str],
    identity_ids: Sequence[str],
) -> dict[str, Any]:
    """The ``summary`` / ``summary_sources`` / ``summary_line_ids`` of a node holding ``stored`` (its stored values,
    absent for a new node) once the lines are appended: a line whose id the node already holds (or that came earlier
    in the same lists) is skipped. ``{}`` when nothing is new (a replay).

    The per-line data (``summary_line_texts``, ``summary_identity_line_ids``: those of ``identity_ids`` appended)
    grows with them only on a node that holds it whole, or holds no line yet (a new node; ``kg_contract.line_data``):
    a stored node built before 1.4.1, or desynced since its backfill (a 1.4.0 append grew its ids and not its texts;
    texts without identity ids), gets no line data, never a partial list, until ``mk bonfire backfill-lines`` fills
    it."""
    have = list(stored.get("summary_line_ids") or [])
    seen = set(have)
    new = []
    for line, source, lid in zip(lines, sources, line_ids, strict=True):
        if lid not in seen:
            seen.add(lid)
            new.append((line, source, lid))
    if not new:
        return {}
    summary = str(stored.get("summary") or "")
    text = "\n".join(line for line, _, _ in new)
    out = {
        "summary": f"{summary}\n{text}" if summary else text,
        "summary_sources": [*(stored.get("summary_sources") or []), *(source for _, source, _ in new)],
        "summary_line_ids": [*have, *(lid for _, _, lid in new)],
    }
    texts, held = stored.get("summary_line_texts"), stored.get("summary_identity_line_ids")
    state = line_data(summary, have, texts, held)
    if state == "broken":  # stored lines without whole per-line data
        return out
    if state == "none":  # no line yet: the data starts with these lines
        texts, held = [], []
    identity = set(identity_ids)
    out["summary_line_texts"] = [*(texts or []), *(line for line, _, _ in new)]
    out["summary_identity_line_ids"] = [*(held or []), *(lid for _, _, lid in new if lid in identity)]
    return out


async def _stored_summaries(driver, group_id: str, uuids: list[str]) -> dict[str, dict[str, Any]]:
    """The stored summary fields (``_SUMMARY_FIELDS``) of the group's ``:Entity`` nodes among ``uuids`` (a missing
    node is absent; a missing field is None)."""
    if not uuids:
        return {}
    async with driver.session() as s:
        result = await s.run(
            "MATCH (n:Entity {group_id:$g}) WHERE n.uuid IN $ids RETURN n.uuid AS u, "
            + ", ".join(f"n.{k} AS {k}" for k in _SUMMARY_FIELDS),
            g=group_id,
            ids=sorted(uuids),
        )
        return {r["u"]: {k: r[k] for k in _SUMMARY_FIELDS} async for r in result}


def _union(var: str, field: str) -> str:
    return (
        f"{var}.{field} = CASE WHEN r.props.{field} IS NULL THEN {var}.{field} ELSE coalesce({var}.{field}, []) + "
        f"[x IN r.props.{field} WHERE NOT x IN coalesce({var}.{field}, [])] END"
    )


class ForeignUuids(ValueError):
    """``write_rows`` refused its batch before writing anything: row uuids another group already holds (a uuid is a
    global identity, so a group cloned with its source's uuids collides with the source), or edge endpoints in neither
    the batch nor the writer's group. ``rows``: ``(uuid, owning group)``, the group None for a missing endpoint."""

    def __init__(self, group_id: str, rows: Iterable[tuple[str, str | None]]) -> None:
        self.rows: list[tuple[str, str | None]] = sorted(set(rows), key=lambda r: (r[0], r[1] or ""))
        held = [(u, g) for u, g in self.rows if g is not None]
        missing = [u for u, g in self.rows if g is None]
        parts = []
        if held:
            parts.append(
                f"{len(held)} row uuids are already held by group(s) {sorted({g for _, g in held})} (uuids are "
                "global: a copy that kept its source's uuids collides with it; re-key the copy or build it fresh): "
                f"{[u for u, _ in held[:5]]}"
            )
        if missing:
            parts.append(
                f"{len(missing)} edge endpoints are in neither the batch nor group {group_id!r}: {missing[:5]}"
            )
        super().__init__(f"write into group {group_id!r} refused: " + "; ".join(parts))


async def _foreign(driver, nodes: list[dict], edges: list[dict], group_id: str) -> list[tuple[str, str | None]]:
    """``(uuid, owning group)`` of each row uuid already stored under a DIFFERENT group_id, and ``(uuid, None)`` of each
    edge endpoint in neither this batch nor THIS group. Every lookup names a label or a relationship type, so the uuid
    indexes are used."""
    ids = {kind: [n["props"]["uuid"] for n in nodes if _primary(n["labels_neo4j"]) == kind] for kind in NODE_BASE}
    rel_ids: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        rel_ids[e["type"]].append(e["props"]["uuid"])
    found: list[tuple[str, str | None]] = []
    async with driver.session() as s:
        for label, uuids in ids.items():
            if not uuids:
                continue
            result = await s.run(
                f"MATCH (n:{label}) WHERE n.uuid IN $ids AND n.group_id <> $g RETURN n.uuid AS u, n.group_id AS g",
                ids=uuids,
                g=group_id,
            )
            found += [(r["u"], r["g"]) async for r in result]
        for etype, uuids in rel_ids.items():
            result = await s.run(
                f"MATCH ()-[e:{_label_clause([etype])}]->() WHERE e.uuid IN $ids AND e.group_id <> $g "
                "RETURN e.uuid AS u, e.group_id AS g",
                ids=uuids,
                g=group_id,
            )
            found += [(r["u"], r["g"]) async for r in result]
        # Every endpoint not written in this batch must already exist in THIS group, as a node of one of the kinds.
        missing = {u for e in edges for u in (e["source"], e["target"])} - {n["props"]["uuid"] for n in nodes}
        for label in NODE_KINDS:
            if not missing:
                break
            result = await s.run(
                f"MATCH (n:{label} {{group_id:$g}}) WHERE n.uuid IN $ids RETURN n.uuid AS u",
                ids=sorted(missing),
                g=group_id,
            )
            missing -= {r["u"] async for r in result}
        found += [(u, None) for u in sorted(missing)]
    return found


async def read_existing_entities(driver, group_id: str) -> list[dict]:
    """The group's resolvable :Entity nodes as {uuid, name, labels[, name_key]} for ``project(existing=...)``."""
    async with driver.session() as s:
        result = await s.run(
            "MATCH (n:Entity {group_id:$g}) WHERE NOT any(l IN labels(n) WHERE l IN $skip) "
            "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, n.name_key AS name_key ORDER BY uuid",
            g=group_id,
            skip=sorted(NOT_CANDIDATES),
        )
        rows = []
        async for r in result:
            row = {"uuid": r["uuid"], "name": r["name"], "labels": sorted(r["labels"])}
            if r["name_key"] is not None:
                row["name_key"] = r["name_key"]
            rows.append(row)
    return rows


async def read_existing_sagas(driver, group_id: str) -> list[dict]:
    """The group's ``:Saga`` nodes as {uuid, name, episodes, ingest_pass} for ``project(existing_sagas=...)``.

    ``ingest_pass`` is the stored ordered list of ingest passes (empty for a saga that has none).

    Graphiti ``_get_or_create_saga`` looks a saga up by (name, group_id) and takes the first record; here the order
    is deterministic (created_at, then uuid). ``episodes`` are the saga's Episodic uuids, most recent first, ordered
    as Graphiti's ``_saga_get_previous_episode_uuid`` orders them (valid_at DESC, created_at DESC; uuid breaks ties).
    """
    async with driver.session() as s:
        result = await s.run(
            "MATCH (s:Saga {group_id:$g}) OPTIONAL MATCH (s)-[:HAS_EPISODE]->(e:Episodic) "
            "WITH s, e ORDER BY e.valid_at DESC, e.created_at DESC, e.uuid "
            "WITH s, [u IN collect(e.uuid) WHERE u IS NOT NULL] AS episodes "
            "RETURN s.uuid AS uuid, s.name AS name, episodes, s.ingest_pass AS ingest_pass, "
            "s.started_at AS started_at, s.ended_at AS ended_at "
            "ORDER BY s.created_at, s.uuid",
            g=group_id,
        )
        return [
            {
                "uuid": r["uuid"],
                "name": r["name"],
                "episodes": list(r["episodes"]),
                "ingest_pass": list(r["ingest_pass"] or []),
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
            }
            async for r in result
        ]


def edge_match_update(update: dict, *, append: bool) -> dict:
    """What a MERGE sets on a STORED edge besides its union fields: ``_split``'s on-match map, or in append mode only
    the writer's own embedding. A later session that states a stored fact again (same uuid: endpoints, name,
    normalized fact, force) grows its ``episodes`` / ``event_keys`` and changes nothing else, so the ``key_text``,
    ``plan`` or role an older message's search key was built from never moves."""
    return {k: v for k, v in update.items() if k == "fact_embedding"} if append else update


async def write_rows(driver, rows: dict, *, group_id: str, embed: Embed, append: bool = False) -> dict:
    """Validate, refuse foreign uuids, embed what is not yet embedded, MERGE nodes then edges by {uuid, group_id}.
    ``append`` (a forward append): a stored edge takes its union fields only (``edge_match_update``). A node row's
    ``kinds_replace`` (structure) also removes the kind labels it does not carry (``kind_label_clause``)."""
    nodes: list[dict[str, Any]] = [
        {"labels_neo4j": list(n["labels_neo4j"]), "props": dict(n["props"])}
        | ({"kinds_replace": True} if n.get("kinds_replace") else {})
        for n in rows["nodes"]
    ]
    edges: list[dict[str, Any]] = [{**e, "props": dict(e["props"])} for e in rows["edges"]]
    appends = {n["props"]["uuid"]: got for n in nodes if (got := strip_appends(n["props"])) is not None}
    _check_group(nodes, edges, group_id)
    errors = validate_rows(nodes, edges, previous=rows.get("external_endpoints"))
    if errors:
        raise ValueError(f"rows violate the graph contract ({len(errors)} errors): {errors[:5]}")
    foreign = await _foreign(driver, nodes, edges, group_id)
    if foreign:
        raise ForeignUuids(group_id, foreign)

    stored_summary = await _stored_summaries(driver, group_id, sorted(appends))
    entities = [n for n in nodes if "Entity" in n["labels_neo4j"]]
    facts = [e for e in edges if e["type"] not in STRUCTURAL_EDGES]
    done_nodes, done_edges = await _already_embedded(
        driver,
        group_id,
        [n["props"]["uuid"] for n in entities],
        [(e["type"], e["props"]["uuid"]) for e in facts],
    )
    node_texts = await _embed_missing(entities, "name", "name_embedding", done_nodes, embed)
    edge_texts = await _embed_missing(facts, "fact", "fact_embedding", done_edges, embed)

    by_labels: dict[tuple[tuple[str, ...], bool], list[dict]] = defaultdict(list)
    # Edge endpoints are validated against the batch plus ``external_endpoints``, so every endpoint has a known kind.
    primary_of: dict[str, str] = {
        u: _primary(r["labels_neo4j"]) for u, r in (rows.get("external_endpoints") or {}).items() if "labels_neo4j" in r
    }
    primary_of.update({n["props"]["uuid"]: _primary(n["labels_neo4j"]) for n in nodes})
    for n in nodes:
        kind = _primary(n["labels_neo4j"])
        create, update = _split(_clean(n["props"]), NODE_BASE[kind])
        if kind == "Episodic" or kind in CREATE_ONLY_KINDS:
            update = {}
        uuid = n["props"]["uuid"]
        if uuid in appends:  # stored: append the new lines on match; new: they are its whole summary
            update |= append_summary(stored_summary.get(uuid, {}), *appends[uuid])
            create = create | append_summary({}, *appends[uuid])
        by_labels[(tuple(n["labels_neo4j"]), bool(n.get("kinds_replace")))].append(
            {"uuid": n["props"]["uuid"], "props": create, "update": update}
        )
    created = {"nodes": 0, "edges": 0}
    async with driver.session() as s:
        for (labels, replace), batch in by_labels.items():
            first, extra = _primary(labels), [label for label in labels if label != _primary(labels)]
            extra_clause = f" SET n:{_label_clause(extra)}" if extra else ""
            extra_clause += kind_label_clause(labels, replace)
            unions = (
                ""
                if first == "Episodic" or first in CREATE_ONLY_KINDS
                else ", " + ", ".join(_union("n", f) for f in UNION_FIELDS)
            )
            if first == "Saga":
                # Graphiti's add_episode sets first_episode_uuid while it is None; a saga stored by a whole-map
                # ``SET n = {...}`` with a null first episode has the property absent.
                unions += ", n.first_episode_uuid = coalesce(n.first_episode_uuid, r.props.first_episode_uuid)"
            query = (
                f"UNWIND $rows AS r MERGE (n:{_label_clause([first])} {{uuid: r.uuid, group_id: $g}}) "
                "ON CREATE SET n += r.props, n._new = true "
                f"ON MATCH SET n += r.update{unions}{extra_clause} "
                "WITH n, coalesce(n._new, false) AS new REMOVE n._new RETURN sum(CASE WHEN new THEN 1 ELSE 0 END) AS c"
            )
            for i in range(0, len(batch), BATCH):
                result = await s.run(query, rows=batch[i : i + BATCH], g=group_id)
                created["nodes"] += (await result.single())["c"]

        by_kind: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for e in edges:
            structural = STRUCTURAL_EDGES.get(str(e["type"]))
            base = structural[0] if structural else BASE_RELATES_TO
            create, update = _split(_clean(e["props"]), base)
            update = edge_match_update(update, append=append)
            key = (e["type"], primary_of[e["source"]], primary_of[e["target"]])
            by_kind[key].append(
                {"uuid": e["props"]["uuid"], "source": e["source"], "target": e["target"], "props": create}
                | {"update": update}
            )
        for (etype, slabel, tlabel), batch in by_kind.items():
            unions = ", ".join(_union("e", f) for f in UNION_FIELDS)
            query = (
                f"UNWIND $rows AS r MATCH (a:{_label_clause([slabel])} {{uuid: r.source, group_id: $g}}) "
                f"MATCH (b:{_label_clause([tlabel])} {{uuid: r.target, group_id: $g}}) "
                f"MERGE (a)-[e:{_label_clause([etype])} {{uuid: r.uuid, group_id: $g}}]->(b) "
                "ON CREATE SET e += r.props, e._new = true "
                f"ON MATCH SET e += r.update, {unions} "
                "WITH e, coalesce(e._new, false) AS new REMOVE e._new "
                "RETURN count(e) AS n, sum(CASE WHEN new THEN 1 ELSE 0 END) AS c"
            )
            for i in range(0, len(batch), BATCH):
                chunk = batch[i : i + BATCH]
                got = await (await s.run(query, rows=chunk, g=group_id)).single()
                if got["n"] != len(chunk):
                    raise ValueError(f"{len(chunk) - got['n']} {etype} edges lost an endpoint in group {group_id!r}")
                created["edges"] += got["c"]
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "nodes_created": created["nodes"],
        "edges_created": created["edges"],
        "node_texts_embedded": node_texts,
        "edge_texts_embedded": edge_texts,
    }


async def read_rows(driver, group_id: str, node_uuids: Iterable[str], edge_keys: Iterable[tuple[str, str]]):
    """The group's stored nodes among ``node_uuids`` and edges among ``edge_keys`` (``(type, uuid)``), in
    ``read_group``'s row shape (typed lookups, so the uuid indexes are used): what a forward append's gate compares its
    rows with before it writes."""
    node_ids = sorted(set(node_uuids))
    by_type: dict[str, set[str]] = defaultdict(set)
    for etype, uuid in edge_keys:
        by_type[etype].add(uuid)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    async with driver.session() as s:
        for label in NODE_KINDS if node_ids else ():
            result = await s.run(
                f"MATCH (n:{label} {{group_id:$g}}) WHERE n.uuid IN $ids RETURN labels(n) AS l, properties(n) AS p",
                g=group_id,
                ids=node_ids,
            )
            async for r in result:
                nodes.setdefault(r["p"]["uuid"], {"labels_neo4j": r["l"], "props": r["p"]})
        for etype, uuids in sorted(by_type.items()):
            result = await s.run(
                f"MATCH (a)-[r:{_label_clause([etype])} {{group_id:$g}}]->(b) WHERE r.uuid IN $ids "
                "RETURN type(r) AS t, a.uuid AS s, b.uuid AS d, properties(r) AS p",
                g=group_id,
                ids=sorted(uuids),
            )
            edges += [{"type": r["t"], "source": r["s"], "target": r["d"], "props": r["p"]} async for r in result]
    return list(nodes.values()), edges


async def read_group(driver, group_id: str) -> tuple[list[dict], list[dict]]:
    """Every node of a node kind (``NODE_KINDS``: one read per label, so the group_id indexes are used; a node of two
    kinds is read once) and every edge of a group, in the stored row shape (labels(n), properties(n))."""
    nodes: dict[str, dict] = {}
    async with driver.session() as s:
        for label in NODE_KINDS:
            result = await s.run(
                f"MATCH (n:{label} {{group_id:$g}}) RETURN labels(n) AS l, properties(n) AS p", g=group_id
            )
            async for r in result:
                nodes.setdefault(r["p"]["uuid"], {"labels_neo4j": r["l"], "props": r["p"]})
        result = await s.run(
            "MATCH (a)-[r {group_id:$g}]->(b) RETURN type(r) AS t, a.uuid AS s, b.uuid AS d, properties(r) AS p",
            g=group_id,
        )
        edges = [{"type": r["t"], "source": r["s"], "target": r["d"], "props": r["p"]} async for r in result]
    return list(nodes.values()), edges


class VoyageEmbedder:
    """Voyage embeddings over HTTP with retry and token counting; the key is held privately and never logged.

    Satisfies the embedder protocol the retrieval index expects (async ``create`` / ``create_batch``); the synchronous ``__call__``
    and ``embed`` take a list of texts. ``input_type`` is "document" for keys and "query" for questions; ``None``
    omits it from the request.
    """

    URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, *, model: str = KEY_EMBED_MODEL, input_type: str | None = "document") -> None:
        import httpx

        self.input_type = input_type
        self._client = httpx.Client(timeout=120.0, headers={"Authorization": f"Bearer {api_key}"})
        self.model = model
        self.name = f"voyage/{model}"
        self.tokens = 0
        self.requests = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"input": list(texts), "model": self.model}
        if self.input_type is not None:
            payload["input_type"] = self.input_type
        for attempt in range(6):
            resp = self._client.post(self.URL, json=payload)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            body = resp.json()
            self.requests += 1
            self.tokens += int(body["usage"]["total_tokens"])
            data = sorted(body["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        raise RuntimeError(f"voyage request failed after retries (status {resp.status_code})")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self(list(texts))

    async def create(self, input_data: str) -> list[float]:
        return (await self.create_batch([input_data]))[0]

    async def create_batch(self, input_data_list: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self, list(input_data_list))
