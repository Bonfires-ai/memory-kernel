"""Graphiti-compatible graph contract on the STORED Neo4j shape.

Rows mirror what Neo4j returns: nodes ``{"labels_neo4j": [...], "props": properties(n)}`` and edges
``{"type", "source", "target", "props": properties(r)}``. Labels are Neo4j labels (no ``labels`` property) and
extension fields are flat top-level properties. Extensions add fields; they never remove or redefine base ones, and
an append only changes base fields Graphiti itself evolves.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


def ext(props: Mapping, key: str):
    """An extension property, flat or inside the JSON-string ``attributes`` property.

    Graphiti's bulk save moves extension fields into ``attributes``; every kernel
    reader of extension properties goes through here so it reads both shapes. Flat wins when both are present.
    """
    value = props.get(key)
    if value is not None:
        return value
    attrs = props.get("attributes")
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except ValueError:
            return None
    return attrs.get(key) if isinstance(attrs, Mapping) else None


def line_data(summary: object, ids: Sequence | None, texts: Sequence | None, identity: Sequence | None) -> str:
    """The state of an Entity's stored per-line data (1.4.1: ``summary_line_texts`` and ``summary_identity_line_ids``
    beside ``summary_line_ids``). ``"whole"``: the texts and the identity ids are both present and the texts are one
    per line id. ``"none"``: the node holds no line (no line id and a blank ``summary``). ``"broken"``: lines without
    whole line data, i.e. built before 1.4.1, or desynced since its backfill by a 1.4.0 append or a rollback (the ids
    grew and the texts did not; one list without the other). Readers read a broken node as legacy, the writer grows
    no line data on it, and ``mk bonfire backfill-lines`` repairs it. The one rule all three apply."""
    if texts is not None and identity is not None and len(texts) == len(ids or ()):
        return "whole"
    if not ids and not str(summary or "").strip():
        return "none"
    return "broken"


#: ``time_source`` values whose ``happened_at`` is a resolved date. ``unresolved`` / ``told_default`` are not: a reader
#: never presents those as "happened on".
RESOLVED_TIME_SOURCES = ("deictic", "explicit")

BASE_ENTITY = frozenset({"uuid", "name", "group_id", "created_at", "name_embedding", "summary"})
BASE_EPISODIC = frozenset(
    {"uuid", "name", "group_id", "created_at", "source", "source_description", "content", "valid_at", "entity_edges"}
)
BASE_RELATES_TO = frozenset(
    {
        "uuid",
        "group_id",
        "created_at",
        "name",
        "fact",
        "fact_embedding",
        "episodes",
        "valid_at",
        "invalid_at",
        "expired_at",
        "reference_time",
        "source_uuid",
        "target_uuid",
    }
)
BASE_MENTIONS = frozenset({"uuid", "group_id", "created_at"})
#: A Graphiti ``SagaNode`` as its save query stores it (``SET n = {uuid, name, group_id, created_at, summary,
#: first_episode_uuid, last_episode_uuid, last_summarized_at}``). The Graphiti base stays embedding-free; the kernel
#: adds extra Saga properties, among them its centroid (``pipeline.saga_vectors``: ``centroid_embedding``, the
#: normalised mean of its messages' voyage-3 vectors, ``centroid_model``, ``centroid_messages``), checked by
#: ``validate_node``. Graphiti's own wholesale ``SET n = {...}`` would drop them: the kernel writer only MERGEs.
BASE_SAGA = frozenset(
    {
        "uuid",
        "name",
        "group_id",
        "created_at",
        "summary",
        "first_episode_uuid",
        "last_episode_uuid",
        "last_summarized_at",
    }
)
#: ``HAS_EPISODE_EDGE_SAVE`` / ``NEXT_EPISODE_EDGE_SAVE`` (Graphiti's edge save queries): MERGE on
#: ``uuid``, SET ``group_id`` and ``created_at``; no source/target uuid properties are stored.
BASE_HAS_EPISODE = frozenset({"uuid", "group_id", "created_at"})
BASE_NEXT_EPISODE = frozenset({"uuid", "group_id", "created_at"})
#: Question / Plan / Purpose / RevisionPass: identity + display name only; their domain
#: fields (qid, text, wh, pass, ...) are flat extension properties.
BASE_CONTRACT_NODE = frozenset({"uuid", "name", "group_id", "created_at"})
#: 2.0 graph-native state: a per-bonfire roleset, route and taught phrase, and the bonfire's own meta node.
BASE_ROLESET = frozenset({"uuid", "group_id", "created_at", "roleset"})
BASE_ROUTE = frozenset({"uuid", "group_id", "created_at", "key"})
BASE_TAUGHT_PHRASE = frozenset({"uuid", "group_id", "created_at", "id"})
BASE_BONFIRE_META = frozenset({"uuid", "group_id", "created_at", "bonfire_id", "schema"})
#: ``Route -> Entity``: the route resolves to the entity it names.
BASE_ROUTES_TO = frozenset({"uuid", "group_id", "created_at"})
BASE_QUESTION = BASE_PLAN = BASE_PURPOSE = BASE_REVISION_PASS = BASE_CONTRACT_NODE
#: Coding edge ``Episodic -> Question``. ``observation`` = uuid of the Event or fact edge it codes, or of the entity
#: whose summary line it codes (C2: ``observation_kind`` event / fact_edge / summary_line; ``summary_line`` = the
#: line's ``summary_line_id``). ``p`` (the Jev2 role probability), ``filler`` and ``told_at`` are extensions that may
#: be None, and Neo4j drops a None property, so they are not base fields.
BASE_ANSWERS = frozenset({"uuid", "group_id", "created_at", "observation", "role", "plan", "pass"})
#: ``Plan -> Question``: the card's purpose row made visible.
BASE_ANSWERS_VIA = frozenset({"uuid", "group_id", "created_at", "role", "wh"})
BASE_DERIVED_FROM = frozenset({"uuid", "group_id", "created_at"})
BASE_INVALIDATED_BY = frozenset({"uuid", "group_id", "created_at"})
#: ``Entity -> Entity`` identity link (kernel 1.2): a node named by an identity claim points at the claim's
#: canonical node. ``claim_message`` (the claim's message id), ``holder``, ``name`` (the claimed name) and ``card`` (the
#: identity construction's id). Search treats a ``SAME_AS`` component as one person.
BASE_SAME_AS = frozenset({"uuid", "group_id", "created_at", "claim_message", "holder", "name", "card"})
#: Structural edge types: (base fields, allowed source kinds, allowed target kinds).
STRUCTURAL_EDGES: dict[str, tuple[frozenset, frozenset[str], frozenset[str]]] = {
    "MENTIONS": (BASE_MENTIONS, frozenset({"Episodic"}), frozenset({"Entity"})),
    "HAS_EPISODE": (BASE_HAS_EPISODE, frozenset({"Saga"}), frozenset({"Episodic"})),
    "NEXT_EPISODE": (BASE_NEXT_EPISODE, frozenset({"Episodic"}), frozenset({"Episodic"})),
    #: The saga chain, one edge per consecutive saga pair.
    "NEXT_SAGA": (BASE_NEXT_EPISODE, frozenset({"Saga"}), frozenset({"Saga"})),
    #: Codings and the derivation DAG.
    "ANSWERS": (BASE_ANSWERS, frozenset({"Episodic"}), frozenset({"Question"})),
    "ANSWERS_VIA": (BASE_ANSWERS_VIA, frozenset({"Plan"}), frozenset({"Question"})),
    "DERIVED_FROM": (BASE_DERIVED_FROM, frozenset({"Question"}), frozenset({"Purpose"})),
    "INVALIDATED_BY": (BASE_INVALIDATED_BY, frozenset({"Plan", "Question"}), frozenset({"RevisionPass"})),
    "SAME_AS": (BASE_SAME_AS, frozenset({"Entity"}), frozenset({"Entity"})),
    "ROUTES_TO": (BASE_ROUTES_TO, frozenset({"Route"}), frozenset({"Entity"})),
}
NODE_KINDS = (
    "Entity",
    "Episodic",
    "Saga",
    "Question",
    "Plan",
    "Purpose",
    "RevisionPass",
    "Roleset",
    "Route",
    "TaughtPhrase",
    "BonfireMeta",
)
NODE_BASE = {
    "Entity": BASE_ENTITY,
    "Episodic": BASE_EPISODIC,
    "Saga": BASE_SAGA,
    "Question": BASE_QUESTION,
    "Plan": BASE_PLAN,
    "Purpose": BASE_PURPOSE,
    "RevisionPass": BASE_REVISION_PASS,
    "Roleset": BASE_ROLESET,
    "Route": BASE_ROUTE,
    "TaughtPhrase": BASE_TAUGHT_PHRASE,
    "BonfireMeta": BASE_BONFIRE_META,
}
#: Base fields Graphiti itself updates on an existing row; everything else in the base is immutable once written.
#: Saga: ``summarize_saga`` rewrites summary + last_summarized_at; ``add_episode`` moves
#: last_episode_uuid on every append and sets first_episode_uuid only while it is None.
EVOLVING = frozenset(
    {
        "name",
        "summary",
        "name_embedding",
        "fact_embedding",
        "episodes",
        "invalid_at",
        "expired_at",
        "last_episode_uuid",
        "last_summarized_at",
    }
)
#: Base fields Neo4j omits from properties() when the value is None, so presence is not required.
NULLABLE = frozenset(
    {
        "name_embedding",
        "fact_embedding",
        "valid_at",
        "invalid_at",
        "expired_at",
        "reference_time",
        "first_episode_uuid",
        "last_episode_uuid",
        "last_summarized_at",
    }
)


#: Base RELATES_TO fields a READ tolerates as absent: Graphiti's ``add_episode`` writes edges through the bulk path,
#: which stores ``source_node_uuid``/``target_node_uuid`` and no
#: ``source_uuid``/``target_uuid``/``reference_time``. The endpoints are the relationship's own ends. A write needs them.
READ_OPTIONAL_RELATES_TO = frozenset({"source_uuid", "target_uuid", "reference_time"})


def _missing(uuid: str, props: dict, base: frozenset, optional: frozenset = frozenset()) -> list[str]:
    return [f"{uuid}: missing base field {field}" for field in sorted(base - NULLABLE - optional - set(props))]


def _is_embedding(value: object) -> bool:
    if value is None:
        return True
    return isinstance(value, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)


def validate_node(row: dict) -> list[str]:
    props = row["props"]
    uuid = str(props.get("uuid", "?"))
    kind = _kind(row)
    if kind is None:
        return [f"{uuid}: node must be exactly one of {', '.join(':' + k for k in NODE_KINDS)}"]
    errors = _missing(uuid, props, NODE_BASE[kind])
    if "name" in props and not isinstance(props["name"], str):
        errors.append(f"{uuid}: name must be str")
    if kind in ("Entity", "Saga") and "summary" in props and not isinstance(props["summary"], str):
        errors.append(f"{uuid}: summary must be str")
    if kind == "Entity" and "name_embedding" in props and not _is_embedding(props["name_embedding"]):
        errors.append(f"{uuid}: name_embedding must be a list of numbers or None")
    if kind == "Episodic" and "entity_edges" in props and not isinstance(props["entity_edges"], list):
        errors.append(f"{uuid}: entity_edges must be a list")
    if kind == "Episodic" and "search_key_embedding" in props and not _is_embedding(props["search_key_embedding"]):
        errors.append(f"{uuid}: search_key_embedding must be a list of numbers or None")
    if kind == "Episodic" and "message_embedding" in props and not _is_embedding(props["message_embedding"]):
        errors.append(f"{uuid}: message_embedding must be a list of numbers or None")
    if kind == "Saga" and "centroid_embedding" in props and not _is_embedding(props["centroid_embedding"]):
        errors.append(f"{uuid}: centroid_embedding must be a list of numbers or None")
    return errors


def validate_edge(row: dict, *, mode: str = "write") -> list[str]:
    """``mode="read"`` tolerates the endpoint uuid props Graphiti's bulk save omits; ``"write"`` requires them."""
    props = row["props"]
    uuid = str(props.get("uuid", "?"))
    if row["type"] in STRUCTURAL_EDGES:
        return _missing(uuid, props, STRUCTURAL_EDGES[row["type"]][0])
    errors = _missing(uuid, props, BASE_RELATES_TO, READ_OPTIONAL_RELATES_TO if mode == "read" else frozenset())
    if "fact" in props and not (isinstance(props["fact"], str) and props["fact"]):
        errors.append(f"{uuid}: fact must be a non-empty str")
    if "episodes" in props and not isinstance(props["episodes"], list):
        errors.append(f"{uuid}: episodes must be a list")
    if "fact_embedding" in props and not _is_embedding(props["fact_embedding"]):
        errors.append(f"{uuid}: fact_embedding must be a list of numbers or None")
    if "source_uuid" in props and props["source_uuid"] != row["source"]:
        errors.append(f"{uuid}: source_uuid does not match edge source")
    if "target_uuid" in props and props["target_uuid"] != row["target"]:
        errors.append(f"{uuid}: target_uuid does not match edge target")
    return errors


def _kind(row: dict | None) -> str | None:
    """The single base kind of a node row, or None when it has none or several."""
    if row is None or "labels_neo4j" not in row:
        return None
    found = [k for k in NODE_KINDS if k in set(row["labels_neo4j"])]
    return found[0] if len(found) == 1 else None


def _endpoint_errors(edge: dict, known: dict[str, dict]) -> list[str]:
    uuid = str(edge["props"].get("uuid", "?"))
    structural = STRUCTURAL_EDGES.get(edge["type"])
    want = structural[1:] if structural else (frozenset({"Entity"}), frozenset({"Entity"}))
    errors = []
    for side, expected in zip(("source", "target"), want, strict=True):
        row = known.get(edge[side])
        if row is None:
            errors.append(f"{uuid}: {side} {edge[side]} does not exist")
        elif _kind(row) not in expected:
            errors.append(f"{uuid}: {side} {edge[side]} must be {' or '.join(':' + k for k in sorted(expected))}")
    return errors


def _append_errors(row: dict, prior: dict) -> list[str]:
    props, before = row["props"], prior["props"]
    uuid = props["uuid"]
    if "labels_neo4j" in row:
        base = NODE_BASE.get(_kind(row) or "", frozenset())
    else:
        base = STRUCTURAL_EDGES[row["type"]][0] if row["type"] in STRUCTURAL_EDGES else BASE_RELATES_TO
    errors = []
    for field in sorted(base & set(before) & set(props)):
        if props[field] == before[field]:
            continue
        if field == "episodes" and set(before[field]) <= set(props[field] or []):
            continue
        if field not in EVOLVING or field == "episodes":
            errors.append(f"{uuid}: base field {field} rewritten on append")
    return errors


def validate_rows(
    nodes: list[dict], edges: list[dict], *, previous: dict | None = None, mode: str = "write"
) -> list[str]:
    prior = previous or {}
    errors = [e for n in nodes for e in validate_node(n)] + [e for r in edges for e in validate_edge(r, mode=mode)]
    known = {u: r for u, r in prior.items() if "labels_neo4j" in r}
    known.update({n["props"]["uuid"]: n for n in nodes if "uuid" in n["props"]})
    for edge in edges:
        errors.extend(_endpoint_errors(edge, known))
    for row in [*nodes, *edges]:
        before = prior.get(row["props"].get("uuid"))
        if before is not None:
            errors.extend(_append_errors(row, before))
    return errors
