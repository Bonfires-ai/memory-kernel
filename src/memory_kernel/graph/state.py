"""Kernel 2.0: the rows that carry a bonfire's state into the graph.

``stamp_state`` adds the new properties to a run's projected rows in place (Episodic ``order`` / ``chat`` / ``session``
/ ``message_embedding`` and friends, Saga ``chat``, a minted entity's ``minted_order`` / ``mint_kind``, kind counts,
every edge's ``run``). ``roleset_rows``, ``route_rows`` and ``meta_row`` build the per-bonfire ``Roleset``, ``Route``
and ``BonfireMeta`` rows. All pure: no database, no embedding call (the vectors come in as an argument)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.graph.g1_kinds import KindEvidence

#: The model of an Episodic's ``message_embedding`` (the events cache's voyage-3 document vectors).
MESSAGE_EMBEDDING_MODEL = "voyage-3/document"
BONFIRE_SCHEMA = "mk-bonfire/v3"
BOUNDARY_POLICY = "segmenter"
HEAD, SUPPLIED = "kind_head_", "kind_supplied_"


def _created_at() -> str:
    from memory_kernel.pipeline.stages._g1_build import CREATED_AT

    return CREATED_AT


# -- kind evidence -----------------------------------------------------------------------------------------------------


def kind_props(row: Mapping[str, Any]) -> dict[str, Any]:
    """A ``KindEvidence.to_row()`` as flat Entity properties: ``kind_person``, ``kind_located``, ``kind_loc_roles``,
    ``kind_head_<K>`` and ``kind_supplied_<K>``."""
    out: dict[str, Any] = {
        "kind_person": bool(row["person"]),
        "kind_located": bool(row["located"]),
        "kind_loc_roles": int(row["loc_roles"]),
    }
    out |= {f"{HEAD}{k}": int(n) for k, n in sorted(row["head"].items())}
    out |= {f"{SUPPLIED}{k}": int(n) for k, n in sorted(row["supplied"].items())}
    return out


def kind_row(props: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse of ``kind_props``: the ``KindEvidence.to_row()`` the properties hold (other properties ignored)."""
    head = {k.removeprefix(HEAD): int(v) for k, v in props.items() if k.startswith(HEAD)}
    supplied = {k.removeprefix(SUPPLIED): int(v) for k, v in props.items() if k.startswith(SUPPLIED)}
    return KindEvidence.from_row(
        {
            "head": head,
            "supplied": supplied,
            "loc_roles": props.get("kind_loc_roles") or 0,
            "located": props.get("kind_located"),
            "person": props.get("kind_person"),
        }
    ).to_row()


# -- the run's rows ----------------------------------------------------------------------------------------------------


def stamp_state(
    rows: Mapping[str, Any],
    *,
    run: int,
    records: Sequence[Mapping[str, Any]],
    chat_of: Mapping[str, str],
    vectors: Mapping[str, Sequence[float]],
) -> dict[str, int]:
    """Stamp the projected ``rows`` in place; returns how many of each were stamped.

    - **Episodic** rows of ``records`` (this run's select rows; a stored Episodic the rows merely re-send is not one of
      them, so it stays as stored): ``order``, ``chat``, ``session``, ``addressees``, ``text``, ``source_row_json``,
      ``message_embedding`` (``vectors[text]``, ``text`` = ``embed.message_texts``), its model and sha, ``run``.
    - **Saga** rows whose ``first_episode_uuid`` is one of those Episodic: ``chat`` and ``run``.
    - ``rows["forward"]["minted"]``: ``minted_order`` ``[run, order]``, ``mint_kind``, ``ground_legacy``, ``run``.
    - ``rows["forward"]["kinds"]`` (when present): ``kind_props`` of each entity's merged evidence.
    - Every edge: ``run``.

    ``ValueError`` on a message with no vector or a minted entity with no row."""
    from memory_kernel.pipeline import chain, embed

    by_id = {r["message_id"]: r for r in records}
    texts = embed.message_texts(records)
    stamped_ep: dict[str, str] = {}  # Episodic uuid -> its chat
    counts = {"episodic": 0, "saga": 0, "minted": 0, "kinds": 0, "edges": 0}
    for n in rows["nodes"]:
        props = n["props"]
        if "Episodic" in n["labels_neo4j"] and props.get("message_id") in by_id:
            mid = props["message_id"]
            record, text = by_id[mid], texts[mid]
            if text not in vectors:
                raise ValueError(f"no message embedding for {mid}")
            props |= {
                "order": record["order"],
                "chat": chat_of[mid],
                "session": chain.session_number(record["session_id"]),
                "addressees": list(record.get("addressees") or ()),
                "text": record["text"],
                "source_row_json": json.dumps(record, sort_keys=True),
                "message_embedding": list(vectors[text]),
                "message_embedding_model": MESSAGE_EMBEDDING_MODEL,
                "message_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "run": run,
            }
            stamped_ep[props["uuid"]] = chat_of[mid]
            counts["episodic"] += 1
    forward = rows.get("forward") or {}
    minted = {m.uuid: m for m in forward.get("minted", ())}
    kinds = forward.get("kinds") or {}
    seen: set[str] = set()
    for n in rows["nodes"]:
        props = n["props"]
        uuid = props["uuid"]
        if "Saga" in n["labels_neo4j"] and props.get("first_episode_uuid") in stamped_ep:
            props |= {"chat": stamped_ep[props["first_episode_uuid"]], "run": run}
            counts["saga"] += 1
        if uuid in minted:
            m = minted[uuid]
            props |= {
                "minted_order": [m.run, m.order],
                "mint_kind": m.kind,
                "ground_legacy": m.run == 0,
                "run": m.run,
            }
            seen.add(uuid)
            counts["minted"] += 1
        if uuid in kinds:
            props |= kind_props(kinds[uuid])
            counts["kinds"] += 1
    if len(seen) != len(minted):
        raise ValueError(f"forward gate: {len(minted) - len(seen)} minted entities have no row")
    for e in rows["edges"]:
        e["props"]["run"] = run
        counts["edges"] += 1
    return counts


# -- Roleset, Route, BonfireMeta ---------------------------------------------------------------------------------------


def roleset_rows(group: str, rolesets: Mapping[str, Mapping[str, Any]], *, run: int) -> list[dict]:
    """One ``Roleset`` node per roleset, from its eventuality row: what was asked (``gloss``, ``gloss_source``,
    ``examples``) and the settled answer (``label``, ``p_yes`` as ``p_yes_json``, ``compiled_sha256``). A row that
    lacks a settled field writes it null (the run had not settled it)."""
    nodes = []
    for roleset, row in sorted(rolesets.items()):
        p_yes = row.get("p_yes")
        props = {
            "uuid": content_uuid("roleset", group, roleset),
            "group_id": group,
            "created_at": _created_at(),
            "roleset": roleset,
            "gloss": row.get("gloss"),
            "gloss_source": row.get("gloss_source"),
            "examples": list(row.get("examples") or ()),
            "label": row.get("label"),
            "p_yes_json": None if p_yes is None else json.dumps(p_yes, sort_keys=True),
            "compiled_sha256": row.get("compiled_sha256"),
            "first_run": run,
        }
        nodes.append({"labels_neo4j": ["Roleset"], "props": props})
    return nodes


def route_rows(group: str, routes: Mapping[str, str], *, run: int) -> tuple[list[dict], list[dict]]:
    """``(Route nodes, ROUTES_TO edges)`` for ``routes`` (key -> entity uuid)."""
    nodes, edges = [], []
    for key, target in sorted(routes.items()):
        route = content_uuid("route", group, key)
        base = {"group_id": group, "created_at": _created_at()}
        nodes.append({"labels_neo4j": ["Route"], "props": {"uuid": route, **base, "key": key, "first_run": run}})
        edges.append(
            {
                "type": "ROUTES_TO",
                "source": route,
                "target": target,
                "props": {"uuid": content_uuid("routes_to", group, key), **base, "run": run},
            }
        )
    return nodes, edges


def _secret_keys(value: Any) -> set[str]:
    from memory_kernel.serve.http import SECRET_FIELDS

    if isinstance(value, Mapping):
        found = {k for k in value if k in SECRET_FIELDS}
        return found.union(*(_secret_keys(v) for v in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(_secret_keys(v) for v in value))
    return set()


def meta_props(
    *,
    bonfire_id: str,
    build_config: Mapping[str, Any],
    recipe_tag: str | None,
    extraction_model: Mapping[str, Any],
    agent_handles: Sequence[str],
    agent_handle_runs: Sequence[int],
    chat_tagged: bool,
    explicit_since: int | None,
    sessions: int,
    committed_run: int,
    generation: str,
) -> dict[str, Any]:
    """The ``BonfireMeta`` properties (without ``uuid`` / ``group_id`` / ``created_at``: ``meta_row`` adds them).
    ``agent_handle_runs`` is parallel to ``agent_handles``: the run at which each handle joined the union (the
    bootstrap's are 0), so a ground fold of run r leaves out exactly the handles known then (``graph.state_read``).
    ``query_cache_key`` is ``sha256(build_config_json)[:12]``. ``generation`` names this seal of the bonfire (each
    seed mints a new one, ``graph_seed.seal_seed``): a commit guards on it, so a run that started before a reseed
    never commits onto the reseeded group (``pipeline.commit``). ``ValueError`` when the config holds a secret field
    or the two handle lists differ in length."""
    secrets = _secret_keys(build_config)
    if secrets:
        raise ValueError(f"the build config holds secret fields: {sorted(secrets)}")
    if len(agent_handle_runs) != len(agent_handles):
        raise ValueError(f"agent_handle_runs has {len(agent_handle_runs)} runs for {len(agent_handles)} agent handles")
    config_json = json.dumps(build_config, sort_keys=True)
    return {
        "bonfire_id": bonfire_id,
        "schema": BONFIRE_SCHEMA,
        "boundary_policy": BOUNDARY_POLICY,
        "build_config_json": config_json,
        "recipe_tag": recipe_tag,
        "extraction_model_json": json.dumps(extraction_model, sort_keys=True),
        "agent_handles": list(agent_handles),
        "agent_handle_runs": [int(r) for r in agent_handle_runs],
        "chat_tagged": chat_tagged,
        "explicit_since": explicit_since,
        "sessions": sessions,
        "committed_run": committed_run,
        "generation": str(generation),
        "query_cache_key": hashlib.sha256(config_json.encode()).hexdigest()[:12],
    }


def meta_row(group: str, props: Mapping[str, Any]) -> dict:
    """The ``BonfireMeta`` node row of ``meta_props`` (its uuid is ``content_uuid("meta", group)``)."""
    base = {"uuid": content_uuid("meta", group), "group_id": group, "created_at": _created_at()}
    return {"labels_neo4j": ["BonfireMeta"], "props": {**base, **props}}
