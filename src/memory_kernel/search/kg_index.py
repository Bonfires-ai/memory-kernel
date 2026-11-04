"""The message search index as graph state: ``search_key*`` extension properties on ``:Episodic``.

``build_keys`` renders each message's expanded search key from reader rows (pure). ``persist_keys`` embeds the keys
whose sha changed and writes ``search_key``, ``search_key_sha``, ``search_key_embedding`` and ``search_key_model`` onto
the message nodes of ONE group (extension properties, never base fields). ``load_index`` reads a group back into a
``KgIndex`` and refuses an index built with another embedding model or dimension before any scoring happens.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from memory_kernel.envelope.render import render as envelope_render
from memory_kernel.graph.g1_nlp import Purpose
from memory_kernel.graph.kg_contract import ext
from memory_kernel.graph.kg_writer import KEY_EMBED_DIM, KEY_EMBED_MODEL
from memory_kernel.search.kg_recipe import expanded_key

EMBED_BATCH = 32
WRITE_BATCH = 100


class IndexModelMismatch(RuntimeError):
    """The stored index (or a query vector) does not match the embedding model and dimension in use."""


def model_tag(model: str, dim: int) -> str:
    return f"{model}:{dim}"


class IndexStale(RuntimeError):
    """Some messages have no persisted key (``missing``) or a key that no longer matches the graph (``stale``)."""

    def __init__(self, stale: frozenset[str], missing: frozenset[str]) -> None:
        super().__init__(
            f"search index is not current: {len(stale)} stale and {len(missing)} missing message keys; run persist_keys"
        )
        self.stale = stale
        self.missing = missing


KEY_RENDERS = ("expanded", "compact", "legacy", "legacy-env")
#: renders whose fact/Event lines are the persisted ``key_text`` fragments
LEGACY_RENDERS = ("legacy", "legacy-env")


def recipe_tag(key_render: str = "expanded") -> str:
    """The key recipe in force: key mode, the deixis flag (it changes key text) and a version. Stored per message."""
    from memory_kernel.knowledge import temporal_deixis as td

    if key_render not in KEY_RENDERS:
        raise ValueError(f"unknown key_render {key_render!r}; expected one of {KEY_RENDERS}")
    deixis = "on" if td.temporal_deixis_enabled() else "off"
    version = {"compact": "v3", "legacy": "v4", "legacy-env": "v5", "expanded": "v1"}[key_render]
    return f"{key_render}|deixis={deixis}|{version}"


def check_recipe(*, stored: str | None, expected: str, uuid: str) -> None:
    if stored != expected:
        raise IndexModelMismatch(
            f"message {uuid} was indexed under key recipe {stored!r}; this reader uses {expected!r}"
        )


def check_model(*, stored_model: str, stored_dim: int, model: str, dim: int) -> None:
    if stored_model != model or stored_dim != dim:
        raise IndexModelMismatch(
            f"index was built with {stored_model} at dim {stored_dim}; this reader uses {model} at dim {dim}"
        )


def check_query_dim(qvec: Sequence[float], dim: int) -> None:
    if len(qvec) != dim:
        raise IndexModelMismatch(f"query vector has dim {len(qvec)}; the index has dim {dim}")


def check_query_model(query_model: str, index_model: str) -> None:
    """The question vector must come from the model that embedded the keys (same dimension is not same space)."""
    if query_model != index_model:
        raise IndexModelMismatch(f"query embedded with {query_model}; the index was built with {index_model}")


@dataclass
class KgIndex:
    keys: dict[str, str]
    vectors: dict[str, list[float]]
    msg_facts: dict[str, list[dict[str, Any]]]  # message -> valid facts (uuid, fact, emb, asserted, modality)
    msg_events: dict[str, list[dict[str, Any]]]  # message -> Event nodes (uuid, name, asserted, modality)
    msg_entities: dict[str, list[dict[str, Any]]]  # message -> plain entities (uuid, name)
    told_at: dict[str, str | None]
    messages: dict[str, dict[str, Any]] = field(default_factory=dict)
    model: str = KEY_EMBED_MODEL
    dim: int = KEY_EMBED_DIM
    stale: frozenset[str] = frozenset()  # stored key sha differs from the key the graph now yields
    missing: frozenset[str] = frozenset()  # no stored key (not indexed); excluded from `vectors`
    key_text_fallbacks: int = (
        0  # legacy renders: facts/Events with no ``key_text`` (Graphiti-written) keyed by display text
    )


_NUMBERED = re.compile(r"^arg\d$")
_FILLER_TAGS = frozenset({"PPT", "GOL", "LOC", "COM"})
#: ``<role>_text`` stems of fillers that are not entities and still say something about the event.
_TEXT_STEMS = re.compile(r"^(arg\d|argm_loc|argm_com)_text$")


def _lemma(roleset: str) -> str:
    return roleset.split(".")[0].replace("_", " ")


def _dated(row: Mapping[str, Any]) -> str | None:
    return (
        str(row["happened_at"])
        if row.get("happened_at") and row.get("time_source") in ("deictic", "explicit")
        else None
    )


def event_fragment(ev: Mapping[str, Any]) -> str:
    """``subject lemma fillers... [date]`` of one Event, read from its role edges and ``<role>_text`` props."""
    roles = sorted(ev.get("roles") or [], key=lambda r: r["role"])
    subject = next((r for r in roles if r["tag"] == "PAG"), None) or next(
        (r for r in roles if _NUMBERED.match(r["role"])), None
    )
    rest = [(r["role"], r["target"]) for r in roles if r is not subject and r["tag"] in _FILLER_TAGS]
    referred = {r["role"] for r in roles}
    for stem, text in (ev.get("texts") or {}).items():
        role = stem.removesuffix("_text").replace("_", "-")
        if _TEXT_STEMS.match(stem) and text and role not in referred:
            rest.append((role, str(text)))
    parts = [
        subject["target"] if subject else "",
        _lemma(ev["roleset"]),
        *(t for _, t in sorted(rest)),
        _dated(ev) or "",
    ]
    return " ".join(p for p in parts if p)


def state_fragment(st: Mapping[str, Any]) -> str:
    parts = [st.get("source") or "", _lemma(st["roleset"]), st.get("target") or "", _dated(st) or ""]
    return " ".join(p for p in parts if p)


def compact_key(
    content: str, events: Sequence[Mapping[str, Any]], states: Sequence[Mapping[str, Any]], entities: Sequence[str]
) -> str:
    """content, then EVERY event/state fragment (exact-string dedupe, no containment drop), then entity names."""
    frags = {event_fragment(e) for e in events} | {state_fragment(s) for s in states}
    kept = sorted(f for f in frags if f)
    return "\n".join(_dedupe([content.strip(), *kept, *sorted({e for e in entities if e})]))


def _dedupe(parts: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    return [p for p in parts if p and not (p in seen or seen.add(p))]


def build_keys(
    rows: Sequence[Mapping[str, Any]], *, key_render: str = "expanded", purpose: Purpose = "extraction"
) -> dict[str, str]:
    """``{message uuid: search key}``. The clock is ``told_at or valid_at``; invalidated facts are skipped. ``purpose``
    is why the keys are built: the build's ``extraction`` (a parser that will not load fails loud) or serve's ``query``
    (it degrades to the small model). ``legacy-env`` keys parse every message with a sender and a clock; with
    temporal deixis on, any render parses the tense of a message with a bare weekday (its relative-time note:
    ``temporal_deixis.content_with_relative_time``); nothing else parses."""
    from memory_kernel.knowledge import temporal_deixis as td

    if key_render not in KEY_RENDERS:
        raise ValueError(f"unknown key_render {key_render!r}; expected one of {KEY_RENDERS}")

    deixis = td.temporal_deixis_enabled()
    keys: dict[str, str] = {}
    for r in rows:
        clock = r.get("told_at") or r.get("valid_at")
        content = td.content_with_relative_time(r["content"], clock) if deixis else r["content"]
        if key_render == "legacy-env" and r.get("sender_name") and clock:
            content = envelope_render(
                r["content"],
                speaker=r["sender_name"],
                addressee=r.get("addressee_name"),
                ts=str(clock),
                purpose=purpose,
            )
        if key_render == "compact":
            keys[r["uuid"]] = compact_key(
                content, r.get("compact_events") or [], r.get("compact_states") or [], r.get("entities") or []
            )
            continue
        invalid = list(r.get("fact_invalid") or [])
        if len(invalid) != len(r.get("facts") or []):
            raise ValueError(
                f"message {r['uuid']}: fact_invalid has {len(invalid)} flags for {len(r.get('facts') or [])} facts"
            )
        facts = [f for f, bad in zip(r.get("facts") or [], invalid, strict=True) if not bad]
        keys[r["uuid"]] = expanded_key(content, facts, r.get("events") or [], r.get("entities") or [])
    return keys


def key_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def _rows(driver, cypher: str, **params: Any) -> list[dict[str, Any]]:
    async with driver.session() as s:
        result = await s.run(cypher, **params)
        return [dict(r) async for r in result]


MESSAGES_CYPHER = (
    "MATCH (m:Episodic {group_id: $g}) OPTIONAL MATCH (s:Saga {group_id: $g})-[:HAS_EPISODE]->(m) "
    "RETURN m.uuid AS uuid, m.message_id AS message_id, m.content AS content, m.sender_name AS sender_name, m.addressee_name AS addressee_name, "
    "toString(m.told_at) AS told_at, toString(m.valid_at) AS valid_at, s.uuid AS saga, "
    "m.search_key AS search_key, m.search_key_sha AS search_key_sha, m.search_key_model AS search_key_model, m.search_key_recipe AS search_key_recipe, "
    "m.search_key_embedding AS search_key_embedding, m.attributes AS attributes"
)
MENTIONS_CYPHER = (
    "MATCH (m:Episodic {group_id: $g})-[:MENTIONS]->(n:Entity {group_id: $g}) "
    "RETURN m.uuid AS m, n.uuid AS uuid, n.name AS name, labels(n) AS labels, n.asserted AS asserted, "
    "n.modality AS modality, n.key_text AS key_text, n.source_text AS source_text, n.attributes AS attributes"
)
FACTS_CYPHER = (
    "MATCH (:Entity {group_id: $g})-[r:RELATES_TO {group_id: $g}]->(:Entity {group_id: $g}) "
    "RETURN r.uuid AS uuid, r.fact AS fact, r.episodes AS episodes, r.fact_embedding AS emb, "
    "r.asserted AS asserted, r.modality AS modality, r.key_text AS key_text, r.attributes AS attributes, "
    "(r.invalid_at IS NOT NULL OR r.expired_at IS NOT NULL) AS invalid"
)


EVENTS_CYPHER = (
    "MATCH (m:Episodic {group_id: $g})-[:MENTIONS]->(e:Event {group_id: $g}) "
    "OPTIONAL MATCH (e)-[r:RELATES_TO {group_id: $g}]->(t:Entity {group_id: $g}) WHERE r.event_uuid = e.uuid "
    "RETURN m.uuid AS m, e.uuid AS uuid, e.roleset AS roleset, toString(e.happened_at) AS happened_at, "
    "e.time_source AS time_source, [k IN keys(e) WHERE k ENDS WITH '_text' | [k, e[k]]] AS texts, "
    "r.name AS tag, r.role AS role, t.name AS target"
)
STATES_CYPHER = (
    "MATCH (a:Entity {group_id: $g})-[r:RELATES_TO {group_id: $g}]->(b:Entity {group_id: $g}) "
    "WHERE r.event_keys IS NOT NULL AND r.invalid_at IS NULL AND r.expired_at IS NULL "
    "UNWIND r.episodes AS m RETURN m, r.uuid AS uuid, r.roleset AS roleset, toString(r.happened_at) AS happened_at, "
    "r.time_source AS time_source, a.name AS source, b.name AS target"
)


def group_compact(event_rows: Sequence[Mapping[str, Any]], state_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    """``{message uuid: {"events": [...], "states": [...]}}`` from the per-role reader rows (deterministic order)."""
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"events": {}, "states": []})
    for r in sorted(event_rows, key=lambda r: (r["m"], r["uuid"], str(r.get("role")), str(r.get("target")))):
        ev = out[r["m"]]["events"].setdefault(
            r["uuid"],
            {
                "uuid": r["uuid"],
                "roleset": r["roleset"],
                "happened_at": r.get("happened_at"),
                "time_source": r.get("time_source"),
                "texts": {k: v for k, v in (r.get("texts") or [])},
                "roles": [],
            },
        )
        if r.get("role") and r.get("target"):
            ev["roles"].append({"tag": r["tag"], "role": r["role"], "target": r["target"]})
    for r in sorted(state_rows, key=lambda r: (r["m"], r["uuid"])):
        out[r["m"]]["states"].append(
            {k: r.get(k) for k in ("roleset", "happened_at", "time_source", "source", "target")}
        )
    return {m: {"events": list(v["events"].values()), "states": v["states"]} for m, v in out.items()}


def _key_text(row: Mapping[str, Any], display: str, fallbacks: list[str] | None = None) -> str:
    """The legacy key fragment of an Event or fact: its ``key_text`` extension (flat or in ``attributes``), else the
    display text (``fact`` / ``name``). A Graphiti-written row has no ``key_text``; it is keyed by what it says, and
    the fallback is recorded in ``fallbacks`` so the index can count it."""
    kt = ext(row, "key_text")
    if kt:
        return str(kt)
    if fallbacks is not None:
        fallbacks.append(str(row.get("uuid")))
    return str(row.get(display) or "")


_MESSAGE_EXTENSIONS = ("sender_name", "addressee_name", "told_at", "act", "mood")


def _resolve(row: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """``row`` with the extension ``keys`` read flat or from its ``attributes`` JSON."""
    out = dict(row)
    for k in keys:
        v = ext(row, k)
        if v is not None:
            out[k] = v
    return out


def build_index(
    messages: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    mentions: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, list[float]],
    *,
    model: str,
    dim: int,
    stored_keys: Mapping[str, str] | None = None,
    key_render: str = "expanded",
    compact: Mapping[str, Mapping[str, Any]] | None = None,
    purpose: Purpose = "extraction",
) -> KgIndex:
    """Assemble a ``KgIndex`` from reader rows. Each fact is attached once per message it supports (``purpose``: see
    ``build_keys``)."""
    stored_cols = ("search_key", "search_key_sha", "search_key_model", "search_key_recipe", "search_key_embedding")
    msgs = {
        m["uuid"]: {k: v for k, v in _resolve(m, _MESSAGE_EXTENSIONS).items() if k not in stored_cols}
        for m in sorted(messages, key=lambda m: m["uuid"])
    }
    fallbacks: list[str] = []
    by_msg_facts: dict[str, dict[str, Mapping[str, Any]]] = {u: {} for u in msgs}
    for f in sorted(facts, key=lambda f: f["uuid"]):
        for ep in f.get("episodes") or []:
            if ep in by_msg_facts:
                by_msg_facts[ep].setdefault(f["uuid"], f)
    events: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    entities: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in mentions:
        if r["m"] not in msgs or "TaxonomyLabel" in r["labels"]:
            continue
        if "Event" in r["labels"]:
            events[r["m"]][r["uuid"]] = {"uuid": r["uuid"], "name": r["name"]} | {
                k: ext(r, k) for k in ("asserted", "modality", "key_text", "source_text")
            }
        else:
            entities[r["m"]][r["uuid"]] = {"uuid": r["uuid"], "name": r["name"]}
    rows = []
    for u, m in msgs.items():
        fs = list(by_msg_facts[u].values())
        rows.append(
            {
                "uuid": u,
                "content": m["content"],
                "told_at": m.get("told_at"),
                "valid_at": m.get("valid_at"),
                "sender_name": m.get("sender_name"),
                "addressee_name": m.get("addressee_name"),
                "facts": [_key_text(f, "fact", fallbacks) if key_render in LEGACY_RENDERS else f["fact"] for f in fs],
                "fact_invalid": [bool(f.get("invalid")) for f in fs],
                "events": [
                    _key_text(e, "name", fallbacks) if key_render in LEGACY_RENDERS else e["name"]
                    for e in events[u].values()
                ],
                "entities": [e["name"] for e in entities[u].values()],
                "compact_events": (compact or {}).get(u, {}).get("events", []),
                "compact_states": (compact or {}).get(u, {}).get("states", []),
            }
        )
    if key_render == "compact" and compact is None and stored_keys is None:
        raise ValueError("key_render='compact' needs the per-event rows (read_compact_rows)")
    keys = dict(stored_keys) if stored_keys is not None else build_keys(rows, key_render=key_render, purpose=purpose)
    return KgIndex(
        keys=keys,
        vectors={u: list(v) for u, v in vectors.items()},
        msg_facts={
            u: [
                {k: f.get(k) for k in ("uuid", "fact", "emb")} | {k: ext(f, k) for k in ("asserted", "modality")}
                for f in by_msg_facts[u].values()
                if not f.get("invalid")
            ]
            for u in msgs
        },
        msg_events={u: sorted(events[u].values(), key=lambda e: e["uuid"]) for u in msgs},
        msg_entities={u: sorted(entities[u].values(), key=lambda e: e["uuid"]) for u in msgs},
        told_at={u: m.get("told_at") or m.get("valid_at") for u, m in msgs.items()},
        messages=msgs,
        model=model,
        dim=dim,
        key_text_fallbacks=len(fallbacks),
    )


#: ``read_key_rows`` scoped to some messages (``$ids``: their uuids): the rows their keys are built from.
SCOPED_MESSAGES_CYPHER = MESSAGES_CYPHER.replace(
    "MATCH (m:Episodic {group_id: $g}) ", "MATCH (m:Episodic {group_id: $g}) WHERE m.uuid IN $ids ", 1
)
SCOPED_FACTS_CYPHER = FACTS_CYPHER.replace(" RETURN ", " WHERE any(x IN r.episodes WHERE x IN $ids) RETURN ", 1)
SCOPED_MENTIONS_CYPHER = MENTIONS_CYPHER.replace(" RETURN ", " WHERE m.uuid IN $ids RETURN ", 1)


async def read_key_rows(
    driver, group_id: str, uuids: Sequence[str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(messages, facts, mentions) of one group, as the reader returns them; with ``uuids``, only what the keys of
    those messages are built from (a forward append's new messages)."""
    if uuids is not None:
        ids = sorted(uuids)
        return (
            await _rows(driver, SCOPED_MESSAGES_CYPHER, g=group_id, ids=ids),
            await _rows(driver, SCOPED_FACTS_CYPHER, g=group_id, ids=ids),
            await _rows(driver, SCOPED_MENTIONS_CYPHER, g=group_id, ids=ids),
        )
    return (
        await _rows(driver, MESSAGES_CYPHER, g=group_id),
        await _rows(driver, FACTS_CYPHER, g=group_id),
        await _rows(driver, MENTIONS_CYPHER, g=group_id),
    )


async def read_compact_rows(driver, group_id: str) -> dict[str, dict]:
    """Per-message event/state structure for the compact key (role edges, ``*_text`` props), grouped."""
    return group_compact(await _rows(driver, EVENTS_CYPHER, g=group_id), await _rows(driver, STATES_CYPHER, g=group_id))


def keys_from_graph(messages, facts, mentions, *, key_render: str = "expanded", compact=None) -> dict[str, str]:
    """Freshly computed keys for the graph rows (what ``persist_keys`` is given)."""
    return build_index(
        messages, facts, mentions, {}, model=KEY_EMBED_MODEL, dim=KEY_EMBED_DIM, key_render=key_render, compact=compact
    ).keys


async def persist_keys(
    driver,
    group_id: str,
    keys: Mapping[str, str],
    embedder: Callable[[list[str]], list[list[float]]],
    *,
    key_render: str = "expanded",
) -> dict[str, int]:
    """Embed only the keys whose ``search_key_sha`` (or model tag) changed and SET ``search_key*`` on the messages.

    Writes to ``:Episodic`` nodes of ``group_id`` matched by uuid; nothing else is touched. Keys for uuids that are not
    messages of the group are ignored (never embedded). A message whose key is unchanged but lacks the current
    ``search_key_recipe`` stamp gets the stamp only (no embedding). Returns ``{"keys", "embedded", "written",
    "stamped"}``.
    """
    tag = model_tag(KEY_EMBED_MODEL, KEY_EMBED_DIM)
    recipe = recipe_tag(key_render)
    stored = {
        r["uuid"]: (r["sha"], r["model"], r["recipe"])
        for r in await _rows(
            driver,
            "MATCH (m:Episodic {group_id: $g}) WHERE m.uuid IN $ids RETURN m.uuid AS uuid, m.search_key_sha AS sha, "
            "m.search_key_model AS model, m.search_key_recipe AS recipe",
            g=group_id,
            ids=sorted(keys),
        )
    }
    ours = [u for u in sorted(keys) if u in stored]
    todo = [u for u in ours if stored[u][:2] != (key_sha(keys[u]), tag)]
    restamp = [u for u in ours if u not in set(todo) and stored[u][2] != recipe]
    written = 0
    for i in range(0, len(todo), EMBED_BATCH):
        batch = todo[i : i + EMBED_BATCH]
        vecs = await asyncio.to_thread(embedder, [keys[u] for u in batch])
        if len(vecs) != len(batch) or any(len(v) != KEY_EMBED_DIM for v in vecs):
            raise IndexModelMismatch(f"embedder returned vectors that are not {KEY_EMBED_DIM}-dimensional")
        rows = [
            {"uuid": u, "key": keys[u], "sha": key_sha(keys[u]), "emb": list(v)}
            for u, v in zip(batch, vecs, strict=True)
        ]
        for j in range(0, len(rows), WRITE_BATCH):
            res = await _rows(
                driver,
                "UNWIND $rows AS r MATCH (m:Episodic {uuid: r.uuid, group_id: $g}) "
                "SET m.search_key = r.key, m.search_key_sha = r.sha, m.search_key_embedding = r.emb, "
                "m.search_key_model = $tag, m.search_key_recipe = $recipe RETURN count(m) AS n",
                rows=rows[j : j + WRITE_BATCH],
                g=group_id,
                tag=tag,
                recipe=recipe,
            )
            written += res[0]["n"]
    stamped = 0
    for j in range(0, len(restamp), WRITE_BATCH):
        res = await _rows(
            driver,
            "MATCH (m:Episodic {group_id: $g}) WHERE m.uuid IN $uuids SET m.search_key_recipe = $recipe "
            "RETURN count(m) AS n",
            uuids=restamp[j : j + WRITE_BATCH],
            g=group_id,
            recipe=recipe,
        )
        stamped += res[0]["n"]
    return {"keys": len(keys), "embedded": len(todo), "written": written, "stamped": stamped}


async def load_index(
    driver, group_id: str, *, model: str, dim: int, key_render: str = "expanded", purpose: Purpose = "extraction"
) -> KgIndex:
    """Read the persisted index of one group.

    Refuses (``IndexModelMismatch``) an index built with another embedding model, dimension or key recipe. Messages
    with no stored key are reported in ``missing`` (and left out of ``vectors``); messages whose stored key sha no
    longer matches the key the current graph yields are reported in ``stale``. Neither raises here: the search does.
    ``purpose``: serve reads with ``query`` (keys re-rendered on the small model if the configured one will not load);
    the build keeps ``extraction``. The reads run on the event loop; the key rebuild and the checks run in a worker
    thread, so a server that is opening never holds the loop (``mk serve``'s ``/healthz``). The rebuild parses only
    where its render does (``build_keys``): serve's ``legacy`` keys parse no message, except the tense of one with a
    bare weekday (its relative-time note); ``legacy-env`` keys parse every message with a sender and a clock.
    """
    messages, facts, mentions = await read_key_rows(driver, group_id)
    compact = await read_compact_rows(driver, group_id) if key_render == "compact" else None
    return await asyncio.to_thread(
        _checked_index,
        messages,
        facts,
        mentions,
        model=model,
        dim=dim,
        key_render=key_render,
        compact=compact,
        purpose=purpose,
    )


def _checked_index(
    messages: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    mentions: Sequence[Mapping[str, Any]],
    *,
    model: str,
    dim: int,
    key_render: str,
    compact: Mapping[str, Mapping[str, Any]] | None,
    purpose: Purpose,
) -> KgIndex:
    """``load_index``'s synchronous part: the stored index of the rows read, its ``stale`` and ``missing`` marked."""
    tag = model_tag(model, dim)
    recipe = recipe_tag(key_render)
    fresh = build_index(
        messages, facts, mentions, {}, model=model, dim=dim, key_render=key_render, compact=compact, purpose=purpose
    ).keys
    keys: dict[str, str] = {}
    vectors: dict[str, list[float]] = {}
    stale: set[str] = set()
    missing: set[str] = set()
    for m in messages:
        u = m["uuid"]
        stored_tag = m.get("search_key_model")
        vec = m.get("search_key_embedding")
        if not m.get("search_key") or vec is None or stored_tag is None:
            missing.add(u)
            continue
        if stored_tag != tag:
            name, _, d = str(stored_tag).rpartition(":")
            check_model(stored_model=name, stored_dim=int(d) if d.isdigit() else -1, model=model, dim=dim)
        check_recipe(stored=m.get("search_key_recipe"), expected=recipe, uuid=u)
        if len(vec) != dim:
            raise IndexModelMismatch(f"message {u} has a {len(vec)}-dim vector; expected {dim}")
        if m.get("search_key_sha") != key_sha(fresh[u]):
            stale.add(u)
        keys[u] = m["search_key"]
        vectors[u] = list(vec)
    idx = build_index(messages, facts, mentions, vectors, model=model, dim=dim, stored_keys=keys, key_render=key_render)
    idx.stale = frozenset(stale)
    idx.missing = frozenset(missing)
    return idx


__all__ = [
    "IndexModelMismatch",
    "IndexStale",
    "KgIndex",
    "build_index",
    "build_keys",
    "check_model",
    "check_query_dim",
    "check_query_model",
    "check_recipe",
    "read_compact_rows",
    "key_sha",
    "keys_from_graph",
    "load_index",
    "model_tag",
    "persist_keys",
    "recipe_tag",
    "read_key_rows",
]
