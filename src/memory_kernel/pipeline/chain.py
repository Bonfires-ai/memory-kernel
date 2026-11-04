"""Chain state: what a bonfire has minted so far (kernel 1.2's forward chain, kernel 2.0's in-memory read of the graph).

A ``Chain`` is the entities, routes, ``SAME_AS`` links, per-chat counters and message tails, and frozen roleset wires of
a bonfire. Kernel 2.0 keeps no ``chain/`` directory: the bonfire's state lives in its Neo4j group, ``graph.state_read``
builds the ``Chain`` from it, and ``graph_seed`` derives the first one from a bootstrap's projection (``from_bootstrap``,
``bootstrap_chain``). The pure helpers here are the shared vocabulary: the dataclasses, the evidence fold
(``fold_evidence``), the boundary states (``closed_boundary``, ``segmenter_state``) and the projection view the write
stage reads (``projection_view`` / ``from_projection_view``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.graph.g1_nlp import normalize_exact
from memory_kernel.graph.referents import Ground, fold_ground
from memory_kernel.pipeline.bonfire import BUILD_CONFIG, SEGMENTER_POLICY, BonfireError, _fields, host_paths
from memory_kernel.pipeline.config import BuildConfig

#: The fewest of a chat's last source rows ``ChatState.tail`` keeps. A forward append runs one whole session, and the
#: per-message stages read no message of another session: graph_picks' previous message
#: (unread: ``_clause_choice.PREVIOUS_MESSAGE_ANTECEDENTS`` is off) and ctx's previous message are both of the SAME
#: session, so the session's first message has none. Only the boundary walk reads across sessions: its day-change and
#: later-part checks read the chat's last message (1), and its window and archived part read back to the open
#: episode's start. So the tail is the open episode (from ``ChatState.boundary["episode_start"]``), and at least TAIL.
TAIL = 1


class ChainError(BonfireError):
    """The chain state is absent, damaged, of another schema, or cannot be derived from a bootstrap."""


@dataclass(frozen=True)
class Minted:
    """One row of ``chain/entities.jsonl``."""

    uuid: str
    key: str  # the resolution key the projection minted it under
    kind: str  # "person" | "entity"
    display: str
    surfaces: tuple[str, ...]
    run: int  # 0 = bootstrap
    order: int  # message order (g1_projection.message_rank) of its first mention inside that run


@dataclass(frozen=True)
class SameAs:
    """One row of ``chain/same_as.jsonl``."""

    from_uuid: str
    to_uuid: str
    message_id: str
    holder: str
    name: str
    card: str
    run: int


@dataclass
class ChatState:
    """One value of ``chain/chats.json``."""

    episodes: int  # episodes minted in this chat so far (the boundary counter continues from it)
    last_episode: str  # external episode id of the chat's last episode
    last_saga: str  # its Saga uuid (NEXT_SAGA source for the chat's next new saga)
    last_order: int
    roster: list[str]  # person names known in this chat, in first-seen order
    tail: list[dict]  # the chat's source rows (select's row shape) from tail_start(boundary), oldest first
    #: the chat's boundary walk (``_boundary_walk.continue_walk``): ``messages`` (the chat's message count, the next
    #: walk position), ``episode_start`` (the walk position the open episode began at), ``cursor`` (the last working
    #: window start). A ``segmenter`` bonfire's chat is closed after every run (``closed_boundary``: no open episode,
    #: ``policy`` "segmenter"); no ``policy`` = ``walk``
    boundary: dict


@dataclass
class Chain:
    entities: dict[str, Minted] = field(default_factory=dict)  # uuid -> Minted
    routes: dict[str, str] = field(default_factory=dict)  # name key -> canonical uuid (first route wins)
    same_as: list[SameAs] = field(default_factory=list)
    chats: dict[str, ChatState] = field(default_factory=dict)
    rolesets: dict[str, dict] = field(default_factory=dict)  # roleset -> {"gloss", "examples": [...], "run": int}
    message_ids: dict[str, int] = field(default_factory=dict)  # message_id -> run that appended it
    messages: int = 0  # messages in the chain (== group Episodic total)
    episodes: int = 0  # episodes in the chain (== group Saga total)
    sessions: int = 0  # sessions in the bonfire: the next session is session_<sessions + 1>
    #: order of the bonfire's first message carrying addressees (any chat); None = none yet. A message in an append
    #: resolves "you" through its addressees when this is at or before it (``addressing.is_explicit_at``)
    explicit_since: int | None = None
    agent_handles: list[str] = field(default_factory=list)  # every agent handle the bonfire was given, first-seen order
    run: int = 0  # last committed run number (0 = bootstrap)
    #: uuid -> where and with whom it was mentioned (None: a chain made before ground)
    ground: dict[str, Ground] | None = None
    #: message_id -> its row (``EVIDENCE_FIELDS``) for every ground row's first and last mention: what a referent
    #: question quotes, kept here because retention prunes the runs' select rows. Append-only. None exactly when
    #: ``ground`` is None; not in ``projection_view`` (the referents stage reads it from the live chain)
    evidence: dict[str, dict] | None = None
    #: uuid -> its merged kind evidence (``g1_kinds.KindEvidence.to_row``), which an append's projection adds its own
    #: evidence to before deciding the kind again. None = a phrase bonfire (kinds conferred by roles, labels frozen)
    kinds: dict[str, dict] | None = None


#: The fields of an evidence row: the select row's, as the referent question quotes them.
EVIDENCE_FIELDS = ("message_id", "speaker", "addressees", "timestamp", "text")


def evidence_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "message_id": row["message_id"],
        "speaker": row.get("speaker"),
        "addressees": list(row.get("addressees") or ()),
        "timestamp": row.get("timestamp"),
        "text": row.get("text"),
    }


def fold_evidence(
    evidence: Mapping[str, dict], ground: Mapping[str, Ground], rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict]:
    """``evidence`` with a row added for every ground row's first or last mention that it lacks, from ``rows``.
    Append-only: a stored row is never removed or overwritten. A mention ``rows`` does not hold is skipped (the
    referents stage refuses a question that would need it, naming it)."""
    out = dict(evidence)
    for g in ground.values():
        for mid in (g.first_mid, g.last_mid):
            if mid not in out and mid in rows:
                out[mid] = evidence_row(rows[mid])
    return out


def empty() -> Chain:
    """A chain with nothing minted (tests of code that only needs a chain to exist)."""
    return Chain()


def tail_start(boundary: Mapping[str, int]) -> int:
    """The walk position of a chat's first tail row: the open episode's start, or ``TAIL`` rows back if earlier."""
    return max(0, min(int(boundary["episode_start"]), int(boundary["messages"]) - TAIL))


#: The boundary policy whose chats keep no open episode (``config.BuildConfig.boundary_policy``).
SEGMENTER = SEGMENTER_POLICY


def closed_boundary(messages: int) -> dict[str, Any]:
    """A ``segmenter`` chat's boundary state after a run: no open episode (each appended session is a new source and
    opens its own Sagas), so ``episode_start`` and ``cursor`` are its message count; ``tail_start`` then keeps
    ``TAIL`` rows and ``write.chain_sagas`` gives no ``prior``. ``messages`` stays the count the bootstrap and
    ``advance`` check."""
    return {"messages": messages, "episode_start": messages, "cursor": messages, "policy": SEGMENTER}


def segmenter_state(boundary: Mapping[str, Any]) -> dict[str, Any]:
    """A chat's boundary state as a ``segmenter`` run continues it. A closed state, and a new chat's (no message), are
    kept. A ``walk`` state (a chat the walk built) is closed at its
    message count: its open episode ends where the next session starts, and that session opens new Sagas
    (``closed_boundary``)."""
    if boundary.get("policy") == SEGMENTER or not int(boundary["messages"]):
        return dict(boundary)
    return closed_boundary(int(boundary["messages"]))


def boundary_walks(
    rows: Sequence[Mapping[str, Any]], conversation_of: Mapping[str, str], policy: str
) -> dict[str, dict[str, Any]]:
    """Each chat's boundary state after a full build's boundary rows (``applied-boundaries.json``), what
    ``from_bootstrap`` stores: the walk's (``_boundary_walk.walk_states``), or under ``segmenter`` closed
    (``closed_boundary``) at the chat's message count."""
    from memory_kernel.pipeline.stages._boundary_walk import walk_states

    walks = walk_states(rows, conversation_of)
    if policy != SEGMENTER:
        return walks
    return {chat: closed_boundary(int(walk["messages"])) for chat, walk in walks.items()}


def _boundary_state(walk: Mapping[str, Any]) -> dict[str, Any]:
    """The stored form of a chat's boundary state: its three counters, and its ``policy`` when it names one (a
    ``walk`` state has none, so its bytes are unchanged)."""
    state: dict[str, Any] = {k: int(walk[k]) for k in ("messages", "episode_start", "cursor")}
    if "policy" in walk:
        state["policy"] = str(walk["policy"])
    return state


def session_number(session_id: str) -> int:
    """N of a source's ``<sample>-session_N``."""
    number = session_id.rpartition("-session_")[2]
    if not number.isdigit():
        raise ChainError(f"session id {session_id!r} is not <sample>-session_N")
    return int(number)


def next_order(chain: Chain) -> int:
    """The order of the bonfire's next message: one past the last order of any chat (orders run over every chat)."""
    return max((s.last_order + 1 for s in chain.chats.values()), default=0)


# -- storage -----------------------------------------------------------------------------------------------------------


def projection_view(chain: Chain) -> dict[str, Any]:
    """What an append's projection reads of the chain (``graph.forward``), as JSON: the entities, routes, links, chats
    and the addressing start (and the kind evidence of a structure bonfire). ``bonfire.forward_input`` pins it in the
    run's forward input when the run starts, so the projection stage never reads the live chain."""
    kinds = {} if chain.kinds is None else {"kinds": dict(sorted(chain.kinds.items()))}
    return {
        "entities": [asdict(m) for m in sorted(chain.entities.values(), key=lambda m: (m.run, m.order, m.uuid))],
        "routes": dict(sorted(chain.routes.items())),
        "same_as": [asdict(s) for s in chain.same_as],
        "chats": {chat: asdict(s) for chat, s in sorted(chain.chats.items())},
        "explicit_since": chain.explicit_since,
        "run": chain.run,
        "ground": None if chain.ground is None else [g.to_row() for _, g in sorted(chain.ground.items())],
        "agent_handles": list(chain.agent_handles),
        **kinds,
    }


def from_projection_view(view: Mapping[str, Any]) -> Chain:
    """The ``Chain`` a ``projection_view`` describes (its other fields empty: the projection does not read them)."""
    try:
        return Chain(
            entities={r["uuid"]: Minted(**{**r, "surfaces": tuple(r["surfaces"])}) for r in view["entities"]},
            routes=dict(view["routes"]),
            same_as=[SameAs(**r) for r in view["same_as"]],
            chats={c: ChatState(**s) for c, s in view["chats"].items()},
            explicit_since=view["explicit_since"],
            run=int(view["run"]),
            ground=None if view.get("ground") is None else {r["uuid"]: Ground.from_row(r) for r in view["ground"]},
            agent_handles=list(view.get("agent_handles") or []),
            kinds=None if view.get("kinds") is None else {str(u): dict(r) for u, r in view["kinds"].items()},
        )
    except (TypeError, KeyError, ValueError) as exc:
        raise ChainError(f"the run's chain view is damaged ({type(exc).__name__}: {exc})") from None


# -- the bootstrap's chain ---------------------------------------------------------------------------------------------


def from_bootstrap(
    rows: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    boundaries: Mapping[str, str],
    conversation_of: Mapping[str, str],
    rolesets: Mapping[str, dict],
    *,
    walks: Mapping[str, Mapping[str, int]],
    sessions: int | None = None,
    agent_handles: Sequence[str] = (),
) -> Chain:
    """The initial chain of a full build, from what it wrote.

    ``rows``: the bootstrap's projection (``project`` output: ``nodes``, ``edges``, ``forward``). ``records``: its select
    sources (``01-select-001/sources.json``), in source order. ``boundaries``: message id -> external episode id (the
    boundary stage's ``episode_by_message``). ``conversation_of``: message id -> chat. ``rolesets``: the events stage's
    ``eventuality.json`` ``rolesets`` (only ``gloss`` and ``examples`` are kept). ``walks``: chat -> its boundary walk
    after the build (``boundary_walks`` of the boundary rows): ``ChatState.boundary``, and where the tail starts
    (``tail_start``). ``sessions``: the build's session count (its input's ``session_N`` lists, an empty one
    included; None = the sessions ``records`` name). ``agent_handles``: its input's handles.

    - **entities** and **routes**: the projection's own first mint (``rows["forward"]``, ``graph.forward.emit``):
      every Entity node but the Events, minted at run 0 in the order of the first message that names it (a MENTIONS
      edge, or the Episodic's ``sender`` / ``addressee``), under the key it was resolved by; and every key that
      resolved onto another key's node (fuzzy merges, the key forms of an identity join, claimed names and their
      possessives). It must cover every Entity node of the rows.
    - **chats**: counters and tail from each chat's last messages; ``roster`` its speakers, addressees and the Person
      entities its messages mention, first-seen order. ``explicit_since``: the bonfire's first message with addressees.
    - **kinds** (``referent_policy`` structure): the first mint's kind evidence per entity; None under phrase.
    - ``same_as`` is empty: the bootstrap is the first mint."""
    order_of = {r["message_id"]: int(r.get("order", i)) for i, r in enumerate(records)}
    if set(boundaries) != set(order_of) or set(conversation_of) != set(order_of):
        raise ChainError("bootstrap chain: the select sources, boundaries and chat map name different messages")
    groups = {n["props"]["group_id"] for n in rows["nodes"]}
    if len(groups) != 1:
        raise ChainError(f"bootstrap chain: the projection spans groups {sorted(groups)}")
    group = next(iter(groups))

    nodes = {n["props"]["uuid"]: n for n in rows["nodes"]}
    episodic = {u: n["props"] for u, n in nodes.items() if "Episodic" in n["labels_neo4j"]}

    def is_entity(uuid: str) -> bool:
        labels = nodes[uuid]["labels_neo4j"] if uuid in nodes else ()
        return "Entity" in labels and "Event" not in labels

    mentioned_in: defaultdict[str, list[str]] = defaultdict(list)  # message id -> entity uuids it names
    for ep in episodic.values():
        mentioned_in[ep["message_id"]] += [u for u in (ep.get("sender"), ep.get("addressee")) if u and is_entity(u)]
    for edge in rows["edges"]:
        if edge["type"] == "MENTIONS" and edge["source"] in episodic and is_entity(edge["target"]):
            mentioned_in[episodic[edge["source"]]["message_id"]].append(edge["target"])

    first_mint = rows.get("forward")
    if first_mint is None:
        raise ChainError("bootstrap chain: the projection carries no forward (first mint)")
    entities = {m.uuid: m for m in first_mint["minted"]}
    if missing := sorted(u for u in nodes if is_entity(u) and u not in entities):
        raise ChainError(
            f"bootstrap chain: entities {missing[:5]} are not in the projection's first mint (never mentioned: no "
            "mint order)"
        )
    if late := sorted(m.uuid for m in entities.values() if m.run != 0):
        raise ChainError(f"bootstrap chain: entities {late[:5]} are not minted at run 0")
    routes = dict(first_mint["routes"])
    kinds = first_mint.get("kinds")

    sagas = {n["props"]["name"]: n["props"]["uuid"] for n in nodes.values() if "Saga" in n["labels_neo4j"]}
    by_chat: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_chat[conversation_of[record["message_id"]]].append(record)
    persons = {u: m.display for u, m in entities.items() if m.kind == "person"}
    chats: dict[str, ChatState] = {}
    for chat, msgs in by_chat.items():
        walk = _boundary_state(walks[chat]) if chat in walks else None
        if walk is None or walk["messages"] != len(msgs):
            raise ChainError(
                f"bootstrap chain: chat {chat!r}: the boundary walk does not cover its {len(msgs)} messages"
            )
        last = msgs[-1]
        last_episode = boundaries[last["message_id"]]
        saga = sagas.get(last_episode)
        if saga != content_uuid("saga", group, last_episode):
            raise ChainError(f"bootstrap chain: chat {chat!r}: the Saga of episode {last_episode!r} is not in the rows")
        roster: dict[str, str] = {}
        for msg in msgs:
            names = [msg["speaker"], *(msg.get("addressees") or ())]
            names += sorted(persons[u] for u in mentioned_in[msg["message_id"]] if u in persons)
            for name in names:
                roster.setdefault(normalize_exact(name), name)
        chats[chat] = ChatState(
            episodes=len({boundaries[m["message_id"]] for m in msgs}),
            last_episode=last_episode,
            last_saga=saga,
            last_order=order_of[last["message_id"]],
            roster=list(roster.values()),
            tail=[json.loads(json.dumps(dict(m))) for m in msgs[tail_start(walk) :]],
            boundary=walk,
        )

    rows_by_mid = {r["message_id"]: r for r in records}
    by_entity: defaultdict[str, list[str]] = defaultdict(list)
    for mid, uuids in mentioned_in.items():
        for u in uuids:
            by_entity[u].append(mid)
    ground = fold_ground({}, by_entity, rows_by_mid, conversation_of.__getitem__, agent_handles, legacy=True)
    evidence = fold_evidence({}, ground, rows_by_mid)

    return Chain(
        entities=entities,
        routes=routes,
        same_as=[],
        chats=chats,
        rolesets={
            rs: {"gloss": row["gloss"], "examples": list(row["examples"]), "run": 0} for rs, row in rolesets.items()
        },
        message_ids=dict.fromkeys(order_of, 0),
        messages=len(order_of),
        episodes=len(set(boundaries.values())),
        sessions=len({r["session_id"] for r in records}) if sessions is None else sessions,
        explicit_since=min((order_of[r["message_id"]] for r in records if r.get("addressees")), default=None),
        agent_handles=list(dict.fromkeys(agent_handles)),
        run=0,
        ground=ground,
        evidence=evidence,
        kinds=None if kinds is None else {u: dict(r) for u, r in sorted(kinds.items())},
    )


# -- one append's session ----------------------------------------------------------------------------------------------


def _fold_forward(chain: Chain, forward: Mapping[str, Any]) -> tuple[dict[str, Minted], dict[str, str], list[SameAs]]:
    """``advance``'s entities, routes and links with a run's ``forward`` (``stages.write.forward_record``) folded in."""
    try:
        minted = [Minted(**{**m, "surfaces": tuple(m["surfaces"])}) for m in forward.get("minted") or ()]
        links = [SameAs(**x) for x in forward.get("same_as") or ()]
        new_routes = {str(k): str(u) for k, u in (forward.get("routes") or {}).items()}
    except (TypeError, KeyError, ValueError) as exc:
        raise ChainError(
            f"forward append: the write's forward result is damaged ({type(exc).__name__}: {exc})"
        ) from None
    if forward and int(forward.get("run", -1)) != chain.run + 1:
        raise ChainError(f"forward append: the write minted at run {forward.get('run')}, not {chain.run + 1}")
    entities = dict(chain.entities)
    for m in minted:
        if m.uuid in entities:
            raise ChainError(f"forward append: entity {m.uuid} ({m.key!r}) is already in the chain")
        if m.run != chain.run + 1:
            raise ChainError(f"forward append: entity {m.uuid} is minted at run {m.run}, not {chain.run + 1}")
        entities[m.uuid] = m
    if stray := sorted(k for k, u in new_routes.items() if u not in entities):
        raise ChainError(f"forward append: routes {stray[:5]} lead to no entity of the chain")
    routes = dict(chain.routes)
    for key, uuid in new_routes.items():
        routes.setdefault(key, uuid)  # the first route for a name wins
    if bad := [x for x in links if x.from_uuid not in entities or x.to_uuid not in entities]:
        raise ChainError(f"forward append: a SAME_AS link leaves the chain ({bad[0].from_uuid} -> {bad[0].to_uuid})")
    return entities, routes, [*chain.same_as, *links]


def _bootstrap_config(root: Path) -> BuildConfig:
    """The bootstrap's config rooted at ``root`` itself (a moved root is fine), reading its own conversation-only copy
    and this host's install paths."""
    path = root / BUILD_CONFIG
    try:
        cfg = BuildConfig.from_json(json.loads(path.read_bytes()), secrets={})
    except (OSError, ValueError) as exc:
        raise ChainError(f"bootstrap {root}: no readable {BUILD_CONFIG} ({type(exc).__name__}: {exc})") from None
    if root.name != cfg.sample_id:
        raise ChainError(f"bootstrap {root}: the root of {cfg.sample_id!r} must be named {cfg.sample_id!r}")
    conversation = root / "01-select-001" / f"{cfg.stem}-conversation-only.json"
    return replace(cfg, workdir=root.parent, conversation=conversation, **host_paths(_fields(cfg)))


def bootstrap_chain(root: Path) -> Chain:
    """The initial chain of the sealed bootstrap build at ``root`` (its conversation root).

    The projection rows are not persisted by a build (``build_dry`` keeps a summary and their hash), but the projection
    is pure: they are projected again from the stage outputs the root keeps (the graph, picks,
    eventuality and boundary files), exactly as ``write`` projected them. The select sources, the boundaries and the
    roleset wires are read from the root too. ``ChainError`` when an input is missing: an empty chain would mint
    duplicates at the first append."""
    from memory_kernel.pipeline.runner import Ctx, path
    from memory_kernel.pipeline.stages import write
    from memory_kernel.pipeline.stages._conversation import session_keys

    root = Path(root)
    cfg = _bootstrap_config(root)
    try:
        sources = json.loads((path(cfg, "select") / "sources.json").read_bytes())
        events = json.loads((path(cfg, "events") / "eventuality.json").read_bytes())
        applied = json.loads((path(cfg, "boundary") / "applied-boundaries.json").read_bytes())
        conv = json.loads(Path(cfg.conversation).read_bytes())["conversation"]
        data, _, rows = write.project(Ctx(cfg=cfg, concurrency=1))
        return from_bootstrap(
            rows,
            sources,
            data["episodes"],
            data["conversation_of"],
            events["rolesets"],
            walks=boundary_walks(applied["rows"], data["conversation_of"], cfg.boundary_policy),
            sessions=len(list(session_keys(conv))),
            agent_handles=conv.get("agent_handles") or (),
        )
    except ChainError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise ChainError(f"bootstrap {root}: cannot derive the chain state ({type(exc).__name__}: {exc})") from None
