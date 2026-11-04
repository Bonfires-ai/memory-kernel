"""The append run's shape: how a session becomes a run's inputs, and the config the run builds under.

Kernel 2.0 keeps a bonfire's state in its Neo4j group (``BonfireMeta`` and the graph), not in a directory, so this module
holds only what ``pipeline.graph_append`` and the stages share:

* the session: ``session_turns`` / ``read_session`` check it, ``plan_session`` numbers and orders it as a full build of
  the whole history would (``SessionPlan``), ``forward_input`` pins what its stages need from the chain state;
* the run: ``run_config`` roots the bonfire's inherited build config at a run, ``host_paths`` keeps the install paths
  that exist on this host;
* the budget of one run: ``append_budget``, ``append_limit`` and ``append_estimate`` (counted from the shard's
  held-or-spent total, ``shard_spend``), and ``CEILING``, the refusal of a run past the bonfire's spend ceiling;
* the errors of an append (``SessionDuplicate``, ``SessionEmpty``, ``PendingSession``, ``AppendBusy``).

With ``on_built="skip"`` a partly built session is not refused: its built message ids are dropped before anything is
planned and the rest are appended (``plan_session``). The bootstrap's reproduction-only inputs
(``CLEARED_ON_APPEND``) are not inherited.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any, Literal, get_args

from memory_kernel.errors import BonfireError
from memory_kernel.grammar import lexicon
from memory_kernel.pipeline.config import BuildConfig
from memory_kernel.pipeline.runner import (
    INDEX_USD_PER_MESSAGE,
    STOP_FACTOR,
    planned_usd,
)
from memory_kernel.pipeline.stages._conversation import (
    FIRST_SESSION,
    build_prepared,
    build_sources,
    split_conversation,
)

_log = logging.getLogger(__name__)
#: The least an append run may add to the shard: a session of one to a few messages still runs every per-session stage
#: (a few cents for one to three new messages), so the floor keeps a short session from stopping on a budget sized to
#: its message count alone. Kept at 1.7's value: lowering it saves nothing (a run pays only what it sends) and could
#: stop a short session.
APPEND_FLOOR_USD = Decimal("0.10")
#: The refusal of an append whose budget would take the shard past the bonfire's ``spend_ceiling_usd``.
CEILING = "bonfire spend ceiling reached"
BUILD_CONFIG = "logs/mk/build-config.json"
#: The run's own copy of the input session, in its run directory (``runs/NNNN``, the run's workdir): the bytes the
#: checks read, never the caller's file, which may change while the run goes on.
INPUT_COPY = "session.json"
#: The run's conversation (``cfg.conversation``, beside ``INPUT_COPY``): the session alone, numbered as the bonfire
#: numbers it (``FIRST_SESSION``), in the input format every stage reads.
RUN_CONVERSATION = "conversation.json"
#: The run's forward input (``cfg.forward_input``, beside ``INPUT_COPY``): what its stages need from the chain
#: (``forward_input``), pinned when the run starts. No stage reads the chain itself.
FORWARD_INPUT = "forward.json"
FORWARD_SCHEMA = "mk-forward-input/v1"
#: The refusal of a session that shares some, not all, message ids with the bonfire (``on_built="refuse"``).
OVERLAP = "session overlaps an already-built session"
#: What an append does with a session's already-built message ids when only some are built: ``refuse`` the session
#: (``OVERLAP``, the default), or ``skip`` them and append the rest (``plan_session``).
OnBuilt = Literal["refuse", "skip"]
ON_BUILT: tuple[OnBuilt, ...] = get_args(OnBuilt)
#: The bootstrap's opt-in reproduction inputs an append run does NOT inherit: a reference build to compare budgets with
#: (sized to the bootstrap's conversation), the boundary replay pair and the freeze manifest. Each can be passed to
#: ``mk append`` explicitly.
CLEARED_ON_APPEND: tuple[str, ...] = (
    "invariants_reference",
    "boundary_replay_archive",
    "boundary_replay_scope",
    "freeze_manifest",
)
#: The bootstrap's build-only flags an append run resets: a genesis bootstrap (``BuildConfig.genesis``) is the one
#: genesis build of its bonfire, and an append is never one.
NOT_INHERITED: dict[str, bool] = {"genesis": False}
#: The boundary policy whose chats keep no open episode (``chain.SEGMENTER``).
SEGMENTER_POLICY = "segmenter"


class SessionDuplicate(BonfireError):
    """Every message id of the session is in the bonfire already: it was appended (a retried job)."""


class SessionEmpty(BonfireError):
    """Every message of the session is blank or markup only: nothing to build."""


class PendingSession(BonfireError):
    """A run after the chain's last wrote ANOTHER session into the group but was not committed (it failed after its
    write): every later write would find that session's rows. Rerun that session first (the append lease refuses any other)."""


class AppendBusy(BonfireError):
    """Another ``append_session`` of the same bonfire is running: RETRIABLE (send the session again later)."""


#: The stored answer of a settled roleset (``stages._events_core._settled``): all three, or none.
SETTLED_ROLESET_FIELDS: tuple[str, ...] = ("label", "p_yes", "compiled_sha256")


def roleset_wire(row: Mapping[str, Any]) -> dict[str, Any]:
    """A roleset's row in the forward input: its frozen ``gloss`` and ``examples``, plus ``SETTLED_ROLESET_FIELDS`` when
    the bonfire stored all three (the events stage then asks nothing for it)."""
    wire: dict[str, Any] = {"gloss": row["gloss"], "examples": row["examples"]}
    settled = {k: row.get(k) for k in SETTLED_ROLESET_FIELDS}
    if all(v is not None for v in settled.values()):
        wire |= settled
    return wire


def require_ground(state: Any, *, where: str = "the chain") -> None:
    """Refused (``BonfireError``) when the chain ``state`` has no ground, or ground but no evidence (a chain made
    before referent resolution): the referents stage plans over both."""
    if state.ground is None or state.evidence is None:
        raise BonfireError(f"{where} has no ground (made before referent resolution): run mk bonfire ground")


def run_config(
    base: BuildConfig,
    run_root: Path,
    *,
    conversation: Path,
    forward_input: Path | None,
    new_messages: int,
    limit_base_usd: Decimal | str | float,
    ceiling_usd: Decimal | str | float | None,
    index_record: Path,
    shard_dir: Path | None,
    capture_cache: Path | None,
    **overrides: Any,
) -> BuildConfig:
    """The append run's config: ``base`` (the bonfire's inherited build config) rooted at ``run_root``, reading
    ``conversation`` and ``forward_input``, with the shard (``shard_dir``: None makes the run's own in its scratch
    root) and capture cache, the run's budget (``new_messages``, counted from ``limit_base_usd``: the shard's
    held-or-spent total before the run, 0 for a fresh scratch shard) under ``ceiling_usd``, the append write mode,
    ``index_record`` (the key recipe tag the group is indexed under: the preflight refuses a run whose tag differs),
    the install paths for THIS host (``host_paths``: ``base``'s where they exist here, else this host's defaults),
    the reproduction inputs cleared (``CLEARED_ON_APPEND``), the build-only flags reset (``NOT_INHERITED``), the
    graph-mode write (``graph_commit``: the stages write nothing, ``pipeline.commit`` does) and ``overrides`` last
    (secrets, runtime flags, an install path or a reproduction input). An override of the boundary policy is refused
    unless it is ``base``'s own, so a run's forward input and the chain's chat states always follow one policy."""
    own = base.boundary_policy
    if overrides.get("boundary_policy", own) != own:
        raise BonfireError(f"boundary_policy {overrides['boundary_policy']!r}: this bonfire appends under {own!r}")
    return replace(
        base,
        conversation=conversation,
        workdir=run_root.parent,
        shard_dir=shard_dir,
        capture_cache=capture_cache,
        bonfire_index_record=index_record,
        write_mode="append",
        append_messages=new_messages,
        append_base_usd=str(limit_base_usd),
        spend_ceiling_usd=None if ceiling_usd is None else str(ceiling_usd),
        forward_input=forward_input,
        graph_commit=True,
        **(host_paths(_fields(base)) | {k: None for k in CLEARED_ON_APPEND} | NOT_INHERITED | dict(overrides)),
    )


#: The ``BuildConfig`` fields that name the host's own install (the code root, the vault, ``mk setup``'s data dir and
#: the frame dirs under it, the harness checkout, the engine, the recipe and cache files, explicit pins). An append run
#: and a served answer keep the bootstrap's value of each when it exists on THIS host, else take this host's default
#: (``host_paths``).
HOST_PATH_FIELDS: tuple[str, ...] = (
    "root",
    "vault",
    "data_dir",
    "frame_dirs",
    "harness_root",
    "engine_binary",
    "engine_image",
    "recipe_manifest",
    "embedding_cache",
    "snapshot_pins",
)


#: The run's evidence rows (``cfg.forward_input``'s directory): message id -> its row (``chain.Chain.evidence``).
EVIDENCE_INPUT = "evidence.json"


def message_rows(run_dir: Path) -> dict[str, dict]:
    """Message id -> its row (``speaker``, ``addressees``, ``timestamp``, ``text``): the bonfire's evidence
    (``chain.Chain.evidence``: every ground row's first and last mention) as the run pinned it in ``run_dir``
    (``EVIDENCE_INPUT``), read only. Never the runs' select rows: retention prunes them. A row never changes once
    written."""
    try:
        return dict(json.loads((Path(run_dir) / EVIDENCE_INPUT).read_bytes()) or {})
    except (OSError, ValueError) as exc:
        raise BonfireError(f"{Path(run_dir) / EVIDENCE_INPUT} does not read ({type(exc).__name__}: {exc})") from None


def _fields(cfg: BuildConfig) -> dict[str, Any]:
    return {name: getattr(cfg, name) for name in HOST_PATH_FIELDS}


#: (field, missing path) pairs ``host_paths`` already warned about in this process.
_WARNED_INSTALL_PATHS: set[tuple[str, str]] = set()


def host_paths(bootstrap: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """``HOST_PATH_FIELDS`` for a run or an answer on THIS host, from the bootstrap's values (``bootstrap``: its
    ``build-config.json`` or a ``BuildConfig``'s fields). A value that is unset, or a path that EXISTS here, is kept as
    it is (on the bootstrap's own host nothing changes: a vault or custom frame dirs keep their ``gloss_source``
    labels, a harness-made shard keeps opening). A path missing here (the bootstrap was built on another host) is
    replaced by this host's default, as ``mk build`` resolves it with no flag: the root is the working directory, the
    data dir ``MK_DATA_DIR`` or the user data dir (``setup_runtime.resolve_data_dir``), the frame dirs derived from it
    (the list is kept only when EVERY dir exists), the rest unset (the ``mk setup`` receipt's engine, the computed
    recipe and pins, no vault, no harness checkout, no embedding cache)."""
    from memory_kernel import setup_runtime

    env = os.environ if environ is None else environ
    defaults: dict[str, Any] = dict.fromkeys(HOST_PATH_FIELDS)
    defaults["root"] = Path(os.path.abspath("."))
    defaults["data_dir"] = setup_runtime.resolve_data_dir(None, env)
    defaults["frame_dirs"] = ()
    out: dict[str, Any] = {}
    for name in HOST_PATH_FIELDS:
        value = bootstrap.get(name)
        if name == "frame_dirs":
            dirs = tuple(Path(v) for v in value or ())
            missing = [str(d) for d in dirs if not d.is_dir()]
            out[name] = defaults[name] if missing else dirs
        elif value is None:
            out[name] = None
            missing = []
        else:
            missing = [] if os.path.exists(value) else [str(value)]
            out[name] = defaults[name] if missing else Path(value)
        key = (name, ", ".join(missing))
        if missing and key not in _WARNED_INSTALL_PATHS:  # once per process: serve_config asks on every question
            _WARNED_INSTALL_PATHS.add(key)
            # a missing embedding cache means paid embedding calls; other frame dirs mean other labels (new wires)
            _log.warning(
                "install path %s %s is missing on this host: using this host's default %s",
                name,
                ", ".join(missing),
                out[name] or "(none)",
            )
    return out


# -- the query shard (retired in 1.8) -------------------------------------------------------------------------------


# -- a bonfire in another place (the store's hydrate) ------------------------------------------------------------------


# -- the budget ------------------------------------------------------------------------------------------------------


def append_budget(new_messages: int, *, planned_jev_usd: Decimal) -> Decimal:
    """What one append run may add to the bonfire shard: ``runner.STOP_FACTOR`` x the per-message projection
    (``runner.INDEX_USD_PER_MESSAGE``) of its new messages, or ``STOP_FACTOR`` x ``planned_jev_usd`` (the Jev spend its
    saved plans project, ``runner.planned_usd``: a plan's ``est_usd``; no stage of the 1.8 set records one, so this term
    is 0 unless a plan does), whichever is larger, and never less than ``APPEND_FLOOR_USD``. Old messages replay at
    $0."""
    stop = Decimal(str(STOP_FACTOR))
    per_message = stop * Decimal(str(INDEX_USD_PER_MESSAGE)) * new_messages
    return max(per_message, stop * planned_jev_usd, APPEND_FLOOR_USD)


def append_limit(cfg: BuildConfig) -> Decimal | None:
    """The held-or-spent total an append run's stages may take the bonfire shard to (``ledger.Shard.limit_usd``): the
    shard's total when the run was planned (``cfg.append_base_usd``) plus ``append_budget`` of the run's new messages
    and of the Jev spend of the plans it has saved so far (``runner.planned_usd``), never past the bonfire's ceiling
    (``cfg.spend_ceiling_usd``). The shard's sealed caps do not bound it. None for a run that is not an append."""
    if cfg.append_base_usd is None or cfg.append_messages is None:
        return None
    budget = append_budget(cfg.append_messages, planned_jev_usd=Decimal(str(planned_usd(cfg))))
    limit = Decimal(cfg.append_base_usd) + budget
    if cfg.spend_ceiling_usd is not None:
        limit = min(limit, Decimal(cfg.spend_ceiling_usd))
    return limit


def append_estimate(new_messages: int, runs: Sequence[Mapping[str, Any]]) -> Decimal:
    """What the preflight expects an append run of ``new_messages`` to add to the shard, before any of its plans
    exist: ``append_budget`` with no planned Jev spend, or the last recorded run's ``budget_usd`` (its budget with the
    Jev spend its plans projected) scaled to ``new_messages`` (per new message, rounded up to a millionth), whichever
    is larger. ``runs`` are ``bonfire.json``'s; a run recorded without ``budget_usd`` (before 1.6) or without new
    messages is skipped. Conservative: a short last run held at the floor scales the floor too."""
    estimate = append_budget(new_messages, planned_jev_usd=Decimal(0))
    for run in reversed(runs):
        budget, messages = run.get("budget_usd"), run.get("new_messages")
        if isinstance(budget, (int, float)) and isinstance(messages, int) and messages > 0:
            scaled = Decimal(str(budget)) * new_messages / messages
            return max(estimate, scaled.quantize(Decimal("0.000001"), rounding=ROUND_CEILING))
    return estimate


def shard_budget(shard: Path) -> dict:
    """Read-only spend of a round shard beside its sealed cap (the manifest's ``budget_cap_usd``;
    ``IntegratedBudget.summary`` over the budget dir, not held against the integrated cap; no lock, no pin write).
    ``RoundBudget.reserve`` itself refuses a send past its cap."""
    from memory_kernel.pipeline.ledger import IntegratedBudget

    cap = Decimal(json.loads((shard / "manifest.json").read_bytes())["budget_cap_usd"])
    reader = object.__new__(IntegratedBudget)
    reader.directory = shard / "budget"
    summary = IntegratedBudget.summary(reader, integrated=False)
    spent = Decimal(summary["held_or_spent_usd"])
    return {
        "cap_usd": str(cap),
        "held_or_spent_usd": str(spent),
        "remaining_usd": str(cap - spent),
        "calls": summary["calls"],
    }


def shard_spend(shard: Path) -> Decimal:
    """The shard's held-or-spent total (``shard_budget``, read-only); 0 for a budget dir never opened."""
    if not (shard / "budget" / "manifest.json").is_file():
        return Decimal(0)
    return Decimal(shard_budget(shard)["held_or_spent_usd"])


# -- retention -------------------------------------------------------------------------------------------------------


# -- mk append -------------------------------------------------------------------------------------------------------


_MONTHS = tuple(lexicon.month_names().values())


def session_date(when: datetime) -> str:
    """``H:MM am|pm on D Month, YYYY`` in UTC (``parse_session_date`` reads the string as UTC)."""
    utc = when.astimezone(UTC)
    hour = utc.hour % 12 or 12
    half = "am" if utc.hour < 12 else "pm"
    return f"{hour}:{utc.minute:02d} {half} on {utc.day} {_MONTHS[utc.month - 1]}, {utc.year}"


def session_turns(session: Mapping[str, Any]) -> tuple[list[dict[str, Any]], datetime]:
    """A session (``SessionInput``'s shape: ``conversation_id``, ``messages`` of ``{message_id, speaker, text,
    timestamp, addressees?}``, ``agent_handles?``) as turns in the input format, and its first message's time. Refuses
    (``BonfireError``) a malformed session."""
    conversation_id = session.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise BonfireError("the session's conversation_id must be a non-empty string")
    messages = session.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BonfireError("the session has no messages")
    handles = session.get("agent_handles")
    if handles is not None and (not isinstance(handles, list) or not all(isinstance(h, str) and h for h in handles)):
        raise BonfireError("the session's agent_handles must be a list of non-empty strings")
    turns: list[dict[str, Any]] = []
    seen: set[str] = set()
    first: datetime | None = None
    for i, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            raise BonfireError(f"message {i}: not an object")
        mid = msg.get("message_id")
        if not isinstance(mid, str) or not mid:
            raise BonfireError(f"message {i}: message_id must be a non-empty string")
        if mid in seen:
            raise BonfireError(f"message id {mid!r} appears twice in the session")
        seen.add(mid)
        raw = msg.get("timestamp")
        if not isinstance(raw, str):
            raise BonfireError(f"message {mid!r}: timestamp must be an ISO 8601 string, not {type(raw).__name__}")
        try:
            when = datetime.fromisoformat(raw)
        except ValueError:
            raise BonfireError(f"message {mid!r}: timestamp {raw!r} is not an ISO 8601 time") from None
        if when.tzinfo is None:
            raise BonfireError(f"message {mid!r}: timestamp {raw!r} has no timezone")
        first = first or when
        for key in ("speaker", "text"):
            if not isinstance(msg.get(key), str):
                raise BonfireError(f"message {mid!r}: {key} must be a string")
        turn = {"speaker": msg["speaker"], "dia_id": mid, "text": msg["text"], "timestamp": raw}
        addressees = msg.get("addressees")
        if addressees is not None:
            if not isinstance(addressees, list) or not all(isinstance(a, str) and a for a in addressees):
                raise BonfireError(f"message {mid!r}: addressees must be a list of non-empty strings")
            turn["addressees"] = list(addressees)
        turns.append(turn)
    assert first is not None
    return turns, first


def read_session(path: Path) -> tuple[dict[str, Any], bytes]:
    """``(session, raw)``: the session file at ``path`` (``session_turns`` checks it) and the bytes read."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise BonfireError(f"session {path}: {type(exc).__name__}: {exc}") from None
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise BonfireError(f"session {path}: not JSON ({exc})") from None
    if not isinstance(doc, dict):
        raise BonfireError(f"session {path}: expected an object {{conversation_id, messages, agent_handles?}}")
    session_turns(doc)
    return doc, raw


@dataclass(frozen=True)
class SessionPlan:
    chat: str
    number: int  # the session's bonfire-wide number (its key is session_<number>)
    session_id: str  # <sample>-session_<number>
    first_order: int  # its first message's order
    conversation: dict  # the run's conversation (``RUN_CONVERSATION``)
    sources: list  # the select rows it gives
    skipped_ids: tuple[str, ...] = ()  # the session's built message ids it does not append (``on_built="skip"``)

    @property
    def new_messages(self) -> int:
        return len(self.sources)


def plan_session(
    session: Mapping[str, Any],
    state: Any,
    base: BuildConfig,
    *,
    sample_id: str,
    chat_tagged: bool,
    on_built: OnBuilt = "refuse",
) -> SessionPlan:
    """What one forward append of ``session`` (``SessionInput``'s shape: ``read_session``) runs, against the chain
    ``state`` and the bootstrap's config ``base``. Creates nothing.

    - **chat**: the session's ``conversation_id`` when the bootstrap's sessions name chats; else the bonfire's one
      chat, the sample (another id is refused: tagging one session would mix a tagged and an untagged history).
    - **number**: one past the bonfire's session count (``Chain.sessions``), so the session id is the one a full
      build of the whole history gives it.
    - **first order**: one past the bonfire's last order (orders run over every chat).
    - **agent handles**: the bonfire's union (``Chain.agent_handles``), then the session's own.

    Refused: ``SessionDuplicate`` when every message id is in the chain already, ``OVERLAP`` when some are and
    ``on_built`` is ``"refuse"``, and ``SessionEmpty`` when nothing is left after input cleanup.

    With ``on_built="skip"`` the built message ids of a partly built session are dropped (``skipped_ids``), never
    appended again: the rest keep their order and timestamps, the budget counts them alone (``new_messages``) and
    the session is dated by the first of them. There is no pending-run check: in 2.0 the append lease covers it
    (``ops.base.check_lease_free``)."""
    from memory_kernel.pipeline import chain as chain_state

    if on_built not in ON_BUILT:
        raise BonfireError(f"on_built must be one of {', '.join(ON_BUILT)}, not {on_built!r}")
    sample = sample_id
    turns, first = session_turns(session)
    # the exact ids, and the turn id of a split message's parts (<id>#k); an unsplit id may itself hold a "#"
    built = set(state.message_ids) | {m.rpartition("#")[0] or m for m in state.message_ids}
    present = [t["dia_id"] for t in turns if t["dia_id"] in built]
    if present and len(present) == len(turns):
        raise SessionDuplicate(f"every message id of the session is in the bonfire already (first {present[0]!r})")
    if present and on_built == "refuse":
        raise BonfireError(
            f"{OVERLAP}: {len(present)} of the session's {len(turns)} message ids are already built (first "
            f"{present[0]!r}); a session is appended whole, never merged onto a built one (on_built='skip', "
            "mk append --on-built skip, appends only its new messages)"
        )
    if present:
        turns = [t for t in turns if t["dia_id"] not in built]
        first = datetime.fromisoformat(turns[0]["timestamp"])
    chat = str(session["conversation_id"])
    if not chat_tagged and chat != sample:
        raise BonfireError(
            f"the bonfire's conversation names no chat (no session_N_conversation): its sessions belong to the "
            f"default chat {sample!r}, so a session of chat {chat!r} cannot be added "
            "(refresh the bonfire with a bootstrap whose sessions are tagged)"
        )
    number = state.sessions + 1
    handles = list(dict.fromkeys([*state.agent_handles, *(session.get("agent_handles") or [])]))
    key = f"session_{number}"
    conv: dict[str, Any] = {FIRST_SESSION: number, key: turns, f"{key}_date_time": session_date(first)}
    if chat_tagged:
        conv[f"{key}_conversation"] = chat
    if handles:
        conv["agent_handles"] = handles
    item = {"sample_id": sample, "conversation": conv}
    first_order = chain_state.next_order(state)
    try:
        prepared = build_prepared(split_conversation(item)[0])
        if not prepared:
            raise SessionEmpty("every message of the session is blank or markup only: nothing to append")
        sources = build_sources(prepared, {"bonfire_id": base.bonfire_id, "conversation_id": sample}, first_order)
    except SessionEmpty:
        raise
    except ValueError as exc:
        raise BonfireError(f"the session does not prepare: {exc}") from None
    return SessionPlan(
        chat=chat,
        number=number,
        session_id=f"{sample}-{key}",
        first_order=first_order,
        conversation=item,
        sources=sources,
        skipped_ids=tuple(present),
    )


def forward_input(
    state: Any, plan: SessionPlan, *, voyage_cache: str | Path, boundary_policy: str | None = None
) -> dict[str, Any]:
    """The run's forward input (``FORWARD_INPUT``): what its stages need from the chain ``state``.

    ``chain``: what the projection resolves against (``chain.projection_view``: entities, routes, links, chats,
    addressing start); its mint run is the view's ``run`` + 1, the same for every replay of the session.

    ``boundary`` is the chat's boundary state to continue: its walk (``_boundary_walk.continue_walk``) under ``walk``,
    or a ``segmenter`` chat's closed state with its ``policy``; counters and tail (empty
    for a new chat), the number of the next new episode (the bonfire's episode count) and whether the bonfire
    has more than one chat. ``explicit_since``: the bonfire's first addressed message (``addressing.is_explicit_at``).
    ``rolesets``: the frozen roleset wires (a settled row, one whose ``label``, ``p_yes`` and ``compiled_sha256`` are
    all stored, carries them, and the events stage asks nothing for it). ``voyage_cache``: the events stage's Voyage
    cache file.
    ``totals``: the chain's message and episode totals (the group's Episodic and Saga totals before the run).
    ``skipped_ids``: the session's built message ids the run does not append (``plan_session``), pinned for the
    run's entry in the result of the append.
    Refused (``BonfireError``, ``require_ground``) when the chain has no ground or no evidence (a chain made
    before referent resolution), which the referents stage plans over; ``append`` refuses that earlier, before the
    run directory exists.

    ``boundary_policy``: the run's (``BuildConfig.boundary_policy``). Under ``segmenter`` each chat the walk built
    is closed at its message count (``chain.segmenter_state``), in ``boundary`` and in the ``chain`` view the
    write stage reads the chat's last Saga from: a chat the walk built ends its open episode where its next session
    starts."""
    from memory_kernel.pipeline import chain as chain_state

    require_ground(state)  # defence: append refuses before the run directory exists
    chat = state.chats.get(plan.chat)
    walk = chat.boundary if chat is not None else {"messages": 0, "episode_start": 0, "cursor": 0}
    view = chain_state.projection_view(state)
    if boundary_policy == SEGMENTER_POLICY:
        walk = chain_state.segmenter_state(walk)
        for row in view["chats"].values():
            row["boundary"] = chain_state.segmenter_state(row["boundary"])
    return {
        "schema": FORWARD_SCHEMA,
        "chain_run": state.run,
        "session_id": plan.session_id,
        "session_number": plan.number,
        "chat": plan.chat,
        "first_order": plan.first_order,
        "explicit_since": state.explicit_since,
        "boundary": {
            "chat": plan.chat,
            **walk,
            "last_episode": chat.last_episode if chat is not None else None,
            "tail": list(chat.tail) if chat is not None else [],
            "next_episode": state.episodes,
            "multi_chat": len(set(state.chats) | {plan.chat}) > 1,
        },
        "rolesets": {rs: roleset_wire(row) for rs, row in state.rolesets.items()},
        "voyage_cache": str(voyage_cache),
        "chain": view,
        # the forward gate's rule 3 and the index gate: the group's totals before this session
        "totals": {"messages": state.messages, "episodes": state.episodes},
        "skipped_ids": list(plan.skipped_ids),
    }


def require_extraction_model(meta: Mapping[str, Any], *, allow_parser_change: bool = False) -> None:
    """Refuse an append under another extraction model than the one ``meta`` (the ``BonfireMeta``'s properties) records
    in ``extraction_model_json`` (``BonfireError``), unless ``allow_parser_change`` says to mix parsers knowingly.
    Reads the record; changes nothing."""
    from memory_kernel.grammar.dependencies import parser_change_problem

    if allow_parser_change:
        return
    recorded = json.loads(meta.get("extraction_model_json") or "null") or None
    if problem := parser_change_problem(recorded, what="bonfire"):
        raise BonfireError(problem)
