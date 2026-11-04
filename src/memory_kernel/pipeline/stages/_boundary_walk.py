"""Boundary-only episode driver (hard UTC-day cut + Jev topic shift), with a replay-equivalence gate.

Ported from a reference-harness module (unreached top-level definitions pruned). Differences: plan / run_boundary take `episode_prefix` (was a rebound module constant), every wire builder takes `window_prefix`, replay_equivalence takes the constants of the archived run (scope, episode and window prefix; they were module globals) and run_boundary a ledger.Shard; the frozen boundary compiler is _boundary_frozen; the key loader is gone.

Mirrors the boundary decision of the harness's day runner (arrival 0 opens episode 0; a known UTC-date
change is a hard ``shift_topic`` with no call; a later part of a split message continues the episode with no call (it
is not an arrival of its own, so the stop-rule shares leave it out); otherwise the frozen ``compile_boundary_step`` wire goes to Jev through a
ledger shard ($2 cap up to the reference conversation's size); ``unresolved`` keeps the episode open; capacity rollover never cuts). It does NOT score statements.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from memory_kernel.construction.memory_jev_wire_ids import MAX_WIRE_BYTES
from memory_kernel.pipeline.ledger import Shard
from memory_kernel.pipeline.llm import cap_scale
from memory_kernel.pipeline.stages._boundary_frozen import (
    BOUNDARY_ID,
    INPUT_TOKEN_MARGIN,
    FullIngestError,
    _interaction_boundary_question,
    apply_boundary_response,
    compile_boundary_step,
)

#: At the reference conversation's size; ``stage_cap`` scales it. Recorded, never a stop (``run_boundary``).
STAGE_CAP_USD = Decimal("0.40")

#: The input cap a boundary wire the ledger sends fits under: the frozen compiler bounds a wire at its bytes +
#: ``INPUT_TOKEN_MARGIN`` <= its input cap (32,768, up to 32,704 bytes), the ledger refuses a wire over
#: ``MAX_WIRE_BYTES`` (31,744) before sending it. ``boundary_wire`` recompiles an over-limit step under this cap.
WIRE_INPUT_CAP = MAX_WIRE_BYTES + INPUT_TOKEN_MARGIN


def stage_cap(messages: int) -> Decimal:
    """The boundary stage cap for a conversation of ``messages``: exactly ``STAGE_CAP_USD`` up to the reference
    conversation's size, proportional above it (``llm.cap_scale``)."""
    scale = cap_scale(messages)
    return STAGE_CAP_USD if scale <= 1 else STAGE_CAP_USD * Decimal(str(scale))


JEV_INPUT_PER_MILLION = Decimal("0.042")

# Pre-registered stop rules: a miss is reported, never re-tuned. They stop the walk only under ``strict`` (the build's
# ``--strict-gates``); otherwise they are recorded metrics (the stage's ``boundary`` gate is advisory).
MAX_UNRESOLVED_SHARE = 0.05

DEFAULT_MAX_SHIFT_SHARE = 0.10  # the build's ``--max-shift-share`` default

PROFILE = "stock-pb-memory-canary"  # the frozen replay profile

HARD_ORIGIN = "hard_utc_calendar_date_change"
PART_ORIGIN = "same_message_later_part"  # part k >= 2 of a split message continues the episode, no call
#: A forward append's session whose first message is EARLIER than its chat's last stored message: appended
#: in arrival order, never re-ordered: it opens a new episode with no call, and the walk's end state records it.
INVERSION_ORIGIN = "session_before_last_stored_message"
#: A same-day arrival whose step cannot fit at all (``window_capacity_pending``: not even the prior message plus the
#: incoming one, with the episode's archived source ids, fits the wire cap): it opens a new episode with no call, as
#: a hard day cut does, so the next window starts afresh. The frozen compiler refuses such a step, so the walk
#: before this cut stopped there.
CAPACITY_ORIGIN = "window_capacity_cut"


def capacity_cuts(rows: Sequence[Mapping]) -> int:
    """The walk's capacity cuts (``CAPACITY_ORIGIN``)."""
    return sum(r["decision_origin"] == CAPACITY_ORIGIN for r in rows)


def later_part(previous_id: str, message_id: str) -> bool:
    """``message_id`` is ``<base>#k`` (k >= 2) and ``previous_id`` is ``<base>#(k-1)``: a later part of the message the
    splitter cut (input ids without such a pair never match)."""
    base, sep, k = message_id.rpartition("#")
    return bool(sep) and k.isdigit() and int(k) >= 2 and previous_id == f"{base}#{int(k) - 1}"


class StopRule(RuntimeError):
    """A pre-registered stop rule fired; the caller reports the numbers and asks."""


def _utc(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.utcoffset() != timedelta(0):
        raise ValueError("source timestamp must be pinned in UTC")
    return stamp


def hard_day_change(prev_ts: str | None, cur_ts: str | None) -> bool:
    """Same rule as the harness's day-boundary module: both stamps known and the UTC date differs."""
    if prev_ts is None or cur_ts is None:
        return False
    prev, cur = _utc(prev_ts), _utc(cur_ts)
    if cur < prev:
        raise ValueError("source timestamps reverse chronological order")
    return cur.date() != prev.date()


def timestamp_rows(sources: Sequence[Mapping]) -> dict[str, dict]:
    return {
        s["message_id"]: {"message_id": s["message_id"], "text": s["text"], "timestamp": s["timestamp"]}
        for s in sources
    }


def messages_view(sources: Sequence[Mapping]) -> list[dict]:
    """The exact six-field message rows the frozen window compiler accepts (``statement_id`` = ``message_id``)."""
    return [
        {
            "message_id": s["message_id"],
            "order": s["order"],
            "speaker": s["speaker"],
            "statement_id": s["message_id"],
            "text": s["text"],
            "text_sha256": s["text_sha256"],
        }
        for s in sources
    ]


def boundary_wire(
    sources: Sequence[Mapping],
    timestamps: Mapping[str, Mapping],
    *,
    index: int,
    history: Sequence[Mapping],
    previous_cursor: int,
    scope: Mapping,
    window_prefix: str,
    step_schema: str | None = None,
    offset: int = 0,
    fit: bool = True,
) -> dict:
    """The frozen same-day boundary step (wire, request_sha256, working window) for one arrival. ``step_schema`` None
    keeps the package's step id; a replay of an archived run passes that run's id (its scope file). ``offset``: the
    walk position of ``sources[0]`` (a continued walk, ``continue_walk``).

    ``fit``: a step whose wire is over the ledger's ``MAX_WIRE_BYTES`` (refused when sent) is compiled again under
    ``WIRE_INPUT_CAP``: the frozen compiler's own rule (the longest contiguous recent suffix of the episode that fits)
    then keeps the MOST RECENT messages of the window that fit; the earlier ones are its archived part, as in any
    capacity rollover. Deterministic, and such a step carries ``window_trimmed``. A step within the limit is the
    frozen step, unchanged. ``fit=False`` (the replay of an archived run) compiles under the frozen cap only."""
    args = dict(
        scope=scope,
        index=index,
        history=history,
        previous_cursor=previous_cursor,
        window_prefix=window_prefix,
        offset=offset,
        **({"step_schema": step_schema} if step_schema is not None else {}),
    )
    rows = messages_view(sources)
    step = compile_boundary_step(rows, timestamps, **args)
    if not fit or step["input_tokens_upper_bound"] <= WIRE_INPUT_CAP:
        return step
    return {**compile_boundary_step(rows, timestamps, input_cap=WIRE_INPUT_CAP, **args), "window_trimmed": True}


def reservation_usd(wire: Mapping) -> Decimal:
    """The ledger's pessimistic hold for one Jev request (a byte per token plus framing)."""
    from memory_kernel.construction.mined_affordance_contract import canonical_json

    return Decimal(len(canonical_json(wire).encode()) + 4096) * JEV_INPUT_PER_MILLION / Decimal(1_000_000)


Decide = Callable[[dict, list], dict]


def window_scope(sources: Sequence[Mapping]) -> dict:
    """The window compiler's scope: the sources' bonfire and conversation plus the frozen replay profile."""
    return {
        "bonfire_id": sources[0]["scope"]["bonfire_id"],
        "profile": PROFILE,
        "conversation_id": sources[0]["scope"]["conversation_id"],
    }


def conversation_parts(sources: Sequence[Mapping], conversations: Mapping[str, str] | None) -> dict[str, list[int]]:
    """``{conversation id: [source position, ...]}`` in source order. ``conversations`` maps a source's ``session_id``
    to its conversation (``_conversation.session_conversations``). None puts every source in one conversation."""
    if conversations is None:
        return {"": list(range(len(sources)))}
    parts: dict[str, list[int]] = {}
    for position, source in enumerate(sources):
        session = source["session_id"]
        if session not in conversations:
            raise ValueError(f"source {source['message_id']}: session {session!r} has no conversation")
        parts.setdefault(conversations[session], []).append(position)
    return parts


def walk(
    sources: Sequence[Mapping],
    decide: Decide,
    *,
    episode_prefix: str,
    window_prefix: str,
    scope: Mapping | None = None,
    step_schema: str | None = None,
    conversations: Mapping[str, str] | None = None,
    continuation: Mapping | None = None,
    fit: bool = True,
) -> dict:
    """Walk the arrivals. ``decide(step, history)`` returns the boundary history row for a same-day arrival.
    ``fit``: over-limit wires keep the most recent messages that fit (``boundary_wire``).

    ``continuation`` (a forward append: ``sources`` are one session of one chat) continues that chat's walk from the
    state the bonfire stored (``continue_walk``); ``conversations`` is then not used.

    ``conversations`` (``session_id`` -> conversation id) runs one independent walk per conversation (chat), over that
    conversation's sources in source order. Each walk has its own history, cursor, indices and day-change rule, and
    its first message opens an episode with no call. Each walk sees a copy of its sources with ``order`` renumbered
    0..n-1, because the window compiler requires a contiguous order. So a step's ``index``, its window ids and its
    ``working_window_start_order`` count positions within the conversation, and a wire carries only that
    conversation's messages. Episode ids stay unique: one counter numbers the episodes of all walks in the source
    order of their first message, so an episode never spans two conversations. A row's ``index`` is its global
    source position, and multi-conversation rows add ``conversation_id`` and ``conversation_index``.

    With one conversation (None, or every session mapped to the same id), the single walk runs over ``sources``
    itself, byte-identical to the walk before conversations existed."""
    scope = dict(scope) if scope is not None else window_scope(sources)
    if continuation is not None:
        return continue_walk(
            sources,
            decide,
            continuation,
            episode_prefix=episode_prefix,
            window_prefix=window_prefix,
            scope=scope,
            step_schema=step_schema,
            fit=fit,
        )
    parts = conversation_parts(sources, conversations)
    if len(parts) == 1:
        rows = _walk_one(sources, decide, scope=scope, window_prefix=window_prefix, step_schema=step_schema, fit=fit)
        out = [_row(r, f"{episode_prefix}{r['episode']}", None, None) for r in rows]
        episodes = rows[-1]["episode"] + 1 if rows else 1
        return {
            "rows": out,
            "episode_by_message": {r["message_id"]: r["episode_id"] for r in out},
            "episodes": episodes,
        }
    walked: list[tuple[int, str, dict]] = []  # (global position, conversation id, local row)
    for conversation, positions in parts.items():
        local = [{**sources[g], "order": i} for i, g in enumerate(positions)]  # copies: ``sources`` stays untouched
        rows = _walk_one(local, decide, scope=scope, window_prefix=window_prefix, step_schema=step_schema, fit=fit)
        walked.extend((positions[r["index"]], conversation, r) for r in rows)
    walked.sort(key=lambda w: w[0])
    numbers: dict[tuple[str, int], int] = {}
    for _, conversation, r in walked:
        numbers.setdefault((conversation, r["episode"]), len(numbers))
    out = [_row(r, f"{episode_prefix}{numbers[(c, r['episode'])]}", g, c) for g, c, r in walked]
    return {
        "rows": out,
        "episode_by_message": {r["message_id"]: r["episode_id"] for r in out},
        "episodes": len(numbers),
    }


def continue_walk(
    sources: Sequence[Mapping],
    decide: Decide,
    state: Mapping,
    *,
    episode_prefix: str,
    window_prefix: str,
    scope: Mapping,
    step_schema: str | None = None,
    fit: bool = True,
) -> dict:
    """One chat's walk continued over one new session (a forward append), exactly as the whole walk would go on.

    ``state`` is the chat's stored walk (``bonfire.forward_input``): ``chat``; ``messages`` (n, the chat's messages
    so far: the session's first message is walk position n); ``episode_start`` (the walk position where the open
    episode began) and ``cursor`` (the last working window start); ``last_episode`` (the open episode's id; None for a
    new chat); ``tail`` (the chat's last source rows, reaching back at least to ``episode_start``: the boundary window
    and its archived part never look further); ``next_episode`` (the bonfire's episode count: the number of the next
    new episode, as the whole walk numbers episodes across chats in order of their first message); and
    ``multi_chat`` (the bonfire has more than one chat: rows then carry the multi-conversation fields, the global
    ``index`` being the source ``order``).

    The rows and wires are the whole walk's for these messages: the window rows carry walk positions, the history
    holds the open episode's start, and the window and archived part read only the tail and the session. A first
    message EARLIER than the tail's last one (``INVERSION_ORIGIN``) opens a new episode with no call (the whole walk
    would refuse it).

    Returns ``walk``'s dict plus ``continuation``: the chat's walk after the session (``chat``, ``messages``,
    ``episode_start``, ``cursor``, ``last_episode``, ``opened``: the new episode ids in order, ``inversion``)."""
    n, start, cursor = int(state["messages"]), int(state["episode_start"]), int(state["cursor"])
    tail = list(state["tail"])
    if n:
        if (
            not tail
            or len(tail) > n
            or not n - len(tail) <= start < n
            or not 0 <= cursor < n
            or not state["last_episode"]
        ):
            raise ValueError(
                f"chat {state['chat']!r}: the stored walk (messages {n}, episode start {start}, cursor {cursor}, "
                f"{len(tail)} tail rows) cannot be continued"
            )
    elif tail or start or cursor:
        raise ValueError(f"chat {state['chat']!r}: a new chat's walk starts empty")
    offset = n - len(tail)
    local = [{**r, "order": offset + j} for j, r in enumerate([*tail, *sources])]  # walk positions, as the window wants
    inverted = bool(n) and _utc(sources[0]["timestamp"]) < _utc(tail[-1]["timestamp"])
    history = [{"index": p, "choice": "shift_topic" if p == start else "continue_active"} for p in range(1, n)]
    rows = _walk_from(
        local,
        decide,
        scope=dict(scope),
        window_prefix=window_prefix,
        step_schema=step_schema,
        offset=offset,
        first=n,
        history=history,
        cursor=cursor,
        inverted=inverted,
        fit=fit,
    )
    first_new = int(state["next_episode"]) - (1 if n else 0)  # the open episode is local episode 0 of a stored chat

    def episode_id(episode: int) -> str:
        return str(state["last_episode"]) if n and episode == 0 else f"{episode_prefix}{first_new + episode}"

    multi = bool(state["multi_chat"])
    out = [
        _row(
            r,
            episode_id(r["episode"]),
            sources[r["index"] - n]["order"] if multi else None,
            state["chat"] if multi else None,
        )
        for r in rows
    ]
    shifts = [r["index"] for r in rows if r["choice"] in {"shift_topic", "initial"}]
    opened = list(dict.fromkeys(r["episode_id"] for r in out if not (n and r["episode_id"] == state["last_episode"])))
    end = {
        "chat": state["chat"],
        "messages": n + len(sources),
        "episode_start": shifts[-1] if shifts else start,
        "cursor": rows[-1]["working_window_start_order"],
        "last_episode": out[-1]["episode_id"],
        "opened": opened,
        "inversion": (
            {
                "message_id": sources[0]["message_id"],
                "timestamp": sources[0]["timestamp"],
                "after": tail[-1]["timestamp"],
            }
            if inverted
            else None
        ),
    }
    return {
        "rows": out,
        "episode_by_message": {r["message_id"]: r["episode_id"] for r in out},
        "episodes": len({r["episode_id"] for r in out}),
        "continuation": end,
    }


def walk_states(rows: Sequence[Mapping], conversation_of: Mapping[str, str]) -> dict[str, dict]:
    """``{chat: {messages, episode_start, cursor}}``: each chat's walk after a full build's boundary rows (``walk``'s
    ``rows`` in source order; ``conversation_of``: message id -> chat), what ``continue_walk`` continues from. A
    ``segmenter`` row's ``segment_cut`` opens an episode too (walk rows never carry it)."""
    out: dict[str, dict] = {}
    for row in rows:
        state = out.setdefault(conversation_of[row["message_id"]], {"messages": 0, "episode_start": 0, "cursor": 0})
        if row["choice"] in {"shift_topic", "initial", "segment_cut"}:
            state["episode_start"] = state["messages"]
        state["cursor"] = int(row["working_window_start_order"])
        state["messages"] += 1
    return out


def _row(r: dict, episode_id: str, position: int | None, conversation: str | None) -> dict:
    """The output row of local walk row ``r``. ``position``/``conversation`` None: the one-conversation walk."""
    row = {
        "index": r["index"] if position is None else position,
        "message_id": r["message_id"],
        "choice": r["choice"],
        "decision_origin": r["decision_origin"],
        "episode_id": episode_id,
        "working_window_start_order": r["working_window_start_order"],
        "request_sha256": r["request_sha256"],
        "boundary_step_sha256": r["boundary_step_sha256"],
        "boundary_history_row": r["boundary_history_row"],
    }
    if conversation is not None:
        row["conversation_id"] = conversation
        row["conversation_index"] = r["index"]
    return row


def _walk_one(
    sources: Sequence[Mapping],
    decide: Decide,
    *,
    scope: dict,
    window_prefix: str,
    step_schema: str | None,
    fit: bool = True,
) -> list[dict]:
    """One sequential walk over ``sources`` (one conversation); each row's ``episode`` is the walk's own counter."""
    return _walk_from(sources, decide, scope=scope, window_prefix=window_prefix, step_schema=step_schema, fit=fit)


def _walk_from(
    sources: Sequence[Mapping],
    decide: Decide,
    *,
    scope: dict,
    window_prefix: str,
    step_schema: str | None,
    offset: int = 0,
    first: int = 0,
    history: Sequence[Mapping] = (),
    cursor: int = 0,
    inverted: bool = False,
    fit: bool = True,
) -> list[dict]:
    """The walk from position ``first`` on (0: the whole walk, ``_walk_one``). ``sources[0]`` is walk position
    ``offset``; ``history`` and ``cursor`` are the walk's state before ``first`` (``continue_walk``); ``inverted``:
    the message at ``first`` is earlier than the one before it (``INVERSION_ORIGIN``). Each row's ``episode`` counts
    the episodes the walk opened from ``first`` on."""
    timestamps = timestamp_rows(sources)
    history = list(history)
    episode, rows = 0, []
    for index in range(first, offset + len(sources)):
        message = sources[index - offset]
        step = None
        if index == 0:
            choice, origin = "initial", "initial"
        elif inverted and index == first:
            choice, origin = "shift_topic", INVERSION_ORIGIN
            history = [
                *history,
                {"index": index, "message_id": message["message_id"], "choice": choice, "origin": origin},
            ]
            cursor = index
        elif hard_day_change(sources[index - offset - 1]["timestamp"], message["timestamp"]):
            choice, origin = "shift_topic", HARD_ORIGIN
            history = [
                *history,
                {"index": index, "message_id": message["message_id"], "choice": choice, "origin": origin},
            ]
            cursor = index
        elif later_part(sources[index - offset - 1]["message_id"], message["message_id"]):
            choice, origin = "continue_active", PART_ORIGIN
            history = [
                *history,
                {"index": index, "message_id": message["message_id"], "choice": choice, "origin": origin},
            ]
        else:
            try:
                step = boundary_wire(
                    sources,
                    timestamps,
                    index=index,
                    history=history,
                    previous_cursor=cursor,
                    scope=scope,
                    window_prefix=window_prefix,
                    step_schema=step_schema,
                    offset=offset,
                    fit=fit,
                )
            except FullIngestError as exc:
                if not str(exc).startswith("window_capacity_pending"):
                    raise
                step = None
            if step is None:  # nothing fits: a capacity cut (CAPACITY_ORIGIN), no call
                choice, origin = "shift_topic", CAPACITY_ORIGIN
                history = [
                    *history,
                    {"index": index, "message_id": message["message_id"], "choice": choice, "origin": origin},
                ]
                cursor = index
            else:
                row = decide(step, history)
                if row["index"] != index or row["message_id"] != message["message_id"]:
                    raise ValueError("decision row does not belong to this arrival")
                history = [*history, row]
                choice, origin = row["choice"], "jev_finite_choice"
                cursor = step["working_window_start_order"]
        if choice == "shift_topic":
            episode += 1
        rows.append(
            {
                "index": index,
                "message_id": message["message_id"],
                "choice": choice,
                "decision_origin": origin,
                "episode": episode,
                "working_window_start_order": cursor,
                "request_sha256": step["request_sha256"] if step else None,
                "boundary_step_sha256": step["content_sha256"] if step else None,
                "boundary_history_row": history[-1] if index else None,
            }
        )
    return rows


def _all_continue(step: dict, history: list) -> dict:
    return {"index": step["index"], "message_id": step["message_id"], "choice": "continue_active"}


def plan(
    sources: Sequence[Mapping],
    *,
    episode_prefix: str,
    window_prefix: str,
    cap: Decimal = STAGE_CAP_USD,
    conversations: Mapping[str, str] | None = None,
    continuation: Mapping | None = None,
) -> dict:
    """No-send plan on the all-continue path (the real path depends on Jev's answers; hard cuts are exact).

    ``conversations`` / ``continuation``: as for ``walk``. With more than one conversation the plan adds
    ``conversations`` (their count); a one-conversation plan is unchanged. Its wires are the frozen compiler's,
    untrimmed (``fit=False``): the plan's figures are what they were, an advisory all-continue estimate (its
    ``wires_within_limit`` may be false where the live walk trims a window, ``run_boundary``). ``cap`` is recorded,
    never a stop: a reservation over it adds ``stage_cap_exceeded``."""
    from memory_kernel.construction.mined_affordance_contract import canonical_json

    holds: list[Decimal] = []
    sizes: list[int] = []

    def decide(step: dict, history: list) -> dict:
        holds.append(reservation_usd(step["wire"]))
        sizes.append(len(canonical_json(step["wire"]).encode()))
        return _all_continue(step, history)

    result = walk(
        sources,
        decide,
        episode_prefix=episode_prefix,
        window_prefix=window_prefix,
        conversations=conversations,
        continuation=continuation,
        fit=False,  # today's plan figures: the plan is sealed in summary.json
    )
    hard = sum(r["decision_origin"] == HARD_ORIGIN for r in result["rows"])
    total = sum(holds, Decimal(0))
    summary = {
        "arrivals": len(sources),
        "hard_day_cuts": hard,
        "same_day_requests": len(holds),
        "reservation_usd": str(total.quantize(Decimal("0.000001"))),
        "mean_hold_usd": str((total / len(holds)).quantize(Decimal("0.000001"))) if holds else "0",
        "max_wire_bytes": max(sizes, default=0),
        "wire_byte_limit": MAX_WIRE_BYTES,
        "wires_within_limit": max(sizes, default=0) <= MAX_WIRE_BYTES,
        "stage_cap_usd": str(cap),
        "within_cap": total <= cap,
        "episodes_if_all_continue": result["episodes"],
    }
    if total > cap:
        summary["stage_cap_exceeded"] = True
    if cuts := capacity_cuts(result["rows"]):
        summary["capacity_cuts"] = cuts
    if len(parts := conversation_parts(sources, conversations)) > 1:
        summary["conversations"] = len(parts)
    return summary


def _combined_contains(attempts: Path, step: dict) -> bool:
    """The archived combined arrival wire carries this step's boundary window state and (interaction-amended) question."""
    state = step["wire"]["state"]
    question = _interaction_boundary_question(step)
    for path in sorted(attempts.glob(f"{step['index']:04d}-arrival-*-attempt0.json")):
        wire = json.loads(path.read_bytes())["wire"]
        if BOUNDARY_ID in wire["questions"]:
            return wire["questions"][BOUNDARY_ID] == question and all(
                key in wire["state"] and wire["state"][key] == value for key, value in state.items()
            )
    return False


def replay_equivalence(
    archived: Path,
    sources: list[dict],
    *,
    scope: Mapping,
    episode_prefix: str,
    window_prefix: str,
    step_schema: str | None = None,
) -> dict:
    """$0: recompile every same-day wire of an archived run from its sources and compare it to the archived decision
    row. ``scope``, ``episode_prefix`` and ``window_prefix`` are the constants the archived run was made with (the
    ``--boundary-replay-scope`` file): they are part of the wires being compared. ``step_schema`` (optional in that
    file) is the step id the archived run sealed its steps with: the archived step digests cover it. The wires are
    compiled as the archived run compiled them: ``fit=False``, no wire-limit trimming (``boundary_wire``)."""
    doc = json.loads(archived.read_bytes())
    old = {r["index"]: r for r in doc["rows"]}
    attempts = archived.parent / "attempts"
    checks: list[tuple[bool, str]] = []

    def decide(step: dict, history: list) -> dict:
        row = old[step["index"]]
        want = row["boundary_history_row"]
        same_window = want["working_window_start_order"] == step["working_window_start_order"]
        if row["reused_score_count"] > 0:  # cached_boundary: the standalone boundary wire was sent as is
            ok = (
                same_window
                and want["request_sha256"] == step["request_sha256"]
                and want["step_sha256"] == step["content_sha256"]
            )
            mode = "standalone_wire_digest"
        else:  # new_combined: boundary + class questions went in one wire; compare the boundary parts inside it
            ok = same_window and _combined_contains(attempts, step)
            mode = "combined_wire_boundary_state_and_question"
        checks.append((ok, mode))
        return want

    result = walk(
        sources,
        decide,
        episode_prefix=episode_prefix,
        window_prefix=window_prefix,
        scope=scope,
        step_schema=step_schema,
        fit=False,  # the archived run's wires, compiled as it compiled them
    )
    hard_new = {r["index"] for r in result["rows"] if r["decision_origin"] == HARD_ORIGIN}
    hard_old = {i for i, r in old.items() if r["decision_origin"] == HARD_ORIGIN}
    return {
        "total": len(checks),
        "match": sum(ok for ok, _ in checks),
        "modes": dict(Counter(mode for _, mode in checks)),
        "hard_cuts_match": len(hard_new & hard_old) if hard_new == hard_old else -1,
        "episodes_match": result["episode_by_message"] == doc["episode_by_message"],
        "arrivals": len(result["rows"]),
    }


#: A forward append's session is small: its shift-share rule fires only over at least this many same-day steps.
APPEND_MIN_SHIFT_STEPS = 10


def check_stop_rules(
    rows: Sequence[Mapping],
    denominator: int,
    *,
    max_shift_share: float = DEFAULT_MAX_SHIFT_SHARE,
    min_shift_steps: int = 0,
) -> dict:
    """Shares are of ALL same-day arrivals; the hard cuts are deterministic, so the denominator is fixed up front.
    With no same-day arrival (a one-message session, a session of hard cuts and later parts) the shares are 0.0.
    ``min_shift_steps``: the shift-share rule fires only over at least that many same-day arrivals (an append's
    session, ``APPEND_MIN_SHIFT_STEPS``; 0 = always, a full build).

    Counts only grow, so a share already above its limit of the full denominator is certain to stay above it."""
    counts = Counter(r["choice"] for r in rows if r["decision_origin"] == "jev_finite_choice")
    out = {
        "same_day_arrivals": denominator,
        "answered": sum(counts.values()),
        "unresolved": counts["unresolved"],
        "shift_topic": counts["shift_topic"],
        "unresolved_share": counts["unresolved"] / denominator if denominator else 0.0,
        "shift_share": counts["shift_topic"] / denominator if denominator else 0.0,
    }
    shifts = out["shift_share"] > max_shift_share and denominator >= min_shift_steps
    out["fired"] = out["unresolved_share"] > MAX_UNRESOLVED_SHARE or shifts
    return out


def run_boundary(
    sources: list[dict],
    out: Path,
    *,
    shard: Shard,
    episode_prefix: str,
    window_prefix: str,
    live: bool,
    plan_only: bool,
    usage: dict | None = None,
    sender: Callable | None = None,
    cap: Decimal = STAGE_CAP_USD,
    max_shift_share: float = DEFAULT_MAX_SHIFT_SHARE,
    strict: bool = False,
    conversations: Mapping[str, str] | None = None,
    continuation: Mapping | None = None,
) -> dict:
    """``shard`` is a ``ledger.Shard`` over a ``make_execution`` directory (its sealed round cap). Nothing is sent unless ``live``.

    ``usage`` is filled as requests complete, so a stop rule or crash still leaves the spend with the caller. The stop
    rules end the walk early only with ``strict``; otherwise their verdict is the summary's ``stop_rules``. The plan's
    wire-size check is advisory (``wire_check``): a window of the walk whose wire would pass the ledger's byte limit keeps
    its most recent messages that fit (``boundary_wire``; the summary's ``window_trimmed`` counts the sent steps it
    trimmed, present when > 0; the plan is not trimmed).
    ``cap`` (the stage cap) is recorded, never a stop: a plan reservation or a spend past it adds
    ``stage_cap_exceeded`` to the summary and the walk goes on; ``--cap-usd`` is the spend stop (the caller's
    ``shard`` guard, ``llm.unit_guard``).
    ``conversations`` (``session_id`` -> conversation id) walks each conversation apart (``walk``); ``continuation``
    continues one chat's stored walk over a forward append's session (``continue_walk``): the applied file then
    carries the chat's walk after it (``continuation``), what the chain stores for the next append. In an append the
    stop rules are advisory only (``strict`` is ignored) and the shift share needs ``APPEND_MIN_SHIFT_STEPS``."""
    from memory_kernel.construction.memory_semantic_jev import sha
    from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256
    from memory_kernel.pipeline.ledger import write_new

    summary = plan(
        sources,
        episode_prefix=episode_prefix,
        window_prefix=window_prefix,
        cap=cap,
        conversations=conversations,
        continuation=continuation,
    )
    if plan_only or not live:
        return {"plan": summary}
    summary["wire_check"] = "advisory" if not summary["wires_within_limit"] else "ok"
    usage = usage if usage is not None else {}
    usage.update({"requests": 0, "replayed": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
    manifest, provenance, _ = shard.load()
    budget = shard.open(manifest, provenance)
    out.mkdir(parents=True, exist_ok=False)
    denominator = summary["same_day_requests"]
    strict = strict and continuation is None  # an append's stop rules are recorded, never a stop
    rules = {"max_shift_share": max_shift_share, "min_shift_steps": APPEND_MIN_SHIFT_STEPS if continuation else 0}
    seen: list[dict] = []
    trimmed: list[int] = []

    def decide(step: dict, history: list) -> dict:
        if step.get("window_trimmed"):
            trimmed.append(step["index"])
        sent_before = (budget.directory / ("jev-" + content_sha256(step["wire"]))).exists()
        archive = budget.evaluate(step["wire"], sender=sender)
        raw = archive["raw"].encode()
        receipt = json.loads(archive["receipt"])["usage"]
        usage["replayed" if sent_before else "requests"] += 1
        if not sent_before:
            usage["input_tokens"] += receipt["input_tokens"]
            usage["output_tokens"] += receipt["output_tokens"]
            usage["usd"] += float(receipt["cost_usd"])
        row = apply_boundary_response(step, json.loads(raw), raw_response_sha256=sha(raw), history=history)[-1]
        seen.append({"decision_origin": "jev_finite_choice", "choice": row["choice"]})
        verdict = check_stop_rules(seen, denominator, **rules)
        if strict and verdict["fired"]:
            raise StopRule("stop rule fired: " + json.dumps(verdict, sort_keys=True))
        return row

    try:
        try:
            result = walk(
                sources,
                decide,
                episode_prefix=episode_prefix,
                window_prefix=window_prefix,
                conversations=conversations,
                continuation=continuation,
            )
        except BaseException as exc:
            write_new(
                out / "stopped.json",
                {
                    "reason": f"{type(exc).__name__}: {exc}",
                    "answered": len(seen),
                    "choices": dict(Counter(r["choice"] for r in seen)),
                    "usage": usage,
                },
            )
            raise
        spend = budget.summary()
    finally:
        budget.close()
    verdict = check_stop_rules(result["rows"], denominator, **rules)
    body = {
        "schema_version": "mk-utc-day-applied-boundaries/v1",
        "source_manifest_sha256": sha(canonical_json(sources).encode()),
        "rows": result["rows"],
        "episode_by_message": result["episode_by_message"],
        "day_change_is_hard_boundary": True,
        "capacity_rollover_does_not_flush_identity": True,
        **({"continuation": result["continuation"]} if continuation is not None else {}),
    }
    applied = {**body, "content_sha256": content_sha256(body)}
    write_new(out / "applied-boundaries.json", applied)
    origins = Counter(f"{r['choice']}/{r['decision_origin']}" for r in result["rows"])
    run_summary = {
        "plan": summary,
        "usage": usage,
        "stop_rules": verdict,
        "episodes": result["episodes"],
        "origins": dict(sorted(origins.items())),
        "budget": spend,
        "applied_sha256": applied["content_sha256"],
    }
    if trimmed:
        run_summary["window_trimmed"] = len(trimmed)
    if cuts := capacity_cuts(result["rows"]):
        run_summary["capacity_cuts"] = cuts
    if not summary["within_cap"] or Decimal(str(usage["usd"])) > cap:
        run_summary["stage_cap_exceeded"] = True
    write_new(out / "summary.json", run_summary)
    return run_summary
