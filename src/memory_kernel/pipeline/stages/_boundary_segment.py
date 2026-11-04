"""The ``segmenter`` boundary policy: each session is a hard boundary, cut inside by the embedding segmenter.

The ``walk`` policy (``_boundary_walk``) asks Jev at every same-day arrival whether a new topic starts. Under
``segmenter`` there is no Jev call and no ledger reservation: each session (a talk, a document) goes to
``MemoryKernelCentroidTimestampBoundaryService`` in long-form mode (``LongForm``: a length-scaled segment cap, a hard
episode ceiling, no smart mask), which cuts on time gaps, topic-shift peaks and centroid splits over voyage-3 vectors.
The vectors come through the events stage's ``{text: vector}`` cache (``embed.TextVoyageCache``) for exactly the
strings that stage embeds (``embed.message_texts``), so a resume re-embeds nothing and events later hits the cache.

The outputs keep the walk's shape: ``applied-boundaries.json`` (same schema version and keys; rows carry the fields
``walk_states`` and the chain read, with no Jev wire) and ``summary.json`` (with the keys ``ingest.boundary`` reads).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memory_kernel.ingest.episode_boundary import LongForm, MemoryKernelCentroidTimestampBoundaryService
from memory_kernel.pipeline.embed import TextVoyageCache, message_texts, token_est, voyage_usd
from memory_kernel.pipeline.stages._boundary_walk import _utc, conversation_parts

POLICY = "segmenter"
#: Row vocabulary. A session's first message opens an episode (``initial``); a cut inside a session opens one
#: (``segment_cut``, origin = the cut reason: time_gap / shift / centroid / ceiling); every other row continues.
INITIAL, CUT, CONTINUE = "initial", "segment_cut", "continue_active"
SESSION_ORIGIN, CONTINUE_ORIGIN = "session_start", "segment_continue"
STOP_RULES = {"not_applicable": True, "fired": False}


def sessions_of(sources: Sequence[Mapping]) -> list[list[int]]:
    """Source positions grouped into sessions: contiguous runs of one ``session_id``, in source order. A session
    whose messages are not contiguous is refused (a session is one hard boundary)."""
    runs: list[list[int]] = []
    seen: set[str] = set()
    for position, source in enumerate(sources):
        session = source["session_id"]
        if runs and sources[runs[-1][0]]["session_id"] == session:
            runs[-1].append(position)
            continue
        if session in seen:
            raise ValueError(f"source {source['message_id']}: session {session!r} is not contiguous in source order")
        seen.add(session)
        runs.append([position])
    return runs


def plan(sources: Sequence[Mapping], conversations: Mapping[str, str] | None = None) -> dict:
    """No-send plan: nothing is reserved; the Voyage tokens are bounded by every message text embedded once."""
    tokens = sum(token_est(t) for t in dict.fromkeys(message_texts(sources).values()))
    out = {
        "policy": POLICY,
        "arrivals": len(sources),
        "sessions": len(sessions_of(sources)),
        "reservation_usd": "0",
        "voyage_tokens_upper_bound": tokens,
        "voyage_usd_upper_bound": voyage_usd(tokens),
    }
    if len(parts := conversation_parts(sources, conversations)) > 1:
        out["conversations"] = len(parts)
    return out


class _Guarded:
    """The cache as the segmenter's embedder, with the spend stop checked after each batch (Voyage holds nothing
    before a request, so the check follows it; the cache already holds what was paid for)."""

    def __init__(self, cache: TextVoyageCache, unit_guard: Callable[[float], None]) -> None:
        self.cache, self.unit_guard = cache, unit_guard

    def create_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self.cache.create_batch(texts)
        self.unit_guard(voyage_usd(self.cache.tokens))
        return vectors


def _settings(service: MemoryKernelCentroidTimestampBoundaryService) -> dict:
    skip = {"_embedder", "_vector_cache", "last_cuts", "_long_form"}
    out = {k.lstrip("_"): v for k, v in vars(service).items() if k not in skip}
    return {**out, "long_form": asdict(service._long_form or LongForm())}


def _continuation(state: Mapping, sources: Sequence[Mapping], rows: Sequence[Mapping]) -> dict:
    """The chat's state after an appended session: closed at its new message count (no open episode), every episode
    of the session opened, and the walk's ``inversion`` record (the session's first message earlier than the chat's
    last stored one: a session is a hard boundary anyway, so it is only reported)."""
    tail = list(state.get("tail") or ())
    messages = int(state["messages"]) + len(sources)
    inverted = bool(tail) and _utc(sources[0]["timestamp"]) < _utc(tail[-1]["timestamp"])
    return {
        "chat": state["chat"],
        "messages": messages,
        "episode_start": messages,
        "cursor": messages,
        "policy": POLICY,
        "last_episode": rows[-1]["episode_id"],
        "opened": list(dict.fromkeys(r["episode_id"] for r in rows)),
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


def run_segment_boundary(
    sources: list[dict],
    out: Path,
    *,
    cache: TextVoyageCache,
    episode_prefix: str,
    conversations: Mapping[str, str] | None,
    first_episode: int,
    unit_guard: Callable[[float], None],
    continuation: Mapping | None = None,
) -> dict:
    """Cut ``sources`` into episodes under the segmenter policy; write ``out/applied-boundaries.json`` and
    ``out/summary.json`` (``out`` must not exist) and return the summary.

    Episode ids are ``f"{episode_prefix}{first_episode + k}"``, k counting episodes in source order across sessions
    and chats (a full build's first is 0, as the walk's). ``conversations`` (``session_id`` -> chat) only adds the
    multi-conversation row fields (``conversation_id``, ``conversation_index``) the walk writes; a session is in one
    chat, so an episode never spans two. ``unit_guard(spend)`` is called with the run's Voyage spend after each
    embedding batch and raises ``llm.StopSpend`` past the unit cap.

    ``continuation``: a forward append's stored chat state (``bonfire.forward_input``'s ``boundary``; ``sources`` are
    one session of that chat). The applied file then carries ``continuation``, the chat's state after the session as
    the walk's has it (``chat``, ``messages``, ``episode_start``, ``cursor``, ``last_episode``, ``opened``: every
    episode of the session, all new; ``inversion``), closed (``chain.closed_boundary``) and naming the policy."""
    from memory_kernel.construction.memory_semantic_jev import sha
    from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256
    from memory_kernel.pipeline.ledger import write_new

    summary_plan = plan(sources, conversations)
    texts = message_texts(sources)
    sessions = sessions_of(sources)
    # ``boundary_text`` of a message with no speaker is its content alone, and a ``message_texts`` string is already
    # stripped: the segmenter embeds exactly the events stage's string for each message
    batches = [
        [
            {
                "content": texts[sources[p]["message_id"]],
                "timestamp": sources[p]["timestamp"],
                "metadata": {"message_id": sources[p]["message_id"]},
                "position": p,
            }
            for p in session
        ]
        for session in sessions
    ]
    embedder = _Guarded(cache, unit_guard)
    # every text once, up front: the segmenter never embeds a short time segment, and the events stage reads this
    # cache for every message; the segmenter's own lookups are then hits
    embedder.create_batch(list(dict.fromkeys(texts.values())))
    service = MemoryKernelCentroidTimestampBoundaryService(embedder=embedder, long_form=LongForm())
    episodes = asyncio.run(service.segment_message_batches(batches))

    if [m["position"] for episode in episodes for m in episode] != list(range(len(sources))):
        raise ValueError("segmenter: the episodes do not tile the sources in order")
    cuts = {(c["session_index"], c["at"]): c["reason"] for c in service.last_cuts}
    session_index = {p: k for k, session in enumerate(sessions) for p in session}
    session_start = {session[0] for session in sessions}
    starts = {episode[0]["position"] for episode in episodes}
    inner = {(session_index[p], p - sessions[session_index[p]][0]) for p in starts - session_start}
    if inner != set(cuts) or not session_start <= starts:
        raise ValueError(f"segmenter: episode starts {sorted(inner)} differ from its cuts {sorted(cuts)}")

    parts = conversation_parts(sources, conversations)
    local = {p: (c, i) for c, positions in parts.items() for i, p in enumerate(positions)}
    multi = len(parts) > 1
    rows = []
    for k, episode in enumerate(episodes):
        episode_id = f"{episode_prefix}{first_episode + k}"
        for j, message in enumerate(episode):
            p = message["position"]
            if j:
                choice, origin = CONTINUE, CONTINUE_ORIGIN
            elif p in session_start:
                choice, origin = INITIAL, SESSION_ORIGIN
            else:
                choice, origin = CUT, cuts[(session_index[p], p - sessions[session_index[p]][0])]
            conversation, position = local[p]
            row: dict[str, Any] = {
                "index": p,
                "message_id": sources[p]["message_id"],
                "choice": choice,
                "decision_origin": origin,
                "episode_id": episode_id,
                "working_window_start_order": position,  # the row's own position in its chat: no window
                "request_sha256": None,
                "boundary_step_sha256": None,
                "boundary_history_row": None,
            }
            if multi:
                row["conversation_id"] = conversation
                row["conversation_index"] = position
            rows.append(row)

    body = {
        "schema_version": "mk-utc-day-applied-boundaries/v1",
        "source_manifest_sha256": sha(canonical_json(sources).encode()),
        "rows": rows,
        "episode_by_message": {r["message_id"]: r["episode_id"] for r in rows},
        "day_change_is_hard_boundary": False,  # sessions are the hard boundaries; a day change is a time gap or not
        "capacity_rollover_does_not_flush_identity": True,  # no window, so no rollover
    }
    if continuation is not None:
        body["continuation"] = _continuation(continuation, sources, rows)
    applied = {**body, "content_sha256": content_sha256(body)}
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "applied-boundaries.json", applied)
    sizes = [len(e) for e in episodes]
    summary = {
        "policy": POLICY,
        "plan": summary_plan,
        "settings": _settings(service),
        "episodes": len(episodes),
        "largest_episode": max(sizes, default=0),
        "per_session": dict(Counter(sources[e[0]["position"]]["session_id"] for e in episodes)),
        "cut_reasons": dict(sorted(Counter(c["reason"] for c in service.last_cuts).items())),
        "origins": dict(sorted(Counter(f"{r['choice']}/{r['decision_origin']}" for r in rows).items())),
        "voyage_tokens": int(cache.tokens),
        "voyage_usd": voyage_usd(int(cache.tokens)),
        "stop_rules": dict(STOP_RULES),
        "applied_sha256": applied["content_sha256"],
    }
    write_new(out / "summary.json", summary)
    return summary
