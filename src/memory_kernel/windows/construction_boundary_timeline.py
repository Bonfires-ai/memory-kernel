"""Opt-in timed, sequential boundary question over an unchanged complete window."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256
from memory_kernel.windows.construction_window import validate_window_input

SCHEMA = "construction-boundary-question-timed-sequential/v1"
CHOICES = ("continue_active", "shift_topic", "unresolved")


class TimedBoundaryError(ValueError):
    """Source time witnesses or chronological scope are invalid."""


def _utc(value: object, *, message_id: str) -> tuple[str | None, datetime | None]:
    if value is None:
        return None, None
    if not isinstance(value, str) or not value:
        raise TimedBoundaryError(f"{message_id} timestamp must be ISO 8601 or unknown")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TimedBoundaryError(f"{message_id} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TimedBoundaryError(f"{message_id} timestamp must include a timezone")
    utc = parsed.astimezone(UTC)
    return utc.isoformat(), utc


def compile_timed_sequential_boundary_question(
    window_input: Mapping[str, Any],
    *,
    timestamp_witnesses: Mapping[str, Mapping[str, Any]],
    expected_witness_sha256: str,
) -> dict[str, Any]:
    """Expose source message times without turning elapsed time into a boundary rule.

    Timestamps describe message delivery, not the dates of events mentioned in text.
    Every active and incoming message must have a source-bound witness; unknown is
    explicit. Earlier closed topics are not return candidates in this mode.
    """
    validate_window_input(window_input)
    if window_input["boundary_status"] != "classify_boundary":
        raise TimedBoundaryError("sequential boundary requires a nonempty active window")
    if window_input["prior_topics"]:
        raise TimedBoundaryError("sequential mode does not accept prior-topic return context")
    if not isinstance(expected_witness_sha256, str) or content_sha256(timestamp_witnesses) != expected_witness_sha256:
        raise TimedBoundaryError("timestamp witnesses differ from externally pinned digest")
    rows = [*window_input["active_window"]["messages"], window_input["incoming_message"]]
    expected_ids = {row["message_id"] for row in rows}
    if not isinstance(timestamp_witnesses, Mapping) or set(timestamp_witnesses) != expected_ids:
        raise TimedBoundaryError("timestamp witnesses must cover exactly active and incoming messages")
    times = []
    last_known: datetime | None = None
    for row in rows:
        witness = timestamp_witnesses[row["message_id"]]
        if (
            not isinstance(witness, Mapping)
            or set(witness) != {"message_id", "source_text_sha256", "timestamp"}
            or witness["message_id"] != row["message_id"]
            or witness["source_text_sha256"] != row["text_sha256"]
        ):
            raise TimedBoundaryError("timestamp witness differs from exact source message")
        timestamp, parsed = _utc(witness["timestamp"], message_id=row["message_id"])
        if parsed is not None and last_known is not None and parsed < last_known:
            raise TimedBoundaryError("known message timestamps contradict source order")
        if parsed is not None:
            last_known = parsed
        times.append(
            {
                "message_id": row["message_id"],
                "timestamp_utc": timestamp,
                "status": "known" if parsed is not None else "unknown",
            }
        )
    prior_time = times[-2]["timestamp_utc"]
    incoming_time = times[-1]["timestamp_utc"]
    if prior_time is None or incoming_time is None:
        gap = {
            "status": "unknown",
            "seconds": None,
            "from_message_id": rows[-2]["message_id"],
            "to_message_id": rows[-1]["message_id"],
        }
    else:
        seconds = (datetime.fromisoformat(incoming_time) - datetime.fromisoformat(prior_time)).total_seconds()
        gap = {
            "status": "known",
            "seconds": seconds,
            "from_message_id": rows[-2]["message_id"],
            "to_message_id": rows[-1]["message_id"],
        }
    state = {
        "window": deepcopy(dict(window_input)),
        "message_times": times,
        "incoming_elapsed_gap": gap,
        "timestamp_witness_sha256": expected_witness_sha256,
        "time_semantics": "message_timestamps_not_event_dates",
    }
    instructions = (
        "Decide whether the incoming message continues the complete active topic or starts a new episode. "
        "Read state.window.active_window.messages in full and state.window.incoming_message separately. "
        "State.message_times are source message timestamps in UTC; state.incoming_elapsed_gap.seconds "
        "is the elapsed time from the last active message, or unknown. A long gap alone does not require "
        "a new episode, and a short gap alone does not require continuation. These timestamps are not "
        "dates of events described in the text. Choose unresolved when the source does not settle the boundary."
    )
    criteria = {}
    time_sha = content_sha256(times)
    for choice in CHOICES:
        criteria[choice] = canonical_json(
            {
                "choice": choice,
                "window_input_sha256": window_input["content_sha256"],
                "message_times_sha256": time_sha,
                "elapsed_gap_status": gap["status"],
                "episode_action": "start_new_episode"
                if choice == "shift_topic"
                else ("keep_active_episode" if choice == "continue_active" else "defer_boundary"),
                "decision_policy": "source_semantics_with_time_as_context_not_cutoff",
            }
        )
    body = {
        "schema": SCHEMA,
        "state": state,
        "wire_question": {"type": "choice", "instructions": instructions, "criteria": criteria},
    }
    return {**body, "content_sha256": content_sha256(body)}


__all__ = ["CHOICES", "SCHEMA", "TimedBoundaryError", "compile_timed_sequential_boundary_question"]
