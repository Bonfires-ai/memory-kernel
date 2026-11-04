"""Pure bounded source contract for whole-window boundary decisions.

The active window is the complete source-ordered prefix before the incoming message.
No database, prompt selection, or source truncation; oversized windows fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from memory_kernel.construction.decision_contract import source_scope_from_dict
from memory_kernel.construction.mined_affordance_contract import content_sha256, source_sha256

WINDOW_INPUT_SCHEMA = "construction-window-input/v1"
MAX_WINDOW_CHARS = 128_000

_MESSAGE_FIELDS = {"message_id", "statement_id", "order", "speaker", "text", "text_sha256"}
_SCOPE_FIELDS = {"bonfire_id", "profile", "conversation_id"}
_STATE_FIELDS = {"candidate_ids", "active_event_ids", "pending_decision_ids"}


class ConstructionWindowError(ValueError):
    """A source window is incomplete, malformed, or exceeds its explicit bound."""


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConstructionWindowError(f"{label} must be a nonempty string")
    return value


def _message(row: object, label: str) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != _MESSAGE_FIELDS:
        raise ConstructionWindowError(f"{label} must contain exactly {sorted(_MESSAGE_FIELDS)}")
    message_id = _nonempty_string(row["message_id"], f"{label}.message_id")
    statement_id = _nonempty_string(row["statement_id"], f"{label}.statement_id")
    speaker = _nonempty_string(row["speaker"], f"{label}.speaker")
    text = row["text"]
    if not isinstance(text, str):
        raise ConstructionWindowError(f"{label}.text must be a string")
    order = row["order"]
    if isinstance(order, bool) or not isinstance(order, int):
        raise ConstructionWindowError(f"{label}.order must be an integer")
    if row["text_sha256"] != source_sha256(text):
        raise ConstructionWindowError(f"{label}.text_sha256 does not match exact source text")
    return {
        "message_id": message_id,
        "statement_id": statement_id,
        "order": order,
        "speaker": speaker,
        "text": text,
        "text_sha256": row["text_sha256"],
    }


def _source_rows(messages: object) -> list[dict[str, Any]]:
    if isinstance(messages, (str, bytes, bytearray)) or not isinstance(messages, Sequence):
        raise ConstructionWindowError("messages must be an ordered sequence")
    rows = [_message(row, f"messages[{index}]") for index, row in enumerate(messages)]
    seen_messages: set[str] = set()
    seen_statements: set[str] = set()
    previous_order: int | None = None
    for row in rows:
        if row["message_id"] in seen_messages or row["statement_id"] in seen_statements:
            raise ConstructionWindowError("source message and statement ids must be unique")
        seen_messages.add(row["message_id"])
        seen_statements.add(row["statement_id"])
        if previous_order is not None and row["order"] != previous_order + 1:
            raise ConstructionWindowError("source messages must have strict contiguous order")
        previous_order = row["order"]
    return rows


def _scope(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _SCOPE_FIELDS:
        raise ConstructionWindowError(f"scope must contain exactly {sorted(_SCOPE_FIELDS)}")
    try:
        normalized = asdict(source_scope_from_dict(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ConstructionWindowError("scope requires nonempty bonfire, profile, and conversation") from exc
    return normalized


def _working_state(value: object) -> dict[str, list[str]]:
    if value is None:
        value = {key: [] for key in _STATE_FIELDS}
    if not isinstance(value, Mapping) or set(value) != _STATE_FIELDS:
        raise ConstructionWindowError(f"working_state must contain exactly {sorted(_STATE_FIELDS)}")
    result: dict[str, list[str]] = {}
    for key in sorted(_STATE_FIELDS):
        raw = value[key]
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise ConstructionWindowError(f"working_state.{key} must be a list of ids")
        ids = [_nonempty_string(item, f"working_state.{key}[]") for item in raw]
        if len(ids) != len(set(ids)):
            raise ConstructionWindowError(f"working_state.{key} ids must be unique")
        result[key] = ids
    return result


def _prior_topics(
    value: object,
    *,
    source_by_id: Mapping[str, dict[str, Any]] | None,
    active_start_order: int,
    excluded_message_ids: set[str] | None = None,
    excluded_statement_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ConstructionWindowError("prior_topics must be an ordered sequence")
    result: list[dict[str, Any]] = []
    seen_windows: set[str] = set()
    seen_messages: set[str] = set()
    previous_start: int | None = None
    for index, raw in enumerate(value):
        label = f"prior_topics[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != {"window_id", "messages", "status"}:
            raise ConstructionWindowError(f"{label} must contain exactly window_id, messages, and status")
        window_id = _nonempty_string(raw["window_id"], f"{label}.window_id")
        if window_id in seen_windows:
            raise ConstructionWindowError("prior topic window ids must be unique")
        seen_windows.add(window_id)
        status = raw["status"]
        if status not in {"suspended", "closed"}:
            raise ConstructionWindowError(f"{label}.status must be suspended or closed")
        raw_messages = raw["messages"]
        if isinstance(raw_messages, (str, bytes, bytearray)) or not isinstance(raw_messages, Sequence):
            raise ConstructionWindowError(f"{label}.messages must be an ordered sequence")
        topic_messages = [_message(row, f"{label}.messages[{i}]") for i, row in enumerate(raw_messages)]
        if not topic_messages:
            raise ConstructionWindowError(f"{label}.messages must be nonempty")
        last_order: int | None = None
        for topic_message in topic_messages:
            message_id = topic_message["message_id"]
            source = source_by_id.get(message_id) if source_by_id is not None else topic_message
            if source is None or topic_message != source:
                raise ConstructionWindowError(f"{label} must reference exact authoritative source rows")
            if message_id in seen_messages or message_id in (excluded_message_ids or set()):
                raise ConstructionWindowError("prior topics may not duplicate source messages")
            seen_messages.add(message_id)
            if topic_message["statement_id"] in (excluded_statement_ids or set()):
                raise ConstructionWindowError("prior topics may not duplicate source statements")
            if topic_message["order"] >= active_start_order:
                raise ConstructionWindowError(f"{label} contains a message at or after the active window")
            if last_order is not None and topic_message["order"] <= last_order:
                raise ConstructionWindowError(f"{label}.messages must be in source order")
            last_order = topic_message["order"]
        topic_start = topic_messages[0]["order"]
        if previous_start is not None and topic_start < previous_start:
            raise ConstructionWindowError("prior_topics must be chronological")
        previous_start = topic_start
        result.append({"window_id": window_id, "messages": topic_messages, "status": status})
    return result


def _source_char_count(window_input: Mapping[str, Any]) -> int:
    rows = list(window_input["active_window"]["messages"]) + [window_input["incoming_message"]]
    rows.extend(message for topic in window_input["prior_topics"] for message in topic["messages"])
    return sum(len(row["text"]) for row in rows)


def _with_hash(body: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(body)
    result["content_sha256"] = content_sha256(body)
    return result


def build_window_input(
    messages: Sequence[Mapping[str, Any]],
    *,
    scope: Mapping[str, Any],
    window_id: str,
    start_message_id: str,
    incoming_message_id: str,
    working_state: Mapping[str, Any] | None = None,
    prior_topics: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build exact active and incoming rows from the authoritative sequence.

    It must end at the incoming row; missing rows are never repaired with a prompt subset.
    Prior topics reference that source and remain separate.
    """
    rows = _source_rows(messages)
    if not rows:
        raise ConstructionWindowError("messages must include the incoming source message")
    positions = {row["message_id"]: index for index, row in enumerate(rows)}
    if start_message_id not in positions or incoming_message_id not in positions:
        raise ConstructionWindowError("start and incoming messages must exist in authoritative source")
    start, incoming = positions[start_message_id], positions[incoming_message_id]
    if start > incoming:
        raise ConstructionWindowError("active window cannot start after incoming message")
    if incoming != len(rows) - 1:
        raise ConstructionWindowError("incoming message must be the final available source row")

    active_messages = rows[start:incoming]
    incoming_row = rows[incoming]
    source_by_id = {row["message_id"]: row for row in rows}
    normalized_prior = _prior_topics(
        prior_topics,
        source_by_id=source_by_id,
        active_start_order=rows[start]["order"],
        excluded_message_ids={row["message_id"] for row in rows[start:]},
        excluded_statement_ids={row["statement_id"] for row in rows[start:]},
    )
    if any(topic["window_id"] == window_id for topic in normalized_prior):
        raise ConstructionWindowError("active and prior topic window ids must be unique")
    body = {
        "schema": WINDOW_INPUT_SCHEMA,
        "boundary_status": "initialize_context" if not active_messages else "classify_boundary",
        "scope": _scope(scope),
        "active_window": {
            "window_id": _nonempty_string(window_id, "window_id"),
            "messages": deepcopy(active_messages),
        },
        "incoming_message": deepcopy(incoming_row),
        "working_state": _working_state(working_state),
        "prior_topics": normalized_prior,
    }
    if not active_messages and start_message_id != incoming_message_id:
        raise ConstructionWindowError("an empty active window is valid only at initialization")
    if _source_char_count(body) > MAX_WINDOW_CHARS:
        raise ConstructionWindowError(f"complete window input exceeds {MAX_WINDOW_CHARS} source characters")
    result = _with_hash(body)
    validate_window_input(result)
    return result


def validate_window_input(value: object) -> None:
    """Validate shape, source ordering, scope, bounds, and the canonical content hash."""
    fields = {
        "schema",
        "boundary_status",
        "scope",
        "active_window",
        "incoming_message",
        "working_state",
        "prior_topics",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ConstructionWindowError(f"window input must contain exactly {sorted(fields)}")
    if value["schema"] != WINDOW_INPUT_SCHEMA:
        raise ConstructionWindowError("unsupported construction window schema")
    active = value["active_window"]
    if not isinstance(active, Mapping) or set(active) != {"window_id", "messages"}:
        raise ConstructionWindowError("active_window must contain window_id and messages")
    _nonempty_string(active["window_id"], "active_window.window_id")
    active_messages = _source_rows(active["messages"])
    incoming = _message(value["incoming_message"], "incoming_message")
    if active_messages and incoming["order"] != active_messages[-1]["order"] + 1:
        raise ConstructionWindowError("active window and incoming must be a contiguous source suffix")
    if not active_messages and value["boundary_status"] != "initialize_context":
        raise ConstructionWindowError("empty active window must be marked initialize_context")
    if active_messages and value["boundary_status"] != "classify_boundary":
        raise ConstructionWindowError("nonempty active window must be marked classify_boundary")
    all_current = active_messages + [incoming]
    current_ids = [row["message_id"] for row in all_current]
    statement_ids = [row["statement_id"] for row in all_current]
    if len(current_ids) != len(set(current_ids)) or len(statement_ids) != len(set(statement_ids)):
        raise ConstructionWindowError("active and incoming ids must be unique")
    _scope(value["scope"])
    _working_state(value["working_state"])

    active_start_order = active_messages[0]["order"] if active_messages else incoming["order"]
    prior = _prior_topics(
        value["prior_topics"],
        source_by_id=None,
        active_start_order=active_start_order,
        excluded_message_ids=set(current_ids),
        excluded_statement_ids=set(statement_ids),
    )
    if prior != value["prior_topics"]:
        raise ConstructionWindowError("prior_topics must use the canonical validated shape")
    if any(topic["window_id"] == active["window_id"] for topic in prior):
        raise ConstructionWindowError("active and prior topic window ids must be unique")
    if _source_char_count(value) > MAX_WINDOW_CHARS:
        raise ConstructionWindowError(f"complete window input exceeds {MAX_WINDOW_CHARS} source characters")
    body = {key: deepcopy(item) for key, item in value.items() if key != "content_sha256"}
    if value["content_sha256"] != content_sha256(body):
        raise ConstructionWindowError("window input content_sha256 does not match content")


__all__ = [
    "MAX_WINDOW_CHARS",
    "WINDOW_INPUT_SCHEMA",
    "ConstructionWindowError",
    "build_window_input",
    "validate_window_input",
]
