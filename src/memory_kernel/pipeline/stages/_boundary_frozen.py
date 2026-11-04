"""Copies of the frozen boundary compiler.

The reference harness reached these through a pilot ingest module (which imports the harness's 7.7k-line stock
PropBank Jev runner and its 2.4k-line window runner) by appending a scripts directory to a package path. Only the
functions the C2 boundary stage actually calls are copied here,
verbatim except that ``stock.MODEL`` / ``stock.digest`` / ``transport.*`` are the local names:

  * from the harness's window runner: MODEL, INPUT_TOKEN_MARGIN, WindowRunnerError, canonical,
    digest, _token_upper_bound, validate_response, _number, _probabilities
  * from the harness's pilot ingest module: INPUT_CAP, BOUNDARY_CHOICES, FullIngestError, _active_episode_start,
    compile_boundary_step, apply_boundary_response
  * from the harness's scheduled-episode module: BOUNDARY_ID, EpisodeScheduledError,
    _interaction_boundary_question (used by the replay-equivalence gate)

One difference: ``compile_boundary_step`` takes ``offset`` (0 = the copy as it was). A forward append walks a chat
from where the bonfire left it, with only the chat's open episode and the new session in hand: ``messages[0]`` is then
the walk's message ``offset``, and every index, cursor and order in the step is still a position of the whole walk,
so the step is the one the whole walk compiles.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from memory_kernel.construction.mined_affordance_contract import content_sha256
from memory_kernel.model_family import model_family_ok
from memory_kernel.windows.construction_boundary_timeline import compile_timed_sequential_boundary_question
from memory_kernel.windows.construction_window import build_window_input

MODEL = "typesafe/jev-1.13"

INPUT_TOKEN_MARGIN = 64


class WindowRunnerError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _token_upper_bound(request: Mapping[str, Any]) -> int:
    # Count every serialized byte as a token, then reserve a small envelope margin.
    return max(1, len(canonical(request))) + INPUT_TOKEN_MARGIN


def validate_response(response: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if not model_family_ok(MODEL, str(response.get("model"))):
        raise WindowRunnerError("response model is not allowlisted")
    questions = request.get("questions")
    answers = response.get("answers")
    if not isinstance(questions, Mapping) or not questions or not isinstance(answers, Mapping):
        raise WindowRunnerError("request/response questions are malformed")
    if set(answers) != set(questions):
        raise WindowRunnerError("response answer key set differs from request exactly")
    validated: dict[str, Mapping[str, Any]] = {}
    for question_id, question in questions.items():
        answer = answers[question_id]
        if not isinstance(question, Mapping) or not isinstance(answer, Mapping):
            raise WindowRunnerError("question or answer is malformed")
        kind = question.get("type")
        if answer.get("type") != kind:
            raise WindowRunnerError("typed answer differs from requested primitive")
        if kind == "choice":
            criteria = question.get("criteria")
            probabilities = answer.get("probabilities")
            if not isinstance(criteria, Mapping) or not isinstance(probabilities, Mapping):
                raise WindowRunnerError("choice criteria/probabilities are malformed")
            if set(probabilities) != set(criteria):
                raise WindowRunnerError("choice probability keys differ from criteria")
            vals = _probabilities(probabilities)
            if abs(sum(vals.values()) - 1.0) > 0.02:
                raise WindowRunnerError("choice probabilities do not sum to one")
            choice = answer.get("choice")
            if choice not in criteria:
                raise WindowRunnerError("choice answer names an unknown option")
            corrected = None
            if vals[choice] < max(vals.values()) - 1e-9:  # the label disagrees with its own probabilities: use the top
                corrected, choice = choice, max(vals, key=vals.__getitem__)
            confidence = _number(answer.get("confidence"), "choice confidence")
            if not 0.0 <= confidence <= 1.0:
                raise WindowRunnerError("choice confidence is outside [0,1]")
            validated[question_id] = {
                "type": "choice",
                "choice": choice,
                "probabilities": vals,
                "confidence": confidence,
                **({"choice_corrected_from": corrected} if corrected else {}),
            }
        elif kind == "noul":
            noul = _number(answer.get("noul"), "Noul yes probability")
            if not 0.0 <= noul <= 1.0:
                raise WindowRunnerError("Noul probability is outside [0,1]")
            validated[question_id] = {"type": "noul", "noul": noul}
        else:
            raise WindowRunnerError("unsupported question primitive")
    return validated


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise WindowRunnerError(f"{label} is invalid")
    return float(value)


def _probabilities(raw: Mapping[str, Any]) -> dict[str, float]:
    values = {str(key): _number(value, f"probability {key}") for key, value in raw.items()}
    if any(value < 0.0 or value > 1.0 for value in values.values()):
        raise WindowRunnerError("choice probability is outside [0,1]")
    return values


INPUT_CAP = 32_768

BOUNDARY_CHOICES = frozenset({"continue_active", "shift_topic", "unresolved"})


class FullIngestError(ValueError):
    """The pinned source or an opt-in question cannot be replayed exactly."""


def _active_episode_start(history: Sequence[Mapping[str, Any]], index: int) -> int:
    if len(history) != index - 1:
        raise FullIngestError("boundary history must cover every previous incoming message")
    active_start = 0
    for position, row in enumerate(history, start=1):
        if row.get("choice") not in BOUNDARY_CHOICES or row.get("index") != position:
            raise FullIngestError("boundary history is not a chronological finite-choice replay")
        if row["choice"] == "shift_topic":
            active_start = position
    return active_start


def compile_boundary_step(
    messages: Sequence[Mapping[str, Any]],
    timestamp_rows: Mapping[str, Mapping[str, Any]],
    *,
    scope: Mapping[str, Any],
    index: int,
    history: Sequence[Mapping[str, Any]],
    previous_cursor: int,
    window_prefix: str,
    step_schema: str = "mk-full-timed-boundary-step/v1",
    input_cap: int = INPUT_CAP,
    offset: int = 0,
) -> dict[str, Any]:
    """Choose the longest contiguous visible suffix that fits the pinned wire cap.

    ``window_prefix`` heads each window id (``<prefix>-<active_start>-cursor-<cursor>``); it is part of the wire, so a
    replay of an archived wire must pass the prefix that wire was made with. ``step_schema`` is the schema id of the
    sealed step body (not part of the wire); a replay of an archived run passes the id that run sealed its steps with.
    ``offset``: ``messages[0]`` is the walk's message ``offset`` (see the module docstring); the messages must reach
    back to the active episode's start."""
    if not 1 <= index < offset + len(messages) or not 0 <= previous_cursor < index or offset < 0:
        raise FullIngestError("boundary index or working cursor is invalid")
    if type(input_cap) is not int or input_cap <= 0 or input_cap > INPUT_CAP:
        raise FullIngestError("input cap exceeds the reviewed provider bound")
    active_start = _active_episode_start(history, index)
    if active_start < offset:
        raise FullIngestError("the messages given start after the active episode's start")
    if [row["message_id"] for row in messages] != list(timestamp_rows):
        raise FullIngestError("timestamp source order differs from archived source chronology")
    for row in messages:
        timestamp_row = timestamp_rows[row["message_id"]]
        if (
            timestamp_row["text"] != row["text"]
            or row["text_sha256"] != hashlib.sha256(row["text"].encode()).hexdigest()
        ):
            raise FullIngestError("timestamp text differs from exact archived source")
    first_cursor = max(active_start, previous_cursor)
    for cursor in range(first_cursor, index):
        window = build_window_input(
            messages[: index - offset + 1],
            scope=scope,
            window_id=f"{window_prefix}-{active_start}-cursor-{cursor}",
            start_message_id=messages[cursor - offset]["message_id"],
            incoming_message_id=messages[index - offset]["message_id"],
            prior_topics=[],
        )
        witnesses = {
            row["message_id"]: {
                "message_id": row["message_id"],
                "source_text_sha256": row["text_sha256"],
                "timestamp": timestamp_rows[row["message_id"]]["timestamp"],
            }
            for row in [*messages[cursor - offset : index - offset], messages[index - offset]]
        }
        question = compile_timed_sequential_boundary_question(
            window,
            timestamp_witnesses=witnesses,
            expected_witness_sha256=content_sha256(witnesses),
        )
        archived = list(messages[active_start - offset : cursor - offset])
        capacity = {
            "policy": "longest_contiguous_recent_suffix_no_episode_change",
            "semantic_episode_start_order": active_start,
            "working_window_start_order": cursor,
            "archived_same_episode_source_count": len(archived),
            "archived_same_episode_source_ids": [row["message_id"] for row in archived],
            "archived_same_episode_sources_sha256": content_sha256(archived),
            "archived_text_not_model_visible": True,
        }
        wire_question = deepcopy(question["wire_question"])
        if "the complete active topic" not in wire_question["instructions"]:
            raise FullIngestError("timed boundary instructions changed upstream")
        wire_question["instructions"] = wire_question["instructions"].replace(
            "the complete active topic", "the visible current working window"
        )
        wire_question["instructions"] += (
            " State.capacity records any earlier messages in this same episode that are "
            "outside the bounded working window. Their text is not available here; do not "
            "infer it or treat this capacity rollover as a semantic episode boundary."
        )
        wire = {
            "model": MODEL,
            "state": {**question["state"], "capacity": capacity},
            "questions": {"boundary": wire_question},
        }
        upper = _token_upper_bound(wire)
        if upper > input_cap:
            continue
        body = {
            "schema_version": step_schema,
            "index": index,
            "message_id": messages[index - offset]["message_id"],
            "semantic_episode_start_order": active_start,
            "working_window_start_order": cursor,
            "capacity_rollover": cursor > first_cursor,
            "capacity": capacity,
            "boundary_question_sha256": question["content_sha256"],
            "wire": wire,
            "request_sha256": digest(wire),
            "input_tokens_upper_bound": upper,
        }
        return {**body, "content_sha256": content_sha256(body)}
    raise FullIngestError(
        f"window_capacity_pending at {messages[index - offset]['message_id']}: even the prior message plus incoming "
        "cannot fit"
    )


def apply_boundary_response(
    step: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    raw_response_sha256: str,
    history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Append one validated boundary choice after the caller archives its receipt."""
    if (
        len(history) != step["index"] - 1
        or type(raw_response_sha256) is not str
        or len(raw_response_sha256) != 64
        or any(char not in "0123456789abcdef" for char in raw_response_sha256)
    ):
        raise FullIngestError("boundary receipt or predecessor history is missing")
    if _active_episode_start(history, step["index"]) != step["semantic_episode_start_order"]:
        raise FullIngestError("boundary step is not the next chronological decision")
    answers = validate_response(response, step["wire"])
    answer = answers["boundary"]
    row = {
        "index": step["index"],
        "message_id": step["message_id"],
        "choice": answer["choice"],
        "probabilities": answer["probabilities"],
        "confidence": answer["confidence"],
        **({"choice_corrected_from": answer["choice_corrected_from"]} if "choice_corrected_from" in answer else {}),
        "step_sha256": step["content_sha256"],
        "request_sha256": step["request_sha256"],
        "raw_response_sha256": raw_response_sha256,
        "working_window_start_order": step["working_window_start_order"],
        "capacity": deepcopy(step["capacity"]),
    }
    return [*map(dict, history), row]


BOUNDARY_ID = "boundary"


class EpisodeScheduledError(ValueError):
    """A source, response, or episode transition violates the opt-in contract."""


def _interaction_boundary_question(boundary: Mapping[str, Any]) -> dict[str, Any]:
    question = deepcopy(boundary["wire"]["questions"][BOUNDARY_ID])
    if set(question["criteria"]) != {"continue_active", "shift_topic", "unresolved"}:
        raise EpisodeScheduledError("boundary options changed upstream")
    question["instructions"] += (
        " Decide continuity of the conversational interaction or session, not merely"
        " whether entities or topics overlap. A later interaction about the same subject"
        " may start a new episode when the source and elapsed context support that;"
        " no elapsed-time threshold decides it. `shift_topic` means start a new"
        " sequential episode, even if the subject recurs. Choose unresolved when"
        " interaction continuity is not established."
    )
    return question
