"""Conversation split, session-date checks, batch preparation and sources (conversation-only, question-blind).

Ported from a reference-harness module (unreached top-level definitions pruned). Differences: parse_session_date (from a data-export script) and PreparationError / prepare_batches (from a pilot script) are copied in; the by-path module loads are gone; split_conversation tolerates a conversation JSON without `qa`. parse_session_date raises ValueError on a missing or unparseable session date instead of returning the wall clock (`session_start` / `check_session_dates` name the field; the driver's preflight calls the latter, so a bad date is a refusal before any stage).
"""

import itertools
import json
import re
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from memory_kernel.construction.memory_discourse_contract import discourse_source
from memory_kernel.pipeline.stages._addressing import agent_mentions
from memory_kernel.pipeline.stages._input_clean import clean_text, split_parts

CONV_ONLY_KEYS = {"sample_id", "conversation", "qa"}


# -- parse_session_date: copied from a data-export script ----------


def parse_session_date(raw: str | None) -> datetime:
    """The session start (UTC) of a ``H:MM am|pm on D Month, YYYY`` string. A missing or unparseable date raises
    ``ValueError``: the build never falls back to the wall clock (that made the message timestamps, and so every
    sealed output after ``select``, differ between two runs over the same file)."""
    match = re.search(r"(\d+):(\d+)\s*(am|pm)\s*on\s*(\d+)\s*(\w+),?\s*(\d+)", raw, re.I) if raw else None
    if not match:
        raise ValueError(f"unparseable session date {raw!r}: expected 'H:MM am|pm on D Month, YYYY'")
    hour_s, minute_s, ampm, day_s, month_s, year_s = match.groups()
    hour = int(hour_s)
    if ampm.lower() == "pm" and hour != 12:
        hour += 12
    if ampm.lower() == "am" and hour == 12:
        hour = 0
    try:
        month = datetime.strptime(month_s[:3], "%b").month
        return datetime(int(year_s), month, int(day_s), hour, int(minute_s), tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"unparseable session date {raw!r}: {exc}") from None


#: The conversation key a forward append's one-session input carries: the bonfire-wide number of its session
#: (``session_<FIRST_SESSION>`` is its only session; every full-build input starts at 1 and has no such key).
FIRST_SESSION = "first_session"


def first_session(conv: Mapping[str, Any]) -> int:
    """The number of ``conv``'s first session: ``conv[FIRST_SESSION]`` (a forward append's session input), else 1."""
    first = conv.get(FIRST_SESSION, 1)
    if isinstance(first, bool) or not isinstance(first, int) or first < 1:
        raise ValueError(f"{FIRST_SESSION} must be a positive integer, not {first!r}")
    return first


def session_keys(conv: Mapping[str, Any]) -> Iterator[str]:
    """``session_<first>`` ... in order (``first_session``: 1 for a full-build input), up to the first number that is
    absent or not a message list (no upper bound)."""
    for i in itertools.count(first_session(conv)):
        session_key = f"session_{i}"
        if session_key not in conv or not isinstance(conv[session_key], list):
            return
        yield session_key


def session_conversations(conv: Mapping[str, Any], sample_id: str) -> dict[str, str]:
    """``{session_id: conversation id}`` for every session of ``conv``. A ``session_id`` is ``<sample_id>-session_N``,
    as the sources carry it. The conversation (chat) of a session is its optional ``session_N_conversation``. A session
    without that key belongs to the default conversation, ``sample_id``. When no session has the key, the whole input
    is one conversation, which is the 1.0.0 behaviour. ``config.load_conversation`` refuses a partly tagged input (a
    build input tags every session or none), so the default applies only to an untagged input."""
    return {f"{sample_id}-{key}": conv.get(f"{key}_conversation") or sample_id for key in session_keys(conv)}


def conversations_of(ctx: Any) -> dict[str, str]:
    """``session_id`` -> conversation (chat) id of a build's conversation input (``session_conversations``); ``ctx``
    is a stage context (``ctx.cfg.conversation`` / ``ctx.cfg.sample_id``)."""
    from memory_kernel.pipeline.config import load_conversation

    item, _ = load_conversation(ctx.cfg.conversation, ctx.cfg.sample_id)
    return session_conversations(item["conversation"], item["sample_id"])


def turn_timestamp(msg: Mapping[str, Any], fallback: datetime) -> str:
    """The turn's own ISO timestamp (timezone required) in UTC, else the 1.0.0 ``session start + offset`` value."""
    raw = msg.get("timestamp")
    if raw is None:
        return fallback.isoformat()
    try:
        when = datetime.fromisoformat(str(raw))
    except ValueError:
        raise ValueError(f"turn {msg.get('dia_id')!r}: timestamp {raw!r} is not an ISO 8601 time") from None
    if when.tzinfo is None:
        raise ValueError(f"turn {msg.get('dia_id')!r}: timestamp {raw!r} has no timezone")
    return when.astimezone(UTC).isoformat()  # the boundary walk takes UTC stamps only


def turn_addressees(msg: Mapping[str, Any]) -> list[str] | None:
    """The turn's explicit addressees, or None when the turn carries none (1.0.0 input)."""
    raw = msg.get("addressees")
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(a, str) and a for a in raw):
        raise ValueError(f"turn {msg.get('dia_id')!r}: addressees must be a list of non-empty strings")
    return list(dict.fromkeys(raw))


def session_start(conv: Mapping[str, Any], session_key: str) -> datetime:
    """``parse_session_date`` of ``<session_key>_date_time``, naming the key in the error."""
    field_name = f"{session_key}_date_time"
    try:
        return parse_session_date(conv.get(field_name))
    except ValueError as exc:
        raise ValueError(f"{field_name}: {exc}") from None


def check_session_dates(conv: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when any session of ``conv`` has no parseable date (``build_prepared`` would raise it)."""
    for session_key in session_keys(conv):
        _ = session_start(conv, session_key)


# -- prepare_batches: copied from a pilot script -----


class PreparationError(ValueError):
    """The source does not satisfy the pinned caption preparation contract."""


def prepare_batches(batches: object) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
    if not isinstance(batches, list):
        raise PreparationError("source batches must be a list")
    prepared: list[list[dict[str, Any]]] = []
    parent_rows = caption_rows = raw_query_rows = raw_url_rows = 0
    source_ids: set[str] = set()
    for batch in batches:
        if not isinstance(batch, list):
            raise PreparationError("source batch must be a list")
        output_batch: list[dict[str, Any]] = []
        for row in batch:
            if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
                raise PreparationError("source row must contain object metadata")
            metadata = row["metadata"]
            source_id = metadata.get("source_message_id")
            if not isinstance(source_id, str) or not source_id:
                raise PreparationError("source message ID must be non-empty")
            if source_id in source_ids:
                raise PreparationError(f"duplicate source message ID: {source_id}")
            source_ids.add(source_id)
            if metadata.get("image_query") is not None:
                raw_query_rows += 1
            if metadata.get("img_url"):
                raw_url_rows += 1
            output_batch.append(row)
            parent_rows += 1
            caption = metadata.get("blip_caption")
            if caption is None:
                continue
            if not isinstance(caption, str) or not caption.strip():
                raise PreparationError(f"invalid BLIP caption for {source_id}")
            caption_id = f"{source_id}-image"
            if caption_id in source_ids:
                raise PreparationError(f"caption source ID collides: {caption_id}")
            source_ids.add(caption_id)
            caption_metadata = {
                "sample_id": metadata.get("sample_id"),
                "sessionId": metadata.get("sessionId"),
                "parent_source_message_id": source_id,
                "parent_dia_id": metadata.get("dia_id"),
                "source_kind": "image_context",
                "source_message_id": caption_id,
                "image_query": metadata.get("image_query"),
                "img_url": list(metadata.get("img_url") or []),
            }
            output_batch.append(
                {
                    "text": caption,
                    "username": row.get("username"),
                    "timestamp": row.get("timestamp"),
                    "metadata": caption_metadata,
                }
            )
            caption_rows += 1
        prepared.append(output_batch)
    return prepared, {
        "parent_rows": parent_rows,
        "caption_rows": caption_rows,
        "prepared_rows": parent_rows + caption_rows,
        "raw_query_rows": raw_query_rows,
        "raw_url_rows": raw_url_rows,
    }


def split_conversation(item: Mapping) -> tuple[dict, dict]:
    """(conversation_only with qa=[], eval_qa). Observation / event_summary / session_summary are dropped."""
    conv_only = {"sample_id": item["sample_id"], "conversation": json.loads(json.dumps(item["conversation"])), "qa": []}
    eval_qa = {"sample_id": item["sample_id"], "qa": json.loads(json.dumps(item.get("qa") or []))}
    return conv_only, eval_qa


def build_prepared(conv_only: Mapping) -> list[list[dict]]:
    """One batch of messages per session of a conversation-only dict (blank ones dropped), then ``prepare_batches``."""
    if set(conv_only) != CONV_ONLY_KEYS or conv_only["qa"] != []:
        raise ValueError("build_prepared takes the conversation-only form (qa == [])")
    sample_id, conv = conv_only["sample_id"], conv_only["conversation"]
    handles = list(conv.get("agent_handles") or [])
    # an addressee naming an agent handle (with or without "@", any case) is that handle's bare name
    agent_names = {h.strip().lstrip("@").casefold(): h.strip().lstrip("@") for h in handles}
    batches: list[list[dict]] = []
    for session_key in session_keys(conv):
        start = session_start(conv, session_key)
        session_id = f"{sample_id}-{session_key}"
        batch: list[dict] = []
        last: datetime | None = None  # the previous kept turn's resolved time
        for offset, msg in enumerate(conv[session_key]):
            text = str(msg.get("text") or "").strip()
            mentioned: list[str] = []
            if handles:
                text, mentioned = agent_mentions(text, handles)
            if not text:
                continue
            text = clean_text(text)
            if not text:  # nothing but markup
                continue
            parts = split_parts(text)
            image_urls = msg.get("img_url") or []
            if isinstance(image_urls, str):
                image_urls = [image_urls]
            base_id = str(msg.get("dia_id") or f"{session_id}-{offset}")
            addressees = turn_addressees(msg)
            timestamp = turn_timestamp(msg, start + timedelta(minutes=offset))
            when = datetime.fromisoformat(timestamp)
            if last is not None and when < last and msg.get("timestamp") is not None:
                raise ValueError(
                    f"turn {msg.get('dia_id')!r}: timestamp {msg['timestamp']!r} is earlier than the previous turn's "
                    f"{last.isoformat()}"
                )
            last = when
            for k, part in enumerate(parts, 1):
                row = {
                    "text": part,
                    "username": str(msg.get("speaker") or ""),
                    "timestamp": timestamp,
                    "metadata": {
                        "sample_id": sample_id,
                        "sessionId": session_id,
                        "dia_id": msg.get("dia_id"),
                        # the image belongs to the first part only (one caption row per image)
                        "blip_caption": msg.get("blip_caption") if k == 1 else None,
                        "image_query": msg.get("query") if k == 1 else None,
                        "img_url": list(image_urls) if k == 1 else [],
                        "source_kind": "message",
                        "source_message_id": base_id if len(parts) == 1 else f"{base_id}#{k}",
                    },
                }
                if mentioned or addressees is not None:
                    named = [
                        agent_names.get(a.strip().lstrip("@").casefold(), a) for a in mentioned + (addressees or [])
                    ]
                    row["metadata"]["addressees"] = list(dict.fromkeys(named))
                batch.append(row)
        if batch:
            batches.append(batch)
    prepared, _counts = prepare_batches(json.loads(json.dumps(batches)))
    return prepared


def build_sources(prepared: list, scope: Mapping, first_order: int = 0) -> list[dict]:
    """discourse_source(...) exactly as native_conversation_handoff --phase prepare (lines 41-64). ``first_order`` is
    the order of the first message: 0 for a full build; a forward append's session continues the bonfire's orders."""
    rows = [r for b in prepared for r in b]
    if any(r["metadata"]["sample_id"] != scope["conversation_id"] for r in rows):
        raise ValueError("source file contains another conversation")
    messages = [r for r in rows if r["metadata"]["source_kind"] == "message"]
    captions = [r for r in rows if r["metadata"]["source_kind"] == "image_context"]
    ids = {r["metadata"]["source_message_id"] for r in messages}
    if (
        len(ids) != len(messages)
        or not 1 <= len(messages) <= 4096
        or len(messages) + len(captions) != len(rows)
        or any(r["metadata"]["parent_source_message_id"] not in ids for r in captions)
    ):
        raise ValueError("source identities, channels or bounds differ")
    sources = []
    for i, row in enumerate(messages):
        meta = row["metadata"]
        attachments = []
        for caption in captions:
            if caption["metadata"]["parent_source_message_id"] != meta["source_message_id"]:
                continue
            if caption["username"] != row["username"] or caption["metadata"]["sessionId"] != meta["sessionId"]:
                raise ValueError("caption parent provenance differs")
            attachments.append(
                {"locator": caption["metadata"]["source_message_id"], "media_type": "image", "caption": caption["text"]}
            )
        sources.append(
            discourse_source(
                scope=scope,
                message_id=meta["source_message_id"],
                order=first_order + i,
                text=row["text"],
                speaker=row["username"],
                session_id=meta["sessionId"],
                timestamp=row["timestamp"],
                attachments=attachments,
                addressees=meta.get("addressees", []),
            )
        )
    return sources
