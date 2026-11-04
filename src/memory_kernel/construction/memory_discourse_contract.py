"""Exact source and bounded syntax/native artifact contracts.

Seals establish content integrity only. Authentication and trusted profile/head
selection belong to the host; model output cannot supply those authorities.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from memory_kernel.construction.memory_semantic_jev import check_seal, seal
from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256

MAX_SOURCE_CHARS = 65536
MAX_SEGMENTS = 32
MAX_ALTERNATIVES = 4
MAX_RECORD_BYTES = {
    "discourse-source/v1": 262144,
    "discourse-syntax/v1": 131072,
    "discourse-parse-agenda/v1": 131072,
    "discourse-native-parse/v1": 4 * 1024 * 1024,
}


class DiscourseContractError(ValueError):
    """Discourse evidence differs from its exact bounded source contract."""


def discourse_id(kind: str, *parts: Any) -> str:
    return f"{kind}:{content_sha256(list(parts))}"


def _check_record(row: Mapping[str, Any], schema: str, fields: set[str]) -> None:
    if not isinstance(row, Mapping) or set(row) != fields | {"schema_version", "content_sha256"}:
        raise DiscourseContractError(f"{schema} fields differ")
    try:
        if len(canonical_json(row).encode("utf-8")) > MAX_RECORD_BYTES[schema]:
            raise DiscourseContractError(f"{schema} byte bound exceeded")
        check_seal(row, schema)
    except ValueError as exc:
        raise DiscourseContractError(str(exc)) from exc


def _text(value: Any, *, empty: bool = False) -> bool:
    return isinstance(value, str) and (empty or bool(value))


def _span(start: Any, end: Any, size: int) -> bool:
    return type(start) is int and type(end) is int and 0 <= start <= end <= size and (start < end or size == 0)


def discourse_source(
    *,
    scope: Mapping[str, Any],
    message_id: str,
    order: int,
    text: str,
    speaker: str,
    session_id: str,
    addressees: Sequence[str] = (),
    timestamp: str | None = None,
    attachments: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    rows = []
    for ordinal, attachment in enumerate(attachments):
        if not isinstance(attachment, Mapping) or set(attachment) != {"locator", "media_type", "caption"}:
            raise DiscourseContractError("attachment input fields differ")
        rows.append(
            {
                **deepcopy(dict(attachment)),
                "ordinal": ordinal,
                "attachment_id": discourse_id(
                    "attachment", dict(scope), session_id, message_id, ordinal, attachment["locator"]
                ),
            }
        )
    result = seal(
        {
            "schema_version": "discourse-source/v1",
            "scope": deepcopy(dict(scope)),
            "message_id": message_id,
            "order": order,
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "speaker": speaker,
            "addressees": list(addressees),
            "session_id": session_id,
            "timestamp": timestamp,
            "attachments": rows,
        }
    )
    check_discourse_source(result)
    return result


def check_discourse_source(source: Mapping[str, Any]) -> None:
    _check_record(
        source,
        "discourse-source/v1",
        {
            "scope",
            "message_id",
            "order",
            "text",
            "text_sha256",
            "speaker",
            "addressees",
            "session_id",
            "timestamp",
            "attachments",
        },
    )
    if (
        not isinstance(source["scope"], Mapping)
        or not source["scope"]
        or any(not _text(k) or not _text(v) for k, v in source["scope"].items())
        or any(not _text(source[k]) for k in ("message_id", "speaker", "session_id"))
        or type(source["order"]) is not int
        or source["order"] < 0
        or not _text(source["text"], empty=True)
        or len(source["text"]) > MAX_SOURCE_CHARS
        or (source["timestamp"] is not None and not _text(source["timestamp"]))
    ):
        raise DiscourseContractError("source metadata differs")
    if source["text_sha256"] != hashlib.sha256(source["text"].encode("utf-8")).hexdigest():
        raise DiscourseContractError("original UTF-8 text digest differs")
    addressees = source["addressees"]
    if (
        not isinstance(addressees, list)
        or len(addressees) > 64
        or any(not _text(x) for x in addressees)
        or len(set(addressees)) != len(addressees)
    ):
        raise DiscourseContractError("addressees differ")
    if not isinstance(source["attachments"], list) or len(source["attachments"]) > 32:
        raise DiscourseContractError("attachment bound differs")
    seen = set()
    for ordinal, row in enumerate(source["attachments"]):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"attachment_id", "ordinal", "locator", "media_type", "caption"}
            or type(row["ordinal"]) is not int
            or row["ordinal"] != ordinal
            or not _text(row["locator"])
            or not _text(row["media_type"])
            or not _text(row["caption"], empty=True)
        ):
            raise DiscourseContractError("attachment occurrence differs")
        expected = discourse_id(
            "attachment", source["scope"], source["session_id"], source["message_id"], ordinal, row["locator"]
        )
        if row["attachment_id"] != expected or expected in seen:
            raise DiscourseContractError("attachment occurrence identity differs or repeats")
        seen.add(expected)


def check_discourse_syntax(syntax: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    check_discourse_source(source)
    _check_record(
        syntax,
        "discourse-syntax/v1",
        {"source_sha256", "parser_stamp", "segments", "alternatives", "status", "diagnostics"},
    )
    if (
        syntax["source_sha256"] != source["content_sha256"]
        or syntax["status"] not in {"parsed", "unavailable"}
        or not _text(syntax["parser_stamp"])
        or not isinstance(syntax["diagnostics"], list)
        or len(syntax["diagnostics"]) > 64
        or any(not _text(x) for x in syntax["diagnostics"])
    ):
        raise DiscourseContractError("syntax source, status or parser differs")
    rows, alternatives = syntax["segments"], syntax["alternatives"]
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= MAX_SEGMENTS
        or not isinstance(alternatives, list)
        or not 1 <= len(alternatives) <= MAX_ALTERNATIVES
    ):
        raise DiscourseContractError("syntax bounds differ")
    by_id: dict[str, Any] = {}
    size = len(source["text"])
    roots = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"segment_id", "start", "end", "parent_id", "kind", "token_indices", "dependency"}
            or not _span(row["start"], row["end"], size)
            or row["kind"] not in {"source", "sentence", "clause"}
            or (row["dependency"] is not None and not _text(row["dependency"]))
            or not isinstance(row["token_indices"], list)
            or any(type(x) is not int or x < 0 for x in row["token_indices"])
            or row["token_indices"] != sorted(set(row["token_indices"]))
        ):
            raise DiscourseContractError("syntax span differs")
        sid = discourse_id("segment", source["content_sha256"], row["start"], row["end"])
        if row["segment_id"] != sid or sid in by_id:
            raise DiscourseContractError("segment identity differs or repeats")
        by_id[sid] = row
        if row["parent_id"] is None:
            roots.append(row)
    if len(roots) != 1 or (roots[0]["start"], roots[0]["end"], roots[0]["kind"]) != (0, size, "source"):
        raise DiscourseContractError("syntax requires exact full-source root")
    for row in rows:
        if row["parent_id"] is None:
            continue
        parent = by_id.get(row["parent_id"])
        if (
            parent is None
            or parent["start"] > row["start"]
            or parent["end"] < row["end"]
            or (parent["start"], parent["end"]) == (row["start"], row["end"])
        ):
            raise DiscourseContractError("segment parent is absent or not strictly enclosing")
        enclosing = [
            other
            for other in rows
            if other["segment_id"] != row["segment_id"]
            and other["start"] <= row["start"]
            and other["end"] >= row["end"]
        ]
        nearest = min(enclosing, key=lambda other: other["end"] - other["start"])
        if row["parent_id"] != nearest["segment_id"]:
            raise DiscourseContractError("segment parent skips nearest enclosing span")
    for left in rows:
        for right in rows:
            if left["start"] < right["start"] < left["end"] < right["end"]:
                raise DiscourseContractError("crossing syntax spans")
    seen = set()
    for alt in alternatives:
        if (
            not isinstance(alt, Mapping)
            or set(alt) != {"alternative_id", "segment_ids"}
            or not isinstance(alt["segment_ids"], list)
            or not alt["segment_ids"]
        ):
            raise DiscourseContractError("segmentation alternative differs")
        ids = alt["segment_ids"]
        if any(not isinstance(x, str) or x not in by_id for x in ids):
            raise DiscourseContractError("segmentation references absent span")
        if len(set(ids)) != len(ids):
            raise DiscourseContractError("segmentation repeats a source span")
        aid = discourse_id("segmentation", source["content_sha256"], ids)
        if alt["alternative_id"] != aid or aid in seen:
            raise DiscourseContractError("segmentation identity differs or repeats")
        seen.add(aid)
        cursor = 0
        for sid in ids:
            if by_id[sid]["start"] != cursor:
                raise DiscourseContractError("segmentation coverage overlaps or has gaps")
            cursor = by_id[sid]["end"]
        if cursor != size:
            raise DiscourseContractError("segmentation does not cover exact source")
