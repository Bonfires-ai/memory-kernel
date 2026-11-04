import json
from pathlib import Path

import pytest

from memory_kernel.pipeline.config import ConversationError, load_conversation
from memory_kernel.pipeline.stages._conversation import build_prepared, build_sources, session_keys, split_conversation

DATE = "1:00 pm on 1 May, 2026"


def conv(sessions: int = 1, **extra) -> dict:
    c = {"speaker_a": "Ana", "speaker_b": "Bo", **extra}
    for i in range(1, sessions + 1):
        c[f"session_{i}"] = [{"speaker": "Ana", "dia_id": f"D{i}:1", "text": "hello there"}]
        c[f"session_{i}_date_time"] = DATE
    return {"sample_id": "s1", "conversation": c, "qa": []}


def sources_of(item: dict) -> list[dict]:
    conv_only, _ = split_conversation(item)
    scope = {"bonfire_id": "b1", "conversation_id": "s1"}
    return build_sources(build_prepared(conv_only), scope)


def test_more_than_99_sessions_load(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(conv(sessions=120)))
    item, _ = load_conversation(p, None)
    assert len(list(session_keys(item["conversation"]))) == 120


def test_addressees_reach_the_source():
    item = conv()
    item["conversation"]["session_1"][0]["addressees"] = ["Bo"]
    assert sources_of(item)[0]["addressees"] == ["Bo"]


def test_turn_timestamp_overrides_session_offset():
    item = conv()
    item["conversation"]["session_1"][0]["timestamp"] = "2026-05-01T13:07:09+00:00"
    assert sources_of(item)[0]["timestamp"] == "2026-05-01T13:07:09+00:00"


def test_naive_timestamp_is_refused():
    item = conv()
    item["conversation"]["session_1"][0]["timestamp"] = "2026-05-01T13:07:09"
    with pytest.raises(ValueError, match="timezone"):
        sources_of(item)


def test_bad_agent_handles_refused(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(conv(agent_handles=["", "x"])))
    with pytest.raises(ConversationError, match="agent_handles"):
        load_conversation(p, None)


def test_plain_turn_metadata_has_no_addressees_key():
    conv_only, _ = split_conversation(conv())
    row = build_prepared(conv_only)[0][0]
    assert "addressees" not in row["metadata"]


def test_malformed_timestamp_names_the_turn():
    item = conv()
    item["conversation"]["session_1"][0]["timestamp"] = "yesterday"
    with pytest.raises(ValueError, match="D1:1"):
        sources_of(item)


def test_offset_timestamp_becomes_utc():
    item = conv()
    item["conversation"]["session_1"][0]["timestamp"] = "2026-05-01T15:07:09+02:00"
    assert sources_of(item)[0]["timestamp"] == "2026-05-01T13:07:09+00:00"


def test_backwards_timestamp_is_refused_naming_the_turn():
    item = conv()
    item["conversation"]["session_1"] = [
        {"speaker": "Ana", "dia_id": "D1:1", "text": "first", "timestamp": "2026-05-01T13:10:00+00:00"},
        {"speaker": "Bo", "dia_id": "D1:2", "text": "second", "timestamp": "2026-05-01T13:05:00+00:00"},
    ]
    with pytest.raises(ValueError, match="D1:2"):
        sources_of(item)


def test_message_count_is_the_message_sources(tmp_path: Path):
    from memory_kernel.pipeline.runner import conversation_counts

    item = conv(agent_handles=["helper_bot"])
    item["conversation"]["session_1"] += [
        {"speaker": "Bo", "dia_id": "D1:2", "text": "A long paste goes here. " * 150},
        {"speaker": "Ana", "dia_id": "D1:3", "text": "@helper_bot"},
    ]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(item))
    sources = sources_of(item)
    assert len(sources) > 3
    assert conversation_counts(p, None).messages == len(sources)
