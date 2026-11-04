import pytest

from memory_kernel.pipeline.stages._addressing import agent_mentions

H = ["@helper_bot"]


@pytest.mark.parametrize(
    ("text", "out"),
    [
        ("@helper_bot what does Ana like?", "what does Ana like?"),
        ("@helper_bot, what does Ana like?", "what does Ana like?"),
        ("what does Ana like @helper_bot?", "what does Ana like?"),
        ("what does Ana like? @helper_bot", "what does Ana like?"),
        ("ask @helper_bot about it", "ask helper_bot about it"),
        ("@Helper_Bot I am Ana.", "I am Ana."),
        ("@helper_bot @helper_bot hi", "hi"),
    ],
)
def test_mention_positions(text, out):
    stripped, names = agent_mentions(text, H)
    assert stripped == out
    assert names == ["helper_bot"]


@pytest.mark.parametrize(
    "text", ["@helper_bot2 hi", "hash x@helper_bot", "no mention here", "visit @helper_bot.com today"]
)
def test_not_a_mention(text):
    assert agent_mentions(text, H) == (text, [])


def test_mention_only_message_becomes_empty():
    assert agent_mentions("@helper_bot ?", H) == ("", ["helper_bot"])


def test_mention_adds_addressee_and_blank_is_dropped():
    from memory_kernel.pipeline.stages._conversation import build_prepared, split_conversation

    c = {
        "speaker_a": "Ana",
        "speaker_b": "Bo",
        "agent_handles": ["helper_bot"],
        "session_1_date_time": "1:00 pm on 1 May, 2026",
        "session_1": [
            {"speaker": "Ana", "dia_id": "D1:1", "text": "@helper_bot I love pears."},
            {"speaker": "Bo", "dia_id": "D1:2", "text": "@helper_bot"},
        ],
    }
    rows = build_prepared(split_conversation({"sample_id": "s1", "conversation": c, "qa": []})[0])[0]
    assert [r["text"] for r in rows] == ["I love pears."]
    assert rows[0]["metadata"]["addressees"] == ["helper_bot"]


def test_leading_strip_keeps_a_minus_sign():
    assert agent_mentions("@helper_bot -5 degrees", H) == ("-5 degrees", ["helper_bot"])


def test_explicit_addressee_and_mention_of_one_agent_are_one_addressee():
    from memory_kernel.pipeline.stages._conversation import build_prepared, split_conversation

    c = {
        "agent_handles": ["@helper_bot"],
        "session_1_date_time": "1:00 pm on 1 May, 2026",
        "session_1": [
            {"speaker": "Ana", "dia_id": "D1:1", "text": "@helper_bot do you know?", "addressees": ["@Helper_Bot"]},
        ],
    }
    rows = build_prepared(split_conversation({"sample_id": "s1", "conversation": c, "qa": []})[0])[0]
    assert rows[0]["metadata"]["addressees"] == ["helper_bot"]
