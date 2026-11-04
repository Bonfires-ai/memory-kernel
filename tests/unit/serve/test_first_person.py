"""``resolve_first_person``: a question's first-person words resolve to the asker; "you" stays the agent."""

from __future__ import annotations

import pytest

from memory_kernel.serve.answer import resolve_first_person


@pytest.mark.parametrize(
    ("question", "resolved"),
    [
        ("@helper_bot what do you know about me?", "@helper_bot what do you know about Ana?"),
        ("what's my favourite fruit", "what's Ana's favourite fruit"),
        ("what did I say about mine?", "what did Ana say about Ana's?"),
        ("I'm wondering what I'll do", "Ana is wondering what Ana will do"),
        ("I've asked before; I'd like to know", "Ana has asked before; Ana would like to know"),
        ("tell me about myself", "tell Ana about Ana"),
        ("what do i like", "what do Ana like"),
        ("what do you know about Milo?", "what do you know about Milo?"),
        ("mime, mentor, India, Iris", "mime, mentor, India, Iris"),
    ],
)
def test_first_person_words_resolve_to_the_speaker(question: str, resolved: str) -> None:
    assert resolve_first_person(question, "Ana") == resolved


@pytest.mark.parametrize("speaker", [None, "", "  ", "@"])
def test_no_speaker_leaves_the_question_alone(speaker: str | None) -> None:
    assert resolve_first_person("what about me?", speaker) == "what about me?"


def test_a_leading_at_is_dropped() -> None:
    assert resolve_first_person("about me", "@Ana") == "about Ana"
