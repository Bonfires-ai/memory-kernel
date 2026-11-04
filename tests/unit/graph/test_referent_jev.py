"""The Jev yes/no wire of a hard referent case. Invented names only."""

from __future__ import annotations

import pytest

from memory_kernel.graph import referent_jev as rj
from memory_kernel.graph.referents import Ground, HardCase, Point


def test_excerpt_centres_on_the_focus_word():
    text = "x" * 900 + " the kiln cracked " + "y" * 900
    out = rj.excerpt(text, "kiln")
    assert "the kiln cracked" in out and len(out) <= 503 and out.startswith("...")


def test_excerpt_keeps_a_short_text_whole():
    assert rj.excerpt("the kiln cracked", "kiln") == "the kiln cracked"


def test_question_wire_carries_both_sides_and_the_criteria():
    rows = {
        "m9": {
            "message_id": "m9",
            "speaker": "Ilse",
            "timestamp": "2026-08-20T10:00:00+00:00",
            "text": "the kiln cracked",
        },
        "m8": {"message_id": "m8", "speaker": "Ilse", "timestamp": "2026-08-20T09:00:00+00:00", "text": "morning"},
        "m1": {
            "message_id": "m1",
            "speaker": "Oren",
            "timestamp": "2026-07-01T10:00:00+00:00",
            "text": "a kiln arrived",
        },
    }
    ground = {"e-k": Ground("e-k", ("tg-a",), ("oren",), "m1", "m1", "2026-07-01T10:00:00+00:00", 1)}
    case = HardCase(0, Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", "e-k"), ("e-k",))
    sealed = rj.question_wire(case, "e-k", rows=rows, before=["m8"], ground=ground, chat="tg-b", names={"e-k": "kiln"})
    wire = sealed["batches"][0]
    state = wire["state"]
    assert state["new_message"]["text"] == "the kiln cracked" and state["new_message_context"][0]["text"] == "morning"
    assert state["earlier_messages_where"] == "another chat" and state["earlier_first_mention"]["speaker"] == "Oren"
    q = wire["questions"]["same"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"yes", "no"} and "not enough" in q["criteria"]["yes"]


def test_answer_yes_refuses_a_missing_answer():
    assert rj.answer_yes({"same": {"probabilities": {"yes": 0.8, "no": 0.2}}}) == 0.8
    with pytest.raises(KeyError):
        rj.answer_yes({})
