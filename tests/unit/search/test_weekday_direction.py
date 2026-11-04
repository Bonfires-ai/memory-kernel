"""Which way a weekday with no ``last``/``next`` points (``temporal_deixis._bare_weekday_resolution_notes``).

"this Friday" and "this coming Friday" name the Friday ahead whatever the sentence's tense: a team says it of a plan
("we demo the plugin on staging this Friday", present tense), and the note under it must not date the plan to the
Friday just gone. A bare "Friday" keeps its rule (forward only with future tense or a nearby ``next``), so the past
reading of a story told in the past is unchanged. Offline; invented sentences only."""

from __future__ import annotations

import pytest

from memory_kernel.knowledge.temporal_deixis import relative_resolution_note

THURSDAY = "2026-10-29T02:18:55Z"
FRIDAY = "2026-10-30T09:00:00Z"


@pytest.mark.parametrize(
    "text",
    [
        "Remember the decision that we demo the plugin on staging this Friday.",
        "We demo the plugin on staging this coming Friday.",
        "This Friday the hall is ours.",
    ],
)
def test_this_weekday_reads_forward_whatever_the_tense(text):
    assert relative_resolution_note(text, THURSDAY) == "friday = 30 October 2026"


def test_this_weekday_said_on_that_day_is_the_same_day():
    assert relative_resolution_note("We demo this Friday.", FRIDAY) == "friday = 30 October 2026"


def test_a_bare_weekday_in_a_past_story_still_reads_back():
    assert relative_resolution_note("We demoed the plugin on Friday.", THURSDAY) == "friday = 23 October 2026"


def test_a_bare_weekday_in_a_future_sentence_still_reads_forward():
    assert relative_resolution_note("We will demo the plugin on Friday.", THURSDAY) == "friday = 30 October 2026"
