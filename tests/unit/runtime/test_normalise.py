"""The normaliser: the text the runtime grammar reads, and the cut back to the surface as written."""

from __future__ import annotations

import pytest

from memory_kernel.runtime.normalise import SENTINEL_END, SENTINEL_START, normalise


def _span(norm, piece: str) -> tuple[int, int]:
    left = norm.text.index(piece)
    return left, left + len(piece)


def test_cut_returns_original_surface():
    norm = normalise("Where does Jean-Paul Vorn's report say it?")
    assert norm.text == "⟦ where do jean - paul vorn 's report ⟪say⟫ it ? ⟧"
    assert norm.spans.cut(*_span(norm, "jean - paul vorn 's report")) == "Jean-Paul Vorn's report"

    norm = normalise("He worked at O'Neil")
    assert norm.text == "⟦ he ⟪work⟫ at o'neil ⟧"
    assert norm.spans.cut(*_span(norm, "o'neil")) == "O'Neil"
    assert norm.spans.cut(*_span(norm, "⟪work⟫ at o'neil")) == "worked at O'Neil"


def test_whitespace_runs_and_astral_characters_keep_offsets_honest():
    text = "Where  is\tMilo's \U0001f600 cat?"
    norm = normalise(text)
    assert norm.text == "⟦ where be milo 's � cat ? ⟧"  # one space between tokens; the engine never sees a surrogate
    assert norm.spans.cut(*_span(norm, "milo 's � cat")) == "Milo's \U0001f600 cat"
    lone = normalise("Ana \ud800 Milo")
    assert lone.text == "⟦ ana \ufffd milo ⟧" and lone.spans.cut(6, 7) == "\ud800"
    for left, right in ((3, 3), (-1, 2), (0, len(norm.text) + 1)):
        with pytest.raises(ValueError):
            norm.spans.cut(left, right)


def test_sentinels_wrap_the_text_and_cut_to_nothing():
    """Every normalised text starts with ⟦ and ends with ⟧ (edge anchors); offsets inside still cut to the surface."""
    norm = normalise("brew a tea")
    assert norm.text == f"{SENTINEL_START} ⟪brew⟫ a tea {SENTINEL_END}"
    assert norm.spans.cut(*_span(norm, "a tea")) == "tea"  # the NP cut drops a leading determiner (Task 6b)
    assert norm.spans.cut(0, len(norm.text)) == "brew a tea"  # a span over the sentinels cuts the tokens inside
    for left, right in ((0, 1), (len(norm.text) - 1, len(norm.text))):
        with pytest.raises(ValueError):
            norm.spans.cut(left, right)
    assert normalise("").text == f"{SENTINEL_START} {SENTINEL_END}"
    forged = normalise(f"Ana {SENTINEL_END} Milo")  # a turn cannot forge an edge
    assert forged.text == f"{SENTINEL_START} ana \ufffd milo {SENTINEL_END}"


def test_verb_marking_reads_pos_and_aux_dependents():
    """A VERB (never an AUX) is written ⟪lemma⟫; so is a non-AUX token governing an ``aux`` (spaCy tags "work" in "Where
    does Kira Dace work?" as a NOUN heading the auxiliary). A marker in the input is written as U+FFFD."""
    assert normalise("Where does Kira Dace work?").text == "⟦ where do kira dace ⟪work⟫ ? ⟧"
    assert normalise("Where does Milo work?").text == "⟦ where do milo ⟪work⟫ ? ⟧"
    assert normalise("the work of Kira Dace").text == "⟦ the work of kira dace ⟧"
    assert normalise("Where has Kira been?").text == "⟦ where have kira be ? ⟧"  # an AUX head stays unmarked
    assert normalise("Ana ⟪ Milo ⟫").text == "⟦ ana � milo � ⟧"
