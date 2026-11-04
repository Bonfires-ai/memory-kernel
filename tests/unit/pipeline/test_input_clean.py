import pytest

from memory_kernel.pipeline.stages._conversation import build_prepared, split_conversation
from memory_kernel.pipeline.stages._input_clean import clean_text, split_parts


@pytest.mark.parametrize(
    ("text", "out"),
    [
        ("see https://example.org/a/b?c=1 now", "see [link: example.org] now"),
        ("see www.example.org/x now", "see [link: example.org] now"),
        ("img data:text/html;base64,PGh0bWw+ end", "img [embedded web page] end"),
        ("<html><body><p>Hi <b>Bo</b></p><script>x()</script></body></html>", "Hi Bo"),
        ("key " + "a" * 90 + " end", "key [blob] end"),
        ("plain words stay as they are.", "plain words stay as they are."),
    ],
)
def test_clean_text(text, out):
    assert clean_text(text) == out


def test_split_parts_respects_limits():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    parts = split_parts(text, max_chars=200, max_sentences=5)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)
    assert " ".join(parts) == text


def test_short_text_is_one_part():
    assert split_parts("short.") == ["short."]


def _conv(turns):
    c = {"speaker_a": "Ana", "speaker_b": "Bo", "session_1_date_time": "1:00 pm on 1 May, 2026", "session_1": turns}
    return split_conversation({"sample_id": "s1", "conversation": c, "qa": []})[0]


def test_long_turn_becomes_parts_with_suffixed_ids():
    long = " ".join(f"Line {i} of the paste." for i in range(400))
    rows = build_prepared(_conv([{"speaker": "Ana", "dia_id": "m1", "text": long}]))[0]
    assert [r["metadata"]["source_message_id"] for r in rows][:2] == ["m1#1", "m1#2"]
    assert all(len(r["text"]) <= 1900 for r in rows)


def test_part_id_collision_is_refused():
    long = " ".join(f"Line {i} of the paste." for i in range(400))
    turns = [{"speaker": "Ana", "dia_id": "m1#1", "text": "x"}, {"speaker": "Ana", "dia_id": "m1", "text": long}]
    with pytest.raises(ValueError, match="m1#1"):
        build_prepared(_conv(turns))


def test_split_turn_keeps_its_image_caption_on_the_first_part_only():
    long = " ".join(f"Line {i} of the paste." for i in range(400))
    turn = {"speaker": "Ana", "dia_id": "m1", "text": long, "blip_caption": "a pear on a table"}
    rows = [r for b in build_prepared(_conv([turn])) for r in b]
    captions = [r for r in rows if r["metadata"]["source_kind"] == "image_context"]
    assert len(captions) == 1
    assert captions[0]["metadata"]["parent_source_message_id"] == "m1#1"


@pytest.mark.parametrize("markup", ["<br>", "<div></div>", "<html><script>x()</script></html>"])
def test_markup_only_turn_is_dropped_like_a_blank_one(markup):
    turns = [{"speaker": "Ana", "dia_id": "m1", "text": markup}, {"speaker": "Bo", "dia_id": "m2", "text": "hi"}]
    rows = build_prepared(_conv(turns))[0]
    assert [r["metadata"]["source_message_id"] for r in rows] == ["m2"]


def test_malformed_url_does_not_raise():
    assert clean_text("see http://[abc now") == "see [link: link] now"


def test_unpunctuated_lines_count_as_sentences():
    text = "\n".join(f"list entry number {i:02d} without any punctuation" for i in range(40))
    assert len(text) < 1900
    parts = split_parts(text, max_sentences=20)
    assert len(parts) >= 2
    assert all(p.count("list entry") <= 20 for p in parts)
    assert sum(p.count("list entry") for p in parts) == 40


@pytest.mark.parametrize("text", ["if x < b and y > 3 then", "metadata:abc,def"])
def test_text_that_only_looks_like_markup_is_unchanged(text):
    assert clean_text(text) == text


def test_sentence_punctuation_stays_after_a_link():
    assert clean_text("Check https://x.org/a. Then go") == "Check [link: x.org]. Then go"
