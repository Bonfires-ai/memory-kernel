"""The de-renderer: units for the unit grammar, offsets back into the original turn."""

from __future__ import annotations

from memory_kernel.runtime.derender import derender

LITERALS = frozenset({"did", "does", "where", "who", "what", "the", "of", "'s", "?"})


def _by_string(units, s):
    return next(u for u in units if u.string == s)


def test_propn_run_is_one_name_unit():
    units = derender("Where does Kira Dace work?", literals=LITERALS)
    name = _by_string(units, "kira dace")
    assert name.cat == "name" and name.lemmas == ()
    assert [u.string for u in units].count("kira") == 0


def test_ambiguous_word_is_acat_with_all_lemmas():
    units = derender("talk", literals=LITERALS)
    (unit,) = units
    assert unit.cat == "acat"
    assert set(unit.lemmas) >= {("noun", "talk"), ("verb", "talk")}


def test_closed_class_literal_gets_no_open_alternative():
    units = derender("Where did Milo work?", literals=LITERALS)
    did = _by_string(units, "did")
    assert (did.cat, did.lemmas) == ("lit", ())
    assert _by_string(units, "where").cat == "lit"


def test_every_lexical_unit_carries_a_tuple_with_a_lemma():
    for u in derender("Milo published three quick reports, then slept! \U0001f600", literals=LITERALS):
        assert isinstance(u.lemmas, tuple)
        if u.cat in {"noun", "verb", "adj", "num", "prep"}:
            assert u.lemmas and all(isinstance(x, str) for x in u.lemmas)
        if u.cat == "acat":
            assert all(len(p) == 2 for p in u.lemmas)
    pub = _by_string(derender("Milo published it", literals=frozenset()), "published")
    assert (pub.cat, pub.lemmas) == ("verb", ("publish",))


def test_offsets_cut_back_to_the_original():
    text = "Where is Jean-Paul Vorn's \U0001f600 report?"
    units = derender(text, literals=LITERALS)
    got = [(u.string, u.start, u.end) for u in units]
    assert got == [
        ("where", 0, 5),
        ("is", 6, 8),
        ("jean - paul vorn", 9, 23),
        ("'s", 23, 25),
        ("\U0001f600", 26, 27),
        ("report", 28, 34),
        ("?", 34, 35),
    ]
    norm = units.spans.normalised
    left = norm.index("jean")
    right = norm.index("report") + len("report")
    assert units.spans.cut(left, right) == "Jean-Paul Vorn's \U0001f600 report"
    lone_text = "Ana \ud800 Milo"
    lone = derender(lone_text, literals=LITERALS)
    # how the parser tags the stand-in depends on the model, so only the offsets are pinned: on the original text
    assert all(0 <= u.start < u.end <= len(lone_text) for u in lone)
    assert lone.units[0].start == 0 and lone.units[-1].end == len(lone_text)
    assert lone.spans.cut(*(lambda n: (n.index("milo"), len(n)))(lone.spans.normalised)) == "Milo"


def test_name_string_collapses_inner_whitespace():
    units = derender("Kira\nDace works", literals=frozenset())
    name = units.units[0]
    assert (name.string, name.cat, name.start, name.end) == ("kira dace", "name", 0, 9)


def test_glued_capitalised_token_is_a_name_whatever_its_pos():
    for apostrophe in ("'", "\u2019"):
        text = f"Where does O{apostrophe}Neil work?"
        units = derender(text, literals=LITERALS)
        name = _by_string(units, f"o{apostrophe}neil")
        assert (name.cat, name.lemmas) == ("name", ()), apostrophe
        assert text[name.start : name.end] == f"O{apostrophe}Neil"
    for text in ("Don't go", "isn't it", "Isn't it"):
        assert all(u.cat != "name" for u in derender(text, literals=frozenset())), text


def test_name_run_negative_controls():
    assert _by_string(derender("Milo works", literals=frozenset()), "milo").cat == "name"
    split = derender("Ana saw Milo", literals=frozenset())
    assert [u.string for u in split if u.cat == "name"] == ["ana", "milo"]
    spaced = derender("Jean - Paul Vorn works", literals=frozenset())
    assert [u.string for u in spaced if u.cat == "name"][-1] == "paul vorn"
    assert "jean" in [u.string for u in spaced]


def test_unambiguous_words_keep_their_class():
    # morphy: "city" has no verb reading, "ran" no noun reading
    noun = _by_string(derender("the city", literals=frozenset()), "city")
    assert (noun.cat, noun.lemmas) == ("noun", ("city",))
    verb = _by_string(derender("Milo ran", literals=frozenset()), "ran")
    assert (verb.cat, verb.lemmas) == ("verb", ("run",))


def test_auxiliaries_are_verbs_never_acat():
    units = derender("Where does Milo work? Is it done?", literals=frozenset())
    for word in ("does", "is"):
        unit = _by_string(units, word)
        assert unit.cat == "verb" and unit.lemmas in (("do",), ("be",)), unit


def test_np_keeps_quantified_flag():
    units = derender("Every team shipped it", literals=frozenset())
    norm = units.spans.normalised
    cut = units.spans.np(norm.index("every"), norm.index("team") + 4)
    assert (cut.text, cut.quantified) == ("team", True)
    plain = derender("The team shipped it", literals=frozenset())
    cut = plain.spans.np(0, plain.spans.normalised.index("team") + 4)
    assert (cut.text, cut.quantified) == ("team", False)
