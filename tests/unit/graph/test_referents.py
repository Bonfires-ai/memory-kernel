"""Referent resolution's pure core. Invented names only."""

from __future__ import annotations

import pytest

from memory_kernel.graph import referents as rf
from memory_kernel.graph.g1_referring import owner_key

CHAT_A, CHAT_B = "tg-ember-a", "tg-ember-b"


def row(mid, speaker, at, text="", addressees=()):
    return {"message_id": mid, "speaker": speaker, "timestamp": at, "text": text, "addressees": list(addressees)}


@pytest.mark.parametrize(
    ("text", "form"),
    [
        ("Ilse Varga", "named"),
        ("@kiln_bot", "named"),
        ("her kiln", "named"),
        ("kilns", "kind"),
        ("clay", "kind"),
        ("it", "pronoun"),
        ("a kiln", "indefinite"),
        ("the kiln", "specific"),
        ("this kiln", "specific"),
        ("at the moment", "specific"),
    ],
)
def test_mention_form_reads_the_parse(text, form):
    assert rf.mention_form(text) == form


@pytest.mark.parametrize(
    ("text", "context", "form"),
    [
        ("Ilse", "I asked Ilse about the kiln", "named"),
        ("the kiln", "Ilse fixed the kiln yesterday", "specific"),
        ("kilns", "unrelated words", "kind"),
    ],
)
def test_mention_form_parses_inside_the_message(text, context, form):
    assert rf.mention_form(text, context) == form


def test_is_question_reads_the_mood():
    assert rf.is_question("did the kiln crack?") is True
    assert rf.is_question("the kiln cracked.") is False


def test_discriminated_key_is_stable():
    k = rf.discriminated_key("kiln", "tg:ember:9")
    assert k.startswith("kiln#") and len(k) == len("kiln#") + 8 and k == rf.discriminated_key("kiln", "tg:ember:9")


def test_fold_ground_grows_and_drops_agents():
    rows = {
        "m1": row("m1", "Ilse", "2026-07-01T10:00:00+00:00", addressees=["kiln_bot"]),
        "m2": row("m2", "Oren", "2026-07-03T10:00:00+00:00", addressees=["Ilse"]),
    }
    chat_of = {"m1": CHAT_A, "m2": CHAT_B}.__getitem__
    g = rf.fold_ground({}, {"e-kiln": ["m1"]}, rows, chat_of, {"kiln_bot"}, legacy=True)
    g = rf.fold_ground(g, {"e-kiln": ["m2"]}, rows, chat_of, {"kiln_bot"}, legacy=False)
    k = g["e-kiln"]
    assert k.chats == (CHAT_A, CHAT_B) and k.people == ("ilse", "oren")
    assert (k.first_mid, k.last_mid, k.last_at, k.mentions) == ("m1", "m2", "2026-07-03T10:00:00+00:00", 2)
    assert rf.Ground.from_row(k.to_row()) == k


def plan(points, rows, ground, stored_keys, chat=CHAT_B):
    return rf.plan_session(points, chat=chat, rows=rows, ground=ground, stored_keys=stored_keys, agents={"kiln_bot"})


def ground_of(uuid, chats, people, last_at, mid="m0"):
    return rf.Ground(uuid, tuple(chats), tuple(people), mid, mid, last_at, 3)


def test_a_candidate_outside_the_common_ground_gives_a_new_referent():
    rows = {"m9": row("m9", "Tamsin", "2026-07-10T10:00:00+00:00", "the kiln cracked")}
    ground = {"e-kiln": ground_of("e-kiln", [CHAT_A], ["ilse"], "2026-07-01T10:00:00+00:00")}
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", "e-kiln")
    provisional, hard, counts = plan([p], rows, ground, {"e-kiln": "kiln"})
    assert provisional == {"u#0|arg0": {"session": 0}} and hard == [] and counts["new_no_ground"] == 1


def test_a_recent_same_chat_candidate_is_an_easy_join():
    rows = {"m9": row("m9", "Tamsin", "2026-07-02T10:00:00+00:00", "the kiln cracked")}
    ground = {"e-kiln": ground_of("e-kiln", [CHAT_B], ["ilse"], "2026-07-01T10:00:00+00:00")}
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", "e-kiln")
    provisional, hard, _ = plan([p], rows, ground, {"e-kiln": "kiln"})
    assert provisional == {"u#0|arg0": {"join": "e-kiln"}} and hard == []


def test_a_shared_but_stale_candidate_is_a_hard_case():
    rows = {"m9": row("m9", "Ilse", "2026-08-20T10:00:00+00:00", "the kiln cracked")}
    ground = {"e-kiln": ground_of("e-kiln", [CHAT_A], ["ilse"], "2026-07-01T10:00:00+00:00")}
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", "e-kiln")
    provisional, hard, _ = plan([p], rows, ground, {"e-kiln": "kiln"})
    assert provisional == {"u#0|arg0": {"hard": 0, "session": 0}}
    assert hard == [rf.HardCase(0, p, ("e-kiln",))]


def test_discriminated_variants_are_candidates_most_recent_first():
    rows = {"m9": row("m9", "Ilse", "2026-08-20T10:00:00+00:00")}
    ground = {
        "e-1": ground_of("e-1", [CHAT_A], ["ilse"], "2026-07-01T10:00:00+00:00"),
        "e-2": ground_of("e-2", [CHAT_A], ["ilse"], "2026-07-05T10:00:00+00:00"),
    }
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", "e-1")
    _, hard, _ = plan([p], rows, ground, {"e-1": "kiln", "e-2": "kiln#0a1b2c3d"})
    assert hard[0].candidates == ("e-2", "e-1")


def test_named_kind_and_pronoun_points_keep_the_default():
    rows = {"m9": row("m9", "Ilse", "2026-08-20T10:00:00+00:00")}
    ground = {"e-k": ground_of("e-k", [CHAT_A], ["ilse"], "2026-07-01T10:00:00+00:00")}
    pts = [
        rf.Point("u#0|arg0", "m9", "kiln", "Kiln", "Kiln", "e-k"),
        rf.Point("u#1|arg0", "m9", "kiln", "kilns", "kilns", "e-k"),
        rf.Point("u#2|arg0", "m9", "kiln", "it", "it", "e-k"),
    ]
    provisional, hard, _ = plan(pts, rows, ground, {"e-k": "kiln"})
    assert provisional == {} and hard == []


def test_an_indefinite_statement_introduces_and_a_question_does_not():
    rows = {
        "m8": row("m8", "Ilse", "2026-08-20T10:00:00+00:00", "a kiln arrived."),
        "m9": row("m9", "Ilse", "2026-08-20T11:00:00+00:00", "did a kiln arrive?"),
    }
    ground = {"e-k": ground_of("e-k", [CHAT_B], ["ilse"], "2026-08-20T09:00:00+00:00")}
    intro = rf.Point("u#0|arg0", "m8", "kiln", "a kiln", "a kiln", "e-k")
    asked = rf.Point("u#1|arg0", "m9", "kiln", "a kiln", "a kiln", "e-k")
    provisional, _, _ = plan([intro, asked], rows, ground, {"e-k": "kiln"})
    assert provisional["u#0|arg0"] == {"session": 0}
    assert provisional["u#1|arg0"] == {"session": 0}  # the question asks about the one just introduced


def test_one_new_referent_takes_all_its_session_mentions():
    rows = {f"m{i}": row(f"m{i}", "Tamsin", f"2026-07-10T1{i}:00:00+00:00") for i in range(3)}
    ground = {"e-kiln": ground_of("e-kiln", [CHAT_A], ["ilse"], "2026-07-01T10:00:00+00:00")}
    pts = [rf.Point(f"u{i}#0|arg0", f"m{i}", "kiln", "the kiln", "the kiln", "e-kiln") for i in range(3)]
    provisional, hard, _ = plan(pts, rows, ground, {"e-kiln": "kiln"})
    assert set(map(str, provisional.values())) == {"{'session': 0}"} and hard == []
    final = rf.settle(provisional, hard, {}, session_keys={0: ("kiln", "m0")})
    assert {d["new"] for d in final.values()} == {rf.discriminated_key("kiln", "m0")}


def test_settle_joins_the_best_yes_at_tau():
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", "e-1")
    hard = [rf.HardCase(0, p, ("e-2", "e-1"))]
    provisional = {"u#0|arg0": {"hard": 0, "session": 0}, "u#1|arg0": {"session": 0}}
    keys = {0: ("kiln", "m9")}
    joined = rf.settle(provisional, hard, {(0, "e-2"): 0.4, (0, "e-1"): 0.81}, session_keys=keys)
    assert joined == {"u#0|arg0": {"join": "e-1"}, "u#1|arg0": {"join": "e-1"}}  # the session follows its referent
    kept = rf.settle(provisional, hard, {(0, "e-2"): 0.4, (0, "e-1"): 0.49}, session_keys=keys)
    assert kept["u#0|arg0"] == {"new": rf.discriminated_key("kiln", "m9")}


def test_settle_refuses_a_missing_answer():
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", "e-1")
    with pytest.raises(KeyError):
        rf.settle({"u#0|arg0": {"hard": 0, "session": 0}}, [rf.HardCase(0, p, ("e-1",))], {},
                  session_keys={0: ("kiln", "m9")})  # fmt: skip


# -- the kind/instance split (spec 2b): the plain key names the kind --------------------------------------------------


def legacy_ground(uuid, last_at, *, legacy, chats=(CHAT_B,), people=("ilse",)):
    return rf.Ground(uuid, tuple(chats), tuple(people), "m0", "m0", last_at, 3, legacy=legacy)


def test_new_specific_referent_is_always_discriminated():
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", None)
    final = rf.settle({"u#0|arg0": {"session": 0}}, [], {}, session_keys={0: ("kiln", "m9")})
    assert final == {"u#0|arg0": {"new": rf.discriminated_key("kiln", "m9")}}
    assert p.default is None


def test_post_split_plain_node_is_not_a_candidate():
    rows = {"m9": row("m9", "Tamsin", "2026-07-02T10:00:00+00:00", "the kiln cracked")}
    ground = {"e-kiln": legacy_ground("e-kiln", "2026-07-01T10:00:00+00:00", legacy=False)}
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", "e-kiln")
    provisional, hard, _ = plan([p], rows, ground, {"e-kiln": "kiln"})
    assert provisional == {"u#0|arg0": {"session": 0}} and hard == []


def test_legacy_plain_node_is_a_candidate():
    rows = {"m9": row("m9", "Tamsin", "2026-07-02T10:00:00+00:00", "the kiln cracked")}
    ground = {"e-kiln": legacy_ground("e-kiln", "2026-07-01T10:00:00+00:00", legacy=True)}
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", "e-kiln")
    provisional, hard, _ = plan([p], rows, ground, {"e-kiln": "kiln"})
    assert provisional == {"u#0|arg0": {"join": "e-kiln"}} and hard == []
    rows = {"m9": row("m9", "Ilse", "2026-08-20T10:00:00+00:00", "the kiln cracked")}
    ground = {"e-kiln": legacy_ground("e-kiln", "2026-07-01T10:00:00+00:00", legacy=True, chats=[CHAT_A])}
    provisional, hard, _ = plan([p], rows, ground, {"e-kiln": "kiln"})
    assert hard == [rf.HardCase(0, p, ("e-kiln",))]


def test_discriminated_node_is_a_candidate_whatever_legacy():
    rows = {"m9": row("m9", "Tamsin", "2026-07-02T10:00:00+00:00", "the kiln cracked")}
    ground = {"e-1": legacy_ground("e-1", "2026-07-01T10:00:00+00:00", legacy=False)}
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", None)
    provisional, hard, _ = plan([p], rows, ground, {"e-1": "kiln#0a1b2c3d"})
    assert provisional == {"u#0|arg0": {"join": "e-1"}} and hard == []


def test_a_ground_uuid_without_a_stored_key_is_not_discriminated():
    rows = {"m9": row("m9", "Tamsin", "2026-07-02T10:00:00+00:00", "the kiln cracked")}
    ground = {"e-1": legacy_ground("e-1", "2026-07-01T10:00:00+00:00", legacy=False)}
    p = rf.Point("u#0|arg0", "m9", "kiln", "the kiln", "the kiln", "e-1")
    provisional, _, _ = plan([p], rows, ground, {})
    assert provisional == {"u#0|arg0": {"session": 0}}


def test_fold_ground_legacy_is_set_once():
    rows = {"m1": row("m1", "Ilse", "2026-07-01T10:00:00+00:00"), "m2": row("m2", "Ilse", "2026-07-02T10:00:00+00:00")}
    first = rf.fold_ground({}, {"e-old": ["m1"]}, rows, lambda _m: CHAT_A, (), legacy=True)
    out = rf.fold_ground(first, {"e-old": ["m2"], "e-new": ["m2"]}, rows, lambda _m: CHAT_A, (), legacy=False)
    assert out["e-old"].legacy is True and out["e-new"].legacy is False


def test_ground_row_without_legacy_reads_as_legacy():
    g = ground_of("e-k", [CHAT_A], ["ilse"], "2026-07-01T10:00:00+00:00")
    old = {k: v for k, v in g.to_row().items() if k != "legacy"}
    assert rf.Ground.from_row(old).legacy is True
    assert rf.Ground.from_row({**old, "legacy": False}).legacy is False
    assert g.to_row()["legacy"] is True


def test_split_discriminated_matches_only_a_trailing_8_hex():
    assert rf.split_discriminated("kiln#0a1b2c3d") == ("kiln", True)
    assert rf.split_discriminated("c#") == ("c#", False)
    assert rf.split_discriminated("c# compiler") == ("c# compiler", False)
    assert rf.split_discriminated("#general channel") == ("#general channel", False)
    assert rf.split_discriminated("kiln#0A1B2C3D") == ("kiln#0A1B2C3D", False)
    assert rf.split_discriminated("kiln#0a1b2c3") == ("kiln#0a1b2c3", False)


def test_a_hash_bearing_plain_post_split_node_is_not_a_candidate():
    rows = {"m9": row("m9", "Tamsin", "2026-07-02T10:00:00+00:00", "the c# compiler cracked")}
    ground = {"e-c": legacy_ground("e-c", "2026-07-01T10:00:00+00:00", legacy=False)}
    p = rf.Point("u#0|arg0", "m9", "c# compiler", "the c# compiler", "the c# compiler", "e-c")
    provisional, hard, _ = plan([p], rows, ground, {"e-c": "c# compiler"})
    assert provisional == {"u#0|arg0": {"session": 0}} and hard == []


def test_hash_led_keys_are_not_one_family():
    rows = {"m9": row("m9", "Tamsin", "2026-07-02T10:00:00+00:00", "the #general channel")}
    ground = {"e-r": legacy_ground("e-r", "2026-07-01T10:00:00+00:00", legacy=False)}
    p = rf.Point("u#0|arg0", "m9", "#general channel", "the #general channel", "the #general channel", None)
    provisional, _, _ = plan([p], rows, ground, {"e-r": "#random channel#0a1b2c3d"})
    assert provisional == {"u#0|arg0": {"session": 0}}


def test_two_referents_of_one_key_in_one_message_get_distinct_keys():
    provisional = {"u#0|arg0": {"session": 0}, "u#1|arg0": {"session": 1}}
    keys = {0: ("red kiln", "m9"), 1: ("red kiln", "m9")}
    final = rf.settle(provisional, [], {}, session_keys=keys)
    a, b = final["u#0|arg0"]["new"], final["u#1|arg0"]["new"]
    assert (
        a != b and rf.split_discriminated(a) == ("red kiln", True) and rf.split_discriminated(b) == ("red kiln", True)
    )
    assert rf.settle(provisional, [], {}, session_keys=keys) == final  # a replay gives the same bytes
    assert a == rf.discriminated_key("red kiln", "m9")  # the first keeps the spec's key


@pytest.mark.parametrize(
    ("surface", "addressee", "expected", "mention"),
    [
        ("my family", "Milo", ("ana's family", "Ana's family"), "possessive"),
        ("with my family", "Milo", ("ana's family", "Ana's family"), "possessive"),
        ("your dog", "Milo", ("milo's dog", "Milo's dog"), "possessive"),
        ("your dog", None, None, "possessive"),
        ("his dog", "Milo", None, "possessive"),
        ("the dog", "Milo", None, "definite"),
        ("a dog", "Milo", None, "indefinite"),
    ],
)
def test_owner_key_reads_the_pole_from_the_pronouns_person(surface, addressee, expected, mention):
    from memory_kernel.construction.referents import classify_mention
    from memory_kernel.graph.g1_referring import _without_leading_adposition

    assert owner_key(surface, speaker="Ana", addressee=addressee) == expected
    assert classify_mention(_without_leading_adposition(surface)).mention == mention
