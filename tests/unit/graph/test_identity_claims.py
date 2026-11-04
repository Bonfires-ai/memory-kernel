import pytest

from memory_kernel.graph.identity import identity_claims
from memory_kernel.graph.identity_cards import load_identity_cards

CARDS = load_identity_cards()
SPEAKERS = {"Ana", "Bo"}


def rec(speaker, roleset, roles, text="", mid="m1"):
    return {
        "message_id": mid,
        "session_id": "s1",
        "speaker": speaker,
        "text": text,
        "events": [{"roleset": roleset, "roles": roles}],
    }


def role(text, deixis_of=None):
    r = {"text": text, "span": [0]}
    if deixis_of:
        r["referent"] = {"status": "deixis", "text": deixis_of}
    return r


def pairs(records):
    return [(c.holder, c.name, c.first_person) for c in identity_claims(records, CARDS, SPEAKERS)]


def test_my_name_is():
    assert pairs([rec("Ana", "be.01", {"arg1": role("my name"), "arg2": role("Dee")}, "my name is Dee.")]) == [
        ("Ana", "Dee", True)
    ]


def test_i_am_via_deixis():
    assert pairs([rec("Ana", "be.01", {"arg1": role("I", deixis_of="Ana"), "arg2": role("Dee")}, "I am Dee.")]) == [
        ("Ana", "Dee", True)
    ]


def test_call_me():
    assert pairs(
        [rec("Bo", "call.01", {"arg1": role("me", deixis_of="Bo"), "arg2": role("Eli")}, "Please call me Eli.")]
    ) == [("Bo", "Eli", True)]


def test_third_person_equative_with_a_speaker():
    assert pairs([rec("Bo", "be.01", {"arg1": role("Dee"), "arg2": role("Ana")}, "Dee is Ana.")]) == [
        ("Dee", "Ana", False)
    ]


def test_same_person_refer():
    r = rec(
        "Ana",
        "refer.01",
        {"arg0": role("Dee and Ana"), "arg1": role("to the same person that is me")},
        "Dee and Ana refer to the same person that is me.",
    )
    assert pairs([r]) == [("Dee", "Ana", False)]


def test_same_person_copula():
    r = rec(
        "Bo",
        "be.01",
        {"arg1": role("Dee and Ana"), "arg2": role("the same person")},
        "Dee and Ana are the same person.",
    )
    assert pairs([r]) == [("Dee", "Ana", False)]


@pytest.mark.parametrize("arg2", ["happy", "a teacher", "here", "what", "him"])
def test_predicative_copula_is_not_a_claim(arg2):
    assert pairs([rec("Ana", "be.01", {"arg1": role("Dee"), "arg2": role(arg2)}, f"Dee is {arg2}.")]) == []


def test_holder_equal_to_name_is_not_a_claim():
    assert pairs([rec("Ana", "be.01", {"arg1": role("I", deixis_of="Ana"), "arg2": role("Ana")})]) == []


@pytest.mark.parametrize("day,text", [("Today", "Today is Monday."), ("Tomorrow", "Tomorrow is Friday.")])
def test_dates_are_not_names(day, text):
    other = text.split()[-1].rstrip(".")
    assert pairs([rec("Ana", "be.01", {"arg1": role(day), "arg2": role(other)}, text)]) == []


@pytest.mark.parametrize(
    "text,h,n", [("Monday is Ana.", "Monday", "Ana"), ("Friday is Saturday.", "Friday", "Saturday")]
)
def test_calendar_words_are_not_names(text, h, n):
    assert pairs([rec("Bo", "be.01", {"arg1": role(h), "arg2": role(n)}, text)]) == []


def test_negated_name_in_its_own_clause_is_no_claim_but_the_next_clause_is():
    r = rec("Ana", "be.01", {}, "i'm not Dee, i'm Eli.")
    r["events"] = [
        {"roleset": "be.01", "roles": {"arg1": role("i", deixis_of="Ana"), "arg2": role("Dee")}},
        {"roleset": "be.01", "roles": {"arg1": role("i", deixis_of="Ana"), "arg2": role("Eli")}},
    ]
    assert pairs([r]) == [("Ana", "Eli", True)]


def test_negation_before_the_name_blocks_it():
    r = rec(
        "Ana", "be.01", {"arg1": role("i", deixis_of="Ana"), "arg2": role("Dee")}, "that fails because i am not Dee"
    )
    assert pairs([r]) == []


def test_stop_calling_me_is_no_claim():
    r = rec(
        "Ana", "call.01", {"arg1": role("me", deixis_of="Ana"), "arg2": role("Dee")}, "you should stop calling me Dee"
    )
    assert pairs([r]) == []


def test_refer_to_me_as():
    r = rec("Bo", "refer.01", {"arg1": role("to me"), "arg2": role("as Eli")}, "please refer to me as Eli")
    assert pairs([r]) == [("Bo", "Eli", True)]


def test_a_filler_absent_from_the_unit_text_makes_no_claim():
    r = rec("Ana", "be.01", {"arg1": role("my name"), "arg2": role("Dee")}, "hello there")
    assert pairs([r]) == []


@pytest.mark.parametrize(
    "text",
    [
        "Dee and Ana are not the same person.",
        "Dee and Ana aren't the same person.",
        "Dee and Ana are never one and the same.",
    ],
)
def test_negated_same_person_is_no_claim(text):
    marker = "one and the same" if "one and" in text else "the same person"
    r = rec("Bo", "be.01", {"arg1": role("Dee and Ana"), "arg2": role(marker)}, text)
    assert pairs([r]) == []


def test_same_person_with_a_negation_elsewhere_is_still_a_claim():
    text = "Dee and Ana are the same person, not two."
    r = rec("Bo", "be.01", {"arg1": role("Dee and Ana"), "arg2": role("the same person")}, text)
    assert pairs([r]) == [("Dee", "Ana", False)]


def test_a_possessive_filler_is_not_a_name():
    r = rec("Ana", "be.01", {"arg1": role("I", deixis_of="Ana"), "arg2": role("Bo's")}, "I'm Bo's.")
    assert pairs([r]) == []
    r = rec("Ana", "be.01", {"arg1": role("I", deixis_of="Ana"), "arg2": role("Dee’s")}, "I am Dee’s.")
    assert pairs([r]) == []


@pytest.mark.parametrize(
    "roleset,text",
    [
        ("be.01", "If I were Dee I would leave."),
        ("be.01", "I wish I were Dee."),
        ("be.01", "Am I Dee?"),
        ("call.01", "Nobody calls me Dee anymore."),
        ("be.01", "Let's pretend I am Dee."),
    ],
)
def test_irrealis_and_question_self_claims_are_no_claim(roleset, text):
    holder = "me" if roleset == "call.01" else "I"
    r = rec("Ana", roleset, {"arg1": role(holder, deixis_of="Ana"), "arg2": role("Dee")}, text)
    assert pairs([r]) == []


def test_a_plain_self_claim_still_claims():
    r = rec("Ana", "be.01", {"arg1": role("I", deixis_of="Ana"), "arg2": role("Dee")}, "I am Dee.")
    assert pairs([r]) == [("Ana", "Dee", True)]


def test_a_same_person_question_is_no_claim():
    text = "Are Dee and Ana the same person?"
    r = rec("Bo", "be.01", {"arg1": role("Dee and Ana"), "arg2": role("the same person")}, text)
    assert pairs([r]) == []


def test_an_irrealis_cue_after_the_names_does_not_block_same_person():
    text = "Dee and Ana were the same person."
    r = rec("Bo", "be.01", {"arg1": role("Dee and Ana"), "arg2": role("the same person")}, text)
    assert pairs([r]) == [("Dee", "Ana", False)]
