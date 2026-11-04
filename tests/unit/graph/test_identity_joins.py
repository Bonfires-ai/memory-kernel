import pytest

from memory_kernel.grammar.dependencies import wordnet_ready
from memory_kernel.graph import identity
from memory_kernel.graph.identity import Claim, alias_occurs, is_common_noun, resolve_joins

SPEAKERS = {"Ana", "Bo", "Cy"}


def c(holder, name, first=False, mid="m"):
    return Claim(holder, name, "equative", first, mid, holder if first else "Cy")


def test_self_declaration_joins_the_speaker():
    j = resolve_joins([c("Ana", "Dee", first=True)], SPEAKERS)
    assert j.canonical == {"dee": "ana"}
    assert j.aliases == {"ana": ("Dee",)} and j.display == {"ana": "Ana"}


def test_third_person_chain_reaches_the_speaker():
    j = resolve_joins([c("Dee", "Ana"), c("Eli", "Dee")], SPEAKERS)
    assert j.canonical == {"dee": "ana", "eli": "ana"}


def test_two_speakers_never_merge():
    j = resolve_joins([c("Ana", "Bo")], SPEAKERS)
    assert j.canonical == {} and len(j.conflicts) == 1


def test_one_name_claimed_by_two_speakers_is_a_conflict():
    j = resolve_joins([c("Ana", "Dee", first=True), c("Bo", "Dee", first=True)], SPEAKERS)
    assert j.canonical == {} and len(j.conflicts) == 1


def test_third_person_cannot_move_a_self_declared_name():
    j = resolve_joins([c("Ana", "Dee", first=True), c("Dee", "Bo")], SPEAKERS)
    assert j.canonical == {"dee": "ana"} and len(j.conflicts) == 1


def test_a_contested_name_is_blocked_for_every_claim():
    claims = [c("Ana", "Dee", first=True), c("Bo", "Dee", first=True), c("Dee", "Ana")]
    j = resolve_joins(claims, SPEAKERS)
    assert j.canonical == {} and len(j.conflicts) == 1 and "Dee" in j.conflicts[0]


def test_speakerless_group_keeps_the_earliest_holder():
    j = resolve_joins([c("X", "Y"), c("Z", "X")], SPEAKERS)
    assert j.canonical == {"y": "x", "z": "x"}


def _common(name: str) -> bool:
    return name in {"egg", "rose"}


@pytest.mark.parametrize(
    "alias,text,expected",
    [
        ("Egg", "Who cooked the egg for breakfast?", False),
        ("Egg", "Egg is life.", False),
        ("Egg", "Fine. Egg is life.", False),
        ("Egg", "I asked Egg about it", True),
        ("Egg", "I asked Egg's sister", True),
        ("Egg", "I asked Eggs about it", False),
        ("Dee", "what fruit does dee like", True),
        ("Dee", "Dee likes pears", True),
        ("Rose", "I met Rose", True),
        ("Rose", "I like rose", False),
    ],
)
def test_alias_occurs(alias, text, expected):
    assert alias_occurs(alias, text, is_common=_common) is expected


def test_alias_occurs_within_a_region_counts_only_that_occurrence():
    text = "I asked Egg about the egg"
    assert alias_occurs("Egg", text, is_common=_common, within=(8, 11))
    assert not alias_occurs("Egg", text, is_common=_common, within=(18, 25))


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
@pytest.mark.parametrize(
    "name,common",
    [("Mary", False), ("Dee", False), ("Eli", False), ("Rose", True), ("Egg", True), ("Hope", True), ("Grace", True)],
)
def test_common_noun_is_a_lowercase_wordnet_noun_lemma(name, common):
    assert is_common_noun(name) is common


def test_without_wordnet_every_name_is_a_common_noun(monkeypatch):
    monkeypatch.setattr(identity, "wordnet_module", lambda: None)
    assert is_common_noun("Dee") and is_common_noun("Mary")
