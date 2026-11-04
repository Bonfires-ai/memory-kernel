"""Mention decomposition (``g1_mentions``): a role filler's referring cores, on the capture spans and the unit's parse."""

import pytest

from memory_kernel.grammar.dependencies import wordnet_ready
from memory_kernel.graph.g1_kinds import filler_evidence
from memory_kernel.graph.g1_mentions import bare_light, decompose, partitive_head, structure_key
from memory_kernel.graph.g1_nlp import lemma_key, nlp

NEEDS_WORDNET = pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup)")


def filler(unit: str, span, frames=(), known=()):
    doc = nlp()(unit)
    return decompose(doc, span, " ".join(doc[i].text for i in span), frames, known)


def keys(dec) -> tuple[str, ...]:
    return tuple(m.key for m in dec.mentions)


@NEEDS_WORDNET
def test_nested_frames_are_cut_but_never_the_head():
    unit = "Ana found a leaked list of 300 Zorvex targets that Kira and Milo had seen."
    frames = [([14], [[10, 11, 12], [9]]), ([3], [[4]])]
    dec = filler(unit, list(range(2, 15)), frames)
    assert len(dec.mentions) == 1
    (m,) = dec.mentions
    assert (m.key, m.surface, m.name) == ("list", "list", False)
    assert m.tokens == (2, 3, 4, 5, 6, 7, 8)
    assert tuple(n.key for n in dec.names) == ("zorvex",)
    assert not any(k in {"kira", "milo"} for k in keys(dec) + tuple(n.key for n in dec.names))
    assert dec.fallback is False

    reduced = filler("Ana saw targets selected by Zorvex.", [2, 3, 4, 5], [([3], [[2], [4, 5]])])
    assert keys(reduced) == ("target",)
    assert reduced.names == ()
    # a later conjunct's head is protected as the root is
    later = filler("Ana saw Kira and targets selected by Zorvex.", [2, 3, 4, 5, 6, 7], [([5], [[4], [6, 7]])])
    assert keys(later) == ("kira", "target")


@NEEDS_WORDNET
def test_coordination_splits_into_one_mention_per_conjunct():
    dec = filler("Kira and Milo visited Brask Labs.", [0, 1, 2])
    assert keys(dec) == ("kira", "milo")
    assert all(m.name for m in dec.mentions)


@NEEDS_WORDNET
def test_partitives_and_light_heads():
    assert keys(filler("This kind of malware worries Ana.", [0, 1, 2, 3])) == ("malware",)
    assert filler("Ana bought a lot.", [2, 3]).mentions == ()
    assert keys(filler("Ana liked the thing.", [2, 3])) == ("thing",)
    assert keys(filler("Ana crossed the line.", [2, 3])) == ("line",)
    assert keys(filler("Ana liked the device.", [2, 3])) == ("device",)
    assert all(bare_light(w) for w in ("lot", "bit"))
    assert not any(bare_light(w) for w in ("thing", "way", "line"))
    assert all(partitive_head(nlp()(p)[1]) for p in ("this kind of attacks", "a sort of tools", "the type of spyware"))
    # a category head is "kind" or a hyponym of it (WordNet); a cognition noun in general is not
    nouns = ("investigation", "sample", "example", "pattern", "precedent", "point")
    for word in ("device", "company", "target", "phone", "spyware", "report", *nouns):
        assert not bare_light(word)
        assert not partitive_head(nlp()(word)[0])
        assert not partitive_head(nlp()(f"the {word} of attacks")[1])
    # a compound on the partitive head is lifted as an inner name ("a Zorvex kind of infection link")
    link = filler("Ana saw a Zorvex kind of infection link.", [2, 3, 4, 5, 6, 7])
    assert keys(link) == ("infection link",) and tuple(n.key for n in link.names) == ("zorvex",)
    probe = filler("Ana read the investigation of Zorvex.", [2, 3, 4, 5])
    assert keys(probe) == ("investigation",) and tuple(n.key for n in probe.names) == ("zorvex",)


@NEEDS_WORDNET
def test_names_in_modifiers_and_lowercase_known_names():
    # a name compound is no re-key: "Zorvex spyware" is its own referent, Zorvex an inner name it supplies no kind to
    spyware = filler("Ana studied Zorvex spyware.", [2, 3])
    (m,) = spyware.mentions
    assert (m.key, m.name, m.head, m.kind_nouns) == ("zorvex spyware", False, "spyware", ())
    assert tuple(n.key for n in spyware.names) == ("zorvex",)
    # a 3rd-person pronoun possessor drops like a determiner; a nominal owner keys by its own key; "my" stays
    assert keys(filler("Ana studied their Zorvex spyware.", [2, 3, 4])) == ("zorvex spyware",)
    owned = keys(filler("Ana fixed the Brask Labs' device.", [2, 3, 4, 5, 6]))
    assert owned == keys(filler("Ana fixed Brask Labs' device.", [2, 3, 4, 5])) and "brask labs" in owned[0]
    assert keys(filler("Ana lost my phone.", [2, 3])) == ("my phone",)
    (dace,) = filler("Ana met the journalist Milo Dace.", [2, 3, 4, 5]).mentions
    assert (dace.key, dace.name, dace.kind_nouns) == ("milo dace", True, ("journalist",))  # a noun before a name head
    assert dict(filler_evidence(False, dace, None).supplied) == {"Person": 1}
    # a comma-set appositive is no restrictive name: the common noun is the mention, the name an inner name
    (vocative,) = filler("Ana saved it on my graph, Kira, today.", [4, 5, 6, 7]).mentions
    assert (vocative.key, vocative.name) == ("my graph", False)
    # a parenthetical name is restrictive (no comma between head and appositive): the name is the mention, no bracket in it
    (paren,) = filler("Ana studied a mapping tool (Brask Labs) yesterday.", [2, 3, 4, 5, 6, 7, 8]).mentions
    assert (paren.key, paren.name, paren.kind_nouns) == (lemma_key("Brask Labs"), True, ("tool",))
    assert paren.surface == "Brask Labs"
    # a listed appositive keeps the group reading: the plural noun is the mention, the listed names are inner names
    group = filler("Ana met my friends Ana and Milo yesterday.", [2, 3, 4, 5, 6])
    assert keys(group) == ("my friend",) and {n.key for n in group.names} == {"ana", "milo"}

    sample = filler("Ana studied the Zorvex sample.", [2, 3, 4])
    assert keys(sample) == ("zorvex sample",)
    assert tuple(n.key for n in sample.names) == ("zorvex",)

    lower = filler("Ana studied zorvex spyware.", [2, 3], known={"zorvex"})
    assert keys(lower) == ("zorvex spyware",) and tuple(n.key for n in lower.names) == ("zorvex",)
    # a lowercased known name is a name: no head class ("labs" is otherwise an Artifact head)
    (labs,) = filler("Ana visited the brask labs.", [2, 3, 4], known={"brask labs"}).mentions
    assert (labs.name, labs.head, labs.key) == (True, "", lemma_key("brask labs"))
    assert not filler_evidence(False, labs, None).head
    (common,) = filler("Ana visited the brask labs.", [2, 3, 4]).mentions
    assert filler_evidence(False, common, None).head
    # an adjective head has no head class ("rich": WordNet's noun "the rich" is a group)
    (rich,) = filler("Kira is rich.", [2]).mentions
    assert rich.head == "" and not filler_evidence(False, rich, None).head
    # parsed alone "Zorvex" is an adjective; the orthographic guard keeps today's key with the name in it
    assert structure_key("Zorvex targets") == "zorvex target"

    # a determiner, and a common noun: not a name (its key is lemma_key's, as analyse's: "rose" alone lemmatises "rise")
    (rose,) = filler("Ana smelled the rose.", [2, 3], known={"rose"}).mentions
    assert (rose.surface, rose.head, rose.name, rose.key) == ("rose", "rose", False, lemma_key("rose"))


@NEEDS_WORDNET
def test_an_unaligned_filler_falls_back_to_its_own_parse():
    dec = decompose(nlp()("Kira met Bo today."), [9, 10], "Kira and Milo", ())
    assert dec.fallback is True
    assert keys(dec) == ("kira", "milo")
    # a name in a dependency-dropped clause (no frame took it) is kept as an inner name
    assert tuple(n.key for n in decompose(nlp()("Kira met Bo."), [], "a report naming Zorvex targets", ()).names) == (
        "zorvex",
    )


@NEEDS_WORDNET
def test_a_shifted_span_realigns_on_the_unit_parse():
    unit = "Ana met the new vendor and Kira yesterday."
    aligned = filler(unit, [2, 3, 4, 5, 6])
    doc = nlp()(unit)
    shifted = decompose(doc, [4, 5, 6, 7, 8], "the new vendor and Kira", ())
    assert (shifted.fallback, shifted.realigned) == (False, (2, 3, 4, 5, 6))
    assert keys(shifted) == keys(aligned) == ("vendor", "kira")
    # several runs: the one nearest the given span's first index
    twice = nlp()("Ana gave the book to Kira and the book stayed.")
    assert decompose(twice, [9, 10], "the book", ()).realigned == (7, 8)
    # an equidistant tie takes the earlier run; no span takes the first run
    cats = nlp()("Ana fed the cat and then the cat slept.")
    assert decompose(cats, [4, 5], "the cat", ()).realigned == (2, 3)
    assert decompose(cats, [], "the cat", ()).realigned == (2, 3)
    # tokenisation-blind (characters exact, whitespace ignored): spaCy's "re - exploit", "Ana 's", whitespace tokens
    bug = nlp()("Kira wants to re-exploit the bug today.")
    assert decompose(bug, [5, 6, 7, 8, 9], "re-exploit the bug", ()).realigned == (3, 4, 5, 6, 7)
    assert decompose(nlp()("Kira took Ana's phone yesterday."), [4, 5], "Ana's phone", ()).realigned == (2, 3, 4)
    assert decompose(nlp()("Kira took  the  phone.\n\nThen left."), [7, 8], "the phone", ()).realigned == (3, 5)
    # a realigned span takes no frame cuts: the unit's other capture indices are suspect too
    unit = "Ana met the vendor of Brask Labs today."
    frames = [([4], [[5, 6]])]  # a frame over "of Brask Labs": on the aligned span it cuts the inner name
    assert filler(unit, [2, 3, 4, 5, 6], frames).names == ()
    realigned = decompose(nlp()(unit), [4, 5, 6, 7, 8], "the vendor of Brask Labs", frames)
    assert realigned.realigned == (2, 3, 4, 5, 6)
    assert tuple(n.key for n in realigned.names) == (lemma_key("Brask Labs"),)


@NEEDS_WORDNET
def test_a_fallback_mention_needs_a_nominal_head():
    dec = decompose(nlp()("Kira met Bo."), [], "until the conditions force a choice", ())
    assert dec.fallback is True
    assert "force" not in keys(dec)
    assert dec.dropped == 1
    # a known name keeps its mention whatever its tag on its own parse
    assert keys(decompose(nlp()("Kira met Bo."), [], "zorvex", (), known={"zorvex"})) == ("zorvex",)
