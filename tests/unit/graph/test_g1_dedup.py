import itertools
import random

from memory_kernel.graph.g1_projection import fuzzy_eligible, fuzzy_form, similar_pairs, trigram_set


def _brute_force(sets):
    """Every pair at Jaccard >= 0.9, by comparing all of them (the test's reference only)."""
    found = []
    for i, j in itertools.combinations(range(len(sets)), 2):
        if sets[i] and sets[j]:
            shared = len(sets[i] & sets[j])
            if 10 * shared >= 9 * (len(sets[i]) + len(sets[j]) - shared):
                found.append((i, j))
    return found


def test_similar_pairs_is_exact_on_random_sets():
    rng = random.Random(20261013)
    pairs = 0
    for _ in range(400):
        alphabet = [f"t{i}" for i in range(rng.randint(5, 50))]
        sets = []
        for _ in range(rng.randint(0, 30)):
            if sets and rng.random() < 0.5:
                near = set(rng.choice(sets))
                for _ in range(rng.randint(0, 2)):
                    if near and rng.random() < 0.5:
                        near.discard(rng.choice(sorted(near)))
                    else:
                        near.add(rng.choice(alphabet))
                sets.append(frozenset(near))
            else:
                sets.append(frozenset(rng.sample(alphabet, rng.randint(0, min(len(alphabet), 25)))))
        expected = _brute_force(sets)
        pairs += len(expected)
        assert similar_pairs(sets) == expected
    assert pairs > 1000


def test_threshold_is_inclusive_and_integer():
    nine_of_ten = [frozenset(range(10)), frozenset(range(9))]
    assert similar_pairs(nine_of_ten) == [(0, 1)]
    just_below = [frozenset(range(18)), frozenset(range(1, 18)) | {"x"}]  # 17 shared of 19: 0.8947
    assert similar_pairs(just_below) == []


def test_comparison_form_guard_and_trigrams():
    assert fuzzy_form("  The Ad-Campaign,  for MY store! ") == "the ad campaign for my store"
    assert not fuzzy_eligible(fuzzy_form("cat"))
    assert not fuzzy_eligible(fuzzy_form("aaaaaaaa"))
    assert fuzzy_eligible(fuzzy_form("road trip"))
    assert trigram_set("road trip") == frozenset({"roa", "oad", "adt", "dtr", "tri", "rip"})
    assert trigram_set("a") == frozenset({"a"})
    assert trigram_set("") == frozenset()
