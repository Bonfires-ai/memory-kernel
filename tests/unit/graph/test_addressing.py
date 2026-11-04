from memory_kernel.graph.addressing import addressee_of, is_explicit


def test_explicit_single_addressee():
    assert addressee_of(["Bo"], "Ana", {"Ana", "Bo", "Cy"}, explicit=True) == "Bo"


def test_explicit_without_addressee_stays_unresolved_even_two_party():
    assert addressee_of([], "Ana", {"Ana", "Bo"}, explicit=True) is None


def test_explicit_two_addressees_unresolved():
    assert addressee_of(["Bo", "Cy"], "Ana", {"Ana", "Bo", "Cy"}, explicit=True) is None


def test_implicit_two_party_rule():
    assert addressee_of([], "Ana", {"Ana", "Bo"}, explicit=False) == "Bo"


def test_implicit_group_is_unresolved_and_deterministic():
    assert addressee_of([], "Ana", {"Ana", "Bo", "Cy"}, explicit=False) is None


def test_is_explicit():
    assert is_explicit([{"addressees": []}, {"addressees": ["Bo"]}])
    assert not is_explicit([{"addressees": []}, {}])
