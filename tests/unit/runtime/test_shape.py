"""Answer shape, role and kind read from the parse; the closed-class rules are data."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from memory_kernel.graph.g1_nlp import turn_doc
from memory_kernel.runtime import shape as shape_module
from memory_kernel.runtime.shape import QuestionShape, load_rules, mine_shape_patterns, read_shape


def test_shape_rules_are_data():
    """Every wh lemma, opener and label is in shapes.json; shape.py holds no such word as a string literal."""
    rules = load_rules()
    assert {"wh_shapes", "degree_shape", "how_degree_shapes", "inference_openers", "role_kinds", "wh_kinds"} <= set(
        rules
    )
    words = set(rules["wh_shapes"]) | set(rules["inference_openers"]) | set(rules["how_degree_shapes"])
    tree = ast.parse(Path(shape_module.__file__).read_text(encoding="utf-8"))
    literals = {n.value.lower() for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not words & literals
    assert json.loads(Path(shape_module.__file__).parent.joinpath("../data/runtime-grammar/en/shapes.json").read_text())


@pytest.mark.parametrize(
    ("question", "shape", "plural", "polar"),
    [
        ("Who did Kira Dace give the book to?", "who", False, False),
        ("How many books did Ana read?", "how_many", False, False),
        ("How long did Milo stay?", "how_long", False, False),
        ("How old is Milo?", "how_degree", False, False),
        ("How did Ana fix the engine?", "what", False, False),
        ("Which books does Ana like?", "list", True, False),
        ("What are Ana's hobbies?", "list", True, False),
        ("When did Kira Dace move?", "when", False, False),
        pytest.param("Did Milo visit Brask Labs?", "polar", False, True, marks=pytest.mark.trf),
        ("Would Ana enjoy Zorvex?", "inference", False, False),
        ("Kira Dace and Milo", None, False, False),
        ("Do Kira Dace and Ana start shops out of what they love?", "polar", False, True),
        ("Is the friend who wrote Milo the note no longer alive?", "polar", False, True),
        ("Did Milo teach himself how to play the lute?", "polar", False, True),
        ("Do you know where Milo lives?", "where", False, False),
        ("Can you tell me what Milo bought?", "what", False, False),
        ("Did Ana say who left?", "polar", False, True),
        ("According to Ana, what did Milo build?", "what", False, False),
        ("Which  old books does Ana like?", "list", True, False),
        ("What are some of the plans of Ana's four cats?", "list", True, False),
        ("What is the name of his dogs?", "what", False, False),
        pytest.param('What is "Little Women" about according to Ana?', "what", False, False, marks=pytest.mark.trf),
    ],
)
def test_shape_reads_structure(question, shape, plural, polar):
    got = read_shape(turn_doc(question), [])
    assert (got.shape, got.plural, got.polar) == (shape, plural, polar)


@pytest.mark.trf
def test_pattern_gives_role_and_kind_from_the_wh_position():
    """The object of "to" asks for the role the patterns mined at that position; a kind follows from shape and role."""
    doc = turn_doc("Who did Kira Dace give the book to?")
    probe = read_shape(doc, [])
    assert probe.answer_role is None and isinstance(probe, QuestionShape)
    key = {"wh": "who", "path": "pobj", "voice": "act", "answer_role": "arg2", "rows": 5, "agree": 4}
    got = read_shape(doc, [key])
    assert (got.shape, got.answer_role, got.expected_kind) == ("who", "arg2", "person")
    assert read_shape(turn_doc("When did Milo leave?"), [key]).answer_role is None


@pytest.mark.trf
def test_mined_patterns_drop_singletons():
    rows = [
        {"question": q, "answer_role": r}
        for q, r in [
            ("Who did Kira Dace meet?", "arg1"),
            ("Who  did Ana call?", "arg1"),
            ("Who did Milo thank?", "arg2"),
            ("Where did Ana sleep?", "argm-loc"),
            ("Where did Milo sleep?", "arg2"),
            ("When did Ana leave?", "argm-tmp"),
            ("When did Milo leave?", "argm-tmp"),
            ("When did Kira Dace leave?", "arg1"),
            ("When did Ana arrive?", "arg2"),
            ("Why did Ana leave?", "arg1"),
            ("Why did Milo leave?", "arg1"),
            ("Why did Kira Dace leave?", "arg2"),
            ("Why did Ana arrive?", "arg2"),
        ]
    ]
    patterns = mine_shape_patterns(rows, turn_doc)
    assert [(p["wh"], p["answer_role"], p["rows"], p["agree"]) for p in patterns] == [
        ("when", "argm-tmp", 4, 2),
        ("who", "arg1", 3, 2),
    ]
