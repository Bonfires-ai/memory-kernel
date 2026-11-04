import pytest

from memory_kernel.runtime.program import (
    Const,
    MeaningError,
    Op,
    OpSpec,
    Program,
    Registry,
    ToolParam,
    ToolSpec,
    Var,
    validate,
)

SPIKE_MEANING = [
    ["find-entity", "?e", "?x"],
    ["identity-lines", "?l", "?e"],
    ["slot-of-kind", "?o", "?l", "Organisation"],
    ["answer", "?o"],
    ["name-span", "?x", "11", "22"],
]


def _registry() -> Registry:
    specs = [
        OpSpec("find-entity", ("name",), ("entity",), "memory"),
        OpSpec("identity-lines", ("entity",), ("lines",), "memory"),
        OpSpec("slot-of-kind", ("lines", "kind"), ("org",), "value"),
        OpSpec("name-span", ("start", "end"), ("name",), "value"),
    ]
    return Registry({s.name: s for s in specs})


def test_from_meaning_builds_ops_and_answer():
    program = Program.from_meaning(SPIKE_MEANING)
    assert len(program.ops) == 4
    assert program.answer == Var("?o")
    assert program.ops[0] == Op("find-entity", (Var("?e"), Var("?x")))
    assert program.ops[3].args == (Var("?x"), Const(11), Const(22))


def test_validate_accepts_spike_program():
    assert validate(Program.from_meaning(SPIKE_MEANING), _registry()) == []


def test_validate_flags_unknown_op_and_unbound_var():
    program = Program(
        ops=(
            Op("mystery", (Var("?a"),)),
            Op("identity-lines", (Var("?l"), Var("?ghost"))),
            Op("find-entity", (Var("?e"),)),
        ),
        answer=Var("?nowhere"),
    )
    problems = validate(program, _registry())
    assert any("unknown op" in p and "mystery" in p for p in problems)
    assert any("?ghost" in p for p in problems)
    assert any("arity" in p and "find-entity" in p for p in problems)
    assert any("?nowhere" in p for p in problems)


def test_validate_flags_cycle():
    program = Program(
        ops=(
            Op("identity-lines", (Var("?l"), Var("?e"))),
            Op("find-entity", (Var("?e"), Var("?l"))),
        ),
        answer=None,
    )
    assert any("cycle" in p for p in validate(program, _registry()))


def test_with_tools_marks_unknown_side_effect():
    registry = _registry().with_tools(
        [ToolSpec.simple("brew", ("what",), True), ToolSpec.simple("lookup", ("what",), False)]
    )
    assert registry.specs["brew"] == OpSpec(
        "brew", ("what",), ("result",), "tool", side_effect=True, params=(ToolParam("what"),)
    )
    assert registry.specs["lookup"].side_effect is False
    assert "find-entity" in registry.specs


def test_from_meaning_rejects_malformed():
    import pytest

    from memory_kernel.runtime.program import MeaningError

    for bad in (
        [[]],
        [["answer"]],
        [["answer", "x"]],
        [["answer", "?a", "?b"]],
        [["find-entity", "?e", "?x"], ["answer", "?e"], ["answer", "?x"]],
    ):
        with pytest.raises(MeaningError):
            Program.from_meaning(bad)
    assert issubclass(MeaningError, ValueError)


def test_canonical_integer_constants():
    p = Program.from_meaning([["name-span", "?x", "007", "-1", "5", "1.5"]])
    assert p.ops[0].args[1:] == (Const("007"), Const(-1), Const(5), Const("1.5"))


def test_json_round_trip():
    p = Program(
        ops=(
            Op("name-span", (Var("?x"), Const("007"), Const(22), Const("11"))),
            Op("find-entity", (Var("?e"), Var("?x"))),
        ),
        answer=Var("?e"),
    )
    assert Program.from_json(p.to_json()) == p
    assert Program.from_json(Program.from_meaning(SPIKE_MEANING).to_json()) == Program.from_meaning(SPIKE_MEANING)


def test_with_tools_refuses_shadowing():
    import pytest

    with pytest.raises(ValueError):
        _registry().with_tools([ToolSpec.simple("find-entity", (), False)])


def test_validate_flags_const_in_output_position():
    p = Program(ops=(Op("find-entity", (Const("x"), Const("y"))),), answer=None)
    assert any("output position" in m for m in validate(p, _registry()))


def test_registry_refuses_answer_and_tool_default_unknown():
    import pytest

    with pytest.raises(ValueError):
        Registry({"answer": OpSpec("answer", (), ("v",), "value")})
    assert OpSpec("t", (), ("r",), "tool").side_effect is None
    assert OpSpec("t", (), ("r",), "memory").side_effect is False
    assert OpSpec("t", (), ("r",), "tool", side_effect=False).side_effect is False


def test_from_meaning_takes_int_offsets_from_a_reading():
    """A Reading's slot offsets come back from the engine as ints: they are constants, never a crash."""
    program = Program.from_meaning([["name-span", "?x", 11, 20], ["find-entity", "?e", "?x"], ["answer", "?e"]])
    assert program.ops[0].args == (Var("?x"), Const(11), Const(20))
    assert program.answer == Var("?e")
    for bad in ([["name-span", "?x", True, 2]], [["name-span", "?x", 1.5, 2]], [["answer", 3]]):
        with pytest.raises(MeaningError):
            Program.from_meaning(bad)
