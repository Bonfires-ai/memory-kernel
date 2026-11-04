import asyncio
from collections.abc import Mapping
from typing import Any

from memory_kernel.runtime.executor import VALUE_SPECS, Execution, Turn, execute
from memory_kernel.runtime.program import OpSpec, Program, Registry, ToolParam, ToolSpec, validate

TOOLS = ("take", "make", "give", "put")


class FakeSpans:
    def __init__(self, text: str) -> None:
        self.text = text

    def cut(self, left: int, right: int) -> str:
        return self.text[left:right]


class FakeMemory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, op: str, args: Mapping[str, Any]) -> Any:
        self.calls.append((op, dict(args)))
        if op == "count":
            return len(args["set"])
        raise KeyError(op)


def _registry(give_side_effect: bool | None = True) -> Registry:
    base = Registry({**VALUE_SPECS, "count": OpSpec("count", ("set",), ("n",), "memory")})
    return base.with_tools(
        [
            ToolSpec.simple("take", ("item",), False),
            ToolSpec.simple("make", ("item", "first", "second"), False),
            ToolSpec.simple("give", ("item", "person"), give_side_effect),
            ToolSpec.simple("put", ("item", "place"), False),
        ]
    )


def _turn(text: str, speaker: str | None = "Kira", participants: tuple[str, ...] = ()) -> Turn:
    return Turn(text=text, speaker=speaker, addressee=None, participants=participants, chat_id="c1", message_id="m1")


# "brew tea" after "to brew X means: take water and herb, make X from them, give X to me" (spike 2 shape).
BREW = [
    ["procedure", "?proc", "brew"],
    ["seq", "?proc", "1", "?b1"],
    ["procedure", "?b1", "take"],
    ["seq", "?b1", "1", "?b1-s1"],
    ["take", "?b1-s1", "?water"],
    ["seq", "?b1", "2", "?b1-s2"],
    ["take", "?b1-s2", "?herb"],
    ["seq", "?proc", "2", "?b2"],
    ["procedure", "?b2", "make"],
    ["seq", "?b2", "1", "?b2-s1"],
    ["make", "?b2-s1", "?x", "?water", "?herb"],
    ["seq", "?proc", "3", "?b3"],
    ["procedure", "?b3", "give"],
    ["seq", "?b3", "1", "?b3-s1"],
    ["give", "?b3-s1", "?x", "?r"],
    ["const", "?water", "water"],
    ["const", "?herb", "herb"],
    ["name-span", "?x", "5", "8"],
    ["indefinite", "?x"],
    ["deixis", "?r", "speaker"],
]


def _run(meaning, turn: Turn, memory: FakeMemory, registry: Registry, resumed=None) -> Execution:
    program = Program.from_meaning(meaning)
    return asyncio.run(execute(program, registry, memory, turn=turn, spans=FakeSpans(turn.text), resumed=resumed))


def _drive(meaning, turn: Turn, memory: FakeMemory, registry: Registry) -> tuple[Execution, list[tuple[str, str]]]:
    """Resume each pending tool step in turn until the program is done; return the steps seen."""
    resumed: dict[str, Any] = {}
    seen: list[tuple[str, str]] = []
    ex = _run(meaning, turn, memory, registry)
    while ex.pending:
        assert len(ex.pending) == 1
        step = ex.pending[0]
        seen.append((step.tool, step.step_id))
        resumed[step.step_id] = f"ok-{step.tool}-{len(seen)}"
        ex = _run(meaning, turn, memory, registry, resumed)
    return ex, seen


def test_tool_step_suspends_and_resume_continues():
    turn = _turn("brew tea")
    first = _run(BREW, turn, FakeMemory(), _registry())
    assert first.problems == []
    assert first.done is False
    (step,) = first.pending
    assert (step.tool, step.step_id, step.args) == ("take", "4:0", {"item": "water"})
    assert not [t for t in first.trace if t.op in TOOLS]  # a pending step is never traced as run
    indefinite = [t for t in first.trace if t.op == "indefinite"]
    assert len(indefinite) == 1 and indefinite[0].ok

    ex, seen = _drive(BREW, turn, FakeMemory(), _registry())
    assert seen == [("take", "4:0"), ("take", "6:0"), ("make", "10:0"), ("give", "14:0")]
    assert ex.done is True and ex.pending == () and ex.problems == []
    tool_trace = [t for t in ex.trace if t.op in TOOLS]
    assert [t.op for t in tool_trace] == ["take", "take", "make", "give"]  # 4 tool trace steps
    assert all(t.ok for t in tool_trace)
    assert tool_trace[2].args == {"item": "tea", "first": "water", "second": "herb"}
    assert tool_trace[3].args == {"item": "tea", "person": "Kira"}
    assert tool_trace[3].out == "ok-give-4"
    assert ex.values["?b3-s1"] == "ok-give-4"
    # a stray resumed id is a problem written with the caller's id prefix (the api exposes <plan_id>:<local id>)
    done = Program.from_meaning([["const", "?x", "tea"], ["count", "?n", "?x"], ["answer", "?n"]])
    stray = asyncio.run(
        execute(
            done,
            _registry(),
            FakeMemory(),
            turn=turn,
            spans=FakeSpans(turn.text),
            resumed={"9:0": "x"},
            id_prefix="p1:",
        )
    )
    assert stray.problems == ["resumed step p1:9:0 is not a step of this program"]


def test_me_without_speaker_is_refused():
    memory = FakeMemory()
    meaning = [
        ["give", "?s", "?x", "?r"],
        ["const", "?x", "tea"],
        ["deixis", "?r", "speaker"],
        ["count", "?n", "?x"],
    ]
    ex = _run(meaning, _turn("give tea to me", speaker=None), memory, _registry())
    assert ex.problems and any("speaker" in p for p in ex.problems)
    assert ex.pending == ()
    assert ex.done is False
    assert ex.trace == []
    assert memory.calls == []  # refused before any op runs


def test_distribute_runs_once_per_participant():
    # "host a party": brew tea for everyone (distributed), then put the cake on the bench (once).
    memory = FakeMemory()
    meaning = [
        ["procedure", "?party", "host-party"],
        ["seq", "?party", "1", "?brew"],
        ["procedure", "?brew", "brew"],
        ["seq", "?brew", "1", "?g"],
        ["give", "?g", "?tea", "?all"],
        ["distribute", "?brew", "?all"],
        ["seq", "?party", "2", "?p"],
        ["put", "?p", "?cake", "?bench"],
        ["everyone", "?all"],
        ["const", "?tea", "tea"],
        ["const", "?cake", "cake"],
        ["const", "?bench", "bench"],
        ["count", "?n", "?all"],
        ["answer", "?n"],
    ]
    turn = _turn("host a party", participants=("Kira", "Milo", "Ana"))
    ex, seen = _drive(meaning, turn, memory, _registry())
    assert ex.done is True and ex.problems == []
    assert seen == [("give", "4:0"), ("give", "4:1"), ("give", "4:2"), ("put", "7:0")]
    gives = [t for t in ex.trace if t.op == "give"]
    assert [t.args["person"] for t in gives] == ["Kira", "Milo", "Ana"]
    assert len([t for t in ex.trace if t.op == "put"]) == 1
    assert ex.values["?n"] == 3  # outside the distributed procedure the set stays whole
    assert ex.values["?g"] == ["ok-give-1", "ok-give-2", "ok-give-3"]


def test_side_effect_unknown_needs_confirmation():
    meaning = [["give", "?s", "?x", "?r"], ["const", "?x", "tea"], ["deixis", "?r", "speaker"]]
    (unknown,) = _run(meaning, _turn("give tea to me"), FakeMemory(), _registry(None)).pending
    assert unknown.side_effect is None and unknown.needs_confirmation is True
    (declared,) = _run(meaning, _turn("give tea to me"), FakeMemory(), _registry(True)).pending
    assert declared.side_effect is True and declared.needs_confirmation is True
    take = [["take", "?s", "?x"], ["const", "?x", "tea"]]
    (pure,) = _run(take, _turn("take tea"), FakeMemory(), _registry()).pending
    assert pure.side_effect is False and pure.needs_confirmation is False


def test_step_carries_actor():
    meaning = [["give", "?s", "?x", "?r"], ["const", "?x", "tea"], ["deixis", "?r", "speaker"]]
    play = ToolSpec.simple("give", ("item", "person"), True, consent="speaker_turn")
    (played,) = _run(meaning, _turn("give tea to me"), FakeMemory(), _registry().with_tools([play])).pending
    assert played.actor == "Kira"


def _registry_plus(*extra: ToolSpec, memory_specs: tuple[OpSpec, ...] = ()) -> Registry:
    base = _registry()
    return Registry({**base.specs, **{s.name: s for s in memory_specs}}).with_tools(list(extra))


def _drive_with(meaning, turn, memory, registry, result, *, prior: bool = False):
    """Resume each pending step with ``result(step)``; with ``prior`` pass the previous values back."""
    resumed: dict[str, Any] = {}
    seen: list[tuple[str, str]] = []
    runs: list[Execution] = []
    ex = _run(meaning, turn, memory, registry)
    runs.append(ex)
    while ex.pending:
        (step,) = ex.pending
        seen.append((step.tool, step.step_id))
        resumed[step.step_id] = result(step)
        extra = {"prior": ex.values} if prior else {}
        program = Program.from_meaning(meaning)
        ex = asyncio.run(
            execute(program, registry, memory, turn=turn, spans=FakeSpans(turn.text), resumed=resumed, **extra)
        )
        runs.append(ex)
    return ex, seen, runs


def test_seq_order_blocks_later_steps_of_the_procedure():
    # errand = seq 1 give(?x, ?who), seq 2 take(?x); ?who comes from a tool outside any procedure.
    meaning = [
        ["procedure", "?errand", "errand"],
        ["seq", "?errand", "1", "?g"],
        ["give", "?g", "?x", "?who"],
        ["seq", "?errand", "2", "?t"],
        ["take", "?t", "?x"],
        ["const", "?x", "cake"],
        ["lookup", "?who"],
    ]
    registry = _registry_plus(ToolSpec.simple("lookup", (), False))
    ex, seen, _ = _drive_with(meaning, _turn("do the errand"), FakeMemory(), registry, lambda s: "Milo")
    assert [tool for tool, _ in seen] == ["lookup", "give", "take"]
    assert ex.done is True and ex.problems == []


def test_step_waiting_on_a_later_step_is_refused():
    meaning = [
        ["procedure", "?p", "swap"],
        ["seq", "?p", "1", "?g"],
        ["give", "?g", "?x", "?t"],
        ["seq", "?p", "2", "?t"],
        ["take", "?t", "?x"],
        ["const", "?x", "cake"],
    ]
    ex = _run(meaning, _turn("swap"), FakeMemory(), _registry())
    assert any("step 1 of procedure ?p (give) waits on a later step (take)" in p for p in ex.problems)
    assert ex.pending == () and ex.done is False


def test_resume_reuses_prior_values():
    # A memory op inside the distributed procedure runs once per member on the first run, never again on resume.
    meaning = [
        ["procedure", "?p", "gift"],
        ["seq", "?p", "1", "?g"],
        ["give", "?g", "?n", "?all"],
        ["distribute", "?p", "?all"],
        ["everyone", "?all"],
        ["count", "?n", "?all"],
    ]
    memory = FakeMemory()
    turn = _turn("gift everyone", participants=("Kira", "Milo", "Ana"))
    ex, seen, runs = _drive_with(meaning, turn, memory, _registry(), lambda s: "given", prior=True)
    assert ex.done is True and len(seen) == 3
    assert len(memory.calls) == 3  # first run only
    for later in runs[1:]:
        counts = [t for t in later.trace if t.op == "count"]
        assert len(counts) == 3 and all(t.reused for t in counts)
    assert [t.args["item"] for t in ex.trace if t.op == "give"] == [4, 4, 3]
    assert ex.values["?n"] == [4, 4, 3]


def test_root_order_does_not_decide_whether_a_program_runs():
    # party (distributed over ?guests) and ask (list-guests writes ?guests) are two roots; either order runs.
    party = [
        ["procedure", "?party", "party"],
        ["seq", "?party", "1", "?brew"],
        ["procedure", "?brew", "brew"],
        ["seq", "?brew", "1", "?g"],
        ["give", "?g", "?tea", "?guests"],
        ["distribute", "?brew", "?guests"],
        ["const", "?tea", "tea"],
    ]
    ask = [["procedure", "?ask", "ask"], ["seq", "?ask", "1", "?guests"], ["list-guests", "?guests"]]
    registry = _registry_plus(ToolSpec.simple("list-guests", (), False))

    def result(step):
        return ["Milo", "Ana"] if step.tool == "list-guests" else "given"

    outcomes = []
    for meaning in (party + ask, ask + party):
        ex, seen, _ = _drive_with(meaning, _turn("party"), FakeMemory(), registry, result)
        outcomes.append((ex.done, ex.problems, [t for t, _ in seen], [s.args for s in ex.trace if s.op == "give"]))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][:3] == (True, [], ["list-guests", "give", "give"])


def test_shape_refusals():
    memory = FakeMemory()
    guests_of = OpSpec("guests-of", ("set",), ("guests",), "memory")
    shadow = OpSpec("shadow", ("x",), ("y",), "value")
    registry = _registry_plus(memory_specs=(guests_of, shadow))
    nested = [  # the inner set is computed inside the outer distributed procedure
        ["procedure", "?o", "outer"],
        ["seq", "?o", "1", "?i"],
        ["procedure", "?i", "inner"],
        ["seq", "?i", "1", "?g"],
        ["give", "?g", "?tea", "?b"],
        ["distribute", "?o", "?a"],
        ["distribute", "?i", "?b"],
        ["everyone", "?a"],
        ["guests-of", "?b", "?a"],
        ["const", "?tea", "tea"],
    ]
    turn = _turn("party", participants=("Kira", "Milo"))
    ex = _run(nested, turn, memory, registry)
    assert any("inside a distributed procedure" in p for p in ex.problems) and not ex.done
    unknown_value = [["shadow", "?y", "?x"], ["const", "?x", "tea"]]
    assert any("unknown value op 'shadow'" in p for p in _run(unknown_value, turn, memory, registry).problems)
    data_step = [["procedure", "?p", "p"], ["seq", "?p", "1", "?x"], ["const", "?x", "tea"]]
    assert any("not a step" in p for p in _run(data_step, turn, memory, registry).problems)
    assert memory.calls == []


def test_gather_lists_its_inputs_in_order():
    """gather takes any number of inputs and lists them in argument order (not the order their ops run): the value a
    parameter that takes several values gets. Only a declared variadic tail relaxes an op's arity."""
    registry = _registry_plus(ToolSpec("make", (ToolParam("item"), ToolParam("material", many=True)), False))
    meaning = [
        ["const", "?b", "y"],
        ["const", "?a", "x"],
        ["gather", "?l", "?a", "?b"],
        ["make", "?s", "?a", "?l"],
    ]
    ex = _run(meaning, _turn("make x"), FakeMemory(), registry)
    assert ex.problems == []
    (step,) = ex.pending
    assert step.args == {"item": "x", "material": ["x", "y"]}
    assert any("arity mismatch" in p for p in validate(Program.from_meaning([["const", "?a", "x", "y"]]), registry))
