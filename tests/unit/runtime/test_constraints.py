"""Slot constraints from tool parameters, checked at plan time: the rule (kind labels, WordNet concept with the
any-sense rule, unknown), the executor blocking only a store-kind mismatch on a resolved entity while a lexical
(WordNet) check rides on the issued step, and a failed check riding on the issued step."""

import asyncio
import threading

import pytest

from memory_kernel.runtime import constraints
from memory_kernel.runtime.api import default_registry
from memory_kernel.runtime.constraints import check_constraint
from memory_kernel.runtime.executor import Turn, execute
from memory_kernel.runtime.program import Program, ToolParam, ToolSpec

TUNE = ToolParam("pattern", role="theme", constraint="tune")
INSTRUMENT = ToolParam("source", role="instrument", constraint="instrument")


@pytest.mark.parametrize(
    "value, param, expected",
    [
        ("kettle", TUNE, ("fail", "kettle is not a tune")),
        ("a brass kettle", TUNE, ("fail", "a brass kettle is not a tune")),
        ("tune", TUNE, ("pass", None)),  # the head equals the constraint
        ("a melody", TUNE, ("pass", None)),  # a sense of melody lies under a sense of tune
        ("anything", ToolParam("x"), ("pass", None)),  # nothing declared, nothing to check
        # any-sense rule: kettle is a kettledrum in one sense, so one qualifying sense admits it; coarse, never blocks
        ("kettle", INSTRUMENT, ("pass", None)),
        ("bench", INSTRUMENT, ("fail", "bench is not a instrument")),
    ],
)
def test_constraint_rule(value, param, expected):
    assert check_constraint(value, param) == expected


def test_unknown_word_passes_to_the_game():
    assert check_constraint("a zorvex", TUNE) == ("unknown", None)  # no WordNet noun sense: a coinage
    assert check_constraint("Kira Dace", TUNE) == ("unknown", None)  # a proper-noun head: a name
    assert check_constraint({"uuid": "u1", "name": "Zorvex", "kinds": ["Organisation"]}, TUNE) == ("unknown", None)


def test_entity_kind_is_checked():
    person = ToolParam("who", role="agent", kind="Person")
    merger = {"uuid": "u6", "name": "the merger", "kinds": ["Artifact"]}
    ana = {"uuid": "u9", "name": "Ana", "kinds": ["Person"]}
    assert check_constraint(merger, person) == ("fail", "the merger is not a Person")
    assert check_constraint(ana, person) == ("pass", None)
    assert check_constraint({"uuid": "u2", "name": "Milo", "kinds": []}, person) == ("unknown", None)  # no labels
    assert check_constraint("Ana", person) == ("unknown", None)  # text names no kind
    assert check_constraint([ana, merger], person)[0] == "fail"  # a set fails on one failing member
    assert check_constraint([ana], person) == ("pass", None)


def _make_sound(constraint: bool) -> ToolSpec:
    return ToolSpec(
        "make_sound",
        (ToolParam("pattern", constraint="tune" if constraint else None), ToolParam("source")),
        True,
    )


def _sound_program(pattern: str) -> Program:
    return Program.from_meaning(
        [
            ["const", "?p", pattern],
            ["const", "?s", "piano"],
            ["procedure", "?proc", "make_sound"],
            ["seq", "?proc", "1", "?step"],
            ["make_sound", "?step", "?p", "?s"],
        ]
    )


class _Spans:
    def cut(self, left: int, right: int) -> str:
        return ""


def _execute(program: Program, tool: ToolSpec):
    turn = Turn(text="x", speaker="Ana", addressee=None, participants=(), chat_id="c1", message_id=None)
    return asyncio.run(execute(program, default_registry([tool]), None, turn=turn, spans=_Spans()))  # type: ignore[arg-type]


def test_lexical_check_never_blocks_a_step(monkeypatch):
    """A failed WordNet check rides on the issued step; the game's Resolve decides. A parser that cannot read the
    phrase (any spaCy failure) makes the check "unknown": the step is still issued."""
    kettle = _execute(_sound_program("kettle"), _make_sound(True))
    assert kettle.problems == [] and [s.tool for s in kettle.pending] == ["make_sound"]
    assert kettle.pending[0].checks == ({"param": "pattern", "constraint": "tune", "result": "fail"},)
    for text, result in (("tune", "pass"), ("zorvex", "unknown")):
        [step] = _execute(_sound_program(text), _make_sound(True)).pending
        assert step.checks == ({"param": "pattern", "constraint": "tune", "result": result},)
    # the parse runs off the event loop (asyncio.run's loop is this thread): a trf read costs tens of ms of CPU
    threads: list[int] = []
    real = constraints._head_lemma
    monkeypatch.setattr(constraints, "_head_lemma", lambda text: threads.append(threading.get_ident()) or real(text))
    _execute(_sound_program("kettle"), _make_sound(True))
    monkeypatch.undo()
    assert threads and threading.get_ident() not in threads
    [bare] = _execute(_sound_program("kettle"), _make_sound(False)).pending  # no constraint declared, nothing checked
    assert bare.checks == ()

    def broken(_text):
        raise ValueError("Shape mismatch for blis.gemm")

    monkeypatch.setattr("memory_kernel.runtime.constraints._head", broken)
    assert check_constraint("kettle", TUNE) == ("unknown", None)
    unread = _execute(_sound_program("kettle"), _make_sound(True))
    assert unread.problems == [] and unread.pending[0].checks == (
        {"param": "pattern", "constraint": "tune", "result": "unknown"},
    )


class _Entities:
    async def run(self, op, args):
        assert op == "find-entity"
        return [{"uuid": "u6", "name": args["name"], "kinds": ["Artifact"]}]


def test_entity_kind_mismatch_repairs_at_plan_time():
    """A resolved entity whose kind labels lack the parameter's kind blocks: no step is issued."""
    tool = ToolSpec("give", (ToolParam("item"), ToolParam("person", kind="Person")), True)
    program = Program.from_meaning(
        [
            ["const", "?i", "tea"],
            ["const", "?n", "the merger"],
            ["find-entity", "?e", "?n"],
            ["procedure", "?proc", "give"],
            ["seq", "?proc", "1", "?step"],
            ["give", "?step", "?i", "?e"],
        ]
    )
    turn = Turn(text="x", speaker="Ana", addressee=None, participants=(), chat_id="c1", message_id=None)
    ex = asyncio.run(execute(program, default_registry([tool]), _Entities(), turn=turn, spans=_Spans()))  # type: ignore[arg-type]
    assert ex.pending == () and not ex.done and ex.problems == ["the merger is not a Person"]
