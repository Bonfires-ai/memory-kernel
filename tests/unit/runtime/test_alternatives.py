"""Alternatives (2.1, spec 1e): every distinct analysis of a text, each a plan walked as ``plan``'s is, with its path;
and a runtime that reads the primary analysis alone (``Runtime(analyses=1)``).

The grammar is the argument grammar's shapes (Task 2's rows): a statement root over one clause slot, two-slot clause
connectives and one operator, with ``statement``, ``relation`` and ``mark`` registered as tools. The connectives score
above the generated clause filler (0.5) and the root below it: otherwise the clause filler wins every slot and nothing
composes. Every sentence is made up.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from memory_kernel.runtime import api as api_module
from memory_kernel.runtime import engine as engine_module
from memory_kernel.runtime.api import (
    ALTERNATIVES,
    ALTERNATIVES_DEADLINE_MS,
    ALTERNATIVES_NODE_CAP,
    ALTERNATIVES_RAW,
    GENERATED,
    Runtime,
    TurnPlan,
    default_registry,
)
from memory_kernel.runtime.catalogue import parse_catalogue
from memory_kernel.runtime.constructions import CxnSpec, spec_id
from memory_kernel.runtime.engine import CLIENT_SLACK_S, GRAMMAR_ID, Reading, RuntimeEngineError
from memory_kernel.runtime.executor import Turn
from memory_kernel.runtime.lexicon import Lexicon, QuestionMap
from memory_kernel.runtime.normalise import normalise
from memory_kernel.runtime.program import ToolSpec
from memory_kernel.runtime.store import CLAUSE_FILLER, composes
from unit.runtime.memory_writer import MemoryWriter, memory_store

PORT = 9729  # 9713-9728 are the other engine test modules'
NOW = "2026-11-24T10:00:00+00:00"
ROOT = "root.s"
ONE = "it works because it is open"
TWO = "it works because it is open but nobody pays"
#: four operators and connectives: more complete analyses than one call's default count
LONG = "it would not work because nobody pays if fees rise but usage holds"
TOOLS = default_registry(
    [
        ToolSpec.simple("statement", ("text",), False),
        ToolSpec.simple("relation", ("whole", "head", "text"), False),
        ToolSpec.simple("mark", ("whole", "scope", "rest"), False),
    ]
)
CONFIG = f"(fcg::get-configuration fcg-go::{GRAMMAR_ID} :max-nr-of-nodes)"


def _slot(name: str) -> dict:
    return {"slot": name, "fills": "clause"}


def _connective(word: str, score: float = 0.7, **kw: Any) -> CxnSpec:
    return CxnSpec(
        id=kw.pop("cxn_id", f"cx.{word}"),
        kind="hand",
        unit="filler",
        fills="clause",
        form=[_slot("h"), word, _slot("d")],
        meaning=[["clause-span", "?x", "?l", "?r"], ["relation", "?rel", "?x", "?h", "?d"]],
        score=score,
        status=kw.pop("status", "live"),
        **kw,
    )


def _rows(**scores: float) -> list[CxnSpec]:
    """Task 2's rows: one statement root over a clause, three connectives and one operator."""
    say = [["procedure", "?p", "say"], ["seq", "?p", "1", "?st"], ["statement", "?st", "?s"]]
    rows = [CxnSpec(id=ROOT, kind="hand", unit="root", form=[_slot("s")], meaning=say, score=0.2, status="live")]
    rows += [_connective(word, scores.get(word, 0.7)) for word in ("because", "if", "but")]
    operator = CxnSpec(
        id="op.not",
        kind="mined",
        unit="filler",
        fills="clause",
        form=[_slot("a"), "not", _slot("b")],
        meaning=[["clause-span", "?x", "?l", "?r"], ["mark", "?m", "?x", "?a", "?b"]],
        score=0.65,
        status="live",
    )
    return [*rows, operator]


# -- the engine and the runtimes ----------------------------------------------------------------------------------------


def _data_dir():
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """One engine for the module, stopped by pid; each runtime loads its own grammar into it on its first plan."""
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("runtime-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


class _NoGraph:
    async def rows(self, cypher: str, params: Any) -> list[dict]:
        return []


def _runtime(engine: Any, *rows: CxnSpec, writer: MemoryWriter | None = None, **kw: Any) -> Runtime:
    kw.setdefault("record_standing", False)
    store = memory_store(*rows, writer=writer)
    return Runtime("bonfire", engine, _NoGraph(), kw.pop("registry", TOOLS), lexicon=_lexicon(), store=store, **kw)


def _lexicon() -> Lexicon:
    return Lexicon([], QuestionMap.load())


@pytest.fixture
def runtime(engine):
    return _runtime(engine, *_rows())


def turn(text: str, speaker: str = "Ana", chat_id: str = "c1") -> Turn:
    return Turn(text, speaker, "Milo", (speaker, "Milo"), chat_id, None, at=NOW)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def walk(runtime: Runtime, plan: TurnPlan) -> tuple[tuple[tuple[str, Any], ...], str]:
    """Every tool step of ``plan`` in order (tool and args; a step id names its plan), each answered as committed with
    value None, until the plan ends; and the status it ends on."""
    steps: list[tuple[str, Any]] = []
    while plan.status == "needs_tools":
        steps += [(s["tool"], s["args"]) for s in plan.tool_steps]
        results = [
            {"step_id": s["step_id"], "tool": s["tool"], "args": s["args"], "verdict": "committed", "value": None}
            for s in plan.tool_steps
        ]
        plan = run(runtime.resume(plan.plan_id, results))
    return tuple(steps), plan.status


def outer(alternative: Any) -> str | None:
    """The construction at address "s": the outermost one under the root."""
    return dict(alternative.path).get("s")


# -- the first alternative is plan's -------------------------------------------------------------------------------------


@pytest.mark.needs_fcg
def test_the_first_alternative_is_the_plans_reading(runtime):
    for text in (ONE, ONE + ".", TWO):  # with a final mark the plan reads the text without it: so do the alternatives
        plan = run(runtime.plan(turn(text), window=0))
        first = run(runtime.alternatives(turn(text))).found[0]
        assert plan.status == "needs_tools" and first.plan.status == plan.status, (text, plan.diagnostics)
        assert first.plan.reading["applied"] == plan.reading["applied"], text
        assert first.plan.reading["normalised"] == plan.reading["normalised"], text
        assert walk(runtime, first.plan) == walk(runtime, plan), text
        assert first.plan.plan_id != plan.plan_id


@pytest.mark.needs_fcg
def test_the_same_program_is_one_alternative(engine):
    """A second connective with another id and the same form and meaning reads the text with the same program,
    variable names aside: one alternative (the earlier, the plan's own)."""
    twin = _connective("because", cxn_id="cx.because-twin")
    runtime = _runtime(engine, *_rows(), twin)
    found = run(runtime.alternatives(turn(ONE), maximal=False)).found
    outers = [outer(a) for a in found]
    assert len([o for o in outers if o in ("cx.because", "cx.because-twin")]) == 1, outers
    plan = run(runtime.plan(turn(ONE), window=0))
    assert found[0].plan.reading["applied"] == plan.reading["applied"]
    assert outers == [outer(found[0]), GENERATED], outers  # the connective's reading, then the whole statement


@pytest.mark.needs_fcg
def test_two_connectives_give_both_nestings(runtime):
    found = run(runtime.alternatives(turn(TWO))).found
    assert {outer(a) for a in found} >= {"cx.because", "cx.but"}
    for alternative in found:  # each walks to its end as a plan does, through its own connectives
        steps, status = walk(runtime, alternative.plan)
        assert status == "answered" and [tool for tool, _ in steps].count("relation") == 2, alternative.path


# -- maximal ------------------------------------------------------------------------------------------------------------

#: with "because" below the clause filler's score, plan's reading of TWO keeps "it works because it is open" whole
BUT_FIRST = (("", ROOT), ("s", "cx.but"), ("s/h", GENERATED), ("s/d", GENERATED))
BUT_BECAUSE = (
    ("", ROOT),
    ("s", "cx.but"),
    ("s/h", "cx.because"),
    ("s/h/h", GENERATED),
    ("s/h/d", GENERATED),
    ("s/d", GENERATED),
)
BECAUSE_BUT = (
    ("", ROOT),
    ("s", "cx.because"),
    ("s/h", GENERATED),
    ("s/d", "cx.but"),
    ("s/d/h", GENERATED),
    ("s/d/d", GENERATED),
)


@pytest.mark.needs_fcg
def test_maximal_drops_a_reading_that_leaves_a_readable_piece_whole_but_never_the_first(engine):
    runtime = _runtime(engine, *_rows(because=0.3))
    every = run(runtime.alternatives(turn(TWO), maximal=False)).found
    assert every[0].path == BUT_FIRST  # it leaves "it works because it is open" to the clause filler
    assert (("", ROOT), ("s", GENERATED)) in [a.path for a in every]  # the whole statement, left to the clause filler
    kept = run(runtime.alternatives(turn(TWO))).found
    assert kept[0].path == BUT_FIRST  # plan's reading: never dropped, whatever it leaves whole
    assert {a.path for a in kept} == {BUT_FIRST, BUT_BECAUSE, BECAUSE_BUT}, [a.path for a in kept]
    assert len(kept) < len(every)


# -- the root, the bounds, the path ---------------------------------------------------------------------------------------


@pytest.mark.needs_fcg
def test_through_keeps_one_root_only(engine):
    """A second root, "because {Y:clause}", reads "because it is open but nobody pays" first (its score is the
    default, above the statement root's): through the statement root every alternative is the statement's."""
    second = CxnSpec(
        id="root.because",
        kind="hand",
        unit="root",
        form=["because", _slot("y")],
        meaning=[["procedure", "?p", "say"], ["seq", "?p", "1", "?st"], ["statement", "?st", "?y"]],
        status="live",
    )
    runtime = _runtime(engine, *_rows(), second)
    text = "because it is open but nobody pays"
    assert run(runtime.plan(turn(text), window=0)).reading["root"] == "root.because"
    every = run(runtime.alternatives(turn(text), maximal=False)).found
    assert every[0].path[0] == ("", "root.because") and ("", ROOT) in [a.path[0] for a in every]
    through = run(runtime.alternatives(turn(text), through=ROOT, maximal=False)).found
    assert through and all(a.path[0] == ("", ROOT) for a in through)
    assert through[0].path == next(a.path for a in every if a.path[0] == ("", ROOT))
    assert {a.path for a in through} == {a.path for a in every if a.path[0] == ("", ROOT)}
    other = run(runtime.alternatives(turn(text), through="root.because", maximal=False)).found
    assert other and all(a.path[0] == ("", "root.because") for a in other)
    with pytest.raises(ValueError, match="cx.but"):  # a filler is no root
        run(runtime.alternatives(turn(text), through="cx.but"))


@pytest.mark.needs_fcg
def test_a_bound_is_reported_and_the_settings_are_restored(runtime, engine):
    before = run(runtime.plan(turn(LONG), window=0))
    nodes = engine._lisp(CONFIG)
    one = run(runtime.alternatives(turn(LONG), limit=1))
    assert one.cut_short and len(one.found) == 1
    assert one.found[0].plan.reading["applied"] == before.reading["applied"]
    full = run(runtime.alternatives(turn(LONG), maximal=False))
    assert full.cut_short and len(full.found) == ALTERNATIVES  # more complete analyses than one call takes
    assert not run(runtime.alternatives(turn(ONE))).cut_short  # the search ran out
    assert engine._lisp(CONFIG) == nodes
    after = run(runtime.plan(turn(LONG), window=0))
    assert after.reading["applied"] == before.reading["applied"] and walk(runtime, after) == walk(runtime, before)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            run(runtime.alternatives(turn(ONE), limit=bad))


@pytest.mark.needs_fcg
def test_the_count_is_what_is_returned(engine):
    """``limit`` counts the alternatives returned, after distinctness, ``through`` and ``maximal``; the search itself
    takes up to ``ALTERNATIVES_RAW``. In this grammar a reading is maximal when it applies all four constructions (no
    connective sits at a piece's edge, so any piece holding one can be read through it)."""
    runtime = _runtime(engine, *_rows())
    every = run(runtime.alternatives(turn(LONG), maximal=False, limit=ALTERNATIVES_RAW))
    assert not every.cut_short  # the search ran out inside its bounds
    four = {"op.not", "cx.because", "cx.if", "cx.but"}
    whole = [a.path for a in every.found if four <= set(dict(a.path).values())]
    assert 1 < len(whole) <= ALTERNATIVES < len(every.found), (len(whole), len(every.found))
    kept = run(runtime.alternatives(turn(LONG)))
    assert [a.path for a in kept.found] == whole and not kept.cut_short
    assert run(runtime.alternatives(turn(LONG), limit=len(whole) - 1)).cut_short  # more survived than returned
    few = run(runtime.alternatives(turn(TWO), limit=2))  # two maximal readings, three that are not
    assert len(few.found) == 2 and not few.cut_short
    twins = _runtime(engine, *_rows(), _connective("because", cxn_id="cx.because-twin"))
    one = run(twins.alternatives(turn(ONE), limit=1))  # the twin's reading is the same program: one alternative
    assert len(one.found) == 1 and not one.cut_short


#: a grammar that does not compose: a root with two np slots, and a taught-style np filler that reads only "the team",
#: scored above the base fillers so the search fills the right-hand slot first
MEETS = CxnSpec(
    id="rt.meets",
    kind="hand",
    unit="root",
    form=[{"slot": "x", "fills": "np"}, "meets", {"slot": "y", "fills": "np"}],
    meaning=[["procedure", "?p", "pair"], ["seq", "?p", "1", "?st"], ["pair", "?st", "?x", "?y"]],
    status="live",
)
TEAM = CxnSpec(
    id="np.team",
    kind="hand",
    unit="filler",
    fills="np",
    form=["the team"],
    meaning=[["name-span", "?x", "?l", "?r"], ["plural", "?x"]],
    score=0.8,
    status="live",
)


@pytest.mark.needs_fcg
def test_a_path_pairs_each_slot_with_its_filler_in_a_grammar_that_does_not_compose(engine):
    assert not composes([MEETS, TEAM])
    registry = default_registry([ToolSpec.simple("pair", ("a", "b"), False)])
    runtime = _runtime(engine, MEETS, TEAM, registry=registry)
    found = run(runtime.alternatives(turn("the report meets the team"), through="rt.meets", maximal=False)).found
    paths = {a.path for a in found}
    assert paths == {
        (("", "rt.meets"), ("x", x), ("y", y)) for x in ("np-the", "np-bare") for y in ("np.team", "np-the", "np-bare")
    }, paths
    assert len(found) == len(paths), [a.path for a in found]  # no path twice: no program counted twice
    # the search filled y before x: pairing slots with fillers in application order would put np.team at x
    assert any(a.plan.reading["applied"][1] == "np.team" for a in found), [a.plan.reading["applied"] for a in found]
    # two readings that part ways at y alone name the construction that read y there, and the same one at x
    team = next(a.path for a in found if dict(a.path)["y"] == "np.team" and dict(a.path)["x"] == "np-the")
    the = next(a.path for a in found if dict(a.path)["y"] == "np-the" and dict(a.path)["x"] == "np-the")
    assert [(p, q) for p, q in zip(team, the, strict=True) if p != q] == [(("y", "np.team"), ("y", "np-the"))]
    nested = run(runtime.alternatives(turn("my report meets the team"), through="rt.meets", maximal=False)).found
    mine = [a.path for a in nested if dict(a.path)["x"] == "np-my"]
    assert mine and all(dict(p)["x/x"] in ("np-bare", "np-the") and dict(p)["y"] for p in mine), mine


@pytest.mark.needs_fcg
def test_every_analysis_is_bounded_by_count_nodes_and_time(runtime, engine):
    run(runtime.plan(turn(ONE), window=0))  # loads the grammar
    text = normalise(LONG).text
    nodes = engine._lisp(CONFIG)
    readings, cut = engine.comprehend_every(text, cap=3, node_cap=5000, deadline_ms=10_000)
    assert len(readings) == 3 and cut and all(r.complete for r in readings)
    primary = engine.comprehend(text)  # the same search's first solution, in the same shape
    assert primary is not None and spec_id(readings[0].applied[0]) == ROOT
    assert (readings[0].meaning, readings[0].applied, readings[0].slots) == (
        primary.meaning,
        primary.applied,
        primary.slots,
    )
    for reading in readings:  # per construction applied, the slot unit it filled; the root filled none
        assert len(reading.fills) == len(reading.applied) and reading.fills[0] == "", reading
        assert sorted(f for f in reading.fills if f) == sorted(slot for _, slot, _, _ in reading.slots), reading
    readings, cut = engine.comprehend_every(normalise(ONE).text, cap=16, node_cap=5000, deadline_ms=10_000)
    assert len(readings) == 2 and not cut  # the search ran out
    readings, cut = engine.comprehend_every(normalise(TWO).text, cap=16, node_cap=7, deadline_ms=10_000)
    assert cut and len(readings) < 5  # the node cap
    readings, cut = engine.comprehend_every(text, cap=16, node_cap=5000, deadline_ms=1)
    assert cut and len(readings) < 16  # the deadline
    assert engine._lisp(CONFIG) == nodes
    every, _ = engine.comprehend_every(text, cap=256, node_cap=5000, deadline_ms=10_000)
    # a grammar whose own node cap ends the first search: that search is comprehend's, then it starts again
    engine._post("/fcg-set-grammar-configuration", {"key": "max-nr-of-nodes", "value": 20, "grammar": GRAMMAR_ID})
    try:
        capped, _ = engine.comprehend_every(text, cap=256, node_cap=5000, deadline_ms=10_000)
        primary = engine.comprehend(text)
        assert engine._lisp(CONFIG) == 20
    finally:
        engine._post(
            "/fcg-set-grammar-configuration", {"key": "max-nr-of-nodes", "value": nodes, "grammar": GRAMMAR_ID}
        )
    assert primary is not None and (capped[0].meaning, capped[0].applied) == (primary.meaning, primary.applied)
    from memory_kernel.runtime.api import _program_key as key

    assert {key(r.meaning) for r in capped} == {key(r.meaning) for r in every}
    for bad in ({"cap": 0}, {"node_cap": 0}, {"deadline_ms": -1}, {"cap": True}):
        with pytest.raises(ValueError):
            engine.comprehend_every(text, **{"cap": 1, "node_cap": 10, "deadline_ms": 10, **bad})


@pytest.mark.needs_fcg
def test_the_settings_are_restored_when_the_search_raises(runtime, engine, monkeypatch):
    run(runtime.plan(turn(ONE), window=0))
    nodes, timeout = engine._lisp(CONFIG), engine.request_timeout
    anchor = "(fcg::set-configuration inv :max-nr-of-nodes NODES)"
    assert anchor in engine_module._EVERY_LISP
    monkeypatch.setattr(
        engine_module, "_EVERY_LISP", engine_module._EVERY_LISP.replace(anchor, f'{anchor} (error "boom")')
    )
    sent: list[float | None] = []
    request = engine._request

    def spy(method, endpoint, data, *, timeout=None):
        sent.append(timeout)
        return request(method, endpoint, data, timeout=timeout)

    monkeypatch.setattr(engine, "_request", spy)
    with pytest.raises(RuntimeEngineError, match="boom"):
        engine.comprehend_every(normalise(ONE).text, cap=4, node_cap=11, deadline_ms=7_000)
    assert max(t for t in sent if t is not None) >= 7.0 + CLIENT_SLACK_S and engine.request_timeout == timeout
    monkeypatch.undo()
    assert engine.pid is not None and engine._lisp(CONFIG) == nodes  # answered with an error: not torn down


@pytest.mark.needs_fcg
def test_a_path_names_every_filled_slot_leftmost_first(runtime):
    found = run(runtime.alternatives(turn(ONE))).found
    assert found[0].path == (("", ROOT), ("s", "cx.because"), ("s/h", GENERATED), ("s/d", GENERATED))
    paths = [a.path for a in run(runtime.alternatives(turn(TWO))).found]
    assert BECAUSE_BUT in paths and BUT_BECAUSE in paths, paths


def test_a_path_is_built_from_the_slot_units_and_what_each_construction_filled():
    """No engine: a reading's slot units (holder, slot, left, right) give the tree, its fill record (the slot unit each
    applied construction filled) the construction in each slot, whatever the order they were applied in; a slot's
    address is its slot names from the root."""
    from memory_kernel.runtime.api import _path

    reading = Reading(
        meaning=[],
        applied=(ROOT, "cx.because", "cx.but", "clause", "clause", "clause"),
        fills=("", "slot-s-1", "slot-d-1", "slot-d-5", "slot-h-1", "slot-h-5"),
        ms=0.0,
        complete=True,
        slots=(
            ("root.s-unit-1", "slot-s-1", 2, 47),
            ("slot-s-1", "slot-h-1", 2, 11),
            ("slot-s-1", "slot-d-1", 20, 47),
            ("slot-d-1", "slot-h-5", 20, 30),
            ("slot-d-1", "slot-d-5", 35, 47),
        ),
    )
    assert GENERATED == CLAUSE_FILLER.id == "clause"
    assert _path(reading) == (
        ("", ROOT, None),
        ("s", "cx.because", (2, 47)),
        ("s/h", GENERATED, (2, 11)),
        ("s/d", "cx.but", (20, 47)),
        ("s/d/h", GENERATED, (20, 30)),
        ("s/d/d", GENERATED, (35, 47)),
    )
    for applied, fills in (
        ((*reading.applied, "clause"), (*reading.fills, "slot-x-9")),  # a slot the reading does not hold
        (reading.applied, ("", "slot-s-1", "slot-d-1", "slot-d-5", "slot-h-1", "slot-h-1")),  # one slot twice
        (reading.applied, ()),  # no fill record: a reading from comprehend, not comprehend_every
    ):
        with pytest.raises(RuntimeError, match="pair"):
            _path(Reading(meaning=[], applied=applied, fills=fills, ms=0.0, complete=True, slots=reading.slots))


# -- drafts and standing --------------------------------------------------------------------------------------------------


@pytest.mark.needs_fcg
def test_no_alternative_applies_a_draft_not_admitted_here(engine):
    draft = _connective("so", provenance={"teacher": "Zed", "chat_id": "c2"}, status="staged")
    runtime = _runtime(engine, *_rows(), draft)
    text = "it works so it is open"
    ana = run(runtime.alternatives(turn(text))).found
    assert ana and not any("cx.so" in dict(a.path).values() for a in ana), [a.path for a in ana]
    zed = run(runtime.alternatives(turn(text, "Zed", "c2"))).found
    assert any(outer(a) == "cx.so" for a in zed), [a.path for a in zed]


@pytest.mark.needs_fcg
def test_standing_is_never_recorded_for_an_alternative(engine):
    writer = MemoryWriter()
    runtime = _runtime(engine, *_rows(), writer=writer, record_standing=True)
    for alternative in run(runtime.alternatives(turn(TWO))).found:
        assert walk(runtime, alternative.plan)[1] == "answered"
    assert not [c for c in writer.calls if c[0] == "add_uses"], writer.calls
    assert runtime.window("c1") == []  # nor is an alternative in the chat's turn window
    plan = run(runtime.plan(turn(TWO), window=0))  # the control: a plan walked to its end records its uses
    walk(runtime, plan)
    assert [c for c in writer.calls if c[0] == "add_uses"]


def test_the_bounds_are_the_briefs():
    assert (ALTERNATIVES, ALTERNATIVES_RAW, ALTERNATIVES_NODE_CAP, ALTERNATIVES_DEADLINE_MS) == (16, 256, 5000, 10_000)


# -- one analysis -----------------------------------------------------------------------------------------------------------

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "office-catalogue.json"


def _cards(engine: Any, **kw: Any) -> Runtime:
    office = parse_catalogue(FIXTURE.read_bytes(), str(FIXTURE))
    return _runtime(engine, registry=default_registry(office.tools), cards=office.cards, **kw)


@pytest.fixture
def engine_calls(engine, monkeypatch):
    """Every ``comprehend`` and ``comprehend_all`` the engine is asked for, by name."""
    calls: list[str] = []
    for name in ("comprehend", "comprehend_all"):
        real = getattr(engine, name)

        def counted(*args: Any, _real: Any = real, _name: str = name, **kwargs: Any) -> Any:
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(engine, name, counted)
    return calls


def _steps(plan: TurnPlan) -> list[tuple[str, Any]]:
    return [(s["tool"], s["args"]) for s in plan.tool_steps]


@pytest.mark.needs_fcg
def test_with_one_analysis_a_plan_makes_one_engine_read_and_the_same_steps(engine, engine_calls):
    default, one = _cards(engine), _cards(engine, analyses=1)
    assert (default.analyses, one.analyses) == (3, 1)
    cases = {
        "take JUJ-3": None,  # a card's sentence
        "yes": "unread",  # a reply
        "lovely weather": "unread",  # nothing reads it
        "search about the monkey situation and write note about it": None,  # two marked verbs
    }
    for text, status in cases.items():
        engine_calls.clear()
        want = run(default.plan(turn(text)))
        wide = list(engine_calls)
        engine_calls.clear()
        got = run(one.plan(turn(text)))
        assert "comprehend_all" not in engine_calls, (text, engine_calls)
        assert got.status == want.status and want.status == (status or want.status), (text, got.diagnostics)
        assert status is not None or (want.status != "unread" and want.reading["complete"]), (text, want.status)
        assert _steps(got) == _steps(want) and got.reading["applied"] == want.reading["applied"], text
        assert got.frames[:1] == want.frames[:1] and all(f["analysis"] == 0 for f in got.frames), (text, got.frames)
        if text.startswith("search"):  # the default reads the bound roles again; one analysis makes no further read
            assert engine_calls == ["comprehend"] and wide.count("comprehend") > 1, (wide, engine_calls)
    assert engine_calls  # the spy saw the reads


def test_analyses_is_a_positive_integer():
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="analyses"):
            Runtime("bonfire", object(), _NoGraph(), TOOLS, lexicon=_lexicon(), store=memory_store(), analyses=bad)  # type: ignore[arg-type]


# -- fix round 1 ----------------------------------------------------------------------------------------------------------


@pytest.mark.needs_fcg
def test_the_first_alternative_is_plans_reading_even_when_the_deadline_ends_the_search(runtime, monkeypatch):
    """The alternatives' deadline bounds the wider search only: the first search is plan's, bounded as plan's is."""
    plan = run(runtime.plan(turn(TWO), window=0))
    monkeypatch.setattr(api_module, "ALTERNATIVES_DEADLINE_MS", 1)
    cut = run(runtime.alternatives(turn(TWO)))
    assert cut.cut_short and cut.found, (cut.cut_short, [a.path for a in cut.found])
    assert cut.found[0].plan.reading["applied"] == plan.reading["applied"]
    assert walk(runtime, cut.found[0].plan) == walk(runtime, plan)


@pytest.mark.needs_fcg
def test_a_search_that_does_not_reproduce_plans_reading_raises_and_keeps_no_plan(runtime, engine, monkeypatch):
    run(runtime.plan(turn(ONE), window=0))
    real = engine.comprehend_every

    def without_the_first(text: str, **bounds: Any):
        readings, cut = real(text, **bounds)
        return readings[1:], cut

    monkeypatch.setattr(engine, "comprehend_every", without_the_first)
    kept = len(runtime._plans)
    with pytest.raises(RuntimeEngineError, match="reproduce"):
        run(runtime.alternatives(turn(ONE)))
    assert len(runtime._plans) == kept


@pytest.mark.needs_fcg
def test_a_reading_with_no_path_is_left_out_with_a_warning(runtime, engine, monkeypatch, caplog):
    run(runtime.plan(turn(TWO), window=0))
    every = run(runtime.alternatives(turn(TWO), maximal=False)).found
    real = engine.comprehend_every
    broken: list[int] = []

    def breaking(text: str, **bounds: Any):
        readings, cut = real(text, **bounds)
        return [replace(r, fills=()) if i in broken else r for i, r in enumerate(readings)], cut

    monkeypatch.setattr(engine, "comprehend_every", breaking)
    broken[:] = [2]
    with caplog.at_level(logging.WARNING, logger="memory_kernel.runtime.api"):
        left = run(runtime.alternatives(turn(TWO), maximal=False)).found
    assert len(left) == len(every) - 1 and left[0].path == every[0].path
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "cx." in warnings[0].getMessage(), [r.getMessage() for r in warnings]
    assert "works" not in warnings[0].getMessage() and "nobody" not in warnings[0].getMessage()
    broken[:] = [0]  # plan's own reading: the caller could not credit it
    with pytest.raises(RuntimeEngineError, match="path"):
        run(runtime.alternatives(turn(TWO)))


@pytest.mark.needs_fcg
def test_alternatives_over_the_office_catalogue(engine):
    """Any bonfire's grammar: the office catalogue's cards over the base set, for the sentences of the one-analysis
    test. No call raises, and the first alternative is plan's reading wherever plan reads the turn."""
    runtime = _cards(engine)
    for text in (
        "take JUJ-3",
        "yes",
        "lovely weather",
        "search about the monkey situation and write note about it",
    ):
        plan = run(runtime.plan(turn(text), window=0))
        found = run(runtime.alternatives(turn(text))).found
        if plan.reading["complete"] and plan.reading["root"]:
            assert found and found[0].plan.reading["applied"] == plan.reading["applied"], (text, found)
            assert found[0].path[0] == ("", plan.reading["root"]), (text, found[0].path)
        else:
            assert found == () or found[0].plan.reading["complete"], text


@pytest.mark.needs_fcg
def test_a_plan_takes_one_snapshot_of_the_constructions(runtime, monkeypatch):
    """The constructions a plan's root comes from are the ones its hand and its gate see: a change to the store while
    the plan reads the bound roles (a teach or a confirm in another chat) does not reach them."""
    want = run(runtime.plan(turn(ONE), window=0))
    snapshot = runtime._all_specs
    hop = runtime._slot_frames
    changed: list[bool] = []

    def specs() -> dict[str, CxnSpec]:
        known = snapshot()
        return {k: v for k, v in known.items() if k != ROOT} if changed else known

    def reading_the_roles(*args: Any) -> Any:
        changed.append(True)
        return hop(*args)

    monkeypatch.setattr(runtime, "_all_specs", specs)
    monkeypatch.setattr(runtime, "_slot_frames", reading_the_roles)
    got = run(runtime.plan(turn(ONE), window=0))
    assert changed and got.hand == want.hand and got.status == want.status, (got.hand, want.hand)


def test_a_programs_key_ignores_the_order_of_its_predicates():
    from memory_kernel.runtime.api import _program_key

    one = [
        ["name-span", "?x-1", 0, 10],
        ["definite", "?x-1"],
        ["name-span", "?y-2", 18, 26],
        ["pair", "?p", "?x-1", "?y-2"],
    ]
    other = [
        ["pair", "?q-7", "?a-3", "?b-9"],
        ["name-span", "?b-9", 18, 26],
        ["definite", "?a-3"],
        ["name-span", "?a-3", 0, 10],
    ]
    assert _program_key(one) == _program_key(other)
    swapped = [["pair", "?q", "?b", "?a"], ["name-span", "?b", 18, 26], ["definite", "?a"], ["name-span", "?a", 0, 10]]
    assert _program_key(one) != _program_key(swapped)  # which referent is the pair's first is not a variable name
    assert _program_key(one) != _program_key([*one, ["plural", "?y-2"]])
