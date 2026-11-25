"""Teach makes a filler (2.1, spec part 3): ``TeachRequest(fills=<slot kind>)`` stages a FILLER of that kind, with as many
slots as it needs, that refers to its own whole span (its body names it ``{X}``), read back inside a live root with a
slot of its kind before it is staged. A root (``fills`` None) teaches exactly as on 2.0.2: the same draft ids and rows.

The grammar is a statement root over one clause slot (``{S:clause}``, a hand row: teach refuses a form with no literal),
with ``statement`` and ``relation`` registered as tools. Every sentence is made up.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from memory_kernel.runtime import teach as teach_module
from memory_kernel.runtime.api import GENERATED, Runtime, TurnPlan, default_registry
from memory_kernel.runtime.base import base_specs
from memory_kernel.runtime.constructions import CxnSpec, Standing, entrenched, spec_id
from memory_kernel.runtime.engine import Reading, RuntimeEngineError
from memory_kernel.runtime.executor import Turn
from memory_kernel.runtime.frames import frame_parts
from memory_kernel.runtime.grammar import GRAMMAR, GRAMMAR_HASH, Example, GrammarError, GrammarFile, install
from memory_kernel.runtime.lexicon import Lexicon, QuestionMap
from memory_kernel.runtime.normalise import normalise
from memory_kernel.runtime.program import ToolSpec
from memory_kernel.runtime.store import ConstructionStore, load_store
from memory_kernel.runtime.teach import (
    MODIFIES,
    TeachError,
    TeachRequest,
    _draft_id,
    _parse_form,
    admissible,
    confirm,
    expand,
    resolver,
    teach,
)
from memory_kernel.serve.runtime import draft_json, parse_teach
from unit.runtime.memory_writer import memory_store
from unit.runtime.test_store_teach import BREW, CARDS, COMPOSED, COMPOSED_REGISTRY, REGISTRY, SITREP

PORT = 9731  # 9713-9730 are the other engine test modules'
NOW = "2026-11-25T10:00:00+00:00"
TOOLS = default_registry(
    [
        ToolSpec.simple("statement", ("text",), False),
        ToolSpec.simple("relation", ("kind", "word", "whole", "head", "text"), False),
    ]
)
ROOT = "root.s"
SAY = [["procedure", "?p", "say"], ["seq", "?p", "1", "?st"], ["statement", "?st", "?s"]]
STATEMENT = CxnSpec(
    id=ROOT, kind="hand", unit="root", form=[{"slot": "s", "fills": "clause"}], meaning=SAY, status="live"
)
EVEN_FORM = "{H:clause} even though {D:clause}"
SENTENCE = "I stay even though it rains"


def _objection(kind: str = "objection", head: str = "{H}", text: str = "{D}") -> dict[str, Any]:
    args = {"kind": kind, "word": "even though", "whole": "{X}", "head": head, "text": text}
    return {"call": "relation", "args": args}


EVEN = TeachRequest([EVEN_FORM], [_objection()], {}, "Ana", "c1", fills="clause")
#: the id the EVEN draft's one row takes (the slug of its first literal, the draft id, the form's index)
EVEN_ID = f"even-though.{_draft_id(EVEN)}.0"
#: no engine: a teach refused before it reads anything never touches one
NO_ENGINE: Any = None


class _Untouched:
    """An engine a teach must not touch: any use of it fails the test."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the teach used the engine ({name})")


class _StandIn:
    """A stand-in engine (no FCG): it records the ``slot_order`` of every load and each read back's text and deadline,
    and answers a read back with ``every(text)``; a whole-turn read (``competes``) finds nothing."""

    def __init__(self, every: Any) -> None:
        self.every = every
        self.slot_orders: list[bool] = []
        self.reads: list[tuple[str, int]] = []

    def load(self, cxns: Any, *, tiers: Any = None, slot_order: bool = False) -> None:
        self.slot_orders.append(slot_order)

    def comprehend(self, text: str) -> None:
        return None

    def comprehend_every(self, text: str, *, cap: int, node_cap: int, deadline_ms: int) -> tuple[list[Reading], bool]:
        self.reads.append((text, deadline_ms))
        return self.every(text)


def _untouched() -> Any:
    return _Untouched()


def _engine(every: Any) -> Any:
    return _StandIn(every)


def _reading(*applied: str) -> Reading:
    return Reading(meaning=[], applied=applied, ms=0.0, complete=True)


def _reads_through(*applied: str) -> Any:
    """A read back answer: one complete reading that applied ``applied``."""
    return lambda text: ([_reading(*applied)], False)


# -- the engine and the helpers ------------------------------------------------------------------------------------------


def _data_dir():
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """One engine for the module, stopped by pid; each test loads its own store into it."""
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("runtime-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


def _store(engine: Any, *rows: CxnSpec) -> ConstructionStore:
    store = memory_store(STATEMENT, *rows)
    load_store(store, engine)
    return store


class _NoGraph:
    async def rows(self, cypher: str, params: Any) -> list[dict[str, Any]]:
        return []


def _runtime(engine: Any, store: ConstructionStore) -> Runtime:
    lexicon = Lexicon([], QuestionMap.load())
    return Runtime("bonfire", engine, _NoGraph(), TOOLS, lexicon=lexicon, store=store, record_standing=False)


def _turn(text: str, speaker: str = "Ana") -> Turn:
    return Turn(text, speaker, "Kira", (speaker, "Kira"), "c1", None, at=NOW)


def _walk(runtime: Runtime, plan: TurnPlan) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """Every tool step of ``plan`` (tool and args), each answered as committed, until the plan ends; and its status."""
    steps: list[tuple[str, dict[str, Any]]] = []
    while plan.status == "needs_tools":
        steps += [(s["tool"], s["args"]) for s in plan.tool_steps]
        results = [
            {"step_id": s["step_id"], "tool": s["tool"], "args": s["args"], "verdict": "committed", "value": None}
            for s in plan.tool_steps
        ]
        plan = asyncio.run(runtime.resume(plan.plan_id, results))
    return steps, plan.status


def _multiset(steps: list[tuple[str, dict[str, Any]]]) -> Counter[str]:
    return Counter(json.dumps(step, sort_keys=True) for step in steps)


# -- a filler, taught, read inside a root, confirmed -------------------------------------------------------------------


@pytest.mark.needs_fcg
def test_a_filler_is_staged_read_inside_a_root_and_confirmed(engine):
    """The connective stages as a taught clause filler with no score (it starts at 0.5), names the root it was read
    inside, and once confirmed reads inside the statement root: the relation gets the whole span, the head and the
    dependent as written, and the statement its whole clause."""
    store = _store(engine)
    draft = teach(store, engine, EVEN, registry=TOOLS)
    assert draft.problems == [] and draft.draft_id, draft.problems
    (spec,) = draft.specs
    assert (spec.kind, spec.unit, spec.fills, spec.status) == ("taught", "filler", "clause", "staged")
    assert spec.form == [{"slot": "h", "fills": "clause"}, "even though", {"slot": "d", "fills": "clause"}]
    assert spec.meaning[0] == ["clause-span", "?x", "?l", "?r"]  # it refers to its own whole span
    assert spec.score is None and "score" not in spec.to_json() and entrenched(spec) == 0.5
    assert spec.provenance["request"]["fills"] == "clause"
    assert draft.inside == {EVEN_FORM: ROOT}
    assert draft_json(draft)["inside"] == {EVEN_FORM: ROOT}
    assert [s.id for s in store.staged()] == [spec.id]

    (live,) = confirm(store, engine, draft.draft_id, by="Ana")
    assert live.status == "live" and live.score is None and entrenched(live) == 0.5
    runtime = _runtime(engine, store)
    plan = asyncio.run(runtime.plan(_turn(SENTENCE), window=0))
    assert plan.reading is not None
    applied = {spec_id(name) for name in plan.reading["applied"]}
    assert {ROOT, live.id} <= applied, plan.reading
    steps, status = _walk(runtime, plan)
    assert status == "answered", status
    relation = {"kind": "objection", "word": "even though", "whole": SENTENCE, "head": "I stay", "text": "it rains"}
    assert _multiset(steps) == _multiset([("statement", {"text": SENTENCE}), ("relation", relation)]), steps


@pytest.mark.needs_fcg
def test_a_staged_filler_is_its_teachers_alone(engine):
    """Before it is confirmed the draft filler applies in a reading only its teacher, in its chat, may use."""
    store = _store(engine)
    draft = teach(store, engine, EVEN, registry=TOOLS)
    assert draft.problems == [], draft.problems
    every, _ = engine.comprehend_every(normalise(SENTENCE).text, cap=16, node_cap=5000, deadline_ms=5000)
    through = [r for r in every if draft.specs[0].id in {spec_id(a) for a in r.applied}]
    assert through, [r.applied for r in every]
    assert admissible(through[0], store, "c1", "Ana") and not admissible(through[0], store, "c1", "Milo")


# -- what a filler teach refuses ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows", [(), (replace(STATEMENT, status="staged", provenance={"teacher": "Ana", "chat_id": "c1"}),)]
)
@pytest.mark.parametrize("body", [[_objection()], ["say {H}", _objection()]])
def test_a_filler_with_no_live_root_of_its_kind_is_refused(rows, body):
    """No live root has a clause slot (none at all, or only the teacher's own draft has one): the problem says which
    kind, nothing is staged, and nothing is read on the engine, a body clause included (the check is the store's)."""
    store = memory_store(*rows)
    before = store.all()
    draft = teach(store, _untouched(), replace(EVEN, body=body), registry=TOOLS)
    assert (draft.draft_id, draft.specs, draft.problems) == ("", [], ["no live root has a clause slot"])
    assert store.all() == before


@pytest.mark.parametrize(
    ("form", "base"),
    [
        ("it", "np-it"),
        ("my {Y}", "np-my"),
        ("before {T}", "time-before"),
        ("the {Y}", "np-the"),
        ("a {Y}", "np-a"),
        ("an {Y}", "np-an"),
    ],
)
def test_a_filler_with_a_base_fillers_skeleton_is_refused(form: str, base: str):
    """A form with the kind and skeleton of a base filler competes with it, as a root does with a base root: a taught
    row would load after it at the same score and win its tie for every speaker for good. A slot stands where a base
    filler consumes its span ("the {Y}" is np-the's "the" and what follows it)."""
    fills = "time" if base.startswith("time") else "np"
    store = memory_store(STATEMENT)
    req = TeachRequest([form], [{"call": "statement", "args": {"text": "{X}"}}], {}, "Ana", "c1", fills=fills)
    draft = teach(store, _untouched(), req, registry=TOOLS)
    problem = (
        f"form {form!r} competes with the base construction {base}: base constructions change only with a kernel "
        "release, never by teaching"
    )
    assert (draft.draft_id, draft.specs, draft.problems) == ("", [], [problem])
    assert store.staged() == []


def test_a_filler_cannot_replace():
    """A filler competes with no live row, so ``replace`` has nothing to take the place of: a problem, nothing read."""
    store = memory_store(STATEMENT)
    draft = teach(store, _untouched(), replace(EVEN, replace=True), registry=TOOLS)
    assert (draft.draft_id, draft.specs) == ("", [])
    assert draft.problems == [
        "a filler cannot replace: it competes with no live row, so there is none for it to take the place of (teach it "
        "without replace)"
    ]


# -- the bonfire's search, the deadline, the engine's errors (a stand-in engine) --------------------------------------


def test_a_staged_filler_leaves_the_bonfires_search_as_it_was():
    """The live rows decide whether the bonfire's grammar composes: staging a filler with clause slots loads the
    bonfire's grammar with the slot-order test off (the self-read searches the grammar as it will be once confirmed,
    composing); confirming it turns the test on."""
    engine = _engine(_reads_through(ROOT, EVEN_ID))
    store = memory_store(STATEMENT)
    draft = teach(store, engine, EVEN, registry=TOOLS)
    assert draft.problems == [] and [s.id for s in store.staged()] == [EVEN_ID], draft.problems
    assert engine.slot_orders == [True, False]  # the self-read's load, then the bonfire's with the draft staged
    load_store(store, engine)
    assert engine.slot_orders[-1] is False
    confirm(store, engine, draft.draft_id, by="Ana")
    assert engine.slot_orders[-1] is True


def test_the_read_back_has_one_deadline_across_roots(monkeypatch):
    """Every root's read gets what is left of one deadline (never more than the alternatives' own); a root reached
    after it is not read, and the problem says the deadline cut the read back short."""
    roots = [
        replace(STATEMENT, id=f"root.{word}", form=[word, {"slot": "s", "fills": "clause"}]) for word in ("so", "yes")
    ]

    def slow(text: str) -> tuple[list[Reading], bool]:
        time.sleep(0.06)
        return [], False

    monkeypatch.setattr(teach_module, "READ_BACK_DEADLINE_MS", 40)
    engine = _engine(slow)
    store = memory_store(STATEMENT, *roots)
    draft = teach(store, engine, EVEN, registry=TOOLS)
    assert (draft.draft_id, draft.specs) == ("", []) and store.staged() == []
    assert len(engine.reads) == 1 and 0 < engine.reads[0][1] <= 40, engine.reads
    (problem,) = draft.problems
    assert f"{ROOT} as 'Zed even though Zed'" in problem, problem
    assert "the read back's 40 ms cut it short before root.so, root.yes could be read" in problem, problem


def test_an_engine_error_in_the_read_back_is_the_engines():
    """The engine failing while a filler is read back leaves nothing staged, the bonfire's grammar reloaded, and the
    error the engine raised reaches the caller as itself (never a problem of the definition)."""
    died = RuntimeEngineError("runtime engine died")

    def fails(text: str) -> tuple[list[Reading], bool]:
        raise died

    engine = _engine(fails)
    store = memory_store(STATEMENT)
    with pytest.raises(RuntimeEngineError) as caught:
        teach(store, engine, EVEN, registry=TOOLS)
    assert caught.value is died
    assert store.staged() == [] and engine.slot_orders == [True, False]  # the self-read's load, then the store's


def test_another_teachers_draft_never_counts_in_a_read_back():
    """Milo's filler is read back: a reading that applied Ana's staged filler too is one Milo may not use, so it does
    not pass the read back; a reading through his own draft alone does."""
    store = memory_store(STATEMENT)
    assert teach(store, _engine(_reads_through(ROOT, EVEN_ID)), EVEN, registry=TOOLS).problems == []
    milo = TeachRequest(
        ["{H:clause} even if {D:clause}"], [_objection(kind="condition")], {}, "Milo", "c1", fills="clause"
    )
    mine = f"even-if.{_draft_id(milo)}.0"
    refused = teach(store, _engine(_reads_through(ROOT, mine, EVEN_ID)), milo, registry=TOOLS)
    assert refused.specs == [] and len(refused.problems) == 1, refused.problems
    assert refused.problems[0].startswith("form '{H:clause} even if {D:clause}' reads inside no live root")
    passed = teach(store, _engine(_reads_through(ROOT, mine)), milo, registry=TOOLS)
    assert passed.problems == [] and [s.id for s in passed.specs] == [mine], passed.problems


#: a root of the bonfire's own with an np slot
NOTE = CxnSpec(
    id="root.note",
    kind="hand",
    unit="root",
    form=["note", {"slot": "y", "fills": "np"}],
    meaning=[["procedure", "?p", "note"], ["seq", "?p", "1", "?st"], ["statement", "?st", "?y"]],
    status="live",
)
#: an np filler of a skeleton no base filler has (it opens on np-the's word, and goes on)
OF = TeachRequest(
    ["the {Y} of {Z}"],
    [{"call": "relation", "args": {"kind": "of", "word": "of", "whole": "{X}", "head": "{Y}", "text": "{Z}"}}],
    {},
    "Ana",
    "c1",
    fills="np",
)


def test_a_filler_is_read_back_in_the_bonfires_own_roots_first():
    """The read back tries the bonfire's own roots (in load order) before the base roots, so a filler that reads in a
    root of the bonfire's is not cut short by the base roots before it, and the draft names that root. "the {Y} of {Z}"
    is no base filler's rival: np-the is "the" and its span alone."""
    store = memory_store(STATEMENT, NOTE)
    mine = f"the.{_draft_id(OF)}.0"
    roots = [s.id for s in [*base_specs(), NOTE] if s.unit == "root" and teach_module._has_slot(s, "np")]
    engine = _engine(_reads_through(*roots, mine))  # every root it is tried in reads it
    draft = teach(store, engine, OF, registry=TOOLS)
    assert draft.problems == [] and [s.id for s in draft.specs] == [mine], draft.problems
    assert draft.inside == {"the {Y} of {Z}": NOTE.id}
    assert engine.reads[0][0] == normalise("note the Zed of Zed").text, engine.reads


@pytest.mark.parametrize(
    ("form", "body", "name"),
    [
        ("{X:clause} even though {D:clause}", _objection(head="{D}"), "X"),
        ("{x:clause} even though {D:clause}", _objection(head="{D}"), "x"),
        ("{H:clause} even though {L:clause}", _objection(text="{L}"), "L"),
        ("{H:clause} even though {r:clause}", _objection(text="{r}"), "r"),
    ],
)
def test_a_filler_slot_named_x_l_or_r_is_a_problem(form: str, body: dict[str, Any], name: str):
    """X is the filler's own span, l and r its edges: a slot of that name is a problem, never an exception, and
    nothing is staged."""
    store = memory_store(STATEMENT)
    req = TeachRequest([form], [body], {}, "Ana", "c1", fills="clause")
    draft = teach(store, NO_ENGINE, req, registry=TOOLS)
    problem = (
        f"form {form!r}: a filler's slot may not be named {name}: {{X}} is the filler's own span, and l and r are its "
        "edges"
    )
    assert (draft.draft_id, draft.specs) == ("", []) and problem in draft.problems, draft.problems
    assert store.staged() == []


@pytest.mark.parametrize(
    ("form", "body", "reason"),
    [
        ("{V:verb} even though {D:clause}", _objection(head="{V}"), "a verb slot inside a filler is not supported"),
        ("{H:bogus} even though {D:clause}", _objection(), "slot 'h' fills 'bogus', not one of"),
        ("{H:clause} {D:clause} even though", _objection(), "adjacent slots need a literal anchor between them"),
    ],
)
def test_a_filler_form_the_compiler_refuses_is_refused_with_its_reason(form: str, body: dict[str, Any], reason: str):
    store = memory_store(STATEMENT)
    req = TeachRequest([form], [body], {}, "Ana", "c1", fills="clause")
    draft = teach(store, NO_ENGINE, req, registry=TOOLS)
    assert (draft.draft_id, draft.specs) == ("", [])
    assert any(p.startswith(f"form {form!r}: ") and reason in p for p in draft.problems), draft.problems
    assert store.staged() == []


def test_a_filler_form_writes_any_slot_kind_and_a_root_form_as_before():
    """A filler's form may write every kind the compiler knows (the compiler judges the form); a root's form is parsed
    as on 2.0.2 (np and clause only)."""
    form = "{H:clause} until {T:time} by {N:name} for {Y}"
    parts, written, problems = _parse_form(form, fills="clause")
    assert problems == [] and written == ["H", "T", "N", "Y"]
    assert [p["fills"] for p in parts if isinstance(p, dict)] == ["clause", "time", "name", "np"]
    assert _parse_form(form)[2] == [
        f"form {form!r}: slot {{T:time}} has an unknown kind 'time': a slot is {{Name}}, {{Name:np}} or {{Name:clause}}",
        f"form {form!r}: slot {{N:name}} has an unknown kind 'name': a slot is {{Name}}, {{Name:np}} or {{Name:clause}}",
    ]


def test_fills_names_a_slot_kind():
    store = memory_store(STATEMENT)
    draft = teach(store, NO_ENGINE, replace(EVEN, fills="sentence"), registry=TOOLS)
    assert draft.specs == [] and any("fills" in p and "'sentence'" in p for p in draft.problems), draft.problems


@pytest.mark.needs_fcg
@pytest.mark.parametrize("cut", [False, True])
def test_a_read_back_that_never_applies_the_draft_is_refused(engine, monkeypatch, cut: bool):
    """The read inside the root finds no reading that applies the draft (or its search was cut short first): a problem
    naming the root and the text read, nothing staged."""
    store = _store(engine)
    monkeypatch.setattr(engine, "comprehend_every", lambda text, **kw: ([], cut))
    draft = teach(store, engine, EVEN, registry=TOOLS)
    assert (draft.draft_id, draft.specs) == ("", []) and store.staged() == []
    (problem,) = draft.problems
    assert problem.startswith(f"form {EVEN_FORM!r} reads inside no live root with a clause slot") and ROOT in problem
    assert ("cut short" in problem) == cut, problem


# -- duplicates and rivals ---------------------------------------------------------------------------------------------


@pytest.mark.needs_fcg
def test_a_duplicate_filler_is_refused_and_a_rival_is_not(engine):
    """The same filler again is refused naming the live row; a filler for the same word with another meaning, or in
    another form, is staged and confirmed (rivals are settled by entrenchment, not at teach). A root with the
    filler's skeleton is no rival either."""
    twin_root = CxnSpec(
        id="root.even-though",
        kind="hand",
        unit="root",
        form=[{"slot": "a", "fills": "clause"}, "even though", {"slot": "b", "fills": "clause"}],
        meaning=[*SAY[:2], ["statement", "?st", "?a"], ["seq", "?p", "2", "?st2"], ["statement", "?st2", "?b"]],
        status="live",
    )
    store = _store(engine, twin_root)
    first = teach(store, engine, EVEN, registry=TOOLS)
    assert first.problems == [], first.problems
    (live,) = confirm(store, engine, first.draft_id, by="Ana")

    again = teach(store, engine, replace(EVEN, teacher="Milo", chat_id="c2"), registry=TOOLS)
    assert (again.draft_id, again.specs) == ("", [])
    assert again.problems == [f"form {EVEN_FORM!r} is already the live construction {live.id}"]

    rival = teach(store, engine, replace(EVEN, body=[_objection(kind="concession")]), registry=TOOLS)
    assert rival.problems == [] and [s.status for s in rival.specs] == ["staged"], rival.problems
    (rival_live,) = confirm(store, engine, rival.draft_id, by="Ana")
    assert rival_live.id != live.id and {live.id, rival_live.id} <= {s.id for s in store.live()}

    fronted = TeachRequest(["even though {D:clause}, {H:clause}"], [_objection()], {}, "Ana", "c1", fills="clause")
    other = teach(store, engine, fronted, registry=TOOLS)
    assert other.problems == [] and [s.status for s in other.specs] == ["staged"], other.problems


@pytest.mark.needs_fcg
def test_a_filler_with_a_new_skeleton_of_a_base_kind_is_staged(engine):
    """An np filler whose skeleton no base filler has is no base row's rival: it stages, read inside a live root with
    an np slot."""
    store = _store(engine)
    req = TeachRequest(["that crew"], [{"call": "statement", "args": {"text": "{X}"}}], {}, "Ana", "c1", fills="np")
    draft = teach(store, engine, req, registry=TOOLS)
    assert draft.problems == [] and [(s.unit, s.fills) for s in draft.specs] == [("filler", "np")], draft.problems
    (root,) = draft.inside.values()
    held = {s.id: s for s in store.known_specs()}[root]
    assert held.unit == "root" and "np" in [p.get("fills") for p in held.form if isinstance(p, dict)]


@pytest.mark.needs_fcg
def test_a_staged_filler_never_applies_in_someone_elses_reading(engine):
    """Ana's staged connective is in no plan and no alternative of Milo's turn; Ana's own alternatives hold it."""
    store = _store(engine)
    draft = teach(store, engine, EVEN, registry=TOOLS)
    assert draft.problems == [] and [s.id for s in draft.specs] == [EVEN_ID], draft.problems
    runtime = _runtime(engine, store)

    def applied(plan: TurnPlan) -> set[str]:
        return {spec_id(name) for name in (plan.reading or {}).get("applied", ())}

    plan = asyncio.run(runtime.plan(_turn(SENTENCE, "Milo"), window=0))
    assert ROOT in applied(plan) and EVEN_ID not in applied(plan), plan.reading
    milos = asyncio.run(runtime.alternatives(_turn(SENTENCE, "Milo"))).found
    assert milos and all(EVEN_ID not in applied(a.plan) for a in milos)
    anas = asyncio.run(runtime.alternatives(_turn(SENTENCE))).found
    assert any(EVEN_ID in applied(a.plan) for a in anas), [a.path for a in anas]


# -- roots, as on 2.0.2 ------------------------------------------------------------------------------------------------


def _canonical_rows(rows: list[dict[str, Any]]) -> str:
    """The rows' JSON (sorted keys) with each row's meaning variables renamed in order of first appearance: the
    engine's per-application suffix on a read clause's variables ("?c1-p-3") is no part of what a teach writes."""
    out = []
    for row in rows:
        names: dict[str, str] = {}
        meaning = [
            [names.setdefault(a, f"?v{len(names)}") if isinstance(a, str) and a.startswith("?") else a for a in p]
            for p in row["meaning"]
        ]
        out.append({**row, "meaning": meaning})
    return hashlib.sha256(json.dumps(out, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


#: what memory-kernel 2.0.2 (fe2fc2b3, its own code, the same engine) stages for two cases of ``test_store_teach``:
#: (draft id, sha256 of ``_canonical_rows`` of the staged rows' ``to_json``). Re-pin only for a change to the base set,
#: the compiler or teach that is meant to change what every root teach stages.
ON_2_0_2 = {
    "brew": ("t5c554e72ab", "d3be47617e69b4e235b1f72fe99a951d9dca7a5f9ead626e54226e16b887232d"),
    "sitrep": ("t3a17b42434", "7700b6395c918bfe7cbcc8d78ef59da1d9839f1be70d3598abc6682dd71f0572"),
}


#: the rows hold the forms as the default parser (trf) normalises them; the small model reads some differently
@pytest.mark.skipif(
    os.environ.get("MK_EXTRACTION_SPACY_MODEL") == "en_core_web_sm", reason="the rows are pinned on the trf parser"
)
@pytest.mark.needs_fcg
@pytest.mark.parametrize("case", sorted(ON_2_0_2))
def test_a_root_teaches_as_on_2_0_2(engine, case: str):
    """A root taught with the old fields (``fills`` None) stages the draft id and rows 2.0.2 staged, and its answer
    says nothing of a filler's read back."""
    store = memory_store()
    if case == "brew":
        for card in CARDS:
            store.put(card)
        req, registry = BREW, REGISTRY
    else:
        store.attach_cards(COMPOSED.cards)
        req, registry = SITREP, COMPOSED_REGISTRY
    load_store(store, engine)
    assert req.fills is None
    draft = teach(store, engine, req, registry=registry)
    assert draft.problems == [], draft.problems
    assert (draft.draft_id, _canonical_rows([s.to_json() for s in draft.specs])) == ON_2_0_2[case]
    assert draft.inside == {} and "inside" not in draft_json(draft)


def test_a_roots_draft_id_is_as_on_2_0_2():
    """The pinned roots' draft ids, in every configuration (the rows' pin needs the engine and the trf parser)."""
    assert {"brew": _draft_id(BREW), "sitrep": _draft_id(SITREP)} == {case: pin[0] for case, pin in ON_2_0_2.items()}


# -- a stated affordance -----------------------------------------------------------------------------------------------

STATEMENT_X = {"call": "statement", "args": {"text": "{X}"}}
DONE = {"call": "statement", "args": {"text": "done"}}
LOG_FORM = "log {X} as a note"
LOG = TeachRequest([LOG_FORM], [STATEMENT_X], {}, "Ana", "c1")


def _empty(engine: Any) -> ConstructionStore:
    """A store of the base set alone (no root reads every sentence, so a partial reading shows its frames)."""
    store = memory_store()
    load_store(store, engine)
    return store


def _live(store: ConstructionStore, engine: Any, req: TeachRequest) -> CxnSpec:
    """``req`` taught and confirmed by its teacher: the one row it puts live."""
    draft = teach(store, engine, req, registry=TOOLS)
    assert draft.problems == [] and draft.draft_id, draft.problems
    (live,) = confirm(store, engine, draft.draft_id, by=req.teacher)
    return live


def _frames(engine: Any, store: ConstructionStore, text: str) -> list[tuple[str, str, bool]]:
    plan = asyncio.run(_runtime(engine, store).plan(_turn(text), window=0))
    return [(f["name"], f["construction"], f["complete"]) for f in plan.frames]


@pytest.mark.needs_fcg
def test_a_root_taught_with_an_affordance_raises_its_frame_on_those_words(engine):
    """A root taught with ``affordance`` raises its frame on each of those words, and not on its verb; one taught with
    ``[]`` raises none on its verb; one taught with no affordance is stored as on 2.0.2 (no key) and raises its verb's
    frame, as it did."""
    store = _empty(engine)
    stated = _live(store, engine, replace(LOG, affordance=["fees", "costs"]))
    assert stated.affordance == ["fees", "costs"] and stated.provenance["request"]["affordance"] == ["fees", "costs"]
    for text, word in (("what about costs", "costs"), ("fees again", "fees")):
        assert _frames(engine, store, text) == [(stated.id, f"frame.{stated.id}.{word}", False)], text
    assert all(name != stated.id for name, _, _ in _frames(engine, store, "log it"))

    none = _live(store, engine, TeachRequest(["file {X} under notes"], [STATEMENT_X], {}, "Ana", "c1", affordance=[]))
    assert none.affordance == [] and frame_parts([none]) == []
    assert all(name != none.id for name, _, _ in _frames(engine, store, "file it"))

    plain = _live(store, engine, TeachRequest(["keep {X} as a memo"], [STATEMENT_X], {}, "Ana", "c1"))
    assert "affordance" not in plain.to_json() and set(plain.provenance["request"]) == {
        "forms",
        "body",
        "defaults",
        "replace",
    }
    assert (plain.id, f"frame.{plain.id}.keep", False) in _frames(engine, store, "keep it")


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"affordance": ["late fees"]}, "affordance word 'late fees' holds a space"),
        ({"affordance": ["fees", "Fees"]}, "affordance words 'fees' and 'Fees' are one word"),
        (
            {"affordance": ["though"], "fills": "clause", "forms": [EVEN_FORM], "body": [_objection()]},
            "a filler raises no",
        ),
    ],
)
def test_an_affordance_teach_refuses_is_a_problem(changes: dict[str, Any], problem: str):
    store = memory_store(STATEMENT)
    draft = teach(store, _untouched(), replace(LOG, **changes), registry=TOOLS)
    assert (draft.draft_id, draft.specs) == ("", []) and any(p.startswith(problem) for p in draft.problems), draft
    assert store.staged() == []


# -- modify ------------------------------------------------------------------------------------------------------------


def _modify(row_id: str, **parts: Any) -> TeachRequest:
    """A modify of ``row_id`` by Ana in c1 giving only ``parts`` (forms, body, affordance)."""
    return TeachRequest(
        parts.get("forms", []),
        parts.get("body", []),
        {},
        "Ana",
        "c1",
        modify=row_id,
        affordance=parts.get("affordance"),
    )


@pytest.mark.needs_fcg
def test_a_live_root_is_modified_one_part_at_a_time(engine):
    """A live taught root modified four times, each modify giving one part alone: a new body, a new affordance, a new
    form of the same skeleton, a new form of another skeleton. After each confirm the parts not given are the old
    row's, the old row is gone, its standing is kept (and the confirm counted) unless the skeleton changed, and a body
    that invoked the first row runs the last one's program."""
    store = _empty(engine)
    l0 = _live(store, engine, LOG)
    keep = _live(store, engine, TeachRequest(["keep {Y}"], ["log {Y} as a note"], {}, "Ana", "c1"))
    assert any(p[:3] == ["invoke", "?t-s1", l0.id] for p in keep.meaning), keep.meaning
    store.record_uses({l0.id: "ok"})
    store.record_uses({l0.id: "ok"})
    store.record_choices([(l0.id, "a-row-not-stored")])
    l0 = store.get(l0.id)
    assert l0 is not None and (l0.standing.uses_ok, l0.standing.won, l0.standing.confirmed) == (2, 1, 1)

    # a new body: until the confirm the old row is what everyone reads
    draft = teach(store, engine, _modify(l0.id, body=[STATEMENT_X, DONE]), registry=TOOLS)
    assert draft.problems == [], draft.problems
    (staged,) = draft.specs
    assert (staged.status, staged.provenance[MODIFIES], staged.provenance["replaces"]) == ("staged", l0.id, [l0.id])
    assert store.get(l0.id) == l0
    (l1,) = confirm(store, engine, draft.draft_id, by="Ana")
    assert store.get(l0.id) is None
    assert (l1.kind, l1.unit, l1.form, l1.affordance) == ("taught", "root", l0.form, l0.affordance)
    assert l1.meaning != l0.meaning and l1.provenance["form"] == LOG_FORM
    assert l1.standing == replace(l0.standing, confirmed=2)
    assert l1.provenance["replaced"] == [
        {"id": l0.id, "kind": "taught", "teacher": "Ana", "by": "Ana", "successor": l1.id}
    ]

    # a new affordance
    l2 = _live(store, engine, _modify(l1.id, affordance=["memo"]))
    assert (l2.form, l2.meaning, l2.affordance) == (l1.form, l1.meaning, ["memo"])
    assert l2.standing == replace(l1.standing, confirmed=3)
    assert [p.id for p in frame_parts([l2])] == [f"frame.{l2.id}.memo"]
    assert (l2.id, f"frame.{l2.id}.memo", False) in _frames(engine, store, "what about the memo")

    # a new form of the same skeleton (the same slots, the same normalised literals: a capital "Log" would not be, the
    # parser reads it as a noun)
    l3 = _live(store, engine, _modify(l2.id, forms=["log {X} as a Note"]))
    assert l3.form == ["log", {"slot": "x", "fills": "np"}, "as a Note"]
    assert (l3.meaning, l3.affordance) == (l2.meaning, l2.affordance)
    assert l3.standing == replace(l2.standing, confirmed=4)

    # a new form of another skeleton: the standing starts again
    l4 = _live(store, engine, _modify(l3.id, forms=["log {X} as a memo"]))
    assert (l4.meaning, l4.affordance) == (l3.meaning, l3.affordance)
    assert l4.standing == Standing(confirmed=1)
    assert {r["id"] for r in l4.provenance["replaced"]} == {l0.id, l1.id, l2.id, l3.id}
    assert {s.id for s in store.live()} == {keep.id, l4.id}

    # the body that invoked the first row runs the last one's program
    expanded, used, problems = expand(keep.meaning, resolver(store, "c1", "Ana"))
    assert problems == [] and used == [l4.id], (used, problems)
    runtime = _runtime(engine, store)
    steps, status = _walk(runtime, asyncio.run(runtime.plan(_turn("keep the cake", "Milo"), window=0)))
    assert status == "answered" and steps == [("statement", {"text": "cake"}), ("statement", {"text": "done"})]
    plan = asyncio.run(runtime.plan(_turn("log the cake as a memo", "Milo"), window=0))
    assert plan.reading is not None and l4.id in {spec_id(a) for a in plan.reading["applied"]}, plan.reading
    # a modify of the first row names the row the chain leads to now, never a row gone since
    refused = teach(store, _untouched(), _modify(l0.id, affordance=[]), registry=TOOLS)
    assert refused.problems == [f"modify {l0.id}: it was replaced by {l4.id} (modify that one)"], refused.problems
    # the stated affordance and the carried standing are what the writer holds: they survive a restart
    assert ConstructionStore(store.writer).all() == store.all()
    # a choice recorded against the first row's id (a pick on alternatives offered before the modifies) counts for the
    # row its chain leads to
    won = l4.standing.won
    store.record_choices([(l0.id, "a-row-not-stored")])
    assert store.get(l4.id).standing.won == won + 1  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.needs_fcg
def test_a_staged_modify_is_its_teachers_alone_and_one_row_takes_one_successor(engine):
    """Two teachers stage a modify of one live row: until a confirm, everyone else reads the old row; the first confirm
    takes its place, and the second is refused at confirm (the row it modifies is gone), changing nothing."""
    store = _empty(engine)
    l0 = _live(store, engine, LOG)
    ana = teach(store, engine, _modify(l0.id, body=[STATEMENT_X, DONE]), registry=TOOLS)
    milo = teach(store, engine, replace(_modify(l0.id, body=[DONE, STATEMENT_X]), teacher="Milo"), registry=TOOLS)
    assert ana.problems == [] and milo.problems == [], (ana.problems, milo.problems)
    plan = asyncio.run(_runtime(engine, store).plan(_turn("log the cake as a note", "Kira"), window=0))
    applied = {spec_id(a) for a in (plan.reading or {}).get("applied", ())}
    assert l0.id in applied and not applied & {s.id for s in store.staged()}, plan.reading
    (live,) = confirm(store, engine, ana.draft_id, by="Ana")
    before = store.all()
    with pytest.raises(TeachError) as refused:
        confirm(store, engine, milo.draft_id, by="Milo")
    assert str(refused.value) == (
        f"draft {milo.draft_id} modifies {l0.id}, which is no longer live: it was replaced by {live.id} (modify that one)"
    )
    assert store.all() == before


def _clause(name: str) -> dict[str, str]:
    return {"slot": name, "fills": "clause"}


def _connective(cxn_id: str, word: str, kind: str, score: float) -> dict[str, Any]:
    meaning = [["clause-span", "?x", "?l", "?r"], ["const", "?k", kind], ["const", "?w", word]]
    return {
        "id": cxn_id,
        "unit": "filler",
        "fills": "clause",
        "score": score,
        "form": [_clause("h"), word, _clause("d")],
        "meaning": [*meaning, ["relation", "?rel", "?k", "?w", "?x", "?h", "?d"]],
    }


THOUGH = "I stay though it rains"
#: a grammar file: a statement root, two connectives, one example (through "though": "unless" is in no example)
ARGUMENT = GrammarFile(
    name="argument-test",
    sha256="a" * 64,
    rows=(
        {"id": "g.say", "unit": "root", "score": 0.2, "form": [_clause("s")], "meaning": SAY},
        _connective("g.though", "though", "concession", 0.7),
        _connective("g.unless", "unless", "exception", 0.6),
    ),
    examples=(
        Example(
            THOUGH,
            (
                ("statement", {"text": THOUGH}),
                (
                    "relation",
                    {"kind": "concession", "word": "though", "whole": THOUGH, "head": "I stay", "text": "it rains"},
                ),
            ),
        ),
    ),
)


@pytest.mark.needs_fcg
def test_a_modified_installed_row_is_taught_and_a_later_install_leaves_it(engine):
    """A modify of an installed filler (its meaning alone; the body writes the stored slots in capitals) gives a taught
    row that keeps the grammar in its provenance and the file's score; installing the file again reports the file's
    row as kept (``kept_taught``) and leaves the taught row as it is."""
    store = memory_store()
    assert install(store, engine, ARGUMENT, registry=TOOLS).examples == 1
    old = store.get("g.unless")
    assert old is not None and old.kind == "hand"
    body = [
        {
            "call": "relation",
            "args": {"kind": "condition", "word": "unless", "whole": "{X}", "head": "{H}", "text": "{D}"},
        }
    ]
    draft = teach(store, engine, _modify(old.id, body=body), registry=TOOLS)
    assert draft.problems == [], draft.problems
    assert draft.inside == {"{H:clause} unless {D:clause}": "g.say"}
    (row,) = confirm(store, engine, draft.draft_id, by="Ana")
    assert (row.kind, row.unit, row.fills, row.form, row.score) == ("taught", "filler", "clause", old.form, 0.6)
    assert (row.provenance[GRAMMAR], row.provenance[GRAMMAR_HASH]) == (ARGUMENT.name, ARGUMENT.sha256)
    assert row.meaning[0] == ["clause-span", "?x", "?l", "?r"] and row.meaning != old.meaning
    assert store.get(old.id) is None

    again = install(store, engine, ARGUMENT, registry=TOOLS)
    assert again.kept_taught == (old.id,) and (again.added, again.replaced, again.removed) == (0, 2, 0)
    assert store.get(old.id) is None and store.get(row.id) == row


@pytest.mark.needs_fcg
def test_a_grammar_files_root_states_its_affordance_and_a_bad_word_refuses_the_file(engine):
    """A grammar file's root may state its affordance: the file installs, the row keeps it (its frame parts are the
    stated words) and the example reads as stated. A word the frame parts cannot take, or an affordance on a filler,
    refuses the whole file with a problem line naming the row, and nothing is written."""
    root, though, unless = ARGUMENT.rows
    store = memory_store()
    done = install(
        store, engine, replace(ARGUMENT, rows=({**root, "affordance": ["say"]}, though, unless)), registry=TOOLS
    )
    installed = store.get("g.say")
    assert done.examples == 1 and installed is not None and installed.affordance == ["say"]
    assert [p.id for p in frame_parts([installed])] == ["frame.say.say"]
    for rows, problem, reason in (
        (
            ({**root, "affordance": ["two words"]}, though, unless),
            "g.say: affordance word 'two words'",
            "holds a space",
        ),
        ((root, {**though, "affordance": ["though"]}, unless), "g.though: does not compile", "only a root states"),
    ):
        fresh = memory_store()
        with pytest.raises(GrammarError) as refused:
            install(fresh, engine, replace(ARGUMENT, rows=rows), registry=TOOLS)
        assert any(p.startswith(problem) and reason in p for p in refused.value.problems), refused.value.problems
        assert fresh.all() == []


#: a live taught wh root that narrows its answer (``answer_shape``)
ZORK = CxnSpec(
    id="zork.t0000000020.0",
    kind="taught",
    unit="root",
    form=["zork", {"slot": "x", "fills": "np"}, "?"],
    meaning=[["find-entity", "?e", "?x"], ["answer", "?e"]],
    answer_shape="who",
    status="live",
    provenance={"teacher": "Kira", "chat_id": "c9", "draft_id": "t0000000020", "form": "zork {X}?"},
)


@pytest.mark.needs_fcg
def test_a_modify_keeps_every_stored_field_it_does_not_name(engine):
    """The parts a modify does not give are kept as stored, and so is everything else it does not name: a wh root
    modified in its affordance alone keeps its answer shape (the plan narrows its answers by it), its unit and its
    score."""
    store = memory_store(ZORK)
    load_store(store, engine)
    draft = teach(store, engine, _modify(ZORK.id, affordance=["oracle"]), registry=TOOLS)
    assert draft.problems == [] and draft.specs[0].answer_shape == "who", draft.problems
    (row,) = confirm(store, engine, draft.draft_id, by="Ana")
    assert (row.answer_shape, row.unit, row.fills, row.score) == ("who", "root", None, None)
    assert (row.form, row.meaning, row.affordance) == (ZORK.form, ZORK.meaning, ["oracle"])


NOTE_ROW = {
    "id": "g.note",
    "unit": "root",
    "form": ["note", {"slot": "y", "fills": "np"}],
    "meaning": [["procedure", "?p", "note"], ["seq", "?p", "1", "?st"], ["statement", "?st", "?y"]],
}
NOTE_CAKE = "note the cake"
#: a grammar of one root with a literal (a plain teach can write a form of its skeleton), one example
NOTES = GrammarFile(
    name="notes-test",
    sha256="b" * 64,
    rows=(NOTE_ROW,),
    examples=(Example(NOTE_CAKE, (("statement", {"text": "cake"}),)),),
)


@pytest.mark.needs_fcg
def test_a_replace_of_a_modified_installed_row_stays_the_grammars(engine):
    """Install, modify the installed root, then replace the modified row with a plain teach of its skeleton: the
    replacing row keeps the grammar's provenance keys and the replaced row's records, so a later install leaves it
    (one live root of that skeleton, the file's row reported in ``kept_taught``), a body that invoked the modified row
    runs the replacing one, and the installed row's id still resolves to it."""
    store = memory_store()
    install(store, engine, NOTES, registry=TOOLS)
    modified = _live(store, engine, _modify("g.note", body=[{"call": "statement", "args": {"text": "{Y}"}}, DONE]))
    keep = _live(store, engine, TeachRequest(["keep {Z}"], ["note {Z}"], {}, "Ana", "c1"))
    assert any(p[:3] == ["invoke", "?t-s1", modified.id] for p in keep.meaning), keep.meaning
    replacing = TeachRequest(
        ["note {Y}"], [{"call": "statement", "args": {"text": "{Y}"}}], {}, "Milo", "c2", replace=True
    )
    draft = teach(store, engine, replacing, registry=TOOLS)
    assert draft.problems == [] and draft.specs[0].provenance["replaces"] == [modified.id], draft.problems
    (row,) = confirm(store, engine, draft.draft_id, by="Milo")
    assert (row.provenance[GRAMMAR], row.provenance[GRAMMAR_HASH]) == (NOTES.name, NOTES.sha256)
    assert [r["id"] for r in row.provenance["replaced"]] == [modified.id, "g.note"]

    again = install(store, engine, NOTES, registry=TOOLS)
    assert again.kept_taught == ("g.note",) and (again.added, again.replaced, again.removed) == (0, 0, 0)
    assert again.through_taught == (NOTE_CAKE,)  # its example reads through the taught row: named, not judged
    roots = [s for s in store.live() if s.unit == "root" and teach_module._same_skeleton(s, row)]
    assert [s.id for s in roots] == [row.id]
    _, used, problems = expand(keep.meaning, resolver(store, "c1", "Ana"))
    assert problems == [] and used == [row.id], (used, problems)
    resolved, _ = resolver(store, "c1", "Ana")("g.note")
    assert resolved is not None and resolved.id == row.id


@pytest.mark.needs_fcg
def test_an_example_through_a_modified_row_is_named_and_not_judged(engine):
    """The door reads the examples on the grammar as it will be. An example whose reading applied a taught row this
    grammar keeps (here a connective modified in its meaning, so its relation is no longer the file's) is not judged
    against its stated steps: it is named in ``through_taught`` and the file installs again. An example that fails with
    no kept taught row in its reading still refuses the whole file."""
    store = memory_store()
    install(store, engine, ARGUMENT, registry=TOOLS)
    body = [
        {
            "call": "relation",
            "args": {"kind": "contrast", "word": "though", "whole": "{X}", "head": "{H}", "text": "{D}"},
        }
    ]
    _live(store, engine, _modify("g.though", body=body))
    again = install(store, engine, ARGUMENT, registry=TOOLS)
    assert again.kept_taught == ("g.though",) and again.through_taught == (THOUGH,)

    broken = Example("I stay", (("statement", {"text": "something else"}),))
    before = store.all()
    with pytest.raises(GrammarError) as refused:
        install(store, engine, replace(ARGUMENT, examples=(*ARGUMENT.examples, broken)), registry=TOOLS)
    judged = [p for p in refused.value.problems if p.startswith("example ")]
    assert len(judged) == 1 and judged[0].startswith("example 'I stay'"), refused.value.problems
    assert not any(THOUGH in p for p in refused.value.problems), refused.value.problems
    assert store.all() == before


@pytest.mark.needs_fcg
def test_a_file_whose_row_was_modified_in_its_form_installs_again(engine, monkeypatch):
    """A modify of an installed row's FORM ("though" taught over as "although") leaves the file's example unread by the
    grammar as it will be: the taught row no longer reads it. The door reads a failing example once more, with the
    file's rows in the kept rows' places: there it reads as stated, through the kept row, so it is named in
    ``through_taught``, not judged, and the file installs again (judging none, which the result says). An example that
    fails there too still refuses the file, and its line names the kept rows. The second read is made only when the
    install keeps taught rows and an example failed."""
    loads: list[int] = []
    real = engine.load

    def counted(*args: Any, **kw: Any) -> None:
        loads.append(1)
        real(*args, **kw)

    monkeypatch.setattr(engine, "load", counted)
    store = memory_store()
    install(store, engine, ARGUMENT, registry=TOOLS)
    assert len(loads) == 2  # the scratch grammar, then the bonfire's
    row = _live(store, engine, _modify("g.though", forms=["{H:clause} although {D:clause}"]))
    assert store.get("g.though") is None and row.form[1] == "although"

    loads.clear()
    again = install(store, engine, ARGUMENT, registry=TOOLS)
    assert (again.kept_taught, again.through_taught, again.examples) == (("g.though",), (THOUGH,), 0)
    assert again.note is not None and again.note.startswith("no example was judged"), again.note
    assert len(loads) == 3  # the scratch grammar, the file's rows in the kept rows' places, the bonfire's
    assert store.get(row.id) == row and store.get("g.though") is None

    broken = Example("I stay", (("statement", {"text": "something else"}),))
    before = store.all()
    with pytest.raises(GrammarError) as refused:
        install(store, engine, replace(ARGUMENT, examples=(*ARGUMENT.examples, broken)), registry=TOOLS)
    judged = [p for p in refused.value.problems if p.startswith("example ")]
    assert len(judged) == 1 and judged[0].startswith("example 'I stay'"), refused.value.problems
    assert "g.though" in judged[0], judged[0]  # the line names the kept taught rows
    assert store.all() == before


def _milos(store: ConstructionStore, engine: Any, text: str) -> tuple[frozenset[str], Counter[str], str]:
    """What Milo (not the teacher) reads ``text`` as: the spec ids its reading applied, its steps walked as committed,
    and how it ends."""
    runtime = _runtime(engine, store)
    plan = asyncio.run(runtime.plan(_turn(text, "Milo"), window=0))
    applied = frozenset(spec_id(name) for name in (plan.reading or {}).get("applied", ()))
    steps, status = _walk(runtime, plan)
    return applied, _multiset(steps), status


#: (Milo's turn, Ana's teach): a root draft at 0.5 over the 0.2 statement root; a FORM modify of an installed filler,
#: staged with the file's 0.6 above the clause filler's 0.5; a body modify staged level with its row, which then loses
#: a choice
STAGED = {
    "root-over-a-low-root": ("log the cake as a note", LOG),
    "filler-form-modify": ("I stay whereas it rains", _modify("g.unless", forms=["{H:clause} whereas {D:clause}"])),
    "row-lost-after-the-teach": (
        THOUGH,
        _modify(
            "g.though",
            body=[
                {
                    "call": "relation",
                    "args": {"kind": "contrast", "word": "though", "whole": "{X}", "head": "{H}", "text": "{D}"},
                }
            ],
        ),
    ),
}


@pytest.mark.needs_fcg
@pytest.mark.parametrize("route", sorted(STAGED))
def test_a_staged_draft_changes_no_one_elses_reading(engine, route: str):
    """Until the confirm, everyone else reads exactly what they read before the draft existed, whatever the draft's own
    score and counters say: a staged draft is loaded below every live row."""
    text, req = STAGED[route]
    store = memory_store()
    install(store, engine, ARGUMENT, registry=TOOLS)
    before = _milos(store, engine, text)
    assert before[2] == "answered" and "g.say" in before[0], before
    draft = teach(store, engine, req, registry=TOOLS)
    assert draft.problems == [] and draft.draft_id, draft.problems
    (staged,) = draft.specs
    if route == "row-lost-after-the-teach":
        store.record_choices([(GENERATED, "g.though")])  # one loss on the live row: the draft's standing is as taught
        load_store(store, engine)
        live = store.get("g.though")
        assert live is not None and entrenched(live) < entrenched(staged) == 0.7
    else:  # its own score is above the row Milo read in its place (g.say's 0.2; the clause filler's 0.5)
        assert entrenched(staged) > {"root-over-a-low-root": 0.2, "filler-form-modify": 0.5}[route]
    assert _milos(store, engine, text) == before


class _Scores:
    """A stand-in engine (no FCG) that records, per load, the score each construction is compiled with, and answers a
    read back with one complete reading through ``applied``; a whole-turn read finds nothing."""

    def __init__(self, *applied: str) -> None:
        self.applied = applied
        self.loads: list[dict[str, float]] = []

    def load(self, cxns: Any, *, tiers: Any = None, slot_order: bool = False) -> None:
        self.loads.append({spec_id(c.name): c.attributes["score"] for c in cxns})

    def comprehend(self, text: str) -> None:
        return None

    def comprehend_every(self, text: str, *, cap: int, node_cap: int, deadline_ms: int) -> tuple[list[Reading], bool]:
        return [_reading(*self.applied)], False


def test_a_staged_draft_compiles_below_every_live_row_wherever_it_loads_for_everyone():
    """``load_store`` and teach's self-read compile every staged draft loaded for everyone at ``DRAFT_SCORE``, below
    any score a live row can have, whatever its own score says; the draft being read back is compiled as it will be
    live, with its own."""
    from memory_kernel.runtime.constructions import DRAFT_SCORE, ENTRENCH_STEP, SCORE_RANGE

    milos = CxnSpec(
        id="whereas.t0000000009.0",
        kind="taught",
        unit="filler",
        fills="clause",
        score=0.6,
        form=[_clause("h"), "whereas", _clause("d")],
        meaning=[["clause-span", "?x", "?l", "?r"], ["statement", "?s1", "?h"], ["statement", "?s2", "?d"]],
        status="staged",
        provenance={"teacher": "Milo", "chat_id": "c2", "draft_id": "t0000000009"},
    )
    store = memory_store(STATEMENT, milos)
    engine = _Scores(ROOT, EVEN_ID)
    draft = teach(store, engine, EVEN, registry=TOOLS)
    assert draft.problems == [] and [s.id for s in draft.specs] == [EVEN_ID], draft.problems
    self_read, reloaded = engine.loads[0], engine.loads[-1]
    assert (self_read[milos.id], self_read[EVEN_ID], self_read[ROOT]) == (DRAFT_SCORE, 0.5, 0.5)
    assert (reloaded[milos.id], reloaded[EVEN_ID], reloaded[ROOT]) == (DRAFT_SCORE, DRAFT_SCORE, 0.5)
    assert SCORE_RANGE[0] - ENTRENCH_STEP >= DRAFT_SCORE  # a live row's score never reaches it (entrenched)
    assert store.get(milos.id) == milos  # the row keeps its own score


def test_a_choice_against_a_replaced_id_counts_for_the_live_row_it_leads_to():
    """``record_choices`` follows the replaced records: a pair naming an id a confirmed replace or modify took out is
    recorded on the live row its chain leads to; an id that leads nowhere is skipped as before; a pair whose two sides
    lead to one row is skipped whole."""
    now = replace(
        PASS,
        id="pass.t0000000031.0",
        provenance={
            "replaced": [
                {"id": "pass.t0000000030.0", "successor": "pass.t0000000031.0"},
                {"id": "pass.t0000000029.0", "successor": "pass.t0000000030.0"},
            ]
        },
    )
    store = memory_store(now)
    store.record_choices([("pass.t0000000029.0", "who-be-x"), ("never.stored", "pass.t0000000030.0")])
    assert (store.get(now.id).standing.won, store.get(now.id).standing.lost) == (1, 1)  # pyright: ignore[reportOptionalMemberAccess]
    assert store.writer.nodes[now.id]["won"] == 1
    store.record_choices([("pass.t0000000029.0", now.id)])
    assert (store.get(now.id).standing.won, store.get(now.id).standing.lost) == (1, 1)  # pyright: ignore[reportOptionalMemberAccess]


#: rows the refusals below look at
PASS = CxnSpec(
    id="pass.t0000000001.0",
    kind="taught",
    unit="root",
    form=["pass", {"slot": "x", "fills": "np"}, "to", {"slot": "y", "fills": "np"}],
    meaning=[
        ["procedure", "?t-proc", "pass"],
        ["seq", "?t-proc", "1", "?s1"],
        ["const", "?k", "pass"],
        ["const", "?w", "to"],
        ["relation", "?s1", "?k", "?w", "?x", "?x", "?y"],
    ],
    status="live",
    provenance={"teacher": "Kira", "chat_id": "c9", "draft_id": "t0000000001", "form": "pass {X} to {Y}"},
)
LIVE_EVEN = CxnSpec(
    id="even-though.t0000000002.0",
    kind="taught",
    unit="filler",
    fills="clause",
    form=[_clause("h"), "even though", _clause("d")],
    meaning=[["clause-span", "?x", "?l", "?r"], ["statement", "?st", "?h"], ["statement", "?st2", "?d"]],
    status="live",
    provenance={"teacher": "Kira", "chat_id": "c9", "draft_id": "t0000000002", "form": EVEN_FORM},
)
MILOS = replace(PASS, id="pass.t0000000003.0", status="staged", provenance={"teacher": "Milo", "chat_id": "c2"})
AFTER = replace(
    PASS,
    id="pass.t0000000004.0",
    form=["hand", {"slot": "x", "fills": "np"}, "to", {"slot": "y", "fills": "np"}],
    provenance={"replaced": [{"id": "pass.t0.0", "successor": "pass.t0000000004.0"}]},
)
CREW = CxnSpec(
    id="crew.t0000000005", kind="taught", unit="part", form=["crew"], meaning=[["const", "?x", "y"]], status="live"
)
CARD = replace(NOTE, id="card.note.0")


#: an installed root (``NOTES``' row, as the door stores it) and a catalogue card of another skeleton
INSTALLED_NOTE = CxnSpec.from_json(
    {**NOTE_ROW, "kind": "hand", "status": "live", "provenance": {GRAMMAR: NOTES.name, GRAMMAR_HASH: NOTES.sha256}}
)
JOT_CARD = replace(NOTE, id="card.jot.0", form=["jot", {"slot": "y", "fills": "np"}])


@pytest.mark.parametrize("replacing", [False, True], ids=["plain", "replace"])
def test_a_root_with_an_installed_rows_skeleton_is_refused_naming_the_modify(replacing: bool):
    """A taught root with the skeleton of an installed root is refused, with ``replace`` too: an installed row is
    modified, never replaced. The refusal calls it an installed row and names the way out; one with a catalogue card's
    skeleton keeps the catalogue's words. Nothing is read (the body is a structured step)."""
    store = memory_store(INSTALLED_NOTE)
    store.attach_cards([JOT_CARD])
    req = TeachRequest(["note {Z}", "jot {Z}"], [{"call": "statement", "args": {"text": "{Z}"}}], {}, "Ana", "c1")
    draft = teach(store, _untouched(), replace(req, replace=replacing), registry=TOOLS)
    assert (draft.draft_id, draft.specs) == ("", [])
    assert draft.problems == [
        "form 'note {Z}' competes with the installed row g.note of grammar 'notes-test': an installed row is changed "
        "by a modify, never replaced (modify g.note)",
        "form 'jot {Z}' competes with the catalogue construction card.jot.0: a catalogue row is never replaced by "
        "chat teaching",
    ], draft.problems


@pytest.mark.parametrize(
    ("req", "problem"),
    [
        (_modify("who-be-x", affordance=[]), "modify who-be-x: a base construction changes only with a kernel release"),
        (
            _modify(CARD.id, affordance=[]),
            "modify card.note.0: a catalogue card changes when its catalogue is installed",
        ),
        (
            _modify(ROOT, affordance=[]),
            "modify root.s: a hand row that no grammar installed changes where it was written",
        ),
        (_modify(MILOS.id, affordance=[]), "modify pass.t0000000003.0: it is a draft staged by 'Milo', not a live"),
        (
            _modify("pass.t0.0", affordance=[]),
            "modify pass.t0.0: it was replaced by pass.t0000000004.0 (modify that one)",
        ),
        (_modify("no.such", affordance=[]), "modify no.such: no stored construction has this id"),
        (_modify(CREW.id, affordance=[]), "modify crew.t0000000005: it is a part; teach makes and modifies roots and"),
        (_modify(PASS.id), "a modify gives nothing to change: give a form, a body or an affordance"),
        (replace(_modify(PASS.id, affordance=[]), replace=True), "modify takes the place of the construction it names"),
        (_modify(PASS.id, forms=["pass {X} to {Y}", "hand {X} to {Y}"]), "a modify changes one construction"),
        (_modify(PASS.id, forms=["pass {Y} to {X}"]), "form 'pass {Y} to {X}': a modify keeps the slots of pass."),
        (_modify(PASS.id, forms=["pass {X}"]), "form 'pass {X}': a modify keeps the slots of pass.t0000000001.0"),
        (replace(_modify(PASS.id, affordance=[]), fills="clause"), "modify pass.t0000000001.0: it is a root; a modify"),
        (_modify(LIVE_EVEN.id, affordance=["though"]), "a filler raises no frame: only a root states an affordance"),
        (_modify(PASS.id, forms=["pass {X} to {Y}"]), "modify pass.t0000000001.0: nothing changes"),
        (_modify(""), "modify names a live construction by its id (a non-empty string); got ''"),
    ],
)
def test_what_a_modify_refuses_is_a_problem(req: TeachRequest, problem: str):
    """A modify that cannot be what it says: of a base row, a card, a hand row no grammar installed, a draft, a part, a
    row gone (naming its successor) or never stored; one giving nothing, two forms, other slots, another unit or an
    affordance for a filler; with replace; or one that changes nothing. Each is a problem, nothing is staged and the
    engine is never touched."""
    store = memory_store(STATEMENT, PASS, LIVE_EVEN, MILOS, AFTER, CREW)
    store.attach_cards([CARD])
    before = store.all()
    draft = teach(store, _untouched(), req, registry=TOOLS)
    assert (draft.draft_id, draft.specs) == ("", []), draft
    assert any(p.startswith(problem) for p in draft.problems), draft.problems
    assert store.all() == before


# -- mk serve's teach body ---------------------------------------------------------------------------------------------


def test_the_serve_body_takes_fills():
    """``fills`` is a slot kind (a string) or null (a root); what kind it names is teach's to check (a draft with
    problems, not a 422). An unknown key is still refused."""
    body = {"forms": [EVEN_FORM], "body": [_objection()], "teacher": "Ana", "chat_id": "c1"}
    assert parse_teach(body | {"fills": "clause"}) == EVEN
    assert parse_teach(body) == replace(EVEN, fills=None)
    assert parse_teach(body | {"fills": None}) == replace(EVEN, fills=None)
    assert parse_teach(body | {"fills": "sentence"}) == replace(EVEN, fills="sentence")
    for bad in (1, "", " ", ["clause"], True, "c" * 257):
        assert isinstance(parse_teach(body | {"fills": bad}), str), bad
    refused = parse_teach(body | {"unit": "filler"})
    assert isinstance(refused, str) and refused.startswith("unknown teach fields ['unit']"), refused


def test_the_serve_body_takes_affordance_and_modify():
    """``affordance`` is a list of words or null; ``modify`` a construction id or null, and with it the forms and the
    body may be empty or absent (each kept as stored). What the words or the id are is teach's to check (a draft with
    problems); without ``modify`` the forms and the body must still be given."""
    body = {"forms": [EVEN_FORM], "body": [_objection()], "teacher": "Ana", "chat_id": "c1"}
    root = replace(EVEN, fills=None)
    assert parse_teach(body) == root and root.affordance is None and root.modify is None
    for words in ([], ["fees", "costs"], ["not one word"]):
        assert parse_teach(body | {"affordance": words}) == replace(root, affordance=words)
    assert parse_teach(body | {"affordance": None}) == root
    modify = {"modify": "log.t1.0", "teacher": "Ana", "chat_id": "c1"}
    assert parse_teach(modify | {"affordance": ["memo"]}) == _modify("log.t1.0", affordance=["memo"])
    assert parse_teach(modify | {"forms": [], "body": [DONE]}) == _modify("log.t1.0", body=[DONE])
    assert parse_teach(body | {"modify": None}) == root
    bad_bodies = [
        body | {"affordance": "fees"},
        body | {"affordance": [1]},
        body | {"affordance": ["f" * 257]},
        body | {"affordance": ["f"] * 33},
        modify | {"modify": ""},
        modify | {"modify": 3},
        modify | {"modify": "m" * 257},
        modify | {"forms": "log {X}"},
        body | {"forms": []},
        body | {"body": []},
    ]
    for bad in bad_bodies:
        assert isinstance(parse_teach(bad), str), bad
