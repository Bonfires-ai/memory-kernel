"""plan / resume: a turn becomes a TurnPlan (answered, needs_tools, repair or unread) on a fake graph; the turn window
resolves "it"; resume continues a program from its tool results and never re-runs a memory op."""

import asyncio
import json
import os
from collections.abc import Mapping
from typing import Any

import pytest

from memory_kernel.runtime.api import ResumeError, Runtime, UnknownPlan, default_registry
from memory_kernel.runtime.constructions import CxnSpec
from memory_kernel.runtime.executor import Turn
from memory_kernel.runtime.lexicon import Lexicon, QuestionMap
from memory_kernel.runtime.program import ToolSpec
from memory_kernel.runtime.teach import TeachRequest
from unit.runtime.memory_writer import memory_store

PORT = 9716  # 9728 the engine tests', 9713 the compiler's, 9714 the base set's; teaching's is 9722
NOW = "2026-10-18T15:30:00+00:00"

KIRA = {"uuid": "u1", "name": "Kira Dace", "labels": ["Entity", "Person"]}
ZORVEX = {"uuid": "u2", "name": "Zorvex", "labels": ["Entity", "Organisation"]}
NEWS = {"uuid": "u6", "name": "the merger", "labels": ["Entity", "Artifact"]}
TALK = {"uuid": "u7", "name": "Zorvex keynote", "labels": ["Entity", "Artifact"]}
MILO = {"uuid": "u8", "name": "Milo", "labels": ["Entity", "Person"]}
ANA = {"uuid": "u9", "name": "Ana", "labels": ["Entity", "Person"]}


class FakeGraph:
    """Answers the StoreMemory queries by their parameters (the test_base_set pattern): find-entity by exact name,
    events by frame and participant groups, role-filler by (frame, role) pairs; records every query."""

    def __init__(self, entities: list[dict], events: list[dict]) -> None:
        self.entities = {e["uuid"]: e for e in entities}
        self.events = events
        self.sent: list[tuple[str, dict]] = []

    async def rows(self, cypher: str, params: Mapping[str, Any]) -> list[dict]:
        self.sent.append((cypher, dict(params)))
        if "n.name = $name" in cypher:
            return [dict(e) for e in self.entities.values() if e["name"] == params["name"]]
        if "$groups" in cypher:
            return [
                {"uuid": ev["uuid"], "name": ev["uuid"]}
                for ev in self.events
                if ev["roleset"] in params["frames"]
                and all(
                    any(x in g["uuids"] and (g["roles"] is None or r in g["roles"]) for r, x in ev["edges"])
                    for g in params["groups"]
                )
            ]
        if "$pairs" in cypher:
            return [
                {**self.entities[x], "event": ev["uuid"]}
                for ev in self.events
                if ev["uuid"] in params["uuids"]
                for r, x in ev["edges"]
                if any(r == role and frame in (None, ev["roleset"]) for frame, role in params["pairs"])
            ]
        return []

    def names_looked_up(self) -> list[str]:
        return [p["name"] for c, p in self.sent if "n.name = $name" in c]


def _event(uuid: str, roleset: str, *edges: tuple[str, dict]) -> dict:
    return {"uuid": uuid, "roleset": roleset, "edges": [(role, entity["uuid"]) for role, entity in edges]}


def _turn(text: str, chat_id: str = "c1", speaker: str | None = "Ana", at: str | None = NOW) -> Turn:
    return Turn(
        text=text,
        speaker=speaker,
        addressee="Milo",
        participants=("Ana", "Milo"),
        chat_id=chat_id,
        message_id=None,
        at=at,
    )


async def _recall(text: str) -> dict:
    return {"context": f"recalled for {text}"}


# -- the engine and the runtime ---------------------------------------------------------------------------------------


def _data_dir():
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """One engine for the module, stopped by pid; each runtime loads its own bonfire's grammar into it."""
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("runtime-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


def _runtime(engine, tmp_path, graph: FakeGraph, rolesets: list[str], **kwargs: Any) -> Runtime:
    lexicon = Lexicon(rolesets, QuestionMap.load())
    store = kwargs.pop("store", None) or memory_store()
    return Runtime("bonfire", engine, graph, kwargs.pop("registry", None), lexicon=lexicon, store=store, **kwargs)


def _plan(runtime: Runtime, text: str, **kwargs: Any):
    window = kwargs.pop("window", 5)
    return asyncio.run(runtime.plan(_turn(text, **kwargs), window=window))


def _result(step: dict, verdict: str = "committed", value: Any = None, detail: str | None = None) -> dict:
    """A resume row for a pending tool step (read from the TurnPlan: ids are never hardcoded)."""
    return {
        "step_id": step["step_id"],
        "tool": step["tool"],
        "args": step["args"],
        "verdict": verdict,
        "value": value,
        "detail": detail,
    }


def _names(answer) -> list[str]:
    return [e["name"] for e in answer]


# -- tests ------------------------------------------------------------------------------------------------------------


@pytest.mark.needs_fcg
def test_plan_answers_where_work(engine, tmp_path):
    graph = FakeGraph([KIRA, ZORVEX], [_event("e1", "work.01", ("arg0", KIRA), ("arg2", ZORVEX))])
    runtime = _runtime(engine, tmp_path, graph, ["work.01"])
    plan = _plan(runtime, "where does Kira Dace work?")
    assert plan.status == "answered", plan.diagnostics
    assert plan.answer == [{"uuid": "u2", "name": "Zorvex", "kinds": ["Organisation"]}]
    assert plan.answer_shape == "where" and plan.reading["root"] == "where-do-x-v" and plan.reading["complete"]
    assert plan.tool_steps == [] and plan.licence == [] and plan.diagnostics == []
    either = next(p for p in plan.program if p[0] == "either")  # the event answer first, the state lines' second
    assert either[2] == next(p[1] for p in plan.program if p[0] == "role-filler")
    assert ["answer", either[1]] in plan.program
    ops = [q["op"] for q in plan.trace["queries"]]  # one StoreMemory; the state path's summary read rides along
    assert sorted(ops) == ["events", "find-entity", "role-filler", "state-lines"] and ops[0] == "find-entity"
    assert plan.trace["queries"][0]["params"]["name"] == "Kira Dace"
    where = next(card for card in plan.hand if card["id"] == "where-do-x-v")
    assert where["kind"] == "base" and where["maker"] == "base" and where["unit"] == "root"
    assert where["standing"] == {"uses_ok": 0, "uses_failed": 0, "uses_world": 0, "confirmed": 0}
    assert "find-entity(?xe ← ?x)" in where["rules"] and "answer(?ans)" in where["rules"]
    assert {card["id"] for card in plan.hand} == {"where-do-x-v", "verb", "np-bare"}
    assert json.loads(json.dumps(plan.to_json()))["plan_id"] == plan.plan_id  # JSON-ready

    bare = _plan(runtime, "where does Kira Dace work")  # no final mark: read once more with " ?"
    assert bare.status == "answered" and bare.reading["retried"] and bare.answer == plan.answer

    hello = _plan(runtime, "hello there")
    assert hello.status == "unread" and hello.hand == [] and hello.program == []
    assert hello.licence is None and hello.to_json()["licence"] is None  # no licence: today's tools (decision 12)
    assert hello.diagnostics == ["no construction read the turn"]


@pytest.mark.needs_fcg
def test_unknown_name_is_repair_not_answer(engine, tmp_path):
    """Review Focus 2: a slot naming nobody is a repair with a diagnostic and a recall result, never an answer built
    from another entity."""
    graph = FakeGraph([KIRA, ZORVEX], [_event("e1", "work.01", ("arg0", KIRA), ("arg2", ZORVEX))])
    runtime = _runtime(engine, tmp_path, graph, ["work.01"], recall=_recall)
    plan = _plan(runtime, "where does Zed Quill work?")
    assert plan.status == "repair" and plan.answer is None and plan.answer_shape == "where"
    assert plan.diagnostics == ["find-entity: 'Zed Quill' names nobody in the graph"]
    assert plan.memory_results["recall"] == {"context": "recalled for where does Zed Quill work?"}
    assert plan.licence is None and plan.hand  # a repair licenses nothing: the turn falls back; the hand shows the draw
    assert not any("$pairs" in c for c, _ in graph.sent)  # no role filler was read for anyone


@pytest.mark.needs_fcg
def test_gate_refuses_what_no_keyword_applies(engine, tmp_path):
    """The gate (carry-in pins): a root read but a constraint left unapplied, a quantified noun phrase or a guarded
    auxiliary is a repair naming it, never a wrong answer; the program never runs."""
    graph = FakeGraph([KIRA, MILO, ZORVEX], [])
    runtime = _runtime(engine, tmp_path, graph, ["meet.01", "work.01", "publish.01", "cover.01"])
    cases = [
        ("when did Kira Dace meet Milo?", "?other (find-entity) is never applied"),
        ("which talk covered Zorvex?", "?noun (const) is never applied"),
        ("where does every team work?", "'every team' is quantified"),
        ("how many reports did Kira Dace publish?", "?noun (const) is never applied: no keyword applies it yet"),
    ]
    for text, reason in cases:
        plan = _plan(runtime, text)
        assert plan.status == "repair", (text, plan.status)
        assert any(reason in d for d in plan.diagnostics), (text, plan.diagnostics)
        assert plan.trace["steps"] == [] and plan.answer is None and plan.licence is None, text
    assert graph.sent == []  # nothing ran
    untimed = _plan(runtime, "what did Milo publish yesterday?", at=None)  # within needs the turn's time
    assert untimed.status == "repair" and any("time" in d for d in untimed.diagnostics), untimed.diagnostics


@pytest.mark.needs_fcg
def test_answers_narrowed_by_kind(engine, tmp_path):
    """Object roots over ditransitive frames return every role filler; the answer shape narrows them by kind."""
    graph = FakeGraph(
        [ANA, MILO, ZORVEX, NEWS],
        [
            _event("e1", "sell.01", ("arg0", ANA), ("arg1", ZORVEX), ("arg2", MILO)),
            _event("e2", "tell.01", ("arg0", ANA), ("arg1", NEWS), ("arg2", MILO)),
        ],
    )
    runtime = _runtime(engine, tmp_path, graph, ["sell.01", "tell.01"])
    sold = _plan(runtime, "what did Ana sell?")
    assert sold.status == "answered" and _names(sold.answer) == ["Zorvex"], (sold.answer, sold.diagnostics)
    assert _names(sold.memory_results[next(p[1] for p in sold.program if p[0] == "answer")]) == ["Zorvex", "Milo"]
    told = _plan(runtime, "who did Ana tell?")
    assert told.status == "answered" and _names(told.answer) == ["Milo"], (told.answer, told.diagnostics)


#: a catalogue card: "pass X to Y" takes X, then gives it to the entity Y names (a memory op feeding a tool step)
PASS = CxnSpec(
    id="card-pass",
    kind="hand",
    unit="root",
    form=["pass", {"slot": "x", "fills": "np"}, "to", {"slot": "y", "fills": "np"}],
    meaning=[
        ["procedure", "?p", "pass"],
        ["seq", "?p", "1", "?s1"],
        ["take", "?s1", "?x"],
        ["seq", "?p", "2", "?s2"],
        ["find-entity", "?ye", "?y"],
        ["give", "?s2", "?x", "?ye"],
    ],
    status="live",
)
TOOLS = default_registry(
    [
        ToolSpec.simple("take", ("item",), False, reads=("bench",), writes=("hands",)),
        ToolSpec.simple("give", ("item", "person"), True, reads=("hands",), writes=("hands", "inventory")),
    ]
)


TOOLS_LICENCE = [
    {"tool": "take", "side_effect": False, "needs_confirmation": False, "reads": ["bench"], "writes": ["hands"]},
    {
        "tool": "give",
        "side_effect": True,
        "needs_confirmation": True,
        "reads": ["hands"],
        "writes": ["hands", "inventory"],
    },
]


def _tool_runtime(engine, tmp_path, graph: FakeGraph, **kwargs: Any) -> Runtime:
    store = kwargs.pop("store", None) or memory_store()
    store.put(PASS)
    return _runtime(engine, tmp_path, graph, [], registry=TOOLS, store=store, **kwargs)


@pytest.mark.needs_fcg
def test_needs_tools_then_resume_answers(engine, tmp_path):
    graph = FakeGraph([ANA], [])
    runtime = _tool_runtime(engine, tmp_path, graph)
    plan = _plan(runtime, "pass the tea to Ana")
    assert plan.status == "needs_tools", plan.diagnostics
    [take] = plan.tool_steps
    assert take == {
        "step_id": take["step_id"],
        "tool": "take",
        "args": {"item": "tea"},
        "side_effect": False,
        "needs_confirmation": False,
        "actor": "Ana",
        "checks": [],
    }
    assert plan.licence == [
        {"tool": "take", "side_effect": False, "needs_confirmation": False, "reads": ["bench"], "writes": ["hands"]},
        {
            "tool": "give",
            "side_effect": True,
            "needs_confirmation": True,
            "reads": ["hands"],
            "writes": ["hands", "inventory"],
        },
    ]
    assert next(card for card in plan.hand if card["id"] == "card-pass")["maker"] == "hand"

    second = asyncio.run(runtime.resume(plan.plan_id, [_result(take, value="taken")]))
    assert second.plan_id == plan.plan_id and second.status == "needs_tools"
    [give] = second.tool_steps
    assert give["tool"] == "give" and give["needs_confirmation"] is True
    assert give["args"] == {"item": "tea", "person": [{"uuid": "u9", "name": "Ana", "kinds": ["Person"]}]}

    done = asyncio.run(runtime.resume(plan.plan_id, [_result(give, value="given")]))
    assert done.status == "answered" and done.tool_steps == [] and done.diagnostics == []
    assert graph.names_looked_up() == ["Ana"]  # resume never re-ran the completed memory op
    assert any(step.get("reused") for step in done.trace["steps"])
    assert runtime.store.get("card-pass").standing.uses_ok == 1  # one use, recorded when the turn ended


@pytest.mark.needs_fcg
def test_resume_unknown_plan_or_step_errors(engine, tmp_path):
    """Review Focus 5: an unknown or expired plan_id, an unknown step_id, or a plan that waits on no tool is a clear
    error, and nothing is half-resumed."""
    now = [1000.0]
    graph = FakeGraph([ANA], [])
    runtime = _tool_runtime(engine, tmp_path, graph, clock=lambda: now[0])
    with pytest.raises(UnknownPlan):
        asyncio.run(runtime.resume("no-such-plan", [_result({"step_id": "0:0", "tool": "take", "args": {}})]))
    plan = _plan(runtime, "pass the tea to Ana")
    step = plan.tool_steps[0]
    with pytest.raises(ResumeError, match="not pending"):
        asyncio.run(runtime.resume(plan.plan_id, [_result({**step, "step_id": f"{plan.plan_id}:99:0"})]))
    with pytest.raises(ResumeError):
        asyncio.run(runtime.resume(plan.plan_id, [{"step_id": step["step_id"], "value": "no verdict"}]))
    resumed = asyncio.run(runtime.resume(plan.plan_id, [_result(step, value="taken")]))
    assert resumed.status == "needs_tools" and resumed.tool_steps[0]["tool"] == "give"  # the refusals changed nothing

    answered = _plan(runtime, "hello there")
    with pytest.raises(ResumeError, match="waits on no tool"):
        asyncio.run(runtime.resume(answered.plan_id, [_result(step, value="taken")]))
    now[0] += 15 * 60 + 1  # the plan expires 15 minutes after its last use
    with pytest.raises(UnknownPlan, match="expired"):
        asyncio.run(runtime.resume(plan.plan_id, [_result(resumed.tool_steps[0])]))


@pytest.mark.needs_fcg
def test_it_refers_to_previous_answer(engine, tmp_path):
    """ "what did Milo publish?" then "who gave it?": "it" is the entity the chat's latest turn answered with,
    handed on resolved (never looked up again by name). The carry-in's first turn, "which talk covered Zorvex?",
    leaves ?noun unapplied, so the gate makes it a repair (pinned below and in the gate test): a turn that did not
    answer leaves no referent, and "it" after it is refused rather than reaching back to an older answer."""
    graph = FakeGraph(
        [MILO, ANA, TALK, ZORVEX],
        [
            _event("e1", "publish.01", ("arg0", MILO), ("arg1", TALK)),
            _event("e2", "give.01", ("arg0", ANA), ("arg1", TALK)),
            _event("e3", "cover.01", ("arg0", TALK), ("arg1", ZORVEX)),
        ],
    )
    runtime = _runtime(engine, tmp_path, graph, ["publish.01", "give.01", "cover.01"])
    first = _plan(runtime, "what did Milo publish?")
    assert first.status == "answered" and _names(first.answer) == ["Zorvex keynote"], first.diagnostics
    second = _plan(runtime, "who gave it?")
    assert second.status == "answered" and _names(second.answer) == ["Ana"], second.diagnostics
    assert "np-it" in {card["id"] for card in second.hand}
    assert graph.names_looked_up() == ["Milo"]  # "it" travelled as the resolved entity
    assert runtime.window("c1")[0]["answer"] == second.answer

    that = _plan(runtime, "who gave that one?")  # "it" (Ana, a person) is now the latest answer: nobody gave Ana
    assert that.status == "repair" and "np-that-one" in {card["id"] for card in that.hand}

    elsewhere = _plan(runtime, "who gave it?", chat_id="c2")  # another chat's window is empty
    assert elsewhere.status == "repair" and any("window" in d for d in elsewhere.diagnostics), elsewhere.diagnostics

    _plan(runtime, "what did Milo publish?")
    unanswered = _plan(runtime, "which talk covered Zorvex?")
    assert unanswered.status == "repair"
    after = _plan(runtime, "who gave it?")
    assert after.status == "repair" and any("no entity" in d for d in after.diagnostics), after.diagnostics
    narrow = _plan(runtime, "who gave it?", window=0)
    assert narrow.status == "repair"


#: a catalogue card whose clause a taught body can read ("give X to Y")
GIVE = CxnSpec(
    id="card-give",
    kind="hand",
    unit="root",
    form=["give", {"slot": "x", "fills": "np"}, "to", {"slot": "y", "fills": "np"}],
    meaning=[["procedure", "?p", "give"], ["seq", "?p", "1", "?s1"], ["give", "?s1", "?x", "?y"]],
    status="live",
)


@pytest.mark.needs_fcg
def test_teach_refuses_the_turn_window_in_a_body(engine, tmp_path):
    """A taught body's referent is a slot, a constant or an earlier step's result: "it" with no step before it would
    read the turn window at use time and could feed an unrelated earlier answer into a side-effecting step. Refused,
    nothing staged."""
    runtime = _runtime(engine, tmp_path, FakeGraph([], []), [], registry=TOOLS, store=memory_store(GIVE))
    request = TeachRequest(
        forms=["hand over {X}"], body=["give it to Ana", "give {X} to Ana"], defaults={}, teacher="Ana", chat_id="c1"
    )
    draft = asyncio.run(runtime.teach(request))
    assert draft.draft_id == "" and draft.specs == []
    assert any("give it to Ana" in p and "slot or a constant" in p for p in draft.problems), draft.problems
    assert runtime.store.staged() == []


# -- fix round 2: the reviewer's probes ---------------------------------------------------------------------------

BRASK = {"uuid": "u3", "name": "Brask Labs", "labels": ["Entity", "Organisation"]}


class SlowGraph(FakeGraph):
    """A FakeGraph whose lookups of some names take a while (a slow graph read)."""

    def __init__(self, entities: list[dict], events: list[dict], slow: set[str], delay: float = 0.5) -> None:
        super().__init__(entities, events)
        self.slow, self.delay = slow, delay

    async def rows(self, cypher: str, params: Mapping[str, Any]) -> list[dict]:
        if params.get("name") in self.slow:
            await asyncio.sleep(self.delay)
        return await super().rows(cypher, params)


@pytest.mark.needs_fcg
def test_an_unresolved_name_hands_out_no_tool_step(engine, tmp_path):
    """C-1: a card whose name names nobody is a repair at plan time: no tool step, but the program's licence."""
    graph = FakeGraph([ANA], [])
    runtime = _tool_runtime(engine, tmp_path, graph)
    plan = _plan(runtime, "pass the tea to Brask Labs")
    assert plan.status == "repair" and plan.tool_steps == [] and plan.licence == TOOLS_LICENCE, plan
    assert plan.diagnostics[0] == "find-entity: 'Brask Labs' names nobody in the graph"
    with pytest.raises(ResumeError):
        asyncio.run(runtime.resume(plan.plan_id, [_result({"step_id": "0:0", "tool": "take", "args": {}})]))


@pytest.mark.needs_fcg
def test_an_unknown_verb_is_repair_not_no(engine, tmp_path):
    """I-1: a verb whose lemma names no frame is an empty result (repair + recall), never a polar "no"."""
    graph = FakeGraph([KIRA, MILO, ANA], [_event("e1", "meet.01", ("arg0", ANA), ("arg1", MILO))])
    runtime = _runtime(engine, tmp_path, graph, ["meet.01"], recall=_recall)
    for text in ("did Ana frobnicate?", "did anyone frobnicate Milo?"):
        plan = _plan(runtime, text)
        assert plan.status == "repair", (text, plan.status, plan.answer)
        assert "the verb 'frobnicate' names no frame" in plan.diagnostics, (text, plan.diagnostics)
        assert "recall" in plan.memory_results, text
    met = _plan(runtime, "did Ana meet?")
    assert met.status == "answered" and [e["uuid"] for e in met.answer] == ["e1"]


@pytest.mark.needs_fcg
def test_turns_of_one_chat_are_serialised(engine, tmp_path):
    """I-2: a turn said while the chat's previous turn is still running waits for it, so "it" is that turn's answer
    (not the one before)."""
    graph = SlowGraph(
        [MILO, ANA, KIRA, TALK, ZORVEX],
        [
            _event("e1", "publish.01", ("arg0", MILO), ("arg1", TALK)),
            _event("e2", "give.01", ("arg0", ANA), ("arg1", TALK)),
            _event("e3", "sell.01", ("arg0", ANA), ("arg1", ZORVEX), ("arg2", MILO)),
            _event("e4", "give.01", ("arg0", KIRA), ("arg1", ZORVEX)),
        ],
        slow={"Ana"},
    )
    runtime = _runtime(engine, tmp_path, graph, ["publish.01", "give.01", "sell.01"])
    assert _names(_plan(runtime, "what did Milo publish?").answer) == ["Zorvex keynote"]

    async def overlapping():
        first = asyncio.create_task(runtime.plan(_turn("what did Ana sell?")))
        await asyncio.sleep(0.15)  # the first turn is waiting on the graph
        second = asyncio.create_task(runtime.plan(_turn("who gave it?")))
        return await first, await second

    sold, gave = asyncio.run(overlapping())
    assert _names(sold.answer) == ["Zorvex"]
    window = next(step for step in gave.trace["steps"] if step["op"] == "window")
    assert window["out"]["uuid"] == "u2"  # Zorvex, the answer of the turn said just before
    assert gave.status == "answered" and _names(gave.answer) == ["Kira Dace"], gave.diagnostics


@pytest.mark.needs_fcg
def test_window_edges(engine, tmp_path):
    """M-7/M-2: an answer naming several entities makes "it" a repair; when the chat's newest plan has expired the
    window is empty (an older plan kept alive does not stand in); a refused resume does not keep a plan alive; a plan
    that expires mid-tools records no standing (M-1). "them" in a live turn refers to the results of a taught phrase's
    earlier steps, which a turn has none of: a repair, even after a turn that answered one entity."""
    now = [0.0]
    graph = FakeGraph(
        [MILO, ANA, TALK, ZORVEX, BRASK],
        [
            _event("e1", "publish.01", ("arg0", MILO), ("arg1", TALK)),
            _event("e2", "give.01", ("arg0", ANA), ("arg1", TALK), ("arg2", BRASK)),
            _event("e3", "sell.01", ("arg0", ANA), ("arg1", ZORVEX), ("arg2", MILO)),
        ],
    )
    runtime = _tool_runtime(engine, tmp_path, graph, store=memory_store(GIVE), clock=lambda: now[0])
    runtime.lexicon = Lexicon(["publish.01", "give.01", "sell.01"], QuestionMap.load())
    gave = _plan(runtime, "what did Ana give?", chat_id="m")
    assert gave.status == "answered" and sorted(_names(gave.answer)) == ["Brask Labs", "Zorvex keynote"]
    it = _plan(runtime, "who gave it?", chat_id="m")
    assert it.status == "repair" and any("2 entities" in d for d in it.diagnostics), it.diagnostics

    one = _plan(runtime, "what did Milo publish?", chat_id="r")
    assert one.status == "answered" and _names(one.answer) == ["Zorvex keynote"], one.diagnostics
    them = _plan(runtime, "give them to Ana", chat_id="r")
    assert them.status == "repair" and them.tool_steps == [], (them.status, them.tool_steps)
    assert any("taught phrase" in d for d in them.diagnostics), them.diagnostics
    assert not any("window" in d for d in them.diagnostics), them.diagnostics  # the turn window held one entity

    tools = _plan(runtime, "pass the tea to Ana", chat_id="t")
    assert tools.status == "needs_tools"
    _plan(runtime, "what did Milo publish?", chat_id="s")
    now[0] = 10.0
    _plan(runtime, "what did Ana sell?", chat_id="s")
    now[0] = 800.0
    with pytest.raises(ResumeError):  # refused: the tool plan is not touched (it expires on its own clock)
        asyncio.run(
            runtime.resume(tools.plan_id, [_result({**tools.tool_steps[0], "step_id": f"{tools.plan_id}:99:0"})])
        )
    now[0] = 950.0  # both "s" plans and the tool plan are past 15 minutes since their last use
    late = _plan(runtime, "who gave it?", chat_id="s")
    assert late.status == "repair" and any("holds 0" in d for d in late.diagnostics), late.diagnostics
    with pytest.raises(UnknownPlan):
        asyncio.run(runtime.resume(tools.plan_id, [_result(tools.tool_steps[0])]))
    standing = runtime.store.get("card-pass").standing
    assert (standing.uses_ok, standing.uses_failed) == (0, 0)  # an expired plan is no evidence either way


def test_an_engine_that_cannot_start_leaves_the_turn_unread(tmp_path):
    """M-5: a failure to start the engine is an unread turn with a diagnostic (today's path), never an exception."""
    from memory_kernel.runtime.engine import RuntimeEngineError

    class DeadEngine:
        pid = None

        def start(self) -> None:
            raise RuntimeEngineError("no FCG Go engine binary")

    lexicon = Lexicon([], QuestionMap.load())
    runtime = Runtime("bonfire", DeadEngine(), FakeGraph([], []), store=memory_store(), lexicon=lexicon)  # type: ignore[arg-type]
    plan = _plan(runtime, "where does Kira Dace work?")
    assert plan.status == "unread" and plan.licence is None
    assert plan.diagnostics == ["runtime engine: no FCG Go engine binary"]
    # nothing read the turn, so nothing consumed it: the whole turn is left, as on every other unread path
    assert plan.frames == [] and plan.left == ["where does Kira Dace work?"]


def test_find_entity_passes_through_a_referent_only():
    """M-6: only the window's Referent passes through find-entity; a dict that merely looks resolved is a name."""
    from memory_kernel.runtime.executor import Referent
    from memory_kernel.runtime.ops_memory import StoreMemory

    graph = FakeGraph([ZORVEX], [])
    memory = StoreMemory(graph, "g1")
    out = asyncio.run(memory.run("find-entity", {"name": Referent("u2", "Zorvex", ("Organisation",))}))
    assert out == [{"uuid": "u2", "name": "Zorvex", "kinds": ["Organisation"]}] and graph.sent == []
    asyncio.run(memory.run("find-entity", {"name": {"uuid": "u2", "name": "Zorvex"}}))
    assert graph.sent  # looked up (as a string), not trusted
