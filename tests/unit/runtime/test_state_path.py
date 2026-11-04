"""The state path of a wh root: when X's events give no answer, the answer is read from X's state lines, restricted to
the store kinds the question's roles expect (derived from their VerbNet themes in the generated question map)."""

import asyncio
import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from memory_kernel.graph.g1_names import KIND_LABELS
from memory_kernel.runtime.api import Runtime
from memory_kernel.runtime.executor import VALUE_SPECS, Turn, execute
from memory_kernel.runtime.lexicon import LEXICON_SPECS, Lexicon, LexiconOps, QuestionMap
from memory_kernel.runtime.normalise import normalise
from memory_kernel.runtime.ops_memory import MEMORY_SPECS, StoreMemory
from memory_kernel.runtime.program import Program, Registry
from unit.runtime.memory_writer import memory_store

PORT = 9718  # the evaluation's port: a test run and an evaluation never share the host's engines
REPO = Path(__file__).parents[3]
REGISTRY = Registry({**VALUE_SPECS, **MEMORY_SPECS, **LEXICON_SPECS})
NOW = "2026-10-18T15:30:00+00:00"

KIRA = {"uuid": "u1", "name": "Kira Dace", "labels": ["Entity", "Person"]}
BRASK = {"uuid": "u2", "name": "Brask Labs", "labels": ["Entity", "Organisation"]}
VELMORA = {"uuid": "u3", "name": "Velmora", "labels": ["Entity", "Place"]}
ZORVEX = {"uuid": "u6", "name": "Zorvex", "labels": ["Entity", "Organisation"]}
MILO = {"uuid": "u5", "name": "Milo", "labels": ["Entity", "Person"]}
ANA = {"uuid": "u9", "name": "Ana", "labels": ["Entity", "Person"]}

SUMMARIES = {
    "u1": "[Ana, 2026-10-01] Kira Dace is a researcher at Brask Labs in Velmora.",
    "u5": "[Ana, 2026-10-02] Milo is an engineer at Zorvex.",
}


def _script():
    path = REPO / "scripts" / "build_question_map.py"
    spec = importlib.util.spec_from_file_location("_build_question_map", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"uuid": row["uuid"], "name": row["name"], "kinds": [k for k in row["labels"] if k in KIND_LABELS]}


class FakeGraph:
    """Answers the StoreMemory queries by their parameters: find-entity by exact name, events by frame and participant
    groups, role-filler by (frame, role) pairs, summaries by uuid, kind mentions by label and text; records every
    query."""

    def __init__(self, entities: list[dict], events: list[dict], summaries: Mapping[str, str]) -> None:
        self.entities = {e["uuid"]: e for e in entities}
        self.events = events
        self.summaries = summaries
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
        if "n.summary AS summary" in cypher:
            return [
                {
                    "uuid": u,
                    "name": self.entities[u]["name"],
                    "summary": self.summaries[u],
                    "line_ids": [f"{u}-l1"],  # the line records the writer stores (1.4.1): one identity line each
                    "line_texts": [self.summaries[u]],
                    "identity_line_ids": [f"{u}-l1"],
                }
                for u in params["uuids"]
                if u in self.summaries
            ]
        if "$kind IN labels" in cypher:
            text = params["text"].lower()
            return [
                dict(e) for e in self.entities.values() if params["kind"] in e["labels"] and e["name"].lower() in text
            ]
        return []

    def ops(self) -> list[str]:
        return [c for c, _ in self.sent]


def _event(uuid: str, roleset: str, *edges: tuple[str, dict]) -> dict:
    return {"uuid": uuid, "roleset": roleset, "edges": [(role, entity["uuid"]) for role, entity in edges]}


# -- the generated kinds ----------------------------------------------------------------------------------------------


def test_theme_labels_are_store_kinds():
    """The theme -> store-label table is closed over VerbNet themes and KIND_LABELS; a role's labels are the union of
    its themes' rows; a theme outside the table gives none."""
    script = _script()
    assert dict(script.THEME_LABELS) == {
        "location": ("Place",),
        "destination": ("Place",),
        "source": ("Place",),
        "initial_location": ("Place",),
        "trajectory": ("Place",),
        "beneficiary": ("Organisation",),
    }
    assert all(set(labels) <= set(KIND_LABELS) for labels in script.THEME_LABELS.values())
    assert script.role_labels(("beneficiary",)) == ("Organisation",)
    assert script.role_labels(("destination", "patient")) == ("Place",)
    assert script.role_labels(("beneficiary", "destination")) == ("Organisation", "Place")
    assert script.role_labels(("Initial-Location",)) == ("Place",)  # VerbNet's spellings are one theme
    assert script.role_labels(("recipient",)) == () and script.role_labels(()) == ()


def _frames(tmp_path: Path) -> Path:
    frames = tmp_path / "frames"
    frames.mkdir()
    roles = "".join(
        (
            '<role descr="worker" f="PAG" n="0"><rolelinks>'
            '<rolelink class="x-1" resource="VerbNet" version="verbnet3.4">agent</rolelink></rolelinks></role>',
            '<role descr="employer" f="GOL" n="2"><rolelinks>'
            '<rolelink class="x-1" resource="VerbNet" version="verbnet3.4">beneficiary</rolelink></rolelinks></role>',
            '<role descr="where" f="LOC" n="M"/>',
            '<role descr="gone to" f="GOL" n="3"/>',
        )
    )
    (frames / "brask.xml").write_text(
        '<?xml version="1.0"?><frameset><predicate lemma="brask">'
        f'<roleset id="brask.01" name="brask"><roles>{roles}</roles></roleset></predicate></frameset>',
        encoding="utf-8",
    )
    return frames


def test_question_map_carries_role_kinds(tmp_path):
    """A role whose themes have store labels carries them as a third element; the shared argm-loc (and a frame's own
    argm-loc with no theme) is a location; ``pairs`` and ``roles`` read as before."""
    built = _script().build_question_map([_frames(tmp_path)])
    assert built["*"] == [["when", "argm-tmp"], ["where", "argm-loc", ["Place"]]]
    assert built["brask.01"] == [
        ["who_or_what", "arg0"],
        ["where", "arg2", ["Organisation"]],
        ["where", "arg3"],
        ["where", "argm-loc", ["Place"]],
    ]
    out = tmp_path / "question-map.json"
    _script().write_question_map(built, out)
    qm = QuestionMap.load(out)
    assert qm.pairs("brask.01") == [
        ("who_or_what", "arg0"),
        ("where", "arg2"),
        ("where", "arg3"),
        ("where", "argm-loc"),
    ]
    assert qm.kinds("brask.01", "arg2") == ("Organisation",) and qm.kinds("brask.01", "arg3") == ()
    assert qm.kinds("*", "argm-loc") == ("Place",) and qm.kinds("unheard.01", "arg0") == ()
    with pytest.raises(ValueError, match="kind"):
        QuestionMap({"brask.01": [["where", "arg2", ["Weekday"]]]})


def test_role_kinds_from_a_frames_own_roles_else_the_shared_ones():
    """``role-kinds``: per frame, the kinds of its own roles among the pairs; a frame with no own role among them
    takes the shared modifiers' kinds."""
    qm = QuestionMap(
        {
            "*": [["when", "argm-tmp"], ["where", "argm-loc", ["Place"]]],
            "brask.01": [["who_or_what", "arg0"], ["where", "arg2", ["Organisation"]]],
            "brask.02": [["who_or_what", "arg0"], ["where", "arg2"]],
            "quill.01": [["who_or_what", "arg0"]],
        }
    )
    lex = Lexicon([], qm)
    assert lex.role_kinds(lex.wh_roles(["brask.01"], "where")) == ["Organisation"]
    assert lex.role_kinds(lex.wh_roles(["brask.02"], "where")) == []  # an own role with no kind: no state answer
    assert lex.role_kinds(lex.wh_roles(["quill.01"], "where")) == ["Place"]
    assert lex.role_kinds(lex.wh_roles(["brask.01", "brask.02", "quill.01"], "where")) == ["Organisation", "Place"]
    assert lex.role_kinds([]) == [] and lex.role_kinds(lex.wh_roles(["quill.01"], "when")) == []
    ops = LexiconOps(lex)
    assert asyncio.run(ops.run("role-kinds", {"roles": [["brask.01", "arg2"]]})) == ["Organisation"]


def test_shipped_map_kinds():
    """The shipped map: the employer (work.01 arg2, a beneficiary) is an Organisation; live.01 has no where role of
    its own, so the shared argm-loc gives Place."""
    lex = Lexicon([], QuestionMap.load())
    assert lex.role_kinds(lex.wh_roles(["work.01", "work.09"], "where")) == ["Organisation"]
    assert lex.role_kinds(lex.wh_roles(["live.01"], "where")) == ["Place"]
    assert lex.role_kinds(lex.wh_roles(["reside.01"], "where")) == ["Place"]


# -- the keywords -----------------------------------------------------------------------------------------------------


def _turn(text: str) -> Turn:
    return Turn(text=text, speaker="Ana", addressee="Milo", participants=("Ana", "Milo"), chat_id="c1", message_id=None)


def test_either_takes_the_first_non_empty():
    graph = FakeGraph([KIRA, MILO], [], {})
    memory = StoreMemory(graph, "g1")
    norm = normalise("x")
    for first, second, expected in (("Zed Quill", "Milo", ["u5"]), ("Kira Dace", "Milo", ["u1"])):
        program = Program.from_meaning(
            [
                ["find-entity", "?a", first],
                ["find-entity", "?b", second],
                ["either", "?c", "?a", "?b"],
                ["answer", "?c"],
            ]
        )
        run = asyncio.run(execute(program, REGISTRY, memory, turn=_turn("x"), spans=norm.spans))
        assert run.done, run.problems
        assert [e["uuid"] for e in run.values["?c"]] == expected


def test_slot_of_kind_takes_a_kind_set():
    graph = FakeGraph([KIRA, BRASK, VELMORA], [], {})
    memory = StoreMemory(graph, "g1")
    lines = [SUMMARIES["u1"]]
    both = asyncio.run(memory.run("slot-of-kind", {"lines": lines, "kind": ["Organisation", "place"]}))
    assert [e["name"] for e in both] == ["Brask Labs", "Velmora"]
    sent = len(graph.sent)
    assert asyncio.run(memory.run("slot-of-kind", {"lines": lines, "kind": []})) == []
    assert len(graph.sent) == sent  # no kind expected: no query
    with pytest.raises(ValueError, match="kind"):
        asyncio.run(memory.run("slot-of-kind", {"lines": lines, "kind": ["Place", "Weekday"]}))


# -- the where root on the engine -------------------------------------------------------------------------------------


def _data_dir():
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("runtime-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


def _plan(engine, tmp_path, graph: FakeGraph, text: str):
    lexicon = Lexicon(["live.01", "work.01"], QuestionMap.load())
    runtime = Runtime("bonfire", engine, graph, lexicon=lexicon, store=memory_store())
    turn = Turn(
        text=text, speaker="Ana", addressee="Milo", participants=("Ana", "Milo"), chat_id="c1", message_id=None, at=NOW
    )
    return asyncio.run(runtime.plan(turn))


def _names(answer) -> list[str]:
    return [e["name"] for e in answer]


@pytest.mark.needs_fcg
def test_where_work_reads_the_organisation_from_the_state_lines(engine, tmp_path):
    graph = FakeGraph([KIRA, BRASK, VELMORA, ANA], [], SUMMARIES)
    plan = _plan(engine, tmp_path, graph, "where does Kira Dace work?")
    assert plan.status == "answered", plan.diagnostics
    assert plan.reading["root"] == "where-do-x-v" and plan.answer_shape == "where"
    assert plan.answer == [{"uuid": "u2", "name": "Brask Labs", "kinds": ["Organisation"]}]
    ops = [p[0] for p in plan.program]
    assert {"role-filler", "role-kinds", "state-lines", "slot-of-kind", "either"} <= set(ops)
    assert plan.diagnostics == []


@pytest.mark.needs_fcg
def test_where_live_reads_the_place_not_the_employer(engine, tmp_path):
    graph = FakeGraph([KIRA, BRASK, VELMORA, ANA], [], SUMMARIES)
    plan = _plan(engine, tmp_path, graph, "where does Kira Dace live?")
    assert plan.status == "answered", plan.diagnostics
    assert _names(plan.answer) == ["Velmora"]


@pytest.mark.needs_fcg
def test_where_live_with_no_place_in_the_lines_is_a_repair(engine, tmp_path):
    """A negative stays negative: Milo's line names his employer only; "where does Milo live?" never answers it."""
    graph = FakeGraph([MILO, ZORVEX, ANA], [], SUMMARIES)
    plan = _plan(engine, tmp_path, graph, "where does Milo live?")
    assert plan.status == "repair" and plan.answer is None
    assert plan.diagnostics[0] == "the program found nothing for the turn"


@pytest.mark.needs_fcg
def test_an_event_answer_wins_over_the_state_lines(engine, tmp_path):
    graph = FakeGraph(
        [KIRA, BRASK, VELMORA, ZORVEX, ANA],
        [_event("e1", "work.01", ("arg0", KIRA), ("arg2", ZORVEX))],
        SUMMARIES,
    )
    plan = _plan(engine, tmp_path, graph, "where does Kira Dace work?")
    assert plan.status == "answered", plan.diagnostics
    assert _names(plan.answer) == ["Zorvex"]  # the line's Brask Labs is not added


def test_only_the_where_root_changed():
    """The state path rides on where-do-x-v alone; both its paths' outputs are read (no dangling output)."""
    from memory_kernel.runtime.api import dangling
    from memory_kernel.runtime.base import base_specs

    by_id = {s.id: s for s in base_specs()}
    where = by_id["where-do-x-v"]
    assert json.dumps(where.meaning) == json.dumps(
        [
            ["frames-of", "?fs", "?v"],
            ["find-entity", "?xe", "?x"],
            ["events", "?evs", "?fs", "?xe", "arg0"],
            ["wh-role", "?wr", "?fs", "where"],
            ["role-filler", "?byev", "?evs", "?wr", "?xe"],
            ["role-kinds", "?k", "?wr"],
            ["state-lines", "?l", "?xe", "?fs"],
            ["slot-of-kind", "?bystate", "?l", "?k"],
            ["either", "?ans", "?byev", "?bystate"],
            ["answer", "?ans"],
        ]
    )
    program = Program.from_meaning([*where.meaning, ["lemma-span", "?v", 0, 1], ["name-span", "?x", 0, 1]])
    assert dangling(program, REGISTRY) == []
    assert not any(
        op[0] in ("either", "role-kinds") for s in base_specs() if s.id != "where-do-x-v" for op in s.meaning
    )
