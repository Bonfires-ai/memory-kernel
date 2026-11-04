"""The base construction set: questions read into programs from base.json and the question map alone (no mining)."""

import asyncio
import os
import statistics
import time
from collections.abc import Mapping
from typing import Any

import pytest

from memory_kernel.runtime.constructions import CONSUME, TOKEN, CompileError, CxnSpec, compile_spec
from memory_kernel.runtime.executor import VALUE_SPECS, Turn, execute
from memory_kernel.runtime.lexicon import LEXICON_SPECS
from memory_kernel.runtime.normalise import normalise
from memory_kernel.runtime.ops_memory import MEMORY_SPECS, StoreMemory
from memory_kernel.runtime.program import ANSWER_PREDICATE, Program, Registry, validate

PORT = 9714  # 9711 is the runtime default, 9728 the engine tests', 9713 the compiler's
REGISTRY = Registry({**VALUE_SPECS, **MEMORY_SPECS, **LEXICON_SPECS})

#: the invented fixture: (question, the root that reads it, answer shape). Swapped, recorded in the Task 6b report:
#: "how many reports did Kira Dace publish?" is the accepted do-support miss (spaCy tags "did" VERB ROOT and
#: "publish" NOUN after the two-token name), so the fixture asks it of Ana.
SHAPES = [
    ("who founded Zorvex?", "who-v-y", "who"),
    ("where does Kira Dace work?", "where-do-x-v", "where"),
    ("when did Kira Dace publish the report?", "when-do-x-v-y", "when"),
    ("how many reports did Ana publish?", "how-many-n-do-x-v", "how_many"),
    ("which talk covered Zorvex?", "which-n-v-y", "which"),
    ("did anyone mention Zorvex?", "polar-do-anyone-v-y", "polar"),
]


class FakeReader:
    """Canned rows keyed by a marker found in the Cypher; records every query."""

    def __init__(self, canned: Mapping[str, list[dict]]) -> None:
        self.canned = canned
        self.sent: list[tuple[str, dict]] = []

    async def rows(self, cypher: str, params: Mapping[str, Any]) -> list[dict]:
        self.sent.append((cypher, dict(params)))
        return next((rows for marker, rows in self.canned.items() if marker in cypher), [])


ZORVEX = {"uuid": "u2", "name": "Zorvex", "labels": ["Entity", "Organisation"]}
BRASK = {"uuid": "u3", "name": "Brask Labs", "labels": ["Entity", "Place"]}


def _base():
    from memory_kernel.runtime.base import base_specs

    return base_specs()


def _turn(text: str, speaker: str | None = "Ana") -> Turn:
    return Turn(
        text=text, speaker=speaker, addressee="Milo", participants=("Ana", "Milo"), chat_id="c1", message_id=None
    )


def test_base_specs_are_live_base_rows_of_known_keywords():
    """Every row parses strictly, is base and live, compiles, and names only keywords (validate cannot see a row no
    fixture reads); every root names its answer shape."""
    from memory_kernel.runtime.frames import is_frame_part, reply_of

    specs = _base()
    assert specs and len({s.id for s in specs}) == len(specs)
    known = set(REGISTRY.specs) | {ANSWER_PREDICATE}
    # 1.5.1: the editor's frame parts and the reply roots afford, they never run (no keyword, no answer); 1.5.2 "phrase"
    affording = [s for s in specs if is_frame_part(s) or reply_of(s) is not None]
    replies = {f"reply.{word}{mark}" for word in ("yes", "no") for mark in ("", ".period", ".exclaim")}
    editor = {"frame.editor.teach", "frame.editor.mean", "frame.editor.phrase"}
    assert {s.id for s in affording} == {*editor, *replies}
    for spec in specs:
        assert spec.kind == "base" and spec.status == "live", spec.id
        assert CxnSpec.from_json(spec.to_json()) == spec
        compile_spec(spec)
        if spec in affording:
            continue
        assert {p[0] for p in spec.meaning} <= known, spec.id
        if spec.unit == "root":
            assert spec.answer_shape, spec.id
            assert sum(p[0] == ANSWER_PREDICATE for p in spec.meaning) == 1, spec.id
    assert {s.answer_shape for s in specs if s.unit == "root"} >= {
        "who",
        "what",
        "where",
        "when",
        "how_many",
        "how_long",
        "which",
        "polar",
    }


def test_slot_of_kind():
    """One keyword reads a slot of any store kind; the kind is a constant of the construction's meaning (the store's
    label, matched whatever case the engine hands back); the two old keywords are gone."""
    assert "slot-of-kind" in MEMORY_SPECS
    assert not {"slot-organisation", "slot-place"} & set(MEMORY_SPECS)
    org = FakeReader({"$kind IN labels": [ZORVEX]})
    out = asyncio.run(
        StoreMemory(org, "g1").run(
            "slot-of-kind", {"lines": ["[Ana] Kira Dace works at Zorvex."], "kind": "Organisation"}
        )
    )
    assert out == [{"uuid": "u2", "name": "Zorvex", "kinds": ["Organisation"]}]
    assert org.sent[0][1]["kind"] == "Organisation"
    place = FakeReader({"$kind IN labels": [BRASK]})
    asyncio.run(
        StoreMemory(place, "g1").run("slot-of-kind", {"lines": ["[Ana] Milo visited Brask Labs."], "kind": "place"})
    )
    assert place.sent[0][1]["kind"] == "Place"  # FCG hands meaning constants back lowercased
    with pytest.raises(ValueError, match="kind"):
        asyncio.run(StoreMemory(place, "g1").run("slot-of-kind", {"lines": ["x"], "kind": "Weekday"}))


def test_lemma_span():
    """lemma-span returns the marked verb's lemma from the normalised text (no second lemmatiser), over the lemma
    alone or the whole marked token; a span over two tokens, or over an unmarked one, is a problem."""
    norm = normalise("Kira Dace published the report")
    assert norm.spans.normalised == norm.text
    left = norm.text.index("⟪publish⟫")
    whole = (left, left + len("⟪publish⟫"))
    inner = (left + 1, left + 1 + len("publish"))
    for span in (whole, inner):
        program = Program.from_meaning([["lemma-span", "?v", *span], ["answer", "?v"]])
        done = asyncio.run(execute(program, REGISTRY, StoreMemory(None, "g1"), turn=_turn("x"), spans=norm.spans))
        assert done.done and done.values["?v"] == "publish", done.problems
    report = norm.text.index("report")
    for span in ((left, report + 6), (report, report + 6)):
        program = Program.from_meaning([["lemma-span", "?v", *span], ["answer", "?v"]])
        done = asyncio.run(execute(program, REGISTRY, StoreMemory(None, "g1"), turn=_turn("x"), spans=norm.spans))
        assert not done.done and "lemma-span" in done.problems[0]


def test_determiner_dropped_at_cut():
    """The NP cut drops leading determiner tokens by POS (spaCy DET), never all of a span's tokens; other leading
    tokens (a possessive pronoun, a name) stay as written."""
    norm = normalise("where does the team work?")
    left = norm.text.index("the team")
    assert norm.spans.cut(left, left + len("the team")) == "team"
    assert norm.spans.cut(left, left + len("the")) == "the"  # a span of determiners only keeps them
    every = normalise("what did every team publish?")
    left = every.text.index("every team")
    assert every.spans.cut(left, left + len("every team")) == "team"
    mine = normalise("where is my report?")
    left = mine.text.index("my report")
    assert mine.spans.cut(left, left + len("my report")) == "my report"  # "my" is a PRON, not a DET


def test_compiler_verb_slot_and_token_primitive():
    """A verb slot anchors on the verb marks the compiler writes (authors never write them), so it may sit next to an
    np slot; TOKEN (one non-space run) is a filler-only primitive; a part never has an edge slot."""
    from memory_kernel.runtime.constructions import SENTINEL_END, SENTINEL_START

    root = CxnSpec.from_json(
        {
            "id": "when-svo",
            "kind": "base",
            "unit": "root",
            "form": [
                "When did",
                {"slot": "x", "fills": "np"},
                {"slot": "v", "fills": "verb"},
                {"slot": "y", "fills": "np"},
                "?",
            ],
            "meaning": [
                ["frames-of", "?fs", "?v"],
                ["find-entity", "?a", "?x"],
                ["find-entity", "?b", "?y"],
                ["answer", "?fs"],
            ],
            "answer_shape": "when",
            "status": "live",
        }
    )
    form = compile_spec(root).cxn.conditional_pole[0][2]["#form"]
    assert [s[1] for s in form] == [f'"{SENTINEL_START} when do "', '" ⟪"', '"⟫ "', f'" [?] {SENTINEL_END}"']
    with pytest.raises(CompileError, match="filler"):
        compile_spec(CxnSpec(id="t", kind="base", unit="root", form=["who", TOKEN], meaning=[["answer", "?x"]]))
    with pytest.raises(CompileError, match="edge"):
        compile_spec(
            CxnSpec(
                id="p",
                kind="base",
                unit="part",
                form=[{"slot": "x", "fills": "np"}, "works"],
                meaning=[["find-entity", "?e", "?x"]],
            )
        )
    with pytest.raises(CompileError, match="verb"):  # a verb slot beside a literal verb: the stand-in context lies
        compile_spec(
            CxnSpec(
                id="vv",
                kind="base",
                unit="root",
                form=["who", {"slot": "v", "fills": "verb"}, "published?"],
                meaning=[["frames-of", "?fs", "?v"], ["answer", "?fs"]],
            )
        )
    verb = next(s for s in _base() if s.unit == "filler" and s.fills == "verb")
    assert verb.form == [TOKEN] and verb.meaning == [["lemma-span", "?x", "?l", "?r"]]


def test_composite_refuses_conflicting_specs():
    from memory_kernel.runtime.lexicon import CompositeMemory, LexiconOps
    from memory_kernel.runtime.program import OpSpec

    lex = LexiconOps(None)  # type: ignore[arg-type]
    CompositeMemory([(MEMORY_SPECS, lex), (LEXICON_SPECS, lex), (dict(LEXICON_SPECS), lex)])
    with pytest.raises(ValueError, match="count"):
        CompositeMemory([(MEMORY_SPECS, lex), ({"count": OpSpec("count", ("set", "by"), ("n",), "memory")}, lex)])


# -- engine ----------------------------------------------------------------------------------------------------------


def _data_dir():
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """One engine for the module, loaded with the whole base set, stopped by pid."""
    from memory_kernel.runtime.base import load_base
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("runtime-fcg"))
    engine.start()
    try:
        # every spec's constructions: one each, but the editor part "means" reads as two (its marked "means" too), and
        # so does "phrase" (a noun alone: the plain token, and the marked verb as written)
        assert load_base(engine) == sum(len(compile_spec(spec).cxns) for spec in _base()) == len(_base()) + 2
        yield engine
    finally:
        engine.stop()


def _read(engine, text: str):
    norm = normalise(text)
    return norm, engine.comprehend(norm.text)


def _roots() -> set[str]:
    return {s.id for s in _base() if s.unit == "root"}


def _spans(norm, reading) -> dict[str, Any]:
    """Slot strings: name-span cuts to the surface ("names", in meaning order), lemma-span reads the lemma."""
    out: dict[str, Any] = {}
    for p in reading.meaning:
        if p[0] == "name-span":
            out.setdefault("names", []).append(norm.spans.cut(p[2], p[3]))
        elif p[0] == "lemma-span":
            out["lemma"] = norm.text[p[2] : p[3]]
    return out


@pytest.mark.needs_fcg
def test_base_shapes_read_without_mining(engine):
    by_id = {s.id: s for s in _base()}
    for text, root, shape in SHAPES:
        norm, reading = _read(engine, text)
        assert reading is not None and reading.complete, text
        assert reading.applied[0] == root, (text, reading.applied)
        program = Program.from_meaning(reading.meaning)
        assert validate(program, REGISTRY) == [], (text, validate(program, REGISTRY))
        assert program.answer is not None
        assert by_id[root].answer_shape == shape
    norm, reading = _read(engine, "where does Kira Dace work?")
    assert _spans(norm, reading) == {"names": ["Kira Dace"], "lemma": "work"}
    norm, reading = _read(engine, "who founded Zorvex?")
    assert _spans(norm, reading) == {"names": ["Zorvex"], "lemma": "found"}
    norm, reading = _read(engine, "which talk covered Zorvex?")  # "talk" is marked a verb; the N slot swallows it
    assert _spans(norm, reading) == {"names": ["talk", "Zorvex"], "lemma": "cover"}


@pytest.mark.needs_fcg
def test_where_program_runs_end_to_end(engine):
    """The data flow validate cannot see: lemma -> frames -> wh-role pairs -> events -> role-filler, on a fake graph."""
    from memory_kernel.runtime.lexicon import CompositeMemory, Lexicon, LexiconOps, QuestionMap

    kira = {"uuid": "u1", "name": "Kira Dace", "labels": ["Entity", "Person"]}
    reader = FakeReader({"n.name = $name": [kira], "$pairs": [dict(ZORVEX, event="e1")], "$groups": [{"uuid": "e1"}]})
    lexicon = LexiconOps(Lexicon(["work.01"], QuestionMap.load()))
    memory = CompositeMemory([(MEMORY_SPECS, StoreMemory(reader, "g1")), (LEXICON_SPECS, lexicon)])
    norm, reading = _read(engine, "where does Kira Dace work?")
    program = Program.from_meaning(reading.meaning)
    run = asyncio.run(execute(program, REGISTRY, memory, turn=_turn(norm.spans.original), spans=norm.spans))
    assert run.done, run.problems
    assert run.values[program.answer.name] == [{"uuid": "u2", "name": "Zorvex", "kinds": ["Organisation"]}]
    events = next(p for c, p in reader.sent if "$frames" in c)
    assert events["frames"] == ["work.01"] and events["groups"] == [{"uuids": ["u1"], "roles": ["arg0"]}]
    pairs = next(p for c, p in reader.sent if "$pairs" in c)["pairs"]
    assert ["work.01", "arg2"] in pairs and ["work.01", "argm-loc"] in pairs


@pytest.mark.needs_fcg
def test_verb_filler_takes_one_mark(engine):
    """With two marked verbs, the verb filler (TOKEN) spans exactly one; a CONSUME verb filler would span both."""
    from memory_kernel.runtime.constructions import VERB_CLOSE, VERB_OPEN

    imperative = CxnSpec(
        id="zz-v-np",
        kind="hand",
        unit="root",
        form=[{"slot": "v", "fills": "verb"}, {"slot": "y", "fills": "np"}],
        meaning=[["frames-of", "?fs", "?v"], ["find-entity", "?e", "?y"], ["answer", "?e"]],
        status="live",
    )
    text = "⟦ ⟪give⟫ the report ⟪say⟫ it ⟧"
    greedy = CxnSpec(
        id="zz-greedy-verb",
        kind="hand",
        unit="filler",
        fills="verb",
        form=[CONSUME],
        meaning=[["lemma-span", "?x", "?l", "?r"]],
        status="live",
    )
    from memory_kernel.runtime.base import load_base

    try:
        engine.add(compile_spec(imperative).cxn)
        reading = engine.comprehend(text)
        assert reading is not None and reading.complete
        [(left, right)] = [tuple(p[2:]) for p in reading.meaning if p[0] == "lemma-span"]
        assert text[left:right] == "give" and " " not in text[left:right]
        assert VERB_OPEN not in text[left:right] and VERB_CLOSE not in text[left:right]

        engine.add(compile_spec(greedy).cxn)  # loads last: the engine prefers it
        reading = engine.comprehend(text)
        [(left, right)] = [tuple(p[2:]) for p in reading.meaning if p[0] == "lemma-span"]
        assert text[left:right] == "give⟫ the report ⟪say"  # the pin above is not vacuous
    finally:
        load_base(engine)  # back to the base set alone, whatever failed


@pytest.mark.needs_fcg
def test_determiner_dropped_at_cut_on_the_engine(engine):
    norm, reading = _read(engine, "where does the team work?")
    assert reading is not None and reading.complete
    assert reading.applied[0] == "where-do-x-v" and "np-the" in reading.applied
    assert _spans(norm, reading)["names"] == ["team"]
    assert any(p[0] == "definite" for p in reading.meaning)


@pytest.mark.needs_fcg
def test_do_support_reads_the_main_verb(engine):
    """Ruling 6: the turn parser (trf) tags "publish" a VERB root after a two-token name, so the turn reads the
    do-support root and the main verb, never the auxiliary as a subject root."""
    norm, reading = _read(engine, "when do Kira Dace publish?")
    assert norm.text == "⟦ when do kira dace ⟪publish⟫ ? ⟧"
    assert reading is not None and reading.complete and reading.applied[0] == "when-do-x-v"
    assert _spans(norm, reading) == {"names": ["Kira Dace"], "lemma": "publish"}
    norm, reading = _read(engine, "how many reports did Kira Dace publish?")
    assert norm.text == "⟦ how many reports do kira dace ⟪publish⟫ ? ⟧"
    assert reading is not None and reading.complete and reading.applied[0] == "how-many-n-do-x-v"


@pytest.mark.needs_fcg
def test_deixis_and_time_fillers(engine):
    cases = [
        ("where does my team work?", "where-do-x-v", "np-my"),
        ("what did Milo publish yesterday?", "what-do-x-v-t", "time-yesterday"),
        ("what did Ana publish before the talk?", "what-do-x-v-t", "time-before"),
    ]
    for text, root, filler in cases:
        norm, reading = _read(engine, text)
        assert reading is not None and reading.complete, text
        assert reading.applied[0] == root and filler in reading.applied, (text, reading.applied)
        program = Program.from_meaning(reading.meaning)
        assert validate(program, REGISTRY) == [], text
    norm, reading = _read(engine, "where does my team work?")
    assert ["deixis", next(p[1] for p in reading.meaning if p[0] == "deixis"), "speaker"] in reading.meaning
    assert _spans(norm, reading)["names"] == ["team"]
    norm, reading = _read(engine, "what did Ana publish before the talk?")
    assert sorted(_spans(norm, reading)["names"]) == ["Ana", "talk"]
    # the speaker is a turn role: a turn with no speaker refuses the program before it runs
    program = Program.from_meaning(_read(engine, "where does my team work?")[1].meaning)
    run = asyncio.run(execute(program, REGISTRY, StoreMemory(None, "g1"), turn=_turn("x", None), spans=norm.spans))
    assert not run.done and any("speaker" in p for p in run.problems)


@pytest.mark.needs_fcg
def test_reference_and_everyone_fillers_read(engine):
    """ "them" reads as a turn-window referent marked plural (never a name), "everyone" as the turn's participants:
    both base rows, read here through a clause root added for the test ("V X to Y")."""
    from memory_kernel.runtime.base import load_base

    clause = CxnSpec(
        id="zz-v-x-to-y",
        kind="hand",
        unit="root",
        form=[{"slot": "v", "fills": "verb"}, {"slot": "x", "fills": "np"}, "to", {"slot": "y", "fills": "np"}],
        meaning=[
            ["frames-of", "?fs", "?v"],
            ["find-entity", "?e", "?x"],
            ["find-entity", "?r", "?y"],
            ["answer", "?e"],
        ],
        status="live",
    )
    try:
        engine.add(compile_spec(clause).cxn)
        norm, reading = _read(engine, "give them to Ana")
        assert reading is not None and reading.complete and "np-them" in reading.applied, reading
        [window] = [p for p in reading.meaning if p[0] == "window"]
        [plural] = [p for p in reading.meaning if p[0] == "plural"]
        assert window[1] == plural[1]
        assert _spans(norm, reading)["names"] == ["Ana"]  # "them" is never cut as a name
        norm, reading = _read(engine, "give tea to everyone")
        assert reading is not None and reading.complete and "np-everyone" in reading.applied, reading
        assert any(p[0] == "everyone" for p in reading.meaning)
        assert _spans(norm, reading)["names"] == ["tea"]
    finally:
        load_base(engine)  # back to the base set alone, whatever failed


@pytest.mark.needs_fcg
def test_base_comprehension_latency(engine, capsys):
    """Report-only: comprehension ms over the fixture questions with the whole base set loaded (flag p95 > 50 ms)."""
    times: list[float] = []
    for _ in range(5):
        for text, _, _ in SHAPES:
            norm = normalise(text)
            started = time.perf_counter()
            engine.comprehend(norm.text)
            times.append((time.perf_counter() - started) * 1000)
    times.sort()
    p95 = times[int(round(0.95 * (len(times) - 1)))]
    with capsys.disabled():
        print(
            f"\nbase set: {len(_base())} constructions; comprehend p50 {statistics.median(times):.1f} ms p95 {p95:.1f} ms"
        )


# -- fix round 1 -------------------------------------------------------------------------------------------------------

#: the turn's time for the time-window pins (Sunday 2026-10-18, 15:30 UTC)
NOW = "2026-10-18T15:30:00+00:00"


def _dangling(spec: CxnSpec) -> list[str]:
    """Op outputs nothing in the spec reads and that are not its answer (a filler's ?x is read by its root)."""
    written, read, answer = set(), set(), None
    for p in spec.meaning:
        if p[0] == ANSWER_PREDICATE:
            answer = p[1]
            continue
        n = len(REGISTRY.specs[p[0]].outputs)
        written |= {a for a in p[1 : 1 + n] if a.startswith("?")}
        read |= {a for a in p[1 + n :] if a.startswith("?")}
    return sorted(written - read - {answer, "?x"})


def test_dangling_outputs_are_only_the_parked_constraints():
    """What no keyword applies yet is exactly the noun restriction, possession, the before/after anchor and a second
    participant (no keyword builds the participant list): Task 8's executable() refuses these programs."""
    from memory_kernel.runtime.frames import is_frame_part, reply_of

    programs = [s for s in _base() if not is_frame_part(s) and reply_of(s) is None]  # frames and replies never run
    dangling = {s.id: _dangling(s) for s in programs if _dangling(s)}
    assert dangling == {
        "when-do-x-v-y": ["?other"],
        "when-do-x-first-v-y": ["?other"],
        "when-do-x-last-v-y": ["?other"],
        "polar-do-x-v-y": ["?other"],
        "which-n-v-y": ["?noun"],
        "which-n-do-x-v": ["?noun"],
        "how-many-n-v-y": ["?noun"],
        "how-many-n-do-x-v": ["?noun"],
        "np-possessive": ["?owner"],
        "np-my": ["?owner"],
        "np-your": ["?owner"],
        "time-before": ["?anchor"],
        "time-after": ["?anchor"],
    }
    assert "np-we" not in {s.id for s in _base()}  # sets are deferred with coordination
    assert "polar-be-there-x" not in {s.id for s in _base()}  # name resolution cannot say a real "no"


def test_auxiliaries_are_data():
    from memory_kernel.runtime.base import base_auxiliaries

    assert base_auxiliaries() == frozenset({"be", "do", "have"})


def test_within_keeps_the_events_of_the_interval():
    """within(events ← events, when): the events whose date falls in the interval ``when`` names, relative to the
    turn's time; undated events are dropped; no turn time, or a constant that is no interval, is a problem."""
    events = [
        {"uuid": "e1", "happened_at": "2026-10-17T09:00:00+00:00"},  # yesterday
        {"uuid": "e2", "happened_at": "2026-10-18T08:00:00+00:00"},  # today
        {"uuid": "e3", "happened_at": "2026-10-16"},  # two days ago, a date only
        {"uuid": "e4", "told_at": "2026-10-17"},  # yesterday, told
        {"uuid": "e5"},  # undated
        {"uuid": "e6", "happened_at": "2026-10-12T23:00:00+00:00"},  # Monday of this week
        {"uuid": "e7", "happened_at": "2026-10-11T23:00:00+00:00"},  # Sunday of last week
    ]

    class Items:
        async def run(self, op, args):
            assert op == "events"
            return events

    def run(when: str, at: str | None = NOW):
        meaning = [["events", "?evs", "f.01", "?p", "arg0"], ["within", "?in", "?evs", when], ["answer", "?in"]]
        meaning.append(["const", "?p", "u1"])
        turn = Turn("x", "Ana", None, (), "c1", None, at=at)
        return asyncio.run(
            execute(Program.from_meaning(meaning), REGISTRY, Items(), turn=turn, spans=normalise("x").spans)
        )

    done = run("d-1")  # FCG hands constants back lowercased
    assert done.done, done.problems
    assert [e["uuid"] for e in done.values["?in"]] == ["e1", "e4"]
    week = run("W-1")
    assert [e["uuid"] for e in week.values["?in"]] == ["e7"]  # ISO weeks start on Monday: 10-05 .. 10-11
    assert not run("d-1", at=None).done and "time" in run("d-1", at=None).problems[0]
    refused = run("before")
    assert not refused.done and "within" in refused.problems[0]


def test_role_filler_except_and_events_all_of():
    """role-filler never returns the given participants; events given a list of participants reads the events
    with ALL of them (an item that is itself a list is one participant's candidates)."""
    kira = {"uuid": "u1", "name": "Kira Dace", "labels": ["Entity", "Person"]}
    milo = {"uuid": "u8", "name": "Milo", "labels": ["Entity", "Person"]}
    reader = FakeReader({"$pairs": [dict(kira, event="e1"), dict(milo, event="e1")], "$groups": [{"uuid": "e1"}]})
    memory = StoreMemory(reader, "g1")
    out = asyncio.run(memory.run("role-filler", {"event": [{"uuid": "e1"}], "role": "arg1", "except": [kira]}))
    assert [e["name"] for e in out] == ["Milo"]
    asyncio.run(memory.run("events", {"frame": "meet.01", "participant": [[kira], [milo]], "role": ["arg0", "arg1"]}))
    cypher, params = reader.sent[-1]
    assert params["groups"] == [{"uuids": ["u1"], "roles": ["arg0"]}, {"uuids": ["u8"], "roles": ["arg1"]}]
    assert "all(" in cypher.lower()
    asyncio.run(memory.run("events", {"frame": "meet.01", "participant": [kira], "role": "arg0"}))
    assert reader.sent[-1][1]["groups"] == [{"uuids": ["u1"], "roles": ["arg0"]}]
    asyncio.run(memory.run("events", {"frame": "meet.01", "participant": None, "role": None}))
    assert reader.sent[-1][1]["groups"] == []


def test_determiner_kept_on_a_proper_name_and_quantifier_marked():
    """A capitalised leading DET before nothing but PROPN tokens is part of the name; a dropped determiner that is
    neither an article nor a demonstrative (no PronType=Art|Dem: every, all, each, no, ...) marks the cut quantified."""
    norm = normalise("where does The Brask Group work?")
    left = norm.text.index("the brask group")
    assert norm.spans.cut(left, left + len("the brask group")) == "The Brask Group"
    lower = normalise("where does the Brask Group work?")
    left = lower.text.index("the brask group")
    assert lower.spans.cut(left, left + len("the brask group")) == "Brask Group"
    team = normalise("where does The team work?")
    left = team.text.index("the team")
    assert team.spans.np(left, left + len("the team")) == ("team", False)
    every = normalise("what did every team publish?")
    left = every.text.index("every team")
    assert every.spans.np(left, left + len("every team")) == ("team", True)
    assert every.spans.cut(left, left + len("every team")) == "team"


@pytest.mark.needs_fcg
def test_reading_guard_refuses_auxiliaries_in_subject_roots(engine, monkeypatch):
    from memory_kernel.graph import g1_nlp
    from memory_kernel.runtime.base import reading_guard

    # the guard exists for the small model's aux-root mis-tag: run it where it can still fire (the sm fallback)
    monkeypatch.setenv("MK_TURN_SPACY_MODEL", "en_core_web_sm")
    g1_nlp.reset_pipelines()
    try:
        _guard_cases(engine, reading_guard)
    finally:
        g1_nlp.reset_pipelines()


def _guard_cases(engine, reading_guard) -> None:
    for text in ("how many reports did Kira Dace publish?", "who did Kira Dace meet yesterday?"):
        norm, reading = _read(engine, text)
        assert reading is not None and reading.complete  # the mis-tag reads, and the guard refuses it
        reason = reading_guard(reading, norm.spans)
        assert reason is not None and "do" in reason, text
    norm, reading = _read(engine, "what did Kira Dace do?")  # an object root: the main verb "do" is fine
    assert reading is not None and reading.complete and reading.applied[0] == "what-do-x-v"
    assert reading_guard(reading, norm.spans) is None
    for text, _, _ in SHAPES:
        norm, reading = _read(engine, text)
        assert reading_guard(reading, norm.spans) is None, text


@pytest.mark.needs_fcg
def test_who_met_x_never_answers_x(engine):
    from memory_kernel.runtime.lexicon import CompositeMemory, Lexicon, LexiconOps, QuestionMap

    kira = {"uuid": "u1", "name": "Kira Dace", "labels": ["Entity", "Person"]}
    milo = {"uuid": "u8", "name": "Milo", "labels": ["Entity", "Person"]}
    reader = FakeReader(
        {
            "n.name = $name": [kira],
            "$pairs": [dict(kira, event="e1"), dict(milo, event="e1")],
            "$groups": [{"uuid": "e1"}],
        }
    )
    memory = CompositeMemory(
        [
            (MEMORY_SPECS, StoreMemory(reader, "g1")),
            (LEXICON_SPECS, LexiconOps(Lexicon(["meet.01"], QuestionMap.load()))),
        ]
    )
    norm, reading = _read(engine, "who met Kira Dace?")
    assert reading.applied[0] == "who-v-y"
    program = Program.from_meaning(reading.meaning)
    run = asyncio.run(execute(program, REGISTRY, memory, turn=_turn(norm.spans.original), spans=norm.spans))
    assert run.done, run.problems
    assert [e["name"] for e in run.values[program.answer.name]] == ["Milo"]


@pytest.mark.needs_fcg
def test_yesterday_applies_within(engine):
    """ "what did Milo publish yesterday?": the time root applies within, so only yesterday's events are read."""
    from memory_kernel.runtime.lexicon import CompositeMemory, Lexicon, LexiconOps, QuestionMap

    milo = {"uuid": "u8", "name": "Milo", "labels": ["Entity", "Person"]}
    report = {"uuid": "u7", "name": "Zorvex report", "labels": ["Entity", "Artifact"]}
    events = [{"uuid": "e1", "happened_at": "2026-10-17T10:00:00+00:00"}, {"uuid": "e2", "happened_at": "2026-10-01"}]
    reader = FakeReader({"n.name = $name": [milo], "$pairs": [dict(report, event="e1")], "$groups": events})
    memory = CompositeMemory(
        [
            (MEMORY_SPECS, StoreMemory(reader, "g1")),
            (LEXICON_SPECS, LexiconOps(Lexicon(["publish.01"], QuestionMap.load()))),
        ]
    )
    norm, reading = _read(engine, "what did Milo publish yesterday?")
    program = Program.from_meaning(reading.meaning)
    assert validate(program, REGISTRY) == [] and "within" in {op.name for op in program.ops}
    turn = Turn(norm.spans.original, "Ana", None, (), "c1", None, at=NOW)
    run = asyncio.run(execute(program, REGISTRY, memory, turn=turn, spans=norm.spans))
    assert run.done, run.problems
    fillers = next(p for c, p in reader.sent if "$pairs" in c)
    assert fillers["uuids"] == ["e1"]  # e2 (two weeks ago) never reached role-filler


# -- fix round 2 -------------------------------------------------------------------------------------------------------


class FakeGraph:
    """A tiny in-memory graph answering the StoreMemory queries by their parameters (find-entity, events,
    role-filler), so role semantics are tested, not canned: events ``[{uuid, roleset, edges: [(role, entity)]}]``."""

    def __init__(self, entities: list[dict], events: list[dict]) -> None:
        self.entities = {e["uuid"]: e for e in entities}
        self.events = events
        self.sent: list[tuple[str, dict]] = []

    async def rows(self, cypher: str, params: Mapping[str, Any]) -> list[dict]:
        self.sent.append((cypher, dict(params)))
        if "n.name = $name" in cypher:
            return [dict(e) for e in self.entities.values() if e["name"] == params["name"]]
        if "$groups" in cypher:
            out = []
            for ev in self.events:
                if ev["roleset"] not in params["frames"]:
                    continue
                if all(
                    any(x in g["uuids"] and (g["roles"] is None or r in g["roles"]) for r, x in ev["edges"])
                    for g in params["groups"]
                ):
                    out.append({"uuid": ev["uuid"], "name": ev["uuid"]})
            return out
        if "$pairs" in cypher:
            out = []
            for ev in self.events:
                if ev["uuid"] not in params["uuids"]:
                    continue
                for r, x in ev["edges"]:
                    if any(r == role and frame in (None, ev["roleset"]) for frame, role in params["pairs"]):
                        out.append({**self.entities[x], "event": ev["uuid"]})
            return out
        return []


KIRA = {"uuid": "u1", "name": "Kira Dace", "labels": ["Entity", "Person"]}
MILO = {"uuid": "u8", "name": "Milo", "labels": ["Entity", "Person"]}
ANA = {"uuid": "u9", "name": "Ana", "labels": ["Entity", "Person"]}
NEWS = {"uuid": "u6", "name": "the merger", "labels": ["Entity", "Artifact"]}


def _run_on(engine, text: str, graph: FakeGraph, rolesets: list[str]):
    from memory_kernel.runtime.lexicon import CompositeMemory, Lexicon, LexiconOps, QuestionMap

    memory = CompositeMemory(
        [(MEMORY_SPECS, StoreMemory(graph, "g1")), (LEXICON_SPECS, LexiconOps(Lexicon(rolesets, QuestionMap.load())))]
    )
    norm, reading = _read(engine, text)
    assert reading is not None and reading.complete, text
    program = Program.from_meaning(reading.meaning)
    run = asyncio.run(execute(program, REGISTRY, memory, turn=_turn(text), spans=norm.spans))
    assert run.done, (text, run.problems)
    return reading, run.values[program.answer.name]


def test_events_role_constants_are_a_closed_set():
    """events' role is one of arg0..arg5, or obj (any core role but arg0); anything else is refused."""
    reader = FakeReader({})
    memory = StoreMemory(reader, "g1")
    asyncio.run(memory.run("events", {"frame": "tell.01", "participant": [MILO], "role": "obj"}))
    assert reader.sent[-1][1]["groups"] == [{"uuids": ["u8"], "roles": ["arg1", "arg2", "arg3", "arg4", "arg5"]}]
    asyncio.run(memory.run("events", {"frame": "tell.01", "participant": [MILO], "role": "OBJ"}))  # FCG case
    assert reader.sent[-1][1]["groups"][0]["roles"][0] == "arg1"
    with pytest.raises(ValueError, match="role"):
        asyncio.run(memory.run("events", {"frame": "tell.01", "participant": [MILO], "role": "argm-tmp"}))


def test_within_input_is_named_interval_and_spans_expose_np():
    from memory_kernel.runtime.base import base_specs
    from memory_kernel.runtime.executor import SpanText

    assert VALUE_SPECS["within"].inputs == ("events", "interval")
    assert "np" in SpanText.__dict__ and "cut" in SpanText.__dict__
    assert base_specs() is not base_specs() and base_specs()[0] is base_specs()[0]  # parsed once, a fresh list


@pytest.mark.needs_fcg
def test_subject_roots_answer_the_subject(engine):
    """ "who sold Zorvex?" with an arg2 buyer: only the seller (arg0), never the buyer nor Zorvex."""
    graph = FakeGraph(
        [KIRA, MILO, ZORVEX],
        [{"uuid": "e1", "roleset": "sell.01", "edges": [("arg0", "u1"), ("arg1", "u2"), ("arg2", "u8")]}],
    )
    reading, answer = _run_on(engine, "who sold Zorvex?", graph, ["sell.01"])
    assert reading.applied[0] == "who-v-y"
    assert [e["name"] for e in answer] == ["Kira Dace"]


@pytest.mark.needs_fcg
def test_obj_reads_any_non_subject_role(engine):
    """The given participant of a subject root is looked up in any core role but arg0 (tell.01's hearer is arg2)."""
    told = {"uuid": "e2", "roleset": "tell.01", "edges": [("arg0", "u9"), ("arg1", "u6"), ("arg2", "u8")]}
    taught = {"uuid": "e3", "roleset": "teach.01", "edges": [("arg0", "u8"), ("arg2", "u1")]}
    graph = FakeGraph([KIRA, MILO, ANA, NEWS], [told, taught])
    _, answer = _run_on(engine, "who told Milo?", graph, ["tell.01", "teach.01"])
    assert [e["name"] for e in answer] == ["Ana"]
    reading, evidence = _run_on(engine, "did anyone tell Milo?", graph, ["tell.01", "teach.01"])
    assert reading.applied[0] == "polar-do-anyone-v-y" and [e["uuid"] for e in evidence] == ["e2"]  # yes
    milo_told = {"uuid": "e4", "roleset": "tell.01", "edges": [("arg0", "u8"), ("arg2", "u1")]}
    _, evidence = _run_on(engine, "did anyone tell Milo?", FakeGraph([KIRA, MILO], [milo_told]), ["tell.01"])
    assert evidence == []  # Milo telling someone is not someone telling Milo: no
