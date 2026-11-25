"""The grammar file and the door that installs it (2.1, spec 1a and 1d): ``parse_grammar``, ``install`` and
``mk grammar install``. A refused file changes nothing: not the store, not the engine's grammar. The grammar is the
argument grammar's shapes (a statement root over one clause slot, two connectives, one operator) with ``statement``,
``relation`` and ``mark`` as tools; the connectives score above the generated clause filler (0.5) and the root below
it, or the clause filler wins every slot and nothing composes. Every sentence is made up.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import os
import re
import signal
import time
from typing import Any

import pytest

from memory_kernel import cli
from memory_kernel.runtime import engine as engine_module
from memory_kernel.runtime import graph_store
from memory_kernel.runtime.api import default_registry
from memory_kernel.runtime.constructions import CxnSpec
from memory_kernel.runtime.engine import RuntimeEngineError
from memory_kernel.runtime.frames import FRAME_PREFIX
from memory_kernel.runtime.grammar import (
    GRAMMAR,
    GRAMMAR_HASH,
    SCHEMA,
    Example,
    GrammarError,
    GrammarFile,
    install,
    parse_grammar,
)
from memory_kernel.runtime.normalise import normalise
from memory_kernel.runtime.program import ToolSpec
from memory_kernel.runtime.store import CLAUSE_FILLER, base_ids, load_store
from memory_kernel.runtime.teach import TeachRequest, teach
from unit.runtime.memory_writer import memory_store

PORT = 9730  # 9713-9729 are the other engine test modules'
NAME = "argument"
ONE = "it works because it is open"
TWO = "it works if it is open"
OWN = "hello there"  # read only by the bonfire's own row
TOOLS = default_registry(
    [
        ToolSpec.simple("statement", ("text",), False),
        ToolSpec.simple("relation", ("whole", "head", "text"), False),
        ToolSpec.simple("mark", ("whole", "scope", "rest"), False),
    ]
)


def _slot(name: str) -> dict:
    return {"slot": name, "fills": "clause"}


def _connective(cxn_id: str, word: str) -> dict:
    return {
        "id": cxn_id,
        "unit": "filler",
        "fills": "clause",
        "score": 0.7,
        "form": [_slot("h"), word, _slot("d")],
        "meaning": [["clause-span", "?x", "?l", "?r"], ["relation", "?rel", "?x", "?h", "?d"]],
    }


ROWS: tuple[dict[str, Any], ...] = (
    {
        "id": "root.s",
        "unit": "root",
        "score": 0.2,
        "form": [_slot("s")],
        "meaning": [["procedure", "?p", "say"], ["seq", "?p", "1", "?st"], ["statement", "?st", "?s"]],
    },
    _connective("cx.because", "because"),
    _connective("cx.if", "if"),
    {
        "id": "op.not",
        "unit": "filler",
        "fills": "clause",
        "score": 0.65,
        "form": [_slot("a"), "not", _slot("b")],
        "meaning": [["clause-span", "?x", "?l", "?r"], ["mark", "?m", "?x", "?a", "?b"]],
    },
)


def _relation(text: str, head: str, rest: str) -> tuple[str, dict[str, str]]:
    return ("relation", {"whole": text, "head": head, "text": rest})


EXAMPLES = (
    Example(ONE, (("statement", {"text": ONE}), _relation(ONE, "it works", "it is open"))),
    Example(TWO, (("statement", {"text": TWO}), _relation(TWO, "it works", "it is open"))),
)
FILE = GrammarFile(name=NAME, sha256="f" * 64, rows=ROWS, examples=EXAMPLES)

#: a catalogue card the door is handed (``cards``): live beside the store's rows, never stored
CARD = CxnSpec(
    id="card.statement.0",
    kind="hand",
    unit="root",
    form=["jot", {"slot": "t", "fills": "np"}],
    meaning=[["procedure", "?p", "jot"], ["seq", "?p", "1", "?st"], ["statement", "?st", "?t"]],
    status="live",
)

#: the bonfire's own row: a root of literals, read by nothing in the file
OWN_ROW = CxnSpec(
    id="own.hello",
    kind="hand",
    unit="root",
    form=["hello there"],
    meaning=[
        ["procedure", "?p", "say"],
        ["seq", "?p", "1", "?st"],
        ["const", "?t", "hello"],
        ["statement", "?st", "?t"],
    ],
    status="live",
)


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
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("runtime-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


@pytest.fixture
def store():
    return memory_store(OWN_ROW)


@pytest.fixture
def registry():
    return TOOLS


def reads(engine: Any, text: str) -> tuple[bool, tuple[str, ...]]:
    """What the engine now holds does with ``text``, with no reload: (complete, applied)."""
    reading = engine.comprehend(normalise(text).text)
    return (False, ()) if reading is None else (reading.complete, tuple(reading.applied))


def rows_of(store: Any) -> list[CxnSpec]:
    return copy.deepcopy(store.all())


def with_file(**changes: Any) -> GrammarFile:
    return dataclasses.replace(FILE, **changes)


def edited(index: int, **fields: Any) -> tuple[dict[str, Any], ...]:
    rows = [copy.deepcopy(r) for r in ROWS]
    rows[index].update(fields)
    return tuple(rows)


def broken(file: GrammarFile, breakage: str, store: Any, *, and_also: str | None = None) -> GrammarFile:
    """``file`` with one breakage (on a row of its own), and ``and_also`` on the operator's row; a breakage that lives in
    the store changes ``store``."""
    rows = [copy.deepcopy(r) for r in file.rows]
    examples = list(file.examples)

    def tool(row: dict, new: str) -> None:
        row["meaning"] = [[new, *p[1:]] if p[0] in ("relation", "mark") else p for p in row["meaning"]]

    def one(what: str, at: int) -> None:
        row = rows[at]
        if what == "a row that does not compile":
            row["form"] = []
        elif what == "an unknown tool":
            tool(row, "nosuchtool")
        elif what == "a tool with one argument too many":
            row["meaning"] = [[*p, "?extra"] if p[0] in ("relation", "mark") else p for p in row["meaning"]]
        elif what == "a filler slot named r":
            row["form"] = [_slot("h"), "if", _slot("r")]
            row["meaning"] = [[s.replace("?d", "?r") for s in p] for p in row["meaning"]]
        elif what == "two rows with one id":
            rows.append(copy.deepcopy(rows[0]))
        elif what == "a base row's id":
            row["id"] = sorted(base_ids())[0]
        elif what == "the generated clause filler's id":
            row["id"] = CLAUSE_FILLER.id
        elif what == "a catalogue card's id":
            row["id"] = CARD.id
        elif what == "an id in the frame parts' namespace":
            row["id"] = f"{FRAME_PREFIX}.statement.because"
        elif what == "a part row stating as_written":
            rows[at] = {"id": row["id"], "unit": "part", "form": ["because"], "meaning": [["const", "?x", "y"]]}
            rows[at]["as_written"] = True
        elif what == "an id a taught row holds":
            store.put(
                CxnSpec(
                    id=row["id"],
                    kind="taught",
                    unit="part",
                    form=["taught"],
                    meaning=[["const", "?x", "y"]],
                    status="live",
                )
            )
        elif what == "an id another grammar's hand row holds":
            store.put(
                CxnSpec(
                    id=row["id"],
                    kind="hand",
                    unit="part",
                    form=["theirs"],
                    meaning=[["const", "?x", "y"]],
                    status="live",
                    provenance={GRAMMAR: "somebody-else"},
                )
            )
        elif what == "an id a mined row holds":
            store.put(  # a mined row that names this grammar is still not one this grammar installed
                CxnSpec(
                    id=row["id"],
                    kind="mined",
                    unit="part",
                    form=["mined"],
                    meaning=[["const", "?x", "y"]],
                    status="live",
                    provenance={GRAMMAR: NAME},
                )
            )
        elif what == "no examples":
            examples.clear()
        elif what == "an example whose first reading differs":
            examples[0] = Example(ONE, (("statement", {"text": "something else"}),))
        elif what == "a row stating kind":
            row["kind"] = "taught"
        else:
            raise AssertionError(what)

    one(breakage, 1)
    if and_also:
        one(and_also, 3)
    return dataclasses.replace(file, rows=tuple(rows), examples=tuple(examples))


# -- the file ---------------------------------------------------------------------------------------------------------


def _bytes(**changes: Any) -> bytes:
    doc = {
        "schema": SCHEMA,
        "name": NAME,
        "rows": list(ROWS),
        "examples": [{"text": e.text, "steps": [{"tool": t, "args": a} for t, a in e.steps]} for e in EXAMPLES],
    }
    doc.update(changes)
    return json.dumps(doc).encode()


def test_parse_reads_a_file_and_hashes_its_bytes():
    import hashlib

    raw = _bytes()
    file = parse_grammar(raw, "argument.json")
    assert (file.name, file.sha256) == (NAME, hashlib.sha256(raw).hexdigest())
    assert file.rows == ROWS and file.examples == EXAMPLES


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        _bytes(schema="kernel-grammar/v2"),
        _bytes(name=""),
        _bytes(rows="x"),
        _bytes(examples=[{"text": "a", "steps": "x"}]),
        _bytes(examples=[{"text": "zzz qqq", "steps": []}]),
        _bytes(extra=1),
    ],
)
def test_parse_refuses_a_file_that_is_not_the_schema(raw):
    with pytest.raises(GrammarError) as err:
        parse_grammar(raw, "argument.json")
    assert err.value.problems and all("argument.json" in p for p in err.value.problems)


def test_an_example_states_at_least_one_step():
    """An example with no steps would pass on any reading, an unread sentence included: it certifies nothing, and the
    file is refused at parse, naming the example."""
    good = {"text": ONE, "steps": [{"tool": "statement", "args": {"text": ONE}}]}
    with pytest.raises(GrammarError) as err:
        parse_grammar(_bytes(examples=[good, {"text": "zzz qqq", "steps": []}]), "argument.json")
    assert err.value.problems == ["argument.json: example 1: an example states at least one step"], err.value.problems


def test_parse_names_every_problem_of_the_shape():
    with pytest.raises(GrammarError) as err:
        parse_grammar(_bytes(schema="nope", name=3, rows="x"), "g.json")
    assert len(err.value.problems) == 3


# -- the door ---------------------------------------------------------------------------------------------------------


@pytest.mark.needs_fcg
def test_a_file_installs_and_reads(store, engine, registry):
    done = install(store, engine, FILE, registry=registry)
    assert (done.added, done.replaced, done.removed, done.examples) == (4, 0, 0, 2)
    assert (done.name, done.sha256, done.kept_taught, done.unexercised) == (NAME, FILE.sha256, (), 1)  # op.not
    installed = [r for r in store.live() if r.id != OWN_ROW.id]
    assert len(installed) == 4
    assert all(r.kind == "hand" and r.status == "live" for r in installed)
    assert all(r.provenance == {GRAMMAR: NAME, GRAMMAR_HASH: FILE.sha256} for r in installed)
    assert store.get(OWN_ROW.id) == OWN_ROW
    assert {r.id: r.score for r in installed}["cx.because"] == 0.7
    # the engine holds what the store now holds, and reads the file's sentences
    assert reads(engine, ONE) == (True, reads(engine, ONE)[1]) and "cx.because" in reads(engine, ONE)[1]
    assert reads(engine, OWN)[0]


BREAKAGES = [
    ("a row that does not compile", "does not compile"),
    ("an unknown tool", "nosuchtool"),
    ("a tool with one argument too many", "the meaning gives 5"),
    ("a filler slot named r", "may not be named r"),
    ("two rows with one id", "appears twice"),
    ("a base row's id", "names a base construction"),
    ("the generated clause filler's id", "the id is the generated clause filler's"),
    ("a catalogue card's id", "the id is a catalogue card's"),
    ("an id in the frame parts' namespace", "the id is in the frame parts' namespace"),
    ("a part row stating as_written", "as_written is the kernel's"),
    ("an id a taught row holds", "did not install"),
    ("an id another grammar's hand row holds", "stored hand row that grammar 'argument' did not install"),
    ("an id a mined row holds", "stored mined row that grammar 'argument' did not install"),
    ("no examples", "no example"),
    ("an example whose first reading differs", "first reading"),
    ("a row stating kind", "the door sets kind"),
]


@pytest.mark.needs_fcg
@pytest.mark.parametrize(("breakage", "problem"), BREAKAGES)
def test_a_bad_file_changes_nothing_and_names_every_problem(breakage, problem, store, engine, registry):
    bad = broken(FILE, breakage, store, and_also="an unknown tool")
    load_store(store, engine)
    before, calls, mine, theirs = rows_of(store), list(store.writer.calls), reads(engine, OWN), reads(engine, ONE)
    assert mine[0] and not theirs[0]
    with pytest.raises(GrammarError) as err:
        install(store, engine, bad, registry=registry, cards=(CARD,))
    assert any(problem in p for p in err.value.problems), (problem, err.value.problems)
    assert any("nosuchtool" in p for p in err.value.problems)  # and_also, on another row
    assert len(err.value.problems) >= 2, err.value.problems
    assert store.all() == before and store.writer.calls == calls
    assert (reads(engine, OWN), reads(engine, ONE)) == (mine, theirs)  # the engine holds the bonfire's grammar still


@pytest.mark.needs_fcg
def test_a_refused_install_leaves_the_engine_on_the_bonfires_grammar_not_the_refused_files(store, engine, registry):
    """The self-test loads a scratch grammar into the engine; a file that fails its example must not leave it there."""
    load_store(store, engine)
    bad = with_file(examples=(Example(ONE, (("statement", {"text": "nope"}),)),))
    with pytest.raises(GrammarError) as err:
        install(store, engine, bad, registry=registry)
    assert any("first reading" in p and ONE in p for p in err.value.problems), err.value.problems
    assert reads(engine, OWN)[0]  # a sentence only the bonfire's own rows read still reads
    assert not reads(engine, ONE)[0]  # a sentence only the refused file would read does not
    assert all(name != "cx.because" for name in reads(engine, ONE)[1])


@pytest.mark.needs_fcg
def test_every_problem_of_a_file_comes_in_one_error(store, engine, registry):
    bad = broken(FILE, "an unknown tool", store, and_also="a tool with one argument too many")
    with pytest.raises(GrammarError) as err:
        install(store, engine, bad, registry=registry)
    text = "\n".join(err.value.problems)
    assert "nosuchtool" in text and "op.not" in text and "cx.because" in text


@pytest.mark.needs_fcg
def test_installing_again_replaces_this_grammar_only(store, engine, registry):
    other = CxnSpec(
        id="other.hi",
        kind="hand",
        unit="root",
        form=["hi"],
        meaning=[["const", "?x", "y"]],
        status="live",
        provenance={GRAMMAR: "somebody-else"},
    )
    mined = CxnSpec(
        id="mined.hey", kind="mined", unit="root", form=["hey"], meaning=[["const", "?x", "y"]], status="live"
    )
    taught = CxnSpec(
        id="taught.yo",
        kind="taught",
        unit="root",
        form=["yo"],
        meaning=[["const", "?x", "y"]],
        status="live",
        provenance={GRAMMAR: NAME},  # a row of this grammar's name, taught since: not the door's to remove
    )
    for row in (other, mined, taught):
        store.put(row)
    install(store, engine, FILE, registry=registry)
    store.record_choices([("cx.because", "cx.if"), ("cx.because", "cx.if")])
    untouched = {r.id: r for r in (store.get(i) for i in (OWN_ROW.id, other.id, mined.id, taught.id))}
    won = store.get("cx.because").standing
    assert (won.won, store.get("cx.if").standing.lost) == (2, 2)

    smaller = with_file(
        sha256="e" * 64,
        rows=(ROWS[0], ROWS[1], ROWS[3]),  # cx.if is dropped
        examples=(EXAMPLES[0],),
    )
    done = install(store, engine, smaller, registry=registry)
    assert (done.added, done.replaced, done.removed) == (0, 3, 1)
    assert store.get("cx.if") is None
    assert {r.id: r for r in (store.get(i) for i in untouched)} == untouched
    kept = store.get("cx.because")
    assert kept.standing == won  # a row that stays keeps its counters
    assert kept.provenance == {GRAMMAR: NAME, GRAMMAR_HASH: "e" * 64}
    assert store.writer.nodes["cx.because"]["won"] == 2
    assert "cx.because" in reads(engine, ONE)[1] and "cx.if" not in reads(engine, TWO)[1]  # the new grammar


@pytest.mark.needs_fcg
def test_a_row_teach_has_modified_is_left_and_reported(store, engine, registry):
    taught = CxnSpec.from_json({**ROWS[2], "kind": "taught", "status": "live", "provenance": {GRAMMAR: NAME}})
    store.put(taught)
    done = install(store, engine, FILE, registry=registry)
    assert done.kept_taught == ("cx.if",)
    assert (done.added, done.replaced, done.removed) == (3, 0, 0)
    assert store.get("cx.if") == taught  # as teach left it
    assert store.get("cx.because").kind == "hand"
    # an example read through the taught row is named, not judged: one example judged, the other named
    assert (done.examples, done.through_taught) == (1, (TWO,)) and done.note is None
    # a file that no longer holds the id leaves the taught row alone
    done = install(store, engine, with_file(rows=(ROWS[0], ROWS[1]), examples=(EXAMPLES[0],)), registry=registry)
    assert done.removed == 1 and store.get("cx.if") == taught  # op.not goes; the taught row stays
    assert store.get("op.not") is None


@pytest.mark.needs_fcg
def test_unexercised_counts_rows_no_example_applied(store, engine, registry):
    done = install(store, engine, with_file(examples=(EXAMPLES[0],)), registry=registry)
    assert (done.examples, done.unexercised) == (1, 2)  # cx.if and op.not


@pytest.mark.needs_fcg
def test_an_install_that_judges_no_example_says_so(store, engine, registry):
    """``examples`` counts the examples JUDGED, and ``unexercised`` counts from them only: a file whose one example
    reads through a taught row the grammar keeps installs with none judged, every row it puts unexercised, and a note
    that says so."""
    store.put(CxnSpec.from_json({**ROWS[2], "kind": "taught", "status": "live", "provenance": {GRAMMAR: NAME}}))
    done = install(store, engine, with_file(examples=(EXAMPLES[1],)), registry=registry)
    assert (done.examples, done.through_taught, done.kept_taught) == (0, (TWO,), ("cx.if",))
    assert done.unexercised == 3  # root.s, cx.because, op.not: applied in no judged example
    assert done.note is not None and done.note.startswith("no example was judged"), done.note


#: a modify of cx.if that writes "because" (its slots kept, its relation's two clauses swapped, so it duplicates nothing)
BECAUSE_SWAPPED = TeachRequest(
    ["{H:clause} because {D:clause}"],
    [{"call": "relation", "args": {"whole": "{X}", "head": "{D}", "text": "{H}"}}],
    {},
    "Ana",
    "c1",
    modify="cx.if",
)


@pytest.mark.needs_fcg
def test_a_staged_draft_takes_no_part_in_an_install(store, engine, registry):
    """The scratch store the examples are read on holds live rows only: a staged modify (taught, carrying the grammar's
    keys) that would take an example's first reading on its own score (cx.if's 0.7, above cx.because's 0.6) neither
    refuses the install nor changes it, and stays staged."""
    file = with_file(rows=edited(1, score=0.6))
    install(store, engine, file, registry=registry)
    draft = teach(store, engine, BECAUSE_SWAPPED, registry=registry)
    assert draft.problems == [] and draft.specs[0].score == 0.7, draft.problems
    done = install(store, engine, file, registry=registry)
    assert (done.examples, done.through_taught, done.kept_taught) == (2, (), ())
    assert done.note is None and store.get(draft.specs[0].id) == draft.specs[0]


def test_the_scratch_store_holds_live_rows_only():
    """What the door reads the examples on: every live row the install leaves, at won = lost = 0 for this grammar's;
    never a staged draft, of this grammar or another."""
    from memory_kernel.runtime.grammar import _scratch_rows

    kept = CxnSpec.from_json({**ROWS[0], "kind": "hand", "status": "live", "provenance": {GRAMMAR: NAME}})
    draft = CxnSpec.from_json(
        {**ROWS[2], "id": "if.t0000000001.0", "kind": "taught", "status": "staged", "provenance": {GRAMMAR: NAME}}
    )
    stored = {s.id: s for s in (kept, draft, OWN_ROW)}
    assert [s.id for s in _scratch_rows(stored, [], set(), [], NAME)] == [kept.id, OWN_ROW.id]


@pytest.mark.needs_fcg
def test_install_works_inside_a_running_loop(store, engine, registry):
    async def go():
        return install(store, engine, FILE, registry=registry)

    done = asyncio.run(go())
    assert (done.added, done.examples) == (4, 2)
    assert reads(engine, OWN)[0]


@pytest.mark.needs_fcg
def test_the_error_says_when_the_examples_were_not_read(store, engine, registry):
    bad = broken(FILE, "a row that does not compile", store)
    rows = list(bad.rows)
    rows[2] = {**rows[2], "id": OWN_ROW.id}  # an id a stored row of nobody's grammar holds
    with pytest.raises(GrammarError) as err:
        install(store, engine, dataclasses.replace(bad, rows=tuple(rows)), registry=registry)
    assert "the examples were not read: 1 rows do not compile, 1 ids clash" in err.value.problems, err.value.problems


@pytest.mark.needs_fcg
def test_a_scratch_load_the_engine_refuses_is_a_problem_and_the_engine_is_restored(
    store, engine, registry, monkeypatch
):
    """A live engine that answers the scratch grammar's load with a refusal (``Refused``: it answered, and said no):
    a problem of the file."""
    load_store(store, engine)
    real, calls = engine.load, []

    def flaky(*args, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise engine_module.Refused("the scratch grammar is refused")
        return real(*args, **kw)

    monkeypatch.setattr(engine, "load", flaky)
    with pytest.raises(GrammarError) as err:
        install(store, engine, FILE, registry=registry)
    assert any("scratch grammar is refused" in p for p in err.value.problems), err.value.problems
    assert engine.pid is not None  # the runtime's start stopped it; the door started it again
    assert reads(engine, OWN)[0] and not reads(engine, ONE)[0]
    assert store.get("cx.because") is None


@pytest.mark.needs_fcg
def test_an_engine_that_cannot_load_the_written_rows_raises_saying_they_are_written(
    store, engine, registry, monkeypatch
):
    """The one path where an engine error leaves ``install`` with the rows WRITTEN: the write landed, then the engine
    could not load the bonfire's new grammar. It is a ``RuntimeEngineError`` (never a ``GrammarError``) whose message
    says the rows are written, and the store holds them."""
    from memory_kernel.runtime import grammar as grammar_module

    def fails(*args: Any, **kw: Any) -> None:
        raise RuntimeEngineError("the load failed")

    load_store(store, engine)
    monkeypatch.setattr(grammar_module, "load_store", fails)  # the restore's load, after the write
    try:
        with pytest.raises(RuntimeEngineError) as err:
            install(store, engine, FILE, registry=registry)
        assert not isinstance(err.value, GrammarError)
        assert f"the rows of grammar {NAME!r} ARE written, but the engine could not load them" in str(err.value)
        assert store.get("cx.because") is not None and "cx.because" in store.writer.nodes
    finally:
        monkeypatch.undo()
        load_store(store, engine)


@pytest.mark.needs_fcg
def test_an_unexpected_error_is_never_masked_and_the_engine_is_restored(store, engine, registry, monkeypatch):
    load_store(store, engine)
    real, calls = engine.load, []

    def boom(*args, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real(*args, **kw)

    monkeypatch.setattr(engine, "load", boom)
    with pytest.raises(RuntimeError, match="boom"):
        install(store, engine, FILE, registry=registry)
    assert engine.pid is not None and reads(engine, OWN)[0]
    assert store.get("cx.because") is None


@pytest.mark.needs_fcg
def test_an_engine_that_was_not_running_is_left_not_running(store, engine, registry):
    engine.stop()
    try:
        done = install(store, engine, FILE, registry=registry)
        assert done.added == 4 and engine.pid is None
        with pytest.raises(GrammarError):
            install(store, engine, with_file(examples=()), registry=registry)
        assert engine.pid is None
    finally:
        engine.start()
        load_store(store, engine)


def _torn_down(engine: Any, monkeypatch: Any) -> None:
    """A transport fault tears ``engine`` down: it has no process, and the next call starts it again (it is still
    started)."""
    hit = _time_out_once(engine, monkeypatch, OWN)
    with pytest.raises(RuntimeEngineError):
        engine.comprehend(normalise(OWN).text)
    assert hit and engine.pid is None


@pytest.mark.needs_fcg
def test_an_engine_a_fault_tore_down_is_still_running_after_an_install(store, engine, registry, monkeypatch):
    """The door decides by whether the engine was started, not by its pid: an engine a fault tore down before the call
    was started, so a good install leaves it running on the new grammar, never stopped."""
    load_store(store, engine)
    _torn_down(engine, monkeypatch)
    try:
        done = install(store, engine, FILE, registry=registry)
        assert done.added == 4
        assert reads(engine, OWN)[0] and "cx.because" in reads(engine, ONE)[1]  # running, holding the new grammar
        assert engine.started
    finally:
        if engine.pid is None:
            engine.start()
        load_store(store, engine)


@pytest.mark.needs_fcg
def test_an_engine_the_door_never_started_is_never_stopped_by_it(store, engine, registry, monkeypatch):
    """A file refused before any example is read never stops the engine: one a fault tore down before the call is
    still started afterwards (the door starts it again and loads the bonfire's grammar)."""
    load_store(store, engine)
    _torn_down(engine, monkeypatch)
    try:
        with pytest.raises(GrammarError) as err:
            install(store, engine, broken(FILE, "a row that does not compile", store), registry=registry)
        assert any("the examples were not read" in p for p in err.value.problems), err.value.problems
        assert reads(engine, OWN)[0] and not reads(engine, ONE)[0]
        assert engine.started
    finally:
        if engine.pid is None:
            engine.start()
        load_store(store, engine)


# -- an engine fault is the engine's error, a refusal the file's problem -------------------------------------------------


def _time_out_once(engine: Any, monkeypatch: Any, text: str) -> list[int]:
    """The engine's next comprehension request for ``text`` gets a timeout it cannot meet: a real HTTP timeout against
    the live engine, which ``comprehend`` meets as every transport failure (it tears the engine down). Returns the list
    the pid of the engine it timed out on is appended to."""
    real, target, hit = engine._request, normalise(text).text, []

    def request(method, endpoint, data, *, timeout=None):
        if not hit and endpoint == "/fcg-comprehend-all" and isinstance(data, dict) and data.get("utterance") == target:
            hit.append(engine.pid)
            timeout = 1e-6
        return real(method, endpoint, data, timeout=timeout)

    monkeypatch.setattr(engine, "_request", request)
    return hit


@pytest.mark.needs_fcg
def test_an_example_read_that_times_out_is_the_engines_error_and_changes_nothing(store, engine, registry, monkeypatch):
    """The first example's read gets no answer in time: the timeout comes out of ``install`` as the engine's error, not
    as a problem of the file; nothing is written; the engine, which the timeout tore down, is started again holding the
    bonfire's grammar."""
    load_store(store, engine)
    before, calls = rows_of(store), list(store.writer.calls)
    hit = _time_out_once(engine, monkeypatch, ONE)
    with pytest.raises(RuntimeEngineError) as err:
        install(store, engine, FILE, registry=registry)
    assert hit and isinstance(err.value, engine_module._Timeout) and not isinstance(err.value, GrammarError)
    assert store.all() == before and store.writer.calls == calls
    assert engine.pid not in (None, hit[0])  # torn down by the timeout, started again by the door
    assert reads(engine, OWN)[0] and not reads(engine, ONE)[0]


@pytest.mark.needs_fcg
@pytest.mark.parametrize("running", [True, False], ids=["running", "not-running"])
def test_an_engine_that_dies_between_two_examples_is_the_engines_error(running, store, engine, registry, monkeypatch):
    """The engine process is killed after the first example was read and before the second is: its death comes out of
    ``install`` as the engine's error; nothing is written; the engine ends as it was given (running and holding the
    bonfire's grammar, or not running, and not started again by a later call)."""
    load_store(store, engine)
    if not running:
        engine.stop()
    before, calls = rows_of(store), list(store.writer.calls)
    real, second, killed = engine.comprehend, normalise(TWO).text, []

    def comprehend(text):
        if text == second and not killed:
            killed.append(engine.pid)
            os.kill(engine.pid, signal.SIGKILL)
            deadline = time.monotonic() + 10
            while engine._engine.process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
        return real(text)

    monkeypatch.setattr(engine, "comprehend", comprehend)
    try:
        with pytest.raises(RuntimeEngineError, match="engine died") as err:
            install(store, engine, FILE, registry=registry)
        assert killed and not isinstance(err.value, GrammarError)
        assert store.all() == before and store.writer.calls == calls
        if running:
            assert engine.pid not in (None, killed[0])
            assert reads(engine, OWN)[0] and not reads(engine, ONE)[0]
        else:
            assert engine.pid is None
            with pytest.raises(RuntimeEngineError, match="not running"):
                engine.comprehend(normalise(OWN).text)
    finally:
        if engine.pid is None:
            engine.start()
            load_store(store, engine)


@pytest.mark.needs_fcg
def test_an_example_read_the_engine_refuses_is_a_problem_of_the_file(store, engine, registry, monkeypatch):
    """A live engine that answers an example's read with a refusal (``Refused``) stays a problem of the file, said
    once, and the engine is left on the bonfire's grammar."""
    load_store(store, engine)
    before = rows_of(store)
    real, second = engine.comprehend, normalise(TWO).text

    def comprehend(text):
        if text == second:
            raise engine_module.Refused("runtime engine /fcg-comprehend-all: HTTP 400: the read is refused")
        return real(text)

    monkeypatch.setattr(engine, "comprehend", comprehend)
    with pytest.raises(GrammarError) as err:
        install(store, engine, FILE, registry=registry)
    assert len(err.value.problems) == 1 and "the read is refused" in err.value.problems[0], err.value.problems
    assert err.value.problems[0] == (  # the engine's own words, said once
        "the engine refused the grammar the examples are read on (runtime engine /fcg-comprehend-all: HTTP 400: the "
        "read is refused)"
    )
    assert store.all() == before
    assert reads(engine, OWN)[0] and not reads(engine, ONE)[0]


@pytest.mark.needs_fcg
def test_the_same_file_installs_again_whatever_picks_have_moved(store, engine, registry):
    """The scratch read uses the file's own scores: counters that dropped a row below the clause filler do not stop the
    same file from installing again, and the write keeps them."""
    low = [copy.deepcopy(r) for r in ROWS]
    low[1]["score"] = 0.55
    file = with_file(rows=tuple(low), examples=(EXAMPLES[0],))
    install(store, engine, file, registry=registry)
    store.record_choices([("root.s", "cx.because")] * 20)  # cx.because lost 20: 0.55 - 0.2 * 20 / 24 < 0.5
    assert store.get("cx.because").standing.lost == 20
    load_store(store, engine)
    assert "cx.because" not in reads(engine, ONE)[1]  # its pick-moved score no longer reads the sentence first
    done = install(store, engine, file, registry=registry)
    assert (done.added, done.replaced, done.examples) == (0, 4, 1)
    assert store.get("cx.because").standing.lost == 20 and store.writer.nodes["cx.because"]["lost"] == 20


# -- mk grammar install -------------------------------------------------------------------------------------------------

CATALOGUE = json.dumps(
    {
        "schema_version": "kernel-catalogue/v1",
        "tools": [
            {
                "name": "statement",
                "x-side-effect": False,
                "input_schema": {"properties": {"text": {"type": "string"}}},
                "x-args": {"text": {"role": "text"}},
            },
            {
                "name": "relation",
                "x-side-effect": False,
                "input_schema": {"properties": {k: {"type": "string"} for k in ("whole", "head", "text")}},
                "x-args": {k: {"role": k} for k in ("whole", "head", "text")},
            },
            {
                "name": "mark",
                "x-side-effect": False,
                "input_schema": {"properties": {k: {"type": "string"} for k in ("whole", "scope", "rest")}},
                "x-args": {k: {"role": k} for k in ("whole", "scope", "rest")},
            },
        ],
    }
)


class _Result:
    def consume(self):
        return None


class _GraphDriver:
    """A sync driver over a dict of ``TaughtPhrase`` nodes: answers the writer's reads, applies its puts and removes."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.ran: list[tuple[str, dict]] = []
        self.closed = False

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        self.closed = True

    def execute_write(self, fn):
        return fn(self)

    def run(self, cypher, **params):
        self.ran.append((cypher, params))
        if cypher == graph_store.READ_PHRASES:
            return [self.nodes[k] for k in sorted(self.nodes)]
        if cypher == graph_store.READ_CATALOGUE:
            return [{"raw": CATALOGUE}]
        if "BonfireMeta" in cypher:
            return [{"g": params["g"]}]
        if cypher == graph_store.REMOVE_PHRASES:
            for i in params["ids"]:
                self.nodes.pop(i, None)
        elif cypher == graph_store.PUT_PHRASES:
            for r in params["rows"]:
                node = self.nodes.setdefault(
                    r["id"], {k: r[k] for k in ("id", "confirmed", "uses_ok", "uses_failed", "uses_world")}
                )
                node["spec_json"] = r["spec_json"]
        return _Result()


class _SharedEngine:
    """The module's engine under another name: the command's own ``stop`` must not stop it."""

    def __init__(self, engine):
        self._engine = engine

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def stop(self):
        return None


@pytest.fixture
def door(engine, monkeypatch):
    driver = _GraphDriver()
    monkeypatch.setattr(cli, "_bonfire_driver", lambda args, secrets: (driver, "bolt://fake", "neo4j"))
    monkeypatch.setattr("memory_kernel.runtime.engine.RuntimeEngine", lambda **kw: _SharedEngine(engine))
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    return driver


@pytest.mark.needs_fcg
def test_the_command_prints_the_summary_of_a_good_file(door, tmp_path, capsys):
    path = tmp_path / "argument.json"
    path.write_bytes(_bytes())
    assert cli.main(["grammar", "install", "b1", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["name"] == NAME and re.fullmatch(r"[0-9a-f]{64}", out["sha256"])
    keys = ("added", "replaced", "removed", "kept_taught", "examples", "through_taught", "unexercised", "note")
    assert {k: out[k] for k in keys} == {
        "added": 4,
        "replaced": 0,
        "removed": 0,
        "kept_taught": [],
        "examples": 2,  # judged
        "through_taught": 0,  # how many examples read through a taught row the grammar keeps (named, not judged)
        "unexercised": 1,
        "note": None,
    }
    assert set(door.nodes) == {r["id"] for r in ROWS}
    assert json.loads(door.nodes["cx.if"]["spec_json"])["provenance"][GRAMMAR] == NAME


@pytest.mark.needs_fcg
def test_the_command_says_plainly_when_no_example_was_judged(door, tmp_path, capsys):
    taught = CxnSpec.from_json({**ROWS[2], "kind": "taught", "status": "live", "provenance": {GRAMMAR: NAME}})
    door.nodes[taught.id] = {
        "id": taught.id,
        **{k: 0 for k in ("confirmed", "uses_ok", "uses_failed", "uses_world")},
        "spec_json": graph_store.phrase_json(taught),
    }
    path = tmp_path / "argument.json"
    path.write_bytes(_bytes(examples=[{"text": TWO, "steps": [{"tool": t, "args": a} for t, a in EXAMPLES[1].steps]}]))
    assert cli.main(["grammar", "install", "b1", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["examples"], out["through_taught"], out["kept_taught"]) == (0, 1, ["cx.if"])
    assert out["note"].startswith("no example was judged"), out


@pytest.mark.needs_fcg
def test_the_command_exits_1_on_a_bad_file_and_writes_nothing(door, tmp_path, capsys):
    path = tmp_path / "argument.json"
    bad = [copy.deepcopy(r) for r in ROWS]
    bad[1]["meaning"][-1][0] = "nosuchtool"
    path.write_bytes(_bytes(rows=bad))
    assert cli.main(["grammar", "install", "b1", str(path)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["problems"] and "nosuchtool" in " ".join(out["problems"])
    assert door.nodes == {} and not [c for c, _ in door.ran if c in graph_store.WRITES]


@pytest.mark.needs_fcg
def test_the_command_reports_an_engine_fault_as_the_engine_and_writes_nothing(
    door, engine, tmp_path, capsys, monkeypatch
):
    path = tmp_path / "argument.json"
    path.write_bytes(_bytes())
    hit = _time_out_once(engine, monkeypatch, ONE)
    assert cli.main(["grammar", "install", "b1", str(path)]) == cli.REFUSED
    out = json.loads(capsys.readouterr().out)
    assert hit and "problems" not in out
    assert out["refused"].startswith("the runtime engine failed") and "no answer within" in out["refused"], out
    assert door.nodes == {} and not [c for c, _ in door.ran if c in graph_store.WRITES]


def test_the_command_refuses_a_file_that_is_not_a_grammar_before_it_connects(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_bonfire_driver", lambda *a: pytest.fail("connected"))
    path = tmp_path / "g.json"
    path.write_bytes(b"[]")
    assert cli.main(["grammar", "install", "b1", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["problems"]
    assert cli.main(["grammar", "install", "b1", str(tmp_path / "missing.json")]) == cli.REFUSED


class _BrokenEngine:
    pid = None
    started = False

    def start(self):
        raise RuntimeEngineError("no pinned FCG Go: run `mk setup`")

    def stop(self):
        return None


def test_an_engine_that_cannot_start_is_reported_as_the_engine(monkeypatch, tmp_path, capsys):
    driver = _GraphDriver()
    monkeypatch.setattr(cli, "_bonfire_driver", lambda args, secrets: (driver, "bolt://fake", "neo4j"))
    monkeypatch.setattr("memory_kernel.runtime.engine.RuntimeEngine", lambda **kw: _BrokenEngine())
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    path = tmp_path / "argument.json"
    path.write_bytes(_bytes())
    assert cli.main(["grammar", "install", "b1", str(path)]) == cli.REFUSED
    refused = json.loads(capsys.readouterr().out)["refused"]
    assert "no pinned FCG Go" in refused and "neo4j" not in refused.lower()
    assert driver.nodes == {}
