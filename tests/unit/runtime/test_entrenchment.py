"""A score on a row, moved by recorded choices (won/lost counters), and the reserved filler slot names."""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

from memory_kernel.runtime.constructions import (
    SCORE,
    CompileError,
    CxnSpec,
    Standing,
    compile_spec,
    entrenched,
    reserved_slots,
)
from memory_kernel.runtime.store import ConstructionStore, compiled, load_store
from unit.runtime.memory_writer import MemoryWriter, memory_store

MEANING = [["clause-span", "?x", "?l", "?r"], ["const", "?k", "reason"], ["relation", "?g", "?k", "?d"]]
D = {"slot": "d", "fills": "clause"}

#: ``to_json`` of ``_filler("cx.a")`` taken on the branch before the score and the counters existed
BEFORE = (
    '{"answer_shape": null, "fills": "clause", "form": ["because ", {"fills": "clause", "slot": "d"}], "id": "cx.a", '
    '"kind": "taught", "meaning": [["clause-span", "?x", "?l", "?r"], ["relation", "?g", "?d"]], "provenance": '
    '{"n": 1}, "standing": {"confirmed": 0, "uses_failed": 0, "uses_ok": 0, "uses_world": 0}, "status": "live", '
    '"unit": "filler"}'
)


def _filler(id_: str, *, score=None, won=0, lost=0, form=None) -> CxnSpec:
    return CxnSpec(
        id=id_,
        kind="taught",
        unit="filler",
        fills="clause",
        form=form or ["because ", D],
        meaning=MEANING,
        status="live",
        score=score,
        standing=Standing(won=won, lost=lost),
    )


@pytest.fixture
def writer() -> MemoryWriter:
    return MemoryWriter()


@pytest.fixture
def store(writer) -> ConstructionStore:
    return memory_store(_filler("cx.a"), _filler("cx.b"), writer=writer)


def test_a_row_without_score_or_choices_is_stored_as_before():
    spec = _filler("cx.a")
    assert "score" not in spec.to_json()
    assert set(spec.to_json()["standing"]) == {"confirmed", "uses_ok", "uses_failed", "uses_world"}
    assert entrenched(spec) == SCORE == 0.5


def test_a_row_without_score_or_choices_serialises_byte_for_byte():
    spec = CxnSpec(
        id="cx.a",
        kind="taught",
        unit="filler",
        fills="clause",
        form=["because ", D],
        meaning=[["clause-span", "?x", "?l", "?r"], ["relation", "?g", "?d"]],
        status="live",
        provenance={"n": 1},
    )
    assert json.dumps(spec.to_json(), sort_keys=True) == BEFORE
    assert CxnSpec.from_json(json.loads(BEFORE)) == spec


def test_a_scored_row_round_trips():
    spec = _filler("cx.a", score=0.65, won=2, lost=1)
    row = spec.to_json()
    assert row["score"] == 0.65 and row["standing"]["won"] == 2 and row["standing"]["lost"] == 1
    assert CxnSpec.from_json(row) == spec
    assert "won" not in _filler("cx.a", lost=1).to_json()["standing"]


def test_the_score_is_a_function_of_the_counters():
    assert entrenched(_filler("cx.a", score=0.65)) == 0.65
    assert entrenched(_filler("cx.a", won=1)) == pytest.approx(0.54)
    assert entrenched(_filler("cx.a", score=0.6, lost=5)) < 0.5
    assert entrenched(_filler("cx.a", score=0.6, lost=4)) >= 0.5 - 1e-9
    assert entrenched(_filler("cx.a", won=10_000)) < 0.7 + 1e-9


@pytest.mark.parametrize("score", [0.19, 0.81, "0.5", True])
def test_a_score_outside_the_range_is_refused(score):
    with pytest.raises(CompileError):
        _filler("cx.a", score=score)


def test_the_engine_gets_the_entrenched_score():
    assert compiled(_filler("cx.a"))[0].attributes["score"] == 0.5
    assert compiled(_filler("cx.a", score=0.65))[0].attributes["score"] == 0.65
    assert compiled(_filler("cx.a", won=1))[0].attributes["score"] == pytest.approx(0.54)


def test_record_choices_is_one_write_and_skips_what_has_no_row(store, writer):
    rows = store.record_choices([("cx.a", "cx.b"), ("clause", "cx.b"), ("cx.a", "cx.a"), ("nope", "cx.a#v1")])
    assert (store.get("cx.a").standing.won, store.get("cx.a").standing.lost) == (1, 1)
    assert (store.get("cx.b").standing.won, store.get("cx.b").standing.lost) == (0, 2)
    assert sorted(rows) == ["cx.a", "cx.b"]
    assert [c for c in writer.calls if c[0] == "add_choices"] == [("add_choices", {"cx.a": (1, 1), "cx.b": (0, 2)})]
    assert (writer.nodes["cx.b"]["won"], writer.nodes["cx.b"]["lost"]) == (0, 2)


def test_record_choices_counts_an_alternates_name_for_its_spec(store, writer):
    store.record_choices([("cx.a#v1", "cx.b")])
    assert store.get("cx.a").standing.won == 1 and store.get("cx.b").standing.lost == 1


def test_record_choices_with_nothing_to_record_writes_nothing(store, writer):
    assert store.record_choices([("clause", "nope"), ("cx.a", "cx.a"), ("cx.b#v0", "cx.b")]) == {}
    assert not [c for c in writer.calls if c[0] == "add_choices"]


def test_the_compile_cache_tells_two_scores_apart():
    a = _filler("cx.a")
    assert compiled(a) is not compiled(dataclasses.replace(a, standing=Standing(won=3)))
    assert compiled(a) is not compiled(dataclasses.replace(a, score=0.6))
    # what compilation does not read still shares one entry
    assert compiled(a) is compiled(dataclasses.replace(a, standing=Standing(uses_ok=9, confirmed=2), status="staged"))


def test_a_stored_filler_with_a_reserved_slot_loads_with_one_warning(caplog):
    class Engine:
        loaded: list = []

        def load(self, cxns, **_kw):
            self.loaded = list(cxns)

    bad = _filler("cx.l", form=["because ", {"slot": "l", "fills": "clause"}])
    store = memory_store(_filler("cx.a"), bad)
    engine = Engine()
    with caplog.at_level(logging.WARNING, logger="memory_kernel.runtime.store"):
        load_store(store, engine)
    messages = [r.getMessage() for r in caplog.records if "cx.l" in r.getMessage()]
    assert len(messages) == 1 and len(caplog.records) == 1
    assert any(c.name == "cx.l" for c in engine.loaded)


def test_reserved_slots_names_l_and_r_in_a_filler_only():
    both = _filler("cx.lr", form=[{"slot": "l", "fills": "clause"}, " and ", {"slot": "r", "fills": "clause"}])
    assert reserved_slots(both) == ["l", "r"]
    assert reserved_slots(_filler("cx.a")) == []
    root = CxnSpec(
        id="cx.root",
        kind="taught",
        unit="root",
        form=["go ", {"slot": "l", "fills": "np"}],
        meaning=[["const", "?l", "a"]],
    )
    assert reserved_slots(root) == []
    part = dataclasses.replace(root, id="cx.part", unit="part", form=["go ", {"slot": "r", "fills": "np"}, " now"])
    assert reserved_slots(part) == []
    assert compile_spec(both).cxns  # compile does not refuse them
