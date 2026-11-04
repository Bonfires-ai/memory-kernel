"""Composition by reference (``invoke``): a body clause read through a taught construction stores a reference to it,
expanded at plan time into that construction's live program; a dangling reference repairs with its name; a cycle is
refused at teach time; the invoked construction records the turn's outcome."""

import asyncio
import json

import pytest
from test_plan_api import FakeGraph, _data_dir, _plan, _result, _runtime

from memory_kernel.runtime.api import default_registry
from memory_kernel.runtime.catalogue import parse_catalogue
from memory_kernel.runtime.teach import TeachRequest, _slots

PORT = 9723  # 9720 engine search, 9721 resume verdicts, 9722 teaching (test_store_teach)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("invoke-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


def _catalogue():
    tool = {
        "name": "make_sound",
        "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "source": {"type": "string"}}},
        "x-side-effect": True,
        "x-args": {
            "pattern": {"role": "theme", "constraint": "tune"},
            "source": {"role": "instrument", "constraint": "instrument"},
        },
        "x-forms": ["play {pattern} on {source}"],
    }
    return parse_catalogue(json.dumps({"schema_version": "kernel-catalogue/v1", "tools": [tool]}).encode(), "catalogue")


def _hum_runtime(engine, tmp_path):
    """A runtime holding the catalogue's card and a confirmed taught "hum" ."""
    cat = _catalogue()
    runtime = _runtime(engine, tmp_path, FakeGraph([], []), [], registry=default_registry(cat.tools), cards=cat.cards)
    req = TeachRequest(
        forms=["hum {X} on {S}", "hum {X}"],
        body=["play {X} on {S}"],
        defaults={"S": "an instrument"},
        teacher="Ana",
        chat_id="c1",
    )
    draft = asyncio.run(runtime.teach(req))
    assert draft.problems == [], draft.problems
    return runtime, asyncio.run(runtime.confirm(draft.draft_id, "Ana"))


def _hum_x(live):
    """The hum form with one slot ("hum {X}": its instrument is the default)."""
    return next(s for s in live if _slots(s) == ["x"])


def _teach(runtime, forms, body, defaults=None, replace=False):
    req = TeachRequest(forms, body, defaults or {}, teacher="Ana", chat_id="c1", replace=replace)
    draft = asyncio.run(runtime.teach(req))
    assert draft.problems == [], draft.problems
    return asyncio.run(runtime.confirm(draft.draft_id, "Ana"))


@pytest.mark.needs_fcg
def test_body_through_a_taught_verb_stores_a_reference(engine, tmp_path):
    runtime, hum = _hum_runtime(engine, tmp_path)
    [serenade] = _teach(runtime, ["serenade {X}"], ["hum {X}"])
    [invoke] = [p for p in serenade.meaning if p[0] == "invoke"]
    assert invoke[2:] == [_hum_x(hum).id, "?x"]  # the construction, then its slot args in its slot order
    assert not any(p[0] == "make_sound" for p in serenade.meaning)  # hum's ops are not copied
    assert serenade.form[1] == {"slot": "x", "fills": "np", "constraint": "tune"}  # notes read through the reference

    plan = _plan(runtime, "serenade a tune")
    assert plan.status == "needs_tools", plan.diagnostics
    assert plan.tool_steps[0]["args"] == {"pattern": "tune", "source": "instrument"}
    assert not any(p[0] == "invoke" for p in plan.program)  # the expanded program is what runs


@pytest.mark.needs_fcg
def test_reference_resolves_to_the_live_version(engine, tmp_path):
    """hum taught again (replace) with another default: serenade issues the new body's step, and the new hum records
    the turn's outcome."""
    runtime, _ = _hum_runtime(engine, tmp_path)
    _teach(runtime, ["serenade {X}"], ["hum {X}"])
    new_hum = _hum_x(_teach(runtime, ["hum {X} on {S}", "hum {X}"], ["play {X} on {S}"], {"S": "a drum"}, True))

    plan = _plan(runtime, "serenade a tune")
    assert plan.status == "needs_tools", plan.diagnostics
    [step] = plan.tool_steps
    assert step["args"] == {"pattern": "tune", "source": "drum"}
    done = asyncio.run(runtime.resume(plan.plan_id, [_result(step, value="hummed")]))
    assert done.status == "answered", done.diagnostics
    assert runtime.store.get(new_hum.id).standing.uses_ok == 1


@pytest.mark.needs_fcg
def test_dangling_reference_repairs_with_its_name(engine, tmp_path):
    runtime, hum = _hum_runtime(engine, tmp_path)
    _teach(runtime, ["serenade {X}"], ["hum {X}"])
    runtime.store.update(removes=[s.id for s in hum], replacing_taught=True)  # gone, with no successor

    plan = _plan(runtime, "serenade a tune")
    assert plan.status == "repair" and plan.tool_steps == []
    assert f"invokes {_hum_x(hum).id}, which is no longer live" in plan.diagnostics
    assert (
        plan.licence == []
    )  # R4: a program with an invoke is tool-bearing; its only tool was in hum: no tools, never null


@pytest.mark.needs_fcg
def test_cycle_is_refused_at_teach_time(engine, tmp_path):
    """hum taught again (replace) through hum itself: the reference would resolve to the new hum, which invokes
    itself."""
    runtime, hum = _hum_runtime(engine, tmp_path)
    req = TeachRequest(["hum {X}"], ["hum {X}"], {}, teacher="Ana", chat_id="c1", replace=True)
    draft = asyncio.run(runtime.teach(req))
    assert draft.specs == [] and any("invoke itself" in p for p in draft.problems), draft.problems
    assert runtime.store.get(_hum_x(hum).id).status == "live"
