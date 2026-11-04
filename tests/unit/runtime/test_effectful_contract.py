"""The effectful contract, end to end on the engine: a walkthrough of taught "hum" and "serenade" over a world-tool
catalogue, one scenario per contract rule (the kernel plans and judges; the game resolves and commits).

The game here is only what each scenario hand-writes: it reads a pending step, decides, and answers with a result row."""

import asyncio
import json
import shutil
from types import SimpleNamespace

import pytest
from test_plan_api import ANA, TOOLS_LICENCE, FakeGraph, _data_dir, _plan, _result, _runtime, _tool_runtime, _turn

from memory_kernel.runtime.api import ResumeError, default_registry
from memory_kernel.runtime.catalogue import parse_catalogue
from memory_kernel.runtime.teach import TeachRequest

PORT = 9725  # 9720-9724 are the other 1.4 engine modules'


def _tool(name, props, forms, side_effect=True, **extra):
    args = {p: dict(note) for p, note in props.items()}
    return {
        "name": name,
        "input_schema": {"type": "object", "properties": {p: {"type": "string"} for p in props}},
        "x-side-effect": side_effect,
        "x-args": args,
        "x-forms": forms,
        **({"x-consent": "speaker_turn"} if side_effect else {}),
        **extra,
    }


CATALOGUE = {
    "schema_version": "kernel-catalogue/v1",
    "tools": [
        _tool("look", {}, ["look around"], side_effect=False),
        _tool("take", {"item": {"role": "theme"}}, ["take {item}"], **{"x-returns": {"of": "item"}}),
        _tool("put", {"item": {"role": "theme"}, "place": {"role": "location"}}, ["put {item} on {place}"]),
        _tool(
            "give",
            {"item": {"role": "theme"}, "person": {"role": "recipient", "kind": "Person"}},
            ["give {item} to {person}"],
        ),
        _tool(
            "make",
            {"item": {"role": "theme"}, "material": {"role": "source", "many": True}},
            ["make {item} from {material}"],
            **{"x-returns": {"of": "item"}},
        ),
        _tool(
            "make_sound",
            {
                "source": {"role": "instrument", "constraint": "instrument"},
                "pattern": {"role": "theme", "constraint": "tune"},
            },
            ["play {pattern} on {source}"],
        ),
    ],
}
HUM = TeachRequest(["hum {X} on {S}", "hum {X}"], ["play {X} on {S}"], {"S": "an instrument"}, "Ana", "c1")
SERENADE = TeachRequest(["serenade {X}"], ["hum a tune", "give a flower to {X}"], {}, "Ana", "c1")


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("contract-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


def _catalogue():
    return parse_catalogue(json.dumps(CATALOGUE).encode(), "catalogue")


def _runtime_with_catalogue(engine, tmp_path):
    cat = _catalogue()
    return _runtime(engine, tmp_path, FakeGraph([ANA], []), [], registry=default_registry(cat.tools), cards=cat.cards)


def _teach(runtime, req, by="Ana"):
    draft = asyncio.run(runtime.teach(req))
    assert draft.problems == [], draft.problems
    return draft, lambda: asyncio.run(runtime.confirm(draft.draft_id, by))


@pytest.fixture(scope="module")
def world(engine, tmp_path_factory):
    """A runtime with the catalogue's cards and a confirmed "hum" and "serenade" (a teach costs seconds: once)."""
    runtime = _runtime_with_catalogue(engine, tmp_path_factory.mktemp("contract"))
    for req in (HUM, SERENADE):
        _teach(runtime, req)[1]()
    return runtime


def _uses(runtime, verb):
    """The standing of a taught verb's forms, summed (each form is its own construction)."""
    forms = [s.standing for s in runtime.store.all() if s.kind == "taught" and s.form[0] == verb]
    return SimpleNamespace(**{f: sum(getattr(st, f) for st in forms) for f in ("uses_ok", "uses_failed", "uses_world")})


def _tools(plan):
    return [row["tool"] for row in plan.licence or []]


@pytest.mark.needs_fcg
def test_teach_and_confirm_hum(engine, tmp_path):
    runtime = _runtime_with_catalogue(engine, tmp_path)
    draft, confirm = _teach(runtime, HUM)
    unread = _plan(runtime, "hum a tune", speaker="Milo")  # a draft is the teacher's alone
    assert unread.status == "unread" and unread.licence is None and unread.tool_steps == []
    slots = {p["slot"]: p for spec in draft.specs for p in spec.form if isinstance(p, dict)}
    assert slots["x"]["constraint"] == "tune" and slots["s"]["constraint"] == "instrument"  # from the tool's params
    confirm()
    assert _plan(runtime, "hum a tune", speaker="Milo").status == "needs_tools"


@pytest.mark.needs_fcg
def test_hum_a_tune_is_licensed_and_commits(world):
    plan = _plan(world, "hum a tune", chat_id="hum")
    assert plan.status == "needs_tools", plan.diagnostics
    [step] = plan.tool_steps
    assert step["tool"] == "make_sound" and step["args"] == {"pattern": "tune", "source": "instrument"}
    assert step["actor"] == "Ana" and step["needs_confirmation"] is False  # the speaker's own move: consent is the turn
    assert {"param": "pattern", "constraint": "tune", "result": "pass"} in step["checks"]
    assert _tools(plan) == ["make_sound"]
    before = _uses(world, "hum").uses_ok
    done = asyncio.run(world.resume(plan.plan_id, [_result(step, value="a tune on the keyboard")]))
    assert done.status == "answered" and done.tool_steps == [] and done.failure is None
    assert _tools(done) == ["make_sound"] and _uses(world, "hum").uses_ok == before + 1


@pytest.mark.needs_fcg
def test_a_direct_catalogue_card_turn_records_its_standing_without_a_diagnostic(world):
    """An attached catalogue card is not a stored row: the turn's standing write skips it instead of failing whole."""
    plan = _plan(world, "play Greensleeves on the lute", chat_id="direct")
    assert plan.status == "needs_tools", plan.diagnostics
    [step] = plan.tool_steps
    assert step["tool"] == "make_sound"
    done = asyncio.run(world.resume(plan.plan_id, [_result(step, value="Greensleeves on the lute")]))
    assert done.status == "answered", done.diagnostics
    assert not [d for d in [*plan.diagnostics, *done.diagnostics] if "standing not recorded" in d]


@pytest.mark.needs_fcg
def test_a_live_everyone_distributes_unless_it_fills_a_many_parameter(world):
    """A live "everyone" distributes over the turn's procedure as a taught one does: give runs once per participant,
    each with one person. Read into a parameter that takes several values (make's material) it stays one list. Read
    into both kinds (a taught "treat {R}" makes tea from R and gives cake to R) it is a repair: say it in two clauses."""
    plan = _plan(world, "give tea to everyone", chat_id="everyone")
    assert plan.status == "needs_tools", plan.diagnostics
    [first] = plan.tool_steps
    assert first["tool"] == "give" and first["args"] == {"item": "tea", "person": "Ana"}, first
    second = asyncio.run(world.resume(plan.plan_id, [_result(first, value="given")]))
    [then] = second.tool_steps
    assert then["tool"] == "give" and then["args"] == {"item": "tea", "person": "Milo"}, then
    done = asyncio.run(world.resume(plan.plan_id, [_result(then, value="given")]))
    assert done.status == "answered" and done.tool_steps == [], done.diagnostics
    made = _plan(world, "make tea from everyone", chat_id="everyone-many")
    assert made.status == "needs_tools", made.diagnostics
    [step] = made.tool_steps
    assert step["tool"] == "make" and step["args"] == {"item": "tea", "material": ["Ana", "Milo"]}, step
    _teach(world, TeachRequest(["treat {R}"], ["make tea from {R}", "give cake to {R}"], {}, "Ana", "c1"))[1]()
    both = _plan(world, "treat everyone", chat_id="everyone-both")
    assert both.status == "repair" and both.tool_steps == [] and sorted(_tools(both)) == ["give", "make"], both
    assert any("'everyone' fills both" in d and "two clauses" in d for d in both.diagnostics), both.diagnostics


@pytest.mark.needs_fcg
def test_composed_phrase_replays_with_identical_args(world):
    """A taught "brew" whose give holds make's result (a structured, non-ASCII value) and whose make holds the list of
    both takes' results: re-planned after a crash with the first three steps committed, give is issued with canonically
    identical args (the list round-trips through `committed`); with give committed too, nothing is issued. Brewed for
    everyone, each member issues identical takes: `committed` in issue order resolves each member's own results."""
    brew = TeachRequest(
        ["brew {X} for {R}", "brew {X}"],
        ["take water", "take herb", "make {X} from them", "give {X} to {R}"],
        {"R": "me"},
        "Ana",
        "c1",
    )
    _teach(world, brew)[1]()
    canon = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)  # noqa: E731
    plan = _plan(world, "brew tea for Ana", chat_id="brew")
    assert plan.status == "needs_tools", plan.diagnostics
    values = {"take": ["water#1", "herb#2"], "make": {"id": "tea#3", "name": "thé"}}
    issued = []
    while plan.tool_steps:
        [step] = plan.tool_steps
        issued.append({"tool": step["tool"], "args": step["args"]})
        value = values[step["tool"]].pop(0) if step["tool"] == "take" else values.get(step["tool"], "given")
        plan = asyncio.run(world.resume(plan.plan_id, [_result(step, value=value)]))
    assert [i["tool"] for i in issued] == ["take", "take", "make", "give"], issued
    assert issued[2]["args"]["material"] == ["water#1", "herb#2"], issued[2]  # the gather's list, in body order
    assert issued[3]["args"]["item"] == {"id": "tea#3", "name": "thé"}, issued[3]
    committed = [
        {**row, "value": v}
        for row, v in zip(issued[:3], ["water#1", "herb#2", {"id": "tea#3", "name": "thé"}], strict=True)
    ]
    again = asyncio.run(world.plan(_turn("brew tea for Ana", chat_id="brew-again"), committed=committed))
    assert again.status == "needs_tools", again.diagnostics
    [give] = again.tool_steps
    assert give["tool"] == "give" and canon(give["args"]) == canon(issued[3]["args"]), give
    last = {**issued[3], "value": "given"}
    done = asyncio.run(world.plan(_turn("brew tea for Ana", chat_id="brew-done"), committed=[*committed, last]))
    assert done.tool_steps == [] and done.status == "answered", done.diagnostics

    plan = _plan(world, "brew tea for everyone", chat_id="brew-all")  # participants Ana and Milo
    rows = []
    while plan.tool_steps:
        [step] = plan.tool_steps
        n = len(rows) + 1
        value = {"id": f"tea#{n}", "name": "thé"} if step["tool"] == "make" else f"{step['tool']}#{n}"
        rows.append({"tool": step["tool"], "args": step["args"], "value": value})
        plan = asyncio.run(world.resume(plan.plan_id, [_result(step, value=value)]))
    assert [r["tool"] for r in rows] == ["take", "take", "make", "give"] * 2, rows
    assert rows[0]["args"] == rows[4]["args"] and rows[6]["args"]["material"] == ["take#5", "take#6"], rows
    again = asyncio.run(world.plan(_turn("brew tea for everyone", chat_id="brew-all-again"), committed=rows[:6]))
    [make] = again.tool_steps  # Milo's make, with Milo's takes: nothing committed is issued again
    assert make["tool"] == "make" and canon(make["args"]) == canon(rows[6]["args"]), make
    done = asyncio.run(world.plan(_turn("brew tea for everyone", chat_id="brew-all-done"), committed=rows))
    assert done.tool_steps == [] and done.status == "answered", done.diagnostics


@pytest.mark.needs_fcg
def test_hum_a_kettle_rides_a_failed_check_and_the_game_decides(world):
    """A lexical (WordNet) constraint never blocks: the step goes out with the check on it and the game decides."""
    plan = _plan(world, "hum a kettle", chat_id="kettle")
    assert plan.status == "needs_tools", plan.diagnostics
    [step] = plan.tool_steps
    assert {"param": "pattern", "constraint": "tune", "result": "fail"} in step["checks"]
    assert _tools(plan) == ["make_sound"]
    refused = asyncio.run(world.resume(plan.plan_id, [_result(step, "declined", detail="a kettle is not a tune")]))
    assert refused.status == "repair" and refused.licence == plan.licence
    assert refused.failure == {
        "step_id": step["step_id"],
        "tool": "make_sound",
        "kind": "declined",
        "detail": "a kettle is not a tune",
    }


@pytest.mark.needs_fcg
def test_foreign_result_is_rejected(world):
    plan = _plan(world, "hum a tune", chat_id="foreign")
    [step] = plan.tool_steps
    for forged in ({"tool": "give"}, {"args": {"pattern": "tune", "source": "kettle"}}):
        with pytest.raises(ResumeError, match="is not for"):
            asyncio.run(world.resume(plan.plan_id, [{**_result(step, value="done"), **forged}]))
        kept = world._get(plan.plan_id).plan  # a refusal changes nothing
        assert kept.status == "needs_tools" and kept.tool_steps == plan.tool_steps and kept.failure is None
    reordered = {**_result(step, value="done"), "args": dict(reversed(list(step["args"].items())))}
    assert asyncio.run(world.resume(plan.plan_id, [reordered])).status == "answered"  # key order is moot


@pytest.mark.needs_fcg
def test_missing_instrument_repairs_with_licence_and_reason(world):
    plan = _plan(world, "hum a tune", chat_id="missing")
    [step] = plan.tool_steps
    standing = _uses(world, "hum")
    world_before, failed_before, ok_before = standing.uses_world, standing.uses_failed, standing.uses_ok
    out = asyncio.run(world.resume(plan.plan_id, [_result(step, "precondition_failed", detail="no instrument here")]))
    assert out.status == "repair" and out.tool_steps == [] and out.licence == plan.licence is not None
    assert out.failure == {
        "step_id": step["step_id"],
        "tool": "make_sound",
        "kind": "precondition_failed",
        "detail": "no instrument here",
    }
    assert out.diagnostics == [f"tool step {step['step_id']} (make_sound) precondition_failed: no instrument here"]
    standing = _uses(world, "hum")  # a refusal the room caused is world, never the construction's failure
    assert (standing.uses_world, standing.uses_failed, standing.uses_ok) == (world_before + 1, failed_before, ok_before)
    with pytest.raises(ResumeError):  # a repair waits on no tool
        asyncio.run(world.resume(plan.plan_id, [_result(step, value="again")]))


@pytest.mark.needs_fcg
def test_serenade_partial_failure_retries_only_the_failed_step(world, monkeypatch):
    plan = _plan(world, "serenade Ana", chat_id="serenade")
    assert plan.status == "needs_tools", plan.diagnostics
    [sound] = plan.tool_steps
    before = _uses(world, "serenade")
    base = (before.uses_world, before.uses_ok, before.uses_failed)
    assert sound["tool"] == "make_sound" and sorted(_tools(plan)) == ["give", "make_sound"]
    with pytest.raises(ResumeError):  # nothing failed yet
        asyncio.run(world.retry(plan.plan_id))
    second = asyncio.run(world.resume(plan.plan_id, [_result(sound, value="a tune")]))
    [give] = second.tool_steps
    assert give["tool"] == "give" and give["args"] == {"item": "flower", "person": "Ana"}
    refused = asyncio.run(world.resume(plan.plan_id, [_result(give, "precondition_failed", detail="no flower")]))
    assert refused.status == "repair" and refused.licence == plan.licence and refused.failure["tool"] == "give"
    record = world._get(plan.plan_id)
    held = (dict(record.values), list(record.queries), list(record.entities), record.referent, dict(record.resumed))

    real_memory = world._memory

    def memory_that_logs(turn):
        memory, store_memory = real_memory(turn)
        store_memory.queries.append({"op": "marker"})  # this replay leaves a trace on the record...
        return memory, store_memory

    def fails_late(*_args):
        raise RuntimeError("the replay died")  # ...and then dies

    monkeypatch.setattr(world, "_memory", memory_that_logs)
    monkeypatch.setattr("memory_kernel.runtime.api._resolved_entities", fails_late)
    with pytest.raises(RuntimeError, match="replay died"):
        asyncio.run(world.retry(plan.plan_id))
    monkeypatch.undo()
    assert (
        dict(record.values),
        list(record.queries),
        list(record.entities),
        record.referent,
        dict(record.resumed),
    ) == held
    assert record.plan is refused and record.plan.failure is not None and record.recorded  # restored whole
    again = asyncio.run(world.retry(plan.plan_id))
    assert (
        again.status == "needs_tools" and again.failure is None and again.tool_steps == [give]
    )  # make_sound stays done
    assert again.licence == refused.licence
    done = asyncio.run(world.resume(plan.plan_id, [_result(give, value="a flower")]))
    assert done.status == "answered"
    after = _uses(world, "serenade")  # the refusal and the retried success are two attempts, each recorded (R3)
    assert (after.uses_world, after.uses_ok, after.uses_failed) == (base[0] + 1, base[1] + 1, base[2])
    with pytest.raises(ResumeError):  # an answered plan has nothing to retry
        asyncio.run(world.retry(plan.plan_id))


@pytest.mark.needs_fcg
def test_a_committed_step_then_an_unresolved_one_repairs_keeping_the_licence(engine, tmp_path):
    """ "pass the tea to Y" takes, then gives to the entity Y names. With the take committed in the world, a Y the memory
    knows resumes at the give (the take never replays); a Y that names nobody is a repair that keeps the licence."""
    runtime = _tool_runtime(engine, tmp_path, FakeGraph([ANA], []))
    take = {"tool": "take", "args": {"item": "tea"}, "value": "taken"}
    plan = asyncio.run(runtime.plan(_turn("pass the tea to Ana"), committed=[take]))
    assert plan.status == "needs_tools" and [s["tool"] for s in plan.tool_steps] == ["give"]
    assert any(step["op"] == "take" and step["reused"] for step in plan.trace["steps"])
    other = {**take, "args": {"item": "coffee"}}  # other args: not this step
    fresh = asyncio.run(runtime.plan(_turn("pass the tea to Ana", chat_id="d"), committed=[other]))
    assert [s["tool"] for s in fresh.tool_steps] == ["take"]
    nobody = asyncio.run(runtime.plan(_turn("pass the tea to Brask Labs", chat_id="e"), committed=[take]))
    assert nobody.status == "repair" and nobody.tool_steps == [], nobody.diagnostics
    assert nobody.licence == TOOLS_LICENCE  # a program that names a tool carries its licence on every outcome


@pytest.mark.needs_fcg
def test_unknown_verb_is_unread_with_no_tool(world):
    plan = _plan(world, "juggle the kettles", chat_id="juggle")
    assert plan.status == "unread" and plan.tool_steps == [] and plan.licence is None and plan.failure is None
    assert plan.to_json()["failure"] is None


@pytest.mark.needs_fcg
def test_same_flow_over_http_with_the_catalogue(tmp_path):
    from fastapi.testclient import TestClient

    from memory_kernel.runtime.api import Runtime
    from memory_kernel.runtime.catalogue import parse_catalogue
    from memory_kernel.runtime.engine import RuntimeEngine
    from memory_kernel.runtime.lexicon import Lexicon, QuestionMap
    from memory_kernel.serve import http
    from memory_kernel.serve.runtime import RuntimeSettings
    from unit.runtime.memory_writer import MemoryWriter, memory_store
    from unit.serve.fakes import FakeRegistry

    writer = MemoryWriter()  # the "graph": the catalogue is put through the route, the runtime reads it back
    data_dir = _data_dir()
    engines: list[RuntimeEngine] = []

    def engine_factory(port):
        engines.append(RuntimeEngine(port, data_dir, out=tmp_path / "fcg" / str(len(engines))))
        return engines[-1]

    def runtime_factory(bonfire_id, meta, engine):
        stored = writer.catalogue()
        cat = parse_catalogue(stored.encode(), "catalogue") if stored else None
        return Runtime(
            bonfire_id, engine, FakeGraph([ANA], []), default_registry(cat.tools if cat else ()),
            store=memory_store(writer=writer), lexicon=Lexicon([], QuestionMap.load()),
            cards=cat.cards if cat else (),
        )  # fmt: skip

    app = http.create_app(
        FakeRegistry().add("ember"),
        runtime_factory=runtime_factory,
        engine_factory=engine_factory,
        writer_factory=lambda bonfire_id: writer,
        runtime_settings=RuntimeSettings(max_runtime_engines=1, runtime_ports="9726-9726", teach_wait_ms=30000),
    )
    base, turn = "/v1/bonfires/ember", {"speaker": "Ana", "participants": ["Ana", "Milo"], "chat_id": "c1"}
    try:
        with TestClient(app) as client:
            assert client.get(f"{base}/catalogue").json() == {"sha256": None, "tools": []}  # none put yet
            put = client.put(f"{base}/catalogue", content=json.dumps(CATALOGUE))
            assert put.status_code == 200 and put.json()["tools"] == CATALOGUE["tools"]
            got = client.get(f"{base}/catalogue").json()
            assert got["sha256"] == put.json()["sha256"] and got["tools"] == CATALOGUE["tools"]
            draft = client.post(f"{base}/teach", json={"forms": HUM.forms, "body": HUM.body, "defaults": HUM.defaults,
                                                        "teacher": "Ana", "chat_id": "c1"})  # fmt: skip
            assert draft.status_code == 200, draft.text
            did = draft.json()["draft_id"]
            assert client.post(f"{base}/teach/{did}/confirm", json={"by": "Ana"}).status_code == 200

            plan = client.post(f"{base}/plan", json=turn | {"text": "hum a tune"}).json()
            assert plan["status"] == "needs_tools", plan["diagnostics"]
            [step] = plan["tool_steps"]
            row = {k: step[k] for k in ("step_id", "tool", "args")}
            assert [r["tool"] for r in plan["licence"]] == ["make_sound"] and step["needs_confirmation"] is False
            refused = client.post(f"{base}/plan/{plan['plan_id']}/resume",
                                  json={"results": [row | {"verdict": "precondition_failed", "value": None, "detail": "no instrument"}]})  # fmt: skip
            assert refused.json()["status"] == "repair" and refused.json()["failure"]["kind"] == "precondition_failed"
            again = client.post(f"{base}/plan/{plan['plan_id']}/retry").json()
            assert again["status"] == "needs_tools" and again["tool_steps"] == [step]
            done = client.post(f"{base}/plan/{plan['plan_id']}/resume",
                               json={"results": [row | {"verdict": "committed", "value": "a tune", "detail": None}]})  # fmt: skip
            assert done.json()["status"] == "answered"
            assert client.post(f"{base}/plan/{plan['plan_id']}/retry").status_code == 409
    finally:
        for engine in engines:
            engine.stop()
    shutil.rmtree(tmp_path / "fcg", ignore_errors=True)
