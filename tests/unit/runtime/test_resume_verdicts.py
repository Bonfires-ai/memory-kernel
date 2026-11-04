"""resume verdicts: a refused effect keeps the plan's licence and carries a structured failure; every result row is
bound to its step's tool and args; each of the game's verdict kinds is structured."""

import asyncio
import json
from dataclasses import asdict, replace

import pytest
from test_plan_api import ANA, FakeGraph, _data_dir, _plan, _result, _tool_runtime

from memory_kernel.runtime.api import VERDICTS, ResumeError

PORT = 9721  # the 1.4 resume-verdict tests' engine (9720 test_engine_search, 9722 test_store_teach)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("verdict-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


def _pending(engine, tmp_path):
    runtime = _tool_runtime(engine, tmp_path, FakeGraph([ANA], []))
    plan = _plan(runtime, "pass the tea to Ana")
    assert plan.status == "needs_tools", plan.diagnostics
    return runtime, plan


@pytest.mark.needs_fcg
def test_each_verdict_kind(engine, tmp_path):
    runtime, first = _pending(engine, tmp_path)
    assert VERDICTS == ("committed", "precondition_failed", "tool_error", "declined")
    committed = asyncio.run(runtime.resume(first.plan_id, [_result(first.tool_steps[0], value="taken")]))
    assert committed.status == "needs_tools" and committed.failure is None
    assert committed.tool_steps[0]["tool"] == "give"
    for verdict in VERDICTS[1:]:
        plan = _plan(runtime, "pass the tea to Ana", chat_id=verdict)
        out = asyncio.run(runtime.resume(plan.plan_id, [_result(plan.tool_steps[0], verdict, detail=verdict)]))
        assert out.status == "repair" and out.failure is not None and out.failure["kind"] == verdict
        assert out.licence == first.licence
    # several rows fail in one resume: the reported failure is the first failed row in REQUEST order
    plan = _plan(runtime, "pass the tea to Ana", chat_id="two-rows")
    record = runtime._get(plan.plan_id)
    twin = replace(record.pending[0], step_id=f"{plan.plan_id}:9:0")  # a second pending step, as a parallel program has
    record.pending = (*record.pending, twin)
    rows = [
        _result(asdict(twin), "tool_error", detail="first row"),
        _result(plan.tool_steps[0], "declined", detail="second row"),
    ]
    out = asyncio.run(runtime.resume(plan.plan_id, rows))
    assert out.failure is not None and out.failure["step_id"] == twin.step_id and out.failure["kind"] == "tool_error"
    assert [d.split(" ")[2] for d in out.diagnostics] == [twin.step_id, plan.tool_steps[0]["step_id"]]
    assert (
        runtime.store.get("card-pass").standing.uses_world == 4
        and runtime.store.get("card-pass").standing.uses_failed == 0
    )


@pytest.mark.needs_fcg
def test_verdict_detail_round_trips_unicode(engine, tmp_path):
    runtime, plan = _pending(engine, tmp_path)
    detail = "pas de thé ici ☕ — Zoë n'a rien « donné »"
    out = asyncio.run(runtime.resume(plan.plan_id, [_result(plan.tool_steps[0], "declined", detail=detail)]))
    assert out.failure is not None and out.failure["detail"] == detail
    wire = json.loads(json.dumps(out.to_json(), ensure_ascii=False))
    assert wire["failure"]["detail"] == detail and detail in wire["diagnostics"][0]


@pytest.mark.needs_fcg
def test_memory_only_repair_hands_off(engine, tmp_path):
    runtime = _tool_runtime(engine, tmp_path, FakeGraph([ANA], []))
    plan = _plan(runtime, "where does Zed Quill work?")
    assert plan.status == "repair" and plan.program and plan.licence is None


@pytest.mark.needs_fcg
def test_step_ids_differ_across_plans(engine, tmp_path):
    runtime = _tool_runtime(engine, tmp_path, FakeGraph([ANA], []))
    one = _plan(runtime, "pass the tea to Ana", chat_id="a")
    two = _plan(runtime, "pass the tea to Ana", chat_id="b")
    first, other = one.tool_steps[0]["step_id"], two.tool_steps[0]["step_id"]
    assert first.startswith(f"{one.plan_id}:") and other.startswith(f"{two.plan_id}:") and first != other


@pytest.mark.needs_fcg
def test_concurrent_resume_is_refused(engine, tmp_path):
    runtime, plan = _pending(engine, tmp_path)
    rows = [_result(plan.tool_steps[0], value="taken")]

    async def both():
        return await asyncio.gather(
            runtime.resume(plan.plan_id, rows), runtime.resume(plan.plan_id, rows), return_exceptions=True
        )

    outcomes = asyncio.run(both())
    applied = [o for o in outcomes if not isinstance(o, BaseException)]
    refused = [o for o in outcomes if isinstance(o, ResumeError)]
    assert len(applied) == 1 and len(refused) == 1 and applied[0].tool_steps[0]["tool"] == "give"
