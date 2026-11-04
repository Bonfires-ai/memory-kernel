"""``mk serve``'s runtime-grammar routes (plan, resume, teach, confirm, constructions) over a fake ``Runtime``: the
wire shapes, the status codes, the busy path while a bonfire teaches, the engine cap; the keyword glossary is current;
one engine-backed smoke (``needs_fcg``) over a fake graph, and ``mk plan``."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from memory_kernel.runtime.api import ResumeError, TurnPlan, UnknownPlan, unread_plan
from memory_kernel.runtime.constructions import CxnSpec
from memory_kernel.runtime.executor import Turn
from memory_kernel.runtime.teach import Draft, NotTeacher, TeachError, TeachRequest
from memory_kernel.serve import http
from memory_kernel.serve.runtime import BUSY_TEACHING, Runtimes, RuntimeSettings
from unit.runtime.memory_writer import MemoryWriter, memory_store
from unit.serve.fakes import FakeRegistry

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIRE = "ember"
OTHER = "cinder"
#: the redaction tests' secret (``test_http``): never in a reason or a log
SECRET = "sk-test-not-a-real-key"
SPEC = CxnSpec(id="brew.d1.0", kind="taught", unit="root", form=["brew ", {"slot": "x", "fills": "np"}], meaning=[])
#: what the live grammar reads a draft's own rendering as (1.5.1: the teach response carries it)
COMPETES = {
    "name": "make",
    "kind": "tool",
    "construction": "frame.make.make",
    "roles": {"item": None},
    "complete": False,
    "analysis": 0,
    "composed_with": None,
}


def _answered(turn: Turn, bonfire_id: str) -> TurnPlan:
    plan = unread_plan(bonfire_id, turn, [])
    return dataclasses.replace(plan, status="answered", answer=[{"uuid": "u2", "name": "Zorvex"}], licence=[])


class FakeEngine:
    def __init__(self, port: int, log: list[tuple[str, int]]) -> None:
        self.port, self.log, self.pid = port, log, None

    def stop(self) -> None:  # a real Runtime's stop() stops its engine
        self.pid = None


class FakeStore:
    def all(self) -> list[CxnSpec]:
        return [SPEC]


class FakeRuntime:
    """Records every call; ``gate`` holds a teach until set; ``start`` / ``stop`` log the engine's port."""

    def __init__(self, bonfire_id: str, engine: FakeEngine, gate: asyncio.Event | None = None) -> None:
        self.bonfire_id, self.engine, self.gate = bonfire_id, engine, gate
        self.turns: list[Turn] = []
        self.committed: list[Any] = []
        self.contexts: list[Any] = []
        self.teaches: list[TeachRequest] = []
        self.store = FakeStore()

    def use_engine(self, engine: FakeEngine) -> None:
        self.engine = engine

    def start(self) -> None:
        self.engine.pid = 1
        self.engine.log.append(("start", self.engine.port))

    def stop(self) -> None:
        if self.engine.pid is not None:
            self.engine.log.append(("stop", self.engine.port))
        self.engine.pid = None

    async def plan(self, turn: Turn, committed: Any = (), context: Any = ()) -> TurnPlan:
        self.turns.append(turn)
        self.committed.append(committed)
        self.contexts.append(context)
        if turn.text == "nothing reads this":
            return unread_plan(self.bonfire_id, turn, ["no construction read the turn"])
        return _answered(turn, self.bonfire_id)

    async def resume(self, plan_id: str, results: list[dict[str, Any]]) -> TurnPlan:
        if plan_id == "gone":
            raise UnknownPlan("plan 'gone' expired 900 s after its last use")
        if plan_id == "never":
            raise UnknownPlan("no plan 'never': this runtime never made it")
        if plan_id == "done":
            raise ResumeError("plan done is answered: it waits on no tool step")
        turn = Turn("brew a tea", "Ana", None, (), "c1", None, "2026-10-18T15:30:00+00:00")
        return _answered(turn, self.bonfire_id)

    async def retry(self, plan_id: str) -> TurnPlan:
        if plan_id != "failed":
            raise ResumeError(f"plan {plan_id} has no failed step to retry")
        return _answered(Turn("brew a tea", "Ana", None, (), "c1", None, "2026-10-18T15:30:00+00:00"), self.bonfire_id)

    async def teach(self, req: TeachRequest) -> Draft:
        self.teaches.append(req)
        if self.gate is not None:
            await self.gate.wait()
        if req.forms == ["{X}!"]:
            return Draft("", [], ["form '{X}!' has no literal"])
        return Draft("d1", [SPEC], [], {"brew {X}": [COMPETES]})

    async def confirm(self, draft_id: str, by: str) -> list[CxnSpec]:
        if by != "Ana":
            raise NotTeacher(f"draft {draft_id} is confirmed by its teacher only ('Ana'), not {by!r}")
        if draft_id != "d1":
            raise TeachError(f"no staged draft {draft_id!r}")
        return [dataclasses.replace(SPEC, status="live")]


class Factory:
    def __init__(self) -> None:
        self.made: dict[str, FakeRuntime] = {}
        self.engines: list[tuple[str, int]] = []
        self.gate: asyncio.Event | None = None

    def runtime(self, bonfire_id: str, fire: Any, engine: FakeEngine) -> FakeRuntime:
        self.made[bonfire_id] = FakeRuntime(bonfire_id, engine, self.gate)
        return self.made[bonfire_id]

    def engine(self, port: int) -> FakeEngine:
        return FakeEngine(port, self.engines)


def _fires(tmp_path: Path, *names: str) -> FakeRegistry:
    """The bonfires these routes serve: a registry that lists ``names`` (no directory)."""
    registry = FakeRegistry()
    for name in names:
        registry.add(name)
    return registry


@pytest.fixture
def factory() -> Factory:
    return Factory()


@pytest.fixture
def client(tmp_path: Path, factory: Factory):
    app = http.create_app(_fires(tmp_path, FIRE), runtime_factory=factory.runtime, engine_factory=factory.engine)
    with TestClient(app) as c:
        yield c


def _raw(client: TestClient, path: str, body: Any) -> Any:
    """POST ``body`` as JSON text with ASCII escapes: a lone surrogate travels as ``\\ud800``, as a client may send it."""
    return client.post(path, content=json.dumps(body), headers={"content-type": "application/json"})


TURN = {"text": "where does Kira Dace work?", "speaker": "Ana", "participants": ["Ana", "Milo"], "chat_id": "c1"}


def test_plan_returns_the_turn_plan_and_fills_the_turn_time(client, factory):
    response = client.post(f"/v1/bonfires/{FIRE}/plan", json=TURN)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered" and body["licence"] == [] and body["answer"][0]["name"] == "Zorvex"
    turn = factory.made[FIRE].turns[0]
    assert turn.participants == ("Ana", "Milo") and turn.chat_id == "c1" and turn.addressee is None
    assert datetime.fromisoformat(turn.at).utcoffset() is not None  # the server's UTC time, ISO 8601
    given = client.post(f"/v1/bonfires/{FIRE}/plan", json=TURN | {"at": "2026-10-18T09:00:00+02:00"})
    assert given.json()["turn"]["at"] == "2026-10-18T09:00:00+02:00"
    unread = client.post(f"/v1/bonfires/{FIRE}/plan", json=TURN | {"text": "nothing reads this"}).json()
    assert unread["status"] == "unread" and unread["licence"] is None  # JSON null: fall back with today's tools


@pytest.mark.parametrize(
    "body",
    [
        {"speaker": "Ana", "chat_id": "c1"},
        TURN | {"text": "  "},
        TURN | {"participants": ["Ana", 7]},
        TURN | {"participants": "Ana"},
        TURN | {"speaker": {"uuid": "u1"}},
        TURN | {"addressee": 3},
        {k: v for k, v in TURN.items() if k != "chat_id"},
        TURN | {"chat_id": 12},
        TURN | {"at": "yesterday"},
        TURN | {"window": [[{"uuid": "u1"}]]},
        [TURN],
        TURN | {"text": "where does \ud800 work?"},
        TURN | {"speaker": "An\udc00a"},
        TURN | {"participants": ["Ana", "\ud800"]},
        TURN | {"text": "x" * 4001},
        TURN | {"participants": ["p"] * 257},
        TURN | {"speaker": "a" * 257},
        TURN | {"chat_id": "c" * 257},
        TURN | {"message_id": "m" * 257},
    ],
)
def test_a_bad_turn_body_is_a_422(client, factory, body):
    assert _raw(client, f"/v1/bonfires/{FIRE}/plan", body).status_code == 422
    assert FIRE not in factory.made or factory.made[FIRE].turns == []


def test_every_route_answers_404_for_an_unknown_bonfire(client):
    teach = {"forms": ["brew {X}"], "body": ["make {X}"], "teacher": "Ana", "chat_id": "c1"}
    assert client.post("/v1/bonfires/nope/plan", json=TURN).status_code == 404
    assert client.post("/v1/bonfires/nope/plan/p1/resume", json={"results": []}).status_code == 404
    assert client.post("/v1/bonfires/nope/teach", json=teach).status_code == 404
    assert client.post("/v1/bonfires/nope/teach/d1/confirm", json={"by": "Ana"}).status_code == 404
    assert client.get("/v1/bonfires/nope/constructions").status_code == 404


def test_resume_shapes_and_errors(client):
    ok = [{"step_id": "0:0", "tool": "take", "args": {}, "verdict": "committed", "value": "done"}]
    no_plan = client.post(f"/v1/bonfires/{FIRE}/plan/p1/resume", json={"results": ok})
    assert no_plan.status_code == 404  # no runtime yet: no plan was ever made here
    client.post(f"/v1/bonfires/{FIRE}/plan", json=TURN)
    assert client.post(f"/v1/bonfires/{FIRE}/plan/p1/resume", json={"results": ok}).json()["status"] == "answered"
    expired = client.post(f"/v1/bonfires/{FIRE}/plan/gone/resume", json={"results": ok})
    unknown = client.post(f"/v1/bonfires/{FIRE}/plan/never/resume", json={"results": ok})
    assert expired.status_code == unknown.status_code == 404
    assert "expired" in expired.json()["reason"] and "never made" in unknown.json()["reason"]
    assert client.post(f"/v1/bonfires/{FIRE}/plan/done/resume", json={"results": ok}).status_code == 409
    retried = client.post(f"/v1/bonfires/{FIRE}/plan/failed/retry")
    assert retried.status_code == 200 and retried.json()["status"] == "answered"
    assert client.post(f"/v1/bonfires/{FIRE}/plan/done/retry").status_code == 409
    assert client.post(f"/v1/bonfires/{FIRE}/plan/never/retry").status_code == 409  # the fake raises ResumeError
    done = [{"tool": "take", "args": {"x": 1}, "value": "done"}]
    assert client.post(f"/v1/bonfires/{FIRE}/plan", json={**TURN, "committed": done}).status_code == 200
    assert client.app.state.pool.runtimes._slots[FIRE].runtime.committed[-1] == done
    for bad_committed in (
        "x",
        [{"tool": 1, "args": {}}],
        [{"tool": "take", "args": []}],
        [{"tool": "take", "args": {}}],
    ):
        assert _raw(client, f"/v1/bonfires/{FIRE}/plan", {**TURN, "committed": bad_committed}).status_code == 422
    for bad in (
        {},
        {"results": "x"},
        {"results": []},
        {"results": [{"step_id": 1, "tool": "take", "args": {}, "verdict": "committed"}]},
        {"results": [{"step_id": "0:0", "ok": True, "value": "old shape"}]},
        {"results": [{"step_id": "0:0", "tool": "take", "args": {}, "verdict": "maybe"}]},
        {"results": [{"step_id": "0:0", "tool": "take", "args": {}, "verdict": "committed", "value": "tea \ud800"}]},
        {"results": [{"step_id": "s" * 257, "tool": "take", "args": {}, "verdict": "committed"}]},
    ):
        assert _raw(client, f"/v1/bonfires/{FIRE}/plan/p1/resume", bad).status_code == 422


FRAME = {
    "name": "take",
    "kind": "tool",
    "construction": "frame.take.take",
    "roles": {"ticket": None},
    "complete": False,
    "analysis": 0,
    "composed_with": None,
}


def test_plan_route_validates_context(client, factory):
    plan = f"/v1/bonfires/{FIRE}/plan"
    assert client.post(plan, json={**TURN, "context": [FRAME]}).status_code == 200
    assert factory.made[FIRE].contexts[-1] == [FRAME]
    done = [{"tool": "take", "args": {"ticket": "JUJ-3"}, "value": "ok"}]
    both = client.post(plan, json={**TURN, "context": [FRAME, FRAME | {"name": "put"}], "committed": done})
    assert both.status_code == 200 and factory.made[FIRE].committed[-1] == done
    assert [f["name"] for f in factory.made[FIRE].contexts[-1]] == ["take", "put"]
    asked = len(factory.made[FIRE].turns)
    for bad in (
        [FRAME] * 9,
        [{k: v for k, v in FRAME.items() if k != "name"}],
        [FRAME | {"name": ""}],
        [FRAME | {"kind": "verb"}],
        [FRAME | {"roles": ["ticket"]}],
        [FRAME | {"roles": {"ticket": 3}}],
        [FRAME | {"complete": "no"}],
        [FRAME | {"colour": "red"}],
        [FRAME | {"roles": {"ticket": "x" * 4001}}],
        ["take"],
        "take",
    ):
        assert _raw(client, plan, {**TURN, "context": bad}).status_code == 422, bad
    assert len(factory.made[FIRE].turns) == asked  # a refused body never reaches the runtime


def test_teach_confirm_and_the_constructions_listing(client):
    teach = {"forms": ["brew {X}"], "body": ["make {X}"], "teacher": "Ana", "chat_id": "c1"}
    draft = client.post(f"/v1/bonfires/{FIRE}/teach", json=teach)
    assert draft.status_code == 200
    assert draft.json() == {
        "draft_id": "d1",
        "specs": [SPEC.to_json()],
        "problems": [],
        "competes": {"brew {X}": [COMPETES]},
    }
    refused = client.post(f"/v1/bonfires/{FIRE}/teach", json=teach | {"forms": ["{X}!"]})
    assert refused.status_code == 200 and refused.json()["draft_id"] == "" and refused.json()["problems"]
    for bad in (
        teach | {"forms": "brew {X}"},
        teach | {"defaults": {"R": 1}},
        teach | {"replace": "yes"},
        {},
        teach | {"forms": ["brew {X} \ud800"]},
        teach | {"teacher": "\udc00"},
        teach | {"forms": ["brew {X}"] * 33},
        teach | {"body": ["make " + "x" * 500]},
        teach | {"teacher": "t" * 257},
    ):
        assert _raw(client, f"/v1/bonfires/{FIRE}/teach", bad).status_code == 422
    assert client.post(f"/v1/bonfires/{FIRE}/teach/d1/confirm", json={"by": "Milo"}).status_code == 403
    assert client.post(f"/v1/bonfires/{FIRE}/teach/d9/confirm", json={"by": "Ana"}).status_code == 409
    assert client.post(f"/v1/bonfires/{FIRE}/teach/d1/confirm", json={}).status_code == 422
    assert _raw(client, f"/v1/bonfires/{FIRE}/teach/d1/confirm", {"by": "\ud800"}).status_code == 422
    live = client.post(f"/v1/bonfires/{FIRE}/teach/d1/confirm", json={"by": "Ana"})
    assert live.status_code == 200 and live.json()["specs"][0]["status"] == "live"
    rows = client.get(f"/v1/bonfires/{FIRE}/constructions")
    assert rows.status_code == 200 and rows.json()["constructions"] == [SPEC.to_json()]


def test_teach_route_accepts_structured_steps(client, factory):
    """1.5.2 (C3): a body entry may be a call ``{"call": <tool>, "args": {<param>: <value>}}``; it reaches the runtime
    as sent. A call that is not that shape, or past the caps, is a 422 (what the call says is the teach's to check)."""
    step = {"call": "web_search", "args": {"query": "{X}"}}
    teach = {"forms": ["sitrep {X}"], "body": [step, "write it"], "teacher": "Ana", "chat_id": "c1"}
    draft = client.post(f"/v1/bonfires/{FIRE}/teach", json=teach)
    assert draft.status_code == 200 and draft.json()["draft_id"] == "d1"
    assert [req.body for req in factory.made[FIRE].teaches] == [[step, "write it"]]
    for bad in (
        {"call": ""},
        {"call": "web_search", "args": {"query": 1}},
        {"call": "web_search", "args": ["{X}"]},
        {"call": "web_search", "with": {}},
        {"call": "web_search", "args": {"query": "q" * 501}},
        {"call": "w" * 257},
        {"call": "web_search", "args": {f"a{n}": "x" for n in range(33)}},
        ["web_search"],
    ):
        assert _raw(client, f"/v1/bonfires/{FIRE}/teach", teach | {"body": [bad]}).status_code == 422, bad
    assert len(factory.made[FIRE].teaches) == 1


def _runtimes(tmp_path: Path, factory: Factory, **settings: Any) -> tuple[Runtimes, Any]:
    pool = http.Pool(_fires(tmp_path, FIRE, OTHER))
    runtimes = Runtimes(
        pool, RuntimeSettings(**settings), runtime_factory=factory.runtime, engine_factory=factory.engine
    )
    return runtimes, pool


def _turn(text: str = "where does Kira Dace work?") -> Turn:
    return Turn(text, "Ana", "Milo", ("Ana", "Milo"), "c1", None, "2026-10-18T15:30:00+00:00")


def test_a_plan_while_the_bonfire_teaches_is_busy_and_a_resume_waits(tmp_path, factory):
    runtimes, pool = _runtimes(tmp_path, factory, resume_wait_s=0.05, teach_wait_ms=0)
    req = TeachRequest(["brew {X}"], ["make {X}"], {}, "Ana", "c1")

    async def scenario() -> None:
        gate = factory.gate = asyncio.Event()
        await runtimes.plan(FIRE, await pool.meta(FIRE), _turn())  # the runtime exists, its engine runs
        teaching = asyncio.create_task(runtimes.teach(FIRE, await pool.meta(FIRE), req))
        await asyncio.sleep(0)
        factory.made[FIRE].gate = gate
        status, busy = await runtimes.plan(FIRE, await pool.meta(FIRE), _turn())
        assert status == 200 and busy["status"] == "unread" and busy["licence"] is None
        assert busy["diagnostics"] == [BUSY_TEACHING] == ["runtime busy: teaching"]
        status, other = await runtimes.plan(OTHER, await pool.meta(OTHER), _turn())
        assert other["status"] == "answered"  # busy per bonfire only
        ok = [{"step_id": "0:0", "tool": "take", "args": {}, "verdict": "committed", "value": None}]
        status, body, headers = await runtimes.resume(FIRE, "p1", ok)
        assert status == 503 and headers["Retry-After"]
        gate.set()
        status, draft = await teaching
        assert status == 200 and draft["draft_id"] == "d1"
        status, body, _ = await runtimes.resume(FIRE, "p1", ok)
        assert status == 200
        status, after = await runtimes.plan(FIRE, await pool.meta(FIRE), _turn())
        assert after["status"] == "answered"
        await runtimes.close()

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize(
    ("wait_ms", "release_after", "status", "diagnostics"),
    [(2000, 0.1, "answered", None), (50, 0.4, "unread", [BUSY_TEACHING])],
)
def test_a_plan_waits_for_a_short_teach_and_past_the_bound_is_unread_busy(
    tmp_path, factory, wait_ms, release_after, status, diagnostics
):
    runtimes, pool = _runtimes(tmp_path, factory, teach_wait_ms=wait_ms)
    req = TeachRequest(["brew {X}"], ["make {X}"], {}, "Ana", "c1")

    async def scenario() -> None:
        gate = factory.gate = asyncio.Event()
        await runtimes.plan(FIRE, await pool.meta(FIRE), _turn())
        factory.made[FIRE].gate = gate
        teaching = asyncio.create_task(runtimes.teach(FIRE, await pool.meta(FIRE), req))
        await asyncio.sleep(0)
        asyncio.get_running_loop().call_later(release_after, gate.set)
        _, body = await runtimes.plan(FIRE, await pool.meta(FIRE), _turn())
        assert body["status"] == status and (diagnostics is None or body["diagnostics"] == diagnostics)
        assert (await teaching)[0] == 200
        await runtimes.close()

    asyncio.run(asyncio.wait_for(scenario(), 10))


CATALOGUE = {
    "schema_version": "kernel-catalogue/v1",
    "tools": [
        {
            "name": "ring",
            "input_schema": {"type": "object", "properties": {"bell": {"type": "string"}}},
            "x-side-effect": True,
            "x-args": {"bell": {"role": "theme"}},
            "x-forms": ["ring {bell}"],
        }
    ],
}


def test_past_the_engine_cap_the_least_recently_used_idle_engine_stops(tmp_path, factory):
    runtimes, pool = _runtimes(tmp_path, factory, max_runtime_engines=1, runtime_ports="9711-9712")

    async def scenario() -> None:
        await runtimes.plan(FIRE, await pool.meta(FIRE), _turn())
        await runtimes.plan(OTHER, await pool.meta(OTHER), _turn())
        assert factory.engines == [("start", 9711), ("stop", 9711), ("start", 9711)]
        assert factory.made[FIRE].engine.pid is None and factory.made[OTHER].engine.pid is not None
        status, again = await runtimes.plan(FIRE, await pool.meta(FIRE), _turn())
        assert again["status"] == "answered" and factory.made[OTHER].engine.pid is None
        await runtimes.close()
        assert all(r.engine.pid is None for r in factory.made.values())  # engines stop on shutdown

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_the_runtime_settings_are_read_from_their_env_names(monkeypatch):
    assert RuntimeSettings.ENV_PREFIX == "MK_SERVE_"
    assert RuntimeSettings.env_names() == {
        "max_runtime_engines": "MK_SERVE_MAX_RUNTIME_ENGINES",
        "runtime_ports": "MK_SERVE_RUNTIME_PORTS",
        "resume_wait_s": "MK_SERVE_RESUME_WAIT_S",
        "teach_wait_ms": "MK_SERVE_TEACH_WAIT_MS",
    }
    assert RuntimeSettings.from_env() == RuntimeSettings()
    assert RuntimeSettings().max_runtime_engines == 2
    assert RuntimeSettings().teach_wait_ms == 15000
    assert RuntimeSettings().runtime_ports == "9712-9719"  # 9711 is mk plan's (MK_RUNTIME_FCG_PORT's default)
    monkeypatch.setenv("MK_SERVE_MAX_RUNTIME_ENGINES", "1")
    monkeypatch.setenv("MK_SERVE_RUNTIME_PORTS", "9721-9722")
    monkeypatch.setenv("MK_SERVE_RESUME_WAIT_S", "5")
    monkeypatch.setenv("MK_SERVE_TEACH_WAIT_MS", "250")
    assert RuntimeSettings.from_env() == RuntimeSettings(1, "9721-9722", 5.0, 250)
    assert RuntimeSettings(max_runtime_engines=1, runtime_ports="9711").ports == (9711,)
    for bad in ("9600-9601", "9711-9900", "9712-9711", "97x1"):
        with pytest.raises(ValueError):
            RuntimeSettings(runtime_ports=bad)
    with pytest.raises(ValueError):
        RuntimeSettings(max_runtime_engines=3, runtime_ports="9711-9712")


def test_the_keyword_glossary_in_docs_is_generated_from_the_code(capsys):
    from memory_kernel.cli import main
    from memory_kernel.runtime.keywords import glossary

    text = glossary()
    assert "←)" not in text and "← )" not in text
    assert "### `invoke(step ← construction, slot…)`" in text  # the one keyword 1.4 adds
    assert (ROOT / "docs" / "keywords.md").read_text(encoding="utf-8") == text, (
        "regenerate: mk keywords > docs/keywords.md"
    )
    assert main(["keywords"]) == 0
    assert capsys.readouterr().out == text


# -- engine-backed (the pinned FCG Go) --------------------------------------------------------------------------------

PORT = 9727  # outside mk serve's runtime ports (9712-9719); 9720-9726 are the other engine test modules'


class FakeGraph:
    """find-entity by exact name; events by frame and participant; role-filler by (frame, role) pairs."""

    KIRA = {"uuid": "u1", "name": "Kira Dace", "labels": ["Entity", "Person"]}
    ZORVEX = {"uuid": "u2", "name": "Zorvex", "labels": ["Entity", "Organisation"]}

    async def rows(self, cypher: str, params: dict[str, Any]) -> list[dict]:
        if "n.name = $name" in cypher:
            return [dict(e) for e in (self.KIRA, self.ZORVEX) if e["name"] == params["name"]]
        if "$groups" in cypher:
            return [{"uuid": "e1", "name": "e1"}] if "work.01" in params["frames"] else []
        if "$pairs" in cypher:
            pairs = [tuple(p) for p in params["pairs"]]
            return [{**self.ZORVEX, "event": "e1"}] if any(r == "arg2" for _, r in pairs) else []
        return []


def _data_dir() -> Path:
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.mark.needs_fcg
def test_smoke_a_real_runtime_answers_a_plan_over_http(tmp_path):
    from memory_kernel.runtime.api import Runtime
    from memory_kernel.runtime.engine import RuntimeEngine
    from memory_kernel.runtime.lexicon import Lexicon, QuestionMap

    data_dir = _data_dir()
    engines: list[RuntimeEngine] = []

    def engine_factory(port: int) -> RuntimeEngine:
        engines.append(RuntimeEngine(port, data_dir, out=tmp_path / "runtime-fcg" / str(port)))
        return engines[-1]

    def runtime_factory(bonfire_id: str, fire: Any, engine: RuntimeEngine) -> Runtime:
        lexicon = Lexicon(["work.01"], QuestionMap.load())
        recall = app.state.pool.runtimes.recall_for(bonfire_id)  # serve's own recall: cache-only
        return Runtime(bonfire_id, engine, FakeGraph(), store=memory_store(), lexicon=lexicon, recall=recall)

    sent: list[str] = []
    speakers: list[str | None] = []

    async def context(bonfire_id: str, query: str, speaker: str | None = None, *, paid: bool = True):
        if paid:
            sent.append(query)
            raise AssertionError("the paid /context path was reached from a plan")
        speakers.append(speaker)
        return 200, http.degraded(None, "no cached context")

    app = http.create_app(
        _fires(tmp_path, FIRE),
        runtime_factory=runtime_factory,
        engine_factory=engine_factory,
        runtime_settings=RuntimeSettings(max_runtime_engines=1, runtime_ports=f"{PORT}-{PORT}"),
    )
    app.state.pool.context = context
    try:
        with TestClient(app) as client:
            body = client.post(f"/v1/bonfires/{FIRE}/plan", json=TURN).json()
            assert body["status"] == "answered", body["diagnostics"]
            assert body["answer"] == [{"uuid": "u2", "name": "Zorvex", "kinds": ["Organisation"]}]
            assert body["licence"] == [] and body["reading"]["complete"]
            nobody = client.post(f"/v1/bonfires/{FIRE}/plan", json=TURN | {"text": "where does Zed Quill work?"})
            repair = nobody.json()
            assert repair["status"] == "repair" and repair["licence"] is None and repair["answer"] is None
            assert "recall unavailable: no cached context" in repair["diagnostics"]
            assert "recall" not in repair["memory_results"] and sent == [] and speakers == ["Ana"]
    finally:
        for engine in engines:
            engine.stop()
    assert [e.pid for e in engines] == [None]


@pytest.mark.needs_fcg
@pytest.mark.needs_neo4j
def test_mk_plan_prints_the_turn_plan_and_names_its_reader(tmp_path, neo4j_target):
    from neo4j import GraphDatabase

    from memory_kernel.graph.state import meta_props, meta_row
    from memory_kernel.runtime.base import base_specs
    from memory_kernel.runtime.graph_store import NarrowWriter

    data_dir = _data_dir()
    group = neo4j_target.group
    meta = meta_props(
        bonfire_id=group,
        build_config={"sample_id": group},
        recipe_tag="t",
        extraction_model={"model": "m"},
        agent_handles=[],
        agent_handle_runs=[],
        chat_tagged=False,
        explicit_since=None,
        sessions=1,
        committed_run=1,
        generation="gen-1",
    )
    where = next(s for s in base_specs() if s.id == "where-do-x-v")
    taught = dataclasses.replace(
        where, id="where-do-x-v-today", kind="taught", status="live", form=[*where.form[:-1], "today?"]
    )
    with GraphDatabase.driver(neo4j_target.uri, auth=(neo4j_target.user, neo4j_target.password)) as driver:
        driver.execute_query("CREATE (m:BonfireMeta) SET m = $p", p=meta_row(group, meta)["props"])
        writer = NarrowWriter(driver, group)
        writer.put_phrases([taught])
        env = {
            **os.environ,
            "MK_DATA_DIR": str(data_dir),
            "MK_RUNTIME_FCG_PORT": str(PORT),
            "PYTHONPATH": str(ROOT / "src"),
            "NEO4J_PASSWORD": neo4j_target.password,
        }
        cmd = [sys.executable, "-m", "memory_kernel.cli", "plan", group]
        addr = ["--neo4j-uri", neo4j_target.uri, "--neo4j-user", neo4j_target.user]
        engine_dir = ["--engine-dir", str(tmp_path / "engine")]  # never the shared data dir
        before = [(p.id, p.standing) for p in writer.phrases()]
        out = subprocess.run(
            [*cmd, "where does Kira Dace work today?", "--speaker", "Ana", "--empty-graph", *addr, *engine_dir],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert out.returncode == 0, out.stderr
        plan = json.loads(out.stdout)
        assert plan["status"] == "repair" and plan["licence"] is None  # the empty reader: Kira Dace names nobody
        assert "names nobody" in " ".join(plan["diagnostics"])
        assert "reader: empty" in out.stderr
        assert taught.id in [card["id"] for card in plan["hand"]]  # the taught row read the turn ...
        assert [(p.id, p.standing) for p in writer.phrases()] == before  # ... and mk plan wrote nothing (no standing)
        source = tmp_path / "tools.json"
        source.write_text(json.dumps(CATALOGUE))
        install = [sys.executable, "-m", "memory_kernel.cli", "catalogue", "install", group]
        cat = subprocess.run([*install, str(source), *addr], capture_output=True, text=True, env=env)
        assert cat.returncode == 0, cat.stderr
        assert (
            json.loads(cat.stdout)["tools"] == ["ring"] and json.loads(writer.catalogue())["tools"][0]["name"] == "ring"
        )
        bad = tmp_path / "bad.json"
        bad.write_text("{}")
        refused = subprocess.run([*install, str(bad), *addr], capture_output=True, text=True, env=env)
        assert refused.returncode == 2 and "schema_version" in refused.stdout
        assert json.loads(writer.catalogue())["tools"][0]["name"] == "ring"  # a refused file changed nothing
        out = subprocess.run(
            [*cmd, "ring the bell", "--speaker", "Ana", "--empty-graph", *addr, *engine_dir],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert out.returncode == 0, out.stderr
        assert "card.ring.0" in [card["id"] for card in json.loads(out.stdout)["hand"]]  # mk plan reads the catalogue


def test_recall_in_serve_asks_the_caches_only(tmp_path, factory):
    """The runtime's recall reaches /context with paid calls off: a server whose paid path raises is never asked to
    pay, and its cache miss comes back as "no cached context" (the plan's diagnostic), never as a context."""
    from memory_kernel.serve.answer import ServeMiss

    asked: list[bool] = []
    questions: list[str] = []

    class CacheOnly:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg

        async def __aenter__(self) -> CacheOnly:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def context(self, question: str, *, paid: bool = True) -> dict[str, Any]:
            asked.append(paid)
            questions.append(question)
            if paid:
                raise AssertionError("the paid path was reached")
            raise ServeMiss("query embedding cache miss and serving is cache-only (--allow-paid to send)")

    pool = http.Pool(_fires(tmp_path, FIRE), overrides={"allow_paid": True}, server_factory=CacheOnly)
    pool.runtimes = Runtimes(pool, RuntimeSettings(), engine_factory=factory.engine)

    async def scenario() -> None:
        recall = pool.runtimes.recall_for(FIRE)
        with pytest.raises(LookupError) as miss:
            await recall("where does my team work?", speaker="Ana")
        assert str(miss.value) == "no cached context" and asked == [False]
        assert questions == ["where does Ana's team work?"]  # the turn's speaker resolves "my"
        status, body = await pool.context(FIRE, "where does Zed Quill work?")  # /context itself still may pay
        assert asked == [False, True] and body["degraded"]
        await pool.close()

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_a_cache_only_context_never_sends_even_when_serving_may_pay(tmp_path, monkeypatch):
    from memory_kernel.serve.answer import ServeConfig, ServeMiss, Server

    cfg = ServeConfig(group="g", allow_paid=True, voyage_api_key="v", openrouter_api_key="o")
    server = Server(cfg)

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a paid call was reached")

    monkeypatch.setattr("memory_kernel.graph.kg_writer.VoyageEmbedder", never)
    with pytest.raises(ServeMiss):
        server.query_vector("where does Zed Quill work?", paid=False)
    assert server.misses.voyage  # recorded as a miss, never sent


# -- review round 2 pins (the reviewer's probes as tests) --------------------------------------------------------------


def test_a_foreign_listener_on_a_port_moves_the_engine_to_the_next_free_one(tmp_path, factory, monkeypatch, caplog):
    from memory_kernel.runtime.engine import PortBusy, RuntimeEngineError

    runtimes, pool = _runtimes(tmp_path, factory)  # default ports 9712-9719
    pool.overrides["openrouter_api_key"] = SECRET  # as mk serve hands the pool its secrets
    tried: list[int] = []
    start = FakeRuntime.start

    def foreign_on_9712(self: FakeRuntime) -> None:
        tried.append(self.engine.port)
        if self.engine.port == 9712:
            raise PortBusy("runtime engine start failed: port 9712 already listening")
        if self.bonfire_id == OTHER:  # any other start failure: the plan is unread, and logged with secrets redacted
            raise RuntimeEngineError(f"runtime engine never answered (key {SECRET})")
        start(self)

    monkeypatch.setattr(FakeRuntime, "start", foreign_on_9712)

    async def scenario() -> None:
        _, body = await runtimes.plan(FIRE, await pool.meta(FIRE), _turn())
        assert body["status"] == "answered", body["diagnostics"]
        assert tried == [9712, 9713] and runtimes.engines() == {FIRE: 9713}
        _, other = await runtimes.plan(OTHER, await pool.meta(OTHER), _turn())
        assert other["status"] == "unread" and other["diagnostics"][0].startswith("runtime unavailable"), other
        await runtimes.close()

    asyncio.run(asyncio.wait_for(scenario(), 10))
    assert f"runtime.plan_unavailable: bonfire {OTHER}" in caplog.text
    assert SECRET not in caplog.text


def test_three_concurrent_first_plans_never_run_more_engines_than_the_cap(tmp_path, factory, monkeypatch):
    from memory_kernel.serve.runtime import ENGINES_BUSY

    root = _fires(tmp_path, "a", "b", "c")
    pool = http.Pool(root)
    runtimes = Runtimes(pool, RuntimeSettings(), runtime_factory=factory.runtime, engine_factory=factory.engine)
    plan = FakeRuntime.plan

    async def scenario() -> None:
        gate = asyncio.Event()

        async def slow(self: FakeRuntime, turn: Turn) -> TurnPlan:
            await gate.wait()
            return await plan(self, turn)

        monkeypatch.setattr(FakeRuntime, "plan", slow)
        tasks = [asyncio.create_task(runtimes.plan(b, await pool.meta(b), _turn())) for b in "abc"]
        await asyncio.sleep(0.05)
        assert len(runtimes.engines()) == 2
        gate.set()
        bodies = [body for _, body in await asyncio.gather(*tasks)]
        assert sorted(b["status"] for b in bodies) == ["answered", "answered", "unread"]
        assert [b["diagnostics"][0].startswith(ENGINES_BUSY) for b in bodies if b["status"] == "unread"] == [True]
        await runtimes.close()

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_a_plan_whose_engine_started_while_a_teach_began_answers_busy(tmp_path, factory, monkeypatch):
    import time as _time

    runtimes, pool = _runtimes(tmp_path, factory)
    start, plan = FakeRuntime.start, FakeRuntime.plan
    ran_while_teaching: list[bool] = []

    def slow_start(self: FakeRuntime) -> None:
        _time.sleep(0.2)
        start(self)

    async def watched(self: FakeRuntime, turn: Turn) -> TurnPlan:
        ran_while_teaching.append(bool(runtimes._teaching.get(FIRE)))
        return await plan(self, turn)

    monkeypatch.setattr(FakeRuntime, "start", slow_start)
    monkeypatch.setattr(FakeRuntime, "plan", watched)
    req = TeachRequest(["brew {X}"], ["make {X}"], {}, "Ana", "c1")

    async def scenario() -> None:
        planning = asyncio.create_task(runtimes.plan(FIRE, await pool.meta(FIRE), _turn()))
        await asyncio.sleep(0.05)  # the plan is inside _engaged, its engine starting
        teaching = asyncio.create_task(runtimes.teach(FIRE, await pool.meta(FIRE), req))
        _, body = await planning
        assert body["status"] == "unread" and body["diagnostics"] == [BUSY_TEACHING]
        assert (await teaching)[0] == 200
        assert ran_while_teaching == []  # the runtime never read a turn while its bonfire taught
        await runtimes.close()

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_a_failing_stop_never_exceeds_the_cap_and_close_goes_on(tmp_path, factory, monkeypatch, caplog):
    root = _fires(tmp_path, FIRE, OTHER)
    pool = http.Pool(root)
    settings = RuntimeSettings(max_runtime_engines=1, runtime_ports="9712-9714")
    runtimes = Runtimes(pool, settings, runtime_factory=factory.runtime, engine_factory=factory.engine)
    stop = FakeRuntime.stop

    def failing(self: FakeRuntime) -> None:
        if self.bonfire_id == FIRE:
            raise OSError("teardown failed")
        stop(self)

    async def scenario() -> None:
        await runtimes.plan(FIRE, await pool.meta(FIRE), _turn())
        monkeypatch.setattr(FakeRuntime, "stop", failing)
        _, body = await runtimes.plan(OTHER, await pool.meta(OTHER), _turn())
        assert body["status"] == "unread" and "teardown failed" in body["diagnostics"][0]
        assert runtimes.engines() == {FIRE: 9712}  # still counted: the cap holds
        assert factory.made[OTHER].engine.pid is None
        await runtimes.close()  # logs FIRE's failure, does not raise

    asyncio.run(asyncio.wait_for(scenario(), 10))
    assert "teardown failed" in caplog.text


def _mk(*args: str | bytes, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    base = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env or {})}
    cmd = [sys.executable, "-m", "memory_kernel.cli", *args]
    return subprocess.run(cmd, capture_output=True, env=base, timeout=120)


def test_mk_plan_refuses_cleanly_before_any_engine_starts(tmp_path):
    bad_argv = _mk("plan", "ember", b"where does \xff work?", "--speaker", "Ana")
    assert bad_argv.returncode == 2 and b"UTF-8" in bad_argv.stdout
    stub = tmp_path / "stub" / "pyfcg"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("raise ImportError('pyfcg stub: not installed')\n")
    no_pyfcg = _mk("plan", "ember", "where does Kira Dace work?", env={"PYTHONPATH": f"{stub.parent}:{ROOT / 'src'}"})
    assert no_pyfcg.returncode == 2 and b"pyfcg" in no_pyfcg.stdout
    no_password = _mk(
        "plan", "ember", "where does Kira Dace work?", env={"NEO4J_PASSWORD": "", "MK_DATA_DIR": str(tmp_path)}
    )
    assert no_password.returncode == 2 and (b"Neo4j password" in no_password.stdout or b"pyfcg" in no_password.stdout)
    for out in (bad_argv, no_pyfcg, no_password):
        assert b"Traceback" not in out.stderr


def test_put_catalogue_writes_bonfire_meta_and_get_reads_it(tmp_path, factory):
    """A PUT validates and sets ``catalogue_json`` through the narrow writer and reloads the runtime; a GET reads it
    back fresh (not from the server's cached meta); none put yet is the empty catalogue; a refused one changes nothing;
    a PUT never moves ``committed_run``."""
    writer = MemoryWriter()
    registry = _fires(tmp_path, FIRE)
    app = http.create_app(
        registry, runtime_factory=factory.runtime, engine_factory=factory.engine, writer_factory=lambda _id: writer
    )
    url = f"/v1/bonfires/{FIRE}/catalogue"
    with TestClient(app) as client:
        assert client.get(url).json() == {"sha256": None, "tools": []}  # absent until the first PUT
        client.post(f"/v1/bonfires/{FIRE}/plan", json=TURN)
        first = factory.made[FIRE]
        put = client.put(url, content=json.dumps(CATALOGUE))
        assert put.status_code == 200 and put.json()["tools"] == CATALOGUE["tools"]
        assert writer.calls == [("set_catalogue", json.dumps(CATALOGUE))]
        got = client.get(url).json()
        assert got == {"sha256": put.json()["sha256"], "tools": CATALOGUE["tools"]}
        assert registry.metas[FIRE]["catalogue_json"] is None  # the cached meta never saw it: GET reads fresh
        assert registry.metas[FIRE]["committed_run"] == 0  # a PUT does not move the run: no swap
        refused = client.put(url, content="{}")
        assert refused.status_code == 400 and "schema_version" in refused.json()["reason"]
        assert writer.calls == [("set_catalogue", json.dumps(CATALOGUE))]  # nothing written
        client.post(f"/v1/bonfires/{FIRE}/plan", json=TURN)
        assert factory.made[FIRE] is not first  # the PUT dropped the runtime: the next plan made a new one
        writer.stored_catalogue = "{not json"
        assert client.get(url).status_code == 503
        writer.fail = RuntimeError("neo4j down")
        assert client.get(url).status_code == 503 and client.put(url, content=json.dumps(CATALOGUE)).status_code == 503


@pytest.mark.needs_neo4j
def test_plan_route_answers_with_no_directory(neo4j_target):
    """The default runtime of a bonfire that is only a Neo4j group: its taught phrases, rolesets and catalogue come
    from the graph (the real registry, the real narrow writer), with no bonfire directory anywhere."""
    from neo4j import AsyncGraphDatabase, GraphDatabase

    from memory_kernel.graph.state import meta_props, meta_row, roleset_rows
    from memory_kernel.runtime.graph_store import NarrowWriter
    from memory_kernel.serve.answer import ReadOnlyDriver
    from memory_kernel.serve.registry import BonfireRegistry

    group = neo4j_target.group
    meta = meta_props(
        bonfire_id=group,
        build_config={"sample_id": group},
        recipe_tag="t",
        extraction_model={"model": "m"},
        agent_handles=[],
        agent_handle_runs=[],
        chat_tagged=False,
        explicit_since=None,
        sessions=1,
        committed_run=1,
        generation="gen-1",
    )
    taught = dataclasses.replace(SPEC, status="live")
    auth = (neo4j_target.user, neo4j_target.password)
    with GraphDatabase.driver(neo4j_target.uri, auth=auth) as sync:
        sync.execute_query("CREATE (m:BonfireMeta) SET m = $p", p=meta_row(group, meta)["props"])
        for node in roleset_rows(group, {"zorb.01": {"label": "zorb"}}, run=1):
            sync.execute_query("CREATE (n:Roleset) SET n = $p", p=node["props"])
        NarrowWriter(sync, group).put_phrases([taught])

    class LazyDriver:
        """The async driver opens inside the app's own event loop, on first use."""

        def __init__(self) -> None:
            self._driver: Any = None

        def session(self, **kwargs: Any):
            if self._driver is None:
                self._driver = ReadOnlyDriver(AsyncGraphDatabase.driver(neo4j_target.uri, auth=auth))
            return self._driver.session(**kwargs)

    overrides = {
        "neo4j_uri": neo4j_target.uri,
        "neo4j_user": neo4j_target.user,
        "neo4j_password": neo4j_target.password,
    }
    app = http.create_app(BonfireRegistry(LazyDriver(), refresh_s=0.0), serve_cfg_overrides=overrides)
    with TestClient(app) as client:
        rows = client.get(f"/v1/bonfires/{group}/constructions")
        put = client.put(f"/v1/bonfires/{group}/catalogue", content=json.dumps(CATALOGUE))
        got = client.get(f"/v1/bonfires/{group}/catalogue")
        after = client.get(f"/v1/bonfires/{group}/constructions")
        slot = app.state.pool.runtimes._slots.get(group)
    assert rows.status_code == 200 and [c["id"] for c in rows.json()["constructions"]] == [SPEC.id]
    assert rows.json()["constructions"][0]["status"] == "live"
    assert put.status_code == 200 and got.json()["tools"] == CATALOGUE["tools"]
    # the PUT dropped the runtime; the second listing made a new one, which read the catalogue and the Roleset node
    assert after.status_code == 200 and slot is not None and slot.runtime.lexicon.frames_of("zorb") == ["zorb.01"]
    assert [c.id for c in slot.runtime.cards] == ["card.ring.0"]


@pytest.mark.needs_neo4j
def test_mk_plan_refuses_a_group_with_no_bonfire_meta(neo4j_target):
    pytest.importorskip("pyfcg")  # without it the command stops earlier, with its own refusal
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "NEO4J_PASSWORD": neo4j_target.password}
    out = subprocess.run(
        [sys.executable, "-m", "memory_kernel.cli", "plan", neo4j_target.group, "hello", "--empty-graph",
         "--neo4j-uri", neo4j_target.uri, "--neo4j-user", neo4j_target.user],
        capture_output=True, text=True, env=env, timeout=120,
    )  # fmt: skip
    assert out.returncode == 2 and "BonfireMeta" in out.stdout and "Traceback" not in out.stderr
