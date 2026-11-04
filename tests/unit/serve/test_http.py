"""``mk serve`` answers from Neo4j alone: a bonfire is a group with a ``BonfireMeta``; one warm ``Server`` per bonfire is
swapped for a fresh one when the bonfire's ``committed_run`` moves; every failure is a JSON answer with ``degraded``
instead of a traceback; and nothing is written to disk. Offline: a fake registry, a fake ``Server`` factory,
FastAPI's ``TestClient``; no Neo4j, no network, no paid call."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import threading
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_kernel.ops.memory import MemoryKernelOps
from memory_kernel.serve import answer, http
from memory_kernel.serve.answer import ServeConfig, ServeMiss
from unit.serve.fakes import SECRET, FakeRegistry, Fakes

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

FIRE = "ember"


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry().add(FIRE)


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


@pytest.fixture
def serve(registry: FakeRegistry, fakes: Fakes):
    """``serve(open_backoff_s, **create_app keywords)`` -> a TestClient (closed at teardown); ``client`` is the default
    60 s backoff, with no warm-up."""
    with contextlib.ExitStack() as stack:

        def make(open_backoff_s: float = 60.0, **kw: Any) -> TestClient:
            app = http.create_app(
                registry,
                serve_cfg_overrides={"allow_paid": False, "openrouter_api_key": SECRET},
                server_factory=fakes,
                open_backoff_s=open_backoff_s,
                **kw,
            )
            return stack.enter_context(TestClient(app))

        yield make


@pytest.fixture
def client(serve):
    return serve()


def ready(r) -> tuple[int, str, dict, dict | None]:
    """A ``/healthz`` answer's readiness: status code, ``status``, ``bonfires`` and ``parser_notes`` (the body also
    has ``version`` and ``spend``)."""
    body = r.json()
    return r.status_code, body["status"], body["bonfires"], body.get("parser_notes")


def eventually(check, *, seconds: float = 5.0):
    """Poll ``check()`` until it returns a truthy value (the swap runs in the background)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if out := check():
            return out
        time.sleep(0.01)
    raise AssertionError("never happened")


def ask(client: TestClient, query: str = "what is warm?", bonfire_id: str = FIRE):
    return client.post(f"/v1/bonfires/{bonfire_id}/context", json={"query": query})


# -- refusals ----------------------------------------------------------------------------------------------------------


def test_an_unknown_bonfire_is_a_404_with_the_degraded_shape(client):
    for name in ("nope", "not-a-bonfire", "..."):  # a group with no BonfireMeta is not a bonfire
        r = ask(client, bonfire_id=name)
        assert r.status_code == 404
        body = r.json()
        assert body["engine"] == "kernel" and body["degraded"] is True and name in body["reason"]


def test_a_registry_that_fails_is_a_503_never_a_traceback(client, registry):
    registry.fail = ConnectionError(f"neo4j said no to {SECRET}")
    r = ask(client)
    assert r.status_code == 503
    body = r.json()
    assert body["engine"] == "kernel" and body["degraded"] is True and "does not load" in body["reason"]
    assert SECRET not in body["reason"] and "<redacted>" in body["reason"]
    assert client.post(f"/v1/bonfires/{FIRE}/entities/sheet", json={"name": "x"}).status_code == 503
    assert client.get(f"/v1/bonfires/{FIRE}/conversations/c/watermark").status_code == 503
    assert client.get("/healthz").status_code == 200  # not warm: ready, with nothing listed


def test_a_body_without_a_query_is_refused(client):
    r = client.post(f"/v1/bonfires/{FIRE}/context", json={"question": "x"})
    assert r.status_code == 422 and r.json()["degraded"] is True


# -- answers -----------------------------------------------------------------------------------------------------------


def test_a_question_is_answered_from_the_graph(client, fakes):
    r = ask(client)
    assert r.status_code == 200
    assert r.json() == {
        "engine": "kernel",
        "context": "[key0] what is warm?",
        "message_ids": ["D1:1"],
        "tokens": 5,
        "frame": {"shape": "what"},
        "run": "bootstrap",
        "degraded": False,
    }
    (cfg,) = fakes.opened
    assert (cfg.group, cfg.query_cache_key, cfg.allow_paid) == (FIRE, "key0", False)
    assert not hasattr(cfg, "conv_root") and not hasattr(cfg, "cache_dir")


def test_a_speaker_resolves_first_person_words_before_retrieval(client):
    body = client.post(f"/v1/bonfires/{FIRE}/context", json={"query": "what do I like?", "speaker": "@Ivo"}).json()
    assert body["context"] == "[key0] what do Ivo like?" and body["resolved_query"] == "what do Ivo like?"


def test_without_a_speaker_the_question_is_unchanged(client):
    body = ask(client, "what do I like?").json()
    assert body["context"] == "[key0] what do I like?" and "resolved_query" not in body


def test_a_non_string_speaker_is_refused(client):
    r = client.post(f"/v1/bonfires/{FIRE}/context", json={"query": "x", "speaker": 3})
    assert r.status_code == 422 and r.json()["degraded"] is True


def test_the_watermark_endpoint_reads_the_graph(client, registry):
    mark = {"message_id": "m-9", "order": 9, "timestamp": "2026-10-01T10:00:00+00:00"}
    registry.marks[(FIRE, "slack-C1-main")] = mark
    r = client.get(f"/v1/bonfires/{FIRE}/conversations/slack-C1-main/watermark")
    assert r.status_code == 200 and r.json() == {
        "engine": "kernel",
        "conversation_id": "slack-C1-main",
        "watermark": mark,
    }
    assert client.get(f"/v1/bonfires/{FIRE}/conversations/slack-C9-main/watermark").json()["watermark"] is None
    assert client.get("/v1/bonfires/nope/conversations/slack-C1-main/watermark").status_code == 404


def test_the_warm_server_is_reused_between_questions(client, fakes):
    """Without the warm-up nothing opens before a question: ``/healthz`` is a 200 that lists the bonfire as null."""
    assert ready(client.get("/healthz")) == (200, "ok", {FIRE: None}, None)
    for _ in range(3):
        assert ask(client).json()["run"] == "bootstrap"
    assert len(fakes.opened) == 1 and fakes.closed == []
    assert ready(client.get("/healthz")) == (200, "ok", {FIRE: "bootstrap"}, None)


def test_newer_committed_run_swaps_the_server(client, registry, fakes):
    assert ask(client).json()["context"] == "[key0] what is warm?"
    registry.move(FIRE, 1)  # an append committed: the question that sees it is still answered by the loaded server
    eventually(lambda: ask(client).json()["run"] == "0001")
    assert ask(client).json()["context"] == "[key1] what is warm?"  # the fresh server, opened from the new meta
    assert [c.query_cache_key for c in fakes.opened] == ["key0", "key1"]
    assert eventually(lambda: fakes.closed == ["key0"])
    registry.move(FIRE, 12)
    eventually(lambda: ask(client).json()["run"] == "0012")


def test_a_reseeded_lower_committed_run_swaps_the_server(client, registry, fakes):
    """The rebuild runbook (empty, then seed) restarts at run 0: any change of committed_run swaps."""
    registry.move(FIRE, 3)
    assert ask(client).json()["run"] == "0003"
    registry.move(FIRE, 0)
    eventually(lambda: ask(client).json()["run"] == "bootstrap")
    assert [c.query_cache_key for c in fakes.opened] == ["key3", "key0"]
    assert eventually(lambda: fakes.closed == ["key3"])
    for _ in range(3):  # unchanged: no more opens
        ask(client)
    assert fakes.attempts == 2


def test_a_new_generation_at_the_same_run_swaps_the_server(client, registry, fakes):
    """A rebuild lands back on the same committed_run (0) with a new generation: the loaded server is replaced."""
    assert ask(client).json()["run"] == "bootstrap"
    registry.metas[FIRE]["generation"] = "gen-two"  # the same run, resealed
    eventually(lambda: ask(client) and len(fakes.opened) == 2)  # a question starts the swap
    assert eventually(lambda: fakes.closed == ["key0"])
    assert ask(client).json()["run"] == "bootstrap"
    for _ in range(3):  # unchanged generation: no more opens
        ask(client)
    assert fakes.attempts == 2


def test_a_newer_run_that_fails_to_open_keeps_the_loaded_one_and_backs_off(client, registry, fakes):
    registry.move(FIRE, 1)
    assert ask(client).json()["run"] == "0001"
    registry.move(FIRE, 2)
    fakes.fail_open = RuntimeError("index stale")
    body = ask(client).json()
    assert body["run"] == "0001" and body["degraded"] is False
    eventually(lambda: fakes.attempts == 2)  # the background open of run 2 failed
    fakes.fail_open = None
    for _ in range(5):  # within the backoff: never tried again, run 1 keeps answering, not degraded
        body = ask(client).json()
        assert body["run"] == "0001" and body["degraded"] is False
    assert fakes.attempts == 2


def test_a_failed_newer_run_is_retried_after_the_backoff(serve, registry, fakes):
    client = serve(open_backoff_s=0.0)
    registry.move(FIRE, 1)
    assert ask(client).json()["run"] == "0001"
    registry.move(FIRE, 2)
    fakes.fail_open = RuntimeError("index stale")
    ask(client)
    eventually(lambda: fakes.attempts == 2)
    fakes.fail_open = None
    eventually(lambda: ask(client).json()["run"] == "0002")


def test_a_first_open_that_fails_is_a_degraded_200_and_backs_off(client, fakes):
    fakes.fail_open = RuntimeError("neo4j down")
    for _ in range(3):
        r = ask(client)
        assert r.status_code == 200
        body = r.json()
        assert body["degraded"] is True and body["context"] == "" and body["run"] == "bootstrap"
        assert "neo4j down" in body["reason"]
    assert fakes.attempts == 1


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bonfire round budget exhausted"),
        ServeMiss("query embedding cache miss and serving is cache-only (--allow-paid to send)"),
        KeyError("message_id"),
    ],
)
def test_an_error_inside_context_is_a_degraded_200(client, fakes, error):
    fakes.fail_context = error
    r = ask(client)
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "kernel" and body["degraded"] is True and body["context"] == ""
    assert body["message_ids"] == [] and body["tokens"] == 0 and body["run"] == "bootstrap"
    assert type(error).__name__ in body["reason"]


def test_a_secret_never_reaches_a_reason(client, fakes, caplog):
    fakes.fail_context = RuntimeError(f"provider said no to key {SECRET}")
    with caplog.at_level("DEBUG"):
        body = ask(client).json()
    assert body["degraded"] is True and SECRET not in body["reason"] and "<redacted>" in body["reason"]
    assert SECRET not in caplog.text


def test_serve_config_hides_secrets():
    cfg = ServeConfig(group="g", openrouter_api_key=SECRET, neo4j_password=SECRET)
    assert SECRET not in repr(cfg)


# -- health and the pool -----------------------------------------------------------------------------------------------


def test_healthz_spend_reads_kernel_ops(serve, registry, monkeypatch):
    """``/healthz`` spend is ``ops.spend`` per listed bonfire: charged (settled plus held) and the ceiling, decimal
    strings; null for a bonfire with no ``kernel_ops`` record; null for all without ops."""
    registry.add("cinder")
    ops = MemoryKernelOps()
    ops.create(FIRE, ceiling_usd=Decimal("40"), state="ready")
    ops.hold(FIRE, run=1, estimate_usd=Decimal("2.5"))
    client = serve(ops=ops)
    assert client.get("/healthz").json()["spend"] == {
        "cinder": None,
        FIRE: {"charged_usd": "2.5", "ceiling_usd": "40"},
    }
    ops.settle(FIRE, run=1, spent_usd=Decimal("1.25"), new_messages=3, budget_usd=Decimal("5"))
    assert client.get("/healthz").json()["spend"][FIRE]["charged_usd"] == "2.5"  # kept for SPEND_TTL_S
    monkeypatch.setattr(http, "SPEND_TTL_S", 0.0)
    assert client.get("/healthz").json()["spend"][FIRE] == {"charged_usd": "1.25", "ceiling_usd": "40"}
    plain = serve()
    assert plain.get("/healthz").json()["spend"] == {"cinder": None, FIRE: None}


def test_healthz_is_not_ready_until_every_bonfire_is_warm_or_has_failed(serve, registry, fakes, monkeypatch):
    """``mk serve`` warms at start: the models load once, then every listed bonfire opens at its committed run.
    ``/healthz`` is a 503 while one still opens. A bonfire whose open is refused is listed as failed, is never retried
    and does not hold the others. A bonfire that appears later is warmed when ``/healthz`` finds it, listed warming
    meanwhile without making a server that has been ready a 503. ``parser_notes`` says when the query parser fell
    back."""
    from neo4j.exceptions import AuthError

    models: list[int] = []
    monkeypatch.setattr(http, "warm_models", lambda: models.append(1))
    registry.add("locked")
    fakes.refuse["locked"] = AuthError("the password is wrong")
    fakes.hold = threading.Event()  # ember's open waits
    client = serve(warm=True, warm_retry_s=0.01)

    def health() -> tuple[int, str, dict, dict | None]:
        return ready(client.get("/healthz"))

    eventually(lambda: health()[2]["locked"] != "warming")
    status, state, bonfires, notes = health()
    refused = bonfires["locked"]
    assert refused.startswith("failed: neo4j: refused (AuthError")
    assert (status, state, bonfires, notes) == (503, "warming", {FIRE: "warming", "locked": refused}, None)
    fakes.hold.set()
    eventually(lambda: health()[0] == 200)
    assert health() == (200, "ok", {FIRE: "bootstrap", "locked": refused}, None)
    assert models == [1] and fakes.attempts == 2  # one open each: the refusal was not retried
    assert ask(client).json()["run"] == "bootstrap" and len(fakes.opened) == 1  # the first question finds it open

    registry.add("late", run=4)
    assert health() == (200, "ok", {FIRE: "bootstrap", "late": "warming", "locked": refused}, None)  # stays ready
    eventually(lambda: health()[2]["late"] == "0004")
    monkeypatch.setattr(
        "memory_kernel.graph.g1_nlp.parser_note", lambda purpose: "sm-fallback" if purpose == "query" else None
    )
    assert health() == (
        200,
        "ok",
        {FIRE: "bootstrap", "late": "0004", "locked": refused},
        {"query": "sm-fallback"},  # the query parser fell back
    )


def test_healthz_with_zero_bonfires_is_ready_once_models_are_warm(serve, registry, monkeypatch):
    monkeypatch.setattr(http, "warm_models", lambda: None)
    registry.metas.clear()
    client = serve(warm=True, warm_retry_s=0.01)
    eventually(lambda: client.get("/healthz").status_code == 200)
    assert ready(client.get("/healthz")) == (200, "ok", {}, None)


def test_healthz_with_zero_bonfires_waits_for_the_models(serve, registry, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(http, "warm_models", release.wait)
    registry.metas.clear()
    client = serve(warm=True, warm_retry_s=0.01)
    assert ready(client.get("/healthz")) == (503, "warming", {}, None)
    release.set()
    eventually(lambda: client.get("/healthz").status_code == 200)


def test_healthz_listing_failure_is_not_ready_with_no_bonfires_open(serve, registry, monkeypatch):
    monkeypatch.setattr(http, "warm_models", lambda: None)
    registry.fail = ConnectionError("neo4j down")
    client = serve(warm=True, warm_retry_s=0.01)
    assert ready(client.get("/healthz"))[0] == 503
    registry.fail = None


def test_a_bonfire_appearing_after_zero_bonfires_keeps_the_server_ready(serve, registry, fakes, monkeypatch):
    monkeypatch.setattr(http, "warm_models", lambda: None)
    registry.metas.clear()
    fakes.hold = threading.Event()  # the new bonfire's open waits
    client = serve(warm=True, warm_retry_s=0.01)
    eventually(lambda: client.get("/healthz").status_code == 200)
    registry.add("fresh")
    assert ready(client.get("/healthz")) == (200, "ok", {"fresh": "warming"}, None)
    fakes.hold.set()
    eventually(lambda: client.get("/healthz").json()["bonfires"]["fresh"] == "bootstrap")
    assert ready(client.get("/healthz")) == (200, "ok", {"fresh": "bootstrap"}, None)


def test_zero_bonfires_then_the_only_bonfire_fails_is_not_ready(serve, registry, fakes, monkeypatch):
    from neo4j.exceptions import AuthError

    monkeypatch.setattr(http, "warm_models", lambda: None)
    registry.metas.clear()
    client = serve(warm=True, warm_retry_s=0.01)
    eventually(lambda: client.get("/healthz").status_code == 200)
    registry.add("locked")
    fakes.refuse["locked"] = AuthError("the password is wrong")
    eventually(lambda: client.get("/healthz").status_code == 503 and "failed" in str(client.get("/healthz").json()))


def test_a_registry_blip_after_zero_bonfires_keeps_the_server_ready(serve, registry, monkeypatch):
    monkeypatch.setattr(http, "warm_models", lambda: None)
    registry.metas.clear()
    client = serve(warm=True, warm_retry_s=0.01)
    eventually(lambda: client.get("/healthz").status_code == 200)
    registry.fail = ConnectionError("neo4j blip")
    assert ready(client.get("/healthz")) == (200, "ok", {}, None)
    registry.fail = None


def test_a_registry_blip_keeps_a_ready_server_ready(serve, registry, monkeypatch):
    monkeypatch.setattr(http, "warm_models", lambda: None)
    client = serve(warm=True, warm_retry_s=0.01)
    eventually(lambda: client.get("/healthz").status_code == 200)
    registry.fail = ConnectionError("neo4j blip")
    assert ready(client.get("/healthz")) == (200, "ok", {FIRE: "bootstrap"}, None)
    registry.fail = None


def test_the_warm_up_loads_what_a_question_reads_with(monkeypatch):
    """``warm_models`` loads the WordNet corpus, the query and turn pipelines (one parse each) and, last, the token
    encoder the context budget counts with, so the first question pays none of them."""
    did: list[str] = []
    monkeypatch.setattr("memory_kernel.grammar.dependencies.wordnet_ready", lambda: did.append("wordnet"))
    monkeypatch.setattr("memory_kernel.graph.g1_nlp.nlp", lambda purpose="extraction": lambda text: did.append(purpose))
    monkeypatch.setattr("memory_kernel.search.kg_context.count_tokens", lambda text: did.append("encoder") or 1)
    http.warm_models()
    assert did == ["wordnet", "query", "turn", "encoder"]


@pytest.mark.parametrize("error", ["service unavailable", "connection refused", "name not resolved"])
def test_an_open_that_cannot_connect_stays_warming_until_a_retry_opens_it(serve, fakes, monkeypatch, error):
    """Right after a restart Neo4j (or its host name) may come up after the kernel: an open that cannot connect is
    retried with backoff (5 s doubling to 30 s; 10 ms capped at 20 ms here), and ``/healthz`` stays a 503 until the
    retry opens the run. The driver raises a name that does not resolve as a ``ValueError`` from the resolver's
    error."""
    from neo4j.exceptions import ServiceUnavailable

    monkeypatch.setattr(http, "warm_models", lambda: None)
    monkeypatch.setattr(http, "WARM_RETRY_CAP_S", 0.02)  # every retry within 20 ms: the wait for the 200 below too
    unresolved = ValueError("Failed to DNS resolve address neo4j:7687")
    unresolved.__cause__ = socket.gaierror(socket.EAI_FAIL, "Non-recoverable failure in name resolution")
    fakes.fail_open = {
        "service unavailable": ServiceUnavailable("Neo4j is not accepting connections yet"),
        "connection refused": ConnectionRefusedError(111, "Connection refused"),
        "name not resolved": unresolved,
    }[error]
    client = serve(warm=True, warm_retry_s=0.01)
    eventually(lambda: fakes.attempts >= 3)  # retried, still failing
    assert ready(client.get("/healthz")) == (503, "warming", {FIRE: "warming"}, None)
    fakes.fail_open = None
    eventually(lambda: client.get("/healthz").status_code == 200)
    assert ready(client.get("/healthz")) == (200, "ok", {FIRE: "bootstrap"}, None)


def test_a_swapped_out_server_closes_only_after_its_question_finishes(registry, fakes):
    async def scenario() -> None:
        pool = http.Pool(registry, overrides={"allow_paid": False}, server_factory=fakes)
        registry.move(FIRE, 1)
        assert (await pool.context(FIRE, "warm?"))[1]["run"] == "0001"
        gate = fakes.gate = asyncio.Event()
        slow = asyncio.create_task(pool.context(FIRE, "slow?"))
        while fakes.entered < 2:  # the slow question is inside run 1's server
            await asyncio.sleep(0.01)
        fakes.gate = None
        registry.move(FIRE, 2)
        assert (await pool.context(FIRE, "fast?"))[1]["run"] == "0001"  # the swap starts in the background
        await pool.wait_swaps()
        assert (await pool.context(FIRE, "fast?"))[1]["run"] == "0002"
        assert fakes.closed == []  # run 1 is still answering the slow question
        gate.set()
        status, body = await slow
        assert (status, body["run"], body["context"]) == (200, "0001", "[key1] slow?")
        assert fakes.closed == ["key1"]
        await pool.close()
        assert fakes.closed == ["key1", "key2"]

    asyncio.run(asyncio.wait_for(scenario(), 10))


# -- nothing is written to disk ----------------------------------------------------------------------------------------


def _tree(*roots: Path) -> dict[str, list]:
    return {str(r): sorted((d, sorted(dirs), sorted(files)) for d, dirs, files in os.walk(r)) for r in roots}


def test_serve_writes_no_file(tmp_path, monkeypatch, registry):
    """The real ``Server`` (its index, retrieval and senders faked) answers two questions over HTTP, one a paid miss
    and one a hit, and an answer is asked of its chat, with the working directory and ``HOME`` in an empty temporary
    directory: nothing is created in either."""
    from memory_kernel.search.kg_index import KgIndex

    monkeypatch.setenv("NLTK_DATA", str(Path.home() / "nltk_data"))  # read before HOME moves
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.chdir(work)
    sent: list[str] = []

    class FakeVoyage:
        def __init__(self, key: str, input_type: str) -> None:
            assert key == "voyage-key" and input_type == "query"

        def __call__(self, texts: list[str]) -> list[list[float]]:
            sent.extend(texts)
            return [[1.0, 0.0] for _ in texts]

    class FakeChat:
        def __init__(self, api_key: str | None) -> None:
            assert api_key == "openrouter-key"

        def __call__(self, request: dict) -> dict:
            sent.append("chat")
            return {"text": "an answer", "usage": {}}

    async def build_answer_context(driver, question, vec, **kw):
        return {"text": f"memories for {question}", "message_ids": ["D1:1"], "tokens": 3}

    class FakeDriver:
        async def close(self) -> None:
            return None

    async def load_index(driver, cfg):
        return KgIndex(
            keys={}, vectors={}, msg_facts={}, msg_events={}, msg_entities={}, told_at={}, messages={}, dim=2
        )

    monkeypatch.setattr("memory_kernel.graph.kg_writer.VoyageEmbedder", FakeVoyage)
    monkeypatch.setattr("memory_kernel.pipeline.llm.OpenRouterChat", FakeChat)
    monkeypatch.setattr("memory_kernel.search.kg_reader.build_answer_context", build_answer_context)
    monkeypatch.setattr(
        "memory_kernel.search.kg_recipe.search_index", lambda idx, **kw: SimpleNamespace(hits=[], messages={})
    )
    monkeypatch.setattr(answer, "open_driver", lambda cfg: FakeDriver())
    monkeypatch.setattr(answer, "load_index", load_index)
    monkeypatch.setattr("memory_kernel.search.kg_context.relative_texts", lambda messages: {})
    registry.metas[FIRE]["query_cache_key"] = "no-file-test"
    before = _tree(work, home)

    app = http.create_app(
        registry,
        serve_cfg_overrides={
            "allow_paid": True,
            "voyage_api_key": "voyage-key",
            "openrouter_api_key": "openrouter-key",
        },
    )
    with TestClient(app) as client:
        for _ in range(2):  # a paid miss, then an LRU hit
            body = ask(client, "Who is Joanna?").json()
            assert body["degraded"] is False and body["context"] == "memories for Who is Joanna?"
        assert sent == ["Who is Joanna?"]
        server = app.state.pool._warm[FIRE].server
        assert asyncio.run(asyncio.to_thread(server.answer_text, "Who is Joanna?", "ctx")) == "an answer"
        assert sent == ["Who is Joanna?", "chat"]
    assert _tree(work, home) == before == _tree(work, home)
    assert list(work.iterdir()) == [] and list(home.iterdir()) == []


# -- the command -------------------------------------------------------------------------------------------------------


def test_mk_serve_sends_by_default_and_never_prints_a_secret(monkeypatch, capsys):
    import uvicorn

    from memory_kernel import cli

    ran: list[tuple[Any, str, int]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: ran.append((app, host, port)))
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    monkeypatch.setenv("NEO4J_PASSWORD", "pw-not-printed")
    monkeypatch.delenv(cli.OPS_URI_ENV, raising=False)  # the quickstart exports it
    assert cli.main(["serve", "--port", "9911", "--neo4j-uri", "bolt://neo4j.test:7687"]) == 0
    app, host, port = ran[0]
    assert (host, port) == ("127.0.0.1", 9911)
    pool = app.state.pool
    assert pool.overrides["allow_paid"] is True and pool.overrides["openrouter_api_key"] == SECRET
    assert pool.overrides["neo4j_uri"] == "bolt://neo4j.test:7687" and pool.ops is None
    assert pool.warm is True  # mk serve warms at start
    out = capsys.readouterr().out
    assert SECRET not in out and "pw-not-printed" not in out
    assert json.loads(out)["secret_sources"]["openrouter_api_key"] == "env"
    assert cli.main(["serve", "--no-allow-paid"]) == 0
    assert ran[1][0].state.pool.overrides["allow_paid"] is False


def test_mk_serve_takes_kernel_ops_from_the_ops_uri(monkeypatch):
    import uvicorn

    from memory_kernel import cli
    from memory_kernel.ops import mongo

    ran: list[Any] = []
    made = MemoryKernelOps()
    uris: list[str] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: ran.append(app))
    monkeypatch.setattr(mongo.MongoKernelOps, "from_uri", classmethod(lambda cls, uri: uris.append(uri) or made))
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    assert cli.main(["serve", "--ops-uri", "mongodb://ops.test:27017"]) == 0
    assert uris == ["mongodb://ops.test:27017"] and ran[0].state.pool.ops is made


def test_mk_serve_refuses_without_a_neo4j_password(monkeypatch, capsys):
    from memory_kernel import cli

    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    assert cli.main(["serve"]) == cli.REFUSED
    assert "no Neo4j password" in capsys.readouterr().out
