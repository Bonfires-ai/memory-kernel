"""The managed syntactic parser: ``mk build`` runs its own parser on a fixed loopback port, only inside the
capture's syntax and run attempts, and refuses clearly when it cannot.

All in-process and offline: no parser, engine or model is started; ``subprocess.Popen`` is replaced where a test must
prove that nothing was launched."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest


def config(tmp_path: Path, **kw):
    from memory_kernel.pipeline.config import BuildConfig

    return BuildConfig(
        **{
            "sample_id": "conv-syn",
            "conversation": tmp_path / "conversation.json",
            "group": "g1-test-c2-syn",
            "purpose": "remember the conversation",
            "workdir": tmp_path / "workdir",
            "root": tmp_path,
            **kw,
        }
    )


def build_argv(tmp_path: Path, *extra: str) -> list[str]:
    conversation = tmp_path / "conversation.json"
    conversation.write_text(json.dumps({"sample_id": "conv-syn", "conversation": {"session_1": []}}))
    return ["build", str(conversation), "--workdir", str(tmp_path / "w"), "--group", "g", "--purpose", "p", *extra]


def placed_model(data: Path) -> Path:
    """What ``mk setup --component parser-model`` leaves behind: the model dir and its receipt entry."""
    from memory_kernel import setup_runtime

    model = data / "nltk_data/models/benepar_en3"
    model.mkdir(parents=True)
    (model / "benepar_model.bin").write_bytes(b"x")
    receipt = setup_runtime.read_receipt(data)
    receipt["components"]["parser-model"] = {"path": "nltk_data/models/benepar_en3", "pinned": True}
    setup_runtime._write_receipt(data, receipt)
    return model


def no_popen(*_args, **_kwargs):
    raise AssertionError("a process was started")


def test_the_managed_parser_is_the_default_its_port_is_checked_and_an_older_config_keeps_its_url(tmp_path):
    from memory_kernel import cli
    from memory_kernel.pipeline.config import BuildConfig, live_parser_url

    cfg = cli.build_config(cli.parser().parse_args(build_argv(tmp_path)))
    assert cfg.parser_url is None and cfg.parser_port == 9801
    assert live_parser_url(cfg) == "http://127.0.0.1:9801"  # the string every new capture seals
    other = cli.build_config(cli.parser().parse_args(build_argv(tmp_path, "--parser-port", "9850")))
    assert live_parser_url(other) == "http://127.0.0.1:9850"
    for bad in ("9701", "9900", "9655"):
        with pytest.raises(ValueError, match="parser port"):
            cli.build_config(cli.parser().parse_args(build_argv(tmp_path, "--parser-port", bad)))
    external = cli.build_config(cli.parser().parse_args(build_argv(tmp_path, "--parser-url", "http://127.0.0.1:8000")))
    assert live_parser_url(external) == "http://127.0.0.1:8000"

    older = {k: v for k, v in config(tmp_path).to_json().items() if k != "parser_port"}
    older["parser_url"] = "http://127.0.0.1:8000"  # the earlier default, written into every older build-config.json
    loaded = BuildConfig.from_json(older, secrets={})
    assert loaded.parser_url == "http://127.0.0.1:8000" and loaded.parser_port == 9801


def test_the_capture_site_describes_the_managed_parser_and_starts_nothing(tmp_path, monkeypatch):
    import subprocess

    from memory_kernel.pipeline.engine import ParserConfig
    from memory_kernel.pipeline.runner import Ctx
    from memory_kernel.pipeline.stages import parse

    monkeypatch.setattr(subprocess, "Popen", no_popen)
    engine = {"engine_binary": tmp_path / "fcg", "engine_image": tmp_path / "image", "allow_unpinned_engine": True}
    managed = parse.site_for(Ctx(cfg=config(tmp_path, data_dir=tmp_path / "data", **engine), concurrency=1))
    assert managed.parser == ParserConfig(port=9801, data_dir=tmp_path / "data")
    assert managed.live_parser == managed.parser.url == "http://127.0.0.1:9801"

    external = config(tmp_path, data_dir=tmp_path / "data", parser_url="http://127.0.0.1:8000", **engine)
    assert parse.site_for(Ctx(cfg=external, concurrency=1)).parser is None  # an external URL starts nothing
    no_data = parse.site_for(Ctx(cfg=config(tmp_path, **engine), concurrency=1))  # a complete capture needs no parser
    assert no_data.parser == ParserConfig(port=9801, data_dir=None)  # the launch refuses it, never site_for


def test_the_managed_parser_refuses_without_its_extra_its_model_or_a_free_port(tmp_path, monkeypatch):
    import importlib.util
    import subprocess

    from memory_kernel.pipeline import engine
    from memory_kernel.pipeline.runner import parser_problems

    data = tmp_path / "data"
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "benepar" else find_spec(name))
    problems = engine.parser_service_problems(engine.ParserConfig(port=9801, data_dir=data), check_port=False)
    assert any("benepar" in p and "memory-kernel[parser]" in p for p in problems)
    assert any("mk setup --component parser-model" in p for p in problems)
    assert "the managed parser needs the `mk setup` data dir (--data-dir); or pass --parser-url" in parser_problems(
        config(tmp_path)
    )
    assert "--allow-remote-parser" in parser_problems(config(tmp_path, parser_url="http://parser.invalid:8000"))[0]
    assert parser_problems(config(tmp_path, parser_url="http://parser.invalid:8000", allow_remote_parser=True)) == []

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())  # the extra "is installed"
    placed_model(data)
    with socket.socket() as holder:  # a busy port in the parser range is refused before anything starts
        for port in engine.PARSER_PORT_RANGE:
            try:
                holder.bind(("127.0.0.1", port))
                break
            except OSError:
                continue
        else:
            pytest.skip("no free port in the parser range")
        holder.listen()
        busy = config(tmp_path, data_dir=data, parser_port=port)
        assert any("already listening" in p for p in parser_problems(busy))
        # a capture under the warm capture slot leases the slot's parser (which holds the port): no cold port check
        assert parser_problems(config(tmp_path, data_dir=data, parser_port=port, capture_slot=tmp_path / "slot")) == []
        monkeypatch.setattr(subprocess, "Popen", no_popen)
        service = engine.ParserService(engine.ParserConfig(port=port, data_dir=data), tmp_path / "attempt")
        with pytest.raises(engine.Stop, match=f"parser port {port} is already listening"):
            service.launch()
    assert not (tmp_path / "attempt").exists()


def test_a_parser_url_must_be_loopback_by_name_or_by_ip_literal_unless_remote_is_allowed():
    """``require_local`` has three ways to accept (``allow_remote``, the name ``localhost``, a loopback IP literal) and two
    ways to refuse: a host that is not an IP literal at all, and an IP literal that parses but is not loopback. The refusals
    must not depend on the name failing to parse. Non-loopback literals come from the documentation prefix and the unspecified
    address, so no routable address is spelled in this file."""
    from memory_kernel.pipeline.engine import RemoteParserRefused, require_local

    for url in ("http://localhost:9801", "http://127.0.0.1:9801/spacy-api", "http://[::1]:9801"):
        require_local(url)
    for url in ("http://[2001:db8::1]:9801", "http://0.0.0.0:9801", "http://[::]:9801", "http://parser.invalid:9801"):
        with pytest.raises(RemoteParserRefused, match="not local"):
            require_local(url)
        require_local(url, allow_remote=True)


def test_an_attempt_runs_inside_the_managed_parser_which_is_always_torn_down(tmp_path, monkeypatch):
    from memory_kernel.pipeline.engine import EngineConfig, ParserConfig, Stop
    from memory_kernel.pipeline.stages import _parse_capture as pc

    calls: list[str] = []
    teardown = {"port_listening_after": False, "own_parser_after": []}

    class FakeService:
        def __init__(self, cfg, out):
            self.url = cfg.url

        def launch(self):
            calls.append("launch")
            if fail_launch:
                raise Stop("no model")
            return {"pid": 1, "ready_s": 1.0, "rss_kib": 1}

        def teardown(self):
            calls.append("teardown")
            return teardown

    monkeypatch.setattr(pc, "ParserService", FakeService)
    parser = ParserConfig(port=9801, data_dir=tmp_path)
    engine = EngineConfig(
        port=9701, binary=tmp_path, image=tmp_path, parser_url=parser.url, binary_sha256=None, image_sha256=None
    )
    site = pc.Site(root=tmp_path, engine=engine, parser_lock=tmp_path / "lock", parser=parser)

    def attempt():
        calls.append("attempt")
        return {"status": "complete"}

    fail_launch = False
    report = pc._with_parser(site, tmp_path, lambda _m: None, attempt)
    assert calls == ["launch", "attempt", "teardown"] and report["status"] == "complete"
    assert report["parser_service"]["teardown"] == teardown

    calls.clear()
    fail_launch = True
    report = pc._with_parser(site, tmp_path, lambda _m: None, attempt, base={"shard": [0, 1]})
    assert calls == ["launch", "teardown"]  # no attempt without a parser, and the teardown still runs
    assert report["status"] == "stopped: parser: no model"
    assert report["shard"] == [0, 1]

    calls.clear()
    fail_launch, teardown = False, {"port_listening_after": True, "own_parser_after": []}
    report = pc._with_parser(site, tmp_path, lambda _m: None, attempt)
    assert report["status"].startswith("stopped: parser teardown dirty")  # a parser left running fails the attempt

    calls.clear()
    external = pc.Site(root=tmp_path, engine=engine, parser_lock=tmp_path / "lock")
    assert pc._with_parser(external, tmp_path, lambda _m: None, attempt) == {"status": "complete"}
    assert calls == ["attempt"]  # an external parser: nothing is started


def test_the_served_parser_surface_is_the_two_syntactic_parser_paths_and_health_with_the_penelope_content_type(
    monkeypatch,
):
    """``create_app``: what the engine's HTTP client sees. The gate never starts a parser (the capture is seeded), so this
    is the public cover of the served surface; the handler's payload is stubbed (no spaCy model, no benepar)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from memory_kernel.parser_service import app as service
    from memory_kernel.parser_service import facade, registry

    loaded: list[str] = []
    monkeypatch.setattr(registry, "get_nlp", lambda: loaded.append("pipeline"))
    monkeypatch.setattr(facade, "_parse_one", lambda sentence: sentence)
    monkeypatch.setattr(facade.ft, "syntactic_parser_payload", lambda sentence: {"trees": [{"sentence": sentence}]})

    app = service.create_app()
    docs = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}  # FastAPI's own, not ours
    assert {route.path for route in app.routes} - docs == {
        "/spacy-api/syntactic-parser",
        "/syntactic-parser",
        "/health",
    }
    assert set(service.SYNTACTIC_PARSER_PATHS) == {"/spacy-api/syntactic-parser", "/syntactic-parser"}

    with TestClient(app) as client:  # the context manager runs the lifespan
        assert loaded == ["pipeline"]  # the spaCy pipeline is loaded at startup, before /health can answer
        health = client.get("/health")
        assert health.status_code == 200 and health.json() == {"status": "ok"}
        for path in service.SYNTACTIC_PARSER_PATHS:  # BOTH paths: the engine strips the /spacy-api prefix for 127.0.0.1
            for body in ({"sentence": "She left.", "model": "en_benepar"}, {"sentence": "She left."}):
                answer = client.post(path, json=body)
                assert answer.status_code == 200, (path, answer.text)
                assert answer.headers["content-type"] == "text/html; charset=utf-8"  # the engine only decodes text/*
                assert answer.json() == {"trees": [{"sentence": "She left."}]}
        assert client.post("/syntactic-parser", json={"model": "en"}).status_code == 422  # a sentence is required
        assert client.post("/health").status_code == 405
        for other in ("/spacy-api/tokenize", "/tokenize", "/spacy-api/texts-dependencies", "/embedding-api/embed"):
            assert client.post(other, json={"sentence": "x", "texts": ["x"], "model": "en"}).status_code == 404
    assert loaded == ["pipeline"]  # one load, not one per request


def test_a_parser_model_that_differs_from_the_pin_is_refused_and_nothing_is_placed(tmp_path):
    from memory_kernel import setup_runtime

    src = tmp_path / "benepar_en3"
    src.mkdir()
    for name in (
        "benepar_model.bin",
        "config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer_config.json",
    ):
        (src / name).write_text(name)
    data = tmp_path / "data"
    comp = setup_runtime.components()["parser-model"]
    with pytest.raises(setup_runtime.SetupError, match=r"parser-model files/benepar_model.bin: expected fc465e82"):
        comp.install(setup_runtime.Context(data), src)
    assert not (data / "nltk_data").exists()


def test_arming_the_engine_without_a_parser_host_is_refused(monkeypatch):
    from memory_kernel.construction.fcg import penelope_host

    monkeypatch.delenv(penelope_host.PENELOPE_HOST_ENV, raising=False)
    assert penelope_host.DEFAULT_PENELOPE_HOST is None
    with pytest.raises(RuntimeError, match="no parser host"):
        penelope_host.install_penelope_host_shim(object())
