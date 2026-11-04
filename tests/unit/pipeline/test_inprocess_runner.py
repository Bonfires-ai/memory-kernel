"""Stages run in the driver's process: the dispatch, the result file, the exception, the import audit and the
process-wide lock. Stub stages only; nothing here parses, spends or reaches Neo4j."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_kernel.pipeline import inprocess, runner
from memory_kernel.pipeline.config import BuildConfig
from memory_kernel.pipeline.stages import STAGES


@pytest.fixture(autouse=True)
def _in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests are about the in-process path, whatever the launching shell sets."""
    monkeypatch.delenv(inprocess.STAGE_CHILDREN_ENV, raising=False)


def _config(tmp_path: Path, **kw) -> BuildConfig:
    conversation = tmp_path / "conversation.json"
    sessions: dict = {}
    for k in (1, 2):
        sessions[f"session_{k}"] = [{"speaker": "Ada", "dia_id": f"D{k}:1", "text": f"the kiln is hot {k}"}]
        sessions[f"session_{k}_date_time"] = f"1:5{k} pm on {k} May, 2023"
    conversation.write_text(json.dumps({"sample_id": "conv-kiln", "conversation": sessions}))
    return BuildConfig(
        **{
            "sample_id": "conv-kiln",
            "conversation": conversation,
            "group": "g1-test-c2-kiln",
            "purpose": "remember the conversation",
            "workdir": tmp_path / "workdir",
            "root": tmp_path,
            "vault": tmp_path,
            **kw,
        }
    )


class RecordingRunner:
    """Stands in for ``subprocess.run`` of a stage child: records argv and writes a passing result."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv, **_kw):
        self.calls.append(list(argv))
        stage, result = argv[argv.index("_stage") + 1], Path(argv[argv.index("--result") + 1])
        mode = "live" if "--live" in argv else "plan"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps({"stage": stage, "mode": mode, "gate": None, "result": {"child": True}}))
        return SimpleNamespace(returncode=0)


def _kiln(seen: list):
    def stage(ctx, live):
        seen.append((ctx.cfg.sample_id, live, ctx.concurrency))
        return {"gate": {"name": "kiln", "pass": True}, "fired": Path("/kiln/fired")}

    return stage


def test_inprocess_stage_runs_without_a_child(tmp_path, monkeypatch):
    cfg, fake, seen = _config(tmp_path), RecordingRunner(), []
    monkeypatch.setitem(STAGES, "kiln", _kiln(seen))
    out = runner._invoke(cfg, "kiln", "live", fake, -1, 3, in_process=True)
    assert fake.calls == []
    assert seen == [("conv-kiln", True, 3)]
    written = json.loads(runner.result_path(cfg, "kiln", "live").read_bytes())
    assert set(written) == {"stage", "mode", "gate", "result"}
    assert written["stage"] == "kiln" and written["mode"] == "live"
    assert written["gate"] == {"name": "kiln", "pass": True}
    assert written["result"]["fired"] == "/kiln/fired"  # JSON shapes, exactly what the driver reads from a child
    assert out == written


def test_child_stage_still_spawns(tmp_path, monkeypatch):
    cfg, fake, seen = _config(tmp_path), RecordingRunner(), []
    monkeypatch.setitem(STAGES, "capture", _kiln(seen))
    assert "capture" in inprocess.CHILD_STAGES
    out = runner._invoke(cfg, "capture", "live", fake, -1, 3, in_process=True)
    (argv,) = fake.calls
    assert "_stage" in argv and argv[argv.index("_stage") + 1] == "capture"
    assert seen == []
    assert out["result"] == {"child": True}


def test_inprocess_exception_is_a_stop(tmp_path, monkeypatch, capsys):
    def boom(ctx, live):
        raise ValueError("boom")

    monkeypatch.setitem(STAGES, "kiln", boom)
    with pytest.raises(runner.Stop) as stop:
        runner._invoke(_config(tmp_path), "kiln", "live", RecordingRunner(), -1, 1, in_process=True)
    assert str(stop.value) == "kiln live: ValueError: boom"
    assert "Traceback" in capsys.readouterr().err  # the traceback goes to the build log


def test_a_spend_refusal_system_exit_in_process_is_a_stop(tmp_path, monkeypatch):
    def refuse(ctx, live):
        raise SystemExit("estimate 9 exceeds 1; not sending")

    monkeypatch.setitem(STAGES, "kiln", refuse)
    with pytest.raises(runner.Stop, match=r"^kiln live: SystemExit: estimate 9 exceeds 1; not sending$"):
        runner._invoke(_config(tmp_path), "kiln", "live", RecordingRunner(), -1, 1, in_process=True)


SECRET = "sk-kiln-glaze-0451"
NEO4J_SECRET = "kiln-neo4j-pw"


def test_a_secret_in_an_in_process_exception_is_redacted(tmp_path, monkeypatch, capsys):
    def leak(ctx, live):
        raise ValueError(f"bad key {SECRET} for {NEO4J_SECRET}")

    monkeypatch.setitem(STAGES, "kiln", leak)
    cfg = _config(tmp_path, openrouter_api_key=SECRET, neo4j_password=NEO4J_SECRET)
    with pytest.raises(runner.Stop) as stop:
        runner._invoke(cfg, "kiln", "live", RecordingRunner(), -1, 1, in_process=True)
    assert str(stop.value) == "kiln live: ValueError: bad key <redacted> for <redacted>"
    assert stop.value.__cause__ is None  # the original message never travels with the Stop
    err = capsys.readouterr().err
    assert "Traceback" in err and "bad key <redacted>" in err
    assert SECRET not in err and NEO4J_SECRET not in err


def test_a_secret_in_a_stage_stop_is_redacted(tmp_path, monkeypatch):
    def refuse(ctx, live):
        raise runner.Stop(f"kiln: key {SECRET} refused")

    monkeypatch.setitem(STAGES, "kiln", refuse)
    cfg = _config(tmp_path, openrouter_api_key=SECRET)
    with pytest.raises(runner.Stop, match=r"^kiln: key <redacted> refused$"):
        runner._invoke(cfg, "kiln", "live", RecordingRunner(), -1, 1, in_process=True)


def test_a_stop_in_process_passes_through_unchanged(tmp_path, monkeypatch):
    raised = runner.Stop("kiln: the glaze is not ready")

    def refuse(ctx, live):
        raise raised

    monkeypatch.setitem(STAGES, "kiln", refuse)
    cfg = _config(tmp_path, openrouter_api_key=SECRET)  # a secret that is not in the text changes nothing
    with pytest.raises(runner.Stop) as stop:
        runner._invoke(cfg, "kiln", "live", RecordingRunner(), -1, 1, in_process=True)
    assert stop.value is raised


def test_import_audit_written_in_process(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setitem(STAGES, "kiln", _kiln([]))
    runner._invoke(cfg, "kiln", "live", RecordingRunner(), -1, 1, in_process=True)
    audit = runner.path(cfg, "logs") / "mk" / "imports-kiln-live.json"
    assert audit.is_file()
    assert json.loads(audit.read_text()) == []  # no harness ``scripts`` module is loaded


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[BuildConfig, RecordingRunner, list]:
    """A build through ``select`` with a stub ``select`` body and no preflight."""
    cfg, fake, seen = _config(tmp_path, through="select"), RecordingRunner(), []
    monkeypatch.setattr(runner, "preflight", lambda cfg, **_: [])
    monkeypatch.setitem(STAGES, "select", _kiln(seen))
    return cfg, fake, seen


def test_a_build_runs_its_stages_in_process(tmp_path, monkeypatch):
    cfg, fake, seen = _build(tmp_path, monkeypatch)
    out = runner.run_build(cfg, runner=fake)
    assert out["status"] == "through"
    assert fake.calls == [] and len(seen) == 1
    events = [e["event"] for e in runner.read_events(cfg)]
    assert "in-process-busy" not in events and "stage-children" not in events
    assert runner.read_marker(cfg, "select")["gate"] == {"name": "kiln", "pass": True}
    with inprocess.build_lock() as held:
        assert held  # released when the build ends


def test_busy_lock_falls_back_to_children(tmp_path, monkeypatch):
    cfg, fake, seen = _build(tmp_path, monkeypatch)
    taken, release = threading.Event(), threading.Event()

    def other_build():
        with inprocess.build_lock() as held:
            assert held
            taken.set()
            release.wait(10)

    other = threading.Thread(target=other_build)
    other.start()
    try:
        assert taken.wait(10)
        out = runner.run_build(cfg, runner=fake)
    finally:
        release.set()
        other.join(10)
    assert out["status"] == "through"
    assert seen == [] and [argv[argv.index("_stage") + 1] for argv in fake.calls] == ["select"]
    assert "in-process-busy" in [e["event"] for e in runner.read_events(cfg)]


def test_stage_children_switch_forces_children(tmp_path, monkeypatch):
    cfg, fake, seen = _build(tmp_path, monkeypatch)
    monkeypatch.setenv(inprocess.STAGE_CHILDREN_ENV, "1")
    out = runner.run_build(cfg, runner=fake)
    assert out["status"] == "through"
    assert seen == [] and [argv[argv.index("_stage") + 1] for argv in fake.calls] == ["select"]
    (event,) = [e for e in runner.read_events(cfg) if e["event"] == "stage-children"]
    assert event["reason"] == "MK_STAGE_CHILDREN"


def _stage_cfg(tmp_path: Path) -> tuple[BuildConfig, runner.Counts]:
    return _config(tmp_path), runner.Counts(messages=2, sessions=2)


def test_plan_mode_stage_runs_once(tmp_path, monkeypatch):
    cfg, counts = _stage_cfg(tmp_path)
    calls: list[bool] = []

    def ctx_stage(ctx, live):
        calls.append(live)
        return {"gate": {"name": "kiln", "pass": True}}

    monkeypatch.setitem(STAGES, "ctx", ctx_stage)
    out = runner.run_stage(cfg, counts, "ctx", runner=RecordingRunner(), lock_fd=-1, in_process=True)
    assert calls == [True]
    assert out["gate"] == {"name": "kiln", "pass": True}
    events = [e["event"] for e in runner.read_events(cfg)]
    assert "plan" not in events and "live" in events
    assert not list(runner.path(cfg, "logs").glob("**/*-plan.json"))  # no driver plan result file


def test_boundary_like_gate_stops_in_one_pass(tmp_path, monkeypatch):
    cfg, counts = _stage_cfg(tmp_path)
    calls: list[bool] = []

    def boundary(ctx, live):
        # the real stage returns its failing gate before cap_check and any send, with or without ``live``
        calls.append(live)
        return {"plan": {}, "live": None, "gate": {"name": "x", "pass": False}}

    monkeypatch.setitem(STAGES, "boundary", boundary)
    with pytest.raises(runner.Stop, match="gate 'x' did not pass"):
        runner.run_stage(cfg, counts, "boundary", runner=RecordingRunner(), lock_fd=-1, in_process=True)
    assert calls == [True]  # one pass, and the gate stopped it before the stage could pay


def test_a_child_stage_is_called_once_too(tmp_path):
    cfg, counts = _stage_cfg(tmp_path)
    fake = RecordingRunner()
    runner.run_stage(cfg, counts, "ctx", runner=fake, lock_fd=-1, in_process=False)
    assert ["--live" in argv for argv in fake.calls] == [True]


# -- process-global state: nothing a stage or a build leaves behind reaches the next ---------------------------------


def _data_dir(root: Path, irregular: str) -> Path:
    """A data dir whose ``nltk_data`` holds a stub WordNet corpus: empty indexes, one ``verb.exc`` line."""
    corpus = root / "nltk_data" / "corpora" / "wordnet"
    corpus.mkdir(parents=True)
    (corpus / "lexnames").write_text("00\tadj.all\t3\n")
    for name in (
        "index.sense",
        *(f"{kind}.{pos}" for kind in ("index", "data") for pos in ("noun", "verb", "adj", "adv")),
    ):
        (corpus / name).write_text("")
    for pos in ("noun", "adj", "adv"):
        (corpus / f"{pos}.exc").write_text("")
    (corpus / "verb.exc").write_text(irregular + "\n")
    return root


@pytest.fixture
def _cold_wordnet():
    """The stub corpora must not outlive the test: every WordNet cache is cleared again afterwards."""
    yield
    inprocess.clear_build_caches()


def test_build_start_clears_data_dir_caches(tmp_path, monkeypatch, _cold_wordnet):
    import importlib

    render = importlib.import_module("memory_kernel.envelope.render")  # the package re-exports a ``render`` function
    seen: list[dict] = []

    def select(ctx, live):
        seen.append(dict(render._verb_exceptions()))  # a WordNet cache a stage reaches (verb agreement)
        return {"gate": {"name": "kiln", "pass": True}}

    monkeypatch.setattr(runner, "preflight", lambda cfg, **_: [])
    monkeypatch.setitem(STAGES, "select", select)
    for name, irregular in (("a", "fired fire"), ("b", "glazed glaze")):
        (tmp_path / name).mkdir()
        cfg = _config(tmp_path / name, through="select", data_dir=_data_dir(tmp_path / name / "data", irregular))
        assert runner.run_build(cfg, runner=RecordingRunner())["status"] == "through"
    assert seen == [{"fire": ("fired",)}, {"glaze": ("glazed",)}]  # each build reads its own data dir's corpus


def test_only_an_in_process_build_clears_the_caches(tmp_path, monkeypatch):
    cleared: list[str] = []
    monkeypatch.setattr(inprocess, "clear_build_caches", lambda: cleared.append("cleared"))
    monkeypatch.setattr(runner, "preflight", lambda cfg, **_: [])
    monkeypatch.setitem(STAGES, "select", _kiln([]))
    fake = RecordingRunner()

    def build(name: str) -> None:
        cfg = _config(tmp_path / name, through="select")
        assert runner.run_build(cfg, runner=fake)["status"] == "through"

    for name in ("in-process", "forced", "busy"):
        (tmp_path / name).mkdir()
    build("in-process")
    assert cleared == ["cleared"] and fake.calls == []
    monkeypatch.setenv(inprocess.STAGE_CHILDREN_ENV, "1")  # forced children: another build may be in-process
    build("forced")
    monkeypatch.delenv(inprocess.STAGE_CHILDREN_ENV)
    with inprocess.build_lock() as held:  # busy: the build holding the lock may be mid-stage
        assert held
        build("busy")
    assert cleared == ["cleared"] and len(fake.calls) == 2  # both ran their stage as a child


def test_a_relative_path_is_refused_before_any_stage(tmp_path):
    cfg = _config(tmp_path, through="select", snapshot_pins=Path("pins.json"), frame_dirs=(Path("frames"),))
    problems = runner.preflight(cfg)
    assert any("snapshot_pins" in p and "relative" in p for p in problems)
    assert any("frame_dirs" in p and "relative" in p for p in problems)
    assert not [p for p in runner.preflight(_config(tmp_path, through="select")) if "relative" in p]


class _FlakyModel:
    """A spaCy stand-in whose load or parse fails until ``healed``."""

    def __init__(self) -> None:
        self.healed = False

    def load(self, *_a, **_kw):
        if not self.healed:
            raise MemoryError("the kiln ran out of room")
        return self

    def __call__(self, text):
        if not self.healed:
            raise MemoryError("the glaze would not set")
        return SimpleNamespace(text=text, kiln=True)


def test_clear_build_caches_forgets_memoized_failures(monkeypatch):
    import importlib

    so = importlib.import_module("memory_kernel.grammar.semantic_operators")
    ws = importlib.import_module("memory_kernel.grammar.wordnet_semantics")
    model = _FlakyModel()
    loaders = (so._load_nlp, ws._load_nlp, ws._porter_stemmer)
    docs = (so._doc_cached, ws._doc)
    for f in (*loaders, *docs):
        f.cache_clear()
    for module in (so, ws):
        monkeypatch.setattr(module, "load_spacy_model", model.load)
    monkeypatch.setattr(so, "doc_to_parsed", lambda doc, text: doc)
    monkeypatch.setitem(sys.modules, "nltk.stem", None)  # the stemmer import fails
    assert [f() for f in loaders] == [None, None, None]  # each load failure is memoized
    assert [f("the kiln") for f in docs] == [None, None]  # and so is each text's degraded parse
    model.healed = True
    monkeypatch.delitem(sys.modules, "nltk.stem")
    assert [f() for f in loaders] == [None, None, None]  # healed, but still memoized

    inprocess.clear_build_caches()

    assert so._load_nlp() is model and ws._load_nlp() is model
    assert type(ws._porter_stemmer()).__name__ == "PorterStemmer"
    assert all(f("the kiln").kiln for f in docs)  # retried, not the memoized None
    monkeypatch.undo()
    inprocess.clear_build_caches()  # the stand-in never outlives the test


def test_the_wordnet_reader_is_swapped_before_the_memos_are_cleared(monkeypatch):
    import importlib

    import nltk.corpus

    render = importlib.import_module("memory_kernel.envelope.render")
    old = nltk.corpus.wordnet
    seen: list[bool] = []
    monkeypatch.setattr(
        render, "_verb_exceptions", SimpleNamespace(cache_clear=lambda: seen.append(nltk.corpus.wordnet is old))
    )
    inprocess.clear_build_caches()
    assert seen == [False]  # a memo cleared while the old reader was still installed could refill from it
    monkeypatch.undo()
    inprocess.clear_build_caches()
