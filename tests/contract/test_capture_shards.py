import sys
import types
from pathlib import Path

import pytest

from memory_kernel.pipeline import engine as eng


def test_reset_ids_sends_the_reset_form(monkeypatch):
    sent = []
    fake = types.SimpleNamespace(routes=types.SimpleNamespace(lisp_eval=lambda code: sent.append(code) or 0))
    monkeypatch.setitem(sys.modules, "pyfcg", fake)
    e = eng.Engine.__new__(eng.Engine)
    e.reset_ids()
    assert sent == [eng.RESET_IDS_FORM]


def test_reset_ids_refuses_a_wrong_answer(monkeypatch):
    fake = types.SimpleNamespace(routes=types.SimpleNamespace(lisp_eval=lambda code: "error"))
    monkeypatch.setitem(sys.modules, "pyfcg", fake)
    e = eng.Engine.__new__(eng.Engine)
    with pytest.raises(eng.Stop):
        e.reset_ids()


def test_select_own_keeps_only_this_port():
    procs = [
        {"pid": 1, "comm": "FCG Go", "argv": ["/x/FCG Go", "--port", "9701", "--address", "127.0.0.1"]},
        {"pid": 2, "comm": "FCG Go", "argv": ["/x/FCG Go", "--port", "9702", "--address", "127.0.0.1"]},
        {"pid": 3, "comm": "python3", "argv": ["python3", "-c", "import pyfcg"]},
    ]
    assert [p["pid"] for p in eng._select_own(procs, 9702)] == [2]
    assert [p["pid"] for p in eng._select_own(procs, None)] == [1, 2, 3]


def test_teardown_scopes_the_stray_check_to_its_port(monkeypatch):
    sibling = {"pid": 2, "comm": "FCG Go", "argv": ["/x/FCG Go", "--port", "9702", "--address", "127.0.0.1"]}
    seen = []

    def fake_scan(exclude=(), port=None):
        seen.append(port)
        return eng._select_own([sibling], port)

    monkeypatch.setattr(eng, "own_fcg_processes", fake_scan)
    monkeypatch.setattr(eng, "port_listening", lambda port: False)
    e = eng.Engine.__new__(eng.Engine)
    e.port, e.process, e.log, e.receipts = 9701, None, None, None
    report = e.teardown()
    assert seen == [9701]
    assert report["own_fcg_after"] == []


# -- parallel capture shards -----------------------------------------------------------------------------------


def test_effective_shards_clamps_to_usable_cpus():
    from memory_kernel.pipeline.stages import _capture_shards as cs

    assert cs.effective_shards(4, usable=2) == 2
    assert cs.effective_shards(2, usable=16) == 2
    assert cs.effective_shards(1, usable=1) == 1


def test_shard_ports_stay_private_and_distinct():
    from memory_kernel.pipeline.stages import _capture_shards as cs

    assert cs.shard_ports(9701, 4) == [9701, 9702, 9703, 9704]
    with pytest.raises(ValueError):
        cs.shard_ports(9798, 4)  # would leave the 97xx range


def test_resume_with_other_shard_count_refused(tmp_path):
    from memory_kernel.pipeline.stages import _capture_shards as cs

    (tmp_path / "report-0-of-2-a0.json").write_text('{"status": "complete", "shard": [0, 2]}')
    with pytest.raises(ValueError, match="shard count"):
        cs.check_shard_count(tmp_path, 4)
    cs.check_shard_count(tmp_path, 2)


def test_file_logger_appends_lines(tmp_path):
    from memory_kernel.pipeline.stages import _capture_shards as cs

    log = cs.file_logger(tmp_path / "shards.log")
    log("a")
    log("b")
    assert (tmp_path / "shards.log").read_text() == "a\nb\n"


def test_capture_shards_bounds():
    from memory_kernel.pipeline.config import BuildConfig

    def make(**kw):
        return BuildConfig(
            sample_id="c",
            conversation=Path("/x/c.json"),
            group="g",
            purpose="p",
            workdir=Path("/x/w"),
            root=Path("/x"),
            **kw,
        )

    assert make().capture_shards == 1
    assert make(capture_shards=8).capture_shards == 8
    for bad in (0, 9, -1):
        with pytest.raises(ValueError, match="capture_shards"):
            make(capture_shards=bad)


def test_cli_declares_the_flag_and_the_hidden_shard_subcommand():
    from memory_kernel import cli

    p = cli.parser()
    ns = p.parse_args(["_capture-shard", "/o", "1", "4", "9702", "{}"])
    assert (ns.out, ns.i, ns.k, ns.port, ns.site_json) == ("/o", 1, 4, 9702, "{}")
    assert ns.func is cli.cmd_capture_shard


def _site(tmp_path):
    from memory_kernel.pipeline.stages import _parse_capture as pc

    ecfg = eng.EngineConfig(
        port=9701,
        binary=tmp_path / "bin",
        image=tmp_path / "img",
        parser_url="http://127.0.0.1:9801",
        binary_sha256="a" * 64,
        image_sha256="b" * 64,
    )
    return pc.Site(root=tmp_path, engine=ecfg, parser_lock=tmp_path / "p.lock", parser=None)


def test_cmd_capture_shard_runs_one_shard_on_its_port(tmp_path, monkeypatch):
    import json

    from memory_kernel import cli
    from memory_kernel.pipeline.stages import _capture_shards as cs
    from memory_kernel.pipeline.stages import _parse_capture as pc

    seen = {}

    def fake_run(out, shard, *, site):
        seen.update(out=out, shard=shard, port=site.engine.port, parser=site.parser, url=site.live_parser)
        return {"status": "complete"}

    monkeypatch.setattr(pc, "run", fake_run)
    args = cli.parser().parse_args(
        ["_capture-shard", str(tmp_path), "1", "2", "9702", json.dumps(cs.site_doc(_site(tmp_path)))]
    )
    assert cli.cmd_capture_shard(args) == 0
    assert seen == {"out": tmp_path, "shard": (1, 2), "port": 9702, "parser": None, "url": "http://127.0.0.1:9801"}
    assert json.loads(cs.result_path(tmp_path, 1, 2).read_text()) == {"shard": [1, 2], "status": "complete"}


class _FakeChild:
    """What run_parallel touches of a Popen; the child 'ran' already: it wrote its result file (or crashed without)."""

    pid = 1

    def __init__(self, rc):
        self.returncode = rc

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        pass


def _fake_popen(out, spawned, *, rc=0, write_result=True):
    import json

    from memory_kernel.pipeline.stages import _capture_shards as cs

    def popen(argv, **kw):
        i, k = int(argv[argv.index("_capture-shard") + 2]), int(argv[argv.index("_capture-shard") + 3])
        spawned.append(argv)
        assert kw["stdout"].name == str(cs.log_path(out, i, k)) and kw["stderr"] == __import__("subprocess").STDOUT
        if write_result:
            cs.result_path(out, i, k).write_text(json.dumps({"shard": [i, k], "status": "complete"}))
        return _FakeChild(rc)

    return popen


def test_run_parallel_spawns_k_children_and_reads_their_result_files(tmp_path, monkeypatch):
    import subprocess

    from memory_kernel.pipeline.stages import _capture_shards as cs

    argvs, lines = [], []
    monkeypatch.setattr(subprocess, "Popen", _fake_popen(tmp_path, argvs))
    reports = cs.run_parallel(tmp_path, _site(tmp_path), 3, log=lines.append)
    assert reports == [{"shard": [i, 3], "status": "complete"} for i in range(3)]
    assert [a[a.index("_capture-shard") + 1 :][1:4] for a in argvs] == [
        ["0", "3", "9701"],
        ["1", "3", "9702"],
        ["2", "3", "9703"],
    ]
    assert all(a[1] == "-P" for a in argvs) and len(lines) == 3


def test_run_parallel_skips_complete_shards(tmp_path, monkeypatch):
    import subprocess

    from memory_kernel.pipeline.stages import _capture_shards as cs

    (tmp_path / "report-0-of-2-a0.json").write_text('{"status": "complete", "shard": [0, 2], "attempt": 0}')
    argvs = []
    monkeypatch.setattr(subprocess, "Popen", _fake_popen(tmp_path, argvs))
    reports = cs.run_parallel(tmp_path, _site(tmp_path), 2, log=lambda _m: None)
    assert [a[-4] for a in argvs] == ["1"] and reports == [{"shard": [1, 2], "status": "complete"}]


def test_a_child_that_crashes_without_a_result_file_is_reported_with_its_log(tmp_path, monkeypatch):
    import subprocess

    from memory_kernel.pipeline.stages import _capture_shards as cs

    monkeypatch.setattr(subprocess, "Popen", _fake_popen(tmp_path, [], rc=3, write_result=False))
    (report,) = cs.run_parallel(tmp_path, _site(tmp_path), 1, log=lambda _m: None)
    assert report["status"] == f"crashed rc=3; see {cs.log_path(tmp_path, 0, 1)}"


def test_a_clean_exit_with_a_missing_or_unparseable_result_is_a_crash(tmp_path, monkeypatch):
    import subprocess

    from memory_kernel.pipeline.stages import _capture_shards as cs

    monkeypatch.setattr(subprocess, "Popen", _fake_popen(tmp_path, [], rc=0, write_result=False))
    (report,) = cs.run_parallel(tmp_path, _site(tmp_path), 1, log=lambda _m: None)
    assert report["status"].startswith("crashed rc=0; see ") and "shard-0-of-1.log" in report["status"]
    cs.result_path(tmp_path, 0, 1).write_text("not json {")
    assert "crashed" in cs.shard_report(tmp_path, 0, 1, 0)["status"]


def test_a_stale_result_file_is_not_trusted(tmp_path, monkeypatch):
    import subprocess

    from memory_kernel.pipeline.stages import _capture_shards as cs

    cs.result_path(tmp_path, 0, 1).write_text('{"shard": [0, 1], "status": "complete"}')
    monkeypatch.setattr(subprocess, "Popen", _fake_popen(tmp_path, [], rc=0, write_result=False))
    (report,) = cs.run_parallel(tmp_path, _site(tmp_path), 1, log=lambda _m: None)
    assert "crashed" in report["status"]


def _sleepers(monkeypatch, on_first_wait, spawned, *, script="import time; time.sleep(60)"):
    """Real ``python -c`` sleepers in place of the shard children; the first untimed ``wait()`` runs ``on_first_wait``."""
    import subprocess
    import sys

    real = subprocess.Popen

    class Sleeper(real):
        first = True

        def wait(self, timeout=None):
            if timeout is None and Sleeper.first:
                Sleeper.first = False
                on_first_wait()
            return super().wait(timeout)

    def popen(argv, **kw):
        child = Sleeper([sys.executable, "-c", script], **kw)
        spawned.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", popen)


def test_run_parallel_terminates_running_children_when_the_wait_raises(tmp_path, monkeypatch):
    from memory_kernel.pipeline.stages import _capture_shards as cs

    def boom():
        raise RuntimeError("boom mid-wait")

    spawned = []
    _sleepers(monkeypatch, boom, spawned)
    with pytest.raises(RuntimeError, match="boom mid-wait"):
        cs.run_parallel(tmp_path, _site(tmp_path), 2, log=lambda _m: None)
    assert len(spawned) == 2 and all(c.poll() is not None for c in spawned)
    assert [c.returncode for c in spawned] == [-15, -15]  # SIGTERM, not left running


def test_sigterm_to_the_parent_unwinds_and_stops_the_children(tmp_path, monkeypatch):
    import os
    import signal

    from memory_kernel.pipeline.stages import _capture_shards as cs

    spawned = []
    _sleepers(monkeypatch, lambda: os.kill(os.getpid(), signal.SIGTERM), spawned)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(cs.Terminated):
        cs.run_parallel(tmp_path, _site(tmp_path), 2, log=lambda _m: None)
    assert all(c.poll() is not None for c in spawned) and len(spawned) == 2
    assert signal.getsignal(signal.SIGTERM) == before  # the previous handler is restored


def test_a_child_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path):
    import subprocess
    import sys
    import time

    from memory_kernel.pipeline.stages import _capture_shards as cs

    ready = tmp_path / "ready"
    code = (
        "import signal, time, pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).write_text('x'); time.sleep(60)"
    )
    child = subprocess.Popen([sys.executable, "-c", code])
    deadline = time.monotonic() + 20
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    cs.stop_children([child], grace_s=0.5)
    assert child.returncode == -9


def test_run_parallel_refuses_ports_outside_the_private_range(tmp_path):
    from memory_kernel.pipeline.stages import _capture_shards as cs

    site = _site(tmp_path)
    object.__setattr__(site.engine, "port", 9799)
    with pytest.raises(ValueError):
        cs.run_parallel(tmp_path, site, 2, log=lambda _m: None)


class _FakeService:
    teardown_receipt: dict = {}
    events: list = []

    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, out
        self.url = "http://127.0.0.1:9801"

    def launch(self):
        self.events.append("launch")
        return {}

    def teardown(self):
        self.events.append("teardown")
        return self.teardown_receipt


def _managed_site(tmp_path):
    from memory_kernel.pipeline.stages import _parse_capture as pc

    site = _site(tmp_path)
    return pc.Site(
        root=site.root,
        engine=site.engine,
        parser_lock=site.parser_lock,
        parser=eng.ParserConfig(port=9801, data_dir=None),
    )


def test_managed_parser_checks_the_teardown_receipt(tmp_path, monkeypatch):
    from memory_kernel.pipeline.stages import parse

    monkeypatch.setattr(parse, "ParserService", _FakeService)
    _FakeService.events = []
    _FakeService.teardown_receipt = {"port_listening_after": False, "own_parser_after": []}
    with parse._managed_parser(_managed_site(tmp_path), tmp_path):
        pass
    assert _FakeService.events == ["launch", "teardown"]

    _FakeService.teardown_receipt = {"port_listening_after": True, "own_parser_after": []}
    with pytest.raises(RuntimeError, match="teardown dirty"), parse._managed_parser(_managed_site(tmp_path), tmp_path):
        pass


def test_managed_parser_is_inert_for_an_external_parser(tmp_path, monkeypatch):
    from memory_kernel.pipeline.stages import parse

    monkeypatch.setattr(parse, "ParserService", lambda *a: pytest.fail("no parser may start"))
    with parse._managed_parser(_site(tmp_path), tmp_path):
        pass


def _capture_ctx(tmp_path, shards):
    import types

    cfg = types.SimpleNamespace(
        capture_shards=shards,
        snapshot_pins=tmp_path / "pins",
        conv_root=tmp_path,
        root=tmp_path,
        capture_cache=None,
        capture_slot=None,
    )
    paths = {n: tmp_path / n for n in ("syntax", "capture", "runs", "select")}
    (paths["select"]).mkdir()
    (paths["select"] / "sources.json").write_text("[]")
    (paths["capture"]).mkdir()
    (paths["capture"] / "manifest.json").write_text("{}")
    return types.SimpleNamespace(cfg=cfg, path=paths.__getitem__), paths


def _patch_capture(monkeypatch, tmp_path, calls):
    from memory_kernel.pipeline.stages import _capture_shards as cs
    from memory_kernel.pipeline.stages import _parse_capture as pc
    from memory_kernel.pipeline.stages import parse

    monkeypatch.setattr(parse, "site_for", lambda ctx: _managed_site(tmp_path))
    monkeypatch.setattr(pc, "syntax_missing", lambda *a: [])
    monkeypatch.setattr(parse, "conserve", lambda *a: {"merged_units": 5})
    monkeypatch.setattr(parse, "_compat_valid", lambda runs: True)
    monkeypatch.setattr(cs, "usable_cpus", lambda: 16)
    monkeypatch.setattr(pc, "run", lambda out, shard, *, site: calls.append(("run", shard)) or {"status": "complete"})
    monkeypatch.setattr(
        pc, "gated_merge", lambda out, site: (out / "merge.json").write_text('{"counts": {"units": 5}}')
    )
    monkeypatch.setattr(parse, "_managed_parser", lambda site, out: __import__("contextlib").nullcontext())
    return parse, cs


def test_capture_with_one_shard_takes_the_single_engine_path(tmp_path, monkeypatch):
    calls = []
    parse, cs = _patch_capture(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(cs, "run_parallel", lambda *a, **k: pytest.fail("k=1 must not shard"))
    ctx, _ = _capture_ctx(tmp_path, 1)
    result = parse.capture(ctx)
    assert calls == [("run", (0, 1))]
    assert result["capture_shards"] == 1 and result["capture_shards_requested"] == 1


def test_capture_with_shards_runs_the_children_then_merges(tmp_path, monkeypatch):
    calls = []
    parse, cs = _patch_capture(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        cs,
        "run_parallel",
        lambda out, site, k, log: calls.append(("parallel", k, site.parser)) or [{"status": "complete"}] * k,
    )
    ctx, _ = _capture_ctx(tmp_path, 4)
    result = parse.capture(ctx)
    assert calls == [("parallel", 4, None)]  # the children get a parser-less site: the parent owns the parser
    assert result["capture_shards"] == 4 and (tmp_path / "capture" / "merge.json").exists()


def test_capture_with_a_failed_shard_raises_before_the_merge(tmp_path, monkeypatch):
    calls = []
    parse, cs = _patch_capture(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        cs, "run_parallel", lambda out, site, k, log: [{"status": "complete"}, {"status": "crashed rc=1"}]
    )
    ctx, _ = _capture_ctx(tmp_path, 2)
    with pytest.raises(RuntimeError, match="shards not complete"):
        parse.capture(ctx)
    assert not (tmp_path / "capture" / "merge.json").exists()


def test_capture_clamps_the_shard_count_to_the_usable_cpus(tmp_path, monkeypatch):
    calls = []
    parse, cs = _patch_capture(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(cs, "usable_cpus", lambda: 1)
    ctx, _ = _capture_ctx(tmp_path, 4)
    result = parse.capture(ctx)
    assert calls == [("run", (0, 1))] and result["capture_shards"] == 1 and result["capture_shards_requested"] == 4


def test_shard_child_imports_this_source_tree(tmp_path):
    import subprocess
    import sys

    from memory_kernel.pipeline.stages import _capture_shards as cs

    done = subprocess.run(
        [sys.executable, "-P", "-m", "memory_kernel.cli", "_capture-shard", "--help"],
        cwd=tmp_path,
        env=cs.child_env(),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0 and "site_json" in done.stdout


def test_resume_uses_the_k_of_the_existing_reports_not_the_clamp(tmp_path, monkeypatch):
    calls = []
    parse, cs = _patch_capture(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(cs, "usable_cpus", lambda: 1)  # the clamp alone would give 1
    ctx, paths = _capture_ctx(tmp_path, 4)
    (paths["capture"] / "report-0-of-2-a0.json").write_text('{"status": "complete", "shard": [0, 2]}')
    (paths["capture"] / "report-1-of-2-a0.json").write_text('{"status": "stopped: x", "shard": [1, 2]}')
    monkeypatch.setattr(
        cs, "run_parallel", lambda out, site, k, log: calls.append(("parallel", k)) or [{"status": "complete"}]
    )
    result = parse.capture(ctx)
    assert calls == [("parallel", 2)]
    assert result["capture_shards"] == 2 and result["capture_shards_requested"] == 4
    assert result["capture_shards_fixed_by_reports"] is True


def test_a_merged_capture_skips_the_shard_count_check_and_runs_nothing(tmp_path, monkeypatch):
    calls = []
    parse, cs = _patch_capture(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(cs, "usable_cpus", lambda: 1)
    monkeypatch.setattr(cs, "check_shard_count", lambda *a: pytest.fail("a merged capture is not re-checked"))
    monkeypatch.setattr(cs, "run_parallel", lambda *a, **k: pytest.fail("a merged capture runs nothing"))
    ctx, paths = _capture_ctx(tmp_path, 1)
    (paths["capture"] / "report-0-of-2-a0.json").write_text('{"status": "complete", "shard": [0, 2]}')
    (paths["capture"] / "report-1-of-2-a0.json").write_text('{"status": "complete", "shard": [1, 2]}')
    (paths["capture"] / "merge.json").write_text('{"counts": {"units": 5}}')
    result = parse.capture(ctx)
    assert calls == [] and result["capture_shards"] == 2


def test_all_shards_complete_starts_no_parser_and_goes_to_the_merge(tmp_path, monkeypatch):
    calls = []
    parse, cs = _patch_capture(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(parse, "_managed_parser", lambda *a: pytest.fail("no parser for a merge-only resume"))
    monkeypatch.setattr(cs, "run_parallel", lambda *a, **k: pytest.fail("nothing left to run"))
    ctx, paths = _capture_ctx(tmp_path, 2)
    for i in (0, 1):
        (paths["capture"] / f"report-{i}-of-2-a0.json").write_text(f'{{"status": "complete", "shard": [{i}, 2]}}')
    result = parse.capture(ctx)
    assert (paths["capture"] / "merge.json").exists() and result["capture_shards"] == 2


def test_managed_parser_is_torn_down_when_the_body_raises(tmp_path, monkeypatch):
    from memory_kernel.pipeline.stages import parse

    monkeypatch.setattr(parse, "ParserService", _FakeService)
    _FakeService.events = []
    _FakeService.teardown_receipt = {"port_listening_after": False, "own_parser_after": []}
    with pytest.raises(KeyError), parse._managed_parser(_managed_site(tmp_path), tmp_path):
        raise KeyError("body")
    assert _FakeService.events == ["launch", "teardown"]
