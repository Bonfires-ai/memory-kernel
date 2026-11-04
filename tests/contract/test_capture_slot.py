"""The warm capture slot's lease protocol: one lease at a time, a busy wait that ends in ``slot busy``, a dropped
lease recycles the engine, pins from another install vacate the slot, and no socket means the capture runs cold. Then
the capture side of a warm attempt: the slot gate, the attach identity check and the merge gate's release receipt.

All in-process against a fake backend, fake /proc entries and a faked-engine capture; nothing here starts an FCG engine
or a parser."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from memory_kernel.pipeline import engine, slot
from memory_kernel.pipeline.stages import _parse_capture as pc
from unit.pipeline import test_capture_cache as tcc

PINS = {
    "binary_sha256": "b" * 64,
    "image_sha256": "i" * 64,
    "binary_path": "/opt/toy/FCG Go",
    "image_path": "/opt/toy/grammar.image",
    "engine_port": 9701,
    "live_parser": "http://127.0.0.1:9801",
    "parser_data_dir": "/opt/toy/data",
    "arm_env": dict.fromkeys(slot.ARM_ENV),
    "code_pins": {"src/memory_kernel/pipeline/slot.py": "c" * 64},
    "walker_sha256": "w" * 64,
}


class FakeBackend:
    """Counts calls; ``health()`` answers from a script (then no problems), ``pins()`` from ``PINS``."""

    def __init__(self, health_script=()):
        self.calls = {"boot": 0, "teardown": 0, "health": 0, "rest": 0}
        self.health_script = list(health_script)

    def boot(self):
        self.calls["boot"] += 1
        return {"engine": {"launch": self.calls["boot"]}, "arm": {"grammar": "toy-grammar"}}

    def identities(self):
        n = self.calls["boot"]
        engine_id = {"pid": 4100 + n, "start_ticks": n, "exe": "/opt/toy/fcg", "port": 9701}
        parser_id = {"pid": 5100 + n, "start_ticks": n, "exe": "/opt/toy/python", "url": PINS["live_parser"]}
        return engine_id, parser_id

    def health(self):
        self.calls["health"] += 1
        return self.health_script.pop(0) if self.health_script else []

    def rest(self):
        self.calls["rest"] += 1
        return []

    def rss_kib(self):
        return 1024

    def pins(self):
        return dict(PINS)

    def teardown(self):
        self.calls["teardown"] += 1
        return {"port_free": True}


def _request(**changes):
    fields = dict(PINS, **changes)
    return slot.LeaseRequest(**fields)


@contextmanager
def _serving(backend):
    # A short /tmp dir: a unix socket path is limited to ~107 bytes, which pytest's tmp_path can exceed.
    root = Path(tempfile.mkdtemp(prefix="mks", dir="/tmp"))
    sock = root / "slot.sock"
    server = slot.SlotServer(sock, backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield sock, thread
    finally:
        server.shutdown()
        thread.join(5)
        shutil.rmtree(root, ignore_errors=True)


def test_grant_warm():
    backend = FakeBackend()
    with _serving(backend) as (sock, _), slot.lease(sock, _request()) as held:
        assert held.grant.kind == "warm"
        assert held.grant.engine["pid"] == 4101 and held.grant.parser["url"] == PINS["live_parser"]
        assert held.grant.boot["pins"] == PINS and held.grant.served == {"leases": 1, "jobs": 0}
        assert backend.calls["boot"] == 1
        assert held.release("complete", 3) == {"recycle": False, "why": []}


def test_busy_waits_then_stops():
    backend = FakeBackend()
    with _serving(backend) as (sock, _):
        with slot.lease(sock, _request()) as held:
            with pytest.raises(slot.SlotBusy), slot.lease(sock, _request(), wait_s=0.5):
                pass
            assert issubclass(slot.SlotBusy, engine.Stop)
            held.release("complete", 1)
        # the waiter that gave up ran no attempt: the next lease is warm on the same engine, nothing recycled
        with slot.lease(sock, _request()) as again:
            assert again.grant.kind == "warm" and again.grant.served == {"leases": 2, "jobs": 1}
            again.release("complete", 0)
        assert backend.calls == {"boot": 1, "teardown": 0, "health": 2, "rest": 2}


def test_dropped_lease_recycles():
    backend = FakeBackend()
    with _serving(backend) as (sock, _):
        with slot.lease(sock, _request()):
            pass  # the connection closes without a release
        with slot.lease(sock, _request()) as again:
            assert backend.calls["teardown"] == 1 and backend.calls["boot"] == 2
            assert again.grant.kind == "warm" and again.grant.engine["pid"] == 4102
            again.release("complete", 0)


def test_code_pins_differ_grant_cold():
    backend = FakeBackend()
    newer = {"src/memory_kernel/pipeline/slot.py": "d" * 64}
    with _serving(backend) as (sock, thread):
        with slot.lease(sock, _request(code_pins=newer)) as held:
            assert held.grant.kind == "cold" and held.grant.engine is None and held.grant.parser is None
            assert held.grant.boot == {} and backend.calls["teardown"] == 1
            held.release("complete", 2)
        thread.join(2)
        assert not thread.is_alive()
        assert backend.calls["boot"] == 1
    # the same files under another path (a copy, even byte-identical): the leased attempt would refuse the exe; another
    # engine port: the slot gate would stop every warm attempt; another arming environment: the shims differ. All cold.
    other_env = dict(PINS["arm_env"], MK_FCG_NO_NEIGHBORS_GUARD="1")
    for change in ({"binary_path": "/srv/other/FCG Go"}, {"engine_port": 9702}, {"arm_env": other_env}):
        backend = FakeBackend()
        with _serving(backend) as (sock, thread):
            with slot.lease(sock, _request(**change)) as held:
                assert held.grant.kind == "cold" and held.grant.engine is None, change
                assert backend.calls["teardown"] == 1, change
                held.release("complete", 0)
            thread.join(2)
            assert not thread.is_alive(), change


def test_want_cold_reboots_after():
    backend = FakeBackend()
    with _serving(backend) as (sock, _):
        with slot.lease(sock, _request(want="cold")) as held:
            assert held.grant.kind == "cold" and backend.calls["teardown"] == 1
            held.release("complete", 4)
        with slot.lease(sock, _request()) as again:
            assert backend.calls["boot"] == 2 and again.grant.kind == "warm"
            again.release("complete", 0)


def test_unhealthy_recycles_before_grant():
    backend = FakeBackend(health_script=[["readback: searches 1"]])
    with _serving(backend) as (sock, _), slot.lease(sock, _request()) as held:
        assert held.grant.kind == "warm"
        assert backend.calls["teardown"] == 1 and backend.calls["boot"] == 2 and backend.calls["health"] == 2
        held.release("complete", 0)


def test_rest_overrun_recycles(monkeypatch):
    monkeypatch.setattr(slot, "REST_DEADLINE_S", 0.3)
    killed = threading.Event()

    class StuckRest(FakeBackend):
        def rest(self):  # the first rest takes the call and never answers, until the engine is killed (teardown)
            self.calls["rest"] += 1
            if self.calls["rest"] == 1:
                killed.wait(10)
                raise ConnectionError("engine gone")
            return []

        def teardown(self):
            killed.set()
            return super().teardown()

    backend = StuckRest()
    with _serving(backend) as (sock, _):
        with slot.lease(sock, _request()) as held:
            reply = held.release("complete", 1)
        assert reply["recycle"] and "no answer within 0.3 s" in reply["why"][0]
        with slot.lease(sock, _request()) as again:  # torn down at the overrun, then recycled before this grant
            assert backend.calls["boot"] == 2 and backend.calls["teardown"] == 3
            assert again.grant.kind == "warm" and again.grant.engine["pid"] == 4102
            again.release("complete", 0)


def test_no_socket_is_unavailable():
    root = Path(tempfile.mkdtemp(prefix="mks", dir="/tmp"))
    try:
        with pytest.raises(slot.SlotUnavailable), slot.lease(root / "absent.sock", _request()):
            pass
    finally:
        shutil.rmtree(root, ignore_errors=True)


# -- a warm attempt in the capture: attach, gate, release receipt, merge gate ------------------------------------------
# The merge-gate fixtures are a real (faked-engine) capture from the capture-cache suite, whose one report is then
# rewritten to end in a release receipt instead of a teardown.

world = tcc.world
#: invented processes: pids above any pid_max, so they never meet this process's own pid
ENGINE_ID = {"pid": 7100001, "start_ticks": "71", "exe": "/opt/toy/FCG Go"}
PARSER_ID = {"pid": 7100002, "start_ticks": "72", "exe": "/opt/toy/python3"}


def _grant(engine_id=ENGINE_ID, parser_id=PARSER_ID):
    return slot.Grant(
        kind="warm",
        engine={**engine_id, "port": 9701},
        parser={**parser_id, "url": PINS["live_parser"]},
        boot={"pins": dict(PINS)},
        served={"leases": 1, "jobs": 0},
    )


def _release(**changes):
    grant = _grant()
    receipt = {
        "engine": {**ENGINE_ID, "ppid": 1, "uid": 1000, "argv": ["fcg"]},
        "parser": {**PARSER_ID, "ppid": 1, "uid": 1000, "argv": ["python3"]},
        "own_fcg_other": [],
        "own_parser_other": [],
        "granted": {"engine": grant.engine, "parser": grant.parser},
    }
    return {**receipt, **changes}


def _captured_report(world, *, end: dict | None):
    """A merged k=1 capture whose only shard report ends with ``end`` (``{"release": ...}``, or None: no receipt)."""
    out, _ = tcc._capture(world, "run-1", 2, cache=None)
    path = out / "report-0-of-1-a0.json"
    report = json.loads(path.read_bytes())
    assert report["status"] == "complete" and "teardown" in report
    report.pop("teardown")
    path.write_text(json.dumps({**report, **(end or {})}))
    return out, path


def test_merge_gate_accepts_clean_release(world):
    out, _ = _captured_report(world, end={"release": _release()})
    gate = pc.merge_gate(out, world.site)
    assert gate["problems"] == [] and gate["ignored"] == []
    assert gate["used"] == [{"file": "report-0-of-1-a0.json", "status": "complete", "i": 0, "k": 1}]


def test_merge_gate_refuses_dirty_release(world):
    stray = {"pid": 7100009, "comm": "FCG Go", "argv": ["/opt/toy/FCG Go", "--port", "9701"]}
    out, _ = _captured_report(world, end={"release": _release(own_fcg_other=[stray])})
    gate = pc.merge_gate(out, world.site)
    assert gate["used"] == [] and gate["problems"]
    (entry,) = gate["ignored"]
    assert entry["file"] == "report-0-of-1-a0.json" and entry["status"] == "complete"
    assert any("7100009" in p for p in entry["release_problems"])


def test_merge_gate_refuses_report_without_receipt(world):
    out, path = _captured_report(world, end=None)
    gate = pc.merge_gate(out, world.site)
    assert gate["used"] == [] and gate["problems"]
    assert gate["ignored"] == [
        {"file": path.name, "status": "complete", "teardown_problems": ["no teardown or release receipt"]}
    ]
    # beside a clean attempt of the same shard, the receipt-less one is not counted twice ("cover 0..0")
    clean = json.loads(path.read_bytes()) | {"release": _release()}
    (out / "report-0-of-1-a1.json").write_text(json.dumps(clean))
    gate = pc.merge_gate(out, world.site)
    assert gate["problems"] == [] and [u["file"] for u in gate["used"]] == ["report-0-of-1-a1.json"]


def _fake_proc(root: Path, pid: int, *, comm: str, argv: list[str], ticks: str, exe: Path) -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "comm").write_text(comm + "\n")
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (d / "stat").write_text(f"{pid} ({comm}) " + " ".join(["S", "1", *["0"] * 17, ticks, *["0"] * 5]))
    (d / "exe").symlink_to(exe)


def test_slot_gate_counts_only_leased_pids(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    fcg, python = tmp_path / "FCG Go", tmp_path / "python3"
    fcg.write_bytes(b"engine")
    python.write_bytes(b"python")
    eid = {**ENGINE_ID, "exe": str(fcg.resolve())}
    pid_ = {**PARSER_ID, "exe": str(python.resolve())}
    _fake_proc(proc, eid["pid"], comm="FCG Go", argv=[str(fcg), "--port", "9701", "--address", "127.0.0.1"],
               ticks=eid["start_ticks"], exe=fcg)  # fmt: skip
    _fake_proc(proc, pid_["pid"], comm="python3", ticks=pid_["start_ticks"], exe=python,
               argv=[str(python), "-P", "-m", "memory_kernel.parser_service", "--port", "9801"])  # fmt: skip
    # the slot itself: its real argv, a python that is neither pyfcg nor the parser service
    _fake_proc(proc, 7100003, comm="python3", ticks="73", exe=python,
               argv=[str(python), "-m", "memory_kernel.cli", "slot", "serve", "--socket", "/tmp/mk-slot/slot.sock",
                     "--data-dir", "/data/mk"])  # fmt: skip
    real_path = engine.Path

    def redirected(*parts):
        p = real_path(*parts)
        return proc / p.relative_to("/proc") if p.is_relative_to("/proc") else p

    monkeypatch.setattr(engine, "Path", redirected)
    monkeypatch.setattr(pc, "port_listening", lambda port: True)
    grant = _grant(eid, pid_)
    assert [p["pid"] for p in engine.own_fcg_processes(port=9701)] == [eid["pid"]]
    assert [p["pid"] for p in engine.own_parser_processes()] == [pid_["pid"]]
    assert pc._slot_gate(9701, grant) is None
    _fake_proc(proc, 7100004, comm="FCG Go", argv=[str(fcg), "--port", "9701"], ticks="74", exe=fcg)
    gate = pc._slot_gate(9701, grant)
    assert gate is not None and gate.startswith("stopped: slot gate") and "7100004" in gate


def test_attached_launch_refuses_other_identity(tmp_path, monkeypatch):
    binary = tmp_path / "FCG Go"
    binary.write_bytes(b"engine")
    cfg = engine.EngineConfig(
        port=9701, binary=binary, image=binary, parser_url=PINS["live_parser"], binary_sha256=None, image_sha256=None
    )
    eid = {**ENGINE_ID, "exe": str(binary.resolve())}
    grant = _grant(eid)
    live = {"pid": eid["pid"], "ppid": 1, "uid": os.getuid(), "exe": eid["exe"], "argv": [],
            "start_ticks": eid["start_ticks"]}  # fmt: skip
    monkeypatch.setattr(engine, "port_listening", lambda port: True)
    seen = {"identity": live}
    monkeypatch.setattr(engine, "proc_identity", lambda pid: seen["identity"])
    attached = engine.AttachedEngine(cfg, tmp_path / "a", grant)
    assert attached.launch()["pid"] == eid["pid"]  # positive control: the granted process
    assert attached.process.pid == eid["pid"] and attached.process.poll() is None
    seen["identity"] = None  # the engine is gone: poll() says so, as _engine_dead reads it
    assert attached.process.poll() is not None

    seen["identity"] = {**live, "start_ticks": "99"}  # same pid, another process
    with pytest.raises(engine.Stop, match="identity"):
        engine.AttachedEngine(cfg, tmp_path / "b", grant).launch()


# -- parse.capture takes the lease; serve clears its orphans ----------------------------------------------------------


def _slot_capture(tmp_path, monkeypatch, shards, calls):
    from contract import test_capture_shards as tcs

    parse, cs = tcs._patch_capture(monkeypatch, tmp_path, calls)
    for name in ("bin", "img"):  # the lease request hashes the engine files, as the capture manifest does
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(pc, "syntax", lambda *a, **k: pytest.fail("the syntax parts are all there"))
    monkeypatch.setattr(pc, "prepare", lambda *a, **k: pytest.fail("the capture is prepared"))
    ctx, paths = tcs._capture_ctx(tmp_path, shards)
    ctx.cfg.capture_slot = Path("/tmp/mk-slot-absent/slot.sock")
    ctx.cfg.data_dir = tmp_path  # the lease request sends the parser's data dir
    return parse, cs, ctx, paths


def test_complete_capture_takes_no_lease(tmp_path, monkeypatch):
    calls = []
    parse, _, ctx, paths = _slot_capture(tmp_path, monkeypatch, 1, calls)
    (paths["capture"] / "report-0-of-1-a0.json").write_text('{"status": "complete", "shard": [0, 1]}')
    (paths["capture"] / "merge.json").write_text('{"counts": {"units": 5}}')
    monkeypatch.setattr(slot, "lease", lambda *a, **k: pytest.fail("a complete capture takes no lease"))
    result = parse.capture(ctx)
    assert calls == [] and "slot" not in result


def test_recorded_k2_asks_cold(tmp_path, monkeypatch):
    calls, asked = [], []
    parse, cs, ctx, paths = _slot_capture(tmp_path, monkeypatch, 1, calls)
    (paths["capture"] / "report-0-of-2-a0.json").write_text('{"status": "complete", "shard": [0, 2]}')
    (paths["capture"] / "report-1-of-2-a0.json").write_text('{"status": "stopped: x", "shard": [1, 2]}')
    monkeypatch.setattr(
        cs, "run_parallel", lambda out, site, k, log: calls.append(("parallel", k)) or [{"status": "complete"}]
    )

    class Held:
        grant = slot.Grant(kind="cold", engine=None, parser=None, boot={}, served={"leases": 0, "jobs": 0})

        def release(self, status, jobs):
            asked.append(("release", status))
            return {"recycle": False, "why": []}

    @contextmanager
    def fake_lease(socket, request, **_):
        asked.append(request)
        yield Held()

    monkeypatch.setattr(slot, "lease", fake_lease)
    result = parse.capture(ctx)
    request = asked[0]
    assert request.want == "cold" and request.live_parser == PINS["live_parser"]
    assert request.binary_path == str((tmp_path / "bin").resolve()) and request.image_path.endswith("img")
    assert request.code_pins == pc.manifest_body([], "x", live_parser="y", site=parse.site_for(ctx))["code_pins"]
    assert calls == [("parallel", 2)] and asked[1] == ("release", "complete")
    assert result["capture_shards"] == 2 and result["slot"]["kind"] == "cold"


def test_unavailable_runs_cold(tmp_path, monkeypatch):
    calls = []
    parse, _, ctx, _ = _slot_capture(tmp_path, monkeypatch, 1, calls)

    def fake_run(out, shard, *, site, grant=None):
        calls.append(("run", shard, grant, site.parser is not None))
        return {"status": "complete"}

    monkeypatch.setattr(pc, "run", fake_run)
    result = parse.capture(ctx)  # nothing listens on the socket: the real lease raises SlotUnavailable
    assert calls == [("run", (0, 1), None, True)]  # today's path: no grant, the managed parser started in the attempt
    assert "slot-absent" in result["slot"]["unavailable"]


def test_clear_orphans_kills_only_matching(tmp_path, monkeypatch):
    import signal

    proc = tmp_path / "proc"
    fcg = tmp_path / "FCG Go"
    fcg.write_bytes(b"engine")
    _fake_proc(proc, 11, comm="FCG Go", argv=[str(fcg), "--port", "9701"], ticks="5", exe=fcg)
    _fake_proc(proc, 12, comm="FCG Go", argv=[str(fcg), "--port", "9702"], ticks="6", exe=fcg)
    real_path = engine.Path

    def redirected(*parts):
        p = real_path(*parts)
        return proc / p.relative_to("/proc") if p.is_relative_to("/proc") else p

    sent = []

    def fake_kill(pid, sig):  # never a real signal: the fake process is gone once signalled
        sent.append((pid, sig))
        shutil.rmtree(proc / str(pid))

    monkeypatch.setattr(engine, "Path", redirected)
    monkeypatch.setattr(slot.os, "kill", fake_kill)
    state = {"engine": {"pid": 11, "start_ticks": "5", "exe": str(fcg.resolve())}}
    with pytest.raises(engine.Stop, match="12"):
        slot.clear_orphans(state)
    assert sent == [(11, signal.SIGTERM)]  # 12 is an own FCG process the state does not prove: never signalled
    sent.clear()
    with pytest.raises(engine.Stop, match="12"):  # same pid, other start ticks: not the process the state names
        slot.clear_orphans({"engine": {"pid": 12, "start_ticks": "99", "exe": str(fcg.resolve())}})
    assert sent == []
    # a zombie FCG Go (an earlier slot's killed engine its parent never reaped: comm kept, no argv, no exe) is no stray
    shutil.rmtree(proc / "12")
    (zombie := proc / "13").mkdir()
    (zombie / "comm").write_text("FCG Go\n")
    (zombie / "cmdline").write_bytes(b"")
    (zombie / "stat").write_text("13 (FCG Go) " + " ".join(["Z", "1", *["0"] * 23]))
    assert slot.clear_orphans({}) == [] and sent == []
    # the negative control: a live process with no readable identity (its binary replaced) still refuses the slot
    (zombie / "stat").write_text("13 (FCG Go) " + " ".join(["S", "1", *["0"] * 23]))
    with pytest.raises(engine.Stop, match="13"):
        slot.clear_orphans({})


def test_clear_orphans_spares_a_proven_runtime_engine(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    fcg = tmp_path / "FCG Go"
    fcg.write_bytes(b"engine")
    _fake_proc(proc, 21, comm="FCG Go", argv=[str(fcg), "--port", "9711", "--address", "127.0.0.1"], ticks="7", exe=fcg)
    real_path = engine.Path

    def redirected(*parts):
        p = real_path(*parts)
        return proc / p.relative_to("/proc") if p.is_relative_to("/proc") else p

    sent = []
    monkeypatch.setattr(engine, "Path", redirected)
    monkeypatch.setattr(slot.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    data = tmp_path / "data"
    state = data / "runtime-fcg" / "9711" / "engine.json"
    state.parent.mkdir(parents=True)
    with pytest.raises(engine.Stop, match="21"):  # no state file: an own FCG process the slot cannot prove
        slot.clear_orphans({}, data_dir=data)
    state.write_text(json.dumps({"pid": 21, "start_ticks": "7", "exe": str(fcg.resolve()), "port": 9711}))
    assert slot.clear_orphans({}, data_dir=data) == [] and sent == []  # the runtime engine: neither killed nor counted
    with pytest.raises(engine.Stop, match="21"):  # without the data dir nothing proves it
        slot.clear_orphans({})
    state.write_text(json.dumps({"pid": 21, "start_ticks": "99", "exe": str(fcg.resolve()), "port": 9711}))
    with pytest.raises(engine.Stop, match="21"):  # a stale file (same pid, other start ticks) proves nothing
        slot.clear_orphans({}, data_dir=data)
    assert sent == []
