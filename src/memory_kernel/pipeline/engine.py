"""FCG engine lifecycle for the capture stage.

mk runs ONE engine per build on a private port: no lanes, no sibling filter, no process-global rebinds
(it replaces the reference harness's rebinding of the spare engine and the parse capture).

Ported from the reference harness:
  * ``EngineConfig`` constants  <- the harness's spare-engine module (PROTECTED_PORTS; the BINARY/IMAGE pins are
    full sha256 pins, see ``pin_problems``)
  * ``parser_lock``  <- the harness's parser flock (lock path explicit)
This module holds the config, the parser-locality guard and the parser lock; ``Engine`` below is the spare-engine
port.

``ParserService`` runs the in-package syntactic parser (``memory_kernel.parser_service``) on a fixed loopback
port for the capture's syntax and run attempts, with the same launch / teardown discipline as ``Engine``.

``AttachedEngine`` drives an engine the capture slot (``memory_kernel.pipeline.slot``) launched and owns: the attempt
attaches as a client under a lease, re-checks the granted identity in /proc and ends with a release receipt
(``release_problems``); it never launches, arms or stops a process.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import http.client
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from memory_kernel.pipeline.slot import Grant

PROTECTED_PORTS: tuple[int, ...] = (9600, 9655)
PORT_RANGE = range(9700, 9800)
#: Lisp form that resets every id counter the engine numbers units, symbols and frames with, so a job's output does not
#: depend on which jobs ran before it on the same engine (the counters were found by probing the engine).
RESET_IDS_FORM = "(progn (utils::reset-id-counters) (setf *gensym-counter* 0) 0)"


@dataclass(frozen=True)
class EngineConfig:
    port: int
    binary: Path
    image: Path
    parser_url: str
    #: full sha256 the binary / image must have (``config.engine_pins``: ``runtime-pins.json``); None = not enforced
    #: (``--allow-unpinned-engine``: the capture manifest still records the actual shas). Required: no silent default
    binary_sha256: str | None
    image_sha256: str | None
    protected_ports: tuple[int, ...] = PROTECTED_PORTS

    def __post_init__(self) -> None:
        if self.port in self.protected_ports or self.port not in PORT_RANGE:
            raise ValueError(f"engine port {self.port}: must be a private 97xx port, never {self.protected_ports}")


def pin_problems(pins: tuple[str | None, str | None], *, binary_sha256: str, image_sha256: str) -> list[str]:
    """What differs between the engine files' ACTUAL full shas and the pins ``(binary, image)`` (``config.engine_pins``;
    a None pin is not enforced). Empty when everything pinned matches. The one check behind preflight (``runner.engine_problems``),
    capture ``prepare`` and ``prepare_syntax``. ``Engine.launch`` does not call it: it has its own, binary-only sha check."""
    out = []
    for what, pin, actual in (("binary", pins[0], binary_sha256), ("grammar image", pins[1], image_sha256)):
        if pin is not None and actual != pin:
            out.append(f"engine {what} sha256 {actual} != pinned {pin}")
    return out


class RemoteParserRefused(ValueError):
    """The parser URL is not on this machine and remote parsers were not allowed."""


def require_local(url: str, *, allow_remote: bool = False) -> None:
    """Refuse a parser URL whose host is not loopback, unless ``allow_remote``."""
    if allow_remote:
        return
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise RemoteParserRefused(f"parser URL {url!r} is not local; pass --allow-remote-parser to use it")


_tls = threading.local()


@contextmanager
def parser_lock(path: Path) -> Iterator[None]:
    """Exclusive flock on ``path``, re-entrant within a thread. Each first acquisition opens a fresh file
    description, so two threads of one process contend exactly like two processes."""
    held = _tls.__dict__.setdefault("held", {})
    key = str(path)
    if held.get(key):
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        held[key] = 1
        try:
            yield
        finally:
            held[key] = 0
            fcntl.flock(handle, fcntl.LOCK_UN)


# -- the engine (ported from the reference harness's spare engine) --------------------------------------------------------


DEAD_PARSER = "http://127.0.0.1:9"


BODY_CAP = 16 * 1024 * 1024


class Stop(Exception):
    """Gate failure: stop (after teardown)."""


def _select_own(procs, port):
    """The processes a shard on ``port`` must not share the host with: all of them (port None, serial), else only the
    engine bound to that port. Other shards' engines are siblings, not strays."""
    if port is None:
        return procs
    return [p for p in procs if str(port) in p["argv"]]


def own_fcg_processes(exclude=(), port=None):
    """uid-scoped scan of /proc for pyfcg / FCG Go processes (comm-filtered; never pgrep -af); ``port`` scopes it."""
    uid, out = os.getuid(), []
    for path in Path("/proc").glob("[0-9]*"):
        pid = int(path.name)
        if pid in exclude or pid == os.getpid():
            continue
        try:
            if path.stat().st_uid != uid:
                continue
            comm = (path / "comm").read_text().strip()
            argv = [s.decode(errors="replace") for s in (path / "cmdline").read_bytes().split(b"\0") if s]
        except OSError:
            continue
        joined = " ".join(argv).lower()
        if comm == "FCG Go" or any(a.endswith("/FCG Go") for a in argv) or ("python" in comm and "pyfcg" in joined):
            out.append({"pid": pid, "comm": comm, "argv": argv})
    return _select_own(out, port)


def port_listening(port):
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        with socket.socket(family) as s:
            s.settimeout(0.5)
            with contextlib.suppress(OSError):
                if s.connect_ex((host, port)) == 0:
                    return True
    return False


def proc_cputime(pid):
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")


def proc_identity(pid: int) -> dict | None:
    """``/proc`` identity of ``pid``: pid, ppid, uid, exe, argv and start_ticks (a string, as /proc writes it).
    None when the process is gone or cannot be read (a zombie's exe link is unreadable too)."""
    proc = Path(f"/proc/{pid}")
    try:
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "ppid": int(stat[1]),
            "uid": proc.stat().st_uid,
            "exe": str((proc / "exe").resolve(strict=True)),
            "argv": [s.decode() for s in (proc / "cmdline").read_bytes().split(b"\0") if s],
            "start_ticks": stat[19],
        }
    except (OSError, ValueError, IndexError):
        return None


def proc_state(pid: int) -> str | None:
    """The one-letter state of ``pid`` from ``/proc/<pid>/stat`` (``Z`` a zombie, ``X`` dead), None when it is gone."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


def same_process(identity, granted) -> bool:
    """True when ``identity`` (``proc_identity``) is the ``granted`` process: same pid, start_ticks and exe. The ticks
    compare as strings (/proc writes a string, a grant may carry an int)."""
    if not isinstance(identity, dict) or not isinstance(granted, dict):
        return False
    if any(granted.get(k) is None for k in ("pid", "start_ticks", "exe")):
        return False
    return (
        identity.get("pid") == granted["pid"]
        and str(identity.get("start_ticks")) == str(granted["start_ticks"])
        and identity.get("exe") == granted["exe"]
    )


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readback_request():
    from memory_kernel.pipeline.stages._resident_stock_capture_runner import stock_readback_request

    return stock_readback_request()


def validate_readback(value, *, parser):
    """``stock_validate_readback`` with the expected parser named by the caller."""
    import re

    from memory_kernel.pipeline.stages._resident_stock_capture_runner import CONFIG

    problems = []
    if not isinstance(value, dict) or value.get("status") != "ok" or not isinstance(value.get("payload"), dict):
        return [f"readback failed: {str(value)[:500]}"], None
    row = value["payload"]
    if row.get("owner") != "FCG-GO" or row.get("class") != "HASHED-FCG-CONSTRUCTION-SET":
        problems.append(f"owner/class {row.get('owner')}/{row.get('class')}")
    if row.get("config") != CONFIG:
        problems.append(f"config differs: {row.get('config')}")
    if row.get("parser") != parser:
        problems.append(f"parser {row.get('parser')}")
    if row.get("searches") != 0:
        problems.append(f"searches {row.get('searches')}")
    counts = dict(re.findall(r"(bucket|current|retired|neither)=(\d+)", str(row.get("supplier", ""))))
    if (
        int(counts.get("bucket", 0)) <= 0
        or counts.get("current") != "3"
        or counts.get("retired") != "0"
        or counts.get("neither") != "0"
    ):
        problems.append(f"supplier contract {counts}")
    return problems, row


class Engine:
    def __init__(self, cfg: EngineConfig, out):
        self.cfg = cfg  # EngineConfig refuses a protected / non-97xx port
        self.port, self.out, self.process = cfg.port, Path(out), None
        self.parser = None
        self.log = None
        self.receipts = None

    def _receipts(self):
        if self.receipts is None:
            self.out.mkdir(parents=True, exist_ok=True)
            self.receipts = (self.out / "receipts.jsonl").open("a")
        return self.receipts

    def launch(self, on_spawn=None):
        """Start the engine and wait until its port listens. ``on_spawn`` (no arguments) runs once the process is
        proven ours, before the readiness wait: a caller that must be able to clean up after its own death (the warm
        capture slot's state file) records the process there."""
        before = {p["pid"] for p in own_fcg_processes(port=self.port)}
        if before:
            raise Stop(f"stray own FCG processes before launch: {sorted(before)}")
        if port_listening(self.port):
            raise Stop(f"port {self.port} already listening")
        binary_sha = sha_file(self.cfg.binary)
        if self.cfg.binary_sha256 is not None and binary_sha != self.cfg.binary_sha256:
            raise Stop(
                f"engine binary sha256 {binary_sha} != pinned {self.cfg.binary_sha256} "
                "(`mk setup` fetches the pinned engine; --allow-unpinned-engine accepts another and records it)"
            )
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = (self.out / "engine.log").open("ab")
        self.process = subprocess.Popen(
            [str(self.cfg.binary), "--port", str(self.port), "--address", "127.0.0.1"],
            cwd=self.out,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        identity = proc_identity(self.process.pid)
        if identity is None or identity["ppid"] != os.getpid() or identity["uid"] != os.getuid():
            raise Stop(f"engine ownership differs: {identity}")
        if Path(identity["exe"]) != Path(self.cfg.binary).resolve():
            self.process.terminate()
            with contextlib.suppress(Exception):
                self.process.wait(10)
            raise Stop(f"engine exe {identity['exe']} is not the pinned binary {self.cfg.binary}")
        if on_spawn is not None:
            on_spawn()
        deadline = time.monotonic() + 30
        while not port_listening(self.port):
            if self.process.poll() is not None:
                raise Stop("engine exited before readiness")
            if time.monotonic() > deadline:
                raise Stop("engine startup timeout")
            time.sleep(0.1)
        return identity

    def arm(self, parser):
        """Resident arming on the private engine: six shims installed individually, parser host = ``parser``,
        shuffle pinned to NIL."""
        import pyfcg

        from memory_kernel.pipeline.stages._fenced_http import GRAMMAR

        os.environ["MK_FCG_PARSER_HOST_REQUIRE_REACHABLE"] = "0"
        pyfcg.init(address="127.0.0.1", port=self.port, directory=str(Path(self.cfg.binary).parent), launch=False)
        deadline = time.monotonic() + 30
        while True:
            try:
                if pyfcg.routes.lisp_eval("(+ 1 1)") == 2:
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
        from memory_kernel.construction.fcg.comprehend_cancel import install_comprehend_cancel
        from memory_kernel.construction.fcg.grammar_store import (
            install_fcg_comprehend_solution_bound,
            install_fcg_condition_hook,
            install_fcg_neighbors_guard,
        )
        from memory_kernel.construction.fcg.penelope_host import install_penelope_host_shim
        from memory_kernel.construction.fcg.provisional_units import install_provisional_units_shim

        shims = {
            "condition_hook": install_fcg_condition_hook(pyfcg),
            "neighbors_guard": install_fcg_neighbors_guard(pyfcg),
            "solution_bound": install_fcg_comprehend_solution_bound(pyfcg),
            "penelope_host": install_penelope_host_shim(pyfcg, host=parser),
            "provisional_units": install_provisional_units_shim(pyfcg),
            "comprehend_cancel": install_comprehend_cancel(pyfcg),
        }
        self.parser = parser
        agent = pyfcg.PropBankAgent("memory-kernel-propbank")
        if agent.grammar.id != GRAMMAR:
            raise Stop(f"grammar id {agent.grammar.id} != {GRAMMAR}")
        started = time.monotonic()
        agent.load_grammar_image(str(self.cfg.image))
        load_s = time.monotonic() - started
        raw = self.http(readback_request(), "readback-raw", timeout=10)
        agent.grammar.set_grammar_configuration("shuffle-cxns-before-application", False)  # NIL
        return {"shims": shims, "image_load_s": load_s, "raw_config": raw.get("payload", {}).get("config")}

    def set_parser(self, url):
        """Repoint the engine's parser host (DEAD_PARSER for a fenced run, LIVE_PARSER for the cold arm)."""
        import pyfcg

        from memory_kernel.construction.fcg.penelope_host import install_penelope_host_shim

        os.environ["MK_FCG_PARSER_HOST_REQUIRE_REACHABLE"] = "0"
        result = install_penelope_host_shim(pyfcg, host=url)
        self.parser = url
        return result

    def reset_ids(self):
        """Reset the engine's id counters before a job. The engine must answer 0; anything else stops the shard."""
        import pyfcg

        answer = pyfcg.routes.lisp_eval(RESET_IDS_FORM)
        if answer != 0:
            raise Stop(f"engine id reset answered {str(answer)[:200]}")

    def readback(self):
        value = self.http(readback_request(), "readback", timeout=10)
        return validate_readback(value, parser=self.parser)

    def http(self, code, label, *, timeout, allowed=None):
        from memory_kernel.pipeline.stages._fenced_http import decode_response, http_body

        started = time.monotonic()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=min(2, timeout))
        response, timer, payload, error = None, None, bytearray(), None
        try:
            connection.connect()
            owned = connection.sock
            timer = threading.Timer(max(0.001, timeout), lambda: owned.shutdown(socket.SHUT_RDWR))
            timer.daemon = True
            timer.start()
            owned.settimeout(timeout)
            connection.request(
                "POST",
                "/fcg-read-eval-print",
                body=http_body(code, utf8=bool(allowed)),
                headers={
                    "Content-Type": "application/json; charset=utf-8" if allowed else "application/json",
                    "Accept-Encoding": "identity",
                },
            )
            response = connection.getresponse()
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("spare-engine HTTP deadline")
                owned.settimeout(remaining)
                chunk = response.read1(65536)
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > BODY_CAP:
                    raise ValueError("spare-engine response byte cap")
            if response.status != 200:
                raise ValueError(f"HTTP status {response.status}: {bytes(payload[:500])!r}")
            return decode_response(bytes(payload), allowed)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            if timer is not None:
                timer.cancel()
            if response is not None:
                response.close()
            connection.close()
            with contextlib.suppress(Exception):  # never mask the original exception
                receipts = self._receipts()
                receipts.write(
                    json.dumps(
                        {
                            "label": label,
                            "elapsed_s": round(time.monotonic() - started, 4),
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "error": error,
                        }
                    )
                    + "\n"
                )
                receipts.flush()

    def idle_cpu(self, seconds=3.0):
        cpu0 = proc_cputime(self.process.pid)
        time.sleep(seconds)
        return proc_cputime(self.process.pid) - cpu0

    def teardown(self):
        report = {}
        if self.process is not None:
            pid = self.process.pid
            if self.process.poll() is None:
                os.kill(pid, signal.SIGTERM)
                try:
                    self.process.wait(10)
                except subprocess.TimeoutExpired:
                    os.kill(pid, signal.SIGKILL)
                    self.process.wait(10)
            report["engine_pid"] = pid
            report["engine_returncode"] = self.process.returncode
        for name in ("log", "receipts"):
            handle = getattr(self, name, None)
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()
        time.sleep(0.5)
        report["port_listening_after"] = port_listening(self.port)
        report["own_fcg_after"] = own_fcg_processes(port=self.port)
        return report

    def close_attempt(self) -> tuple[str, dict]:
        """End an attempt on the engine this process launched: ``("teardown", self.teardown())``."""
        return "teardown", self.teardown()


class _LeasedProcess:
    """The leased engine as ``_engine_dead`` and ``idle_cpu`` see a process: its ``pid``, and ``poll()`` None while that
    pid is still the granted process (``same_process``), -1 from the first time it is not. Never a kill handle."""

    def __init__(self, granted: dict):
        self.pid, self.granted, self.returncode = granted["pid"], granted, None

    def poll(self):
        if self.returncode is None and not same_process(proc_identity(self.pid), self.granted):
            self.returncode = -1
        return self.returncode


class AttachedEngine(Engine):
    """An engine the capture slot owns, leased to this attempt (``slot.Grant``, warm). The attempt attaches as a client:
    ``launch`` re-checks the granted identity in /proc (the slot ran the cold launch at boot), ``arm`` points pyfcg at
    the port and sets the parser (the slot armed at boot; its receipt rides along), and ``close_attempt`` writes a
    release receipt. Nothing here starts or stops a process."""

    def __init__(self, cfg: EngineConfig, out, grant: Grant):
        super().__init__(cfg, out)
        if grant.kind != "warm" or grant.engine is None or grant.parser is None:
            raise ValueError(f"an attached engine needs a warm grant, got {grant.kind!r}")
        self.grant = grant

    def launch(self):
        granted = self.grant.engine
        if granted.get("port") != self.port:
            raise Stop(f"leased engine port {granted.get('port')} is not {self.port}")
        identity = proc_identity(granted["pid"])
        if not same_process(identity, granted) or identity["uid"] != os.getuid():
            raise Stop(f"leased engine identity differs: {identity} (granted {granted})")
        if Path(identity["exe"]) != Path(self.cfg.binary).resolve():
            raise Stop(f"leased engine exe {identity['exe']} is not the pinned binary {self.cfg.binary}")
        if not port_listening(self.port):
            raise Stop(f"leased engine port {self.port} is not listening")
        self.process = _LeasedProcess(granted)
        return identity

    def arm(self, parser):
        """Attach: pyfcg's client globals only (no engine call), then the parser host set explicitly."""
        import pyfcg

        pyfcg.init(address="127.0.0.1", port=self.port, launch=False)
        self.set_parser(parser)
        return {"attached": True, "slot": self.grant.boot}

    def teardown(self):
        raise RuntimeError("a leased engine is never torn down by the attempt: close_attempt releases it")

    def close_attempt(self) -> tuple[str, dict]:
        """``("release", receipt)``: the leased engine and parser as /proc shows them now, any other own FCG process on
        the port or own parser process, and the granted identities (``release_problems`` checks it)."""
        engine, parser = self.grant.engine, self.grant.parser
        if self.receipts is not None:
            with contextlib.suppress(Exception):
                self.receipts.close()
        return "release", {
            "engine": proc_identity(engine["pid"]),
            "parser": proc_identity(parser["pid"]),
            "own_fcg_other": [p for p in own_fcg_processes(port=self.port) if p["pid"] != engine["pid"]],
            "own_parser_other": [p for p in own_parser_processes() if p["pid"] != parser["pid"]],
            "granted": {"engine": engine, "parser": parser},
        }


def release_problems(receipt) -> list[str]:
    """Empty only when the release receipt proves the leased engine and parser are alive as the granted processes
    (pid, start_ticks, exe) and no other own FCG process (on the port) or own parser process exists."""
    if not isinstance(receipt, dict) or "error" in receipt:
        return [f"release failed: {receipt}"]
    granted = receipt.get("granted") if isinstance(receipt.get("granted"), dict) else {}
    problems = []
    for what in ("engine", "parser"):
        if not same_process(receipt.get(what), granted.get(what)):
            problems.append(
                f"leased {what} is not the granted process: {receipt.get(what)} (granted {granted.get(what)})"
            )
    if receipt.get("own_fcg_other") != []:
        problems.append(f"other own FCG processes at release: {receipt.get('own_fcg_other')}")
    if receipt.get("own_parser_other") != []:
        problems.append(f"other own parser processes at release: {receipt.get('own_parser_other')}")
    return problems


# -- the managed syntactic-parser service ---------------------------------------------------------

#: The managed parser's port range and default. The port is part of the URL every capture seals (the capture
#: manifest's phase-B parser and each syntax capture's ``parser``), so changing it voids a resume: it is fixed,
#: configurable (``--parser-port``) and refused when busy, never picked at random.
PARSER_PORT_RANGE = range(9800, 9900)
DEFAULT_PARSER_PORT = 9801
PARSER_MODULE = "memory_kernel.parser_service"
#: What the service imports beyond the base install (the ``[parser]`` extra) and the spaCy pipeline it loads.
PARSER_IMPORTS: tuple[str, ...] = ("fastapi", "uvicorn", "benepar", "torch_struct", "en_core_web_sm")
#: The setup component that holds the benepar model, under ``<data>/nltk_data`` (the service's ``NLTK_DATA``).
PARSER_MODEL_COMPONENT = "parser-model"
PARSER_NLTK_DATA = "nltk_data"
#: Where that model must sit, relative to the data dir: nltk's ``models/benepar_en3`` lookup under ``NLTK_DATA``.
PARSER_MODEL_REL = "nltk_data/models/benepar_en3"
#: One parse after ``/health``: benepar attaches on the first parse (torch and the model load then), and the engine's
#: parser probe allows 8 s, so the cold load is paid here.
PARSER_WARMUP_TEXT = "The parser is alive."
PARSER_STARTUP_S = 600.0


def managed_parser_url(port: int) -> str:
    """The URL of the managed parser on ``port``. It contains ``127.0.0.1``, so the engine calls the bare
    ``/syntactic-parser`` path; the service serves both paths."""
    return f"http://127.0.0.1:{port}"


def check_parser_port(port: int) -> None:
    if port not in PARSER_PORT_RANGE or port in PROTECTED_PORTS:
        raise ValueError(
            f"parser port {port}: must be in {PARSER_PORT_RANGE.start}-{PARSER_PORT_RANGE.stop - 1} (default "
            f"{DEFAULT_PARSER_PORT})"
        )


@dataclass(frozen=True)
class ParserConfig:
    """The managed parser of one build: its port and the ``mk setup`` data dir that holds its model."""

    port: int
    #: None = no data dir: the launch refuses (the model is looked up there)
    data_dir: Path | None
    python: str = sys.executable
    startup_s: float = PARSER_STARTUP_S

    def __post_init__(self) -> None:
        check_parser_port(self.port)

    @property
    def url(self) -> str:
        return managed_parser_url(self.port)


def _parser_model_problems(data_dir: Path) -> list[str]:
    from memory_kernel import setup_runtime

    try:
        model = setup_runtime.managed_path(data_dir, PARSER_MODEL_COMPONENT)
    except setup_runtime.SetupError as exc:
        model, detail = None, str(exc)
    else:
        detail = f"looked under {data_dir}"
    if model is None or not (model / "benepar_model.bin").is_file():
        return [
            f"no parser model: run `mk setup --component {PARSER_MODEL_COMPONENT}` ({detail}); or pass --parser-url"
        ]
    if model != data_dir / PARSER_MODEL_REL:
        return [f"parser model {model} is not {data_dir / PARSER_MODEL_REL} (where NLTK_DATA finds it)"]
    return []


def parser_service_problems(cfg: ParserConfig, *, check_port: bool = True) -> list[str]:
    """What would stop the managed parser from starting, found without starting it: the ``[parser]`` extra and the spaCy
    model in the interpreter that runs it, the benepar model ``mk setup`` placed, and (``check_port``) a free port."""
    import importlib.util

    problems = []
    missing = [name for name in PARSER_IMPORTS if importlib.util.find_spec(name) is None]
    if missing:
        problems.append(
            f"the managed parser needs {', '.join(missing)} in {cfg.python}: pip install 'memory-kernel[parser]'"
            + (" and run `mk setup --component spacy-model`" if "en_core_web_sm" in missing else "")
            + "; or pass --parser-url"
        )
    if cfg.data_dir is None:
        problems.append("the managed parser needs the `mk setup` data dir (--data-dir); or pass --parser-url")
    else:
        problems += _parser_model_problems(cfg.data_dir)
    if check_port and port_listening(cfg.port):
        problems.append(
            f"parser port {cfg.port} is already listening: stop what holds it, or choose another with --parser-port "
            "(the port is sealed into every capture: a new port means a new capture)"
        )
    return problems


def own_parser_processes(exclude=()):
    """uid-scoped scan of /proc for managed parser services (``python ... -m memory_kernel.parser_service``)."""
    uid, out = os.getuid(), []
    for path in Path("/proc").glob("[0-9]*"):
        pid = int(path.name)
        if pid in exclude or pid == os.getpid():
            continue
        try:
            if path.stat().st_uid != uid:
                continue
            argv = [s.decode(errors="replace") for s in (path / "cmdline").read_bytes().split(b"\0") if s]
        except OSError:
            continue
        if any(a == "-m" and b == PARSER_MODULE for a, b in zip(argv, argv[1:], strict=False)):
            out.append({"pid": pid, "argv": argv})
    return out


def rss_kib(pid: int) -> int | None:
    """Resident set size of ``pid`` in KiB (``/proc/<pid>/status`` VmRSS), None when it cannot be read."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def parser_http(port: int, method: str, path: str, body: bytes | None = None, *, timeout: float):
    """``(status, content type, body)`` of one request to the local parser (no proxy: ``http.client`` reads none)."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.getheader("content-type") or "", response.read()
    finally:
        connection.close()


class ParserService:
    """The managed parser: ``python -P -m memory_kernel.parser_service`` on ``127.0.0.1:<port>``, a private child of
    this process, with the discipline of ``Engine.launch`` / ``teardown``: no stray own parser and a free port before
    launch, a new session, ``/proc`` ownership, readiness by ``/health`` plus one warm-up parse, SIGTERM then SIGKILL,
    and a log at ``<out>/parser.log``. The child sees ``NLTK_DATA=<data>/nltk_data`` (the benepar model), the Hugging
    Face libraries offline, and no ``MK_PARSER_*`` setting (the copied config reads those: a stray one would change the
    pipeline)."""

    def __init__(self, cfg: ParserConfig, out):
        self.cfg, self.out = cfg, Path(out)
        self.process = None
        self.log = None

    @property
    def url(self) -> str:
        return self.cfg.url

    def child_env(self) -> dict[str, str]:
        from memory_kernel.pipeline.provenance import package_dir

        env = {k: v for k, v in os.environ.items() if not k.startswith("MK_PARSER_")}
        pkg = package_dir()
        if pkg.parent.name == "src":  # a source tree: the child imports this tree, never another install
            env["PYTHONPATH"] = str(pkg.parent)
        else:
            env.pop("PYTHONPATH", None)
        if self.cfg.data_dir is not None:
            env["NLTK_DATA"] = str(self.cfg.data_dir / PARSER_NLTK_DATA)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    def argv(self) -> list[str]:
        return [self.cfg.python, "-P", "-m", PARSER_MODULE, "--port", str(self.cfg.port), "--address", "127.0.0.1"]

    def launch(self, on_spawn=None) -> dict:
        """Start the parser and wait for its health check and warm-up parse. ``on_spawn`` as ``Engine.launch``."""
        strays = own_parser_processes()
        if strays:
            raise Stop(f"stray own parser processes before launch: {sorted(p['pid'] for p in strays)}")
        problems = parser_service_problems(self.cfg)
        if problems:
            raise Stop("; ".join(problems))
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = (self.out / "parser.log").open("ab")
        started = time.monotonic()
        self.process = subprocess.Popen(
            self.argv(),
            cwd=self.out,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=self.child_env(),
        )
        proc = Path(f"/proc/{self.process.pid}")
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        identity = {
            "pid": self.process.pid,
            "ppid": int(stat[1]),
            "uid": proc.stat().st_uid,
            "exe": str((proc / "exe").resolve()),
            "argv": [s.decode() for s in (proc / "cmdline").read_bytes().split(b"\0") if s],
            "start_ticks": stat[19],
        }
        if identity["ppid"] != os.getpid() or identity["uid"] != os.getuid():
            raise Stop(f"parser ownership differs: {identity}")
        if Path(identity["exe"]) != Path(self.cfg.python).resolve():
            raise Stop(f"parser exe {identity['exe']} is not the interpreter {self.cfg.python}")
        if on_spawn is not None:
            on_spawn()
        deadline = started + self.cfg.startup_s

        def remaining() -> float:
            left = deadline - time.monotonic()
            if left <= 0:
                raise Stop(f"parser startup timeout ({self.cfg.startup_s:.0f} s; see {self.out / 'parser.log'})")
            return left

        while True:
            if self.process.poll() is not None:
                raise Stop(
                    f"parser exited before readiness, rc {self.process.returncode} (see {self.out / 'parser.log'})"
                )
            try:
                if parser_http(self.cfg.port, "GET", "/health", timeout=min(2.0, remaining()))[0] == 200:
                    break
            except OSError:
                pass
            remaining()
            time.sleep(0.25)
        health_s = time.monotonic() - started
        body = json.dumps({"sentence": PARSER_WARMUP_TEXT, "model": "en_benepar"}).encode()
        try:
            status, ctype, answer = parser_http(self.cfg.port, "POST", "/syntactic-parser", body, timeout=remaining())
        except OSError as exc:
            raise Stop(f"parser warm-up parse failed: {type(exc).__name__}: {exc}") from None
        if status != 200 or not ctype.startswith("text/html") or b'"trees"' not in answer:
            raise Stop(f"parser warm-up parse: status {status}, content type {ctype!r}, body {answer[:200]!r}")
        if self.process.poll() is not None:
            raise Stop(f"parser exited after readiness, rc {self.process.returncode}")
        return {
            **identity,
            "url": self.url,
            "health_s": round(health_s, 3),
            "ready_s": round(time.monotonic() - started, 3),
            "rss_kib": rss_kib(self.process.pid),
        }

    def teardown(self) -> dict:
        report: dict = {}
        if self.process is not None:
            pid = self.process.pid
            if self.process.poll() is None:
                report["rss_kib_at_stop"] = rss_kib(pid)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGTERM)  # its own session: the group is the service and nothing else
                try:
                    self.process.wait(15)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pid, signal.SIGKILL)
                    self.process.wait(10)
            report["parser_pid"] = pid
            report["parser_returncode"] = self.process.returncode
        if self.log is not None:
            with contextlib.suppress(Exception):
                self.log.close()
        deadline = time.monotonic() + 5
        while port_listening(self.cfg.port) and time.monotonic() < deadline:
            time.sleep(0.2)
        report["port_listening_after"] = port_listening(self.cfg.port)
        report["own_parser_after"] = own_parser_processes()
        return report


def parser_teardown_problems(teardown) -> list[str]:
    """Empty only when the receipt proves the parser port is free and no own parser process is left."""
    if not isinstance(teardown, dict) or "error" in teardown:
        return [f"parser teardown failed: {teardown}"]
    problems = []
    if teardown.get("port_listening_after") is not False:
        problems.append(f"parser port listening after teardown: {teardown.get('port_listening_after')}")
    if teardown.get("own_parser_after") != []:
        problems.append(f"own parser processes after teardown: {teardown.get('own_parser_after')}")
    return problems
