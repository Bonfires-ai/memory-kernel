"""The warm capture slot: one long-lived process keeps an FCG engine and the managed parser up, and the capture child
leases them instead of booting its own, which saves both cold boots (parser start, grammar image load) per capture.

Why a lease and not a shared engine: a capture seals the engine's pins and the parser URL, resets the engine per job
and reads it back after every switch. Those checks only mean something if nothing else drives the engine meanwhile,
so the slot grants ONE lease at a time: it is the capture mutex among the captures that name its socket. A second
capture waits for the lease (``BUSY_WAIT_S``, then ``SlotBusy``, a retriable stop); it never falls back to cold,
because the slot holds the parser port the capture must seal.

The lease is a held connection on a unix socket, JSON lines, one op per line:
  * ``lease`` (client) -> ``grant`` (slot), or ``refused`` when the slot has exited; the request carries the pins
    (``PIN_KEYS``: the engine binary and image sha256 and resolved paths, the engine port, the live parser URL, the
    parser's data dir, the arm environment ``ARM_ENV``, the code pins and the walker sha), and warm is granted only
    when they equal the slot's boot;
  * ``release`` (client: status + job count) -> ``released`` (slot: whether it recycles and why);
  * ``status`` and ``stop`` on a fresh connection, answered at once, even while a lease is held.
Closing the connection ends the lease, so a capture child that dies releases it too. A lease that ends without a
``release``, or with a status other than ``complete``, is never reused: the slot recycles (teardown + boot) first.

The engine and the parser sit behind ``SlotBackend``. This module holds the protocol, the client side (``lease``), the
server (``SlotServer``), the real backend (``FcgBackend``: the engine and the parser launched with a cold attempt's
checks, the engine armed on the live parser) and ``main_serve`` (``mk slot serve``). A slot that dies alone leaves
its engine and parser running (each starts in its own session); ``slot-state.json`` beside the socket names them
(pid, start_ticks, exe), and the next slot kills only processes matching it and refuses to start beside any other own
FCG or parser process (``clear_orphans``).
"""

from __future__ import annotations

import errno
import json
import os
import select
import signal
import socket as _sock
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from memory_kernel.pipeline.engine import (
    Stop,
    own_fcg_processes,
    own_parser_processes,
    proc_identity,
    proc_state,
    same_process,
)

SLOT_SCHEMA = "mk-capture-slot/v1"
#: How long a capture waits for the lease before its attempt stops ``slot busy`` (15 min).
BUSY_WAIT_S = 900.0
#: Recycle when the engine's RSS after a release is above this (6 GiB; the soak's peak was 3.1 GB): a safety net.
RSS_CAP_KIB = 6 * 1024 * 1024
#: How long a client waits for the ``released`` reply: the slot rests the parser and reads it back before answering.
RELEASE_WAIT_S = 120.0
#: How long a ``status`` / ``stop`` caller, and the slot reading a connection's first op, wait.
OP_WAIT_S = 10.0
#: Deadlines for the backend calls made under the lease lock with no timeout of their own: resting the parser after a
#: release, and a boot (launch, arm, grammar image load). On overrun the engine and parser are torn down (which closes
#: the stuck call's connection): a late rest recycles, a late boot fails the slot.
REST_DEADLINE_S = 30.0
BOOT_DEADLINE_S = 180.0
#: How long a call that overran gets to return once its engine and parser are torn down; past it the slot fails.
_OVERRUN_GRACE_S = 10.0
#: How often a waiting lease checks that its client is still there and the slot still serving.
_POLL_S = 0.2
#: How long the accept loop backs off after running out of file descriptors (EMFILE / ENFILE) before it retries.
_ACCEPT_RETRY_S = 1.0
_MAX_LINE = 1024 * 1024
#: The state file beside the socket: the engine and parser identities of the current boot (``clear_orphans``).
STATE_FILE = "slot-state.json"
#: How long an orphan has after SIGTERM (then SIGKILL), and after SIGKILL, to be gone.
ORPHAN_GRACE_S = 10.0
#: The request fields that must equal the slot's ``pins()``: a difference means the slot booted from another install.
#: The paths too: a leased attempt refuses an engine whose exe is not the build's binary, so a slot booted from another
#: (even byte-identical) copy must vacate rather than grant warm.
PIN_KEYS: tuple[str, ...] = (
    "binary_sha256",
    "image_sha256",
    "binary_path",
    "image_path",
    "engine_port",
    "live_parser",
    "parser_data_dir",
    "arm_env",
    "code_pins",
    "walker_sha256",
)
#: The environment the engine's arming shims read (``grammar_store``, ``provisional_units``, ``comprehend_cancel``): a
#: warm engine was armed in the slot's environment, a cold one in the capture's, so the two must agree.
ARM_ENV: tuple[str, ...] = (
    "MK_FCG_PROVISIONAL_SPAN_WIDTH",
    "MK_FCG_PROVISIONAL_TOKEN_CEILING",
    "MK_FCG_PROVISIONAL_SPAN_MODE",
    "MK_FCG_NO_COMPREHEND_CANCEL",
    "MK_FCG_NO_NEIGHBORS_GUARD",
    "MK_FCG_DIALOG_DEBUGGER",
)


def arm_env() -> dict[str, str | None]:
    """``ARM_ENV`` as this process reads it (None = unset)."""
    return {name: os.environ.get(name) for name in ARM_ENV}


@dataclass(frozen=True)
class LeaseRequest:
    """What the capture will seal: the slot grants warm only when its boot pins equal these."""

    binary_sha256: str | None
    image_sha256: str | None
    binary_path: str  # the engine binary, strictly resolved
    image_path: str  # the grammar image, strictly resolved
    engine_port: int
    live_parser: str
    parser_data_dir: str  # the managed parser's data dir (its model), strictly resolved
    arm_env: dict[str, str | None]  # ARM_ENV as read by the requesting process
    code_pins: dict[str, str]  # PIN_FILES -> sha256, as the capture manifest seals them
    walker_sha256: str
    want: Literal["warm", "cold"] = "warm"


@dataclass(frozen=True)
class Grant:
    kind: Literal["warm", "cold"]
    engine: dict | None  # {"pid", "start_ticks", "exe", "port"}; None for cold
    parser: dict | None  # {"pid", "start_ticks", "exe", "url"}; None for cold
    boot: dict  # the slot's launch + arm receipts and its pins ({} for cold)
    served: dict  # {"leases": int, "jobs": int} since boot


class SlotUnavailable(Exception):
    """No socket, a refused connection, or a slot that has exited: the caller runs cold, without a lease."""


class SlotBusy(Stop):
    """No grant within the wait: another capture holds the lease (a retriable stop, never a cold fallback)."""


class SlotBackend(Protocol):
    def boot(self) -> dict: ...  # launch + arm, return the boot receipt
    def identities(self) -> tuple[dict, dict]: ...  # (engine, parser) as granted
    def health(self) -> list[str]: ...  # problems: identity, readback (parser LIVE, searches 0), idle CPU
    def rest(self) -> list[str]: ...  # set the parser LIVE + readback
    def rss_kib(self) -> int | None: ...
    def pins(self) -> dict: ...  # every PIN_KEYS field, as of the current boot
    def teardown(self) -> dict: ...  # clean stop of engine + parser; receipt


def _log(event: str, **fields) -> None:
    # One JSON line per event on stderr: the slot's log (the worker owns where it goes).
    print(json.dumps({"slot": event, **fields}, default=str, sort_keys=True), file=sys.stderr, flush=True)


def _send(conn: _sock.socket, message: dict) -> None:
    conn.sendall(json.dumps(message, sort_keys=True).encode() + b"\n")


class _Lines:
    """Reads one JSON object per line; ``None`` when the peer closed. A socket timeout propagates (``TimeoutError``)."""

    def __init__(self, conn: _sock.socket) -> None:
        self.conn, self.buf = conn, b""

    def read(self) -> dict | None:
        while b"\n" not in self.buf:
            chunk = self.conn.recv(65536)
            if not chunk:
                return None
            self.buf += chunk
            if len(self.buf) > _MAX_LINE:
                raise ValueError(f"slot message over {_MAX_LINE} bytes")
        line, self.buf = self.buf.split(b"\n", 1)
        message = json.loads(line)
        if not isinstance(message, dict):
            raise ValueError(f"slot message is not an object: {line[:80]!r}")
        return message


def _connect(path: Path) -> _sock.socket:
    conn = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
    try:
        conn.connect(str(path))
    except OSError as exc:  # missing socket, refused, path too long: nothing is listening there
        conn.close()
        raise SlotUnavailable(f"{path}: {exc.strerror or exc!r}") from exc
    return conn


def _peer_gone(conn: _sock.socket) -> bool:
    """True when the client hung up (readable with EOF). A waiting lease checks this so a client that gave up is not
    granted (and its dead connection not taken for a dropped lease)."""
    try:
        readable, _, _ = select.select([conn], [], [], 0)
        return bool(readable) and conn.recv(1, _sock.MSG_PEEK) == b""
    except OSError:
        return True


# -- the client --------------------------------------------------------------------------------------------------------


class Lease:
    """A held lease: the connection stays open until ``release`` (or the ``lease`` block ends, which drops it)."""

    def __init__(self, conn: _sock.socket, lines: _Lines, grant: Grant) -> None:
        self.grant = grant
        self._conn, self._lines, self._released = conn, lines, False

    def release(self, status: str, jobs: int) -> dict:
        """Tell the slot how the attempt ended; returns its ``{"recycle": bool, "why": [...]}``."""
        if self._released:
            raise RuntimeError("lease already released")
        self._released = True
        try:
            _send(self._conn, {"op": "release", "status": status, "jobs": jobs})
            self._conn.settimeout(RELEASE_WAIT_S)
            reply = self._lines.read()
        except (OSError, ValueError) as exc:
            raise Stop(f"slot release: {exc!r}") from exc
        if reply is None or reply.get("op") != "released":
            raise Stop(f"slot release: no released reply ({reply!r})")
        return {"recycle": reply["recycle"], "why": reply["why"]}


@contextmanager
def lease(socket: Path, request: LeaseRequest, *, wait_s: float = BUSY_WAIT_S) -> Iterator[Lease]:
    """Hold the slot's lease for the block. ``SlotUnavailable``: no slot (run cold, no lease). ``SlotBusy``: no grant
    within ``wait_s``. Leaving the block without ``Lease.release`` drops the lease, and the slot recycles."""
    with _connect(socket) as conn:
        lines = _Lines(conn)
        try:
            _send(conn, {"op": "lease", "request": asdict(request)})
            conn.settimeout(wait_s)
            reply = lines.read()
        except TimeoutError as exc:
            raise SlotBusy(f"slot busy: no grant within {wait_s:g} s") from exc
        except (OSError, ValueError) as exc:
            raise SlotUnavailable(f"slot lease: {exc!r}") from exc
        if reply is None:
            raise SlotUnavailable("slot closed the connection before a grant")
        if reply.get("op") == "refused":
            raise SlotUnavailable(f"slot refused: {'; '.join(reply.get('why') or [])}")
        if reply.get("op") != "grant":
            raise SlotUnavailable(f"slot lease: unexpected reply {reply.get('op')!r}")
        conn.settimeout(None)  # the lease lasts as long as the capture's attempt
        yield Lease(conn, lines, Grant(**reply["grant"]))


def control(socket: Path, op: Literal["status", "stop"], *, wait_s: float = OP_WAIT_S) -> dict:
    """One ``status`` / ``stop`` round trip (``mk slot status|stop``). ``SlotUnavailable`` when no slot answers."""
    with _connect(socket) as conn:
        try:
            _send(conn, {"op": op})
            conn.settimeout(wait_s)
            reply = _Lines(conn).read()
        except (OSError, ValueError) as exc:
            raise SlotUnavailable(f"slot {op}: {exc!r}") from exc
    if reply is None:
        raise SlotUnavailable(f"slot {op}: closed without a reply")
    return reply


# -- the server --------------------------------------------------------------------------------------------------------


class SlotServer:
    """One lease at a time over ``socket``; the engine and parser behind ``backend``.

    The socket listens from construction, so a lease that arrives while ``serve_forever`` sets up and boots queues on
    the lease lock instead of being refused (as after a restart of the slot). ``setup``, when given, runs first under
    that lock and returns the backend (``main_serve``: clear the orphans, then build ``FcgBackend``); if it raises, the
    slot fails and the waiting leases are refused. ``status`` and ``stop`` never take that lock."""

    def __init__(
        self,
        socket: Path,
        backend: SlotBackend | None = None,
        *,
        rss_cap_kib: int = RSS_CAP_KIB,
        setup: Callable[[], SlotBackend] | None = None,
    ) -> None:
        if (backend is None) == (setup is None):
            raise ValueError("slot server: give exactly one of a backend and a setup")
        self.socket, self._backend, self._setup, self.rss_cap_kib = socket, backend, setup, rss_cap_kib
        self._lease_lock = threading.Lock()  # held by a lease, a boot, a recycle and the final teardown
        self._state_lock = threading.Lock()  # guards the fields below for ``status``; held only briefly
        self._state = "booting"
        self._state_since = time.time()  # epoch seconds when ``_state`` was entered (``status``: the worker's watchdog)
        self._served = {"leases": 0, "jobs": 0}  # this engine's (reset at each boot)
        self._lifetime = {"leases": 0, "jobs": 0}  # this slot's, across boots
        self._boot: dict = {}
        self._engine: dict | None = None
        self._parser: dict | None = None
        self._up = False
        self._failure: str | None = None
        self._stopping = threading.Event()  # a stop is pending: grant nothing more, tear down after the current lease
        self._done = threading.Event()  # torn down: refuse every lease, serve_forever returns
        self._stop_guard = threading.Lock()  # taken once, never blocking, by the first shutdown(): one stopper thread
        self._listener = self._listen(socket)
        self._inode = os.stat(socket).st_ino  # unlink only our own socket at exit, never a successor's

    @property
    def backend(self) -> SlotBackend:
        if self._backend is None:  # only before ``setup`` returned; nothing calls the backend then
            raise Stop("slot backend: not set up")
        return self._backend

    @staticmethod
    def _listen(path: Path) -> _sock.socket:
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(path):
            # A leftover socket from a slot that died is replaced; a live slot, or anything not a socket, is refused.
            if not stat.S_ISSOCK(os.lstat(path).st_mode):
                raise Stop(f"slot socket {path}: exists and is not a socket")
            with suppress(SlotUnavailable), _connect(path):
                raise Stop(f"slot socket {path}: another slot is serving")
            path.unlink()
        listener = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        listener.bind(str(path))
        os.chmod(path, 0o600)
        listener.listen(16)
        listener.settimeout(_POLL_S)  # the accept loop wakes to notice the exit
        return listener

    def _set(self, **fields) -> None:
        with self._state_lock:
            if "state" in fields and fields["state"] != self._state:
                self._state_since = time.time()
            for name, value in fields.items():
                setattr(self, "_" + name, value)

    # -- lifecycle (every method below that touches the backend runs under the lease lock) --

    def serve_forever(self) -> None:
        """Boot, then serve leases until a ``stop`` (or a vacate) ends it; raises ``Stop`` if a boot failed."""
        accept = threading.Thread(target=self._accept_loop, name="slot-accept", daemon=True)
        with self._lease_lock:
            accept.start()  # status answers during the setup and the boot; leases queue on the lock
            if self._setup is not None and not (self._stopping.is_set() or self._done.is_set()):
                try:
                    self._backend = self._setup()
                except Exception as exc:  # noqa: BLE001 - a refused setup (unproven processes, no engine) ends the slot
                    self._fail(f"setup: {exc!r}")
            if not (self._stopping.is_set() or self._done.is_set()):  # stopped before serving: boot nothing
                self._boot_backend()
        self._done.wait()
        accept.join(2 * _POLL_S + 1)
        self._listener.close()
        with suppress(FileNotFoundError):
            if os.stat(self.socket).st_ino == self._inode:
                self.socket.unlink()
        _log("exit", failure=self._failure, served=self._lifetime)
        if self._failure is not None:
            raise Stop(f"slot: {self._failure}")

    def shutdown(self) -> None:
        """Ask the slot to stop: no lease is granted from now on, and it tears down and returns from ``serve_forever``
        once the current lease (or boot) ends. Returns at once and never blocks on a lock (safe from a signal handler,
        which may interrupt a thread holding ``_state_lock``)."""
        if self._done.is_set() or not self._stop_guard.acquire(blocking=False):
            return
        self._stopping.set()
        threading.Thread(target=self._stop_after_lease, name="slot-stop", daemon=True).start()

    def _stop_after_lease(self) -> None:
        with self._lease_lock:
            self._exit("stop")

    def _exit(self, why: str) -> None:
        if self._done.is_set():
            return
        if self._up:
            self._teardown(why)
        self._set(state="vacated")
        _log("vacated", why=why)
        self._done.set()

    def _fail(self, why: str) -> None:
        # A boot that raised, or an engine still unhealthy after a fresh boot: the slot cannot serve warm. Captures run
        # cold from now on (their process gate checks the ports), until the worker starts a new slot.
        _log("failed", why=why)
        self._set(failure=why)
        self._exit(why)

    def _boot_backend(self) -> bool:
        self._set(state="booting")
        self._up = True  # before boot(): a half-booted backend (engine up, parser not) is still torn down on failure
        try:
            receipt = self._bounded(self.backend.boot, BOOT_DEADLINE_S, "boot")
            engine, parser = self.backend.identities()
        except Exception as exc:  # noqa: BLE001 - any boot failure ends the slot, with the reason
            self._fail(f"boot: {exc!r}")
            return False
        self._set(state="idle", boot=receipt, engine=engine, parser=parser, served={"leases": 0, "jobs": 0})
        _log("booted", engine=engine, parser=parser)
        return True

    def _teardown(self, why: str) -> dict:
        try:
            receipt = self.backend.teardown()
        except Exception as exc:  # noqa: BLE001 - recorded; the next boot's stray checks refuse a dirty host
            receipt = {"error": repr(exc)}
        self._up = False
        self._set(engine=None, parser=None, boot={})
        _log("teardown", why=why, receipt=receipt)
        return receipt

    def _bounded[T](self, call: Callable[[], T], deadline_s: float, what: str) -> T:
        """``call()`` in a helper thread, bounded by ``deadline_s``. On overrun: tear the engine and parser down (the
        stuck call's connection closes with the engine), give the call ``_OVERRUN_GRACE_S`` to return, tear down again
        (whatever it started meanwhile), then raise ``Stop``. A call still running after the grace fails the slot: the
        backend is never booted again beside it."""
        box: dict = {}

        def run() -> None:
            try:
                box["value"] = call()
            except BaseException as exc:  # noqa: BLE001 - handed to the caller below
                box["error"] = exc

        helper = threading.Thread(target=run, name=f"slot-{what}", daemon=True)
        helper.start()
        helper.join(deadline_s)
        if helper.is_alive():
            _log("overrun", what=what, deadline_s=deadline_s)
            self._teardown(f"{what} overrun")
            helper.join(_OVERRUN_GRACE_S)
            if helper.is_alive():
                self._fail(f"{what}: no answer within {deadline_s:g} s, still running after the teardown")
            else:
                self._teardown(f"{what} overrun: after the call returned")
            raise Stop(f"{what}: no answer within {deadline_s:g} s")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _recycle(self, why: list[str]) -> bool:
        self._set(state="recycling")
        _log("recycle", why=why)
        self._teardown("recycle")
        if self._stopping.is_set():  # a stop that came in meanwhile: boot nothing, the stopper exits next
            return False
        return self._boot_backend()

    # -- connections --

    def _accept_loop(self) -> None:
        while not self._done.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError as exc:
                if self._done.is_set():  # the listener closed at exit
                    break
                if exc.errno in (errno.ECONNABORTED, errno.EINTR):  # that one client went away: keep accepting
                    _log("accept", error=repr(exc), retry=True)
                    continue
                if exc.errno in (errno.EMFILE, errno.ENFILE):  # out of descriptors: back off, then retry
                    _log("accept", error=repr(exc), retry=True)
                    self._done.wait(_ACCEPT_RETRY_S)
                    continue
                # Anything else: the slot cannot accept leases. Close the listener so clients are refused at once (they
                # run cold) instead of waiting out the busy wait in the backlog, then stop through the lease lock.
                _log("accept failed", error=repr(exc))
                self._set(failure=f"accept: {exc!r}")
                with suppress(OSError):
                    self._listener.close()
                self.shutdown()
                break
            conn.settimeout(None)
            threading.Thread(target=self._handle, args=(conn,), name="slot-conn", daemon=True).start()

    def _handle(self, conn: _sock.socket) -> None:
        with conn:
            lines = _Lines(conn)
            try:
                conn.settimeout(OP_WAIT_S)
                message = lines.read()
                conn.settimeout(None)
                if message is None:
                    return
                op = message.get("op")
                if op == "status":
                    _send(conn, self._status())
                elif op == "stop":
                    _send(conn, {"op": "stopping", "schema": SLOT_SCHEMA})
                    self.shutdown()
                elif op == "lease":
                    self._serve_lease(conn, lines, LeaseRequest(**message["request"]))
                else:
                    _send(conn, {"op": "error", "why": [f"unknown op {op!r}"]})
            except (OSError, ValueError, TypeError, KeyError) as exc:
                _log("connection", error=repr(exc))

    def _status(self) -> dict:
        with self._state_lock:
            state, up = self._state, self._up
            out = {
                "op": "status",
                "schema": SLOT_SCHEMA,
                "pid": os.getpid(),
                "state": state,
                "state_since": self._state_since,
                "stopping": self._stopping.is_set(),
                "served": dict(self._served),
                "lifetime": dict(self._lifetime),
                "engine": self._engine,
                "parser": self._parser,
                "failure": self._failure,
            }
        rss = None
        if up and state in ("idle", "leased"):
            with suppress(Exception):
                rss = self.backend.rss_kib()
        return out | {"rss_kib": rss, "rss_cap_kib": self.rss_cap_kib}

    def _acquire(self, conn: _sock.socket) -> bool:
        """Wait for the lease lock. False when the client hung up or the slot has exited (the caller refuses). While a
        stop is pending the lock is left to the stopper."""
        while not self._done.is_set():
            if _peer_gone(conn):
                return False
            if self._stopping.is_set():
                self._done.wait(_POLL_S)
                continue
            if self._lease_lock.acquire(timeout=_POLL_S):
                if self._done.is_set() or self._stopping.is_set():
                    self._lease_lock.release()
                    continue
                return True
        return False

    def _serve_lease(self, conn: _sock.socket, lines: _Lines, request: LeaseRequest) -> None:
        if not self._acquire(conn):
            if self._done.is_set():
                with suppress(OSError):
                    _send(conn, {"op": "refused", "why": [f"slot exited: {self._failure or 'stopped'}"]})
            return
        try:
            if _peer_gone(conn):  # gave up while we took the lock: no attempt ran, nothing to recycle
                return
            try:
                mode = self._prepare(request)
            except Exception as exc:  # noqa: BLE001 - a backend that raises cannot be trusted warm: exit, captures run cold
                self._fail(f"prepare: {exc!r}")
                mode = None
            if mode is None:  # the slot failed while preparing (and has exited)
                with suppress(OSError):
                    _send(conn, {"op": "refused", "why": [f"slot exited: {self._failure}"]})
                return
            granted = not _peer_gone(conn)
            ended: tuple[str, int] | None = None
            if granted:
                try:
                    _send(conn, {"op": "grant", "grant": asdict(self._grant(mode))})
                except Exception as exc:  # noqa: BLE001 - the peer left, or pins() raised: no lease ran
                    # Still falls through to _after_lease + _finish, so a vacate exits and a cold grant reboots.
                    _log("grant failed", error=repr(exc), mode=mode)
                    granted = False
            if granted:
                self._set(state="leased")
                _log("grant", kind="warm" if mode == "warm" else "cold", mode=mode)
                ended = self._wait_release(lines)
            reply = self._after_lease(mode, ended, granted=granted)
            if ended is not None:
                with suppress(OSError):
                    _send(conn, {"op": "released", **reply})
            self._finish(mode, reply)
        finally:
            self._lease_lock.release()

    def _wait_release(self, lines: _Lines) -> tuple[str, int] | None:
        """The release's ``(status, jobs)``, or None when the connection closed (or misbehaved) first: a dropped lease."""
        try:
            message = lines.read()
            if message is None or message.get("op") != "release":
                return None
            return str(message.get("status")), int(message.get("jobs") or 0)
        except (OSError, ValueError, TypeError):  # a malformed release counts as dropped: the engine is not reused
            return None

    def _prepare(self, request: LeaseRequest) -> Literal["warm", "cold", "vacate"] | None:
        """Every check before a grant; returns how to grant. ``vacate`` = cold, then exit when the lease ends."""
        pins = self.backend.pins()
        differ = [k for k in PIN_KEYS if getattr(request, k) != pins.get(k)]
        if differ:
            # The slot booted from another install (older code, other engine files or parser): it must not serve this
            # capture warm, nor boot again from its own code. Free the ports for the capture's cold path and exit after.
            _log("pins differ", fields=differ)
            if self._up:
                self._teardown("pins differ: " + ", ".join(differ))
            return "vacate"
        if request.want == "cold":
            if self._up:
                self._teardown("cold lease")
            return "cold"
        if not self._up and not self._boot_backend():
            return None
        problems = self.backend.health()
        if problems:
            if not self._recycle([f"health: {p}" for p in problems]):
                return None
            problems = self.backend.health()
        if problems:
            # Still unhealthy on a fresh engine: grant cold (the capture keeps the mutex, on free ports), then exit.
            _log("unhealthy after recycle", problems=problems)
            self._teardown("unhealthy after recycle")
            self._set(failure="unhealthy after recycle: " + "; ".join(problems))
            return "vacate"
        return "warm"

    def _grant(self, mode: str) -> Grant:
        pins = self.backend.pins() if mode == "warm" else {}
        with self._state_lock:
            if mode != "warm":
                return Grant(kind="cold", engine=None, parser=None, boot={}, served=dict(self._served))
            self._served["leases"] += 1
            self._lifetime["leases"] += 1
            return Grant(
                kind="warm",
                engine=self._engine,
                parser=self._parser,
                boot={**self._boot, "pins": pins},
                served=dict(self._served),
            )

    def _after_lease(self, mode: str, ended: tuple[str, int] | None, *, granted: bool) -> dict:
        """The ``released`` answer: whether the engine is reused. Only a warm lease that ended ``complete``, with the
        parser back at rest and the RSS under the cap, keeps it."""
        if mode != "warm" or not granted:
            return {"recycle": False, "why": []}
        why: list[str] = []
        if ended is None:
            why.append("dropped: the connection closed without a release")
        else:
            status, jobs = ended
            with self._state_lock:
                self._served["jobs"] += jobs
                self._lifetime["jobs"] += jobs
            if status != "complete":
                why.append(f"release status {status!r}")
            else:
                try:
                    why += [f"rest: {p}" for p in self._bounded(self.backend.rest, REST_DEADLINE_S, "rest")]
                    rss = self.backend.rss_kib()  # None = unknown, not over the cap
                except Exception as exc:  # noqa: BLE001 - an engine that cannot be rested is not reused
                    why.append(f"rest: {exc!r}")
                    rss = None
                if rss is not None and rss > self.rss_cap_kib:
                    why.append(f"rss {rss} KiB > cap {self.rss_cap_kib} KiB")
        return {"recycle": bool(why), "why": why}

    def _finish(self, mode: str, reply: dict) -> None:
        # Runs after the ``released`` reply, still holding the lease: the capture child does not wait for a reboot,
        # and the next lease does not start before it.
        if self._done.is_set():
            return  # the slot failed during the lease (a rest that never returned): nothing is booted again
        if mode == "vacate":
            self._exit(self._failure or "pins differ")
        elif self._stopping.is_set():
            return  # the stopper tears down next
        elif mode == "cold":
            self._boot_backend()
        elif reply["recycle"]:
            self._recycle(reply["why"])
        else:
            self._set(state="idle")


# -- the real backend and ``mk slot serve`` ----------------------------------------------------------------------------


def _write_state(path: Path, state: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, sort_keys=True))
    os.replace(tmp, path)


def _gone(pid: int, named: dict, wait_s: float) -> bool:
    """True once ``pid`` is no longer the ``named`` process (exited, a zombie, or the pid reused), within ``wait_s``."""
    deadline = time.monotonic() + wait_s
    while same_process(proc_identity(pid), named):
        if time.monotonic() > deadline:
            return False
        time.sleep(_POLL_S)
    return True


def runtime_engine_pids(data_dir: Path | None) -> set[int]:
    """The runtime-grammar engines (``memory_kernel.runtime.engine``) running from ``data_dir``: the pids their state
    files (``<data_dir>/runtime-fcg/*/engine.json``) name AND ``/proc`` confirms (``same_process``). Only those: a state
    file naming a pid that is now another process proves nothing."""
    if data_dir is None:
        return set()
    pids = set()
    for path in sorted(Path(data_dir).glob("runtime-fcg/*/engine.json")):
        try:
            named = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(named, dict) and isinstance(named.get("pid"), int):
            identity = proc_identity(named["pid"])
            if identity is not None and identity["uid"] == os.getuid() and same_process(identity, named):
                pids.add(named["pid"])
    return pids


def clear_orphans(state: dict, *, data_dir: Path | None = None) -> list[dict]:
    """Kill the engine and parser a dead slot left running, and nothing else. A process is signalled only when it is
    this user's and its /proc identity matches ``state`` (``slot-state.json``) on pid, start_ticks and exe: SIGTERM,
    then SIGKILL if it is still that process after ``ORPHAN_GRACE_S``. Then any other own FCG or parser process raises
    ``Stop``: the slot never starts beside a process it cannot prove it started. A runtime-grammar engine its state file
    under ``data_dir`` proves (``runtime_engine_pids``) is neither killed nor counted. Returns what it killed."""
    killed: list[dict] = []
    for what in ("engine", "parser"):
        named = state.get(what)
        if not isinstance(named, dict) or not isinstance(named.get("pid"), int):
            continue
        identity = proc_identity(named["pid"])
        if identity is None or identity["uid"] != os.getuid() or not same_process(identity, named):
            continue
        pid = named["pid"]
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
        gone = _gone(pid, named, ORPHAN_GRACE_S)
        if not gone:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            gone = _gone(pid, named, ORPHAN_GRACE_S)
        killed.append(
            {"what": what, "pid": pid, "start_ticks": named["start_ticks"], "exe": named["exe"], "gone": gone}
        )
    gone_pids = {k["pid"] for k in killed if k["gone"]}  # a killed engine may linger as a zombie still named FCG Go
    # A zombie (state Z) holds no port, memory or engine state, and only its parent can reap it: in a container whose
    # PID 1 never reaps, an earlier slot's killed engine stays a zombie named FCG Go for the container's life, and
    # counting it would refuse every later slot. Every other process counts, readable identity or not.
    procs = own_fcg_processes() + own_parser_processes()
    spared = gone_pids | runtime_engine_pids(data_dir)
    unproven = sorted({p["pid"] for p in procs if p["pid"] not in spared and proc_state(p["pid"]) not in ("Z", "X")})
    if unproven:
        raise Stop(
            f"own FCG or parser processes the slot state does not name (or that survived SIGKILL): {unproven}; "
            f"killed {killed}. Stop them, then start the slot"
        )
    return killed


class FcgBackend:
    """The slot's engine and managed parser, launched as a cold capture attempt launches them (``Engine.launch``,
    ``ParserService.launch``, with their ownership and pin checks), from the engine files and the pins ``mk setup``
    placed under ``data_dir``. The engine is armed with the same ``Engine.arm``, but on the live parser (its rest value)
    and in this process's environment: a cold run attempt arms on the dead parser and switches to the live one before
    phase B. The lease pins (``ARM_ENV``, the parser's data dir) make a capture whose environment or data dir differs
    run cold. Each boot logs under ``<run_dir>/<boot timestamp>/``.

    The code pins are hashed once, at construction, after importing what arming and the readback load: they describe
    the code this process runs, so a slot left running across a deploy reports its own code and vacates. With
    ``state_file``, the engine and parser identities are written there as each one starts, and removed after a clean
    teardown (``clear_orphans`` reads it)."""

    def __init__(
        self,
        data_dir: Path,
        run_dir: Path,
        *,
        engine_port: int = 9701,
        parser_port: int = 9801,
        state_file: Path | None = None,
    ) -> None:
        import memory_kernel.construction.fcg.comprehend_cancel  # noqa: F401 - imported before hashing (see above)
        import memory_kernel.construction.fcg.grammar_store  # noqa: F401
        import memory_kernel.construction.fcg.penelope_host  # noqa: F401
        import memory_kernel.construction.fcg.provisional_units  # noqa: F401
        import memory_kernel.pipeline.stages._fenced_http  # noqa: F401
        import memory_kernel.pipeline.stages._resident_stock_capture_runner  # noqa: F401
        from memory_kernel import setup_runtime
        from memory_kernel.pipeline.engine import EngineConfig, ParserConfig, managed_parser_url
        from memory_kernel.pipeline.provenance import code_path
        from memory_kernel.pipeline.stages import _parse_capture as pc

        binary = setup_runtime.managed_path(data_dir, "fcg-go")
        image = setup_runtime.managed_path(data_dir, "fcg-image")
        if binary is None or image is None:
            raise ValueError(f"no FCG engine under {data_dir}: run `mk setup --component fcg-go --component fcg-image`")
        components = setup_runtime.load_pins()["components"]
        self.live_parser = managed_parser_url(parser_port)
        self.engine_cfg = EngineConfig(
            port=engine_port,
            binary=binary,
            image=image,
            parser_url=self.live_parser,
            binary_sha256=components["fcg-go"]["binary_sha256"],
            image_sha256=components["fcg-image"]["sha256"],
        )
        self.parser_cfg = ParserConfig(port=parser_port, data_dir=data_dir)
        self.run_dir, self.state_file = run_dir, state_file
        self._code_pins = {p: pc.sha(code_path(p).read_bytes()) for p in pc.PIN_FILES}
        self._walker = pc.current_walker_sha256()
        self._shas: tuple[str | None, str | None] = (None, None)  # of the files the current boot launched
        self._paths: tuple[str | None, str | None] = (None, None)  # their strictly resolved paths
        self._arm_env: dict[str, str | None] | None = None  # ARM_ENV as read at the current boot
        self._parser_data_dir: str | None = None  # the parser's data dir, strictly resolved at the current boot
        self._boots = 0
        self.engine = None
        self.parser = None
        self._ids: tuple[dict, dict] | None = None

    def _save_state(self) -> None:
        if self.state_file is None:
            return
        state = {}
        for what, handle in (("engine", self.engine), ("parser", self.parser)):
            process = getattr(handle, "process", None)
            identity = proc_identity(process.pid) if process is not None else None
            if identity is not None:
                state[what] = {k: identity[k] for k in ("pid", "start_ticks", "exe")}
        _write_state(self.state_file, state)

    def boot(self) -> dict:
        from memory_kernel.pipeline.engine import Engine, ParserService, sha_file
        from memory_kernel.pipeline.stages._fenced_http import GRAMMAR

        if self.engine is not None or self.parser is not None:
            raise Stop("slot backend: boot while still booted (teardown first)")
        self._boots += 1
        boot_dir = self.run_dir / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}-{self._boots}"
        self._shas = (sha_file(self.engine_cfg.binary), sha_file(self.engine_cfg.image))
        self._paths = (
            str(self.engine_cfg.binary.resolve(strict=True)),
            str(self.engine_cfg.image.resolve(strict=True)),
        )
        self._arm_env = arm_env()
        data_dir = self.parser_cfg.data_dir
        self._parser_data_dir = str(data_dir.resolve(strict=True)) if data_dir is not None else None
        self.parser = ParserService(self.parser_cfg, boot_dir / "parser")  # set first: teardown stops a half boot
        # Each process is in the state file from its spawn on, before the readiness waits: a slot killed mid-boot leaves
        # nothing the next slot's clear_orphans cannot prove (an unnamed half-started parser would refuse it, and the
        # cold path's stray check with it).
        parser_launch = self.parser.launch(on_spawn=self._save_state)
        self.engine = Engine(self.engine_cfg, boot_dir / "engine")
        engine_launch = self.engine.launch(on_spawn=self._save_state)
        arm = self.engine.arm(self.live_parser)
        problems, _ = self.engine.readback()
        if problems:
            raise Stop(f"slot boot readback: {problems}")
        parser_id = proc_identity(self.parser.process.pid)
        if parser_id is None:
            raise Stop("slot boot: the parser is gone after the arm")
        keys = ("pid", "start_ticks", "exe")
        self._ids = (
            {**{k: engine_launch[k] for k in keys}, "port": self.engine_cfg.port},
            {**{k: parser_id[k] for k in keys}, "url": self.live_parser},
        )
        return {
            "dir": str(boot_dir),
            "engine": engine_launch,
            "parser": parser_launch,
            "arm": {**arm, "grammar": GRAMMAR},
        }

    def identities(self) -> tuple[dict, dict]:
        if self._ids is None:
            raise Stop("slot backend: not booted")
        return self._ids

    def health(self) -> list[str]:
        """The checks before a grant: both processes still the booted ones, a clean readback (parser at rest =
        LIVE, searches 0, the supplier and config contract) and idle engine CPU over 3 s."""
        from memory_kernel.pipeline.stages._parse_capture import IDLE_CPU_MAX_S

        if self.engine is None or self._ids is None:
            return ["not booted"]
        problems = [
            f"{what} is no longer the booted process {granted}"
            for what, granted in zip(("engine", "parser"), self._ids, strict=True)
            if not same_process(proc_identity(granted["pid"]), granted)
        ]
        if problems:
            return problems
        if self.engine.parser != self.live_parser:
            return [f"engine parser {self.engine.parser} is not at rest ({self.live_parser})"]
        try:
            bad, _ = self.engine.readback()
            idle = self.engine.idle_cpu(3.0)
        except Exception as exc:  # noqa: BLE001 - an engine that cannot answer is unhealthy
            return [f"readback / idle CPU: {exc!r}"]
        problems += [f"readback: {p}" for p in bad]
        if idle > IDLE_CPU_MAX_S:
            problems.append(f"engine not idle: {idle:.2f}s cpu over 3s")
        return problems

    def rest(self) -> list[str]:
        if self.engine is None:
            return ["not booted"]
        self.engine.set_parser(self.live_parser)
        bad, _ = self.engine.readback()
        return [f"readback: {p}" for p in bad]

    def rss_kib(self) -> int | None:
        from memory_kernel.pipeline.engine import rss_kib

        return rss_kib(self._ids[0]["pid"]) if self._ids is not None else None

    def pins(self) -> dict:
        return {
            "binary_sha256": self._shas[0],
            "image_sha256": self._shas[1],
            "binary_path": self._paths[0],
            "image_path": self._paths[1],
            "engine_port": self.engine_cfg.port,
            "live_parser": self.live_parser,
            "parser_data_dir": self._parser_data_dir,
            "arm_env": dict(self._arm_env) if self._arm_env is not None else None,
            "code_pins": dict(self._code_pins),
            "walker_sha256": self._walker,
        }

    def teardown(self) -> dict:
        """Stop the engine, then the parser, whatever the boot reached (parser up and engine not, engine launched and
        not armed). Idempotent. The state file is removed unless a stop raised (its processes may still run)."""
        from memory_kernel.pipeline.engine import parser_teardown_problems
        from memory_kernel.pipeline.stages._parse_capture import teardown_problems

        receipt: dict = {}
        try:
            if self.engine is not None:
                try:
                    receipt["engine"] = self.engine.teardown()
                except Exception as exc:  # noqa: BLE001 - recorded; the parser is still stopped
                    receipt["engine"] = {"error": repr(exc)}
        finally:
            if self.parser is not None:
                try:
                    receipt["parser"] = self.parser.teardown()
                except Exception as exc:  # noqa: BLE001 - recorded
                    receipt["parser"] = {"error": repr(exc)}
            self.engine, self.parser, self._ids = None, None, None
        receipt["problems"] = (teardown_problems(receipt["engine"]) if "engine" in receipt else []) + (
            parser_teardown_problems(receipt["parser"]) if "parser" in receipt else []
        )
        raised = any("error" in part for part in (receipt.get("engine") or {}, receipt.get("parser") or {}))
        if self.state_file is not None and not raised:
            with suppress(FileNotFoundError):
                self.state_file.unlink()
        return receipt


def main_serve(
    socket: Path, data_dir: Path, *, engine_port: int = 9701, parser_port: int = 9801, rss_cap_gib: float = 6.0
) -> int:
    """``mk slot serve``: refuse if a slot already answers on ``socket``; listen on it at once, so a capture arriving
    from now on waits for the lease instead of running cold beside the boot; then, under the lease lock, clear the
    orphans ``slot-state.json`` names (``clear_orphans``: any unproven own FCG or parser process fails the slot, and the
    waiting captures run cold), boot, and serve until ``mk slot stop``, SIGTERM or SIGINT. 0 after a clean stop, 2 when
    refused or when the slot failed (one JSON line on stderr says why)."""
    directory = socket.parent
    if not directory.exists():
        directory.mkdir(mode=0o700, parents=True)
    try:
        probe = _connect(socket)
    except SlotUnavailable:
        probe = None
    if probe is not None:  # checked before anything else: a live slot's engine matches the state file too
        probe.close()
        _log("refused", why=f"another slot is serving on {socket}")
        return 2
    state_file = directory / STATE_FILE

    def setup() -> SlotBackend:
        try:
            state = json.loads(state_file.read_text())
        except FileNotFoundError:
            state = {}
        except (OSError, ValueError) as exc:  # unreadable: it proves nothing, so nothing is killed
            _log("state unreadable", file=str(state_file), error=repr(exc))
            state = {}
        killed = clear_orphans(state if isinstance(state, dict) else {}, data_dir=data_dir)
        if killed:
            _log("orphans killed", killed=killed)
        with suppress(FileNotFoundError):
            state_file.unlink()
        return FcgBackend(
            data_dir, data_dir / "slot", engine_port=engine_port, parser_port=parser_port, state_file=state_file
        )

    try:
        server = SlotServer(socket, rss_cap_kib=int(rss_cap_gib * 1024 * 1024), setup=setup)
    except Exception as exc:  # noqa: BLE001 - a busy socket, or a path that is not a socket: the start is refused
        _log("refused", why=repr(exc))
        return 2

    def on_signal(_signum, _frame) -> None:
        server.shutdown()

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        server.serve_forever()
    except Stop as exc:
        _log("failed", why=str(exc))
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0
