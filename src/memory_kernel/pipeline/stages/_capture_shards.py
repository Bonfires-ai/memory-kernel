"""Parallel capture: k shard children, one engine port each, one shared managed parser, then the gated merge."""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from memory_kernel.pipeline.engine import PORT_RANGE, PROTECTED_PORTS
from memory_kernel.pipeline.stages import _parse_capture as pc

MAX_SHARDS = 8


def effective_shards(requested: int, usable: int) -> int:
    return max(1, min(requested, usable, MAX_SHARDS))


def usable_cpus() -> int:
    return len(os.sched_getaffinity(0))


def shard_ports(base: int, k: int) -> list[int]:
    ports = [base + i for i in range(k)]
    if any(p not in PORT_RANGE or p in PROTECTED_PORTS for p in ports):
        raise ValueError(f"shard ports {ports} leave the private 97xx range")
    return ports


def recorded_shard_counts(out: Path) -> set[int]:
    return {
        int(m.group(2))
        for f in out.glob("report-*-of-*-a*.json")
        if (m := re.match(r"report-(\d+)-of-(\d+)-a", f.name))
    }


def recorded_shard_count(out: Path) -> int | None:
    """The shard count the capture's existing reports were made with (None = no report yet). It is fixed for the
    capture's life: a resume uses it whatever count is requested or the CPU clamp would give."""
    seen = recorded_shard_counts(out)
    if len(seen) > 1:
        raise ValueError(f"capture {out} holds reports of several shard counts {sorted(seen)}")
    return next(iter(seen), None)


def check_shard_count(out: Path, k: int) -> None:
    """A capture keeps one shard count for life: reports of another k mean a resume with a different k (refused)."""
    seen = recorded_shard_counts(out)
    if seen and seen != {k}:
        raise ValueError(f"capture {out} was started with shard count {sorted(seen)}, not {k}")


def pending_shards(out: Path, k: int) -> list[int]:
    """The shards whose latest report is not complete (a complete shard is never run again: the merge gate wants
    exactly one complete report per shard)."""
    done = pc.latest_reports(out)
    return [i for i in range(k) if (done.get(pc.shard_tag((i, k))) or {}).get("status") != "complete"]


def result_path(out: Path, i: int, k: int) -> Path:
    return out / f"shard-{i}-of-{k}.result.json"


def log_path(out: Path, i: int, k: int) -> Path:
    return out / f"shard-{i}-of-{k}.log"


class Terminated(RuntimeError):
    """SIGTERM reached the parent: unwound as an exception so every ``finally`` (children, shared parser) runs."""


@contextlib.contextmanager
def sigterm_as_exception() -> Iterator[None]:
    def handler(_signum, _frame):
        raise Terminated("SIGTERM")

    try:
        previous = signal.signal(signal.SIGTERM, handler)
    except ValueError:  # not the main thread: no handler can be installed
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)


def site_doc(site) -> dict:
    e = site.engine
    return {
        "root": str(site.root),
        "parser_lock": str(site.parser_lock),
        "parser_url": e.parser_url,
        "binary": str(e.binary),
        "image": str(e.image),
        "binary_sha256": e.binary_sha256,
        "image_sha256": e.image_sha256,
    }


def file_logger(path: Path) -> Callable[[str], None]:
    """A line logger appending to ``path`` (the stage context has no logger; the capture dir keeps the shard log)."""

    def log(message: str) -> None:
        with path.open("a") as handle:
            handle.write(message + "\n")

    return log


def child_env() -> dict[str, str]:
    """The shard child's environment: a source tree is imported from THIS tree (as ``ParserService.child_env`` does),
    never from another install; an installed package needs no ``PYTHONPATH``."""
    from memory_kernel.pipeline.provenance import package_dir

    env = dict(os.environ)
    pkg = package_dir()
    if pkg.parent.name == "src":
        env["PYTHONPATH"] = str(pkg.parent)
    else:
        env.pop("PYTHONPATH", None)
    return env


def stop_children(children: list, *, grace_s: float = 60.0) -> None:
    """SIGTERM every child still running (its ``_driven`` turns that into Stop and its ``run_shard`` tears the engine
    down), then SIGKILL the ones that outlast ``grace_s``."""
    live = [c for c in children if c.poll() is None]
    for child in live:
        with contextlib.suppress(ProcessLookupError):
            child.terminate()
    deadline = time.monotonic() + grace_s
    for child in live:
        try:
            child.wait(max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                child.kill()
            child.wait()


def shard_report(out: Path, i: int, k: int, returncode: int | None) -> dict:
    """The child's result file; a missing or unreadable one (or a non-zero exit) is a crash, reported with the log."""
    crashed = {"shard": [i, k], "status": f"crashed rc={returncode}; see {log_path(out, i, k)}"}
    if returncode != 0:
        return crashed
    try:
        report = json.loads(result_path(out, i, k).read_bytes())
    except (OSError, ValueError):
        return crashed
    return report if isinstance(report, dict) else crashed


def run_parallel(out: Path, site, k: int, log) -> list[dict]:
    """Start a ``_capture-shard`` child for every pending shard (engine ports base..base+k-1) and wait for all; return
    the reports of the shards it ran. Each child writes ``shard-<i>-of-<k>.result.json`` and logs to
    ``shard-<i>-of-<k>.log`` in ``out``; the parent reads only the result file. Whatever way this exits (an exception,
    SIGTERM), every child still running is terminated, then killed after a grace period."""
    check_shard_count(out, k)
    ports = shard_ports(site.engine.port, k)
    pending = pending_shards(out, k)
    doc = json.dumps(site_doc(site))
    env = child_env()
    children: dict[int, subprocess.Popen] = {}
    try:
        with sigterm_as_exception():
            for i in range(k):
                if i not in pending:
                    log(f"capture shard {i}/{k}: already complete")
                    continue
                result_path(out, i, k).unlink(missing_ok=True)
                argv = [
                    sys.executable,
                    "-P",
                    "-m",
                    "memory_kernel.cli",
                    "_capture-shard",
                    str(out),
                    str(i),
                    str(k),
                    str(ports[i]),
                    doc,
                ]
                with log_path(out, i, k).open("ab") as handle:
                    children[i] = subprocess.Popen(
                        argv, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, env=env
                    )
                log(f"capture shard {i}/{k} on port {ports[i]} (pid {children[i].pid})")
            for child in children.values():
                child.wait()
    finally:
        stop_children(list(children.values()))
    return [shard_report(out, i, k, child.returncode) for i, child in children.items()]
