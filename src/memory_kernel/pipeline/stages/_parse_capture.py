"""Parse capture, ported from a reference-harness module (unreached top-level definitions pruned).

Differences from the reference module: a frozen Site(root, EngineConfig, parser lock) replaces the module globals
(ROOT / BINARY / IMAGE / LIVE_PARSER and the spare-engine class) and the harness lane rebinds; the parser flock is
taken around parser_probe / capture_syntax / run_phase_b; PIN_FILES names this package's files. With no external
parser URL, the managed parser (Site.parser) is started around each syntax / run attempt (_with_parser). The code
pins and the snapshot files are hashed in the IMPORTED package (provenance.code_path / tree_census; Site.code_root
only in tests and the gate's negative controls); Site.root is the base of the data paths (syntax_from) only.

The capture is hybrid (fence-first, per-pass 5 s, stand-in fallback for timed-out passes) and runs on the
build's own engine (``engine.Engine``), or, under a capture-slot lease (``grant``), on the slot's engine
(``engine.AttachedEngine``) with the slot's parser.

hybrid = phase A (every unit's fenced or whole-tree passes, parser host DEAD: no pass can re-parse) ->
phase B (stand-in fallback over each timed-out pass's subtree, parser host LIVE: segments need a fresh parse)
-> merge (offline, deterministic, shard-independent) writing one sealed fenced-unit record per unit.

The parse stage drives it through these functions (``Site`` carries the data root, the engine config, the
parser lock and the managed parser):

    status(out)                                     # what is already on disk
    syntax(syn, sources_path, site=site)            # stored parses, if absent
    prepare(out, sources_path, syn, live_parser=..., site=site, snapshot_pins=...)
    run(out, (i, k), site=site)                     # one shard; shards run serially
    gated_merge(out, site)                          # after every shard is complete

``syntax`` writes ``NNNN/syntax-XX.json`` (the stock capture's stored-parse layout and bytes) with ONE
parse per non-blank message part through ``capture_warm_syntax`` and no FCG unit pass; it is
write-once and resumable, and its per-part seconds land in ``syntax-seconds/`` for the projection.

Phase B runs only after a live parser probe (one trivial parse) succeeds. A fallback whose syntax call failed
on parser transport stops the shard (``stopped: parser outage``) and writes nothing, so a resume retries it.
The live parser URL is sealed in the manifest and must match on every run. ``--snapshot-pins`` seals a list
of file sha256s into the manifest; ``run``, ``merge`` and acceptance re-check it and VOID on any difference
(or when the declared pins file is missing). ``merge`` builds in ``.merge-N/`` and promotes each message dir
by rename, so a failure partway leaves the capture re-mergeable.

Shards run SERIALLY on one host: ``Engine.launch`` refuses while any own FCG process exists. A leased attempt
launches nothing: ``_slot_gate`` admits it only when the own FCG and parser processes are exactly the leased ones.
A re-run of a shard (resume after a stop/crash) is a new attempt: ``run-<i>-of-<k>-aN.log`` / ``report-...-aN.json``
/ ``driver-...-aN.pid`` (N = first free); phase rows already on disk are skipped, never rewritten. The
merge gate counts only ``complete`` reports with a clean teardown, or for a leased attempt a clean release
(``end_problems``; one shard count k, every i exactly once);
stopped/crashed attempts are listed in ``merge.json`` and ignored. Every row/report write is atomic.

Resuming a capture made elsewhere: a COMPLETE capture made elsewhere is read as is; an INCOMPLETE one cannot
be resumed by mk (VOID); start a new capture. The manifest of a capture seals the code pins, the live parser URL and
the snapshot pins of the run that wrote it, and ``run`` / ``merge`` re-check all three, so a half-done capture made by
another harness (other pinned files, another parser address) fails that check on its first resume.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memory_kernel.construction.fcg import clause_units as cu
from memory_kernel.construction.fcg import clause_units_request as rq
from memory_kernel.construction.fcg import clause_units_standin as si
from memory_kernel.construction.memory_semantic_jev import check_seal, seal, sha
from memory_kernel.construction.mined_affordance_contract import canonical_json
from memory_kernel.grammar.dependencies import extraction_model_record, parser_change_problem
from memory_kernel.pipeline.engine import (
    DEAD_PARSER,
    AttachedEngine,
    Engine,
    EngineConfig,
    ParserConfig,
    ParserService,
    Stop,
    own_fcg_processes,
    own_parser_processes,
    parser_lock,
    parser_teardown_problems,
    pin_problems,
    port_listening,
    proc_identity,
    release_problems,
    same_process,
    sha_file,
)
from memory_kernel.pipeline.provenance import code_path, package_dir, tree_census

if TYPE_CHECKING:
    from memory_kernel.pipeline.slot import Grant

HYBRID_CAPTURE_SCHEMA = "mk-hybrid-capture/v1"

MESSAGE_SCHEMA = "mk-hybrid-message/v1"

SERVING_S = 5

ENCODING = "B1"


# sha256 of the NATIVE_ANALYSIS_WALKER string (not the file).
WALKER_SHA256 = "d75865d2e353576cfd244167f9e4e039f51fb5b33a60e7327f4caeb1ee6dde11"

# The capture and gate path's import closure: the hybrid modules, the transport (``engine`` and the
# probe module it takes http_body / decode_response / GRAMMAR from, with that module's own imports), the
# phase-B request/record builders, the stored-parse readers, the arming shims, the walker's file and the
# consumer that loads the merged capture, plus the load-time imports of the stored-parse readers (the Lisp
# session wrapper of every syntax request, the sub-unit split, seal/sha and canonical_json of every row).
# One import level deep, by design: the transitive closure would pull in unrelated modules. The walker
# STRING is pinned separately (walker_sha256). ``_capture_cache`` writes cached rows into a capture, so it is pinned too.
# ``slot`` is pinned because a leased attempt runs on the engine and parser it booted, armed and checked.
PIN_FILES = (
    "src/memory_kernel/construction/fcg/clause_units.py",
    "src/memory_kernel/construction/fcg/clause_units_request.py",
    "src/memory_kernel/construction/fcg/clause_units_standin.py",
    "src/memory_kernel/construction/fcg/native_analysis.py",
    "src/memory_kernel/construction/fcg/native_warm_source_units.py",
    "src/memory_kernel/construction/fcg/native_frame_transport.py",
    "src/memory_kernel/construction/fcg/grammar_store.py",
    "src/memory_kernel/construction/fcg/penelope_host.py",
    "src/memory_kernel/construction/fcg/provisional_units.py",
    "src/memory_kernel/construction/fcg/comprehend_cancel.py",
    "src/memory_kernel/construction/fcg/native_warm_session_lisp.py",
    "src/memory_kernel/construction/memory_warm_message.py",
    "src/memory_kernel/construction/memory_discourse_units.py",
    "src/memory_kernel/construction/memory_discourse_contract.py",
    "src/memory_kernel/construction/memory_discourse_native.py",
    "src/memory_kernel/construction/memory_semantic_jev.py",
    "src/memory_kernel/construction/mined_affordance_contract.py",
    "src/memory_kernel/pipeline/engine.py",
    "src/memory_kernel/pipeline/slot.py",
    "src/memory_kernel/pipeline/stages/_parse_capture.py",
    "src/memory_kernel/pipeline/stages/_capture_cache.py",
    "src/memory_kernel/pipeline/stages/_parse_hybrid_pairs.py",
    "src/memory_kernel/pipeline/stages/_fenced_http.py",
    "src/memory_kernel/pipeline/stages/_unicode_repair.py",
    "src/memory_kernel/pipeline/stages/_capture_record.py",
    "src/memory_kernel/pipeline/stages/_resident_stock_capture_runner.py",
    "src/memory_kernel/parser_service/__init__.py",
    "src/memory_kernel/parser_service/__main__.py",
    "src/memory_kernel/parser_service/app.py",
    "src/memory_kernel/parser_service/cache.py",
    "src/memory_kernel/parser_service/config.py",
    "src/memory_kernel/parser_service/constituency.py",
    "src/memory_kernel/parser_service/facade.py",
    "src/memory_kernel/parser_service/facade_transform.py",
    "src/memory_kernel/parser_service/registry.py",
)

SNAPSHOT_FILE = "snapshot-pins.json"

SNAPSHOT_SCHEMA = "parse-hybrid-snapshot-pins/v1"

DECLARED_SECTIONS = ("snapshot_pins", "syntax_manifest_sha256")  # sealed at prepare, re-checked, not recomputed

SYNTAX_CAPTURE_SCHEMA = "mk-hybrid-syntax/v1"

SYNTAX_MANIFEST = "syntax-manifest.json"

PROBE_TEXT = "The parser is alive."

# Engine error text that names the parser transport (the Lisp side of a refused / reset / unreachable host).
PARSER_TRANSPORT_MARKERS = ("USOCKET", "CONNECTION-REFUSED", "CONNECTION REFUSED", "ECONNREFUSED", "UNREACHABLE")

IDLE_CPU_MAX_S = 0.0  # global constraint: idle engine CPU must be 0.0 s over 3 s (gate is `idle > 0.0`)


@dataclass(frozen=True)
class Site:
    """Where a capture runs: the data root its ``syntax_from`` path is relative to and the engine it drives.

    Replaces the module globals ``ROOT`` / ``BINARY`` / ``IMAGE`` / ``LIVE_PARSER`` of the reference harness and its
    lane rebinds: one engine per build, no sibling-lane process filter, and the parser flock taken explicitly around
    the three calls the lanes wrapped. The code
    pins and snapshot files are hashed in ``code_root`` (None = the imported package; a copy only in tests and the
    gate's negative controls)."""

    root: Path
    engine: EngineConfig
    parser_lock: Path
    pin_files: tuple[str, ...] = PIN_FILES
    code_root: Path | None = None
    #: the managed parser to run around each syntax / run attempt; None = ``engine.parser_url`` is an external
    #: service (``--parser-url``) and nothing is started
    parser: ParserConfig | None = None

    @property
    def live_parser(self) -> str:
        return self.engine.parser_url


class ParserOutage(Stop):
    """The live parser cannot parse: the shard stops (nothing written), and a resume retries."""


def in_shard(order: int, shard: tuple[int, int]) -> bool:
    return order % shard[1] == shard[0]


def parse_shard(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)/(\d+)", text)
    if not m or not 0 <= int(m[1]) < int(m[2]):
        raise ValueError(f"shard must be i/k with 0 <= i < k, got {text!r}")
    return int(m[1]), int(m[2])


def shard_tag(shard: tuple[int, int]) -> str:
    return f"{shard[0]}-of-{shard[1]}"


def astral(text: str) -> set[str]:
    return {c for c in text if ord(c) > 0xFFFF}


def _atomic_write_once(path: Path, text: str) -> None:
    """Write-once AND atomic: the bytes go to a temp name in the same directory (fsynced), then appear under
    the final name in one step via ``os.link``, which refuses an existing name (FileExistsError) -- a
    rename-if-absent without the check/rename race. A truncated row is never visible under the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def _write_once(path: Path, value) -> None:
    _atomic_write_once(path, canonical_json(value))


def write_sources(out: Path, sources) -> None:
    _write_once(out / "sources.json", sources)


def _engine_dead(engine) -> bool:
    """True when the engine process has exited or its port is down (fakes without a process are alive)."""
    process = getattr(engine, "process", None)
    if process is None:
        return False
    if process.poll() is not None:
        return True
    port = getattr(engine, "port", None)
    return port is not None and not port_listening(port)


def hybrid_jobs(sources, syntax_dir: Path):
    """``(order, k, job)`` over the STORED syntax captures (the parse is reused, never re-fetched).

    ``k`` numbers a message's jobs across its parts exactly as the stock capture numbers ``unit-KK``."""
    from memory_kernel.construction.memory_warm_message import warm_message_jobs, warm_message_parts

    jobs = []
    for source in sources:
        root = Path(syntax_dir) / f"{source['order']:04d}"
        k = 0
        for index, part in enumerate(warm_message_parts(source)):
            if not part["source"]["text"].strip():
                continue
            syntax = json.loads((root / f"syntax-{index:02d}.json").read_bytes())
            for job in warm_message_jobs(source, part, syntax):
                jobs.append((source["order"], k, job))
                k += 1
    return jobs


def _phase_a_rows(out: Path):
    # an order is zero-padded to four digits at least: wider past 9999, so sort numerically
    for path in sorted((out / "phase-a").glob("[0-9][0-9][0-9][0-9]*/unit-[0-9][0-9].json"), key=_order_k):
        yield path, json.loads(path.read_bytes())


def _order_k(path: Path) -> tuple[int, int]:
    return int(path.parent.name), int(path.stem.split("-")[1])


def load_job(out: Path, order: int, k: int):
    job = json.loads((out / "jobs" / f"{order:04d}-{k:02d}.json").read_bytes())
    row = json.loads((out / "phase-a" / f"{order:04d}" / f"unit-{k:02d}.json").read_bytes())
    if sha(canonical_json(job).encode()) != row["job_sha256"]:
        raise ValueError(f"job {order:04d}-{k:02d} differs from its phase-A row")
    return job


def planned_paths(nodes) -> str:
    """Scheduler routing: a fenceable unit is always fenced; the rest get one whole pass."""
    return "fenced" if cu.fenceable(nodes) else "whole"


def phase_a_request(job, label: str) -> tuple[str, float, set[str]]:
    """The phase-A request of one job as sent to the engine: ``(request, deadline_s, allowed astral chars)``."""
    analysis = cu.fence_analysis(job["nodes"], ENCODING)
    if (planned_paths(job["nodes"]) == "fenced") != (len(analysis["fences"]) > 0):
        raise ValueError(f"ratchet: routing disagrees with fence analysis for {label}")
    allowed = astral(job["source"]["text"])
    request, deadline = rq.fenced_request(job, analysis, t_pass=rq.T_PASS, utf8=bool(allowed))
    return request, deadline, allowed


def run_phase_a(engine, jobs, out: Path, *, shard, resume=True):
    if engine.parser != DEAD_PARSER:
        raise Stop(f"phase A needs the dead parser host, engine parser is {engine.parser}")
    for order, k, job in jobs:
        if not in_shard(order, shard):
            continue
        path = out / "phase-a" / f"{order:04d}" / f"unit-{k:02d}.json"
        if path.exists() and resume:
            continue
        job_path = out / "jobs" / f"{order:04d}-{k:02d}.json"
        if job_path.exists():
            if json.loads(job_path.read_bytes()) != job:
                raise ValueError(f"job {order:04d}-{k:02d} differs from the stored job")
        else:
            _write_once(job_path, job)
        request, deadline, allowed = phase_a_request(job, f"{order:04d}-{k:02d}")
        engine.reset_ids()
        started = time.monotonic()
        try:
            response = engine.http(request, f"A-{order:04d}-{k:02d}", timeout=deadline + 5, allowed=allowed)
        except Stop:  # SIGTERM / gate stops are never recorded as a unit failure
            raise
        except Exception as exc:  # noqa: BLE001 -- transport failure is recorded, never retried silently
            if _engine_dead(engine):
                raise Stop(f"engine down during A-{order:04d}-{k:02d}: {exc}") from exc
            response = {"status": "error", "message": f"transport: {exc}"[:500], "payload": {"passes": []}}
        _write_once(
            path,
            {
                "job_sha256": sha(canonical_json(job).encode()),
                "request_sha256": hashlib.sha256(request.encode()).hexdigest(),
                "deadline_s": deadline,
                "elapsed_s": round(time.monotonic() - started, 3),
                "response": response,
            },
        )


def _remembering(last: dict, call):
    """Wrap a sender so the raw engine answer survives ``capture_warm_syntax``'s generic ValueError."""

    def sender(code):
        last["response"] = None
        last["response"] = result = call(code)
        return result

    return sender


def parser_probe(engine, live_parser: str) -> dict:
    """One trivial parse through ``capture_warm_syntax`` on the live engine; raises ParserOutage unless it
    returns a valid stored-parse capture of the probe text, parsed via ``live_parser``."""
    from memory_kernel.construction.fcg.native_warm_source_units import capture_warm_syntax

    last: dict = {}
    started = time.monotonic()
    try:
        capture = capture_warm_syntax(
            _remembering(last, lambda code: engine.http(code, "parser-probe", timeout=8)),
            PROBE_TEXT,
            grammar_id=rq.GRAMMAR,
        )
    except Stop:
        raise
    except Exception as exc:  # noqa: BLE001 -- every probe failure is an outage
        answer = str(last.get("response"))[:300]
        raise ParserOutage(f"parser outage: probe failed ({type(exc).__name__}: {exc}; answer {answer})") from exc
    if capture["parser"][0] != live_parser:
        raise ParserOutage(f"parser outage: probe parsed via {capture['parser'][0]}, want {live_parser}")
    return {
        "seconds": round(time.monotonic() - started, 3),
        "parser": capture["parser"],
        "trees": len(capture["trees"]),
        "syntax_sha256": capture["content_sha256"],
    }


def parser_failure(engine, exc: BaseException, last: dict) -> str | None:
    """Classify a failed syntax call: None = specific to this text (record it as a lost segment); otherwise
    the parser-outage reason (stop; nothing is written, a resume retries).

    The live probe decides: a dead engine is an engine stop; an engine answer that names the parser transport,
    a Python transport error, or a failing probe is an outage; a failure with a healthy probe is the text's."""
    if _engine_dead(engine):
        raise Stop(f"engine down during a syntax call: {type(exc).__name__}: {exc}")
    answer = last.get("response")
    message = str(answer.get("message", "")) if isinstance(answer, dict) else ""
    if any(marker in message.upper() for marker in PARSER_TRANSPORT_MARKERS):
        return f"parser transport: {message[:300]}"
    try:
        parser_probe(engine, engine.parser)
    except ParserOutage as outage:
        return f"{outage} (after {type(exc).__name__}: {exc})"[:600]
    if isinstance(exc, OSError):  # ConnectionError / TimeoutError with a live engine: transient, retry
        return f"transport {type(exc).__name__}: {exc}"[:600]
    return None


def redo_pass(engine, job, analysis, pass_id):
    """Stand-in fallback over one pass subtree (a port of the reference harness's redo_pass onto src helpers)."""
    engine.reset_ids()
    from memory_kernel.construction.fcg.native_warm_source_units import capture_warm_syntax, warm_source_units
    from memory_kernel.construction.memory_warm_message import warm_child_source
    from memory_kernel.pipeline.stages._capture_record import round_capture_record, round_native_request

    p = next(x for x in analysis["passes"] if x["id"] == pass_id)
    spans = [tuple(f["span"]) for f in analysis["fences"] if f["collapsed_in"] == pass_id]
    rows, frames = [], []
    for number, segment in enumerate(si.pass_segments(p["nodes"], job["nodes"], job["source"]["text"], spans)):
        allowed = astral(segment["text"])
        label = f"B-{job['origin']['source_unit_id'][-8:]}-{pass_id}-s{number:02d}"
        started = time.monotonic()
        row = {"text": segment["text"], "charmap": segment["charmap"]}
        try:
            last: dict = {}
            try:
                syntax = capture_warm_syntax(
                    _remembering(
                        last,
                        lambda code, label=label, allowed=allowed: engine.http(
                            code, label + "-syntax", timeout=8, allowed=allowed
                        ),
                    ),
                    segment["text"],
                    grammar_id=rq.GRAMMAR,
                )
            except Stop:
                raise
            except Exception as exc:  # noqa: BLE001 -- an outage stops; a text-specific failure is recorded
                reason = parser_failure(engine, exc, last)
                if reason is not None:
                    raise ParserOutage(
                        reason if reason.startswith("parser outage") else f"parser outage: {reason}"
                    ) from exc
                raise
            row["syntax_sha256"], row["syntax_s"] = syntax["content_sha256"], round(time.monotonic() - started, 3)
            source = warm_child_source(
                job["source"], segment["text"], f"{job['source']['message_id']}#ph-{pass_id}-{number:02d}"
            )
            jobs = []
            for sub in warm_source_units(source, syntax):
                sub_job = {
                    "source": warm_child_source(source, sub["text"], sub["unit_id"]),
                    "span": [sub["start"], sub["end"]],
                    "nodes": sub["nodes"],
                    "parser": sub["parser"],
                }
                request = round_native_request(sub_job, SERVING_S, utf8=bool(allowed))
                sent = time.monotonic()
                response = engine.http(request, label, timeout=SERVING_S + 3, allowed=allowed)
                message = str(response.get("message", "")) if isinstance(response, dict) else ""
                if any(marker in message.upper() for marker in PARSER_TRANSPORT_MARKERS):
                    raise ParserOutage(f"parser outage: native unit request: {message[:300]}")
                jobs.append(
                    {
                        "elapsed_s": round(time.monotonic() - sent, 3),
                        "record": round_capture_record(sub_job, response, request),
                    }
                )
            row["jobs"] = [
                {
                    "request_sha256": j["record"]["request_sha256"],
                    "elapsed_s": j["elapsed_s"],
                    "status": j["record"]["capture_status"],
                    "frames": (j["record"]["response"].get("payload") or {}).get("frames"),
                }
                for j in jobs
            ]
            row["status"] = "cutoff" if any(j["record"]["capture_status"] == "cutoff" for j in jobs) else "ok"
            frames += si.map_segment_frames(segment, jobs, job["nodes"], pass_id=pass_id, number=number)
        except Stop:  # SIGTERM / gate stops are never recorded as a unit failure
            raise
        except Exception as exc:  # noqa: BLE001 -- a failed segment is a lost segment, recorded
            row["status"], row["error"] = "error", f"{type(exc).__name__}: {exc}"[:500]
        row["seconds"] = round(time.monotonic() - started, 3)
        rows.append(row)
    return {
        "status": "ok" if rows and all(r["status"] == "ok" for r in rows) else "partial",
        "seconds": round(sum(r["seconds"] for r in rows), 3),
        "segments": rows,
        "frames": frames,
    }


def run_phase_b(engine, out: Path, *, shard, live_parser):
    """Rows are built in full before the write-once; a ParserOutage (a Stop) raised by ``redo_pass`` leaves
    the unit's phase-B file absent, so a resume redoes it."""
    if engine.parser != live_parser:
        raise Stop(f"phase B needs the live parser host, engine parser is {engine.parser}")
    for path, row in _phase_a_rows(out):
        order, k = _order_k(path)
        if not in_shard(order, shard):
            continue
        target = out / "phase-b" / path.parent.name / path.name
        if target.exists():
            continue
        job = load_job(out, order, k)
        analysis = cu.fence_analysis(job["nodes"], ENCODING)
        rows, _, _ = cu.pass_rows(analysis, row["response"])
        redo = {
            r["id"]: redo_pass(engine, job, analysis, r["id"]) for r in rows if r["status"] in ("time-out", "not-run")
        }
        if redo and _engine_dead(engine):
            raise Stop(f"engine down during phase B of {order:04d}-{k:02d}")
        _write_once(target, redo)


def _new_stage(out: Path) -> Path:
    n = 0
    while True:
        stage = out / f".merge-{n}"
        try:
            stage.mkdir(parents=True, exist_ok=False)
            return stage
        except FileExistsError:
            n += 1


def _same_tree(a: Path, b: Path) -> bool:
    names = sorted(p.name for p in a.iterdir())
    return names == sorted(p.name for p in b.iterdir()) and all(
        (a / n).read_bytes() == (b / n).read_bytes() for n in names
    )


def _promote(stage: Path, out: Path) -> None:
    """Each staged ``NNNN`` dir appears under ``out`` in one ``os.rename`` onto a not-yet-existing name. A dir
    already promoted by an earlier, interrupted merge must be byte-identical (merge is deterministic)."""
    for staged in sorted(stage.iterdir()):
        final = out / staged.name
        if final.exists():
            if not _same_tree(staged, final):
                raise ValueError(f"merged {staged.name} differs from the staged merge; refusing to overwrite")
            continue
        os.rename(staged, final)


def merge(out: Path) -> dict:
    """``NNNN/unit-KK.json`` (sealed fenced-unit record) + ``NNNN/complete.json`` for every source.

    Builds everything in ``out/.merge-N/`` first; only a complete build is promoted (``_promote``), so an
    exception partway leaves no partial message dir and a second merge succeeds. A stage that failed is left
    as evidence. Reads jobs only from ``jobs/`` and rows only from ``phase-a`` / ``phase-b``; byte-identical
    for any shard split or run order of the same rows. Permissive about missing phase-B files (``merge_gate``
    is not)."""
    stage = _new_stage(out)
    counts = _merge_into(out, stage)
    _promote(stage, out)
    shutil.rmtree(stage)
    return counts


def _merge_into(out: Path, stage: Path) -> dict:
    sources = json.loads((out / "sources.json").read_bytes())
    by_order = {s["order"]: s for s in sources}
    counts = {"units": 0, "framed": 0, "empty": 0, "cutoff": 0}
    per_message: dict[int, list[str]] = {order: [] for order in by_order}
    for path, row in _phase_a_rows(out):
        order, k = _order_k(path)
        if order not in by_order:
            raise ValueError(f"phase-A row {order:04d}-{k:02d} has no source")
        job = load_job(out, order, k)
        fb_path = out / "phase-b" / path.parent.name / path.name
        fallbacks = json.loads(fb_path.read_bytes()) if fb_path.exists() else {}
        analysis = cu.fence_analysis(job["nodes"], ENCODING)
        phase_a = {**row["response"], "request_sha256s": [row["request_sha256"]]}
        record = cu.fenced_unit_record(
            job, analysis, phase_a, fallbacks, t_pass=rq.T_PASS, deadline_s=row["deadline_s"]
        )
        _write_once(stage / path.parent.name / path.name, record)
        per_message[order].append(sha(canonical_json(record).encode()))
        counts["units"] += 1
        counts[record["capture_status"]] += 1
    for order, shas in sorted(per_message.items()):
        _write_once(
            stage / f"{order:04d}" / "complete.json",
            seal(
                {
                    "schema_version": MESSAGE_SCHEMA,
                    "source_sha256": by_order[order].get("content_sha256"),
                    "units": len(shas),
                    "unit_sha256": shas,
                }
            ),
        )
    return counts


def current_walker_sha256() -> str:
    from memory_kernel.construction.fcg.native_analysis import NATIVE_ANALYSIS_WALKER

    return hashlib.sha256(NATIVE_ANALYSIS_WALKER.encode()).hexdigest()


def _display_path(path: Path, root: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _resolve(text: str, root: Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else root / path


def manifest_body(sources, syntax_from: str, *, live_parser: str, site: Site) -> dict:
    """Every pin that is recomputed from the working tree (``check_pins``)."""
    from memory_kernel.pipeline.stages._resident_stock_capture_runner import CONFIG

    return {
        "schema_version": HYBRID_CAPTURE_SCHEMA,
        "sources_sha256": sha(canonical_json(sources).encode()),
        "source_count": len(sources),
        "syntax_from": syntax_from,
        "t_pass": rq.T_PASS,
        "deadline_cap": int(rq.DEADLINE_CAP),
        "encoding": ENCODING,
        "binary_sha256": sha_file(site.engine.binary),
        "image_sha256": sha_file(site.engine.image),
        "config_sha256": sha(canonical_json(CONFIG).encode()),
        "walker_sha256": current_walker_sha256(),
        "code_pins": {p: sha(code_path(p, code_root=site.code_root).read_bytes()) for p in site.pin_files},
        "parsers": {"phase_a": DEAD_PARSER, "phase_b": live_parser},
        "extraction_model": extraction_model_record(),
    }


def _snapshot_roots(site: Site) -> dict:
    """The package dirs a snapshot pin list is checked against (acceptance adds the one it was copied from)."""
    return {"this_tree": site.code_root if site.code_root is not None else package_dir()}


def _file_sha(path: Path) -> str | None:
    return sha(path.read_bytes()) if path.is_file() else None


def load_snapshot_pins(raw: bytes) -> dict:
    pins = json.loads(raw)
    files = pins.get("files") if isinstance(pins, dict) else None
    if (
        not isinstance(files, dict)
        or not files
        or not all(
            isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for k, v in files.items()
        )
        or any(Path(k).is_absolute() or ".." in Path(k).parts for k in files)
    ):
        raise ValueError("snapshot pins need a non-empty {relative path: sha256} map under 'files'")
    if pins.get("schema", SNAPSHOT_SCHEMA) != SNAPSHOT_SCHEMA:
        raise ValueError(f"snapshot pins schema {pins.get('schema')!r} != {SNAPSHOT_SCHEMA}")
    return files


def _snapshot_file_problems(files: dict, roots: dict) -> list[dict]:
    """Each pinned file against the census of each package dir in ``roots`` (``tree_census``: the same keys and sha a
    computed pin list has); a file absent from a census is ``got: None``."""
    problems = []
    censuses = {label: tree_census(Path(root)) for label, root in roots.items()}
    for rel, want in sorted(files.items()):
        for label, current in censuses.items():
            got = current.get(rel)
            if got != want:
                problems.append({"file": rel, "tree": label, "want": want, "got": got})
    return problems


def snapshot_problems(out: Path, manifest: dict, roots: dict) -> list[dict]:
    """[] when the manifest declares no snapshot, or when the declared pins file is present, unchanged, and
    every pinned file hashes to its pin in every root. A declared-but-missing pins file is a problem."""
    declared = manifest.get("snapshot_pins")
    if declared is None:
        return []
    path = Path(out) / declared["file"]
    if not path.is_file():
        return [{"problem": "declared snapshot pins file is missing", "file": declared["file"]}]
    problems = []
    if sha(path.read_bytes()) != declared["sha256"]:
        problems.append({"problem": "snapshot pins file differs from its sealed sha256", "file": declared["file"]})
    return problems + _snapshot_file_problems(declared["files"], roots)


def _declared_problems(out: Path, manifest: dict, site: Site) -> list:
    problems = list(snapshot_problems(out, manifest, _snapshot_roots(site)))
    want = manifest.get("syntax_manifest_sha256")
    if want is not None and _file_sha(_resolve(manifest["syntax_from"], site.root) / SYNTAX_MANIFEST) != want:
        problems.append({"problem": "syntax manifest differs from its sealed sha256", "file": SYNTAX_MANIFEST})
    return problems


def _check_syntax_supply(syntax_dir: Path, sources) -> str | None:
    """A ``parse_capture syntax`` dir (it has a syntax manifest) must be complete for exactly these sources;
    the sha of its manifest is sealed. A stock capture dir (no syntax manifest) is accepted as before."""
    path = Path(syntax_dir) / SYNTAX_MANIFEST
    if not path.exists():
        return None
    raw = path.read_bytes()
    syntax_manifest = json.loads(raw)
    check_seal(syntax_manifest, SYNTAX_CAPTURE_SCHEMA)
    if syntax_manifest["sources_sha256"] != sha(canonical_json(sources).encode()):
        raise ValueError("the syntax capture was made for different sources")
    missing = syntax_missing(Path(syntax_dir), sources)
    if missing:
        raise ValueError(f"the syntax capture is incomplete: {len(missing)} parts missing, e.g. {missing[:5]}")
    return sha(raw)


def prepare(
    out: Path,
    sources_path: Path,
    syntax_dir: Path,
    *,
    live_parser: str,
    site: Site,
    snapshot_pins: Path | None = None,
) -> dict:
    if current_walker_sha256() != WALKER_SHA256:
        raise ValueError("native walker differs from the 5c0d905 pin")
    sources = json.loads(Path(sources_path).read_bytes())
    orders = [s["order"] for s in sources]
    if orders != sorted(set(orders)):
        raise ValueError("sources must be strictly ordered")
    body = manifest_body(sources, _display_path(syntax_dir, site.root), live_parser=live_parser, site=site)
    if problems := pin_problems(
        (site.engine.binary_sha256, site.engine.image_sha256),
        binary_sha256=body["binary_sha256"],
        image_sha256=body["image_sha256"],
    ):
        raise ValueError(
            "; ".join(problems) + " (`mk setup` fetches the pinned engine; --allow-unpinned-engine accepts another)"
        )
    syntax_manifest_sha = _check_syntax_supply(syntax_dir, sources)
    if syntax_manifest_sha is not None:
        body["syntax_manifest_sha256"] = syntax_manifest_sha
    pins_raw = None
    if snapshot_pins is not None:
        pins_raw = Path(snapshot_pins).read_bytes()
        files = load_snapshot_pins(pins_raw)
        problems = _snapshot_file_problems(files, _snapshot_roots(site))
        if problems:
            raise ValueError(f"snapshot pins differ from the tree now: {problems[:5]}")
        body["snapshot_pins"] = {"file": SNAPSHOT_FILE, "sha256": sha(pins_raw), "files": files}
    manifest = seal(body)
    out.mkdir(parents=True, exist_ok=False)
    write_sources(out, sources)
    if pins_raw is not None:
        _atomic_write_once(out / SNAPSHOT_FILE, pins_raw.decode())
    _write_once(out / "manifest.json", manifest)
    return manifest


def check_pins(out: Path, *, live_parser: str | None, site: Site) -> dict:
    """The sealed manifest, after recomputing every pin from the working tree and re-checking the declared
    sections (snapshot pins, syntax manifest). Raises ``ValueError("VOID: ...")`` on any difference, and when
    ``live_parser`` is given and differs from the sealed phase-B parser."""
    manifest = json.loads((out / "manifest.json").read_bytes())
    check_seal(manifest, HYBRID_CAPTURE_SCHEMA)
    if manifest["walker_sha256"] != WALKER_SHA256:
        raise ValueError("manifest walker pin differs from the 5c0d905 pin")
    sealed_parser = manifest["parsers"]["phase_b"]
    if live_parser is not None and live_parser != sealed_parser:
        raise ValueError(f"live parser {live_parser} differs from the manifest's {sealed_parser}")
    sources = json.loads((out / "sources.json").read_bytes())
    now = manifest_body(sources, manifest["syntax_from"], live_parser=sealed_parser, site=site)
    now["extraction_model"] = manifest.get(
        "extraction_model"
    )  # a record, not a pin: ``require_extraction_model`` reads it
    differs = sorted(k for k in now if now[k] != manifest.get(k))
    differs += sorted(k for k in manifest if k not in now and k not in (*DECLARED_SECTIONS, "content_sha256"))
    declared = _declared_problems(out, manifest, site)
    if differs or declared:
        raise ValueError(f"VOID: pin differs: {differs}; declared sections: {declared[:10]}")
    return manifest


def require_extraction_model(manifest: dict, *, what: str) -> None:
    """Refuse a resume under another extraction model than the one the capture recorded (a manifest with none was made
    before 1.4: ``en_core_web_sm``). The version alone differing is no refusal."""
    if problem := parser_change_problem(manifest.get("extraction_model"), what=what, flag=False):
        raise ValueError(problem)


def _readback(engine, expected: str, when: str):
    problems, row = engine.readback()
    if problems or not row or row.get("parser") != expected:
        raise Stop(f"{when} readback: parser {row and row.get('parser')} (want {expected}); problems {problems}")
    return row


def run_shard(engine, jobs, out: Path, *, shard, live_parser, site: Site, log=lambda _m: None) -> dict:
    """launch -> arm(DEAD) -> readback DEAD -> phase A -> every shard row on disk -> set_parser(LIVE) ->
    readback LIVE -> live parser probe -> phase B -> idle CPU -> final readback -> ``close_attempt`` (always):
    a teardown, or for a leased engine (launch = identity re-check, arm = attach) a release."""
    mine = [(o, k) for o, k, _ in jobs if in_shard(o, shard)]
    report: dict[str, Any] = {"shard": list(shard), "jobs": len(mine), "status": "started"}
    try:
        report["engine"] = engine.launch()
        report["arm"] = engine.arm(DEAD_PARSER)
        _readback(engine, DEAD_PARSER, "phase-A")
        log(f"phase A: {len(mine)} jobs, parser {DEAD_PARSER}")
        run_phase_a(engine, jobs, out, shard=shard)
        missing = [
            f"{o:04d}-{k:02d}" for o, k in mine if not (out / "phase-a" / f"{o:04d}" / f"unit-{k:02d}.json").exists()
        ]
        if missing:
            raise Stop(f"phase A rows missing before the parser switch: {missing[:10]}")
        engine.set_parser(live_parser)
        _readback(engine, live_parser, "phase-B")
        with parser_lock(site.parser_lock):
            report["parser_probe"] = parser_probe(engine, live_parser)  # refuses phase B on a parser outage
        log(f"phase B: parser {live_parser}, probe {report['parser_probe']}")
        with parser_lock(site.parser_lock):
            run_phase_b(engine, out, shard=shard, live_parser=live_parser)
        report["idle_cpu_s_over_3s"] = idle = engine.idle_cpu(3.0)
        if idle > IDLE_CPU_MAX_S:
            raise Stop(f"engine not idle after the phases: {idle:.2f}s cpu over 3s")
        _readback(engine, live_parser, "final")
        report["status"] = "complete"
    except Stop as exc:
        report["status"] = f"stopped: {exc}"[:2000]
        log(f"STOP {exc}")
    except BaseException as exc:  # noqa: BLE001 -- recorded; teardown below always runs
        import traceback

        report["status"] = f"crashed: {type(exc).__name__}: {exc}"[:2000]
        log(traceback.format_exc())
    finally:
        end = _close_attempt(engine, report)
        report["phase_a_rows"] = sum(
            1 for o, k in mine if (out / "phase-a" / f"{o:04d}" / f"unit-{k:02d}.json").exists()
        )
        report["phase_b_rows"] = sum(
            1 for o, k in mine if (out / "phase-b" / f"{o:04d}" / f"unit-{k:02d}.json").exists()
        )
        log(f"{end} {report[end]}")
    return report


def _close_attempt(engine, report: dict) -> str:
    """End an attempt through ``engine.close_attempt()``: its receipt lands under ``teardown`` (cold) or ``release``
    (leased), an exception's under the engine's own key; a receipt that is not clean (``end_problems``) stops the
    attempt. Returns the key."""
    end = "release" if isinstance(engine, AttachedEngine) else "teardown"
    try:
        end, report[end] = engine.close_attempt()
    except BaseException as exc:  # noqa: BLE001 -- a failed teardown / release is a failed attempt
        report[end] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    dirty = end_problems(report)
    if dirty:
        report["status"] = f"stopped: {end} dirty {dirty}; was {report['status']}"[:2000]
    return end


def end_problems(report: dict) -> list[str]:
    """How an attempt ended, checked: its teardown receipt (``teardown_problems``) or, for a leased engine, its release
    receipt (``release_problems``). A report with neither never counts."""
    if "teardown" not in report and "release" not in report:
        return ["no teardown or release receipt"]
    problems = []
    if "teardown" in report:
        problems += teardown_problems(report["teardown"])
    if "release" in report:
        problems += release_problems(report["release"])
    return problems


def teardown_problems(teardown) -> list[str]:
    """Empty only when the receipt proves the port is free (127.0.0.1 and ::1) and no own FCG process is left."""
    problems = []
    if not isinstance(teardown, dict) or "error" in teardown:
        problems.append(f"teardown failed: {teardown}")
        return problems
    if teardown.get("port_listening_after") is not False:
        problems.append(f"port listening after teardown: {teardown.get('port_listening_after')}")
    if teardown.get("own_fcg_after") != []:
        problems.append(f"own FCG processes after teardown: {teardown.get('own_fcg_after')}")
    return problems


def _attempt(out: Path, tag: str) -> int:
    n = 0
    while (out / f"run-{tag}-a{n}.log").exists():
        n += 1
    return n


def _driven(out: Path, tag: str, work) -> dict:
    """One attempt N of a driver: pid file, log, SIGTERM -> Stop, ``work(n, log)`` -> write-once report."""
    n = _attempt(out, tag)
    with (out / f"driver-{tag}-a{n}.pid").open("x") as handle:
        handle.write(str(os.getpid()))
    logfile = (out / f"run-{tag}-a{n}.log").open("x")

    def log(message):
        logfile.write(f"{time.strftime('%H:%M:%S')} {message}\n")
        logfile.flush()

    def on_term(_s, _f):
        raise Stop("SIGTERM")

    previous = signal.signal(signal.SIGTERM, on_term)
    try:
        report = work(n, log)
        report["attempt"] = n
        _atomic_write_once(out / f"report-{tag}-a{n}.json", json.dumps(report, indent=1, default=str))
        log("DONE")
    finally:
        signal.signal(signal.SIGTERM, previous)
        logfile.close()
    return report


def _with_parser(site: Site, attempt_dir: Path, log, attempt, base: dict | None = None) -> dict:
    """``attempt()`` (one engine attempt; it returns its report and never raises) with the managed parser up around it
    when the site has one. The parser starts here, inside the attempt, so a complete capture (which runs no
    attempt) starts none. Its launch and teardown receipts land in the report under ``parser_service``; a parser that
    cannot start, or that does not stop clean, makes the attempt ``stopped``."""
    if site.parser is None:
        return attempt()
    service = ParserService(site.parser, attempt_dir)
    receipt: dict[str, Any] = {"url": service.url}
    report: dict[str, Any] = dict(base or {})
    try:
        if service.url != site.live_parser:
            raise Stop(f"the managed parser {service.url} is not the site's live parser {site.live_parser}")
        receipt["launch"] = launch = service.launch()
        log(f"parser: {service.url} ready in {launch['ready_s']}s (pid {launch['pid']}, rss {launch['rss_kib']} KiB)")
        report = attempt()
    except Stop as exc:
        report["status"] = f"stopped: parser: {exc}"[:2000]
        log(f"STOP parser {exc}")
    except BaseException as exc:  # noqa: BLE001 -- recorded; the parser teardown below always runs
        import traceback

        report["status"] = f"crashed: parser: {type(exc).__name__}: {exc}"[:2000]
        log(traceback.format_exc())
    finally:
        try:
            receipt["teardown"] = teardown = service.teardown()
        except BaseException as exc:  # noqa: BLE001 -- a failed teardown is a failed attempt
            receipt["teardown"] = teardown = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        report["parser_service"] = receipt
        dirty = parser_teardown_problems(teardown)
        if dirty:
            report["status"] = f"stopped: parser teardown dirty {dirty}; was {report.get('status')}"[:2000]
        log(f"parser teardown {teardown}")
    return report


def _process_gate(port: int) -> str | None:
    strays = own_fcg_processes(port=port)
    if strays or port_listening(port):
        return f"stopped: process gate (strays {strays}, port {port})"
    return None


def _slot_gate(port: int, grant: Grant) -> str | None:
    """The leased attempt's ``_process_gate``: the own FCG processes on the port are exactly the leased engine and the
    own parser processes exactly the leased parser, each the granted process (pid, start_ticks, exe), and the port is
    listening."""
    engines = sorted(p["pid"] for p in own_fcg_processes(port=port))
    parsers = sorted(p["pid"] for p in own_parser_processes())
    granted = (grant.engine, grant.parser)
    same = all(same_process(proc_identity(g["pid"]), g) for g in granted)
    listening = port_listening(port)
    if engines != [grant.engine["pid"]] or parsers != [grant.parser["pid"]] or not same or not listening:
        return (
            f"stopped: slot gate (own FCG {engines}, own parser {parsers}, leased {[g['pid'] for g in granted]}, "
            f"granted identities {'match' if same else 'differ'}, port {port} listening {listening})"
        )
    return None


def _leased(site: Site, grant: Grant | None) -> None:
    """A leased attempt runs on the slot's engine and parser: refuse a cold grant, or a site that starts a parser."""
    if grant is None:
        return
    if grant.kind != "warm" or grant.engine is None or grant.parser is None:
        raise ValueError(f"only a warm grant leases an engine, got {grant.kind!r}")
    if site.parser is not None:
        raise ValueError("a leased attempt uses the slot's parser: site.parser must be None")


def _recheck(report: dict, recheck, log) -> None:
    """Pins re-checked AFTER the attempt: any change during it voids the report (never ``complete``)."""
    try:
        recheck()
    except ValueError as exc:
        report["status"] = f"void: pins changed during the run: {exc}; was {report['status']}"[:2000]
        log(report["status"])


def run(out: Path, shard: tuple[int, int], *, site: Site, grant: Grant | None = None) -> dict:
    _leased(site, grant)
    live_parser, port = site.live_parser, site.engine.port
    manifest = check_pins(out, live_parser=live_parser, site=site)
    require_extraction_model(manifest, what="capture")  # a resume: the shards already on disk were parsed with it
    sources = json.loads((out / "sources.json").read_bytes())
    jobs = hybrid_jobs(sources, _resolve(manifest["syntax_from"], site.root))
    tag = shard_tag(shard)

    def work(n, log):
        gate = _process_gate(port) if grant is None else _slot_gate(port, grant)
        if gate:
            report: dict[str, Any] = {"shard": list(shard), "status": gate}
            log(gate)
        else:
            attempt_dir = out / f"engine-{tag}-a{n}"
            engine = (
                Engine(site.engine, attempt_dir) if grant is None else AttachedEngine(site.engine, attempt_dir, grant)
            )
            report = _with_parser(
                site,
                attempt_dir,
                log,
                lambda: run_shard(engine, jobs, out, shard=shard, live_parser=live_parser, site=site, log=log),
                base={"shard": list(shard)},
            )
        report["manifest_sha256"] = manifest["content_sha256"]
        _recheck(report, lambda: check_pins(out, live_parser=live_parser, site=site), log)
        return report

    return _driven(out, tag, work)


def syntax_parts(sources):
    """``(order, index, text)`` for every non-blank part, indexed exactly as the stock capture and
    ``hybrid_jobs`` index ``syntax-XX.json`` (blank parts keep their index and are skipped)."""
    from memory_kernel.construction.memory_warm_message import warm_message_parts

    for source in sources:
        for index, part in enumerate(warm_message_parts(source)):
            if part["source"]["text"].strip():
                yield source["order"], index, part["source"]["text"]


def _syntax_path(out: Path, order: int, index: int) -> Path:
    return out / f"{order:04d}" / f"syntax-{index:02d}.json"


def _syntax_seconds_path(out: Path, order: int, index: int) -> Path:
    return out / "syntax-seconds" / f"{order:04d}-{index:02d}.json"


def syntax_missing(out: Path, sources) -> list[str]:
    return [f"{o:04d}-{i:02d}" for o, i, _ in syntax_parts(sources) if not _syntax_path(out, o, i).exists()]


def syntax_manifest_body(sources, *, live_parser: str, site: Site) -> dict:
    from memory_kernel.pipeline.stages._resident_stock_capture_runner import CONFIG

    return {
        "schema_version": SYNTAX_CAPTURE_SCHEMA,
        "sources_sha256": sha(canonical_json(sources).encode()),
        "source_count": len(sources),
        "parts": sum(1 for _ in syntax_parts(sources)),
        "grammar": rq.GRAMMAR,
        "live_parser": live_parser,
        "extraction_model": extraction_model_record(),
        "binary_sha256": sha_file(site.engine.binary),
        "image_sha256": sha_file(site.engine.image),
        "config_sha256": sha(canonical_json(CONFIG).encode()),
        "code_pins": {p: sha(code_path(p, code_root=site.code_root).read_bytes()) for p in site.pin_files},
    }


def prepare_syntax(out: Path, sources, *, live_parser: str, site: Site) -> dict:
    """Create ``out`` (sources + sealed syntax manifest), or, on resume, refuse unless the sources, the live
    parser and every pin are unchanged."""
    body = syntax_manifest_body(sources, live_parser=live_parser, site=site)
    path = out / SYNTAX_MANIFEST
    if path.exists():
        manifest = json.loads(path.read_bytes())
        check_seal(manifest, SYNTAX_CAPTURE_SCHEMA)
        if json.loads((out / "sources.json").read_bytes()) != sources:
            raise ValueError("syntax resume: sources differ from the stored sources")
        require_extraction_model(manifest, what="syntax capture")
        differs = sorted(
            k
            for k in set(body) | set(manifest)
            if k not in ("content_sha256", "extraction_model") and body.get(k) != manifest.get(k)
        )
        if differs:
            raise ValueError(f"syntax resume: pin differs: {differs}")
        return manifest
    if problems := pin_problems(
        (site.engine.binary_sha256, site.engine.image_sha256),
        binary_sha256=body["binary_sha256"],
        image_sha256=body["image_sha256"],
    ):
        raise ValueError(
            "; ".join(problems) + " (`mk setup` fetches the pinned engine; --allow-unpinned-engine accepts another)"
        )
    manifest = seal(body)
    out.mkdir(parents=True, exist_ok=False)
    write_sources(out, sources)
    _write_once(path, manifest)
    return manifest


def capture_syntax(engine, sources, out: Path, *, live_parser: str, log=lambda _m: None) -> dict:
    """Write-once ``NNNN/syntax-XX.json`` + ``syntax-seconds/NNNN-XX.json`` for every missing part. Any
    failure stops with nothing written for that part (a hybrid prepare needs every part): an outage as
    ``parser outage``, a text-specific failure by name."""
    from memory_kernel.construction.fcg.native_warm_source_units import capture_warm_syntax

    written = 0
    for order, index, text in syntax_parts(sources):
        path = _syntax_path(out, order, index)
        if path.exists():
            continue
        allowed = astral(text)
        last: dict = {}
        started = time.monotonic()
        try:
            capture = capture_warm_syntax(
                _remembering(
                    last,
                    lambda code, label=f"syntax-{order}-{index}", allowed=allowed: engine.http(
                        code, label, timeout=8, allowed=allowed
                    ),
                ),
                text,
                grammar_id=rq.GRAMMAR,
            )
        except Stop:
            raise
        except Exception as exc:  # noqa: BLE001 -- classified; never written
            reason = parser_failure(engine, exc, last)
            if reason is not None:
                raise ParserOutage(
                    reason if reason.startswith("parser outage") else f"parser outage: {reason}"
                ) from exc
            raise Stop(f"syntax failed for {order:04d}-{index:02d}: {type(exc).__name__}: {exc}"[:600]) from exc
        seconds = round(time.monotonic() - started, 3)
        if capture["parser"][0] != live_parser:
            raise Stop(f"syntax {order:04d}-{index:02d} parsed via {capture['parser'][0]}, want {live_parser}")
        _write_once(path, capture)
        _write_once(
            _syntax_seconds_path(out, order, index), {"seconds": seconds, "syntax_sha256": capture["content_sha256"]}
        )
        written += 1
        if written % 100 == 0:
            log(f"syntax: {written} parts written")
    return {"written": written}


def run_syntax_engine(engine, sources, out: Path, *, live_parser: str, site: Site, log=lambda _m: None) -> dict:
    """launch -> arm(LIVE) -> readback LIVE -> parser probe -> stored parses -> final readback -> ``close_attempt``
    (a teardown, or a release for a leased engine)."""
    report: dict[str, Any] = {"parts": sum(1 for _ in syntax_parts(sources)), "status": "started"}
    started = time.monotonic()
    try:
        report["engine"] = engine.launch()
        report["arm"] = engine.arm(live_parser)
        _readback(engine, live_parser, "syntax")
        with parser_lock(site.parser_lock):
            report["parser_probe"] = parser_probe(engine, live_parser)
        log(f"syntax: {report['parts']} parts, parser {live_parser}, probe {report['parser_probe']}")
        with parser_lock(site.parser_lock):
            report |= capture_syntax(engine, sources, out, live_parser=live_parser, log=log)
        _readback(engine, live_parser, "final")
        missing = syntax_missing(out, sources)
        report["status"] = "complete" if not missing else f"stopped: {len(missing)} parts missing"
    except Stop as exc:
        report["status"] = f"stopped: {exc}"[:2000]
        log(f"STOP {exc}")
    except BaseException as exc:  # noqa: BLE001 -- recorded; teardown below always runs
        import traceback

        report["status"] = f"crashed: {type(exc).__name__}: {exc}"[:2000]
        log(traceback.format_exc())
    finally:
        end = _close_attempt(engine, report)
        report["seconds"] = round(time.monotonic() - started, 3)
        report["syntax_files"] = report["parts"] - len(syntax_missing(out, sources))
        log(f"{end} {report[end]}")
    return report


def syntax(out: Path, sources_path: Path, *, site: Site, grant: Grant | None = None) -> dict:
    """The ``syntax`` step: prepare-or-resume, then one attempt on the site's engine, or on the leased engine with
    ``grant`` (report-syntax-aN)."""
    _leased(site, grant)
    live_parser, port = site.live_parser, site.engine.port
    sources = json.loads(Path(sources_path).read_bytes())
    manifest = prepare_syntax(out, sources, live_parser=live_parser, site=site)

    def work(n, log):
        gate = _process_gate(port) if grant is None else _slot_gate(port, grant)
        if gate:
            report: dict[str, Any] = {"status": gate}
            log(gate)
        else:
            attempt_dir = out / f"engine-syntax-a{n}"
            engine = (
                Engine(site.engine, attempt_dir) if grant is None else AttachedEngine(site.engine, attempt_dir, grant)
            )
            report = _with_parser(
                site,
                attempt_dir,
                log,
                lambda: run_syntax_engine(engine, sources, out, live_parser=live_parser, site=site, log=log),
            )
        report["manifest_sha256"] = manifest["content_sha256"]
        _recheck(report, lambda: prepare_syntax(out, sources, live_parser=live_parser, site=site), log)
        return report

    return _driven(out, "syntax", work)


def syntax_seconds(syntax_dir: Path) -> dict:
    """Syntax seconds behind a stored-parse dir, for the projection. A ``syntax`` dir: its own per-part rows.
    A stock capture dir: its ``batch-*.jsonl`` http receipts labelled ``syntax-*`` (predecessors included);
    when receipts cover fewer parts than files, the total is the measured mean x parts (``basis``)."""
    syntax_dir = Path(syntax_dir)
    parts = len(list(syntax_dir.glob("[0-9][0-9][0-9][0-9]*/syntax-[0-9][0-9].json")))
    if (syntax_dir / SYNTAX_MANIFEST).exists():
        source = "parse_capture syntax receipts"
        seconds = [
            json.loads(p.read_bytes())["seconds"] for p in sorted((syntax_dir / "syntax-seconds").glob("*.json"))
        ]
    else:
        source = "stock capture batch receipts"
        seconds = []
        for path in sorted(syntax_dir.rglob("batch-*.jsonl")):
            for line in path.read_text().splitlines():
                if '"syntax-' not in line:
                    continue
                row = json.loads(line)
                if row.get("event") == "http" and str(row.get("label", "")).startswith("syntax-"):
                    seconds.append(float(row["elapsed_seconds"]))
    result = {"source": source, "parts": parts, "measured": len(seconds), "seconds_total": None}
    if parts and seconds:
        per_part = sum(seconds) / len(seconds)
        exact = len(seconds) == parts
        result |= {
            "seconds_per_part": per_part,
            "seconds_total": sum(seconds) if exact else per_part * parts,
            "basis": "exact" if exact else "mean x parts",
        }
    return result


def latest_reports(out: Path) -> dict:
    latest: dict[str, tuple[int, dict]] = {}
    for path in out.glob("report-*-of-*-a*.json"):
        m = re.fullmatch(r"report-(\d+-of-\d+)-a(\d+)\.json", path.name)
        if m and int(m[2]) >= latest.get(m[1], (-1, None))[0]:
            latest[m[1]] = (int(m[2]), json.loads(path.read_bytes()))
    return {tag: value for tag, (_, value) in sorted(latest.items())}


def _all_reports(out: Path) -> list[dict]:
    rows = []
    for path in sorted(out.glob("report-*-of-*-a*.json")):
        m = re.fullmatch(r"report-(\d+)-of-(\d+)-a(\d+)\.json", path.name)
        if m:
            report = json.loads(path.read_bytes())
            rows.append({"file": path.name, "i": int(m[1]), "k": int(m[2]), "attempt": int(m[3]), "report": report})
    return rows


def merge_gate(out: Path, site: Site) -> dict:
    """``{"problems", "used", "ignored"}``; the merge runs only on no problems.

    Only ``complete`` shard reports count, and only with a clean teardown receipt or, for a leased attempt, a clean
    release receipt (``end_problems``); a report with neither receipt never counts. They must all share ONE shard
    count k and cover every i in 0..k-1 exactly once.
    Non-complete reports of any k are listed under ``ignored`` and never block a later full run. Then every
    job must have its ``jobs/`` file, phase-A row and phase-B file, and no extra phase-A row may exist."""
    manifest = check_pins(out, live_parser=None, site=site)
    problems, used, ignored = [], [], []
    for row in _all_reports(out):
        status_ = row["report"].get("status")
        dirty = end_problems(row["report"])
        entry = {"file": row["file"], "status": status_}
        if status_ == "complete" and not dirty:
            used.append({**entry, "i": row["i"], "k": row["k"]})
        else:
            key = "release_problems" if "release" in row["report"] else "teardown_problems"
            ignored.append({**entry, key: dirty} if dirty else entry)
    ks = sorted({u["k"] for u in used})
    if len(ks) != 1:
        problems.append(f"complete shard reports use shard counts {ks} (need exactly one)")
    else:
        (k,) = ks
        seen = [u["i"] for u in used]
        if sorted(seen) != list(range(k)):
            problems.append(f"complete shard reports {sorted(seen)} do not cover 0..{k - 1} exactly once")
    sources = json.loads((out / "sources.json").read_bytes())
    expected = {(o, k) for o, k, _ in hybrid_jobs(sources, _resolve(manifest["syntax_from"], site.root))}
    rows = {_order_k(p) for p, _ in _phase_a_rows(out)}
    if rows != expected:
        problems.append(f"phase-A rows differ from jobs: missing {len(expected - rows)}, extra {len(rows - expected)}")
    for o, k in sorted(expected):
        if not (out / "jobs" / f"{o:04d}-{k:02d}.json").exists():
            problems.append(f"job {o:04d}-{k:02d} missing")
        if not (out / "phase-b" / f"{o:04d}" / f"unit-{k:02d}.json").exists():
            problems.append(f"phase-B {o:04d}-{k:02d} missing")
    return {"problems": problems, "used": used, "ignored": ignored}


def gated_merge(out: Path, site: Site) -> dict:
    """``merge_gate`` -> ``merge`` -> ``merge.json`` receipt (counts, used and ignored shard reports), written
    last: it is the commit point, and a merge without it can be re-run."""
    if (out / "merge.json").exists():
        raise SystemExit("merge refused: merge.json exists (the capture is already merged)")
    gate = merge_gate(out, site)
    if gate["problems"]:
        raise SystemExit(f"merge refused: {gate['problems'][:20]}")
    counts = merge(out)
    receipt = {"counts": counts, "used_reports": gate["used"], "ignored_reports": gate["ignored"]}
    _atomic_write_once(out / "merge.json", canonical_json(receipt))
    return receipt


def status(out: Path) -> dict:
    return {
        "jobs": len(list((out / "jobs").glob("*.json"))),
        "phase_a": sum(1 for _ in _phase_a_rows(out)),
        "phase_b": len(list((out / "phase-b").glob("[0-9][0-9][0-9][0-9]*/unit-[0-9][0-9].json"))),
        "merged_units": len(list(out.glob("[0-9][0-9][0-9][0-9]*/unit-[0-9][0-9].json"))),
        "complete_messages": len(list(out.glob("[0-9][0-9][0-9][0-9]*/complete.json"))),
        "reports": {tag: r.get("status") for tag, r in latest_reports(out).items()},
    }
