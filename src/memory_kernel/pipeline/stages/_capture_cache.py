"""Per-job capture cache: a cumulative capture (the whole conversation so far, rebuilt on every ``mk append``) runs
the engine only for the jobs it has not seen.

A capture stays sealed to its own full sources list (``sources_sha256``), and ``conserve`` still counts every message.
The cache only supplies job RESULTS: before the shards run, ``prefill`` writes the phase-A row (and ``jobs/`` file) of
every job whose result is cached, and the phase-B file of every phase-A row whose redo is cached, exactly as the
engine would have; ``run_phase_a`` / ``run_phase_b`` then skip those rows as they skip any row already on disk, and the
merge reads them like any other. After the shards are complete, ``store`` files every result the engine produced.

Why per-job reuse is sound: the engine's id counters are reset before every job (``engine.RESET_IDS_FORM``, sent by
``Engine.reset_ids`` in ``run_phase_a`` and ``redo_pass``), so a job's result depends only on its request and the pinned
engine / parser / package, never on the other jobs of its run (S8: parallel shards give the serial bytes).

The key (sha256 of the canonical JSON of):

* ``pins``: from the capture's sealed manifest (``check_pins`` has just recomputed them from the tree):
  ``schema_version``, ``t_pass``, ``deadline_cap``, ``encoding``, ``binary_sha256``, ``image_sha256``,
  ``config_sha256``, ``walker_sha256``, ``code_pins``, and the snapshot pins' ``files`` map (the census of the whole
  package, ``runtime-pins.json`` with its parser-model pin included). NOT the pins file's own sha (its ``basis``
  names the kernel head, which changes on a commit that changes no file), and NOT what seals the run to its sources
  (``sources_sha256``, ``source_count``, ``syntax_from``, ``syntax_manifest_sha256``, ``content_sha256``): a cumulative
  run has a new sources list every time. A manifest without snapshot pins has no cache.
* ``RESET_IDS_FORM`` and ``CACHE_SCHEMA``.
* phase A: ``parsers.phase_a`` (the dead parser host), the job's sha (``job_sha256`` of its row: the job JSON with the
  unit's text, its stored parse nodes, the parser [url, model] that made them and the parse's sha), the sha of the
  request as sent (``phase_a_request``) and its deadline.
* phase B: ``parsers.phase_b`` (the live parser url), ``SERVING_S``, the job's sha and the sha of the phase-A response
  the redo follows (which passes are redone depends on it).

A result whose pins differ is never reused: that falls out of the key. An entry that cannot be read, or whose schema,
phase or key differ, is a miss. A phase-A row the engine never answered (the ``transport:`` error the driver records)
is never stored, nor the redo that followed it, nor a redo with a lost segment (``status == "error"``), so the next run
redoes it. Each entry carries ``result_sha256`` (of the canonical result); ``_read`` verifies it, and a bad entry is a
miss that the next store moves to ``quarantine/`` and replaces. Stored entries are write-once (link-if-absent): a
valid entry under an existing key is kept. Each hit leaves a receipt ``cache-hits/<A|B>-NNNN-KK.json`` with ``"cached": true`` in the capture, and a
phase-A row from the cache carries ``"cached": true`` (the merge reads only its named fields, so the merged capture
equals a cold one).

Cached time-outs are cached too: a pass that timed out under load replays as a time-out, as a resumed capture would.
Likewise a few passes mint a run-dependent NUMBER of binding variables, so their ``?NAME-<n>`` suffixes differ between
two fresh engines given the same job (the warm-capture-slot probe, 2026-10-16: no engine counter outside
``RESET_IDS_FORM`` moves; only the names differ, never the structure); the cache replays the first run's names.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path

from memory_kernel.construction.memory_semantic_jev import check_seal, sha
from memory_kernel.construction.mined_affordance_contract import canonical_json
from memory_kernel.pipeline.engine import RESET_IDS_FORM
from memory_kernel.pipeline.stages import _parse_capture as pc

CACHE_SCHEMA = "mk-capture-cache/v1"

RECEIPTS = "cache-hits"

QUARANTINE = "quarantine"

PIN_KEYS = (
    "schema_version",
    "t_pass",
    "deadline_cap",
    "encoding",
    "binary_sha256",
    "image_sha256",
    "config_sha256",
    "walker_sha256",
    "code_pins",
)

PHASES = ("A", "B")


def pins_of(manifest: dict) -> dict | None:
    """The key's pin section from a capture manifest; None (no cache) when the manifest declares no snapshot pins."""
    snapshot = manifest.get("snapshot_pins")
    if snapshot is None:
        return None
    return {
        **{k: manifest[k] for k in PIN_KEYS},
        "snapshot_files": snapshot["files"],
        "parsers": manifest["parsers"],
    }


def _key(doc: dict) -> str:
    return sha(canonical_json({"schema": CACHE_SCHEMA, "reset_ids_form": RESET_IDS_FORM, **doc}).encode())


def key_a(pins: dict, *, job_sha256: str, request_sha256: str, deadline_s: float) -> str:
    base = {k: v for k, v in pins.items() if k != "parsers"}
    return _key(
        {
            "phase": "A",
            "pins": base,
            "parser": pins["parsers"]["phase_a"],
            "job_sha256": job_sha256,
            "request_sha256": request_sha256,
            "deadline_s": deadline_s,
        }
    )


def key_b(pins: dict, *, job_sha256: str, phase_a_sha256: str) -> str:
    base = {k: v for k, v in pins.items() if k != "parsers"}
    return _key(
        {
            "phase": "B",
            "pins": base,
            "parser": pins["parsers"]["phase_b"],
            "serving_s": pc.SERVING_S,
            "job_sha256": job_sha256,
            "phase_a_sha256": phase_a_sha256,
        }
    )


def _entry(cache: Path, phase: str, key: str) -> Path:
    return Path(cache) / phase / key[:2] / f"{key}.json"


def _read(cache: Path, phase: str, key: str):
    """The cached result, or None (absent, unreadable, another schema / phase / key, or a result whose bytes differ
    from the entry's ``result_sha256``)."""
    try:
        doc = json.loads(_entry(cache, phase, key).read_bytes())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != CACHE_SCHEMA or doc.get("phase") != phase:
        return None
    if doc.get("key") != key or "result" not in doc:
        return None
    if doc.get("result_sha256") != _result_sha(doc["result"]):
        return None
    return doc["result"]


def _result_sha(result) -> str:
    return sha(canonical_json(result).encode())


def _store(cache: Path, phase: str, key: str, result) -> bool:
    """Write-once (link-if-absent). A valid entry under the key is kept (False); a bad or unreadable one is moved to
    ``quarantine/`` first, then replaced."""
    path = _entry(cache, phase, key)
    if path.exists():
        if _read(cache, phase, key) is not None:
            return False
        quarantine = Path(cache) / QUARANTINE / f"{phase}-{key}.{time.time_ns()}.json"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):  # a concurrent store moved it first
            os.replace(path, quarantine)
    doc = {"schema": CACHE_SCHEMA, "phase": phase, "key": key, "result": result, "result_sha256": _result_sha(result)}
    try:
        pc._atomic_write_once(path, canonical_json(doc))
    except FileExistsError:
        return False
    return True


def _response_sha(response) -> str:
    return sha(canonical_json(response).encode())


def _unanswered(row: dict) -> bool:
    """The driver's own record of a transport failure (``run_phase_a``): the engine never answered."""
    response = row.get("response")
    return (
        isinstance(response, dict)
        and response.get("status") == "error"
        and str(response.get("message", "")).startswith("transport: ")
    )


def _errored_redo(redo: dict) -> bool:
    """A phase-B redo with a lost segment (``status == "error"``: the syntax or unit call never answered)."""
    return any(
        isinstance(segment, dict) and segment.get("status") == "error"
        for fallback in redo.values()
        if isinstance(fallback, dict)
        for segment in fallback.get("segments") or []
    )


def _manifest(out: Path) -> dict:
    manifest = json.loads((Path(out) / "manifest.json").read_bytes())
    check_seal(manifest, pc.HYBRID_CAPTURE_SCHEMA)
    return manifest


def _receipt(out: Path, phase: str, order: int, k: int, key: str) -> None:
    path = Path(out) / RECEIPTS / f"{phase}-{order:04d}-{k:02d}.json"
    if not path.exists():
        pc._write_once(path, {"cached": True, "phase": phase, "key": key, "cache_schema": CACHE_SCHEMA})


def prefill(out: Path, site: pc.Site, cache: Path) -> dict:
    """Write every cached phase-A row and phase-B file this capture lacks; ``{"hits": {"A": n, "B": n}}``.

    Re-checks the pins first (``check_pins``: VOID on any difference), so a cached row never lands in a capture whose
    tree has moved since ``prepare``."""
    manifest = pc.check_pins(out, live_parser=site.live_parser, site=site)
    hits = {"A": 0, "B": 0}
    pins = pins_of(manifest)
    if pins is None:
        return {"hits": hits, "disabled": "the capture manifest declares no snapshot pins"}
    sources = json.loads((Path(out) / "sources.json").read_bytes())
    for order, k, job in pc.hybrid_jobs(sources, pc._resolve(manifest["syntax_from"], site.root)):
        name = f"{order:04d}/unit-{k:02d}.json"
        row_path = Path(out) / "phase-a" / name
        job_sha = sha(canonical_json(job).encode())
        if row_path.exists():
            row = json.loads(row_path.read_bytes())
        else:
            request, deadline, _ = pc.phase_a_request(job, f"{order:04d}-{k:02d}")
            request_sha = hashlib.sha256(request.encode()).hexdigest()
            key = key_a(pins, job_sha256=job_sha, request_sha256=request_sha, deadline_s=deadline)
            cached = _read(cache, "A", key)
            if not isinstance(cached, dict) or cached.get("request_sha256") != request_sha:
                continue
            job_path = Path(out) / "jobs" / f"{order:04d}-{k:02d}.json"
            if job_path.exists():
                if json.loads(job_path.read_bytes()) != job:
                    raise ValueError(f"job {order:04d}-{k:02d} differs from the stored job")
            else:
                pc._write_once(job_path, job)
            row = {
                "job_sha256": job_sha,
                "request_sha256": request_sha,
                "deadline_s": cached["deadline_s"],
                "elapsed_s": cached["elapsed_s"],
                "response": cached["response"],
                "cached": True,
            }
            pc._write_once(row_path, row)
            _receipt(out, "A", order, k, key)
            hits["A"] += 1
        b_path = Path(out) / "phase-b" / name
        if b_path.exists() or row.get("job_sha256") != job_sha:
            continue
        key = key_b(pins, job_sha256=job_sha, phase_a_sha256=_response_sha(row["response"]))
        cached = _read(cache, "B", key)
        if not isinstance(cached, dict) or not cached:
            continue
        pc._write_once(b_path, cached)
        _receipt(out, "B", order, k, key)
        hits["B"] += 1
    return {"hits": hits}


def store(out: Path, cache: Path) -> dict:
    """File every phase-A row and non-empty phase-B file of a capture whose shards are complete under its keys;
    ``{"stored": {"A": n, "B": n}}`` counts the entries written (an existing key is kept; a phase-A row that came from
    the cache is not re-filed, but a redo the engine ran after it is)."""
    manifest = _manifest(out)
    stored = {"A": 0, "B": 0}
    pins = pins_of(manifest)
    if pins is None:
        return {"stored": stored}
    for path, row in pc._phase_a_rows(Path(out)):
        if _unanswered(row):
            continue
        if not row.get("cached"):
            key = key_a(
                pins, job_sha256=row["job_sha256"], request_sha256=row["request_sha256"], deadline_s=row["deadline_s"]
            )
            result = {k: row[k] for k in ("request_sha256", "deadline_s", "elapsed_s", "response")}
            stored["A"] += _store(cache, "A", key, result)
        b_path = Path(out) / "phase-b" / path.parent.name / path.name
        redo = json.loads(b_path.read_bytes()) if b_path.exists() else None
        if redo and not _errored_redo(redo):
            key = key_b(pins, job_sha256=row["job_sha256"], phase_a_sha256=_response_sha(row["response"]))
            stored["B"] += _store(cache, "B", key, redo)
    return {"stored": stored}
