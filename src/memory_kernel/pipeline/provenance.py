"""Provenance helpers: the cost ledger, code identity, the freeze check, sha helpers.

Ported from the reference harness:
  * ``CostRecorder`` / ``ledger_summary`` / ``STAGES`` / ``PROVIDERS``  <- the harness's stage module
    (NO default ledger; ``tags`` replaces the harness's ledger-row tagging rebinding)
  * ``kernel_head`` / ``_git_dir``  <- the harness's worktree-head reader
  * ``check`` / ``census`` / ``sha_file`` / ``FrozenCodeChanged``  <- the harness's freeze module
  * ``rel``  <- the harness's stage helpers (lexical, never resolved)
Row and file formats are byte-identical to the originals. ``tree_census`` / ``write_content_addressed`` replace the
stored snapshot-pin list and the freeze manifest as "what code built this".

Code identity comes from the code that RUNS, never from the data root ``cfg.root``: ``package_dir`` is the
directory of the imported ``memory_kernel`` package (a source checkout, a wheel install or a copy). A pinned key
``src/memory_kernel/<rel>`` resolves under that package; any other key (a harness-relative path)
resolves under an explicit harness root (``--harness-root``) or is refused.
"""

from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: Cost-ledger stage names (the harness stage module's STAGES).
STAGES = (
    "freeze",
    "select",
    "stage",
    "boundaries",
    "taxonomy",
    "capture",
    "clauses",
    "picks",
    "graph",
    "jev1",
    "taxonomy_events",
    "taxonomy_questions",
    "label_questions",
    "jev2",
    "gaploop",
    "remine",
    "eventuality",
    "referents",
    "build",
    "index",
    "answer",
    "rejudge",
    "report",
)
PROVIDERS = ("jev", "openrouter", "voyage")  # jev = typesafe/jev-1.13 via OpenRouter; openrouter = gpt-4.1-mini etc.


class CostRecorder:
    """One appended JSON line per stage run: paid USD, tokens and requests per provider, wall and engine seconds.

        with CostRecorder("taxonomy", run="taxonomy-001", ledger=cfg.ledger, tags={"conv": "conv-syn"}) as cost:
            cost.add("openrouter", requests=5, input_tokens=..., output_tokens=..., usd=...)
            cost.engine(seconds)

    The line carries the sha of the previous line (append-only chain). An exception inside the block is still
    recorded (status "error"), then re-raised. ``tags`` are appended to every row (``{"conv": label}`` is what the
    harness's ``tag_ledger_rows`` rebinding added).
    """

    def __init__(
        self,
        stage: str,
        *,
        run: str = "",
        ledger: Path,
        clock: Callable[[], float] = time.monotonic,
        note: str = "",
        tags: Mapping[str, str] | None = None,
    ):
        if stage not in STAGES:
            raise ValueError(f"unknown cost stage {stage!r}")
        self.stage, self.run, self.ledger, self.clock, self.note = stage, run, ledger, clock, note
        self.tags = dict(tags or {})
        self.providers: dict[str, dict[str, float]] = {}
        self.engine_seconds = 0.0
        self.started = 0.0
        self.started_at = ""

    def add(
        self, provider: str, *, requests: int = 0, input_tokens: int = 0, output_tokens: int = 0, usd: float = 0.0
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        if min(requests, input_tokens, output_tokens, usd) < 0:
            raise ValueError("cost quantities must be non-negative")
        row = self.providers.setdefault(provider, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        row["requests"] += requests
        row["input_tokens"] += input_tokens
        row["output_tokens"] += output_tokens
        row["usd"] += usd

    def engine(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("engine seconds must be non-negative")
        self.engine_seconds += seconds

    def __enter__(self) -> CostRecorder:
        self.started = self.clock()
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.write(status="error" if exc_type else "ok")
        return False

    def row(self, status: str = "ok") -> dict:
        return {
            "schema_version": "mk-cost/v1",
            "stage": self.stage,
            "run": self.run,
            "started_at": self.started_at,
            "wall_seconds": round(self.clock() - self.started, 3),
            "engine_seconds": round(self.engine_seconds, 3),
            "paid_usd": round(sum(p["usd"] for p in self.providers.values()), 9),
            "providers": {k: dict(v) for k, v in sorted(self.providers.items())},
            "status": status,
            "note": self.note,
            **self.tags,
        }

    def write(self, status: str = "ok") -> dict:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            data = handle.read()
            lines = [line for line in data.splitlines() if line.strip()]
            prev = hashlib.sha256(lines[-1]).hexdigest() if lines else None
            row = {**self.row(status), "prev_sha256": prev}
            handle.write(json.dumps(row, sort_keys=True).encode() + b"\n")
            handle.flush()
        return row


def ledger_rows(ledger: Path) -> list[dict]:
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.is_file() else []


def ledger_summary(ledger: Path) -> dict:
    """Totals by stage and overall; verifies the prev_sha256 chain."""
    rows = ledger_rows(ledger)
    raw = [line.encode() for line in ledger.read_text().splitlines() if line.strip()] if ledger.is_file() else []
    for i, row in enumerate(rows):
        expect = hashlib.sha256(raw[i - 1]).hexdigest() if i else None
        if row["prev_sha256"] != expect:
            raise ValueError(f"cost ledger chain broken at line {i + 1}")
    by_stage: dict[str, dict[str, float]] = {}
    for row in rows:
        s = by_stage.setdefault(row["stage"], {"paid_usd": 0.0, "wall_seconds": 0.0, "engine_seconds": 0.0, "runs": 0})
        s["paid_usd"] += row["paid_usd"]
        s["wall_seconds"] += row["wall_seconds"]
        s["engine_seconds"] += row["engine_seconds"]
        s["runs"] += 1
    return {
        "rows": len(rows),
        "paid_usd": round(sum(r["paid_usd"] for r in rows), 9),
        "wall_seconds": round(sum(r["wall_seconds"] for r in rows), 3),
        "engine_seconds": round(sum(r["engine_seconds"] for r in rows), 3),
        "by_stage": by_stage,
    }


# -- kernel head (worktree-aware; no git process) ---------------------------------------------------------------------


def _git_dir(root: Path) -> Path:
    """``root/.git`` as a directory: follows the ``gitdir:`` pointer of a worktree's ``.git`` file."""
    dot = root / ".git"
    if not dot.is_file():
        return dot
    first = dot.read_text().strip()
    if not first.startswith("gitdir:"):
        raise ValueError(f"{dot} is not a gitdir pointer")
    target = Path(first[len("gitdir:") :].strip())
    return target if target.is_absolute() else (root / target).resolve()


def kernel_head(root: Path) -> str:
    """The commit HEAD names (same files, same return value as the reference harness's ``kernel_head``).

    HEAD is read from the gitdir; a ``ref:`` is resolved against the gitdir, then its ``commondir`` (the main
    ``.git``), loose file first and ``packed-refs`` second.
    """
    gitdir = _git_dir(root)
    head = (gitdir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    dirs = [gitdir]
    common = gitdir / "commondir"
    if common.is_file():
        dirs.append((gitdir / common.read_text().strip()).resolve())
    for d in dirs:
        if (d / ref).is_file():
            return (d / ref).read_text().strip()
    for d in dirs:
        packed = d / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    raise ValueError(f"cannot resolve {ref}")


# -- freeze check -----------------------------------------------------------------------------------------------------


class FrozenCodeChanged(RuntimeError):
    def __init__(self, diffs):
        self.diffs = list(diffs)
        super().__init__("frozen code changed: " + "; ".join(self.diffs))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def census(root: Path, globs: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in globs:
        for hit in glob.glob(str(root / pattern), recursive=True):
            if os.path.isfile(hit) and "__pycache__" not in hit:
                out[os.path.relpath(hit, root)] = sha_file(Path(hit))
    return dict(sorted(out.items()))


def _diffs(recorded: Mapping[str, str], current: Mapping[str, str], label: str = "") -> list[str]:
    out = [f"{label}{k}: changed" for k in recorded if k in current and current[k] != recorded[k]]
    out += [f"{label}{k}: missing" for k in recorded if k not in current]
    out += [f"{label}{k}: added" for k in current if k not in recorded]
    return out


def _keyed_census(patterns: Sequence[str], *, harness_root: Path | None, code_root: Path | None) -> dict[str, str]:
    """``census`` over root-relative patterns (a v1 freeze manifest), each resolved through ``code_path``: a
    ``src/memory_kernel/`` pattern under the package, any other under the harness root. Keys keep the v1 spelling."""
    package = code_root if code_root is not None else package_dir()
    out: dict[str, str] = {}
    for pattern in patterns:
        if pattern.startswith(SRC_PREFIX):
            hits = census(package, [pattern[len(SRC_PREFIX) :]])
            out.update({SRC_PREFIX + Path(k).as_posix(): v for k, v in hits.items()})
        elif harness_root is None:
            raise FileNotFoundError(f"{pattern}: a harness file; pass --harness-root")
        else:
            out.update(census(Path(harness_root), [pattern]))
    return dict(sorted(out.items()))


def check(manifest: Path, *, harness_root: Path | None = None, code_root: Path | None = None) -> None:
    """Every file matching the manifest's globs hashes as recorded; none added or missing.

    A v2 manifest (``census_root: "memory_kernel"``, keys relative to the package) censuses the imported package
    (``code_root`` in tests) for ``globs`` / ``code`` and the harness root for ``bench_globs`` / ``bench_code``. A v1
    manifest (root-relative keys) resolves every key and glob through ``code_path``. A harness half with no
    ``harness_root`` raises ``FileNotFoundError``."""
    doc = json.loads(manifest.read_bytes())
    recorded = doc["code"]
    globs = list(doc.get("globs") or recorded)
    if doc.get("census_root") == "memory_kernel":
        diffs = _diffs(recorded, census(code_root if code_root is not None else package_dir(), globs))
        if doc.get("bench_code"):
            if harness_root is None:
                raise FileNotFoundError(f"{manifest}: bench_code pins harness files; pass --harness-root")
            diffs += _diffs(doc["bench_code"], census(Path(harness_root), doc["bench_globs"]), "bench:")
    else:
        diffs = _diffs(recorded, _keyed_census(globs, harness_root=harness_root, code_root=code_root))
    if diffs:
        raise FrozenCodeChanged(diffs)


# -- code identity (the imported package, never cfg.root) --------------------------------------------------------

#: The key prefix of a file of this package in every pin list, census and code key (the source-tree spelling).
SRC_PREFIX = "src/memory_kernel/"


def package_dir() -> Path:
    """The directory of the IMPORTED ``memory_kernel`` package: ``Path(memory_kernel.__file__).parent``, lexical (never
    resolved). A source checkout gives ``<tree>/src/memory_kernel``, a wheel ``.../site-packages/memory_kernel``."""
    import memory_kernel

    return Path(memory_kernel.__file__).parent


def code_path(key: str, *, harness_root: Path | None = None, code_root: Path | None = None) -> Path:
    """Where the pinned file ``key`` is: a ``src/memory_kernel/<rel>`` key under ``code_root`` (default: the imported
    package), any other key under ``harness_root`` (lexical join, ``..`` allowed). FileNotFoundError when a harness
    key has no harness root."""
    if key.startswith(SRC_PREFIX):
        return (code_root if code_root is not None else package_dir()) / key[len(SRC_PREFIX) :]
    if harness_root is None:
        raise FileNotFoundError(f"{key}: a harness file; pass --harness-root")
    return Path(harness_root) / key


def code_key(module: Any) -> str:
    """The pin key of a module of this package: ``src/memory_kernel/<path relative to the package>``."""
    return SRC_PREFIX + Path(module.__file__).relative_to(package_dir()).as_posix()


def _source_tree(code_root: Path) -> Path | None:
    """The checkout around a package laid out as ``<tree>/src/memory_kernel``; None for any other layout (a wheel)."""
    return code_root.parents[1] if code_root.parent.name == "src" else None


def tree_census(code_root: Path | None = None) -> dict[str, str]:
    """``{"src/memory_kernel/<rel>": sha256, ...}`` over the package dir (default: the imported package), sorted, plus
    ``pyproject.toml`` when the package sits in a source tree (``<tree>/src/memory_kernel`` and ``<tree>/pyproject.toml``
    is a file). ValueError when the package dir holds no file.

    The sha is ``memory_semantic_jev.sha`` over the file bytes: the function ``_parse_capture._file_sha`` uses, so a
    census is a valid capture snapshot-pin ``files`` map."""
    from memory_kernel.construction.memory_semantic_jev import sha

    root = Path(code_root) if code_root is not None else package_dir()
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            path = Path(dirpath) / name
            out[SRC_PREFIX + path.relative_to(root).as_posix()] = sha(path.read_bytes())
    if not out:
        raise ValueError(f"empty code census: no file under {root}")
    tree = _source_tree(root)
    if tree is not None and (tree / "pyproject.toml").is_file():
        out["pyproject.toml"] = sha((tree / "pyproject.toml").read_bytes())
    return dict(sorted(out.items()))


def code_head(code_root: Path | None = None) -> str:
    """The git HEAD of the checkout around the package (``.git`` dir or worktree file), else ``"unknown"`` (a wheel, a
    copy, an archive extract). Never raises."""
    try:
        tree = _source_tree(Path(code_root) if code_root is not None else package_dir())
        return "unknown" if tree is None else kernel_head(tree)
    except Exception:  # noqa: BLE001 -- a head is a record, never a reason to stop
        return "unknown"


def write_content_addressed(directory: Path, prefix: str, doc: dict) -> Path:
    """Write ``doc`` as canonical JSON to ``directory/<prefix>-<sha12>.json`` (sha of those bytes), write-once.

    A retry with the same content reuses the file; an existing file with different bytes is an error."""
    raw = json.dumps(doc, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    path = directory / f"{prefix}-{hashlib.sha256(raw).hexdigest()[:12]}.json"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError(f"{path} exists with different content (write-once)") from None
    return path


def rel(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` when it lies under it (lexically), else the path itself. Never resolves."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
