"""The build driver: stage order, done markers, resume, money stops, stages in this process (a few as children).

Every value comes from an explicit ``BuildConfig``: nothing here reads a reference-harness environment variable or
rebinds a module.

Per stage, in order: skip on a done marker (``<stage dir>/.done-<stage>``); stop on a started-but-unmarked stage
unless it is in ``RESUMABLE``; prerequisites (rewrite needs write's marker);
for a paid stage the build cap (the stop) and the per-conversation 1.5x limit (recorded); plan (stages with a plan mode), then live; a gate
that does not pass stops with no marker; else the marker ``{stage, gate, usd, wall_s, finished_at}`` where ``usd``
is this conversation's ledger delta across the stage. One JSON line per event goes to ``<conv_root>/driver-events.jsonl``
with stable event names, so ``mk inspect`` and the bench read the log.

A stage runs in this process (``inprocess.stage_body``, a fresh ``Ctx`` per stage) unless it is in
``inprocess.CHILD_STAGES`` (``capture``), which runs as a child (``python -P -m memory_kernel.cli _stage ...``). Every
stage of a build runs as a child when another build of this process holds ``inprocess.build_lock`` (event
``in-process-busy``) or when ``MK_STAGE_CHILDREN=1`` (event ``stage-children``).
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from functools import cached_property
from pathlib import Path
from typing import Any

from memory_kernel.grammar.dependencies import extraction_model_record
from memory_kernel.pipeline import graphdb, inprocess, ledger, llm, provenance
from memory_kernel.pipeline.config import (
    DEFAULT_THROUGH,
    LAST_STAGE,
    BuildConfig,
    EngineNotSetUp,
    default_recipe,
    engine_image,
    engine_paths,
    engine_pins,
    engine_record,
    grammar_image_sha,
    load_conversation,
    load_replay_scope,
    session_numbers,
)

#: Kernel 1.8: the taxonomy and question layer (taxonomy, jev1_pilot, jev1, taxonomy_events, taxq, labelq, jev2, remine,
#: graph_b) is archived; an older root or config that names one reads it through
#: ``config.ARCHIVED_STAGES``.
ORDER: tuple[str, ...] = (
    "select",
    "boundary",
    "capture",  # syntax + capture + compat runs, the only stage that holds the engine
    "graph_picks",
    "graph_a",
    "events",
    "ctx",
    "referents",  # an append's decisions on specific mentions (a full build writes a marker)
    "build_dry",
    "write",
    "rewrite",
    "index",
    "sanity",
    "seal",
)  # mk stops at seal (config.LAST_STAGE); the evaluation stages belong to the benchmark harness
STAGE_DIRS: dict[str, str] = {
    "select": "01-select-001",
    "boundary": "03-boundary-001",
    "shard": "03-boundary-001-ledger",
    "syntax": "04-syntax-001",
    "capture": "04-capture-001",
    "runs": "05-runs-001",
    "graph_a": "06-graph-001",  # graph_picks writes into the same dir
    "events": "10-events-001",
    "ctx": "10-eventuality-ctx-001",
    "referents": "10-referents-001",
    "build_dry": "11-dryrun-001",
    "write": "11-write-001",
    "index": "11-index-001",
    "sanity": "11-sanity-001",
    "seal": "11-seal-001",
    "cache": "cache",
    "logs": "logs",
}
#: Stages whose re-run over a started dir is safe (boundary is NOT: ``mkdir(exist_ok=False)``).
RESUMABLE: frozenset[str] = frozenset({"capture", "syntax", "events", "ctx"})
#: The 369-message reference conversation's measured per-stage cost under the adopted recipe.
INDEX_USD: dict[str, float] = {
    "boundary": 0.053184,
    "graph_picks": 0.004192,
    "events": 0.014047,
    "ctx": 0.086998,
    "referents": 0.0001,  # not measured on the reference (a full build asks nothing): an append's hard cases
    "write": 0.000694,
    "rewrite": 0.0,
    "index": 0.001319,
    "sanity": 0.0,  # a receipt (stages.receipt): no search, no Voyage
}
REF_MESSAGES = llm.REF_MESSAGES  # the 369-message reference conversation: the scale of the bootstrap's projections
PAID: frozenset[str] = frozenset(INDEX_USD)
STOP_FACTOR = 1.5
#: The per-message projection: a full build's recorded limit (``conv_limit``) and an append's budget
#: (``bonfire.append_budget``) are ``STOP_FACTOR`` x it per message. Kernel 1.8: 1.7's 0.0025 x the kept stages' share
#: of the reference conversation's measured cost (``INDEX_USD``: 0.160534 of 1.7's 0.649053), rounded up.
INDEX_USD_PER_MESSAGE = 0.00062
#: STAGE_DIRS key of a stage that has none of its own (graph_picks writes into graph_a's 06-graph-001).
DIR_KEY = {"graph_picks": "graph_a", "rewrite": "write"}
#: The concurrency recipe: the first paid parallel stage (``events``) at 4; after a transport error or a
#: 429, every later stage of the conversation at 4.
PILOT_STAGE, PILOT_CONCURRENCY, RAMP_DOWN_CONCURRENCY = "events", 4, 4
RAMP_EVENT = "concurrency-ramp"
TRANSPORT_TYPES: tuple[str, ...] = ("ProviderTransportError", "TransportError", "Timeout", "URLError", "Connection")
#: Provider HTTP statuses that mean "try again later", never "this request is wrong" (``classify_failure``).
TRANSIENT_HTTP: frozenset[int] = frozenset({429, 499, 529})
#: A stage stopped ONLY by transient failures is re-run in the same driver at the ramped concurrency, at most this
#: many times. The failed wire is never resent: the re-run asks its twin (``llm.lost_wires``), as a resume does.
TRANSIENT_RERUNS = 3


class Stop(RuntimeError):
    """The driver stops here; the reason goes to driver-events.jsonl."""


# -- the conversation (counts from the conversation JSON) ------------------------------------------------------------


@dataclass(frozen=True)
class Counts:
    messages: int
    sessions: int


def conversation_counts(conversation: Path, sample_id: str | None = None) -> Counts:
    """Messages = the message rows ``select`` emits (the ONE message count: capture conserve, the episode map, the write
    and index checks compare with it, and the caps scale with it). It equals the turn count unless a turn is split into
    parts or dropped (only a mention, only markup, blank). Sessions = the ``session_<n>`` lists."""
    from memory_kernel.pipeline.stages._conversation import build_prepared, split_conversation

    item, _ = load_conversation(conversation, sample_id)
    prepared = build_prepared(split_conversation(item)[0])
    messages = sum(row["metadata"]["source_kind"] == "message" for batch in prepared for row in batch)
    return Counts(messages=messages, sessions=len(session_numbers(item["conversation"])))


# -- the stage context (built in every stage child) ------------------------------------------------------------------


@dataclass
class Ctx:
    cfg: BuildConfig
    concurrency: int
    lock_fd: int | None = None
    batcher: llm.QuestionBatcher = field(default_factory=llm.QuestionBatcher)

    @property
    def root(self) -> Path:
        return self.cfg.conv_root

    def path(self, key: str) -> Path:
        return path(self.cfg, key)

    def save_plan(self, name: str, plan: dict) -> None:
        logs = self.path("logs")  # the harness's save_plan, byte-identical
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f"{name}-plan.json").write_text(json.dumps(plan, indent=1, default=str, sort_keys=True))

    def cost(self, stage: str, *, run: str, note: str) -> provenance.CostRecorder:
        return provenance.CostRecorder(
            stage, run=run, ledger=self.cfg.ledger, note=note, tags={"conv": self.cfg.sample_id}
        )

    def freeze_check(self) -> None:
        if self.cfg.freeze_manifest is not None:
            provenance.check(self.cfg.freeze_manifest, harness_root=self.cfg.harness_root)

    @cached_property
    def counts(self) -> Counts:
        return conversation_counts(self.cfg.conversation, self.cfg.sample_id)

    @property
    def messages(self) -> int:
        """The conversation's message count: what the stage caps scale with (``llm.cap_scale``)."""
        return self.counts.messages

    def pilot_sessions(self) -> frozenset[str]:
        """First, middle (ceil) and last session: 19 -> 1, 10, 19."""
        n = self.sessions()
        return frozenset(f"{self.cfg.sample_id}-session_{k}" for k in (1, (n + 1) // 2, n))

    def sources(self) -> list[dict]:
        return json.loads((self.path("select") / "sources.json").read_bytes())

    # -- what a ported stage body needs, built once per stage child (no module-global caches) ----------

    @cached_property
    def sender(self) -> llm.JevSender:
        """The Jev HTTP sender; it owns its httpx client and only connects on a ledger cache miss."""
        return llm.JevSender(api_key=self.cfg.openrouter_api_key)

    @cached_property
    def prefetch(self) -> llm.Prefetcher:
        return llm.Prefetcher(self.concurrency)

    def shard(self, **hooks: Any) -> ledger.Shard:
        """The conversation's round-ledger shard (``03-boundary-001-ledger``, or a bonfire's ``cfg.shard_dir``) with
        this build's sender and key; a harness-made shard re-hashes its harness pins under ``cfg.harness_root``. An
        append run's shard carries the run's limit (``bonfire.append_limit``, from the plans saved so far), not the
        shard's sealed caps."""
        from memory_kernel.pipeline.bonfire import append_limit

        return ledger.Shard(
            path=self.path("shard"),
            harness_root=self.cfg.harness_root,
            sender=self.sender,
            api_key=self.cfg.openrouter_api_key,
            limit_usd=append_limit(self.cfg),
            **hooks,
        )

    def sessions(self) -> int:
        return self.counts.sessions


StageFn = Callable[[Ctx, bool], Any]


# -- paths -----------------------------------------------------------------------------------------------------------


def path(cfg: BuildConfig, key: str) -> Path:
    """``<conv_root>/<stage dir>``; the shard is ``cfg.shard_dir`` when one is set (a bonfire's persistent shard, see
    ``pipeline/bonfire.py``). The shard's ``-pairs`` sibling is named from the shard path (``ledger.make_execution``),
    so it follows the shard."""
    if key == "shard" and (shard_dir := getattr(cfg, "shard_dir", None)) is not None:
        return shard_dir
    return cfg.conv_root / STAGE_DIRS[key]


def stage_dir(cfg: BuildConfig, stage: str) -> Path:
    """Where a stage's done marker lives. capture -> 04-capture-001."""
    return path(cfg, DIR_KEY.get(stage, stage))


def marker(cfg: BuildConfig, stage: str) -> Path:
    return stage_dir(cfg, stage) / f".done-{stage}"


def evidence(cfg: BuildConfig, stage: str) -> list[Path]:
    """Paths whose existence means the stage started."""
    if stage == "graph_a":
        return [path(cfg, "graph_a") / "graph.json"]
    if stage == "rewrite":
        return [path(cfg, "write") / "rerun.json"]
    if stage == "capture":
        return [path(cfg, "syntax"), path(cfg, "capture"), path(cfg, "runs")]
    return [path(cfg, DIR_KEY.get(stage, stage))]


def events_path(cfg: BuildConfig) -> Path:
    return cfg.conv_root / "driver-events.jsonl"


def lock_path(cfg: BuildConfig) -> Path:
    return cfg.conv_root / "driver.lock"


def result_path(cfg: BuildConfig, stage: str, mode: str) -> Path:
    """The child's result JSON (``logs`` is not sealed)."""
    return path(cfg, "logs") / "driver" / f"{stage}-{mode}.json"


def read_marker(cfg: BuildConfig, stage: str) -> dict | None:
    m = marker(cfg, stage)
    return json.loads(m.read_bytes()) if m.is_file() else None


# -- events ----------------------------------------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log_event(cfg: BuildConfig, event: str, stage: str | None = None, **fields: Any) -> dict:
    row = {"at": now_iso(), "conv": cfg.sample_id, "event": event, "stage": stage, **fields}
    p = events_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return row


def read_events(cfg: BuildConfig) -> list[dict]:
    p = events_path(cfg)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.is_file() else []


# -- money -----------------------------------------------------------------------------------------------------------


def paid_usd(cfg: BuildConfig, label: str | None = None) -> float:
    """Build ledger paid total; with ``label``, only the rows tagged ``conv: label``."""
    if label is None:
        return float(provenance.ledger_summary(cfg.ledger)["paid_usd"])
    rows = provenance.ledger_rows(cfg.ledger)
    return round(sum(float(r.get("paid_usd") or 0.0) for r in rows if r.get("conv") == label), 9)


def projected_share(cfg: BuildConfig, counts: Counts, stage: str) -> float:
    if stage in INDEX_USD:
        return INDEX_USD[stage] * counts.messages / REF_MESSAGES
    return 0.0


def planned_usd(cfg: BuildConfig) -> float:
    """The ``est_usd`` total of this run's saved stage plans (``logs/*-plan.json``); a plan without one counts 0."""
    total = 0.0
    for plan in sorted(path(cfg, "logs").glob("*-plan.json")):
        try:
            est = json.loads(plan.read_bytes()).get("est_usd")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(est, (int, float)):
            total += float(est)
    return total


def conv_limit(cfg: BuildConfig, counts: Counts) -> float:
    """Per-conversation limit (recorded past it, never a stop: ``_money``): 1.5 x the per-message projection (no
    evaluation share: mk runs none). A forward append's limit is its budget (``bonfire.append_budget`` of its messages
    and of its stages' planned spend), the one its shard holds it to."""
    if getattr(cfg, "forward_input", None) is not None:
        from memory_kernel.pipeline.bonfire import append_budget

        return float(append_budget(counts.messages, planned_jev_usd=Decimal(str(planned_usd(cfg)))))
    return STOP_FACTOR * INDEX_USD_PER_MESSAGE * counts.messages


def conv_spent(cfg: BuildConfig) -> float:
    """This conversation's spend: its tagged ledger rows."""
    return paid_usd(cfg, cfg.sample_id)


def _money(cfg: BuildConfig, counts: Counts, stage: str) -> dict:
    paid, share = paid_usd(cfg), projected_share(cfg, counts, stage)
    spent, limit = conv_spent(cfg), conv_limit(cfg, counts)
    verdict = {
        "paid_usd": paid,
        "projected_share_usd": round(share, 6),
        "cap_usd": cfg.cap_usd,
        "conv_spent_usd": spent,
        "conv_limit_usd": round(limit, 6),
    }
    if paid + share > cfg.cap_usd:
        raise Stop(f"build cap: paid {paid} + projected {share:.6f} > {cfg.cap_usd}")
    if spent > limit:  # recorded (the verdict rides on the stage's ``start`` event), never a stop: --cap-usd is
        verdict["conv_limit_exceeded"] = True
        log_event(cfg, "conv-limit-exceeded", stage, conv_spent_usd=spent, conv_limit_usd=round(limit, 6))
    return verdict


# -- the concurrency ramp ----------------------------------------------------------------------------------------------


def write_code_census(cfg: BuildConfig) -> Path:
    """``logs/mk/code-census-<sha12>.json``: the head and the census of the IMPORTED package at driver start. A record of what code built this; nothing compares it."""
    doc = {"kernel_head": provenance.code_head(), "files": provenance.tree_census()}
    return provenance.write_content_addressed(path(cfg, "logs") / "mk", "code-census", doc)


def recipe_record(cfg: BuildConfig, *, code_census: Path | None = None) -> dict:
    """The effective recipe for ``driver-start`` (the adopted recipe constants and the freeze sha), plus the
    sha of this driver's code census file. A genesis build adds ``"genesis": true``; any other record is unchanged."""
    sha = None
    if cfg.freeze_manifest is not None:
        try:
            sha = hashlib.sha256(cfg.freeze_manifest.read_bytes()).hexdigest()
        except OSError:
            sha = None
    return {
        "name": "mk-build",
        "jev_concurrency": cfg.jev_concurrency,
        "pilot_stage": PILOT_STAGE,
        "pilot_concurrency": PILOT_CONCURRENCY,
        "ramp_down_concurrency": RAMP_DOWN_CONCURRENCY,
        "freeze_manifest_sha256": sha,
        "extraction_model": extraction_model_record(),
        "code_census_sha256": None if code_census is None else hashlib.sha256(code_census.read_bytes()).hexdigest(),
        **({"genesis": True} if cfg.genesis else {}),
    }


def ramped_down(cfg: BuildConfig) -> bool:
    return any(e["event"] == RAMP_EVENT for e in read_events(cfg))


def stage_concurrency(cfg: BuildConfig, stage: str, ramped: bool) -> int:
    """Jev requests in flight for ``stage``: ``cfg.jev_concurrency``, capped at 4 for the pilot stage and, after a
    ramp-down, for every stage."""
    n = cfg.jev_concurrency
    if stage == PILOT_STAGE:
        n = min(n, PILOT_CONCURRENCY)
    if ramped:
        n = min(n, RAMP_DOWN_CONCURRENCY)
    return n


def lost_wire_roots(cfg: BuildConfig) -> list[Path]:
    """Where the twin re-ask looks for lost attempts: the conversation root, and a bonfire's shard outside it (an
    earlier run's failed attempt lives there, and ``rglob`` does not follow the run root's link to it)."""
    return [cfg.conv_root] + ([cfg.shard_dir] if getattr(cfg, "shard_dir", None) is not None else [])


def failure_files(cfg: BuildConfig) -> set[Path]:
    roots = {cfg.conv_root, path(cfg, "shard")}
    return {p for r in roots if r.is_dir() for p in r.rglob("failure.json")}


def classify_failure(p: Path) -> dict | None:
    """``{kind: rate_limit|transport, ...}`` for a 429 / 499 / 529 or a transport loss; None for any other failure.
    499 ("internal_error: Unable to complete this request") and 529 ("overloaded") are the provider failing under
    load: a chat bootstrap probe met five in one bootstrap and its appends, each answered by the same resume."""
    try:
        doc = json.loads(p.read_bytes())
    except (OSError, ValueError):
        return None
    kind, reason = str(doc.get("type", "")), str(doc.get("reason", ""))
    status = None
    http = p.parent / "http-error.json"
    if http.is_file():
        try:
            status = json.loads(http.read_bytes()).get("status")
        except (OSError, ValueError):
            status = None
    rate = status in TRANSIENT_HTTP or any(f"HTTP {s}" in reason or f"HTTP Error {s}" in reason for s in TRANSIENT_HTTP)
    transport = any(t in kind for t in TRANSPORT_TYPES) or "outcome_unknown" in reason
    if not (rate or transport):
        return None
    return {"kind": "rate_limit" if rate else "transport", "type": kind, "reason": reason[:200], "path": str(p)}


def transient_only(cfg: BuildConfig, before: set[Path]) -> bool:
    """True when the attempt left at least one new failure and every new one is transient (``classify_failure``). A
    stop with no new failure (a cap, a gate, a refusal) or with any other failure is never re-run."""
    new = sorted(failure_files(cfg) - before)
    return bool(new) and all(classify_failure(p) is not None for p in new)


def ramp_if_transport(cfg: BuildConfig, stage: str, before: set[Path]) -> dict | None:
    hits = [c for p in sorted(failure_files(cfg) - before) if (c := classify_failure(p)) is not None]
    if not hits:
        return None
    return log_event(cfg, RAMP_EVENT, stage, to=RAMP_DOWN_CONCURRENCY, failures=len(hits), first=hits[:8])


# -- the conversation lock -------------------------------------------------------------------------------------------


def acquire_lock(cfg: BuildConfig, *, stamp: bool = True) -> int:
    """Exclusive non-blocking flock on ``<conv_root>/driver.lock``; returns the fd. Stop when any process holds it.
    Every stage child inherits the fd (``pass_fds``), so the lock lives on the shared open file description."""
    cfg.conv_root.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path(cfg), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise Stop(
            f"{lock_path(cfg)} is held: a driver or an orphaned stage child of {cfg.sample_id} is running"
        ) from None
    if stamp:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def lock_held(cfg: BuildConfig) -> bool:
    """True while any process holds the conversation lock. Read-only (``mk inspect`` calls it): an EXISTING lock file
    is opened ``O_RDONLY`` and probed with a non-blocking SHARED flock (it conflicts with the driver's exclusive one);
    an absent file means nobody holds it. Nothing is created: no directory, no file."""
    try:
        fd = os.open(lock_path(cfg), os.O_RDONLY)
    except (FileNotFoundError, NotADirectoryError):
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)  # closing the description also drops a shared lock this probe just took
    return False


# -- the child: one stage in this process ----------------------------------------------------------------------------


def _gate(stage: str, out: Any) -> dict | None:
    """``out["gate"]`` when the stage returns one, else a gate from ``out["passed"]`` (build_dry), else None."""
    if isinstance(out, Mapping):
        if "gate" in out and (out["gate"] is None or "pass" in out["gate"]):
            return out["gate"]
        if "passed" in out:
            return {"name": stage, "pass": bool(out["passed"]), "values": {"passed": out["passed"]}}
    return None


def audited_modules(cfg: BuildConfig) -> list[str]:
    """Loaded modules that come from a ``scripts`` tree (the harness or a pilot tree), by name or by file."""
    dirs = [(cfg.root / "scripts").resolve()] + ([(cfg.vault / "scripts").resolve()] if cfg.vault is not None else [])
    out: set[str] = set()
    for name, module in list(sys.modules.items()):
        if name == "scripts" or name.startswith("scripts.") or name == "integrated_jev_provider":
            out.add(name)
            continue
        file = getattr(module, "__file__", None)
        if not file:
            continue
        try:
            resolved = Path(file).resolve()
        except OSError:
            continue
        if any(resolved.is_relative_to(d) for d in dirs):
            out.add(name)
    return sorted(out)


def run_stage_child(
    config_path: Path,
    stage: str,
    *,
    live: bool,
    result: Path | None,
    lock_fd: int | None,
    concurrency: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    """The ``mk _stage`` child: rebuild the config, hold the driver lock, run one stage (``inprocess.stage_body``),
    write its result."""
    from memory_kernel.pipeline.stages import STAGES

    env = os.environ if environ is None else environ
    cfg = BuildConfig.from_json(json.loads(config_path.read_bytes()), secrets=env)
    if stage not in STAGES:
        raise Stop(f"stage {stage!r} is not runnable in this build (mk stops at {LAST_STAGE!r})")
    if not 1 <= concurrency <= cfg.jev_concurrency:
        raise Stop(f"--concurrency {concurrency}: must be 1..{cfg.jev_concurrency}")
    if lock_fd is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the driver's inherited lock: same description, no-op
        held = lock_fd
    else:
        held = acquire_lock(cfg)  # a hand-run stage holds the conversation lock too; refused under a driver
    try:
        doc = inprocess.stage_body(cfg, stage, live=live, lock_fd=held, concurrency=concurrency)
    finally:
        if lock_fd is None:
            os.close(held)
    text = _result_text(doc)
    if result is not None:
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(text)
    print(text[:20000])
    return 0


def _result_text(doc: dict) -> str:
    """A stage's result doc as the driver reads it, from a child or in-process."""
    return json.dumps(doc, sort_keys=True, indent=1, default=str)


# -- the loop --------------------------------------------------------------------------------------------------------

Runner = Callable[..., Any]


def package_file() -> str:
    import memory_kernel

    return str(Path(memory_kernel.__file__))


def child_env(cfg: BuildConfig, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The parent's env + the secrets (they reach the child only through its environment), with ``PYTHONPATH`` derived
    from THIS process's package and never inherited (the venv's editable install may point at another tree): the
    package's source root for a source tree (its directory sits in a ``src`` dir), unset for an installed package. The
    build root is never on the path: ``stage_argv`` starts the child with ``-P`` and the child's cwd is the root, so a
    stray ``./memory_kernel`` or ``./scripts`` in a data directory cannot shadow the code that ran the driver."""
    env = dict(os.environ if base is None else base)
    pkg = provenance.package_dir()
    if pkg.parent.name == "src":
        env["PYTHONPATH"] = str(pkg.parent)
    else:
        env.pop("PYTHONPATH", None)
    if cfg.data_dir is not None:  # the child resolves the same setup data (models, tokenizer, corpora) as the driver
        env["MK_DATA_DIR"] = str(cfg.data_dir)
    return env | cfg.secret_env()


def stage_argv(cfg: BuildConfig, stage: str, *, live: bool, result: Path, lock_fd: int, concurrency: int) -> list[str]:
    argv = [sys.executable, "-P", "-m", "memory_kernel.cli", "_stage", stage, "--config", str(cfg.config_path)]
    argv += ["--result", str(result)]
    if live:
        argv.append("--live")
    argv += ["--lock-fd", str(lock_fd), "--concurrency", str(concurrency), "--expect-package", package_file()]
    return argv


def _invoke(cfg: BuildConfig, stage: str, mode: str, runner: Runner, lock_fd: int, n: int, *, in_process: bool) -> dict:
    """One call of a stage: in this process (``inprocess.stage_body``) when ``in_process`` and the stage is not in
    ``inprocess.CHILD_STAGES``, else a ``mk _stage`` child. Either way the result doc is written to ``result_path``
    and read back from its JSON."""
    result = result_path(cfg, stage, mode)
    if result.exists():
        result.unlink()  # a stale result from an earlier attempt must never be read as this one's
    if in_process and stage not in inprocess.CHILD_STAGES:
        try:
            doc = inprocess.stage_body(cfg, stage, live=mode == "live", lock_fd=lock_fd, concurrency=n)
        except Stop as stop:
            if (text := inprocess.redact(cfg, str(stop))) != str(stop):
                raise Stop(text) from None
            raise
        except (Exception, SystemExit) as exc:  # a stage's spend refusal is a SystemExit (ctx): a child's rc 1
            # the build log, where a child's traceback went; no secret in it, nor in the Stop (no cause carried)
            print(inprocess.redact(cfg, traceback.format_exc()), end="", file=sys.stderr)
            raise Stop(inprocess.redact(cfg, f"{stage} {mode}: {type(exc).__name__}: {exc}")) from None
        text = _result_text(doc)
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(text)
        print(text[:20000])
        return json.loads(text)
    argv = stage_argv(cfg, stage, live=mode == "live", result=result, lock_fd=lock_fd, concurrency=n)
    rc = runner(argv, cwd=cfg.root, env=child_env(cfg), stdin=subprocess.DEVNULL, pass_fds=(lock_fd,)).returncode
    if rc != 0:
        raise Stop(f"{stage} {mode}: child exited rc={rc} (see the build log)")
    if not result.is_file():
        raise Stop(f"{stage} {mode}: child wrote no result {result}")
    return json.loads(result.read_bytes())


def _adoptable(cfg: BuildConfig, stage: str) -> bool:
    if stage == "write":
        return (path(cfg, "write") / "summary.json").is_file()
    if stage == "rewrite":
        return (path(cfg, "write") / "rerun.json").is_file()
    return False


def _check_started(cfg: BuildConfig, stage: str) -> bool:
    """True when the stage started earlier (evidence without a marker) and is resumable; Stop when not resumable."""
    started = [str(p) for p in evidence(cfg, stage) if p.exists()]
    if not started:
        return False
    if stage in RESUMABLE or _adoptable(cfg, stage):
        return True
    raise Stop(f"{stage}: stage dir exists without .done: bump the suffix or inspect ({started})")


def _prerequisites(cfg: BuildConfig, stage: str) -> None:
    if stage == "rewrite" and read_marker(cfg, "write") is None:
        raise Stop("rewrite: requires write's done marker")


def _gate_or_stop(cfg: BuildConfig, stage: str, out: dict, mode: str) -> dict | None:
    gate = out.get("gate")
    if gate is not None:
        log_event(cfg, "gate", stage, mode=mode, gate=gate)
        if not gate.get("pass"):
            if gate.get("advisory") and not getattr(cfg, "strict_gates", False):
                log_event(cfg, "gate-advisory", stage, mode=mode, gate=gate)
                return gate
            raise Stop(f"{stage} {mode}: gate {gate.get('name')!r} did not pass")
    return gate


def run_stage(
    cfg: BuildConfig, counts: Counts, stage: str, *, runner: Runner, lock_fd: int, in_process: bool = False
) -> dict:
    """One stage end to end; returns the marker body. Raises Stop (the caller logs it with the usd delta)."""
    resumed = _check_started(cfg, stage)
    _prerequisites(cfg, stage)
    money = _money(cfg, counts, stage) if stage in PAID else None
    n = stage_concurrency(cfg, stage, ramped_down(cfg))
    log_event(cfg, "start", stage, resumed=resumed, money=money, jev_concurrency=n)
    # one pass: a stage's live branch recomputes and saves its plan and runs its own checks before its first paid call
    live = _invoke(cfg, stage, "live", runner, lock_fd, n, in_process=in_process)
    log_event(cfg, "live", stage)
    gate = _gate_or_stop(cfg, stage, live, "live")
    return {"stage": stage, "gate": gate}


def _spent_since(cfg: BuildConfig, before: float) -> float | None:
    try:
        return round(conv_spent(cfg) - before, 9)
    except Exception:  # noqa: BLE001 -- a broken ledger must not hide the stop event
        return None


# -- preflight: refuse before any stage runs -----------------------------------------------------------

#: Stages whose bodies open the PropBank frame dirs (``_g1_build_u2``) / talk to Neo4j.
FRAME_STAGES = frozenset({"events", "ctx", "referents", "build_dry", "write", "rewrite"})
NEO4J_STAGES = frozenset({"write", "rewrite", "index"})  # sanity is a receipt: it reads no graph
RECIPE_STAGES = frozenset({"index"})
#: Stages that call ``embed.embedder`` (Voyage, or the cache in front of it): ``write`` and ``rewrite`` embed the
#: documents they MERGE (``stages/write.py``), ``index`` the search keys (``stages/index.py``); ``sanity`` is a receipt
#: and embeds nothing. ``embedder`` raises only when the stage runs, after every paid stage before it, so the preflight
#: asks for a source.
EMBED_STAGES = frozenset({"write", "rewrite", "index"})
#: The ``events`` stage embeds the conversation's messages with Voyage (to cut it into segments) through its OWN cache
#: file, not through ``embed.embedder`` (``stages/events.py`` hands this path to the sealed embedder). Without a key an
#: uncached message raises only when the stage runs, after the paid stages, so the preflight asks for a key or that file.
EVENTS_VOYAGE_CACHE = "10-events-voyage-cache.json"


def events_voyage_cache(cfg: BuildConfig) -> Path:
    """The events stage's message-embedding cache: under the conversation root (``logs/``), or for a forward append
    the bonfire's persistent one (``<bonfire>/chain/embeddings/``, named by its run input), so nothing is paid twice."""
    from memory_kernel.pipeline.config import load_forward

    forward = load_forward(cfg)
    if forward is not None:
        return Path(forward["voyage_cache"])
    return path(cfg, "logs") / EVENTS_VOYAGE_CACHE


def wordnet_problems() -> list[str]:
    """The NLTK WordNet corpus must be findable by nltk's own lookup (``NLTK_DATA``, ``nltk.data.path``, ``~/nltk_data``,
    and the ``<data>/nltk_data`` that ``cli.wire_runtime_data`` put first). The envelope renderer needs it for verb
    agreement and raises when it is missing (``envelope/render.py``), which would be after the paid stages. The two forms
    are the ones ``nltk.corpus.wordnet`` itself tries (``LazyCorpusLoader``): the extracted ``corpora/wordnet`` directory
    and the ``corpora/wordnet.zip`` archive (a host with only the zip loads fine). Presence only: nothing is loaded and no
    verdict is cached."""
    import nltk.data

    for resource in ("corpora/wordnet.zip/wordnet/", "corpora/wordnet"):
        try:
            nltk.data.find(resource)
        except LookupError:
            continue
        return []
    return [
        "wordnet: the NLTK WordNet corpus is not found (every stage after select renders with it): "
        "run `mk setup --component wordnet`"
    ]


def pending_stages(cfg: BuildConfig) -> list[str]:
    """Stages up to ``through`` whose done marker is absent (read-only: nothing is created)."""
    through = cfg.through or DEFAULT_THROUGH
    return [s for s in ORDER[: ORDER.index(through) + 1] if not marker(cfg, s).is_file()]


#: Wall-clock bound of the ONE Neo4j preflight attempt (connect + the two count queries). A closed port is refused at
#: once; this only bounds a black-holed host.
NEO4J_PREFLIGHT_TIMEOUT_S = 30.0
NEO4J_RATE_LIMIT_CODE = "Neo.ClientError.Security.AuthenticationRateLimit"


#: The sealed eventuality code labels each frame dir by its path below the fourth parent of the FIRST dir
#: (``frame_index``: ``Path(dirs[0]).parents[3]``), and raises a bare IndexError when the first dir has fewer parents. That
#: stage runs after the paid ones, so a shallow first dir is refused here, before any spend.
FRAME_DIR_MIN_PARENTS = 4


def frame_dir_problems(cfg: BuildConfig, stages: list[str]) -> list[str]:
    """Every frame dir must exist (the frame stages read them all), and the first must lie deep enough
    (``FRAME_DIR_MIN_PARENTS``). The setup-managed upstream dir names the fix."""
    from memory_kernel.pipeline import frames

    if not cfg.frame_dirs:
        return [
            f"no frame dirs (needed by {stages}): pass --frame-dir, or run `mk setup --component propbank-frames` "
            "(--data-dir / MK_DATA_DIR says where)"
        ]
    managed = frames.upstream_dir(cfg.data_dir) if cfg.data_dir is not None else None
    problems = []
    parents = len(Path(cfg.frame_dirs[0]).parents)
    if parents < FRAME_DIR_MIN_PARENTS:
        problems.append(
            f"frame dir {cfg.frame_dirs[0]} has only {parents} parent directories (needed by {stages}): the build derives a "
            f"label from the {FRAME_DIR_MIN_PARENTS}th parent of the FIRST frame dir and would fail after the paid stages; "
            "place that dir at least four levels below the filesystem root (for example /srv/mk/propbank/frames)"
        )
    problems += [
        f"frame dir {d} is not a directory (needed by {stages})"
        + (": run `mk setup --component propbank-frames`" if d == managed else "")
        for d in cfg.frame_dirs
        if not d.is_dir()
    ]
    return problems


def engine_problems(cfg: BuildConfig) -> list[str]:
    """The engine binary and image must resolve, be files, and have the pinned full sha256 (with
    ``allow_unpinned_engine`` the shas are not enforced). Hashes the two files, so it runs only when ``capture`` is pending."""
    from memory_kernel.pipeline.engine import pin_problems

    try:
        binary, image = engine_paths(cfg)
    except EngineNotSetUp as exc:
        return [f"capture: {exc}"]
    problems = [
        f"capture: {what} {p} is not a file"
        for what, p in (("engine binary", binary), ("engine image", image))
        if not p.is_file()
    ]
    if not problems:
        problems = [
            f"capture: {text} (`mk setup` fetches the pinned engine; --allow-unpinned-engine accepts another)"
            for text in pin_problems(
                engine_pins(cfg),
                binary_sha256=provenance.sha_file(binary),
                image_sha256=provenance.sha_file(image),
            )
        ]
    return problems


def parser_problems(cfg: BuildConfig) -> list[str]:
    """An external ``--parser-url`` must be local unless remote parsers are allowed; the managed parser (no URL) needs the
    ``[parser]`` extra, the model ``mk setup`` placed and a free port (``engine.parser_service_problems``); under the warm
    capture slot (``cfg.capture_slot``) the capture leases the slot's parser, which holds that port, so the port is not
    checked here (a cold fallback's own launch still checks it). Starts nothing; runs only when ``capture`` is pending."""
    from memory_kernel.pipeline.engine import (
        ParserConfig,
        RemoteParserRefused,
        parser_service_problems,
        require_local,
    )

    if cfg.parser_url is not None:
        try:
            require_local(cfg.parser_url, allow_remote=cfg.allow_remote_parser)
        except RemoteParserRefused as exc:
            return [str(exc)]
        return []
    return parser_service_problems(
        ParserConfig(port=cfg.parser_port, data_dir=cfg.data_dir), check_port=cfg.capture_slot is None
    )


def boundary_replay_problems(pair: tuple[Path, Path], scope_doc: Mapping[str, Any]) -> list[str]:
    """The boundary stage's $0 replay gate, run up front: a replay pair this code does not reproduce (for example a scope
    file without the archived run's ``step_schema``) would otherwise stop the build only when the boundary stage
    runs. Same pass rule as the stage's gate."""
    from memory_kernel.pipeline.stages import _boundary_walk as ub

    archive, sources = pair
    try:
        r = ub.replay_equivalence(archive, json.loads(sources.read_bytes()), **scope_doc)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"boundary replay gate: the replay pair does not replay ({type(exc).__name__}: {exc})"]
    if r["match"] == r["total"] and r["total"] and r["episodes_match"]:
        return []
    return [
        f"boundary replay gate: {r['match']}/{r['total']} archived same-day steps reproduce (episodes match: "
        f"{r['episodes_match']}); check the --boundary-replay-scope constants (step_schema among them)"
    ]


def relative_path_problems(cfg: BuildConfig) -> list[str]:
    """Every path of the build config, and ``MK_STOPWORDS_FILE``, must be absolute: a stage that runs in this process
    would resolve a relative one against THIS process's cwd, a ``mk _stage`` child against ``cfg.root``, and the
    preflight here against the cwd again. ``mk build`` and ``mk append`` make every path absolute; the API takes
    ``overrides`` as given."""
    from dataclasses import fields

    from memory_kernel.stopwords import STOPWORDS_PATH_ENV

    out = []
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        values = value if isinstance(value, tuple) else (value,)
        bad = [str(v) for v in values if isinstance(v, Path) and not v.is_absolute()]
        if bad:
            out.append(f"{f.name} is relative ({', '.join(bad)}): pass an absolute path")
    if (raw := os.environ.get(STOPWORDS_PATH_ENV, "").strip()) and not Path(raw).expanduser().is_absolute():
        out.append(f"{STOPWORDS_PATH_ENV} is relative ({raw}): set an absolute path")
    return out


def genesis_resume_problems(cfg: BuildConfig) -> list[str]:
    """A conversation root resumes with the ``genesis`` it was started with: the flag decides whether the picks check
    accepts a zero, so a resume that flipped it would hold the stages still to run to other rules than the ones
    already done.
    The root's ``build-config.json`` (written at every driver start, before the first stage) says what it was started
    as; a root without one has not started, and a config without the key is not a genesis build. A config that does
    not read refuses only a genesis resume: a real build's checks are the strict ones (and such a config is rewritten
    at the driver start, as before)."""
    recorded = cfg.config_path
    if not recorded.is_file():
        return []
    try:
        doc = json.loads(recorded.read_bytes())
        started = isinstance(doc, dict) and doc.get("genesis", False) is True
    except (OSError, ValueError) as exc:
        if not cfg.genesis:
            return []
        return [
            f"genesis: {recorded} does not read ({type(exc).__name__}), so nothing says this root was started as a "
            "genesis build: start a new conversation root with --genesis"
        ]
    if started == cfg.genesis:
        return []
    flag = "with --genesis" if started else "without --genesis"
    return [
        f"genesis: {cfg.conv_root} was started with genesis={started}: resume it {flag}, or start a new conversation "
        "root"
    ]


def preflight(cfg: BuildConfig, *, secret_sources: Mapping[str, str | None] | None = None) -> list[str]:
    """Every problem that would stop a pending stage later, found now. No OpenRouter key is required up front (a missing
    key fails only on a cache miss). A Voyage source is, for the stages that embed text: a key, or the cache the stage
    reads (``--embedding-cache`` for ``EMBED_STAGES``, ``events_voyage_cache`` for ``events``), checked for presence
    only; a forward append's ``events`` and a segmenter ``boundary`` need the key. The Neo4j password is required for
    the stages that certainly call Neo4j, and for those stages ONE read-only connection checks the server and the group
    (``neo4j_preflight``). ``secret_sources`` (the CLI's field -> source map) only names the source of a rejected
    password in the refusal."""
    problems: list[str] = relative_path_problems(cfg)
    problems += genesis_resume_problems(cfg)
    pending = set(pending_stages(cfg))
    counts: Counts | None = None
    try:
        item, _ = load_conversation(cfg.conversation, cfg.sample_id)  # the counts need it on every build
        if "select" in pending:  # a bad session date is refused here, not inside the select child
            from memory_kernel.pipeline.stages._conversation import check_session_dates

            check_session_dates(item["conversation"])
        # the message count prepares every turn: a bad or backwards turn timestamp is refused here, not in run_build
        counts = conversation_counts(cfg.conversation, cfg.sample_id)
    except ValueError as exc:
        problems.append(str(exc))
    if counts is not None and "boundary" in pending and not path(cfg, "shard").exists():
        try:  # a bootstrap seals its round cap with the shard; one above the integrated cap is refused before any stage
            ledger.round_cap(counts.messages)
        except ValueError as exc:
            problems.append(str(exc))
    if cfg.freeze_manifest is not None and not cfg.freeze_manifest.is_file():
        problems.append(f"freeze manifest {cfg.freeze_manifest} is not a file")
    if pending & FRAME_STAGES:
        problems += frame_dir_problems(cfg, sorted(pending & FRAME_STAGES))
    if "boundary" in pending and cfg.boundary_policy == "segmenter" and not cfg.voyage_api_key:
        problems.append(
            "boundary: --boundary segmenter embeds the messages with Voyage (to cut episodes): set VOYAGE_API_KEY "
            "(it would fail only when the stage runs)"
        )
    if cfg.referent_policy == "structure" and pending & {"referents", "build_dry", "write"}:
        from memory_kernel.grammar.dependencies import wordnet_ready

        if not wordnet_ready():
            problems.append("referents: --referents structure reads WordNet: run mk setup (or --referents phrase)")
    if "boundary" in pending:
        if not path(cfg, "shard").exists():
            try:
                if cfg.grammar_image_sha256 or cfg.recipe_manifest is not None:
                    grammar_image_sha(cfg)
                elif not (image := engine_image(cfg)).is_file():  # hashed at the shard's creation, not here
                    raise ValueError(
                        f"no --grammar-image-sha256 and no --recipe-manifest, and the engine image {image} is not a file"
                    )
            except ValueError as exc:
                problems.append(f"boundary: a new ledger shard needs the G0 image pin: {exc}")
        pair, scope = cfg.boundary_replay_archive, cfg.boundary_replay_scope
        if (pair is None) != (scope is None):
            problems.append(
                "boundary replay gate: --boundary-replay-archive and --boundary-replay-scope go together (the gate is "
                "opt-in)"
            )
        elif pair is not None and scope is not None:
            missing = [str(p) for p in (*pair, scope) if not p.is_file()]
            if missing:
                problems.append(f"boundary replay gate: files missing {missing}")
            else:
                try:
                    scope_doc = load_replay_scope(scope)
                except ValueError as exc:
                    problems.append(f"boundary replay gate: {exc}")
                else:
                    problems += boundary_replay_problems(pair, scope_doc)
    if "capture" in pending:
        if cfg.snapshot_pins is not None and not cfg.snapshot_pins.is_file():
            problems.append(f"capture: snapshot pins {cfg.snapshot_pins} is not a file")
        problems += engine_problems(cfg)
        problems += [f"capture: {text}" for text in parser_problems(cfg)]
    if "build_dry" in pending and cfg.invariants_reference is not None and not cfg.invariants_reference.is_dir():
        problems.append(f"build_dry: invariants reference {cfg.invariants_reference} is not a directory")
    if pending & EMBED_STAGES and cfg.embedding_cache is None and not cfg.voyage_api_key:
        problems.append(
            f"embedding: {[s for s in ORDER if s in pending & EMBED_STAGES]} embed text and there is neither an embedding cache nor a "
            "Voyage key: pass --embedding-cache FILE, or set VOYAGE_API_KEY (the embedder would fail only when the "
            "stage runs, after the paid stages)"
        )
    if "events" in pending and cfg.forward_input is not None and not cfg.voyage_api_key:
        problems.append(
            "events: an append embeds its new session's messages with Voyage (to segment them); the bonfire's cache "
            "never holds a new session's texts: set VOYAGE_API_KEY"
        )
    elif "events" in pending and not cfg.voyage_api_key and not (cache := events_voyage_cache(cfg)).is_file():
        problems.append(
            "events: the stage embeds the conversation's messages with Voyage (to segment them) and there is neither a "
            f"Voyage key nor its cache {cache}: set VOYAGE_API_KEY (--embedding-cache does not serve this stage; it would "
            "fail only when it runs, after the paid stages)"
        )
    if pending - {"select"}:  # every stage after select projects rows through the envelope renderer
        problems += wordnet_problems()
    if pending & NEO4J_STAGES:
        problems += neo4j_preflight(cfg, pending, secret_sources=secret_sources)
        if cfg.embedding_cache is not None and not cfg.embedding_cache.is_file():
            problems.append(f"embedding cache {cfg.embedding_cache} is not a file")
    if pending & RECIPE_STAGES:
        problems += _recipe_problems(cfg)
    return problems


async def _neo4j_probe(cfg: BuildConfig) -> dict[str, Any]:
    """ONE connection. ``verify_connectivity`` runs on the raw driver (``ReadOnlyDriver`` does not wrap it; it raises
    AuthError / ServiceUnavailable), then the group's counts go through the READ-only wrapper, and in append mode the
    group's Saga ingest passes too (``saga_passes``, for ``write.check_owner``). Writes nothing."""
    from memory_kernel.serve.answer import ReadOnlyDriver

    raw = graphdb.driver(cfg)
    try:
        await raw.verify_connectivity()
        counts: dict[str, Any] = await graphdb.group_counts(ReadOnlyDriver(raw), cfg.group)
        if cfg.write_mode == "append":
            counts["saga_passes"] = await graphdb.saga_passes(ReadOnlyDriver(raw), cfg.group)
        return counts
    finally:
        await raw.close()


def neo4j_preflight(
    cfg: BuildConfig, pending: Collection[str], *, secret_sources: Mapping[str, str | None] | None = None
) -> list[str]:
    """Refusals found by ONE read-only Neo4j attempt, or ``[]`` when no Neo4j stage is pending (L-C, L-D).

    A single attempt, never retried (a wrong password trips the server's auth rate limiter). Mapped to refusals:
    ``AuthError``, ``ServiceUnavailable`` / ``OSError`` (unreachable), the ``AuthenticationRateLimit`` client error and
    any other driver error. The group check follows the resume state, so a resumed build is not refused for work it
    already did: a first write still ahead (``write`` pending and no ``summary.json``) needs an EMPTY group; once the
    first write has landed (done, or its summary is there to be adopted) nothing demands emptiness, and a pending
    rewrite / index needs a NON-empty group (sanity's receipt reads no graph: it is not a Neo4j stage). ``write._check_counts`` states both rules. In append mode
    (``mk append``) every Neo4j stage meets the bonfire's existing group: it must be NON-empty and the bonfire's own
    (``write.check_owner``), whether the write is still ahead or not."""
    stages = sorted(set(pending) & NEO4J_STAGES)
    if not stages:
        return []
    if cfg.neo4j_password is None:
        return [f"no Neo4j password (NEO4J_PASSWORD) for {stages}"]
    from neo4j.exceptions import AuthError, DriverError, Neo4jError, ServiceUnavailable

    from memory_kernel.pipeline.stages.write import SUMMARY, Refused, _check_counts, check_owner

    where = f"{cfg.neo4j_uri} as {cfg.neo4j_user!r}"
    try:
        counts = asyncio.run(asyncio.wait_for(_neo4j_probe(cfg), NEO4J_PREFLIGHT_TIMEOUT_S))
    except AuthError:
        source = (secret_sources or {}).get("neo4j_password")
        came_from = f"the password came from {source}" if source else "check NEO4J_PASSWORD"
        return [f"neo4j: authentication failed at {where} ({came_from}; precedence is env, container, env-file)"]
    except Neo4jError as exc:  # after AuthError: it is a ClientError
        code = getattr(exc, "code", None)
        if code == NEO4J_RATE_LIMIT_CODE:
            return [f"neo4j: authentication rate limited at {where} (too many failed logins; wait, no retry made)"]
        return [f"neo4j: refused at {where} ({type(exc).__name__} {code})"]
    except (ServiceUnavailable, OSError) as exc:  # OSError covers TimeoutError
        return [f"neo4j: unreachable at {where} ({type(exc).__name__})"]
    except DriverError as exc:
        return [f"neo4j: driver error at {where} ({type(exc).__name__})"]
    first_write_ahead = "write" in pending and not (path(cfg, "write") / SUMMARY).is_file()
    needs_group = sorted(set(pending) & {"rewrite", "index", "sanity"})
    if cfg.write_mode == "append":  # the run merges into (and then reads) the bonfire's own, non-empty group
        try:
            _check_counts(cfg.group, counts, rewrite=False, append=True)
            check_owner(cfg.group, counts["saga_passes"], cfg.ingest_pass)
        except Refused as exc:
            return [f"neo4j: {exc} (pending: {stages})"]
        return []
    try:
        # the first write MERGEs into an empty group, never a taken one; the later stages read the group it made
        if first_write_ahead:
            _check_counts(cfg.group, counts, rewrite=False)
        elif needs_group:
            _check_counts(cfg.group, counts, rewrite=True)
    except Refused as exc:
        if first_write_ahead:
            return [f"neo4j: {exc}"]
        # _check_counts words the empty-group refusal for a rewrite; index reads the same group (sanity's receipt none)
        detail = str(exc) if "rewrite" in needs_group else f"group {cfg.group!r} is empty; the first write is missing"
        return [f"neo4j: {detail} (pending: {needs_group})"]
    return []


def _recipe_problems(cfg: BuildConfig) -> list[str]:
    if cfg.recipe_manifest is None:
        source, final = "default recipe", default_recipe()
    else:
        source = f"recipe manifest {cfg.recipe_manifest}"
        try:
            doc = json.loads(cfg.recipe_manifest.read_bytes())
        except (OSError, ValueError) as exc:
            return [f"{source}: {type(exc).__name__}"]
        final = (doc.get("final_recipe") if isinstance(doc, dict) else None) or {}
    missing = [] if (final.get("kg_recipe_config") or {}).get("key_render") else ["kg_recipe_config.key_render"]
    missing += [] if final.get("key_recipe_tag") else ["key_recipe_tag"]
    if missing:
        return [f"{source} lacks final_recipe {missing}"]
    # L-B(a): the deixis flag changes the key recipe tag; index refuses a mismatch only after the write has run
    # (sanity's receipt checks the tag the index recorded)
    from memory_kernel.knowledge.temporal_deixis import TEMPORAL_DEIXIS_ENV
    from memory_kernel.search import kg_index

    try:
        got = kg_index.recipe_tag(final["kg_recipe_config"]["key_render"])
    except ValueError as exc:
        return [f"{source}: {exc}"]
    if got != final["key_recipe_tag"]:
        return [
            f"key recipe tag {got!r} differs from the recipe's {final['key_recipe_tag']!r} "
            f"({TEMPORAL_DEIXIS_ENV}={os.environ.get(TEMPORAL_DEIXIS_ENV)!r}); index would refuse it after the write"
        ]
    if cfg.bonfire_index_record is not None:
        return bonfire_recipe_problems(cfg.bonfire_index_record, got)
    return []


def bonfire_recipe_problems(record: Path, got: str) -> list[str]:
    """An append run indexes keys into the bonfire's group, whose stored keys carry the bootstrap's recipe tag (its
    index record, ``BuildConfig.bonfire_index_record``): a different tag (another key render, or another
    ``MK_TEMPORAL_DEIXIS``) would re-embed every key as stale and mix two recipes in one group. Refused here."""
    from memory_kernel.knowledge.temporal_deixis import TEMPORAL_DEIXIS_ENV

    try:
        stored = json.loads(record.read_bytes()).get("recipe_tag")
    except (OSError, ValueError, AttributeError) as exc:
        return [f"bonfire index record {record}: {type(exc).__name__} (the bootstrap's index stage must have run)"]
    if not stored:
        return [f"bonfire index record {record} has no recipe_tag"]
    if stored != got:
        return [
            f"key recipe tag {got!r} differs from the bonfire's {stored!r} (its bootstrap's index record {record}; "
            f"{TEMPORAL_DEIXIS_ENV}={os.environ.get(TEMPORAL_DEIXIS_ENV)!r}): an append must index under the "
            "bootstrap's recipe (set the same deixis flag, or refresh the bonfire)"
        ]
    return []


def refuse_unless_ready(cfg: BuildConfig, *, secret_sources: Mapping[str, str | None] | None = None) -> None:
    problems = preflight(cfg, secret_sources=secret_sources)
    if problems:
        raise Stop("refused before any stage: " + "; ".join(problems))


def write_config(cfg: BuildConfig, *, engine: Mapping[str, Any] | None = None) -> Path:
    """The no-secrets config every stage child reads (overwritten at each driver start; ``logs`` is not sealed).
    ``engine`` (``config.engine_record``: an ``--allow-unpinned-engine`` build) is recorded beside the fields; the
    children do not read it back."""
    p = cfg.config_path
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = cfg.to_json() | ({} if engine is None else {"engine_record": dict(engine)})
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return p


def run_build(
    cfg: BuildConfig,
    *,
    runner: Runner = subprocess.run,
    clock: Callable[[], float] = time.monotonic,
    secret_sources: Mapping[str, str | None] | None = None,
) -> dict:
    """The loop. Returns ``{"status": complete|through|stopped, ...}``. Stops at ``cfg.through`` (default build_dry).

    ``secret_sources`` (the CLI's secret field -> source name map; never values) is logged in ``driver-start``."""
    through = cfg.through or DEFAULT_THROUGH
    refuse_unless_ready(cfg, secret_sources=secret_sources)  # before ANY mkdir: a refused build leaves no workdir
    counts = conversation_counts(cfg.conversation, cfg.sample_id)
    cfg.ledger.parent.mkdir(parents=True, exist_ok=True)
    cfg.ledger.touch(exist_ok=True)  # a gate's ledger read never crashes on a build that sent nothing
    lock_fd = acquire_lock(cfg)  # held (with every child) for the life of the loop
    try:
        engine = engine_record(cfg)  # None unless --allow-unpinned-engine
        write_config(cfg, engine=engine)
        forced = os.environ.get(inprocess.STAGE_CHILDREN_ENV) == "1"
        # one build at a time runs stages in this process; a build that finds it busy runs every stage as a child
        with contextlib.nullcontext(False) if forced else inprocess.build_lock() as in_process:
            if forced:
                log_event(cfg, "stage-children", reason=inprocess.STAGE_CHILDREN_ENV)
            elif not in_process:
                log_event(cfg, "in-process-busy")
            # only the lock holder touches process-wide caches: a busy build's would clear them under a running stage
            with inprocess.build_data(cfg) if in_process else contextlib.nullcontext():
                return _loop(
                    cfg,
                    counts,
                    through=through,
                    runner=runner,
                    clock=clock,
                    lock_fd=lock_fd,
                    secret_sources=secret_sources,
                    engine=engine,
                    in_process=in_process,
                )
    finally:
        os.close(lock_fd)


def _loop(
    cfg: BuildConfig,
    counts: Counts,
    *,
    through: str,
    runner: Runner,
    clock: Callable[[], float],
    lock_fd: int,
    secret_sources: Mapping[str, str | None] | None = None,
    engine: Mapping[str, Any] | None = None,
    in_process: bool = False,
) -> dict:
    log_event(
        cfg,
        "driver-start",
        through=through,
        pid=os.getpid(),
        jev_concurrency=cfg.jev_concurrency,
        ramped_down=ramped_down(cfg),
        recipe=recipe_record(cfg, code_census=write_code_census(cfg)),
        secret_sources=None if secret_sources is None else dict(secret_sources),  # field -> source name, never a value
        **({} if engine is None else {"engine": dict(engine)}),  # only an --allow-unpinned-engine build records it
    )
    for stage in ORDER:
        if marker(cfg, stage).is_file():
            log_event(cfg, "skip", stage, reason="done marker")
        else:
            reruns = 0
            while True:
                before, t0 = conv_spent(cfg), clock()
                failures = failure_files(cfg)
                try:
                    body = run_stage(cfg, counts, stage, runner=runner, lock_fd=lock_fd, in_process=in_process)
                    break
                except Exception as exc:  # Stop, and any parent-side failure: the attempt's spend is always logged
                    reason = str(exc) if isinstance(exc, Stop) else f"driver error {type(exc).__name__}: {exc}"
                    usd = _spent_since(cfg, before)
                    log_event(cfg, "stop", stage, reason=reason, usd=usd, wall_s=round(clock() - t0, 3))
                    try:
                        ramp_if_transport(cfg, stage, failures)
                        transient = transient_only(cfg, failures)
                    except Exception as ramp_exc:  # noqa: BLE001 -- a scan error must not hide the stop
                        log_event(cfg, "ramp-scan-error", stage, error=f"{type(ramp_exc).__name__}: {ramp_exc}")
                        transient = False
                    if not transient or reruns >= TRANSIENT_RERUNS:
                        return {"status": "stopped", "stage": stage, "reason": reason}
                    reruns += 1
                    log_event(cfg, "transient-rerun", stage, attempt=reruns, of=TRANSIENT_RERUNS)
            done = {
                **body,
                "usd": round(conv_spent(cfg) - before, 9),
                "wall_s": round(clock() - t0, 3),
                "finished_at": now_iso(),
            }
            m = marker(cfg, stage)
            m.parent.mkdir(parents=True, exist_ok=True)
            with m.open("x") as handle:
                handle.write(json.dumps(done, sort_keys=True, indent=1, default=str))
            log_event(cfg, "done", stage, usd=done["usd"], wall_s=done["wall_s"], gate=done["gate"])
        if stage == through:
            log_event(cfg, "stop", stage, reason=f"--through {through}", usd=0.0)
            return {"status": "through", "stage": stage}
    log_event(cfg, "driver-complete")
    return {"status": "complete"}


# -- status ----------------------------------------------------------------------------------------------------------


def status_at(conv_root: Path, ledger: Path) -> dict:
    """Read-only: per-stage done|started|pending, ledger paid, the current stage and the last stop."""
    probe = _Probe(conv_root)
    stages: dict[str, str] = {}
    for stage in ORDER:
        if marker(probe, stage).is_file():  # type: ignore[arg-type]
            stages[stage] = "done"
        elif any(p.exists() for p in evidence(probe, stage)):  # type: ignore[arg-type]
            stages[stage] = "started"
        else:
            stages[stage] = "pending"
    events = read_events(probe)  # type: ignore[arg-type]
    current = None
    for e in events:
        if e["event"] == "start":
            current = e["stage"]
        elif e["event"] in ("done", "stop") and e["stage"] == current:
            current = None
    last_stop = next((e for e in reversed(events) if e["event"] == "stop"), None)
    summary = provenance.ledger_summary(ledger)
    label = conv_root.name
    rows = provenance.ledger_rows(ledger)
    return {
        "conv": label,
        "conv_root": str(conv_root),
        "stages": stages,
        "ledger": str(ledger),
        "ledger_paid_usd": summary["paid_usd"],
        "conv_spent_usd": round(sum(float(r.get("paid_usd") or 0.0) for r in rows if r.get("conv") == label), 9),
        "current_stage": current,
        "last_stop": last_stop,
        "lock_held": lock_held(probe),  # type: ignore[arg-type]
    }


@dataclass(frozen=True)
class _Probe:
    """Just enough of a BuildConfig for the path helpers (status of a conversation root without its config)."""

    conv_root: Path

    @property
    def sample_id(self) -> str:
        return self.conv_root.name


def status(cfg: BuildConfig) -> dict:
    return status_at(cfg.conv_root, cfg.ledger) | {"cap_usd": cfg.cap_usd, "group": cfg.group}
