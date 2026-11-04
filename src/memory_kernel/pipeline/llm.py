"""LLM plumbing for the build: the Jev sender, replay refusal, twin re-ask batching, the stage meter.

Ported from the reference harness:
  * ``JevSender`` (+ ``ProviderTransportError``, ``_RequestTrace``, ``_causes``, ``_connect_reason``, ``canonical``)
    <- the harness's provider ``evaluate`` / ``_client`` (an injected object that owns its
    httpx client, instead of a module loaded by path into a global on the first cache miss)
  * ``lost_wires`` / ``QuestionBatcher``  <- the harness's re-ask module (an explicit batcher instead
    of rebinding ``partition_question_wire`` in every loaded module)
  * ``Meter`` / ``StopSpend`` / ``STAGE_CAPS``  <- the harness's Jev and event stage modules (no defaults)

It adds the objects the ported stages use: ``metered`` / ``guard`` (the stage meter), ``Serial`` /
``Pool`` / ``SpendCapGate`` / ``LedgerView`` / ``Prefetcher`` (the parallel Jev prefetch of the harness; an explicit object handed to the stage, no patching of ``RoundBudget`` / ``Meter``) and ``OpenRouterChat``
(one OpenRouter chat completion, with the key passed in). Kernel 1.8 archived ``GapSender`` with the question layer.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import ssl
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_kernel.pipeline.ledger import RoundBudget, Shard
    from memory_kernel.pipeline.runner import Ctx

MAX_CONCURRENCY = 8  # = the httpx Limits(max_connections=8) below
JEV_ENDPOINTS = {"openrouter": "https://openrouter.ai/api/alpha/decisions"}

#: The event stage caps (kernel 1.8: the taxonomy layer's Jev, question and remine caps left with it).
STAGE_CAPS: dict[str, float] = {"events": 0.10, "ctx": 0.35}
#: The reference conversation every cap was measured on: the scale of a bootstrap's round cap (``ledger.round_cap``) and
#: of the recorded stage caps. An append's budget does not scale with it (``bonfire.append_budget``).
REF_MESSAGES = 369


def cap_scale(messages: int) -> float:
    """1.0 up to the reference conversation's size, then proportional to the message count."""
    return max(1.0, messages / REF_MESSAGES)


def stage_cap(stage: str, messages: int) -> float:
    return STAGE_CAPS[stage] * cap_scale(messages)


# -- the Jev sender ---------------------------------------------------------------------------------------------------


class ProviderTransportError(RuntimeError):
    """A transport failure with an explicit HTTP-send boundary."""

    def __init__(self, *, phase: str, reason: str) -> None:
        if phase not in {"connect", "outcome_unknown"}:
            raise ValueError("invalid provider transport phase")
        self.phase = phase
        self.reason = reason
        super().__init__(f"provider transport failed: {phase}/{reason}")


class _RequestTrace:
    def __init__(self) -> None:
        self.connect_failure: str | None = None
        self.connect_causes: list[BaseException] = []
        self.http_send_started = False

    def __call__(self, event: str, info: dict[str, Any]) -> None:
        if event.endswith(("send_request_headers.started", "send_request_body.started")):
            self.http_send_started = True
        if event.endswith("connect_tcp.failed"):
            self.connect_failure = "tcp"
            self.connect_causes = _causes(info["exception"])
        elif event.endswith("start_tls.failed"):
            self.connect_failure = "tls"
            self.connect_causes = _causes(info["exception"])


def _causes(exc: BaseException) -> list[BaseException]:
    pending = [exc]
    seen: set[int] = set()
    chain: list[BaseException] = []
    while pending and len(chain) < 16:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return chain if not pending else []


def _connect_reason(exc: BaseException, trace: _RequestTrace, httpx: Any) -> str | None:
    if trace.http_send_started or trace.connect_failure is None:
        return None
    if not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return None
    chain = trace.connect_causes
    if not chain or any(isinstance(item, (ssl.SSLCertVerificationError, httpx.ProxyError)) for item in chain):
        return None
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout"
    if any(isinstance(item, socket.gaierror) for item in chain):
        return "dns_resolution" if trace.connect_failure == "tcp" else None
    if any(isinstance(item, ssl.SSLEOFError) for item in chain):
        return "tls_eof" if trace.connect_failure == "tls" else None
    transient_errnos = {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETUNREACH,
    }
    if any(isinstance(item, OSError) and item.errno in transient_errnos for item in chain):
        return "connection_failure" if trace.connect_failure == "tcp" else None
    return None


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def request_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


class JevSender:
    """The Jev HTTP sender (the harness provider's ``evaluate``), one httpx client per sender and process.

    Called exactly like the frozen ``evaluate`` (``sender(payload, provider=..., api_key=..., timeout=...)``) so a
    ``RoundBudget`` can take it as its ``sender``. ``api_key`` passed to the call wins over the constructor's."""

    def __init__(
        self,
        *,
        api_key: str | None,
        endpoints: dict[str, str] | None = None,
        max_connections: int = MAX_CONCURRENCY,
    ) -> None:
        self.api_key = api_key
        self.endpoints = dict(JEV_ENDPOINTS if endpoints is None else endpoints)
        self.max_connections = max_connections
        self._client_obj: Any | None = None
        self._client_pid: int | None = None
        self._lock = Lock()

    def _client(self, httpx: Any) -> Any:
        with self._lock:
            if self._client_obj is None or self._client_pid != os.getpid():
                self._client_obj = httpx.Client(
                    http1=True,
                    http2=False,
                    trust_env=True,
                    follow_redirects=False,
                    limits=httpx.Limits(
                        max_connections=self.max_connections, max_keepalive_connections=4, keepalive_expiry=120.0
                    ),
                )
                self._client_pid = os.getpid()
            return self._client_obj

    def close(self) -> None:
        with self._lock:
            if self._client_obj is not None and self._client_pid == os.getpid():
                self._client_obj.close()
            self._client_obj = None
            self._client_pid = None

    def __call__(
        self,
        payload: dict[str, Any],
        *,
        provider: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        on_response: Callable[[bytes, Any | None, float], None] | None = None,
        on_http_error: Callable[[int, bytes], None] | None = None,
    ) -> tuple[dict[str, Any], float]:
        key = api_key or self.api_key
        if not key:
            raise RuntimeError("JevSender: no API key configured")
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise RuntimeError("provider transport requires httpx in the runner environment") from exc
        endpoint = self.endpoints[provider]
        body = canonical(payload)
        client = self._client(httpx)
        trace = _RequestTrace()
        request = client.build_request(
            "POST",
            endpoint,
            content=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            timeout=httpx.Timeout(
                connect=min(timeout, 15.0), read=timeout, write=min(timeout, 30.0), pool=min(timeout, 15.0)
            ),
            extensions={"trace": trace},
        )
        started = time.perf_counter()
        try:
            response = client.send(request, stream=True)
            try:
                if not 200 <= response.status_code < 300:
                    if on_http_error is not None:
                        on_http_error(response.status_code, b"".join(response.iter_raw()))
                    raise RuntimeError(f"provider request failed: HTTP {response.status_code}")
                raw = b"".join(response.iter_raw())
            finally:
                response.close()
        except httpx.TransportError as exc:
            reason = _connect_reason(exc, trace, httpx)
            raise ProviderTransportError(
                phase="connect" if reason else "outcome_unknown",
                reason=reason or type(exc).__name__,
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if on_response is not None:
                on_response(raw, None, latency_ms)
            raise RuntimeError("provider returned invalid JSON") from exc
        if on_response is not None:
            on_response(raw, decoded, latency_ms)
        if not isinstance(decoded, dict) or not isinstance(decoded.get("answers"), dict):
            raise RuntimeError("provider response missing answers map")
        if not isinstance(decoded.get("model"), str) or not isinstance(decoded.get("usage"), dict):
            raise RuntimeError("provider response missing model or usage")
        return decoded, latency_ms


class ReplayMiss(RuntimeError):
    """Test seam (``tests/contract/test_provider_replay.py``; no driver code uses it): a wire was not in the cache, and
    a replay-only sender never sends."""

    def __init__(self, sha: str) -> None:
        self.sha = sha
        super().__init__(f"replay miss: wire {sha} is not archived (replay mode never sends)")


class RefusingSender:
    """Test seam: a sender that refuses every call with ``ReplayMiss`` and records the wire sha, so a test can prove a
    replay answers from the cache alone. The driver never installs it (a live build sends through ``JevSender``)."""

    def __init__(self) -> None:
        self.misses: list[str] = []

    def __call__(self, payload: dict[str, Any], **_: Any) -> tuple[dict[str, Any], float]:
        sha = request_digest(payload)
        self.misses.append(sha)
        raise ReplayMiss(sha)

    def close(self) -> None:
        return None


# -- twin re-ask ---------------------------------------------------------------------------------------

MAX_TWIN_ROUNDS = 3  # a twin that is itself lost moves to the next round, at most this many times


def lost_wires(roots: Iterable[Path]) -> frozenset[str]:
    """Shas of shard attempts that failed with no archive (``budget/jev-<sha>/failure.json``, no ``archive.json``)."""
    out: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for failure in root.glob("**/budget/jev-*/failure.json"):
            if not (failure.parent / "archive.json").exists():
                out.add(failure.parent.name.removeprefix("jev-"))
    return frozenset(out)


class QuestionBatcher:
    """``partition_question_wire`` with the standing twin re-ask.

    A compiled batch whose sha is lost is replaced by its twin from ``round + k`` (k = 1..``max_twin_rounds``); every
    replacement is appended to ``twins``. With nothing lost, ``partition`` returns the frozen function's result object
    unchanged (same as the harness's twin re-ask when ``not lost``)."""

    def __init__(self, lost: frozenset[str] = frozenset(), *, max_twin_rounds: int = MAX_TWIN_ROUNDS) -> None:
        self.lost = frozenset(lost)
        self.max_twin_rounds = max_twin_rounds
        self.twins: list[dict[str, Any]] = []

    def partition(self, wire: Any, policy: Any, *, round: int) -> list:  # noqa: A002 -- the frozen keyword
        from memory_kernel.construction.memory_discourse_question_batches import partition_question_wire

        batches = partition_question_wire(wire, policy, round=round)
        if not self.lost:
            return batches
        from memory_kernel.construction.mined_affordance_contract import content_sha256

        idx = [i for i, b in enumerate(batches) if content_sha256(b) in self.lost]
        if not idx:
            return batches
        out = list(batches)
        for i in idx:
            for k in range(1, self.max_twin_rounds + 1):
                twin = partition_question_wire(wire, policy, round=round + k)
                if len(twin) != len(batches):
                    raise ValueError(f"twin partition (round {round + k}) has {len(twin)} batches, not {len(batches)}")
                if content_sha256(twin[i]) not in self.lost:
                    self.twins.append(
                        {"lost": content_sha256(batches[i]), "twin": content_sha256(twin[i]), "round": round + k}
                    )
                    out[i] = twin[i]
                    break
            else:
                raise ValueError(
                    f"lost wire {content_sha256(batches[i])}: every twin up to round +{self.max_twin_rounds} lost"
                )
        return out


# -- the stage meter (the harness's Jev stage module) -----------------------------------------------------------------


class StopSpend(RuntimeError):
    """Reservation refused: the unit cap (``--cap-usd``) would be crossed. A stage cap is recorded, never a stop."""


def stage_cap_record(cfg: Any, stage: str, cap_usd: Any, spend_usd: Any, **extra: Any) -> dict:
    """A stage that went past its stage cap: the fields its plan / summary / result records (``stage_cap_usd``, the
    spend or estimate, ``stage_cap_exceeded``) and ONE ``stage-cap-exceeded`` event in the build's event log. The
    stage goes on: ``--cap-usd`` (the unit cap) is the one spend stop."""
    from memory_kernel.pipeline.runner import log_event

    record = {"stage_cap_usd": str(cap_usd), "stage_cap_spend_usd": str(spend_usd), "stage_cap_exceeded": True}
    record |= {k: str(v) for k, v in extra.items()}
    log_event(cfg, "stage-cap-exceeded", stage, **record)
    return record


def unit_guard(cfg: Any, usage: Mapping, stage_cap: float) -> Callable[[float], None]:
    """The ``reserve_guard`` of a stage the ``Meter`` does not guard (boundary, picks, referents). Its stage cap is
    recorded, never a stop; once its spend (``usage["usd"]``, filled as sends complete) is past ``stage_cap``, a hold
    that would take the build's paid total (the cost ledger's, read then: the stage's own row is written when it
    ends, so it is the total at stage start) plus the spend over ``cfg.cap_usd`` (``--cap-usd``) is refused. Under the
    stage cap nothing is read or refused: the stage's ``cap_check`` (paid + stage cap <= unit cap) holds the unit cap."""
    from memory_kernel.pipeline import provenance

    paid: list[float] = []

    def guard(amount: float) -> None:
        spent = float(usage.get("usd", 0.0))
        if spent <= stage_cap:
            return
        if not paid:
            paid.append(provenance.ledger_summary(cfg.ledger)["paid_usd"])
        if paid[0] + spent + amount > cfg.cap_usd:
            raise StopSpend(f"unit cap {cfg.cap_usd} would be crossed (cumulative {paid[0] + spent:.4f})")

    return guard


class Meter:
    """Spend of one stage = sum of ``receipt.cost_usd`` over budget dirs that completed after the meter was made.

    No defaults: the caller passes ``unit_spent`` (the programme ledger's paid total) and ``unit_cap`` (the build's
    ``cap_usd``) explicitly. ``check`` refuses a hold past the unit cap only; the stage cap is recorded
    (``cap_record``), never a stop."""

    def __init__(self, budget: Path, stage_cap: float, *, unit_spent: float, unit_cap: float) -> None:
        self.budget, self.stage_cap, self.unit_cap = budget, stage_cap, unit_cap
        self.unit_spent = unit_spent
        self.seen = self._completed()
        self.rows: dict[str, dict] = {}
        self.cost: Any = None
        self.pool: Any = None  # set by ``serialised`` for the duration of a prefetch

    def _completed(self) -> set[str]:
        return {p.parent.name for p in self.budget.glob("*/completed.json")}

    def refresh(self) -> None:
        for p in self.budget.glob("*/completed.json"):
            name = p.parent.name
            if name not in self.seen and name not in self.rows:
                self.rows[name] = json.loads(p.read_bytes())["receipt"]

    def _spent(self) -> float:
        self.refresh()
        return sum(float(r["cost_usd"]) for r in self.rows.values())

    def spent(self) -> float:
        pool = self.pool
        if pool is not None and pool.owns_thread():  # a prefetch worker: no per-call glob of the shard
            return pool.meter_spent(self, Meter._spent)
        return self._spent()

    def would_cross(self, reserve_usd: float) -> str | None:
        """The cap a hold of ``reserve_usd`` would cross (the stage cap first, then the unit cap), else None. A
        predicate (``cap_could_bind``): it records and refuses nothing."""
        spent = self.spent()
        if spent + reserve_usd > self.stage_cap:
            return f"stage cap {self.stage_cap:.4f} would be crossed (spent {spent:.6f}, next {reserve_usd:.6f})"
        if self.unit_spent + spent + reserve_usd > self.unit_cap:
            return f"unit cap {self.unit_cap} would be crossed (cumulative {self.unit_spent + spent:.4f})"
        return None

    def check(self, reserve_usd: float) -> None:
        """Refuse a hold that would cross the unit cap. A hold past the stage cap goes on (``cap_record``)."""
        spent = self.spent()
        if self.unit_spent + spent + reserve_usd > self.unit_cap:
            raise StopSpend(f"unit cap {self.unit_cap} would be crossed (cumulative {self.unit_spent + spent:.4f})")

    def cap_record(self) -> dict:
        """``{}`` while the stage's spend is within its stage cap, else what the stage records: ``stage_cap_usd``,
        ``stage_cap_spend_usd`` and ``stage_cap_exceeded``. (While the stage cap was a stop, a hold never took the
        spend past it, so a run within the cap records exactly what it did before.)"""
        spent = self.spent()
        if spent <= self.stage_cap:
            return {}
        return {"stage_cap_usd": str(self.stage_cap), "stage_cap_spend_usd": str(spent), "stage_cap_exceeded": True}

    def usage(self) -> dict[str, dict]:
        self.refresh()
        out: dict[str, dict] = {}
        for name, r in self.rows.items():
            provider = "jev" if name.startswith("jev-") else "openrouter"
            u = out.setdefault(provider, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
            u["requests"] += 1
            u["input_tokens"] += int(r["input_tokens"])
            u["output_tokens"] += int(r["output_tokens"])
            u["usd"] += float(r["cost_usd"])
        return out


# -- the metered stage (the harness's Jev stage module) --------------------------------------------------------------


@contextmanager
def metered(
    ctx: Ctx, stage: str, run: str, shard: Shard, *, note: str, stage_cap_override: float | None = None
) -> Iterator[Meter]:
    """One stage's meter and ONE cost-ledger row.

    The harness's ``metered`` rebound ``RoundBudget.reserve`` to a meter-guarded version for the block and restored
    it; here the guard is the meter itself, handed to the ledger as ``shard.with_hooks(reserve_guard=guard(meter))``.
    The programme paid total and the cap are read from the config (no module defaults). The row is written on exit
    with what the shard's receipts show, as before. A stage whose spend went past its stage cap logs one
    ``stage-cap-exceeded`` event on exit; the stage adds ``meter.cap_record()`` to its result."""
    from memory_kernel.pipeline import provenance
    from memory_kernel.pipeline.runner import log_event

    cap = stage_cap(stage, ctx.messages) if stage_cap_override is None else stage_cap_override
    unit_spent = provenance.ledger_summary(ctx.cfg.ledger)["paid_usd"]
    meter = Meter(shard.budget, cap, unit_spent=unit_spent, unit_cap=ctx.cfg.cap_usd)
    with ctx.cost(stage, run=run, note=note) as cost:
        meter.cost = cost  # callers may add non-Jev providers (voyage) to the same ledger row
        try:
            yield meter
        finally:
            for provider, u in meter.usage().items():
                cost.add(provider, **u)
            if record := meter.cap_record():
                log_event(ctx.cfg, "stage-cap-exceeded", stage, **record)


def guard(meter: Meter) -> Callable[[float], None]:
    """The ``reserve_guard`` of a ledger: refuse a hold that would cross the unit cap (``Meter.check``)."""
    return meter.check


# -- parallel Jev sending ---------------------------------------------------------------------------
#
# A paid stage's runner sends one wire at a time through ``RoundBudget.evaluate``; each wire
# is cached by content sha in the shard's ``budget/`` dir and a re-ask replays ``archive.json`` at $0. The stage
# calls ``Prefetcher.jev`` with the wire list it is about to send sequentially (it HAS the arguments, so
# the harness's argument capture and first-send hook are not needed); the pool sends the uncached wires with N
# threads through the ledger's own ``evaluate``, and the sequential loop then replays every wire from the cache.
#
# ``evaluate`` is not safe for concurrent callers (``reserve`` is check-then-act; ``summary`` reads files that
# ``write_new`` may be half-way through writing). During a prefetch only, the ledger instance routes its pool
# threads' ``reserve`` / ``complete`` / ``summary`` through ``Serial`` (one lock, reservations in wire order); the
# HTTP call runs outside it. The files are written by the ledger's own code, byte-identical to the sequential path.
# A refusal (cap, gate, a pre-reservation check) is SOFT: stop submitting, drain, let the sequential loop go on and
# refuse at the same wire if the refusal was real. A failure after a reservation (lost wire, ``outcome_unknown``, a
# bad archive) is HARD: stop, drain, re-raise; nothing is re-asked.

_TL = threading.local()


class Refused(RuntimeError):
    """A prefetch-only gate refused a wire (soft: the sequential runner decides)."""


class _Aborted(Exception):
    """A worker's wire was never reserved because the prefetch stopped first."""


def _sha(wire: Mapping) -> str:
    from memory_kernel.construction.mined_affordance_contract import content_sha256

    return content_sha256(wire)


@dataclass(frozen=True)
class Item:
    sha: str
    payload: Any


def plan_items(budget_dir: Path, payloads: Iterable[Any], *, kind: str) -> tuple[list[Item], dict]:
    """Ordered, deduplicated, uncached wires, truncated where the sequential runner would raise.

    ``kind`` ``jev``: cached = ``jev-<sha>/archive.json``. A dir that
    exists without it is an attempt the runner refuses to retry: nothing at or after it is prefetched."""
    seen: set[str] = set()
    items: list[Item] = []
    stats = {"wires": 0, "unique": 0, "cached": 0, "truncated_at": None}
    done = "archive.json"
    for payload in payloads:
        stats["wires"] += 1
        sha = _sha(payload)
        if sha in seen:
            continue
        seen.add(sha)
        stats["unique"] += 1
        d = budget_dir / f"{kind}-{sha}"
        if d.exists():
            if (d / done).exists():
                stats["cached"] += 1
                continue
            stats["truncated_at"] = d.name
            break
        items.append(Item(sha, payload))
    return items, stats


class SpendCapGate:
    """ctx's stop: before an uncached wire, the running receipt sum of every PRIOR wire must be < cap.

    Prior cached wires count at their receipt; in-flight ones at their reservation until their receipt lands."""

    def __init__(self, order: list[str], budget_dir: Path, cap: Decimal):
        self.order, self.cap = order, Decimal(str(cap))
        self.position: dict[str, int] = {}
        for i, sha in enumerate(order):
            self.position.setdefault(sha, i)
        self.cost: dict[str, Decimal] = {}
        for sha in set(order):
            done = budget_dir / f"jev-{sha}" / "completed.json"
            if done.exists():
                self.cost[sha] = Decimal(str(json.loads(done.read_bytes())["receipt"]["cost_usd"]))

    def check(self, sha: str, amount: Decimal) -> None:  # noqa: ARG002 (the rule ignores the next wire's hold)
        total = Decimal(0)
        for prior in self.order[: self.position[sha]]:
            if prior not in self.cost:
                raise Refused(f"prior wire {prior[:12]} has no receipt or hold")
            total += self.cost[prior]
        if total >= self.cap:
            raise Refused(f"running receipt sum {total} reached the cap {self.cap}")

    def reserved(self, sha: str, amount: Decimal) -> None:
        self.cost[sha] = amount

    def completed(self, sha: str, cost: Decimal) -> None:
        self.cost[sha] = cost


class LedgerView:
    """``RoundBudget.summary()`` kept incrementally during a prefetch (the ledger's own globs and parses every
    ``reservation.json`` / ``completed.json`` per reservation: ~70 ms per 1,000 attempts, serialised under the lock).

    Exact by construction: the base is the ledger's ``summary()`` at prefetch start; the ledger instance holds the
    shard's exclusive ``execution.lock``, so the only writers until the prefetch ends are this pool's reserve /
    complete, which update the view with the same ``Decimal(str(float))`` terms the summary adds. Verified against
    the ledger's ``summary()`` at the end of every prefetch (``verify``); a mismatch raises."""

    KEYS = ("reserved_usd", "actual_usd", "held_or_spent_usd", "unresolved_reservations_usd")

    def __init__(self, base: Mapping):
        from memory_kernel.pipeline.ledger import LEGACY_INTEGRATED_CAP

        self.cap = Decimal(base["cap_usd"])  # the round's sealed cap or an append's limit, as the summary reports it
        # the shard's own recorded integrated cap (a shard made before 1.1 has $10); None under an append's limit
        integrated = base.get("integrated_cap_usd", LEGACY_INTEGRATED_CAP)
        self.integrated_cap = None if integrated is None else Decimal(integrated)
        self.reserved = Decimal(base["reserved_usd"])
        self.actual = Decimal(base["actual_usd"])
        self.unresolved = Decimal(base["unresolved_reservations_usd"])
        self.by_stage = {k: Decimal(v) for k, v in base["held_or_spent_by_stage"].items()}
        self.calls, self.completed = base["calls"], base["completed"]

    def reserved_row(self, row: Mapping) -> None:
        amount = Decimal(str(row["reserved_usd"]))
        self.reserved += amount
        self.unresolved += amount
        self.by_stage[row["stage"]] = self.by_stage.get(row["stage"], Decimal(0)) + amount
        self.calls += 1

    def completed_row(self, done: Mapping) -> None:
        amount, cost = Decimal(str(done["reserved_usd"])), Decimal(str(done["receipt"]["cost_usd"]))
        self.actual += cost
        self.unresolved -= amount
        self.by_stage[done["stage"]] += cost - amount
        self.completed += 1

    def summary(self) -> dict:
        """The ledger's ``summary()`` as kept here: read, never held against a cap (the ledger's ``reserve`` holds a new
        hold against them, over this held total)."""
        held = self.actual + self.unresolved
        return {
            "cap_usd": str(self.cap),
            "reserved_usd": str(self.reserved),
            "actual_usd": str(self.actual),
            "held_or_spent_usd": str(held),
            "unresolved_reservations_usd": str(self.unresolved),
            "held_or_spent_by_stage": {k: str(v) for k, v in self.by_stage.items()},
            "calls": self.calls,
            "completed": self.completed,
            "unknown_or_failed": self.calls - self.completed,
        }

    def verify(self, frozen: Mapping) -> None:
        mine = self.summary()
        same = all(Decimal(mine[k]) == Decimal(frozen[k]) for k in self.KEYS) and all(
            mine[k] == frozen[k] for k in ("calls", "completed", "unknown_or_failed")
        )
        stages = set(mine["held_or_spent_by_stage"]) | set(frozen["held_or_spent_by_stage"])
        same = same and all(
            Decimal(mine["held_or_spent_by_stage"].get(k, "0")) == Decimal(frozen["held_or_spent_by_stage"].get(k, "0"))
            for k in stages
        )
        if not same:
            raise RuntimeError(f"prefetch ledger view {mine} != ledger summary {frozen}")


class Pool:
    """N workers over ordered items; reservations in item order under one lock; sends in parallel."""

    def __init__(
        self,
        n: int,
        send_one: Callable[[Item], Any],
        gate: SpendCapGate | None = None,
        *,
        budget: Any = None,
        view: LedgerView | None = None,
    ):
        self.n, self.send_one, self.gate, self.budget, self.view = n, send_one, gate, budget, view
        self.costs: list[float] = []  # receipts completed by this pool, in completion order
        self.meter_base: dict[int, tuple[float, int]] = {}
        self.cond = threading.Condition(threading.Lock())
        self.inflight: dict[str, Decimal] = {}
        self.items: list[Item] = []
        self.next = self.ticket = 0
        self.limit = sys.maxsize  # soft stop: no ticket >= limit is reserved
        self.hard = False
        self.sent = 0
        self.refusals: list[dict] = []
        self.failures: list[tuple[int, str, BaseException]] = []

    def owns_thread(self) -> bool:
        return getattr(_TL, "pool", None) is self

    # called by the ledger's ``Serial`` routes, on a worker thread

    def reserve(self, original: Callable, kwargs: dict) -> Any:
        i, sha = _TL.index, _TL.sha
        with self.cond:
            self.cond.wait_for(lambda: self.ticket == i or i >= self.limit or self.hard)
            if i >= self.limit or self.hard:
                raise _Aborted
            try:
                amount = (
                    Decimal(kwargs["input_cap"]) * kwargs["input_rate"]
                    + Decimal(kwargs["output_cap"]) * kwargs["output_rate"]
                ) / 1_000_000
                if self.gate is not None:
                    self.gate.check(sha, amount)
                out = original(**kwargs)
                if self.view is not None:
                    self.view.reserved_row(out[1])
            except BaseException:
                self.limit = min(self.limit, i)
                self.ticket = i + 1
                self.cond.notify_all()
                raise
            self.inflight[sha] = amount
            if self.gate is not None:
                self.gate.reserved(sha, amount)
            _TL.reserved = True
            self.ticket = i + 1
            self.cond.notify_all()
            return out

    def complete(self, original: Callable, args: tuple, kwargs: dict) -> Any:
        with self.cond:
            out = original(*args, **kwargs)
            if self.view is not None:
                self.view.completed_row(out)
            self.costs.append(float(out["receipt"]["cost_usd"]))
            self.inflight.pop(_TL.sha, None)
            if self.gate is not None:
                self.gate.completed(_TL.sha, Decimal(str(out["receipt"]["cost_usd"])))
            return out

    def inflight_usd(self) -> float:
        return float(sum(self.inflight.values(), Decimal(0)))

    def meter_spent(self, meter: Any, frozen_spent: Callable[[Any], float]) -> float:
        """``Meter.spent()`` without its per-call glob: the real value once, plus this pool's later receipts."""
        if id(meter) not in self.meter_base:
            self.meter_base[id(meter)] = (frozen_spent(meter), len(self.costs))
        base, start = self.meter_base[id(meter)]
        return base + sum(self.costs[start:])

    # the workers

    def _worker(self) -> None:
        _TL.pool = self
        try:
            while True:
                with self.cond:
                    if self.hard or self.next >= min(len(self.items), self.limit):
                        return
                    i = self.next
                    self.next += 1
                item = self.items[i]
                _TL.index, _TL.sha, _TL.reserved = i, item.sha, False
                try:
                    self.send_one(item)
                    with self.cond:
                        self.sent += 1
                except _Aborted:
                    return
                except BaseException as exc:  # noqa: BLE001 - classified, recorded, re-raised by run()
                    with self.cond:
                        if _TL.reserved:
                            self.hard = True
                            self.failures.append((i, item.sha, exc))
                        else:
                            self.limit = min(self.limit, i)
                            self.refusals.append({"index": i, "sha": item.sha, "error": f"{type(exc).__name__}: {exc}"})
                            if self.ticket == i:
                                self.ticket = i + 1
                        self.cond.notify_all()
        finally:
            _TL.pool = None

    def run(self, items: list[Item]) -> dict:
        self.items = items
        threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(min(self.n, len(items)))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        report = {
            "concurrency": self.n,
            "items": len(items),
            "sent": self.sent,
            "refused": self.refusals[:1],
            "failed": [{"index": i, "sha": s, "error": f"{type(e).__name__}: {e}"} for i, s, e in self.failures],
            "inflight_after": len(self.inflight),
        }
        if self.failures:  # the ledger's own exception, as a sequential run would raise it; nothing re-asked
            print(json.dumps({"prefetch_failed": report}, default=str), file=sys.stderr, flush=True)
            raise min(self.failures, key=lambda f: f[0])[2]
        return report


class Serial:
    """What a ``RoundBudget`` consults (``budget.serial``) while a pool is prefetching: the pool's worker threads go
    through the pool's lock and tickets, every other caller gets the ledger's own method."""

    def __init__(self, pool: Pool):
        self.pool = pool

    def owns_thread(self) -> bool:
        return self.pool.owns_thread()

    def reserve(self, original: Callable, kwargs: dict) -> Any:
        return self.pool.reserve(original, kwargs)

    def complete(self, original: Callable, *args: Any, **kwargs: Any) -> Any:
        return self.pool.complete(original, args, kwargs)

    def summary(self, original: Callable) -> dict:
        view = self.pool.view  # inside the reserve, i.e. under pool.cond
        return original() if view is None else view.summary()


@contextmanager
def serialised(budget: RoundBudget, pool: Pool, meter: Meter | None) -> Iterator[None]:
    """Route ``budget``'s pool-thread reserve / complete / summary through ``pool`` for the prefetch only.

    The stage-cap guard also counts every in-flight hold, and the meter's ``spent`` stops globbing the shard. On a
    clean exit the incremental ledger view is verified against the ledger's own ``summary()``."""
    guard_before = budget.reserve_guard
    if guard_before is not None:
        budget.reserve_guard = lambda amount: guard_before(amount + pool.inflight_usd())
    budget.serial = Serial(pool)
    if meter is not None:
        meter.pool = pool
    try:
        yield
    finally:
        budget.serial = None
        budget.reserve_guard = guard_before
        if meter is not None:
            meter.pool = None
    if pool.view is not None:
        pool.view.verify(budget.summary())


def worst_case_usd(wire: Mapping) -> Decimal:
    """The ledger's hold of one Jev wire: (canonical bytes + 4096) input tokens at ``JEV_INPUT_PER_MILLION``,
    4096 output tokens at $0."""
    from memory_kernel.construction.mined_affordance_contract import canonical_json
    from memory_kernel.pipeline.ledger import JEV_INPUT_PER_MILLION

    return Decimal(len(canonical_json(wire).encode()) + 4096) * JEV_INPUT_PER_MILLION / 1_000_000


def cap_could_bind(budget: RoundBudget, meter: Meter | None, worst: Decimal) -> dict:
    """Could a plan whose holds total ``worst`` hit a cap? Uses the ledger's own predicates: the shard cap
    (the round's ``cap_usd`` - ``held_or_spent_usd``, both from ``summary()``) and the meter's stage / unit caps
    (``would_cross``). No meter -> could bind (stay sequential). Port of the harness's cap check. A stage past its
    stage cap still binds here: the rest of it is sent sequentially, never stopped."""
    summary = budget.summary()
    remaining = Decimal(summary["cap_usd"]) - Decimal(summary["held_or_spent_usd"])
    out: dict[str, Any] = {"worst_case_usd": str(worst), "shard_remaining_usd": str(remaining)}
    binds = worst > remaining
    if meter is None:
        out["stage_cap"], binds = "unknown: no Meter captured", True
    elif (crossed := meter.would_cross(float(worst))) is not None:
        out["stage_cap"], binds = crossed, True
    else:
        out["stage_cap"] = f"ok (cap {meter.stage_cap})"
    return {**out, "could_bind": binds}


class Prefetcher:
    """N-in-flight sending of a stage's exact wires, called by the stage before its sequential loop.

    ``n == 1`` is no prefetch at all (today's sequential path). ``wires`` may be a list or a zero-argument callable
    building it: a build failure is logged and the stage runs sequentially (nothing was sent); a failure inside the
    pool is the ledger's own and propagates."""

    def __init__(self, n: int, *, emit: Callable[[dict], None] | None = None):
        if not 1 <= n <= MAX_CONCURRENCY:
            raise ValueError(f"concurrency {n}: must be 1..{MAX_CONCURRENCY}")
        self.n = n
        self.emit = emit or self._stderr

    @staticmethod
    def _stderr(report: dict) -> None:
        print(json.dumps({"prefetch": report.get("stage"), **report}, default=str), file=sys.stderr, flush=True)

    def _wires(self, stage: str, wires: Sequence[Mapping] | Callable[[], Sequence[Mapping]]) -> list | None:
        if not callable(wires):
            return list(wires)
        try:
            return list(wires())
        except Exception as exc:  # noqa: BLE001 - a wire-list build failure falls back to sequential
            self.emit(
                {
                    "stage": stage,
                    "skipped": "wire-list build failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback": "sequential",
                }
            )
            return None

    def jev(
        self,
        budget: RoundBudget,
        wires: Sequence[Mapping] | Callable[[], Sequence[Mapping]],
        *,
        stage: str,
        meter: Meter | None = None,
        gate_cap: Decimal | float | None = None,
    ) -> dict:
        """Send the uncached ``wires`` through ``budget.evaluate`` (semantic=False, the ledger's own sender)."""
        if self.n == 1:
            return {"skipped": "concurrency 1"}
        built = self._wires(stage, wires)
        if built is None:
            return {"skipped": "wire-list build failed"}
        items, stats = plan_items(budget.directory, built, kind="jev")
        if not items:
            report = {**stats, "items": 0, "sent": 0}
        else:
            gate = (
                None
                if gate_cap is None
                else SpendCapGate([_sha(w) for w in built], budget.directory, Decimal(str(gate_cap)))
            )
            view = LedgerView(budget.summary())
            pool = Pool(self.n, lambda item: budget.evaluate(item.payload), gate, budget=budget, view=view)
            with serialised(budget, pool, meter):
                report = {**stats, **pool.run(items)}
        self.emit({"stage": stage, **report})
        return report


#: The OpenRouter chat completions endpoint (``OpenRouterChat``'s default). Kernel 1.8 archived the gap sender
#: (``GapSender``, stage ``grammar_gap``) with the question and remine stages.
GAP_URL = "https://openrouter.ai/api/v1/chat/completions"


# -- OpenRouter chat (the harness taxonomy module's HTTP sender) ----------------------------------------------------


class OpenRouterChat:
    """One OpenRouter chat completion with the API key passed in (the answer server's sender; 1.7's taxonomy stage
    used it too); no retry on a paid response (a transport error before any response is raised). Called as
    ``sender(request) -> dict``."""

    def __init__(self, *, api_key: str | None, url: str = GAP_URL, timeout: float = 180.0) -> None:
        self.api_key, self.url, self.timeout = api_key, url, timeout

    def __call__(self, request: dict) -> dict:
        import httpx

        if not self.api_key:
            raise RuntimeError("OpenRouterChat: no API key configured")
        response = httpx.post(
            self.url, json=request, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout
        )
        response.raise_for_status()
        body = response.json()
        return {
            "text": body["choices"][0]["message"]["content"] or "",
            "usage": body.get("usage", {}),
            "id": body.get("id"),
        }
