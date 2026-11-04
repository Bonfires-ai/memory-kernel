"""The ``kernel_ops`` contract: one record per bonfire holding its per-run spend hold, its append lease (which is also
the in-flight marker) and a short spend history. The API is synchronous: an append runs in a worker thread."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from memory_kernel.pipeline import bonfire
from memory_kernel.pipeline.bonfire import AppendBusy, BonfireError, PendingSession

_log = logging.getLogger(__name__)

#: how many settled runs ``history`` keeps
HISTORY_KEEP = 10


class LeaseLost(BonfireError):
    """The append lease is no longer this owner's (expired, or taken): nothing may be committed under it."""


@dataclass(frozen=True)
class Lease:
    owner: str
    expires_at: float
    run: int
    session_id: str


@dataclass(frozen=True)
class OpsRecord:
    id: str
    state: str
    #: None = no ceiling (recorded explicitly)
    ceiling_usd: Decimal | None
    spent_usd: Decimal
    #: the outstanding hold ``(run, estimate_usd)``, or None
    held: tuple[int, Decimal] | None
    lease: Lease | None


class KernelOps(Protocol):
    def create(self, id: str, *, ceiling_usd: Decimal | None, state: str = "seeding") -> None: ...
    def get(self, id: str) -> OpsRecord | None: ...
    def set_state(self, id: str, state: str) -> None: ...
    def set_ceiling(self, id: str, ceiling_usd: Decimal | None) -> None: ...
    def take_lease(
        self, id: str, *, owner: str, session_id: str, run: int, committed_run: int, ttl_s: float
    ) -> Lease: ...
    def renew(self, id: str, *, owner: str, ttl_s: float) -> Lease: ...
    def release(self, id: str, *, owner: str, committed: bool) -> None: ...
    def clear_stale_lease(self, id: str) -> None:
        """Clear a lease that is NOT live (expired, or released uncommitted). ``AppendBusy`` for a live lease."""
        ...

    def hold(self, id: str, *, run: int, estimate_usd: Decimal) -> Decimal | None:
        """Hold ``estimate_usd`` for ``run``. A hold still held is an orphan and is added to ``spent_usd`` first."""
        ...

    def settle(self, id: str, *, run: int, spent_usd: Decimal, new_messages: int, budget_usd: Decimal) -> None: ...
    def spend(self, id: str) -> dict[str, str | None]: ...
    def history(self, id: str) -> list[dict[str, Any]]: ...


def check_lease_free(current: Lease | None, *, owner: str, session_id: str, committed_run: int, now: float) -> None:
    """Raise unless ``owner``/``session_id`` may take the lease over ``current``. A live lease of another owner is
    ``AppendBusy``; the same owner takes it back. An expired lease follows the expiry rules whoever owns it. An expired lease of an uncommitted run (``run > committed_run``)
    naming another session is ``PendingSession`` (that session stopped; send it again first); the same session takes
    it. An expired lease of a committed run counts as released."""
    if current is None:
        return
    if current.expires_at > now:
        if current.owner == owner:
            return
        raise AppendBusy(f"busy: bonfire is leased by {current.owner}")
    if current.run > committed_run and current.session_id != session_id:
        raise PendingSession(
            f"run {current.run} of session {current.session_id!r} stopped before it committed: send that session "
            f"again before session {session_id!r}"
        )


def ceiling_refusal(id: str, *, spent: Decimal, estimate: Decimal, ceiling: Decimal) -> BonfireError:
    return BonfireError(
        f"{bonfire.CEILING}: bonfire {id!r} has spent ${spent} of ${ceiling}; this run's estimate is ${estimate}"
    )


def headroom(ceiling: Decimal | None, spent: Decimal, estimate: Decimal) -> Decimal | None:
    return None if ceiling is None else ceiling - spent - estimate


def spend_view(rec: OpsRecord) -> dict[str, str | None]:
    """Charged = settled spend plus any outstanding hold (a crashed worker's hold stays charged at its estimate)."""
    charged = rec.spent_usd + (rec.held[1] if rec.held else Decimal(0))
    return {"charged_usd": str(charged), "ceiling_usd": None if rec.ceiling_usd is None else str(rec.ceiling_usd)}


class LeaseKeeper:
    """Renews ``owner``'s lease every ``ttl_s / 3`` in a daemon thread until ``stop``. A renew that RAISES (not
    ``LeaseLost``) is retried at the same cadence until the lease's own expiry; ``LeaseLost`` or an expiry marks it
    lost. ``check`` is the synchronous renew a writer runs right before it commits."""

    def __init__(
        self,
        ops: KernelOps,
        bonfire_id: str,
        owner: str,
        ttl_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        self.ops, self.bonfire_id, self.owner, self.ttl_s = ops, bonfire_id, owner, ttl_s
        self.clock = clock
        self.lost = False
        self.valid_until = clock() + ttl_s  # the lease was just taken
        self._stop = threading.Event()
        self._wait = wait if wait is not None else self._stop.wait
        self._thread = threading.Thread(target=self._run, name=f"mk-lease-{bonfire_id}", daemon=True)

    def _renew(self) -> None:
        self.ops.renew(self.bonfire_id, owner=self.owner, ttl_s=self.ttl_s)
        self.valid_until = self.clock() + self.ttl_s

    def _run(self) -> None:
        while not self._wait(self.ttl_s / 3):
            try:
                self._renew()
            except LeaseLost:
                self.lost = True
                return
            except Exception as exc:  # noqa: BLE001 - a transient store error: retried until the lease expires
                if self.clock() >= self.valid_until:
                    self.lost = True
                    return
                _log.warning("lease renew of bonfire %r failed (%s); retrying", self.bonfire_id, type(exc).__name__)

    def start(self) -> LeaseKeeper:
        self._thread.start()
        return self

    def check(self) -> None:
        """A synchronous renew; ``LeaseLost`` when the lease is lost or the renew cannot be confirmed."""
        if not self.lost:
            try:
                self._renew()
                return
            except Exception:  # noqa: BLE001 - not confirmed is not held: never commit on it
                self.lost = True
        raise LeaseLost(f"the lease on bonfire {self.bonfire_id!r} was lost; nothing was committed")

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
