"""In-memory ``KernelOps``: the contract's reference behaviour, for tests and single-process use."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from typing import Any

from memory_kernel.ops.base import (
    HISTORY_KEEP,
    Lease,
    LeaseLost,
    OpsRecord,
    ceiling_refusal,
    check_lease_free,
    headroom,
    spend_view,
)
from memory_kernel.pipeline.bonfire import AppendBusy, BonfireError


class MemoryKernelOps:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self._recs: dict[str, OpsRecord] = {}
        self._hist: dict[str, list[dict[str, Any]]] = {}

    def _rec(self, id: str) -> OpsRecord:
        try:
            return self._recs[id]
        except KeyError:
            raise BonfireError(f"no kernel_ops record for bonfire {id!r}") from None

    def create(self, id: str, *, ceiling_usd: Decimal | None, state: str = "seeding") -> None:
        with self._lock:
            if id in self._recs:
                raise BonfireError(f"kernel_ops record for bonfire {id!r} already exists")
            self._recs[id] = OpsRecord(id, state, ceiling_usd, Decimal(0), None, None)
            self._hist[id] = []

    def get(self, id: str) -> OpsRecord | None:
        with self._lock:
            return self._recs.get(id)

    def set_state(self, id: str, state: str) -> None:
        with self._lock:
            self._recs[id] = replace(self._rec(id), state=state)

    def set_ceiling(self, id: str, ceiling_usd: Decimal | None) -> None:
        with self._lock:
            self._recs[id] = replace(self._rec(id), ceiling_usd=ceiling_usd)

    def take_lease(self, id: str, *, owner: str, session_id: str, run: int, committed_run: int, ttl_s: float) -> Lease:
        with self._lock:
            rec = self._rec(id)
            now = self.clock()
            check_lease_free(rec.lease, owner=owner, session_id=session_id, committed_run=committed_run, now=now)
            lease = Lease(owner, now + ttl_s, run, session_id)
            self._recs[id] = replace(rec, lease=lease)
            return lease

    def renew(self, id: str, *, owner: str, ttl_s: float) -> Lease:
        with self._lock:
            rec = self._rec(id)
            now = self.clock()
            if rec.lease is None or rec.lease.owner != owner or rec.lease.expires_at <= now:
                raise LeaseLost(f"the lease on bonfire {id!r} is no longer held by {owner!r}")
            lease = replace(rec.lease, expires_at=now + ttl_s)
            self._recs[id] = replace(rec, lease=lease)
            return lease

    def release(self, id: str, *, owner: str, committed: bool) -> None:
        """Committed: the lease is cleared. Not committed: it expires now but keeps naming the run and session, so a
        different session is refused as pending until the stopped one is sent again."""
        with self._lock:
            rec = self._rec(id)
            if rec.lease is None or rec.lease.owner != owner:
                return
            lease = None if committed else replace(rec.lease, expires_at=self.clock())
            self._recs[id] = replace(rec, lease=lease)

    def clear_stale_lease(self, id: str) -> None:
        with self._lock:
            rec = self._rec(id)
            if rec.lease is not None:
                if rec.lease.expires_at > self.clock():
                    raise AppendBusy(f"bonfire {id!r} is leased by {rec.lease.owner} and the lease is live")
                self._recs[id] = replace(rec, lease=None)

    def hold(self, id: str, *, run: int, estimate_usd: Decimal) -> Decimal | None:
        with self._lock:
            rec = self._rec(id)
            # a hold still held is an orphan (a live lease excludes a concurrent hold): charge it at its estimate
            spent = rec.spent_usd + (rec.held[1] if rec.held else Decimal(0))
            if rec.ceiling_usd is not None and spent + estimate_usd > rec.ceiling_usd:
                raise ceiling_refusal(id, spent=spent, estimate=estimate_usd, ceiling=rec.ceiling_usd)
            self._recs[id] = replace(rec, spent_usd=spent, held=(run, estimate_usd))
            return headroom(rec.ceiling_usd, spent, estimate_usd)

    def settle(self, id: str, *, run: int, spent_usd: Decimal, new_messages: int, budget_usd: Decimal) -> None:
        with self._lock:
            rec = self._rec(id)
            held = None if rec.held and rec.held[0] == run else rec.held
            self._recs[id] = replace(rec, spent_usd=rec.spent_usd + spent_usd, held=held)
            self._hist[id] = [
                *self._hist[id],
                {"run": run, "spent_usd": spent_usd, "new_messages": new_messages, "budget_usd": budget_usd},
            ][-HISTORY_KEEP:]

    def spend(self, id: str) -> dict[str, str | None]:
        with self._lock:
            return spend_view(self._rec(id))

    def history(self, id: str) -> list[dict[str, Any]]:
        with self._lock:
            self._rec(id)
            return list(self._hist[id])
