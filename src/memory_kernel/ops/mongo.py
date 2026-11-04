"""Mongo ``KernelOps``: one ``kernel_ops`` document per bonfire (``_id`` = the bonfire id). pymongo (the ``ops``
extra) is imported only when a client is built. Money is stored as Decimal128 so the ceiling check is one atomic
``find_one_and_update``; the history keeps decimal strings."""

from __future__ import annotations

import time
from collections.abc import Callable
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

_TAKE_ATTEMPTS = 5


def _d128(value: Decimal) -> Any:
    from bson.decimal128 import Decimal128  # noqa: PLC0415

    return Decimal128(value)


def _dec(value: Any) -> Decimal | None:
    return None if value is None else value.to_decimal()


def _lease(doc: dict[str, Any] | None) -> Lease | None:
    if not doc:
        return None
    return Lease(doc["owner"], doc["expires_at"], doc["run"], doc["session_id"])


def _record(doc: dict[str, Any]) -> OpsRecord:
    held = doc.get("held")
    return OpsRecord(
        id=doc["_id"],
        state=doc["state"],
        ceiling_usd=_dec(doc.get("ceiling_usd")),
        spent_usd=_dec(doc["spent_usd"]) or Decimal(0),
        held=(held["run"], _dec(held["estimate_usd"])) if held else None,
        lease=_lease(doc.get("lease")),
    )


class MongoKernelOps:
    def __init__(self, collection: Any, clock: Callable[[], float] = time.time) -> None:
        self.col = collection
        self.clock = clock

    @classmethod
    def from_uri(
        cls, uri: str, db: str = "mk_ops", collection: str = "kernel_ops", clock: Callable[[], float] = time.time
    ) -> MongoKernelOps:
        try:
            from pymongo import MongoClient  # noqa: PLC0415
        except ImportError:
            raise BonfireError("kernel_ops needs pymongo: install memory-kernel[ops]") from None
        return cls(MongoClient(uri, tz_aware=True)[db][collection], clock)

    def _doc(self, id: str) -> dict[str, Any]:
        doc = self.col.find_one({"_id": id})
        if doc is None:
            raise BonfireError(f"no kernel_ops record for bonfire {id!r}")
        return doc

    def create(self, id: str, *, ceiling_usd: Decimal | None, state: str = "seeding") -> None:
        from pymongo.errors import DuplicateKeyError  # noqa: PLC0415

        doc = {
            "_id": id,
            "state": state,
            "ceiling_usd": None if ceiling_usd is None else _d128(ceiling_usd),
            "spent_usd": _d128(Decimal(0)),
            "held": None,
            "lease": None,
            "history": [],
        }
        try:
            self.col.insert_one(doc)
        except DuplicateKeyError:
            raise BonfireError(f"kernel_ops record for bonfire {id!r} already exists") from None

    def get(self, id: str) -> OpsRecord | None:
        doc = self.col.find_one({"_id": id})
        return None if doc is None else _record(doc)

    def _set(self, id: str, fields: dict[str, Any]) -> None:
        if self.col.update_one({"_id": id}, {"$set": fields}).matched_count == 0:
            raise BonfireError(f"no kernel_ops record for bonfire {id!r}")

    def set_state(self, id: str, state: str) -> None:
        self._set(id, {"state": state})

    def set_ceiling(self, id: str, ceiling_usd: Decimal | None) -> None:
        self._set(id, {"ceiling_usd": None if ceiling_usd is None else _d128(ceiling_usd)})

    def take_lease(self, id: str, *, owner: str, session_id: str, run: int, committed_run: int, ttl_s: float) -> Lease:
        for _ in range(_TAKE_ATTEMPTS):
            doc = self._doc(id)
            observed = doc.get("lease")
            now = self.clock()
            check_lease_free(_lease(observed), owner=owner, session_id=session_id, committed_run=committed_run, now=now)
            lease = Lease(owner, now + ttl_s, run, session_id)
            # compare-and-set on the lease exactly as observed: a racing taker makes this miss and re-decide
            swapped = self.col.update_one(
                {"_id": id, "lease": observed},
                {
                    "$set": {
                        "lease": {"owner": owner, "expires_at": lease.expires_at, "run": run, "session_id": session_id}
                    }
                },
            )
            if swapped.matched_count:
                return lease
        raise BonfireError(f"could not take the lease on bonfire {id!r}: it kept changing")

    def renew(self, id: str, *, owner: str, ttl_s: float) -> Lease:
        now = self.clock()
        doc = self.col.find_one_and_update(
            {"_id": id, "lease.owner": owner, "lease.expires_at": {"$gt": now}},
            {"$set": {"lease.expires_at": now + ttl_s}},
            return_document=True,
        )
        if doc is None:
            raise LeaseLost(f"the lease on bonfire {id!r} is no longer held by {owner!r}")
        return _lease(doc["lease"])  # type: ignore[return-value]

    def release(self, id: str, *, owner: str, committed: bool) -> None:
        if committed:
            self.col.update_one({"_id": id, "lease.owner": owner}, {"$set": {"lease": None}})
        else:
            # expires now but keeps naming the run and session: a different session is refused as pending
            self.col.update_one({"_id": id, "lease.owner": owner}, {"$set": {"lease.expires_at": self.clock()}})

    def clear_stale_lease(self, id: str) -> None:
        now = self.clock()
        done = self.col.update_one(
            {"_id": id, "$or": [{"lease": None}, {"lease.expires_at": {"$lte": now}}]}, {"$set": {"lease": None}}
        )
        if done.matched_count == 0:
            lease = self._doc(id).get("lease")  # a missing record raises here
            if not lease:
                return  # cleared between the update and this read: nothing left to clear
            raise AppendBusy(f"bonfire {id!r} is leased by {lease['owner']} and the lease is live")

    def close(self) -> None:
        client = getattr(self.col, "database", None)
        client = getattr(client, "client", None)
        if client is not None:
            client.close()

    def hold(self, id: str, *, run: int, estimate_usd: Decimal) -> Decimal | None:
        est = _d128(estimate_usd)
        # a hold still held is an orphan (a live lease excludes a concurrent hold): charge it at its estimate
        orphan = {"$ifNull": ["$held.estimate_usd", 0]}
        for _ in range(_TAKE_ATTEMPTS):
            doc = self.col.find_one_and_update(
                {
                    "_id": id,
                    "$or": [
                        {"ceiling_usd": None},
                        {"$expr": {"$lte": [{"$add": ["$spent_usd", orphan, est]}, "$ceiling_usd"]}},
                    ],
                },
                [
                    {
                        "$set": {
                            "spent_usd": {"$add": ["$spent_usd", orphan]},
                            "held": {"$literal": {"run": run, "estimate_usd": est}},
                        }
                    }
                ],
                return_document=True,
            )
            if doc is not None:
                rec = _record(doc)
                return headroom(rec.ceiling_usd, rec.spent_usd, estimate_usd)
            rec = _record(self._doc(id))  # a missing record raises here
            spent = rec.spent_usd + (rec.held[1] if rec.held else Decimal(0))
            # the ceiling may have been lifted since the refusal: then hold again; else refuse from this state
            if rec.ceiling_usd is not None and spent + estimate_usd > rec.ceiling_usd:
                raise ceiling_refusal(id, spent=spent, estimate=estimate_usd, ceiling=rec.ceiling_usd)
        raise BonfireError(f"could not hold on bonfire {id!r}: its record kept changing")

    def settle(self, id: str, *, run: int, spent_usd: Decimal, new_messages: int, budget_usd: Decimal) -> None:
        entry = {
            "run": run,
            "spent_usd": str(spent_usd),
            "new_messages": new_messages,
            "budget_usd": str(budget_usd),
        }
        # one pipeline update: add the spend, drop the hold only if it is this run's, append the history
        result = self.col.update_one(
            {"_id": id},
            [
                {
                    "$set": {
                        "spent_usd": {"$add": ["$spent_usd", _d128(spent_usd)]},
                        "held": {"$cond": [{"$eq": ["$held.run", run]}, None, "$held"]},
                        "history": {
                            "$slice": [
                                {"$concatArrays": [{"$ifNull": ["$history", []]}, [{"$literal": entry}]]},
                                -HISTORY_KEEP,
                            ]
                        },
                    }
                }
            ],
        )
        if result.matched_count == 0:
            raise BonfireError(f"no kernel_ops record for bonfire {id!r}")

    def spend(self, id: str) -> dict[str, str | None]:
        return spend_view(_record(self._doc(id)))

    def history(self, id: str) -> list[dict[str, Any]]:
        return [
            {
                **e,
                "spent_usd": Decimal(e["spent_usd"]),
                "budget_usd": Decimal(e["budget_usd"]),
            }
            for e in self._doc(id).get("history", [])
        ]
