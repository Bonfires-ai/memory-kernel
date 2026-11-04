"""``kernel_ops`` over the memory backend and, when ``MK_TEST_MONGO_URI`` is set, the Mongo one (a throwaway db)."""

from __future__ import annotations

import os
import uuid
from decimal import Decimal

import pytest

from memory_kernel.ops.base import LeaseKeeper, LeaseLost
from memory_kernel.ops.memory import MemoryKernelOps
from memory_kernel.pipeline import bonfire
from memory_kernel.pipeline.bonfire import AppendBusy, BonfireError, PendingSession

MONGO_URI = os.environ.get("MK_TEST_MONGO_URI")
D = Decimal


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture(params=["memory", "mongo"])
def ops(request, clock):
    if request.param == "memory":
        yield MemoryKernelOps(clock)
        return
    if not MONGO_URI:
        pytest.skip("MK_TEST_MONGO_URI is not set")
    from memory_kernel.ops.mongo import MongoKernelOps

    db = f"mk_ops_test_{uuid.uuid4().hex[:10]}"
    o = MongoKernelOps.from_uri(MONGO_URI, db=db, clock=clock)
    try:
        yield o
    finally:
        o.col.database.client.drop_database(db)
        o.col.database.client.close()


def fresh(ops, ceiling="10"):
    ops.create("b1", ceiling_usd=None if ceiling is None else D(ceiling))
    return ops


def take(ops, owner="w1", session="s1", run=1, committed=0, ttl=30):
    return ops.take_lease("b1", owner=owner, session_id=session, run=run, committed_run=committed, ttl_s=ttl)


def test_live_lease_of_another_owner_is_busy(ops):
    fresh(ops)
    take(ops, owner="w1")
    with pytest.raises(AppendBusy):
        take(ops, owner="w2", session="s2")


def test_same_owner_takes_its_live_lease_back(ops, clock):
    fresh(ops)
    take(ops, owner="w1")
    clock.now += 5
    lease = take(ops, owner="w1", session="s1")
    assert lease.owner == "w1" and lease.expires_at == clock.now + 30


def test_expired_lease_naming_another_session_is_pending(ops, clock):
    fresh(ops)
    take(ops, owner="w1", session="s1", run=1)
    clock.now += 31
    with pytest.raises(PendingSession):
        take(ops, owner="w2", session="s2", run=1)


def test_same_owner_expired_lease_of_another_session_is_pending(ops, clock):
    fresh(ops)
    take(ops, owner="w1", session="s1", run=1)
    ops.release("b1", owner="w1", committed=False)
    with pytest.raises(PendingSession):
        take(ops, owner="w1", session="s2", run=1)
    assert take(ops, owner="w1", session="s1", run=1).session_id == "s1"


def test_expired_lease_of_a_committed_run_does_not_pend_the_next_session(ops, clock):
    fresh(ops)
    take(ops, owner="w1", session="s1", run=1)
    clock.now += 31
    lease = take(ops, owner="w2", session="s2", run=2, committed=1)
    assert lease.owner == "w2" and lease.run == 2


def test_retry_of_the_stopped_session_takes_the_lease(ops, clock):
    fresh(ops)
    take(ops, owner="w1", session="s1", run=1)
    ops.release("b1", owner="w1", committed=False)
    with pytest.raises(PendingSession):
        take(ops, owner="w2", session="s2", run=1)
    lease = take(ops, owner="w2", session="s1", run=1)
    assert lease.owner == "w2" and lease.session_id == "s1"


def test_hold_past_the_ceiling_is_refused(ops):
    fresh(ops, "1.00")
    assert ops.hold("b1", run=1, estimate_usd=D("0.40")) == D("0.60")
    ops.settle("b1", run=1, spent_usd=D("0.40"), new_messages=3, budget_usd=D("0.40"))
    with pytest.raises(BonfireError) as exc:
        ops.hold("b1", run=2, estimate_usd=D("0.70"))
    assert str(exc.value).startswith(bonfire.CEILING)
    assert ops.hold("b1", run=2, estimate_usd=D("0.60")) == D("0")


def test_no_ceiling_means_no_refusal_and_no_headroom(ops):
    fresh(ops, None)
    assert ops.hold("b1", run=1, estimate_usd=D("999")) is None
    assert ops.spend("b1")["ceiling_usd"] is None


def test_an_orphaned_hold_is_charged_at_its_estimate_by_the_next_hold(ops):
    fresh(ops, "2.00")
    ops.hold("b1", run=1, estimate_usd=D("0.90"))  # the worker crashes here and its scratch is lost
    assert ops.spend("b1")["charged_usd"] == "0.90"
    assert ops.hold("b1", run=2, estimate_usd=D("0.50")) == D("0.60")  # 2.00 - 0.90 orphan - 0.50
    rec = ops.get("b1")
    assert rec.spent_usd == D("0.90") and rec.held == (2, D("0.50"))
    assert ops.spend("b1")["charged_usd"] == "1.40"
    with pytest.raises(BonfireError) as exc:  # the orphan counts toward the ceiling check too
        ops.hold("b1", run=3, estimate_usd=D("0.70"))
    assert str(exc.value).startswith(bonfire.CEILING)
    assert ops.get("b1").spent_usd == D("0.90") and ops.get("b1").held == (2, D("0.50"))


def test_clear_stale_lease_clears_expired_and_released_leases_and_refuses_a_live_one(ops, clock):
    fresh(ops)
    ops.clear_stale_lease("b1")  # no lease: nothing to do
    take(ops, owner="w1")
    with pytest.raises(AppendBusy):
        ops.clear_stale_lease("b1")
    assert ops.get("b1").lease is not None
    clock.now += 31  # expired
    ops.clear_stale_lease("b1")
    assert ops.get("b1").lease is None
    take(ops, owner="w2", session="s2", run=1)
    ops.release("b1", owner="w2", committed=False)  # released uncommitted: expires now, still names the session
    assert ops.get("b1").lease is not None
    ops.clear_stale_lease("b1")
    assert ops.get("b1").lease is None
    with pytest.raises(BonfireError):
        ops.clear_stale_lease("nope")


def test_settle_moves_the_hold_to_spent_and_records_history(ops):
    fresh(ops)
    for run in range(1, 13):
        ops.hold("b1", run=run, estimate_usd=D("0.30"))
        ops.settle("b1", run=run, spent_usd=D("0.10"), new_messages=run, budget_usd=D("0.30"))
    rec = ops.get("b1")
    assert rec.held is None and rec.spent_usd == D("1.20")
    hist = ops.history("b1")
    assert [h["run"] for h in hist] == list(range(3, 13))
    assert hist[-1] == {"run": 12, "spent_usd": D("0.10"), "new_messages": 12, "budget_usd": D("0.30")}


def test_spend_is_decimal_strings(ops):
    fresh(ops, "2.50")
    ops.hold("b1", run=1, estimate_usd=D("0.25"))
    ops.settle("b1", run=1, spent_usd=D("0.1234"), new_messages=1, budget_usd=D("0.25"))
    assert ops.spend("b1") == {"charged_usd": "0.1234", "ceiling_usd": "2.50"}


def test_renew_by_another_owner_is_lost(ops):
    fresh(ops)
    take(ops, owner="w1")
    with pytest.raises(LeaseLost):
        ops.renew("b1", owner="w2", ttl_s=30)
    assert ops.renew("b1", owner="w1", ttl_s=60).expires_at == 1060.0


def test_lease_keeper_check_raises_lease_lost(ops, clock):
    fresh(ops)
    take(ops, owner="w1")
    keeper = LeaseKeeper(ops, "b1", "w1", 30, clock=clock, wait=lambda _s: True)
    keeper.check()
    clock.now += 31
    with pytest.raises(LeaseLost):
        keeper.check()
    assert keeper.lost


def test_set_state_and_ceiling(ops):
    fresh(ops)
    ops.set_state("b1", "ready")
    ops.set_ceiling("b1", None)
    rec = ops.get("b1")
    assert rec.state == "ready" and rec.ceiling_usd is None
    assert ops.get("nope") is None


def test_mongo_close_closes_its_client():
    from types import SimpleNamespace

    from memory_kernel.ops.mongo import MongoKernelOps

    closed: list[bool] = []
    col = SimpleNamespace(database=SimpleNamespace(client=SimpleNamespace(close=lambda: closed.append(True))))
    MongoKernelOps(col).close()
    assert closed == [True]


def test_clear_stale_lease_treats_a_lease_gone_after_the_miss_as_success():
    from types import SimpleNamespace

    from memory_kernel.ops.mongo import MongoKernelOps

    col = SimpleNamespace(
        update_one=lambda *_a, **_k: SimpleNamespace(matched_count=0),
        find_one=lambda *_a, **_k: {"_id": "b1", "lease": None},
    )
    MongoKernelOps(col).clear_stale_lease("b1")  # no TypeError
