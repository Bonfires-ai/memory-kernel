"""Kernel 2.0: the seed (``pipeline.graph_seed``) builds a new bonfire into an empty Neo4j group and seals it with its
``BonfireMeta``, guarded by the ``kernel_ops`` record.

``seed`` is tested with the build, the seal and the group stood in (a fake world: node count, meta, a call log), so the
order of its steps and its refusals are what is pinned. ``empty_group`` and ``seal_seed`` run on the test Neo4j: the
seal over a hand-made bootstrap root, read back through ``state_read``. Invented names only."""

from __future__ import annotations

import asyncio
import json
import re
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_kernel.graph.state import MESSAGE_EMBEDDING_MODEL  # noqa: F401  (the module under test builds on state)
from memory_kernel.ops.memory import MemoryKernelOps
from memory_kernel.pipeline import graph_seed
from memory_kernel.pipeline.bonfire import BonfireError
from memory_kernel.pipeline.config import BuildConfig

GROUP = "bf-kiln"


def config(tmp_path: Path, **kw) -> BuildConfig:
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"sample_id": GROUP, "conversation": {"session_1": []}}))
    base = {
        "sample_id": GROUP,
        "conversation": seed,
        "group": GROUP,
        "workdir": tmp_path / "workdir",
        "root": tmp_path,
        "genesis": True,
    }
    return BuildConfig(**(base | kw))


class World:
    """The group as ``seed`` sees it, and a log of what it did to it."""

    def __init__(self, monkeypatch, *, nodes: int = 0, meta: bool = False, build: str = "complete") -> None:
        self.nodes, self.meta, self.calls, self.build = nodes, meta, [], build
        self.ops: MemoryKernelOps | None = None
        self.through: list[str | None] = []
        self.caps: list[float] = []
        self.on_build: Any = None
        monkeypatch.setattr(graph_seed, "_spent", lambda _cfg: 0.25)
        monkeypatch.setattr(graph_seed, "_messages", lambda _cfg: 2)
        driver = SimpleNamespace(close=self._close)
        monkeypatch.setattr(graph_seed, "_driver", lambda _cfg: driver)
        monkeypatch.setattr(graph_seed, "read_meta", self._read_meta)
        monkeypatch.setattr(graph_seed, "_node_count", self._node_count)
        monkeypatch.setattr(graph_seed, "empty_group", self._empty_group)
        monkeypatch.setattr(graph_seed, "run_build", self._run_build)
        monkeypatch.setattr(graph_seed, "seal_seed", self._seal_seed)

    async def _close(self) -> None:
        return None

    async def _read_meta(self, _driver, _group):
        return {"committed_run": 0} if self.meta else None

    async def _node_count(self, _driver, _group):
        return self.nodes

    async def _empty_group(self, _driver, group, **_kw):
        self.calls.append("empty")
        assert group == GROUP
        gone, self.nodes = self.nodes, 0
        return gone

    def _run_build(self, cfg, **_kw):
        self.calls.append("build")
        self.through.append(cfg.through)
        self.caps.append(cfg.cap_usd)
        if self.on_build is not None:
            self.on_build()
        assert not self.meta, "the group must stay invisible to readers while it is built"
        assert self.ops is not None and self.ops.get(GROUP).state == "seeding"
        self.nodes = 7
        return {"status": self.build, "stage": "seal"}

    async def _seal_seed(self, _driver, root, *, bonfire_id):
        self.calls.append("seal")
        assert bonfire_id == GROUP and root.name == GROUP
        self.meta = True
        return {"committed_run": 0}


def seed_of(world: World, cfg: BuildConfig, ops: MemoryKernelOps, **kw) -> Any:
    world.ops = ops
    return graph_seed.seed(cfg, ops=ops, **kw)


def test_seed_writes_bonfire_meta_last(tmp_path, monkeypatch):
    world, ops = World(monkeypatch), MemoryKernelOps()
    out = seed_of(world, config(tmp_path), ops, ceiling_usd=Decimal("3"))
    assert out["status"] == "seeded"
    assert world.calls == ["build", "seal"]  # the build writes the rows, then the seal writes the meta
    assert world.through == ["seal"]
    record = ops.get(GROUP)
    assert record.state == "ready" and record.ceiling_usd == Decimal("3")


def test_seed_of_a_seeded_bonfire_is_done_and_deletes_nothing(tmp_path, monkeypatch):
    world, ops = World(monkeypatch, nodes=40, meta=True), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=None, state="ready")
    assert seed_of(world, config(tmp_path), ops) == {"status": "done"}
    assert world.calls == [] and world.nodes == 40
    assert ops.get(GROUP).state == "ready"
    # no ops record either: still done, and none is made
    other = World(monkeypatch, nodes=40, meta=True)
    fresh = MemoryKernelOps()
    assert seed_of(other, config(tmp_path), fresh) == {"status": "done"}
    assert other.calls == [] and fresh.get(GROUP) is None


def test_seed_refuses_a_nonempty_group_it_did_not_start(tmp_path, monkeypatch):
    world, ops = World(monkeypatch, nodes=12), MemoryKernelOps()
    with pytest.raises(BonfireError, match="group holds data no 2.0 seed started"):
        seed_of(world, config(tmp_path), ops)
    assert world.calls == [] and world.nodes == 12
    assert ops.get(GROUP) is None  # a refused seed leaves no record that would later license an emptying


def test_failed_seed_retry_empties_the_group_first(tmp_path, monkeypatch):
    world, ops = World(monkeypatch, nodes=12), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=None, state="seeding")
    out = seed_of(world, config(tmp_path), ops)
    assert out["status"] == "seeded"
    assert world.calls == ["empty", "build", "seal"]
    assert ops.get(GROUP).state == "ready"


def test_a_seeding_record_over_an_empty_group_builds_without_emptying(tmp_path, monkeypatch):
    world, ops = World(monkeypatch), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=None, state="seeding")
    seed_of(world, config(tmp_path), ops)
    assert world.calls == ["build", "seal"]


def test_a_stopped_build_raises_and_leaves_the_seed_unsealed(tmp_path, monkeypatch):
    world, ops = World(monkeypatch, build="stopped"), MemoryKernelOps()
    with pytest.raises(BonfireError, match="stopped"):
        seed_of(world, config(tmp_path), ops)
    assert world.calls == ["build"] and not world.meta
    assert ops.get(GROUP).state == "seeding"


def test_seed_needs_the_group_to_be_the_bonfire_id(tmp_path, monkeypatch):
    world, ops = World(monkeypatch), MemoryKernelOps()
    with pytest.raises(BonfireError, match="group"):
        seed_of(world, config(tmp_path, group="other"), ops)
    assert world.calls == [] and ops.get(GROUP) is None


def test_a_ready_record_without_a_meta_is_refused(tmp_path, monkeypatch):
    world, ops = World(monkeypatch, nodes=3), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=None, state="ready")
    with pytest.raises(BonfireError, match="ready"):
        seed_of(world, config(tmp_path), ops)
    assert world.calls == []


def test_a_finished_seed_with_a_seeding_record_is_healed_to_ready(tmp_path, monkeypatch):
    world, ops = World(monkeypatch, nodes=40, meta=True), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=None, state="seeding")
    assert seed_of(world, config(tmp_path), ops) == {"status": "done"}
    assert world.calls == [] and ops.get(GROUP).state == "ready"


def cumulative(monkeypatch, totals):
    """The ledger of one workdir: cumulative across attempts, so each call reads the next total."""
    it = iter(totals)
    monkeypatch.setattr(graph_seed, "_spent", lambda _cfg: next(it))


def test_a_failed_seed_attempt_settles_its_spend_and_its_retry_is_not_refused(tmp_path, monkeypatch):
    world, ops = World(monkeypatch, build="stopped"), MemoryKernelOps()
    cumulative(monkeypatch, [0.25, 0.5])  # attempt 1 spent 0.25; the retry's ledger reads 0.25 + 0.25
    cfg = config(tmp_path, cap_usd=0.6)
    with pytest.raises(BonfireError, match="stopped"):
        seed_of(world, cfg, ops, ceiling_usd=Decimal("1.00"))
    record = ops.get(GROUP)
    assert record.state == "seeding" and record.held is None and record.spent_usd == Decimal("0.25")
    world.build = "complete"  # the retry holds 0.6 on top of 0.25 spent: 0.85 fits 1.00; an orphan would make 1.2
    assert seed_of(world, cfg, ops, ceiling_usd=Decimal("1.00"))["status"] == "seeded"
    record = ops.get(GROUP)
    # the same workdir: the ledger total is 0.50, of which 0.25 was settled -> L1 + L2, not 2*L1 + L2
    assert record.state == "ready" and record.spent_usd == Decimal("0.50") and record.held is None


def test_a_settle_that_fails_never_masks_the_build_failure_and_frees_the_lease(tmp_path, monkeypatch):
    world, ops = World(monkeypatch, build="stopped"), MemoryKernelOps()

    def broken(*_a, **_k):
        raise RuntimeError("mongo is down")

    monkeypatch.setattr(ops, "settle", broken)
    with pytest.raises(BonfireError, match="stopped"):  # the original error, not the settle's
        seed_of(world, config(tmp_path, cap_usd=0.6), ops, ceiling_usd=Decimal("1.00"))
    lease = ops.get(GROUP).lease  # released uncommitted: expired now (it keeps naming the stopped session)
    assert lease is None or lease.expires_at <= ops.clock()


def test_a_build_that_raises_also_settles_its_spend(tmp_path, monkeypatch):
    world, ops = World(monkeypatch), MemoryKernelOps()

    def boom():
        raise RuntimeError("the runner died")

    world.on_build = boom
    with pytest.raises(RuntimeError):
        seed_of(world, config(tmp_path, cap_usd=0.6), ops, ceiling_usd=Decimal("1.00"))
    record = ops.get(GROUP)
    assert record.held is None and record.spent_usd == Decimal("0.25") and record.state == "seeding"


def test_healing_a_sealed_bonfire_settles_a_hold_still_held(tmp_path, monkeypatch):
    world, ops = World(monkeypatch, nodes=40, meta=True), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=Decimal("2"), state="seeding")
    ops.hold(GROUP, run=0, estimate_usd=Decimal("0.75"))  # the seed died after the seal, before it settled
    assert seed_of(world, config(tmp_path), ops) == {"status": "done"}
    record = ops.get(GROUP)
    assert record.state == "ready" and record.held is None and record.spent_usd == Decimal("0.75")


def test_rebuild_lease_refuses_a_live_lease_holds_its_own_and_resets_the_record(tmp_path, monkeypatch):
    from memory_kernel.pipeline.bonfire import AppendBusy

    ops = MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=Decimal("5"), state="ready")
    ops.hold(GROUP, run=1, estimate_usd=Decimal("0.5"))
    ops.settle(GROUP, run=1, spent_usd=Decimal("0.4"), new_messages=1, budget_usd=Decimal("0.5"))
    ops.take_lease(GROUP, owner="appender", session_id="s", run=2, committed_run=1, ttl_s=600)
    with pytest.raises(AppendBusy), graph_seed.rebuild_lease(ops, GROUP):
        raise AssertionError("not reached")
    assert ops.get(GROUP).state == "ready"  # refused: nothing changed
    ops.release(GROUP, owner="appender", committed=True)
    with graph_seed.rebuild_lease(ops, GROUP):
        inside = ops.get(GROUP)  # the lease is ours for the whole delete: an append is refused
        assert inside.lease is not None and inside.lease.owner.startswith("rebuild-") and inside.state == "seeding"
        with pytest.raises(AppendBusy):
            ops.take_lease(GROUP, owner="appender", session_id="s2", run=2, committed_run=0, ttl_s=60)
    record = ops.get(GROUP)
    assert record.state == "seeding" and record.lease is None
    assert record.spent_usd == Decimal("0.4") and record.ceiling_usd == Decimal("5")
    with pytest.raises(BonfireError, match="no kernel_ops record"), graph_seed.rebuild_lease(ops, "bf-none"):
        pass


def test_a_seed_after_a_rebuild_reseeds_the_empty_group(tmp_path, monkeypatch):
    world, ops = World(monkeypatch), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=Decimal("5"), state="ready")
    with graph_seed.rebuild_lease(ops, GROUP):
        pass  # the group was emptied, the meta with it
    assert seed_of(world, config(tmp_path), ops)["status"] == "seeded"
    assert world.calls == ["build", "seal"] and ops.get(GROUP).state == "ready"


def test_a_crash_after_the_meta_delete_leaves_a_failed_seed_that_is_emptied_not_healed(tmp_path, monkeypatch):
    """The meta is gone, rows remain, the record says seeding: the seed empties and rebuilds, it never heals to ready."""
    world, ops = World(monkeypatch, nodes=12, meta=False), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=None, state="seeding")
    assert seed_of(world, config(tmp_path), ops)["status"] == "seeded"
    assert world.calls == ["empty", "build", "seal"]


def test_a_retry_removes_the_stale_markers_and_outputs_of_the_group_writing_stages(tmp_path, monkeypatch):
    from memory_kernel.pipeline.runner import STAGE_DIRS, marker, path

    cfg = config(tmp_path)
    world, ops = World(monkeypatch, nodes=12), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=None, state="seeding")
    kept = [marker(cfg, "build_dry"), marker(cfg, "events")]
    stale = [marker(cfg, s) for s in ("write", "index", "sanity", "seal")]
    for m in [*kept, *stale]:
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text("{}")
    (path(cfg, "write") / "summary.json").write_text("{}")
    seen: dict[str, bool] = {}

    def check():
        seen["stale_gone"] = not any(m.exists() for m in stale) and not (path(cfg, "write") / "summary.json").exists()
        seen["upstream_kept"] = all(m.exists() for m in kept)

    world.on_build = check
    seed_of(world, cfg, ops)
    assert seen == {"stale_gone": True, "upstream_kept": True}
    assert STAGE_DIRS["write"] in str(path(cfg, "write"))


def test_the_ceiling_caps_the_build_and_its_spend_is_recorded(tmp_path, monkeypatch):
    world, ops = World(monkeypatch), MemoryKernelOps()
    seed_of(world, config(tmp_path, cap_usd=20.0), ops, ceiling_usd=Decimal("3"))
    assert world.caps == [3.0]
    record = ops.get(GROUP)
    assert record.spent_usd == Decimal("0.25") and record.held is None
    assert ops.history(GROUP)[0]["new_messages"] == 2
    # a ceiling above the build's own cap leaves the cap
    other, ops2 = World(monkeypatch), MemoryKernelOps()
    seed_of(other, config(tmp_path, cap_usd=1.0), ops2, ceiling_usd=Decimal("9"))
    assert other.caps == [1.0]


def test_a_seed_never_overrides_an_existing_ceiling(tmp_path, monkeypatch):
    world, ops = World(monkeypatch), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=Decimal("2"), state="seeding")
    seed_of(world, config(tmp_path), ops, ceiling_usd=Decimal("1.5"))
    assert ops.get(GROUP).ceiling_usd == Decimal("2") and world.caps == [2.0]  # `mk bonfire ceiling` is the only way
    # a record with no ceiling takes the seed's
    other, ops2 = World(monkeypatch), MemoryKernelOps()
    ops2.create(GROUP, ceiling_usd=None, state="seeding")
    seed_of(other, config(tmp_path), ops2, ceiling_usd=Decimal("1.5"))
    assert ops2.get(GROUP).ceiling_usd == Decimal("1.5") and other.caps == [1.5]


def test_a_held_lease_refuses_the_seed_as_busy(tmp_path, monkeypatch):
    from memory_kernel.pipeline.bonfire import AppendBusy

    world, ops = World(monkeypatch), MemoryKernelOps()
    ops.create(GROUP, ceiling_usd=None, state="seeding")
    ops.take_lease(GROUP, owner="appender", session_id="s", run=1, committed_run=0, ttl_s=600)
    with pytest.raises(AppendBusy):
        seed_of(world, config(tmp_path), ops)
    assert world.calls == []


def test_the_seed_holds_the_lease_while_it_builds_and_frees_it_after(tmp_path, monkeypatch):
    world, ops = World(monkeypatch), MemoryKernelOps()
    held: dict[str, Any] = {}
    world.on_build = lambda: held.update(lease=ops.get(GROUP).lease)
    seed_of(world, config(tmp_path), ops)
    assert held["lease"] is not None and held["lease"].owner.startswith("seed-")
    assert ops.get(GROUP).lease is None


# -- on the test Neo4j ---------------------------------------------------------------------------------------------


def neo(target, fn) -> Any:
    from neo4j import AsyncGraphDatabase

    async def go():
        driver = AsyncGraphDatabase.driver(target.uri, auth=(target.user, target.password))
        try:
            return await fn(driver)
        finally:
            await driver.close()

    return asyncio.run(go())


async def count(driver, group: str) -> int:
    async with driver.session() as s:
        return [r["n"] async for r in await s.run("MATCH (n {group_id: $g}) RETURN count(n) AS n", g=group)][0]


@pytest.mark.needs_neo4j
def test_empty_group_refuses_a_group_with_bonfire_meta(neo4j_target):
    g = neo4j_target.group

    async def go(driver):
        async with driver.session() as s:
            await (
                await s.run("CREATE (:Entity {group_id: $g, uuid: 'e1'}), (:Entity {group_id: $g, uuid: 'e2'})", g=g)
            ).consume()
        assert await graph_seed.empty_group(driver, g, batch=1) == 2  # batched: one node per statement
        assert await count(driver, g) == 0
        async with driver.session() as s:
            await (
                await s.run(
                    "CREATE (:Entity {group_id: $g, uuid: 'e3'}), (:BonfireMeta {group_id: $g, uuid: 'm'})", g=g
                )
            ).consume()
        with pytest.raises(BonfireError, match="BonfireMeta"):
            await graph_seed.empty_group(driver, g)
        assert await count(driver, g) == 2

    neo(neo4j_target, go)


@pytest.mark.needs_neo4j
def test_rebuild_empties_a_sealed_group_and_a_reseal_mints_a_new_generation(neo4j_target, tmp_path, monkeypatch):
    from memory_kernel.graph.kg_writer import ensure_constraints, ensure_indexes, write_rows
    from memory_kernel.graph.state_read import read_meta
    from unit.pipeline import test_commit as tc

    group = neo4j_target.group
    root, _, rows = sealable(group, tmp_path, monkeypatch)

    async def go(driver):
        await ensure_indexes(driver)
        await ensure_constraints(driver)
        await write_rows(driver, rows, group_id=group, embed=tc.FakeVoyage())
        first = await graph_seed.seal_seed(driver, root, bonfire_id=tc.SAMPLE)
        with pytest.raises(BonfireError, match="BonfireMeta"):
            await graph_seed.empty_group(driver, group)  # without --rebuild: refused
        gone = await graph_seed.empty_group(driver, group, batch=3, rebuild=True)
        assert gone > 0 and await count(driver, group) == 0 and await read_meta(driver, group) is None
        await write_rows(driver, rows, group_id=group, embed=tc.FakeVoyage())
        second = await graph_seed.seal_seed(driver, root, bonfire_id=tc.SAMPLE)
        return first, second

    first, second = neo(neo4j_target, go)
    assert first["generation"] != second["generation"] and second["committed_run"] == 0


@pytest.mark.needs_neo4j
def test_rebuild_deletes_the_meta_first_so_a_crash_mid_delete_leaves_no_meta(neo4j_target, tmp_path, monkeypatch):
    from memory_kernel.graph.kg_writer import ensure_constraints, ensure_indexes, write_rows
    from memory_kernel.graph.state_read import read_meta
    from unit.pipeline import test_commit as tc

    group = neo4j_target.group
    root, _, rows = sealable(group, tmp_path, monkeypatch)
    real = graph_seed._scalar

    async def crash_on_batches(driver, query, **params):
        if query == graph_seed._EMPTY_BATCH:
            raise RuntimeError("crash mid-delete")
        return await real(driver, query, **params)

    async def go(driver):
        await ensure_indexes(driver)
        await ensure_constraints(driver)
        await write_rows(driver, rows, group_id=group, embed=tc.FakeVoyage())
        await graph_seed.seal_seed(driver, root, bonfire_id=tc.SAMPLE)
        monkeypatch.setattr(graph_seed, "_scalar", crash_on_batches)
        with pytest.raises(RuntimeError, match="mid-delete"):
            await graph_seed.empty_group(driver, group, rebuild=True)
        monkeypatch.setattr(graph_seed, "_scalar", real)
        return await read_meta(driver, group), await count(driver, group)

    meta, remaining = neo(neo4j_target, go)
    assert meta is None and remaining > 0  # invisible at once, rows left for the next seed to empty


def sealable(group: str, tmp_path: Path, monkeypatch) -> tuple[Path, Any, dict]:
    """A bootstrap root at ``tmp_path`` ready to seal into ``group``: ``(root, its chain, its projected rows)``."""
    from memory_kernel.pipeline import chain as ch
    from unit.pipeline import test_commit as tc

    root = tmp_path / tc.SAMPLE
    chain, rows = bootstrap(group, tc)
    conv_only = {
        "sample_id": tc.SAMPLE,
        "conversation": {"agent_handles": ["@helper"], "session_1": [], "session_1_conversation": "k1"},
    }
    for name, doc in {
        "01-select-001/convkiln-conversation-only.json": conv_only,
        "10-events-001/eventuality.json": {"rolesets": tc.ROLESETS0},
        "11-index-001/index.json": {"recipe_tag": "tag-123"},
        "driver-events.jsonl": {"event": "driver-start", "recipe": {"extraction_model": tc.MODEL}},
        "logs/mk/build-config.json": config(
            tmp_path, sample_id=tc.SAMPLE, group=group, conversation=tmp_path / "seed.json", neo4j_password="s3cret"
        ).to_json(),
    }.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(doc))
    raw = json.loads((root / "logs/mk/build-config.json").read_text())
    raw["openrouter_api_key"] = "s3cret-value"  # a raw secret, not the "<set>" to_json writes
    raw["neo4j_password"] = "s3cret-value"
    (root / "logs/mk/build-config.json").write_text(json.dumps(raw))
    cfg = BuildConfig.from_json(json.loads((root / "logs/mk/build-config.json").read_text()), secrets={})
    assert cfg.stem == "convkiln"
    monkeypatch.setattr(ch, "bootstrap_chain", lambda _root: chain)
    return root, chain, rows


@pytest.mark.needs_neo4j
def test_seal_seed_meta_reads_back(neo4j_target, tmp_path, monkeypatch):
    from memory_kernel.graph.kg_writer import ensure_constraints, ensure_indexes, write_rows
    from memory_kernel.graph.state_read import read_chain, read_meta
    from memory_kernel.pipeline import chain as ch
    from unit.pipeline import test_commit as tc

    group = neo4j_target.group
    root, chain, rows = sealable(group, tmp_path, monkeypatch)

    async def go(driver):
        await ensure_indexes(driver)
        await ensure_constraints(driver)
        await write_rows(driver, rows, group_id=group, embed=tc.FakeVoyage())  # the build's rows: no meta yet
        with pytest.raises(ch.ChainError):
            await read_chain(driver, group)  # invisible until the seal
        before = await count(driver, group)
        sealed = await graph_seed.seal_seed(driver, root, bonfire_id=tc.SAMPLE)
        meta = await read_meta(driver, group)
        back = await read_chain(driver, group)
        return before, sealed, meta, back, await count(driver, group)

    before, sealed, meta, back, after = neo(neo4j_target, go)
    assert meta["committed_run"] == 0 and sealed["committed_run"] == 0
    assert re.fullmatch(r"[0-9a-f]{32}", meta["generation"]) and sealed["generation"] == meta["generation"]
    assert meta["bonfire_id"] == tc.SAMPLE and meta["schema"] == "mk-bonfire/v3"
    assert meta["recipe_tag"] == "tag-123" and json.loads(meta["extraction_model_json"]) == tc.MODEL
    assert meta["agent_handles"] == chain.agent_handles and meta["agent_handle_runs"] == [0] * len(chain.agent_handles)
    assert meta["sessions"] == chain.sessions and meta["explicit_since"] == chain.explicit_since
    assert meta["chat_tagged"] is True
    config_json = json.loads(meta["build_config_json"])
    assert config_json["group"] == group and "neo4j_password" not in config_json
    assert "s3cret-value" not in meta["build_config_json"]
    assert back.sessions == chain.sessions and back.agent_handles == chain.agent_handles and back.run == 0
    assert back.routes == chain.routes and set(back.rolesets) == set(tc.ROLESETS0)
    assert after == before + 1 + len(tc.ROLESETS0) + len(chain.routes)  # the meta, the rolesets and the routes


def test_the_stored_build_config_holds_no_address_and_an_append_takes_its_own(tmp_path, monkeypatch):
    root, _, _ = sealable(GROUP, tmp_path, monkeypatch)
    raw = json.loads((root / "logs/mk/build-config.json").read_text())
    assert raw["neo4j_uri"] and raw["neo4j_user"]  # the build did have them
    _, rows = graph_seed.seal_rows(root, bonfire_id="convkiln", generation="g" * 32)
    meta = next(n for n in rows["nodes"] if "BonfireMeta" in n["labels_neo4j"])["props"]
    stored = json.loads(meta["build_config_json"])
    assert "neo4j_uri" not in stored and "neo4j_user" not in stored and "neo4j_password" not in stored
    # the append's own address comes from its overrides, never from the stored config
    from memory_kernel.pipeline import graph_append

    assert graph_append._settings({"neo4j_uri": "bolt://elsewhere:7687"}).neo4j_uri == "bolt://elsewhere:7687"
    assert graph_append._settings({}).neo4j_uri == "bolt://localhost:7687"


@pytest.mark.needs_neo4j
def test_two_seals_of_a_group_are_two_generations(neo4j_target, tmp_path, monkeypatch):
    """A reseed (the group emptied, built and sealed again) is always another generation, so a commit of a run that
    started before it never lands on it (``pipeline.commit``'s guard)."""
    from memory_kernel.graph.kg_writer import ensure_constraints, ensure_indexes, write_rows
    from memory_kernel.graph.state_read import read_meta
    from unit.pipeline import test_commit as tc

    group = neo4j_target.group
    root, _, rows = sealable(group, tmp_path, monkeypatch)

    async def go(driver):
        await ensure_indexes(driver)
        await ensure_constraints(driver)
        await write_rows(driver, rows, group_id=group, embed=tc.FakeVoyage())
        first = await graph_seed.seal_seed(driver, root, bonfire_id=tc.SAMPLE)
        async with driver.session() as s:  # the bonfire taken down for a rebuild
            await (await s.run("MATCH (n {group_id: $g}) DETACH DELETE n", g=group)).consume()
        await write_rows(driver, rows, group_id=group, embed=tc.FakeVoyage())
        second = await graph_seed.seal_seed(driver, root, bonfire_id=tc.SAMPLE)
        return first, second, await read_meta(driver, group)

    first, second, meta = neo(neo4j_target, go)
    assert first["generation"] != second["generation"] and meta["generation"] == second["generation"]
    assert all(re.fullmatch(r"[0-9a-f]{32}", x["generation"]) for x in (first, second))


def bootstrap(group: str, tc) -> tuple[Any, dict]:
    """The bootstrap's projected, stamped rows (no Roleset / Route / meta rows) and the chain they give."""
    import copy
    from collections import Counter
    from datetime import datetime

    from memory_kernel.graph.g1_projection import project
    from memory_kernel.graph.state import stamp_state
    from memory_kernel.pipeline import chain as ch
    from memory_kernel.pipeline.bonfire import session_date
    from memory_kernel.pipeline.embed import message_texts
    from memory_kernel.pipeline.stages._conversation import build_prepared, build_sources, split_conversation
    from memory_kernel.pipeline.stages._g1_build import CREATED_AT

    conv: dict = {"agent_handles": ["@helper"]}
    chat_of: dict[str, str] = {}
    spec: dict = {}
    for n, (chat, start, msgs) in tc.BOOT.items():
        conv[f"session_{n}_date_time"] = session_date(datetime.fromisoformat(start))
        conv[f"session_{n}_conversation"] = chat
        conv[f"session_{n}"] = tc.turns(start, msgs)
        chat_of |= {m[0]: chat for m in msgs}
        spec |= {m[0]: m[3] for m in msgs}
    conv_only = split_conversation({"sample_id": tc.SAMPLE, "conversation": conv, "qa": []})[0]
    sources = build_sources(build_prepared(conv_only), tc.SCOPE)
    records = tc.graph_records(sources, spec)
    rows = project(
        records, eventuality=tc.eventuality(records), episodes=tc.EPISODES0, group_id=group, created_at=CREATED_AT,
        ingest_pass=tc.PASS, conversation_of=chat_of, referent_policy=tc.POLICY,
    )  # fmt: skip
    counts = Counter(chat_of.values())
    chain = ch.from_bootstrap(
        copy.deepcopy(rows), sources, tc.EPISODES0, chat_of, tc.ROLESETS0,
        walks={c: ch.closed_boundary(k) for c, k in counts.items()}, sessions=len(tc.BOOT),
        agent_handles=conv["agent_handles"],
    )  # fmt: skip
    stamp_state(
        rows,
        run=0,
        records=sources,
        chat_of=chat_of,
        vectors={t: tc.vector(t) for t in message_texts(sources).values()},
    )
    return chain, rows


@pytest.mark.needs_neo4j
def test_bonfire_exists_is_true_only_with_a_bonfire_meta(neo4j_target):
    from memory_kernel import api

    overrides = {
        "neo4j_uri": neo4j_target.uri,
        "neo4j_user": neo4j_target.user,
        "neo4j_password": neo4j_target.password,
    }
    assert api.bonfire_exists(neo4j_target.group, overrides=overrides) is False

    async def put(driver):
        async with driver.session() as s:
            await (await s.run("CREATE (:Entity {group_id: $g, uuid: 'e1'})", g=neo4j_target.group)).consume()

    neo(neo4j_target, put)
    assert api.bonfire_exists(neo4j_target.group, overrides=overrides) is False  # rows without a meta: not a bonfire

    async def meta(driver):
        async with driver.session() as s:
            await (await s.run("CREATE (:BonfireMeta {group_id: $g, uuid: 'm'})", g=neo4j_target.group)).consume()

    neo(neo4j_target, meta)
    assert api.bonfire_exists(neo4j_target.group, overrides=overrides) is True


def test_mk_build_seed_and_the_bonfire_commands_parse():
    from memory_kernel import cli

    p = cli.parser()
    args = p.parse_args(["build", "c.json", "--workdir", "w", "--group", "g", "--seed", "--ops-uri", "mongodb://x"])
    assert args.seed is True and args.ops_uri == "mongodb://x" and args.spend_ceiling_usd is None
    assert p.parse_args(["bonfire", "empty", "g"]).func is cli.cmd_bonfire_empty
    ceiling = p.parse_args(["bonfire", "ceiling", "g", "2.5", "--ops-uri", "mongodb://x"])
    assert ceiling.func is cli.cmd_bonfire_ceiling and ceiling.usd == "2.5"
    assert cli._amount("USD", "2.5") == Decimal("2.5")
    with pytest.raises(ValueError, match="not an amount"):
        cli._amount("USD", "-1")
