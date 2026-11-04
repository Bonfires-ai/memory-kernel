"""Kernel 2.0 dev receipts (``pipeline.receipts``): ``MK_RECEIPTS=on`` keeps the run dir, the Voyage cache and the Jev
shard of the 2.0 append. The append is the one of ``test_graph_append`` (build and commit stood in), $0."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from memory_kernel.pipeline import graph_append
from memory_kernel.pipeline.bonfire import BonfireError
from memory_kernel.pipeline.receipts import Receipts, receipts_from_env
from unit.pipeline.test_graph_append import bf, leave_crashed_attempt, session_of  # noqa: F401


def test_receipts_off_by_default():
    assert receipts_from_env({}) is None
    assert receipts_from_env({"MK_RECEIPTS": "off", "MK_RECEIPTS_DIR": "/x"}) is None


def test_receipts_on_without_dir_refused():
    for env in ({"MK_RECEIPTS": "on"}, {"MK_RECEIPTS": "on", "MK_RECEIPTS_DIR": " "}):
        with pytest.raises(ValueError, match="MK_RECEIPTS_DIR"):
            receipts_from_env(env)


def test_receipts_paths_live_under_the_bonfire(tmp_path):
    r = receipts_from_env({"MK_RECEIPTS": "on", "MK_RECEIPTS_DIR": str(tmp_path)})
    assert isinstance(r, Receipts)
    assert r.shard("bf") == tmp_path / "bf" / "shard"
    assert r.voyage_cache("bf").parent == tmp_path / "bf"
    assert r.run_dir("bf", 3) == tmp_path / "bf" / "runs" / "0003"


def test_receipts_on_keeps_the_run_dir_and_reuses_the_shard(bf, tmp_path):  # noqa: F811
    r = Receipts(tmp_path / "receipts")
    r.voyage_cache(bf.group).parent.mkdir(parents=True)
    r.voyage_cache(bf.group).write_text(json.dumps({"kept text": [0.5]}))
    make_shard(r.shard(bf.group))
    out = bf.append(receipts=r)
    assert out["status"] == "appended"
    seen = bf.runs[0]
    assert seen.cfg.shard_dir == r.shard(bf.group) and seen.cfg.append_base_usd == "0"  # no shard yet: nothing spent
    assert seen.run_dir == r.run_dir(bf.group, 1) and r.run_dir(bf.group, 1).is_dir()  # kept
    assert Path(seen.forward["voyage_cache"]) == r.voyage_cache(bf.group)
    assert seen.cache["kept text"] == [0.5] and len(seen.cache) > 1  # persisted and still seeded
    assert json.loads(r.voyage_cache(bf.group).read_bytes()) == seen.cache
    assert bf.scratch_left() == []


def test_the_budget_counts_from_the_shards_spend(bf, tmp_path, monkeypatch):  # noqa: F811
    r = Receipts(tmp_path / "receipts")
    make_shard(r.shard(bf.group))
    monkeypatch.setattr(graph_append, "shard_spend", lambda shard: Decimal("1.25") if shard == r.shard(bf.group) else 0)
    bf.append(receipts=r)
    assert bf.runs[0].cfg.append_base_usd == "1.25"


def test_without_receipts_nothing_is_kept(bf):  # noqa: F811
    bf.append()
    assert bf.runs[0].cfg.shard_dir is None and bf.scratch_left() == []


def test_crashed_attempt_is_kept_aside(bf, tmp_path):  # noqa: F811
    r = Receipts(tmp_path / "receipts")
    bf.build_raises = RuntimeError("worker killed")
    with pytest.raises(RuntimeError):
        bf.append(receipts=r)
    crashed = r.run_dir(bf.group, 1)
    assert crashed.is_dir()  # kept at once
    bf.build_raises = None
    out = bf.append(receipts=r)  # the retry takes the same run number and a fresh dir
    assert out["status"] == "appended" and out["run"] == "0001"
    runs = sorted(p.name for p in crashed.parent.iterdir())
    assert runs == ["0001", "0001.attempt-1"]
    assert bf.ops.get(bf.group).held is None


def test_a_killed_attempts_spend_is_settled_then_its_dir_renamed(bf, tmp_path):  # noqa: F811
    r = Receipts(tmp_path / "receipts")
    bf.scratch = tmp_path / "receipts"  # leave_crashed_attempt lays out <scratch>/<id>/runs/<run>
    left = leave_crashed_attempt(bf, "0.3")
    assert left == r.run_dir(bf.group, 1)
    bf.append(receipts=r)
    assert bf.ops.get(bf.group).spent_usd == Decimal("0.55")  # settled as with no receipts: 0.3 + 0.25
    assert (left.parent / "0001.attempt-1").is_dir() and left.is_dir()


def test_a_landed_run_dir_is_not_renamed_by_the_next_append(bf, tmp_path):  # noqa: F811
    r = Receipts(tmp_path / "receipts")
    bf.append(receipts=r)
    bf.set_meta(committed_run=1)  # the stood-in commit landed run 1
    bf.append(session_of(("c1", "c2")), receipts=r)
    assert sorted(p.name for p in r.run_dir(bf.group, 1).parent.iterdir()) == ["0001", "0002"]


def make_shard(shard: Path) -> None:
    (shard / "budget").mkdir(parents=True)
    (shard / "manifest.json").write_text("{}")
    pairs = shard.parent / (shard.name + "-pairs")
    pairs.mkdir()
    (pairs / "pair.json").write_text("{}")


def test_a_fresh_receipts_bonfire_runs_with_its_own_shard_and_adopts_it(bf, tmp_path):  # noqa: F811
    r = Receipts(tmp_path / "receipts")

    def build_made_a_shard(cfg):
        make_shard(cfg.conv_root / "03-boundary-001-ledger")

    bf.on_build = build_made_a_shard
    bf.append(receipts=r)
    cfg = bf.runs[0].cfg
    assert cfg.shard_dir is None and cfg.append_base_usd == "0"  # no preflight against a shard that is not there
    assert (r.shard(bf.group) / "manifest.json").is_file() and (r.shard(bf.group).parent / "shard-pairs").is_dir()
    assert not (cfg.conv_root / "03-boundary-001-ledger").exists()
    bf.set_meta(committed_run=1)
    bf.on_build = None  # the shard exists now: the build does not make one
    bf.append(session_of(("c1", "c2")), receipts=r)  # the next append replays the adopted shard
    assert bf.runs[1].cfg.shard_dir == r.shard(bf.group)


def test_the_shard_is_adopted_even_when_the_run_stopped(bf, tmp_path):  # noqa: F811
    r = Receipts(tmp_path / "receipts")
    bf.on_build = lambda cfg: make_shard(cfg.conv_root / "03-boundary-001-ledger")
    bf.build_raises = RuntimeError("killed after paying")
    with pytest.raises(RuntimeError):
        bf.append(receipts=r)
    assert (r.shard(bf.group) / "manifest.json").is_file()


def test_a_shard_with_spend_lifts_the_ceiling_by_that_spend(bf, tmp_path, monkeypatch):  # noqa: F811
    r = Receipts(tmp_path / "receipts")
    make_shard(r.shard(bf.group))
    monkeypatch.setattr(graph_append, "shard_spend", lambda shard: Decimal(40))
    bf.ops.set_ceiling(bf.group, Decimal("5"))  # headroom 5, far below the shard's 40
    bf.append(receipts=r)
    cfg = bf.runs[0].cfg
    assert cfg.append_base_usd == "40" and Decimal(cfg.spend_ceiling_usd) == Decimal(45)


def test_a_corrupt_kept_voyage_cache_is_refused_before_the_hold(bf, tmp_path):  # noqa: F811
    r = Receipts(tmp_path / "receipts")
    r.voyage_cache(bf.group).parent.mkdir(parents=True)
    r.voyage_cache(bf.group).write_text("{not json")
    with pytest.raises(BonfireError, match="Voyage cache"):
        bf.append(receipts=r)
    assert bf.ops.get(bf.group).held is None and bf.runs == []


def test_a_failing_adoption_leaves_the_spend_settled_and_the_build_error_propagates(bf, tmp_path, monkeypatch):  # noqa: F811
    r = Receipts(tmp_path / "receipts")

    def boom(cfg, shard):
        raise OSError("disk full")

    monkeypatch.setattr(graph_append, "adopt_shard", boom)
    bf.build_raises = RuntimeError("killed after paying")
    with pytest.raises(RuntimeError, match="killed after paying"):
        bf.append(receipts=r)
    rec = bf.ops.get(bf.group)
    assert rec.held is None and rec.spent_usd == Decimal("0.25")


def test_a_manifestless_leftover_shard_dir_is_replaced_by_the_adopted_one(bf, tmp_path):  # noqa: F811
    r = Receipts(tmp_path / "receipts")
    (r.shard(bf.group) / "junk").mkdir(parents=True)
    bf.on_build = lambda cfg: make_shard(cfg.conv_root / "03-boundary-001-ledger")
    bf.append(receipts=r)
    assert (r.shard(bf.group) / "manifest.json").is_file() and not (r.shard(bf.group) / "junk").exists()


def test_an_existing_receipts_shard_is_linked_into_the_run_root_and_passes_the_seal_check(bf, tmp_path):  # noqa: F811
    from memory_kernel.pipeline.runner import STAGE_DIRS
    from memory_kernel.pipeline.stages.index import build_seal

    r = Receipts(tmp_path / "receipts")
    make_shard(r.shard(bf.group))
    qa = tmp_path / "eval-qa.json"
    qa.write_text("{}")
    seen = {}

    def seal_check(cfg):
        link = cfg.conv_root / STAGE_DIRS["shard"]
        seen["link"] = link.is_symlink() and link.resolve() == r.shard(bf.group).resolve()
        seen["missing"] = build_seal(cfg.conv_root, (STAGE_DIRS["shard"],), qa, ledger=cfg.ledger)["stages_missing"]

    bf.on_build = seal_check
    bf.append(receipts=r)
    assert seen == {"link": True, "missing": []}
    assert (r.shard(bf.group) / "manifest.json").is_file() and (r.shard(bf.group).parent / "shard-pairs").is_dir()
    link = r.run_dir(bf.group, 1) / bf.group / STAGE_DIRS["shard"]
    assert link.is_symlink()  # the kept run dir holds the link; adoption did not touch it
    bf.set_meta(committed_run=1)
    bf.append(session_of(("c1", "c2")), receipts=r)  # a second receipts-on append
    assert bf.runs[1].cfg.shard_dir == r.shard(bf.group) and (r.shard(bf.group) / "manifest.json").is_file()


def test_renaming_a_crashed_run_dir_with_the_link_leaves_the_receipts_shard_intact(bf, tmp_path):  # noqa: F811
    r = Receipts(tmp_path / "receipts")
    make_shard(r.shard(bf.group))
    bf.build_raises = RuntimeError("killed")
    with pytest.raises(RuntimeError):
        bf.append(receipts=r)
    bf.build_raises = None
    bf.append(receipts=r)  # the retry renames the crashed dir (with its link) aside, then builds afresh
    runs = r.run_dir(bf.group, 1).parent
    assert (runs / "0001.attempt-1").is_dir()
    assert (r.shard(bf.group) / "manifest.json").is_file() and (r.shard(bf.group) / "budget").is_dir()
    assert (runs / "0001" / bf.group / "03-boundary-001-ledger").is_symlink()
