"""``mk build --genesis``: a bonfire bootstrapped from a seed of a few Setup sentences.

Such a seed gives zeros a full build's checks refuse: no message needs an antecedent pick (kernel 1.8's sanity is a
receipt and searches nothing). Each check accepts its zero only in a genesis build (``config.genesis_zero_ok``); without
the flag every one keeps its rule. The pairs below
drive the real stage functions (their slow or paid inner steps faked), so each "with genesis" test also proves the
stage asks the predicate. The "without genesis" half of a pair is a regression guard: it passed before the flag
existed and must keep passing.

Also here: the flag is recorded (``build-config.json``, the ``driver-start`` recipe, ``sealed.json``), a root resumed
with the other flag is refused before any stage, and an append of a genesis bonfire is not a genesis build. Offline:
nothing is sent, no engine runs, no Neo4j. Invented names only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_kernel.pipeline import runner
from memory_kernel.pipeline.config import BuildConfig
from memory_kernel.pipeline.stages import graph, index

SAMPLE, GROUP = "conv-kiln", "g-kiln"
PURPOSE = "keep the kiln studio's firing schedule"


def seed(path: Path) -> Path:
    """A genesis seed: one session of two Setup sentences."""
    path.write_text(
        json.dumps(
            {
                "sample_id": SAMPLE,
                "conversation": {
                    "session_1_date_time": "9:00 am on 3 June, 2026",
                    "session_1": [
                        {"speaker": "Setup", "dia_id": "D1:1", "text": "This group runs the kiln studio."},
                        {"speaker": "Setup", "dia_id": "D1:2", "text": "It keeps the firing schedule."},
                    ],
                },
            }
        )
    )
    return path


def config(tmp_path: Path, **kw) -> BuildConfig:
    base = {
        "sample_id": SAMPLE,
        "conversation": seed(tmp_path / "seed.json"),
        "group": GROUP,
        "purpose": PURPOSE,
        "workdir": tmp_path / "workdir",
        "root": tmp_path,
    }
    return BuildConfig(**(base | kw))


# -- the flag ----------------------------------------------------------------------------------------------------------


def test_genesis_is_off_by_default_and_the_predicate_reads_it(tmp_path):
    from memory_kernel.pipeline.config import genesis_zero_ok

    assert config(tmp_path).genesis is False and not genesis_zero_ok(config(tmp_path))
    assert genesis_zero_ok(config(tmp_path, genesis=True))
    assert not genesis_zero_ok(SimpleNamespace(forward_input=None))  # a stage context's stand-in config


def test_a_genesis_build_is_never_an_append(tmp_path):
    for kw in ({"write_mode": "append"}, {"forward_input": tmp_path / "forward.json"}):
        with pytest.raises(ValueError, match="genesis"):
            config(tmp_path, genesis=True, **kw)


# -- 1. picks: the recorded picks plan counted 0 messages --------------------------------------------------------------


def picks_ctx(tmp_path: Path, cfg: BuildConfig, *, plan_messages: int | None) -> SimpleNamespace:
    ctx = SimpleNamespace(cfg=cfg, path=lambda key: tmp_path / key)
    (tmp_path / "graph_a" / "picks").mkdir(parents=True)
    if plan_messages is not None:
        plan = graph.picks_plan_path(ctx)
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps({"messages": plan_messages}))
    return ctx


def test_genesis_accepts_zero_picks(tmp_path):
    ctx = picks_ctx(tmp_path, config(tmp_path, genesis=True), plan_messages=0)
    assert graph.check_picks(ctx) == {"pick_files": 0, "plan_messages": 0}


def test_zero_picks_without_genesis_still_fails(tmp_path):
    ctx = picks_ctx(tmp_path, config(tmp_path), plan_messages=0)
    with pytest.raises(FileNotFoundError, match="no picks"):
        graph.check_picks(ctx)


@pytest.mark.parametrize("plan_messages", [2, None])
def test_genesis_still_needs_the_picks_its_plan_counted(tmp_path, plan_messages):
    """Only the zero the plan recorded is accepted: a plan that counted messages, or no recorded plan, still fails."""
    ctx = picks_ctx(tmp_path, config(tmp_path, genesis=True), plan_messages=plan_messages)
    with pytest.raises(FileNotFoundError, match="no picks"):
        graph.check_picks(ctx)


# -- the record ----------------------------------------------------------------------------------------------------


def sealed(tmp_path: Path, cfg: BuildConfig) -> dict:
    """``seal`` over a hand-made conversation root (every sealed stage dir, the eval-qa)."""
    for d in index.SEALED_DIRS:
        (cfg.conv_root / d).mkdir(parents=True, exist_ok=True)
    (cfg.conv_root / runner.STAGE_DIRS["select"] / f"{cfg.stem}-eval-qa.json").write_text("[]")
    index.seal(runner.Ctx(cfg=cfg, concurrency=1), True)
    return json.loads((cfg.conv_root / runner.STAGE_DIRS["seal"] / "sealed.json").read_text())


def test_genesis_recorded_in_provenance(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "p").mkdir()
    genesis, plain = config(tmp_path / "g", genesis=True), config(tmp_path / "p")
    # build-config.json: the stage children read it back; a real build's carries no new key
    doc = json.loads(json.dumps(genesis.to_json()))
    assert doc["genesis"] is True and BuildConfig.from_json(doc, secrets={}).genesis is True
    assert "genesis" not in plain.to_json()
    assert BuildConfig.from_json(json.loads(json.dumps(plain.to_json())), secrets={}).genesis is False
    # the driver-start recipe
    assert runner.recipe_record(genesis)["genesis"] is True
    assert "genesis" not in runner.recipe_record(plain)
    # the seal: a sealed bootstrap says it was a genesis build; a real build's sealed.json is unchanged
    assert sealed(tmp_path / "g", genesis)["genesis"] is True
    assert "genesis" not in sealed(tmp_path / "p", plain)


# -- resume ----------------------------------------------------------------------------------------------------------


def genesis_problems(cfg: BuildConfig) -> list[str]:
    return [p for p in runner.preflight(cfg) if "genesis" in p]


@pytest.mark.parametrize("started, resumed", [(True, False), (False, True)])
def test_genesis_flag_mismatch_on_resume_is_refused(tmp_path, started, resumed):
    first = config(tmp_path, genesis=started, through="select")
    assert genesis_problems(first) == []  # a root that has not started takes either flag
    runner.write_config(first)  # what every driver start writes before its first stage
    (problem,) = genesis_problems(config(tmp_path, genesis=resumed, through="select"))
    assert f"started with genesis={started}" in problem
    assert genesis_problems(config(tmp_path, genesis=started, through="select")) == []  # the same flag resumes


def test_a_genesis_resume_over_an_unreadable_config_is_refused(tmp_path):
    """A real build's resume over a torn config is not refused (as before: its checks are the strict ones); a genesis
    resume is, since nothing says the root was started as one."""
    cfg = config(tmp_path, through="select")
    cfg.config_path.parent.mkdir(parents=True)
    cfg.config_path.write_text("{")
    assert genesis_problems(cfg) == []
    (problem,) = genesis_problems(config(tmp_path, genesis=True, through="select"))
    assert "does not read" in problem


def test_mk_build_genesis_records_the_flag_and_refuses_a_resume_without_it(child_env, tmp_path):
    from contract.test_cli import SAMPLE_ID, build_args, mk

    args = build_args(tmp_path, through="select", vault=tmp_path / "vault")
    conv_root = tmp_path / "workdir" / SAMPLE_ID
    done = mk(child_env, *args, "--genesis")
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads((conv_root / "logs/mk/build-config.json").read_text())["genesis"] is True
    events = [json.loads(line) for line in (conv_root / "driver-events.jsonl").read_text().splitlines()]
    assert [e["recipe"]["genesis"] for e in events if e["event"] == "driver-start"] == [True]

    refused = mk(child_env, *args)  # the same root without --genesis
    assert refused.returncode == 2, refused.stdout + refused.stderr
    assert "refused" in refused.stdout and "genesis" in refused.stdout
    assert json.loads((conv_root / "logs/mk/build-config.json").read_text())["genesis"] is True  # left as it was

    again = mk(child_env, *args, "--genesis")
    assert again.returncode == 0, again.stdout + again.stderr


# -- appends -----------------------------------------------------------------------------------------------------------


def test_an_append_of_a_genesis_bonfire_is_not_a_genesis_build(tmp_path):
    from memory_kernel.pipeline import bonfire

    base = config(tmp_path, genesis=True)
    run = bonfire.run_config(
        base,
        tmp_path / "runs" / "0001" / SAMPLE,
        conversation=tmp_path / "c.json",
        forward_input=None,
        new_messages=1,
        limit_base_usd=0,
        ceiling_usd=None,
        index_record=tmp_path / "index.json",
        shard_dir=None,
        capture_cache=None,
    )
    assert run.genesis is False and run.write_mode == "append"
    assert "genesis" not in run.to_json()
