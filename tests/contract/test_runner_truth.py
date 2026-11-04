"""What the driver and the CLI say about themselves is true: ``mk inspect`` lists the stages ``mk`` runs, the
per-conversation stop is the one the code enforces, a conversation of a hundred sessions is refused, a shallow frame
dir is refused before the stage that needs it, and the ingest-pass default is neutral.

All in-process and offline; nothing here starts a stage child, reaches Neo4j or spends."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_kernel.pipeline import runner
from memory_kernel.pipeline.config import BuildConfig

#: The 14 stages ``mk`` runs, select to seal (kernel 1.8).
STAGES_MK_RUNS = [
    "select",
    "boundary",
    "capture",
    "graph_picks",
    "graph_a",
    "events",
    "ctx",
    "referents",
    "build_dry",
    "write",
    "rewrite",
    "index",
    "sanity",
    "seal",
]
EVALUATION_STAGES = {"predictions", "unseal", "retrieval", "frames", "answer", "membench"}


def _sessions(n: int) -> dict:
    return {f"session_{k}": [{"speaker": "A", "text": f"line {k}"}] for k in range(1, n + 1)}


def _conversation(tmp_path: Path, n_sessions: int) -> Path:
    path = tmp_path / "conversation.json"
    path.write_text(json.dumps({"sample_id": "conv-syn", "conversation": _sessions(n_sessions)}))
    return path


def _config(tmp_path: Path, **kw) -> BuildConfig:
    return BuildConfig(
        **{
            "sample_id": "conv-syn",
            "conversation": _conversation(tmp_path, 2),
            "group": "g1-test-c2-syn",
            "purpose": "remember the conversation",
            "workdir": tmp_path / "workdir",
            "root": tmp_path,
            "vault": tmp_path,
            **kw,
        }
    )


def test_inspect_lists_the_14_stages_mk_runs_and_no_evaluation_stage(tmp_path):
    assert len(STAGES_MK_RUNS) == 14
    assert tuple(STAGES_MK_RUNS) == runner.ORDER
    conv_root = tmp_path / "workdir" / "conv-syn"
    conv_root.mkdir(parents=True)
    doc = runner.status_at(conv_root, tmp_path / "workdir" / "cost-ledger.jsonl")
    assert list(doc["stages"]) == STAGES_MK_RUNS
    assert not EVALUATION_STAGES & set(doc["stages"])
    leftovers = EVALUATION_STAGES & (set(runner.STAGE_DIRS) | runner.RESUMABLE | runner.PAID)
    assert not leftovers


def test_conv_limit_is_one_and_a_half_times_the_per_message_projection(tmp_path):
    cfg = _config(tmp_path)
    assert runner.conv_limit(cfg, runner.Counts(messages=369, sessions=19)) == pytest.approx(0.34317)
    assert runner.conv_limit(cfg, runner.Counts(messages=0, sessions=1)) == 0.0
    assert runner.projected_share(cfg, runner.Counts(messages=369, sessions=19), "sanity") == 0.0
    assert "sanity" in runner.PAID


@pytest.mark.parametrize("pending_stage", ["write", "rewrite"])
def test_a_shallow_frame_dir_is_refused_when_only_write_or_rewrite_is_pending(tmp_path, pending_stage):
    cfg = _config(tmp_path, through=pending_stage, frame_dirs=(Path("/f"), Path("/tmp")))
    order = runner.ORDER
    for stage in order[: order.index(pending_stage)]:  # every earlier stage is done: only the one under test is pending
        done = runner.marker(cfg, stage)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("{}")
    assert runner.pending_stages(cfg) == [pending_stage]
    problems = runner.preflight(cfg)
    assert any("parent directories" in text and pending_stage in text for text in problems), problems


def test_the_ingest_pass_default_is_neutral(tmp_path):
    assert _config(tmp_path).ingest_pass == "mk-conv-syn-001"
    assert _config(tmp_path, ingest_pass="my-pass").ingest_pass == "my-pass"


def test_an_unknown_through_stage_names_the_range(tmp_path):
    with pytest.raises(ValueError, match="unknown through stage 'answer': mk runs select..seal"):
        _config(tmp_path, through="answer")


# -- transient provider failures: classified, ramped, re-run in the same driver -------------------------------------


def _fail(cfg: BuildConfig, sha: str, status: int) -> None:
    d = cfg.conv_root / "03-boundary-001-ledger" / "budget" / f"jev-{sha}"
    d.mkdir(parents=True)
    (d / "failure.json").write_text(
        json.dumps({"type": "RuntimeError", "reason": f"provider request failed: HTTP {status}"})
    )
    (d / "http-error.json").write_text(json.dumps({"status": status}))


def _drive(tmp_path: Path, attempts: list) -> tuple[dict, list[dict], int]:
    """Run the driver loop through ``select`` with a stage that plays ``attempts`` in order: a status code = fail with
    that provider error, None = succeed, "cap" = a money stop with no failure file."""
    cfg = _config(tmp_path)
    calls = []

    def fake_run_stage(cfg_, counts, stage, **_kw):
        step = attempts[len(calls)]
        calls.append(stage)
        if step is None:
            return {"gate": None}
        if step == "cap":
            raise runner.Stop("build cap: paid 1 + projected 9 > 2")
        _fail(cfg_, f"{len(calls):064x}", step)
        raise runner.Stop(f"{stage} live: child exited rc=1 (see the build log)")

    import unittest.mock as mock

    with mock.patch.object(runner, "run_stage", fake_run_stage):
        out = runner._loop(
            cfg,
            runner.Counts(messages=3, sessions=1),
            through=runner.ORDER[0],
            runner=None,
            clock=lambda: 0.0,
            lock_fd=-1,
        )
    return out, runner.read_events(cfg), len(calls)


@pytest.mark.parametrize("status", [429, 499, 529])
def test_transient_provider_failure_reruns_the_stage(tmp_path, status):
    out, events, calls = _drive(tmp_path, [status, None])
    assert out["status"] == "through" and calls == 2
    kinds = [e["event"] for e in events]
    assert "concurrency-ramp" in kinds and "transient-rerun" in kinds


@pytest.mark.parametrize("attempts", [[400], ["cap"]])
def test_other_stops_are_not_rerun(tmp_path, attempts):
    out, events, calls = _drive(tmp_path, attempts)
    assert out["status"] == "stopped" and calls == 1
    assert "transient-rerun" not in [e["event"] for e in events]


def test_transient_reruns_are_bounded(tmp_path):
    out, events, calls = _drive(tmp_path, [499] * (runner.TRANSIENT_RERUNS + 1))
    assert out["status"] == "stopped" and calls == runner.TRANSIENT_RERUNS + 1


def test_write_projection_is_the_write_stages_own_cost(tmp_path):
    from types import SimpleNamespace

    from memory_kernel.pipeline.stages.write import index_projection_usd

    ctx = SimpleNamespace(counts=runner.Counts(messages=1746, sessions=546))
    projection = index_projection_usd(ctx)
    assert projection == pytest.approx(1.5 * (0.000694 + 0.0 + 0.001319 + 0.0) * 1746 / 369)
    assert projection < 0.02  # the probe's bootstrap: was $4.365, its write+index+sanity cost ~$0.01
