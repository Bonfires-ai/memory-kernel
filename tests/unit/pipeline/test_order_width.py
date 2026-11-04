"""Capture directories are named by zero-padded order, four digits at least: an order past 9999 is five digits."""

from __future__ import annotations

import json
from pathlib import Path

from memory_kernel.pipeline.stages import _clause_choice, _parse_capture


def unit(root: Path, order: int, k: int, sha: str) -> None:
    path = root / f"{order:04d}" / f"unit-{k:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"content_sha256": sha, "job": {"nodes": []}}))


def test_the_capture_globs_accept_an_order_past_9999(tmp_path):
    for order in (9999, 10000, 123456):
        unit(tmp_path / "phase-a", order, 0, f"a{order}")
        unit(tmp_path / "phase-b", order, 0, f"b{order}")
        unit(tmp_path, order, 0, f"m{order}")
        (tmp_path / f"{order:04d}" / "complete.json").write_text("{}")
    got = [_parse_capture._order_k(p) for p, _ in _parse_capture._phase_a_rows(tmp_path)]
    assert got == [(9999, 0), (10000, 0), (123456, 0)]  # all found, in numeric order
    status = _parse_capture.status(tmp_path)
    assert (status["phase_a"], status["phase_b"], status["merged_units"], status["complete_messages"]) == (3, 3, 3, 3)


def test_capture_context_indexes_an_order_past_9999(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"capture": {"path": "cap"}}))
    cap = tmp_path / "cap"
    unit(cap, 10000, 1, "s1")
    (cap / "sources.json").write_text(json.dumps([{"order": 10000}]))
    _, index, by_order = _clause_choice.capture_context(run, tmp_path)
    assert index == {"s1": "10000/unit-01.json"} and set(by_order) == {10000}
