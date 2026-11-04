"""An append's sanity is the receipt: gates, resolution and referents, no search. Invented names only."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest

from memory_kernel.pipeline.stages import receipt
from memory_kernel.pipeline.stages.write import Refused

FWD = {"run": 3, "minted": [{"uuid": "e-new", "key": "kiln#0a1b2c3d", "display": "the kiln", "kind": "entity",
                             "surfaces": ["the kiln"], "run": 3, "order": 0}],
       "routes": {}, "same_as": [], "conflicts": [], "touched": ["e-new", "e-old"], "episodic": ["ep-9"],
       "mentioned": {"e-new": ["m9"]}, "own": 1}  # fmt: skip
SUMMARY = {"append": {"passed": True, "episodic": {"got": 5, "expected": 5, "passed": True},
                      "saga": {"got": 2, "expected": 2, "passed": True}, "created": {"nodes": 3, "edges": 4},
                      "inversion": None, "forward": FWD}}  # fmt: skip
INDEX = {"passed": True, "append_gate": {"passed": True, "keys_equal_group": True}}
DECISIONS = {"mode": "append", "decisions": {"u#0|arg1": {"new": "kiln#0a1b2c3d"}},
             "hard": [{"point": "u#0|arg1", "key": "kiln", "candidates": [["e-old", 0.31]]}],
             "counts": {"points": 1, "hard": 1}}  # fmt: skip


def test_the_receipt_carries_gates_resolution_and_referents():
    out = receipt.build_receipt(write_summary=SUMMARY, index_result=INDEX, decisions=DECISIONS)
    assert out["schema"] == receipt.RECEIPT_SCHEMA and out["passed"] is True
    assert out["gates"]["write"]["episodic"]["passed"] is True and out["gates"]["index"]["passed"] is True
    assert out["minted"] == [{"uuid": "e-new", "key": "kiln#0a1b2c3d", "display": "the kiln", "kind": "entity"}]
    assert out["reused"] == ["e-old"]  # touched minus minted
    assert out["referents"]["discriminated"] == ["kiln#0a1b2c3d"] and out["referents"]["hard"] == DECISIONS["hard"]
    assert out["voyage_requests"] == 0 and out["voyage_tokens"] == 0


def test_a_failed_gate_is_refused():
    bad = {"append": {**SUMMARY["append"], "passed": False}}
    with pytest.raises(Refused, match="write gate"):
        receipt.build_receipt(write_summary=bad, index_result=INDEX, decisions=DECISIONS)
    with pytest.raises(Refused, match="index gate"):
        receipt.build_receipt(write_summary=SUMMARY, index_result={"passed": False}, decisions=DECISIONS)


def test_a_write_without_forward_is_refused():
    with pytest.raises(Refused, match="forward"):
        receipt.build_receipt(write_summary={"append": {"passed": True}}, index_result=INDEX, decisions=DECISIONS)


def test_the_stage_writes_sanity_json_and_reads_no_graph(tmp_path, monkeypatch):
    from memory_kernel.pipeline import graphdb

    monkeypatch.setattr(graphdb, "read_only", lambda _cfg: pytest.fail("the receipt reads no graph"))
    for key, name, body in (("write", "summary.json", SUMMARY), ("index", "index.json", INDEX),
                            ("referents", "decisions.json", DECISIONS)):  # fmt: skip
        (tmp_path / key).mkdir()
        (tmp_path / key / name).write_text(json.dumps(body))
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(sample_id="conv-ember"),
        path=lambda key: tmp_path / key,
        cost=lambda *_a, **_k: contextlib.nullcontext(SimpleNamespace(add=lambda *_a, **_k: None, note="")),
    )
    receipt.append_receipt(ctx)
    assert json.loads((tmp_path / "sanity" / "sanity.json").read_text())["schema"] == receipt.RECEIPT_SCHEMA
