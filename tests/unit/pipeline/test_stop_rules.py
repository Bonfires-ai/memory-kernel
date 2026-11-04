import pytest

from memory_kernel.pipeline.runner import Stop, _gate_or_stop
from memory_kernel.pipeline.stages import _boundary_frozen as bf


class Cfg:
    def __init__(self, strict: bool):
        self.strict_gates = strict


@pytest.fixture
def no_events(monkeypatch):
    seen = []
    monkeypatch.setattr("memory_kernel.pipeline.runner.log_event", lambda cfg, event, stage, **kw: seen.append(event))
    return seen


def test_failed_advisory_gate_continues(no_events):
    gate = {"name": "boundary", "pass": False, "advisory": True}
    assert _gate_or_stop(Cfg(False), "boundary", {"gate": gate}, "live") == gate
    assert "gate-advisory" in no_events


def test_failed_advisory_gate_stops_when_strict(no_events):
    with pytest.raises(Stop):
        _gate_or_stop(Cfg(True), "boundary", {"gate": {"name": "x", "pass": False, "advisory": True}}, "live")


def test_failed_hard_gate_stops(no_events):
    with pytest.raises(Stop):
        _gate_or_stop(Cfg(False), "write", {"gate": {"name": "x", "pass": False}}, "live")


def test_choice_not_top_uses_top(monkeypatch):
    monkeypatch.setattr(bf, "MODEL", "m")
    request = {"questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}}
    response = {
        "model": "m",
        "answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.2, "b": 0.8}, "confidence": 0.8}},
    }
    out = bf.validate_response(response, request)["q"]
    assert out["choice"] == "b"
    assert out["choice_corrected_from"] == "a"


def test_parts_of_one_message_are_one_boundary_arrival():
    from memory_kernel.pipeline.stages import _boundary_walk as ub
    from memory_kernel.pipeline.stages._conversation import build_prepared, build_sources, split_conversation

    c = {
        "session_1_date_time": "1:00 pm on 1 May, 2026",
        "session_1": [
            {"speaker": "Ana", "dia_id": "D1:1", "text": "hello there"},
            {"speaker": "Bo", "dia_id": "D1:2", "text": "A long paste goes here. " * 50},
        ],
    }
    prepared = build_prepared(split_conversation({"sample_id": "s1", "conversation": c, "qa": []})[0])
    sources = build_sources(prepared, {"bonfire_id": "b1", "conversation_id": "s1"})
    assert [s["message_id"] for s in sources] == ["D1:1", "D1:2#1", "D1:2#2", "D1:2#3"]
    summary = ub.plan(sources, episode_prefix="ep-", window_prefix="w")
    assert summary["same_day_requests"] == 1
