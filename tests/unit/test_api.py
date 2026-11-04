"""The library entry points: ``context`` answers from the bonfire's committed graph (its ``BonfireMeta``; no directory).
Offline: ``Server`` and the driver are fakes; no stage child, no Neo4j, no network. (``append_session`` is covered with
``pipeline.graph_append`` in ``unit/pipeline/test_graph_append.py``.)"""

from __future__ import annotations

from typing import Any

import pytest

from memory_kernel import api
from memory_kernel.pipeline.bonfire import BonfireError

# -- context ----------------------------------------------------------------------------------------------------------


class FakeServer:
    opened: list[Any] = []

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    async def __aenter__(self) -> FakeServer:
        FakeServer.opened.append(self.cfg)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def context(self, question: str) -> dict[str, Any]:
        return {"frame": {"shape": "what"}, "text": f"ctx for {question}", "message_ids": ["D1:1"], "tokens": 7}


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> type[FakeServer]:
    FakeServer.opened = []
    monkeypatch.setattr(api.answer, "Server", FakeServer)
    return FakeServer


def test_context_answers_from_the_graph(fake_server, monkeypatch):
    from memory_kernel.graph.state_read import META
    from unit.serve.test_registry import FakeDriver

    driver = FakeDriver({META: [{"p": {"committed_run": 2, "query_cache_key": "abc123"}}]})
    monkeypatch.setattr(api.answer, "open_driver", lambda cfg: driver)
    out = api.context("b1", "what is warm?", serve_cfg_overrides={"allow_paid": False})
    assert out["run"] == "0002" and out["engine"] == "kernel" and out["text"] == "ctx for what is warm?"
    assert "resolved_query" not in out
    (cfg,) = fake_server.opened
    assert (cfg.group, cfg.query_cache_key, cfg.allow_paid) == ("b1", "abc123", False)
    assert driver.queries == [(META, {"g": "b1"})]  # the bonfire is its BonfireMeta: no directory is read
    said = api.context("b1", "what do I like?", speaker="@Ivo")
    assert said["resolved_query"] == "what do Ivo like?"


def test_context_of_a_group_without_bonfire_meta_is_refused(fake_server, monkeypatch):
    from memory_kernel.graph.state_read import META
    from unit.serve.test_registry import FakeDriver

    monkeypatch.setattr(api.answer, "open_driver", lambda cfg: FakeDriver({META: []}))
    with pytest.raises(BonfireError, match="not a bonfire"):
        api.context("g18", "what is warm?")
    assert fake_server.opened == []


def test_run_label_is_bootstrap_for_run_zero():
    assert [api.run_label(n) for n in (0, 1, 12)] == ["bootstrap", "0001", "0012"]
