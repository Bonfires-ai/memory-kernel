"""``POST /v1/bonfires/{id}/entities/sheet``: who is X, each summary line with its source message ids. A fake Neo4j
reader and a fake ``Server``; no network, no paid call."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from memory_kernel.search import kg_reader as kr
from memory_kernel.serve import http
from unit.serve.fakes import SECRET, FakeRegistry, Fakes, FakeServer

FIRE = "ember"

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        async def gen():
            for r in self.rows:
                yield r

        return gen()


class _Graph:
    """Ana (a Person, alias "Dee") joined by SAME_AS to Anabel; both hold lines sourced from messages."""

    NODES = {
        "u-ana": {
            "uuid": "u-ana", "name": "Ana", "labels": ["Entity", "Person"], "summary": "Ana is a chemist.\nAna moved.",
            "line_ids": ["l1", "l2"], "line_texts": ["Ana is a chemist.", "Ana moved."], "identity_line_ids": ["l1"],
            "sources": ["e1", "e2"], "identity_aliases": ["Dee"],
        },
        "u-anabel": {
            "uuid": "u-anabel", "name": "Anabel", "labels": ["Entity"], "summary": "Ana is a chemist.",
            "line_ids": ["l3"], "line_texts": ["Ana is a chemist."], "identity_line_ids": [],
            "sources": ["e3"], "identity_aliases": None,
        },
    }  # fmt: skip
    EPISODES = {"e1": "m-1", "e2": "m-2", "e3": "m-3"}

    def session(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, cypher, **p):
        if cypher == kr.SHEET_BY_NAME_CYPHER:
            return _Rows([{"uuid": u} for u, n in self.NODES.items() if str(n["name"]).lower() == p["name"].lower()])
        if cypher == kr.IDENTITY_ALIASES_CYPHER:
            return _Rows([n for n in self.NODES.values() if n["identity_aliases"]])
        if cypher == kr.SAME_AS_COMPONENT_CYPHER:
            if not set(p["uuids"]) & set(self.NODES):
                return _Rows([])
            a, b = self.NODES["u-ana"], self.NODES["u-anabel"]
            rows = []
            for s, t in ((a, b), (b, a)):
                rows.append({
                    "start": s["uuid"], "start_name": s["name"], "start_minted_order": [0, 1],
                    "uuid": t["uuid"], "name": t["name"], "minted_order": [0, 2], "edges": [["u-anabel", "u-ana"]],
                })  # fmt: skip
            return _Rows([r for r in rows if r["start"] in p["uuids"]])
        if cypher == kr.SHEET_VIEW_CYPHER:
            return _Rows([self.NODES[u] for u in p["uuids"] if u in self.NODES])
        if cypher == kr.SOURCE_MESSAGE_IDS_CYPHER:
            return _Rows([{"uuid": u, "message_id": self.EPISODES[u]} for u in p["uuids"] if u in self.EPISODES])
        if cypher == kr.TIMELINE_CYPHER:
            return _Rows([
                {
                    "uuid": "ev1", "name": "Ana moved", "happened_at": "2026-05-01", "told_at": "2026-05-02",
                    "time_source": "explicit", "entities": ["u-ana"], "attributes": None,
                },
                {
                    "uuid": "ev2", "name": "Ana asked", "happened_at": "2026-05-03", "told_at": "2026-05-03",
                    "time_source": "told_default", "asserted": False, "entities": ["u-anabel"], "attributes": None,
                },
            ])  # fmt: skip
        raise AssertionError(f"unexpected query: {cypher[:60]}")


def test_entity_sheet_returns_lines_with_sources_and_identities():
    for query in ({"name": "ana"}, {"name": "Dee"}, {"uuid": "u-anabel"}):
        out = asyncio.run(kr.entity_sheet_view(_Graph(), "g", **query))
        assert len(out) == 1, query
        sheet = out[0]
        assert sheet["uuid"] == "u-ana" and sheet["name"] == "Ana"  # the SAME_AS sink is canonical
        assert sheet["identities"] == ["u-ana", "u-anabel"]
        assert sheet["kinds"] == ["Person"]
        assert sheet["aliases"] == ["Dee", "Anabel"]
        # the duplicate line is one line, with the message ids of both sources
        assert sheet["lines"] == [
            {"text": "Ana is a chemist.", "message_ids": ["m-1", "m-3"], "episode_ids": ["e1", "e3"]},
            {"text": "Ana moved.", "message_ids": ["m-2"], "episode_ids": ["e2"]},
        ]
        assert sheet["summary"] == "Ana is a chemist.\nAna moved."
        # a told_default day is never shown as happened-on; an unasserted event keeps its flag
        assert [(t["name"], t["happened_at"], t["asserted"]) for t in sheet["timeline"]] == [
            ("Ana moved", "2026-05-01", None),
            ("Ana asked", None, False),
        ]
    assert asyncio.run(kr.entity_sheet_view(_Graph(), "g", name="Nobody")) == []


class _SheetServer(FakeServer):
    async def entity_sheet(self, *, name=None, uuid=None):
        return await kr.entity_sheet_view(_Graph(), "g", name=name, uuid=uuid)


class _SheetFakes(Fakes):
    def __call__(self, cfg):
        return _SheetServer(self, cfg)


def test_entity_sheet_route_statuses():
    fakes = _SheetFakes()
    app = http.create_app(
        FakeRegistry().add(FIRE),
        serve_cfg_overrides={"allow_paid": False, "openrouter_api_key": SECRET},
        server_factory=fakes,
    )
    url = f"/v1/bonfires/{FIRE}/entities/sheet"
    with contextlib.ExitStack() as stack:
        client = stack.enter_context(TestClient(app))
        ok = client.post(url, json={"name": "Dee"})
        assert ok.status_code == 200
        assert ok.json()["entities"][0]["lines"][0] == {
            "text": "Ana is a chemist.",
            "message_ids": ["m-1", "m-3"],
            "episode_ids": ["e1", "e3"],
        }
        assert client.post(url, json={"name": "  Dee ", "uuid": None}).status_code == 200
        miss = client.post(url, json={"name": "Nobody"})
        assert miss.status_code == 404 and miss.json()["entities"] == []
        for bad in ({}, {"name": ""}, {"name": 3}, {"name": "a", "uuid": "b"}, {"name": "  "}, [1]):
            r = client.post(url, json=bad)
            assert r.status_code == 422 and r.json()["entities"] == [], bad
        assert client.post(url, content=b"not json").status_code == 422
        assert client.post("/v1/bonfires/nope/entities/sheet", json={"name": "x"}).status_code == 404
