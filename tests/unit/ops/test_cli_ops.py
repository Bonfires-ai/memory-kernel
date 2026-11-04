"""``mk``'s kernel_ops commands: the address comes from ``--ops-uri`` or ``MK_OPS_URI`` (never argv for a credentialed
URI), ``mk bonfire release`` and ``mk bonfire empty --rebuild``."""

from __future__ import annotations

import json
import subprocess
import sys
from decimal import Decimal

import pytest

from memory_kernel import cli
from memory_kernel.ops.memory import MemoryKernelOps
from memory_kernel.pipeline import graph_seed


@pytest.fixture
def ops(monkeypatch):
    shared = MemoryKernelOps()
    shared.create("b1", ceiling_usd=Decimal("5"), state="ready")
    seen: list[str | None] = []

    def open_ops(uri):
        seen.append(uri)
        return shared

    monkeypatch.setattr(cli, "open_ops", open_ops)
    shared.seen = seen  # type: ignore[attr-defined]
    return shared


def run(*argv: str) -> int:
    args = cli.parser().parse_args(list(argv))
    return args.func(args)


def test_a_missing_ops_extra_is_a_refusal_that_names_the_extra():
    """`mk build --seed` and `mk serve --ops-uri` import the Mongo ops: without pymongo they refuse, with no traceback.
    In a child process, because this imports ``ops.mongo`` with pymongo hidden."""
    code = """
import sys
sys.modules.update({"bson": None, "bson.decimal128": None, "pymongo": None})
from memory_kernel import cli
from memory_kernel.pipeline.bonfire import BonfireError
try:
    cli.open_ops("mongodb://localhost:1/")
except BonfireError as exc:
    print(exc)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "memory-kernel[ops]" in out.stdout


def test_open_ops_falls_back_to_mk_ops_uri(monkeypatch):
    from memory_kernel.ops import mongo

    got: list[str] = []
    monkeypatch.setattr(mongo.MongoKernelOps, "from_uri", classmethod(lambda cls, uri, **kw: got.append(uri) or "ops"))
    monkeypatch.setenv(cli.OPS_URI_ENV, "mongodb://u:pw@host/")
    assert cli.open_ops(None) == "ops" and got == ["mongodb://u:pw@host/"]


def test_serve_reads_mk_ops_uri_when_no_flag(monkeypatch):
    pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    from memory_kernel.ops import mongo
    from memory_kernel.serve import answer

    got: list[str] = []
    monkeypatch.setattr(
        mongo.MongoKernelOps, "from_uri", classmethod(lambda cls, uri, **kw: got.append(uri) or object())
    )

    def stop(_cfg):
        raise ValueError("stop here")  # after the ops are opened, before anything serves

    monkeypatch.setattr(answer, "open_driver", stop)
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv(cli.OPS_URI_ENV, "mongodb://u:pw@host/")
    assert run("serve") == cli.REFUSED
    assert got == ["mongodb://u:pw@host/"]
    got.clear()
    monkeypatch.delenv(cli.OPS_URI_ENV)
    assert run("serve") == cli.REFUSED and got == []  # neither flag nor env: no ops
    assert run("serve", "--ops-uri", "mongodb://flag/") == cli.REFUSED and got == ["mongodb://flag/"]


def test_release_clears_a_stale_lease_and_refuses_a_live_one(ops, capsys):
    ops.take_lease("b1", owner="w", session_id="s", run=1, committed_run=0, ttl_s=600)
    assert run("bonfire", "release", "b1", "--ops-uri", "mongodb://x") == cli.REFUSED
    assert "lease is live" in capsys.readouterr().out and ops.get("b1").lease is not None
    ops.release("b1", owner="w", committed=False)  # released uncommitted: stopped, still naming the session
    assert run("bonfire", "release", "b1") == 0
    assert json.loads(capsys.readouterr().out) == {"id": "b1", "lease": None}
    assert ops.get("b1").lease is None


def test_rebuild_is_refused_under_a_live_lease_and_resets_the_record(ops, monkeypatch, capsys):
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    emptied: list[bool] = []

    inside: list = []

    async def fake_empty(_driver, group, *, rebuild=False, **_kw):
        emptied.append(rebuild)
        inside.append(ops.get("b1"))
        return 7

    class Driver:
        async def close(self):
            return None

    from neo4j import AsyncGraphDatabase

    monkeypatch.setattr(graph_seed, "empty_group", fake_empty)
    monkeypatch.setattr(AsyncGraphDatabase, "driver", staticmethod(lambda *a, **k: Driver()))
    ops.take_lease("b1", owner="w", session_id="s", run=1, committed_run=0, ttl_s=600)
    assert run("bonfire", "empty", "b1", "--rebuild") == cli.REFUSED
    assert emptied == [] and ops.get("b1").state == "ready"
    ops.release("b1", owner="w", committed=True)
    capsys.readouterr()
    assert run("bonfire", "empty", "b1", "--rebuild") == 0
    assert emptied == [True] and ops.get("b1").state == "seeding" and ops.get("b1").lease is None
    assert inside[0].lease is not None and inside[0].lease.owner.startswith("rebuild-")  # held during the delete
    assert ops.get("b1").ceiling_usd == Decimal("5")
    assert json.loads(capsys.readouterr().out)["removed_nodes"] == 7


def test_empty_without_rebuild_keeps_the_meta_refusal_and_never_opens_ops(ops, monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    seen_rebuild: list[bool] = []

    async def fake_empty(_driver, group, *, rebuild=False, **_kw):
        seen_rebuild.append(rebuild)
        return 0

    class Driver:
        async def close(self):
            return None

    from neo4j import AsyncGraphDatabase

    monkeypatch.setattr(graph_seed, "empty_group", fake_empty)
    monkeypatch.setattr(AsyncGraphDatabase, "driver", staticmethod(lambda *a, **k: Driver()))
    assert run("bonfire", "empty", "b1") == 0
    assert seen_rebuild == [False] and ops.seen == []  # type: ignore[attr-defined]
