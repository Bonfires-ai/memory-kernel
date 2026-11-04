"""Offline stand-ins for ``mk serve``'s tests: a registry that lists bonfires from memory (``BonfireRegistry``'s
``ids`` / ``meta`` / ``watermark``) and a ``Server`` factory that records what it was opened with."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from memory_kernel.serve.answer import ServeConfig

SECRET = "sk-test-not-a-real-key"


class FakeRegistry:
    """The bonfires in memory. ``add(id, run)`` makes a ``BonfireMeta``: its ``query_cache_key`` is ``key<run>``, so a
    server opened from it can tell which committed run it was opened at. ``fail`` makes every call raise."""

    def __init__(self) -> None:
        self.metas: dict[str, dict[str, Any]] = {}
        self.marks: dict[tuple[str, str], dict[str, Any]] = {}
        self.fail: Exception | None = None
        self.meta_reads = 0

    def add(self, bonfire_id: str, run: int = 0) -> FakeRegistry:
        self.metas[bonfire_id] = {
            "group_id": bonfire_id,
            "bonfire_id": bonfire_id,
            "committed_run": run,
            "query_cache_key": f"key{run}",
            "catalogue_json": None,
        }
        return self

    def move(self, bonfire_id: str, run: int) -> None:
        """The bonfire's ``committed_run`` moves (an append committed)."""
        self.add(bonfire_id, run)

    def _check(self) -> None:
        if self.fail is not None:
            raise self.fail

    async def ids(self) -> list[str]:
        self._check()
        return sorted(self.metas)

    async def meta(self, id: str) -> dict[str, Any] | None:
        self._check()
        self.meta_reads += 1
        meta = self.metas.get(id)
        return dict(meta) if meta is not None else None

    async def watermark(self, id: str, chat: str) -> dict[str, Any] | None:
        self._check()
        return self.marks.get((id, chat))


class Fakes:
    """The fake ``Server`` factory: records every config it opens, every context call and every close."""

    def __init__(self) -> None:
        self.opened: list[ServeConfig] = []
        self.closed: list[str] = []
        self.entered = 0  # every context call begun
        self.fail_open: Exception | None = None
        self.fail_context: Exception | None = None
        self.gate: asyncio.Event | None = None
        self.hold: threading.Event | None = None  # an open waits until it is set (set from the test's thread)
        self.refuse: dict[str, Exception] = {}  # bonfire id -> the error its open raises
        self.attempts = 0  # every open tried, failed or not

    def __call__(self, cfg: ServeConfig) -> FakeServer:
        return FakeServer(self, cfg)


class FakeServer:
    def __init__(self, fakes: Fakes, cfg: ServeConfig) -> None:
        self.fakes, self.cfg = fakes, cfg

    async def __aenter__(self) -> FakeServer:
        self.fakes.attempts += 1
        if (refused := self.fakes.refuse.get(self.cfg.group)) is not None:
            raise refused
        while self.fakes.hold is not None and not self.fakes.hold.is_set():
            await asyncio.sleep(0.01)
        if self.fakes.fail_open is not None:
            raise self.fakes.fail_open
        self.fakes.opened.append(self.cfg)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.fakes.closed.append(self.cfg.query_cache_key)

    async def context(self, question: str) -> dict[str, Any]:
        self.fakes.entered += 1
        if self.fakes.gate is not None:
            await self.fakes.gate.wait()
        if self.fakes.fail_context is not None:
            raise self.fakes.fail_context
        return {
            "frame": {"shape": "what"},
            "text": f"[{self.cfg.query_cache_key}] {question}",
            "message_ids": ["D1:1"],
            "tokens": 5,
            "retrieved": ["D1:1"],
        }
