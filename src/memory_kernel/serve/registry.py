"""The bonfires a Neo4j holds, read from the graph alone (no bonfire directory).

A group is a bonfire when it has a ``BonfireMeta`` node; the group name is the bonfire id. A group that is still
seeding, or a 1.8 group, has none and is never listed. Reads only: the driver may be a ``ReadOnlyDriver``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

IDS = "MATCH (m:BonfireMeta) RETURN m.group_id AS id"
#: the newest message the chat holds: its ``order`` and the select row it was stored from
WATERMARK = (
    "MATCH (e:Episodic {group_id: $g, chat: $c}) WHERE e.order IS NOT NULL "
    "RETURN e.order AS order, e.source_row_json AS row ORDER BY e.order DESC LIMIT 1"
)


class BonfireRegistry:
    """``ids()`` (cached for ``refresh_s``), ``meta(id)`` (always fresh) and ``watermark(id, chat)`` over ``driver``."""

    def __init__(self, driver: Any, *, refresh_s: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.driver = driver
        self.refresh_s = refresh_s
        self.clock = clock
        self._ids: list[str] = []
        self._read_at: float | None = None

    async def _rows(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        async with self.driver.session() as session:
            result = await session.run(cypher, params)
            return [dict(record) async for record in result]

    async def ids(self) -> list[str]:
        """The groups that have a ``BonfireMeta``, sorted; read again once the last read is ``refresh_s`` old."""
        now = self.clock()
        if self._read_at is None or now - self._read_at >= self.refresh_s:
            rows = await self._rows(IDS, {})
            self._ids = sorted({str(r["id"]) for r in rows if r["id"]})
            self._read_at = now
        return list(self._ids)

    async def meta(self, id: str) -> dict[str, Any] | None:
        """The ``BonfireMeta`` properties of bonfire ``id``, or None when the group has none (not a bonfire):
        ``state_read.read_meta``."""
        from memory_kernel.graph.state_read import read_meta

        return await read_meta(self.driver, id)

    async def watermark(self, id: str, chat: str) -> dict[str, Any] | None:
        """How far the bonfire has taken ``chat``: ``{message_id, order, timestamp}`` of its highest-``order``
        Episodic (``message_id`` and ``timestamp`` from the stored select row), or None when it never saw the chat."""
        rows = await self._rows(WATERMARK, {"g": id, "c": chat})
        if not rows:
            return None
        row = json.loads(rows[0]["row"]) if rows[0]["row"] else {}
        order = rows[0]["order"]
        return {
            "message_id": row.get("message_id"),
            "order": None if order is None else int(order),
            "timestamp": row.get("timestamp"),
        }


__all__ = ["BonfireRegistry"]
