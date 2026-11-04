"""The runtime's state on the graph: taught phrases, their standing counters and the tool catalogue.

``NarrowWriter(driver, group)`` is the serving process's ONLY write path (a sync Neo4j driver). Its statements
(``WRITES``) name two labels and nothing else: a ``TaughtPhrase`` (``id``, ``spec_json`` = the live row's
``CxnSpec.to_json()`` with standing zeroed, and the counters ``confirmed`` / ``uses_ok`` / ``uses_failed`` /
``uses_world``) and the bonfire's ``BonfireMeta``, of which it sets ``catalogue_json`` alone. Counters are only ever
added to (``SET n.uses_ok = coalesce(n.uses_ok, 0) + d``), so two processes counting the same phrase add up.
Reads (``READS``) are ``phrases()`` and ``catalogue()``; ``catalogue()`` is the stored JSON text, or ``None`` until
the first ``set_catalogue``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.runtime.constructions import CxnSpec, Standing

PUT_PHRASES = (
    "UNWIND $rows AS r "
    "MERGE (n:TaughtPhrase {group_id: $g, id: r.id}) "
    "ON CREATE SET n.uuid = r.uuid, n.created_at = r.created_at, n.confirmed = r.confirmed, "
    "n.uses_ok = r.uses_ok, n.uses_failed = r.uses_failed, n.uses_world = r.uses_world "
    "SET n.spec_json = r.spec_json"
)
REMOVE_PHRASES = "MATCH (n:TaughtPhrase {group_id: $g}) WHERE n.id IN $ids DETACH DELETE n"
ADD_CONFIRMED = (
    "UNWIND $rows AS r MATCH (n:TaughtPhrase {group_id: $g, id: r.id}) SET n.confirmed = coalesce(n.confirmed, 0) + r.d"
)
ADD_USES = (
    "UNWIND $rows AS r MATCH (n:TaughtPhrase {group_id: $g, id: r.id}) "
    "SET n.uses_ok = coalesce(n.uses_ok, 0) + r.ok, "
    "n.uses_failed = coalesce(n.uses_failed, 0) + r.failed, "
    "n.uses_world = coalesce(n.uses_world, 0) + r.world "
    "RETURN n.id AS id, coalesce(n.confirmed, 0) AS confirmed, n.uses_ok AS uses_ok, "
    "n.uses_failed AS uses_failed, n.uses_world AS uses_world"
)
SET_CATALOGUE = "MATCH (m:BonfireMeta {group_id: $g}) SET m.catalogue_json = $raw RETURN m.group_id AS g"
#: every statement the serving process writes with
WRITES: tuple[str, ...] = (PUT_PHRASES, REMOVE_PHRASES, ADD_CONFIRMED, ADD_USES, SET_CATALOGUE)

READ_PHRASES = (
    "MATCH (n:TaughtPhrase {group_id: $g}) RETURN n.id AS id, n.spec_json AS spec_json, "
    "coalesce(n.confirmed, 0) AS confirmed, coalesce(n.uses_ok, 0) AS uses_ok, "
    "coalesce(n.uses_failed, 0) AS uses_failed, coalesce(n.uses_world, 0) AS uses_world ORDER BY n.id"
)
READ_CATALOGUE = "MATCH (m:BonfireMeta {group_id: $g}) RETURN m.catalogue_json AS raw"
READ_ROLESETS = "MATCH (n:Roleset {group_id: $g}) RETURN n.roleset AS roleset"
READS: tuple[str, ...] = (READ_PHRASES, READ_CATALOGUE, READ_ROLESETS)

#: a use delta: ``(ok, failed, world)``
Uses = tuple[int, int, int]


def _created_at() -> str:
    from memory_kernel.graph.state import _created_at as created_at

    return created_at()


def phrase_json(spec: CxnSpec) -> str:
    """The ``spec_json`` of a row: its JSON with standing zeroed (the counters live beside it on the node)."""
    return json.dumps(replace(spec, standing=Standing()).to_json(), ensure_ascii=False, sort_keys=True)


def phrase_spec(row: Mapping[str, Any]) -> CxnSpec:
    """The row a ``TaughtPhrase`` holds: its ``spec_json`` with the node's counters as the standing."""
    spec = CxnSpec.from_json(json.loads(row["spec_json"]))
    standing = Standing(
        confirmed=int(row.get("confirmed") or 0),
        uses_ok=int(row.get("uses_ok") or 0),
        uses_failed=int(row.get("uses_failed") or 0),
        uses_world=int(row.get("uses_world") or 0),
    )
    return replace(spec, standing=standing)


class NarrowWriter:
    """The serving process's only write path: ``driver`` is a sync Neo4j driver, ``group`` the bonfire's group."""

    def __init__(self, driver: Any, group: str) -> None:
        self.driver, self.group = driver, group

    # -- reads --------------------------------------------------------------------------------------------------------

    def _read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            return [dict(r) for r in session.run(cypher, g=self.group, **params)]

    def phrases(self) -> list[CxnSpec]:
        """Every stored phrase, sorted by id, its standing the node's counters."""
        return [phrase_spec(r) for r in self._read(READ_PHRASES)]

    def catalogue(self) -> str | None:
        """The bonfire's ``catalogue_json``, read now; ``None`` until the first ``set_catalogue``."""
        rows = self._read(READ_CATALOGUE)
        return rows[0]["raw"] if rows else None

    # -- writes -------------------------------------------------------------------------------------------------------

    def update(
        self, *, puts: Iterable[CxnSpec] = (), removes: Iterable[str] = (), confirmed: Iterable[str] = ()
    ) -> None:
        """Remove, put and add confirmations in ONE transaction (all or nothing). A put of a phrase the graph already
        holds replaces its ``spec_json`` only; a new one starts with its standing's counters."""
        puts, removes = list(puts), sorted(set(removes))
        rows = [
            {
                "id": s.id,
                "uuid": content_uuid("taught", self.group, s.id),
                "created_at": _created_at(),
                "spec_json": phrase_json(s),
                "confirmed": s.standing.confirmed,
                "uses_ok": s.standing.uses_ok,
                "uses_failed": s.standing.uses_failed,
                "uses_world": s.standing.uses_world,
            }
            for s in puts
        ]
        adds = [{"id": i, "d": d} for i, d in sorted(Counter(confirmed).items())]
        if not (rows or removes or adds):
            return

        def tx_fn(tx: Any) -> None:
            if removes:
                tx.run(REMOVE_PHRASES, g=self.group, ids=removes).consume()
            if rows:
                tx.run(PUT_PHRASES, g=self.group, rows=rows).consume()
            if adds:
                tx.run(ADD_CONFIRMED, g=self.group, rows=adds).consume()

        with self.driver.session() as session:
            session.execute_write(tx_fn)

    def put_phrases(self, specs: Iterable[CxnSpec]) -> None:
        self.update(puts=specs)

    def remove_phrases(self, ids: Iterable[str]) -> None:
        self.update(removes=ids)

    def add_confirmed(self, ids: Iterable[str]) -> None:
        """Add one confirmation per occurrence of an id."""
        self.update(confirmed=ids)

    def add_uses(self, deltas: Mapping[str, Uses]) -> dict[str, Standing]:
        """Add use deltas ``{id: (ok, failed, world)}`` to the counters; returns the counters now stored, by id (a phrase
        the graph does not hold is left out)."""
        rows = [{"id": i, "ok": ok, "failed": f, "world": w} for i, (ok, f, w) in sorted(deltas.items())]
        if not rows:
            return {}

        def tx_fn(tx: Any) -> list[dict[str, Any]]:
            return [dict(r) for r in tx.run(ADD_USES, g=self.group, rows=rows)]

        with self.driver.session() as session:
            stored = session.execute_write(tx_fn)
        return {
            r["id"]: Standing(
                confirmed=int(r["confirmed"]),
                uses_ok=int(r["uses_ok"]),
                uses_failed=int(r["uses_failed"]),
                uses_world=int(r["uses_world"]),
            )
            for r in stored
        }

    def set_catalogue(self, raw: str) -> None:
        """Make ``raw`` (the catalogue's JSON text) the bonfire's ``catalogue_json``. ``LookupError`` when the group has no
        ``BonfireMeta``."""

        def tx_fn(tx: Any) -> int:
            return len(list(tx.run(SET_CATALOGUE, g=self.group, raw=raw)))

        with self.driver.session() as session:
            if not session.execute_write(tx_fn):
                raise LookupError(f"{self.group!r} has no BonfireMeta")


def open_sync_driver(uri: str, user: str, password: str) -> Any:
    """A sync Neo4j driver for the narrow writer and the rolesets read."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5, connection_acquisition_timeout=10)


WriterFactory = Callable[[str], NarrowWriter]

__all__ = [
    "READS",
    "WRITES",
    "NarrowWriter",
    "WriterFactory",
    "open_sync_driver",
    "phrase_json",
    "phrase_spec",
]
