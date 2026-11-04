"""An in-memory stand-in for ``NarrowWriter``: the same calls, the same counter arithmetic, no Neo4j. ``calls`` records
every write, so a test can see what reached the "graph" and what stayed in a process."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace

from memory_kernel.runtime.constructions import CxnSpec, Standing
from memory_kernel.runtime.graph_store import Uses, phrase_json, phrase_spec
from memory_kernel.runtime.store import ConstructionStore


class MemoryWriter:
    def __init__(self, catalogue: str | None = None) -> None:
        self.nodes: dict[str, dict] = {}
        self.stored_catalogue = catalogue
        self.calls: list[tuple[str, object]] = []
        self.fail: Exception | None = None

    def _check(self) -> None:
        if self.fail is not None:
            raise self.fail

    def phrases(self) -> list[CxnSpec]:
        return [phrase_spec(self.nodes[k]) for k in sorted(self.nodes)]

    def catalogue(self) -> str | None:
        self._check()
        return self.stored_catalogue

    def update(
        self, *, puts: Iterable[CxnSpec] = (), removes: Iterable[str] = (), confirmed: Iterable[str] = ()
    ) -> None:
        self._check()
        puts, removes, confirmed = list(puts), list(removes), list(confirmed)
        self.calls.append(("update", ([s.id for s in puts], sorted(removes), sorted(confirmed))))
        for i in removes:
            self.nodes.pop(i, None)
        for s in puts:
            node = self.nodes.get(s.id)
            if node is None:
                node = self.nodes[s.id] = {
                    "confirmed": s.standing.confirmed,
                    "uses_ok": s.standing.uses_ok,
                    "uses_failed": s.standing.uses_failed,
                    "uses_world": s.standing.uses_world,
                }
            node["spec_json"] = phrase_json(s)
        for i, d in Counter(confirmed).items():
            if i in self.nodes:
                self.nodes[i]["confirmed"] += d

    def put_phrases(self, specs: Iterable[CxnSpec]) -> None:
        self.update(puts=specs)

    def remove_phrases(self, ids: Iterable[str]) -> None:
        self.update(removes=ids)

    def add_confirmed(self, ids: Iterable[str]) -> None:
        self.update(confirmed=ids)

    def add_uses(self, deltas: Mapping[str, Uses]) -> dict[str, Standing]:
        self._check()
        self.calls.append(("add_uses", dict(deltas)))
        out = {}
        for i, (ok, failed, world) in deltas.items():
            node = self.nodes.get(i)
            if node is None:
                continue
            node["uses_ok"] += ok
            node["uses_failed"] += failed
            node["uses_world"] += world
            out[i] = replace(phrase_spec(node).standing)
        return out

    def set_catalogue(self, raw: str) -> None:
        self._check()
        self.calls.append(("set_catalogue", raw))
        self.stored_catalogue = raw


def memory_store(*rows: CxnSpec, writer: MemoryWriter | None = None, read_only: bool = False) -> ConstructionStore:
    """A store over ``writer`` (a fresh one by default) holding ``rows`` (live ones reach the writer)."""
    store = ConstructionStore(writer or MemoryWriter(), read_only=False)
    for row in rows:
        store.put(row)
    store.read_only = read_only
    return store
