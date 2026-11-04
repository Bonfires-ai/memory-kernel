"""A bonfire's construction store: the mined, hand and taught constructions beside the shipped base set.

The live rows live on the graph, one ``TaughtPhrase`` node each (``runtime/graph_store``): the row's AUTHORED form
(literals written naturally) in ``spec_json``, compiled when it is loaded, so a kernel release that changes the
normaliser recompiles every row; its standing is the node's counters. The store reads them once when it opens and
writes through ``writer`` (the ``NarrowWriter``) on every change to a live row; the write is one transaction, and the
store's memory changes only after it lands. Staged rows are drafts (``runtime/teach``): they stay in this process's
memory and never reach the graph. One bonfire may be served by several processes: counters are added to, never set, so
their uses add up.

Rules a write keeps (``StoreError`` otherwise):

* base constructions ship with the kernel (``runtime/base``) and are never stored: no ``kind="base"`` row, and no row
  whose id names a base construction;
* a ``taught`` row is never overwritten by a row of another kind (a re-mine writes ``mined`` rows only), even within
  one ``update`` that also removes it (puts are admitted against the rows before the removals), and a taught row is
  removed only by a confirmed replace (``update(..., replacing_taught=True)``, which ``teach.confirm`` passes).

``record_uses`` counts a turn's uses in one write.

Standing and order: ``record_uses({id: outcome})`` counts a use (``Standing.uses_ok`` / ``uses_failed`` /
``uses_world``; ``world`` never moves ``precedence``) and ``precedence`` reads the first two. ``load_order()`` is the
base set and the live rows in ascending precedence: the highest loads last and wins the engine's recency tie-break (taught over hand over mined over base). Staged rows are drafts (``runtime/teach``):
``load_store`` loads them BELOW everything live (then the generated frame parts, ``runtime/frames``, then the live
order), so a draft never changes what anyone else reads (``admissible``
refuses a reading that applied someone else's draft) and a draft only reads where nothing live does.
"""

from __future__ import annotations

import functools
import json
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from memory_kernel.runtime.base import base_specs
from memory_kernel.runtime.constructions import CompileError, CxnSpec, Standing, compile_spec, precedence, spec_id
from memory_kernel.runtime.frames import frame_parts

if TYPE_CHECKING:
    import pyfcg

    from memory_kernel.runtime.engine import RuntimeEngine

#: a turn's outcome for one construction: ``ok`` (answered), ``failed`` (a misreading, an empty result, an unresolved
#: name, a constraint or store-kind fail: evidence against the construction), ``world`` (the room lacked something:
#: counted in ``uses_world`` only, never in precedence)
Outcome = Literal["ok", "failed", "world"]
OUTCOMES: tuple[Outcome, ...] = ("ok", "failed", "world")


class StoreError(ValueError):
    """A store row breaks a store rule, or the store file is malformed."""


@functools.cache
def _base_ids() -> frozenset[str]:
    return frozenset(s.id for s in base_specs())


def _is_outcome(value: object) -> bool:
    return isinstance(value, str) and value in OUTCOMES


class ConstructionStore:
    """The construction rows of one bonfire. Live rows persist through ``writer`` (a ``NarrowWriter``: ``phrases()``,
    ``update(puts=, removes=, confirmed=)``, ``add_uses``); staged drafts stay in this process's memory. ``read_only``
    (``mk plan``: a process that does not own the bonfire's state) refuses every write (``StoreError``)."""

    def __init__(self, writer: Any, *, read_only: bool = False) -> None:
        self.writer = writer
        self.read_only = read_only
        self._lock = threading.RLock()
        #: the rows the graph held when the store opened (live), then every change this process made
        self._rows: dict[str, CxnSpec] = {}
        for spec in writer.phrases():
            _admit(spec, self._rows)
            self._rows[spec.id] = spec
        self._cards: tuple[CxnSpec, ...] = ()

    def attach_cards(self, cards: Iterable[CxnSpec]) -> None:
        """The catalogue's generated cards: live in this process (``live``, ``load_order``), never stored, never listed
        by ``all``."""
        self._cards = tuple(cards)

    # -- reads --------------------------------------------------------------------------------------------------------

    def all(self) -> list[CxnSpec]:
        """Every row, sorted by id."""
        return [self._rows[k] for k in sorted(self._rows)]

    def live(self) -> list[CxnSpec]:
        """The live rows, sorted by id, then the attached catalogue cards."""
        return [*(s for s in self.all() if s.status == "live"), *self._cards]

    def known_specs(self) -> list[CxnSpec]:
        """Every construction that exists here: the base set, every stored row (drafts included) and the attached cards."""
        return [*base_specs(), *self.all(), *self._cards]

    def staged(self) -> list[CxnSpec]:
        """The staged rows (drafts), sorted by id."""
        return [s for s in self.all() if s.status == "staged"]

    def get(self, cxn_id: str) -> CxnSpec | None:
        return self._rows.get(cxn_id)

    def load_order(self) -> list[CxnSpec]:
        """The base set and the live rows, in ascending ``precedence`` (the highest loads last)."""
        return sorted([*base_specs(), *self.live()], key=precedence)

    # -- writes -------------------------------------------------------------------------------------------------------

    def put(self, spec: CxnSpec) -> None:
        """Add or replace one row (by id)."""
        self.update(puts=[spec])

    def update(
        self, *, puts: Iterable[CxnSpec] = (), removes: Iterable[str] = (), replacing_taught: bool = False
    ) -> None:
        """Remove rows by id and add or replace rows, in ONE atomic write (all or nothing). Every put is admitted
        against the rows as they were BEFORE the removals (removing a taught row never lets a mined one take its id). A
        taught row is removed only with ``replacing_taught`` (a confirmed replace: ``teach.confirm``). Only live rows
        reach the graph; a live row the graph already holds keeps its use counters (they are added to, never set)."""
        with self._lock:
            puts, removes = list(puts), list(removes)
            for cxn_id in removes:
                existing = self._rows.get(cxn_id)
                if existing is None:
                    raise StoreError(f"no stored construction {cxn_id!r} to remove")
                if existing.kind == "taught" and not replacing_taught:
                    raise StoreError(f"{cxn_id}: a taught row is removed only by a confirmed replace")
            for spec in puts:
                _admit(spec, self._rows)
            self._check_writable()
            rows = dict(self._rows)
            for cxn_id in removes:
                del rows[cxn_id]
            persisted_removes = [i for i in removes if self._rows[i].status == "live"]
            live_puts: list[CxnSpec] = []
            confirmed: list[str] = []
            for spec in puts:
                before = self._rows.get(spec.id)
                kept = before is not None and before.status == "live" and spec.id not in removes
                if kept:
                    assert before is not None
                    confirmed += [spec.id] * max(0, spec.standing.confirmed - before.standing.confirmed)
                    spec = replace(
                        spec,
                        standing=replace(
                            before.standing, confirmed=max(spec.standing.confirmed, before.standing.confirmed)
                        ),
                    )
                rows[spec.id] = spec
                if spec.status == "live":
                    live_puts.append(spec)
                elif before is not None and before.status == "live" and spec.id not in removes:
                    persisted_removes.append(spec.id)  # a live row turned into a draft leaves the graph
            if live_puts or persisted_removes or confirmed:
                self.writer.update(puts=live_puts, removes=persisted_removes, confirmed=confirmed)
            self._rows = rows

    def record_use(self, cxn_id: str, outcome: Outcome) -> CxnSpec | None:
        """Count one use of a construction (``record_uses`` for one id)."""
        return self.record_uses({cxn_id: outcome}).get(spec_id(cxn_id))

    def record_uses(self, uses: Mapping[str, Outcome]) -> dict[str, CxnSpec]:
        """Count one use per construction in ONE write (a turn's applied constructions), and return the updated rows by
        spec id: ``ok`` counts ``uses_ok``, ``failed`` ``uses_failed``, ``world`` ``uses_world``. An alternate's name
        (``<id>#v<n>``) counts for its spec. A base construction's use is not stored (base standing changes only with a
        kernel release), nor an attached catalogue card's (no stored row); a name that is none of these is a
        ``KeyError``, and an outcome that is not one of ``OUTCOMES`` a ``StoreError``: nothing is written in either
        case. The counters are added to on the graph; the row returned carries the counters now stored. A staged
        draft's use is counted in this process only."""
        bad = {name: outcome for name, outcome in uses.items() if not _is_outcome(outcome)}
        if bad:
            raise StoreError(f"unknown outcome {bad!r}: one of {', '.join(OUTCOMES)}")
        with self._lock:
            deltas: dict[str, list[int]] = {}
            for name, outcome in uses.items():
                cxn_id = spec_id(name)
                if cxn_id not in self._rows:
                    if cxn_id in _base_ids() or any(card.id == cxn_id for card in self._cards):
                        continue  # base and attached catalogue cards keep no stored standing
                    raise KeyError(f"no construction {cxn_id!r} in the store or the base set")
                delta = deltas.setdefault(cxn_id, [0, 0, 0])
                delta[OUTCOMES.index(outcome)] += 1
            if not deltas:
                return {}
            self._check_writable()
            live = {i: tuple(d) for i, d in deltas.items() if self._rows[i].status == "live"}
            stored = self.writer.add_uses(live) if live else {}
            updated: dict[str, CxnSpec] = {}
            for cxn_id, (ok, failed, world) in ((i, d) for i, d in deltas.items()):
                spec = self._rows[cxn_id]
                s = spec.standing
                standing = stored.get(cxn_id) or Standing(
                    confirmed=s.confirmed,
                    uses_ok=s.uses_ok + ok,
                    uses_failed=s.uses_failed + failed,
                    uses_world=s.uses_world + world,
                )
                updated[cxn_id] = replace(spec, standing=standing)
            self._rows = {**self._rows, **updated}
            return updated

    def _check_writable(self) -> None:
        if self.read_only:
            raise StoreError("the store is open read-only")


def _admit(spec: CxnSpec, rows: dict[str, CxnSpec]) -> None:
    if spec.kind == "base":
        raise StoreError(f"{spec.id}: base constructions ship with the kernel and are never stored")
    if spec.id in _base_ids():
        raise StoreError(f"{spec.id}: the id names a base construction")
    existing = rows.get(spec.id)
    if existing is not None and existing.kind == "taught" and spec.kind != "taught":
        raise StoreError(f"{spec.id}: a taught row is never overwritten by a {spec.kind} row")


@functools.lru_cache(maxsize=4096)
def _compiled(key: str) -> tuple[pyfcg.Construction, ...]:
    return compile_spec(CxnSpec.from_json(json.loads(key))).cxns


def compiled(spec: CxnSpec) -> tuple[pyfcg.Construction, ...]:
    """``compile_spec(spec).cxns`` (its alternates), cached on what compilation reads (standing, status and provenance
    do not change it)."""
    row = spec.to_json()
    row.update(standing=None, status="live", provenance=None)
    return _compiled(json.dumps(row, ensure_ascii=False, sort_keys=True))


def _compiles(spec: CxnSpec) -> bool:
    try:
        compiled(spec)
    except CompileError:
        return False
    return True


def loaded_frame_parts(order: Sequence[CxnSpec]) -> list[CxnSpec]:
    """The frame parts ``load_store`` loads for the load order ``order`` (``frames.frame_parts``: generated, never
    stored; a part whose word does not compile is left out). The runtime reports frames from these and no others."""
    return [part for part in frame_parts(order) if _compiles(part)]


def load_store(store: ConstructionStore, engine: RuntimeEngine) -> int:
    """Load the bonfire's grammar into ``engine`` (replacing it): the staged drafts first (ascending precedence), then
    the frame parts of the live cards and taught phrases (``loaded_frame_parts``), then ``store.load_order()``. Loaded
    below every live row, a part never takes a sentence a card or phrase reads completely. Returns the number of specs
    loaded."""
    order = store.load_order()
    parts = loaded_frame_parts(order)
    specs = [*sorted(store.staged(), key=precedence), *parts, *order]
    engine.load([cxn for spec in specs for cxn in compiled(spec)])
    return len(specs)


__all__ = [
    "OUTCOMES",
    "ConstructionStore",
    "Outcome",
    "StoreError",
    "load_store",
    "loaded_frame_parts",
]
