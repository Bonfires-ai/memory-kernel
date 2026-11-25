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
``load_store`` loads them BELOW everything live (first, then the generated frame parts, ``runtime/frames``, and the
generated fillers, ``generated_fillers``, then the live order), compiled at ``constructions.DRAFT_SCORE``, below any
score a live row can have, so a draft never changes what anyone else reads (``admissible`` refuses a reading that
applied someone else's draft) and a draft only reads where nothing live does.

The one generated filler is ``CLAUSE_FILLER``, loaded only when a loaded root (a draft, a live row, a card) has a clause
slot (``{Y:clause}``, ``runtime/teach``): it consumes the slot's whole text and names it as written (``clause-span``),
up to 200 characters of the NORMALISED turn (its ``.{1,200}``: a verb as its marked lemma, tokens joined by one space;
as written that may be more or fewer). It holds no words, is never stored and records no standing; a grammar with no
clause slot loads exactly what it loaded before it existed. A clause filler a store holds itself is a live row, so it
loads after this one and wins the engine's recency tie-break. A stored row with the generated filler's id is not
refused (a store that holds one still opens): it loads after the filler and replaces it in the engine, and the load
logs one warning naming the id.
"""

from __future__ import annotations

import functools
import json
import logging
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from memory_kernel.runtime.base import base_specs
from memory_kernel.runtime.constructions import (
    CONSUME,
    DRAFT_SCORE,
    CompileError,
    CxnSpec,
    compile_spec,
    precedence,
    reserved_slots,
    spec_id,
)
from memory_kernel.runtime.frames import CLAUSE_FILLS, CLAUSE_SPAN, frame_parts

if TYPE_CHECKING:
    import pyfcg

    from memory_kernel.runtime.engine import RuntimeEngine

log = logging.getLogger("memory_kernel.runtime.store")

#: a turn's outcome for one construction: ``ok`` (answered), ``failed`` (a misreading, an empty result, an unresolved
#: name, a constraint or store-kind fail: evidence against the construction), ``world`` (the room lacked something:
#: counted in ``uses_world`` only, never in precedence)
Outcome = Literal["ok", "failed", "world"]
OUTCOMES: tuple[Outcome, ...] = ("ok", "failed", "world")


#: a confirmed row's provenance key (``runtime/teach``): its records of the rows it took the place of, each ``{"id",
#: "kind", "teacher", "by", SUCCESSOR}`` (a replaced row's own records carried along)
REPLACED = "replaced"
#: a ``REPLACED`` record's key: the row that took that row's place (None: none did, a 1.5.0 replace by generalisation)
SUCCESSOR = "successor"


class StoreError(ValueError):
    """A store row breaks a store rule, or the store file is malformed."""


@functools.cache
def base_ids() -> frozenset[str]:
    """The ids of the base set (``base.base_specs``): no stored row may hold one."""
    return frozenset(s.id for s in base_specs())


def _is_outcome(value: object) -> bool:
    return isinstance(value, str) and value in OUTCOMES


class ConstructionStore:
    """The construction rows of one bonfire. Live rows persist through ``writer`` (a ``NarrowWriter``: ``phrases()``,
    ``update(puts=, removes=, confirmed=)``, ``add_uses`` and ``add_choices``, which adds to a row's ``won`` / ``lost``
    counters and returns the standing now stored); staged drafts stay in this process's memory. ``read_only``
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

    def successor(self, cxn_id: str) -> str | None:
        """The row a confirmed replace or modify put in the place of ``cxn_id``: the ``SUCCESSOR`` of the ``REPLACED``
        record of a live row that names it, else None. One step: that row may be gone since (``live_successor``)."""
        for row in self.live():
            for record in row.provenance.get(REPLACED, []):
                if isinstance(record, Mapping) and record.get("id") == cxn_id:
                    after = record.get(SUCCESSOR)
                    return after if isinstance(after, str) else None
        return None

    def live_successor(self, cxn_id: str) -> str | None:
        """The stored live row a chain of confirmed replaces and modifies leads to from ``cxn_id`` (``successor``
        followed, as teach's resolver follows it, past rows gone since), else None: the row a choice against a replaced
        id counts for, and the row a refused modify names."""
        seen = {cxn_id}
        at = self.successor(cxn_id)
        while at is not None and at not in seen:
            row = self._rows.get(at)
            if row is not None and row.status == "live":
                return at
            seen.add(at)
            at = self.successor(at)
        return None

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
                    if cxn_id in base_ids() or any(card.id == cxn_id for card in self._cards):
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
                standing = stored.get(cxn_id) or replace(
                    s, uses_ok=s.uses_ok + ok, uses_failed=s.uses_failed + failed, uses_world=s.uses_world + world
                )
                updated[cxn_id] = replace(spec, standing=standing)
            self._rows = {**self._rows, **updated}
            return updated

    def record_choices(self, pairs: Iterable[tuple[str, str]]) -> dict[str, CxnSpec]:
        """Count a list of choices, each ``(winner, loser)`` as construction names or spec ids, in ONE write, and return
        the updated rows by spec id: the winner's ``won`` and the loser's ``lost`` grow by one (they move the score the
        construction compiles with, ``constructions.entrenched``). An alternate's name counts for its spec. A side whose
        row a confirmed replace or modify took out counts for the live row its chain leads to (``live_successor``: a
        pick on alternatives offered before the confirm, a pick ledger replayed). A side with no stored row and no such
        chain (a base row, a generated filler, an attached catalogue card, an unknown id) is skipped, and a pair whose
        two sides are one spec is skipped whole. The counters are added to on the graph; a returned row carries the
        counters now stored. A staged draft's choice is counted in this process only. Nothing reads differently until
        the grammar is loaded again (``load_store``): an engine keeps the scores it was loaded with."""
        with self._lock:
            deltas: dict[str, list[int]] = {}
            for winner, loser in pairs:
                won, lost = self._choosing(spec_id(winner)), self._choosing(spec_id(loser))
                if won == lost:
                    continue
                if won in self._rows:
                    deltas.setdefault(won, [0, 0])[0] += 1
                if lost in self._rows:
                    deltas.setdefault(lost, [0, 0])[1] += 1
            if not deltas:
                return {}
            self._check_writable()
            live = {i: (d[0], d[1]) for i, d in deltas.items() if self._rows[i].status == "live"}
            stored = self.writer.add_choices(live) if live else {}
            updated: dict[str, CxnSpec] = {}
            for cxn_id, (won, lost) in deltas.items():
                spec = self._rows[cxn_id]
                s = spec.standing
                standing = stored.get(cxn_id) or replace(s, won=s.won + won, lost=s.lost + lost)
                updated[cxn_id] = replace(spec, standing=standing)
            self._rows = {**self._rows, **updated}
            return updated

    def _choosing(self, cxn_id: str) -> str:
        """The row a choice naming ``cxn_id`` counts for: itself when stored, else the live row a replaced record leads
        to; a base row, the generated filler or an attached card (which keep no stored standing) is looked up no
        further, and an id that leads nowhere is returned as it is (and skipped)."""
        if cxn_id in self._rows or cxn_id in base_ids() or cxn_id == CLAUSE_FILLER.id:
            return cxn_id
        if any(card.id == cxn_id for card in self._cards):
            return cxn_id
        return self.live_successor(cxn_id) or cxn_id

    def _check_writable(self) -> None:
        if self.read_only:
            raise StoreError("the store is open read-only")


def _admit(spec: CxnSpec, rows: dict[str, CxnSpec]) -> None:
    if spec.kind == "base":
        raise StoreError(f"{spec.id}: base constructions ship with the kernel and are never stored")
    if spec.id in base_ids():
        raise StoreError(f"{spec.id}: the id names a base construction")
    existing = rows.get(spec.id)
    if existing is not None and existing.kind == "taught" and spec.kind != "taught":
        raise StoreError(f"{spec.id}: a taught row is never overwritten by a {spec.kind} row")


@functools.lru_cache(maxsize=4096)
def _compiled(key: str, score: float | None = None) -> tuple[pyfcg.Construction, ...]:
    return compile_spec(CxnSpec.from_json(json.loads(key)), score=score).cxns


def compiled(spec: CxnSpec, *, draft: bool = False) -> tuple[pyfcg.Construction, ...]:
    """``compile_spec(spec).cxns`` (its alternates), cached on what compilation reads: the row, its ``score`` and its
    ``won`` / ``lost`` counters (the rest of standing, status and provenance do not change it). A row with no choices
    keys the cache exactly as before choices existed. ``draft`` True: a staged row as it loads for everyone, compiled at
    ``DRAFT_SCORE`` (below any live row's score) whatever its own score and counters say (cached apart)."""
    row = spec.to_json()
    row.update(standing=_choices(spec), status="live", provenance=None)
    key = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return _compiled(key, DRAFT_SCORE) if draft else _compiled(key)


def _choices(spec: CxnSpec) -> dict[str, int] | None:
    s = spec.standing
    return {k: v for k, v in (("won", s.won), ("lost", s.lost)) if v} or None


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


#: the clause slot's filler (see the module docstring): generated on load, never stored, never in the base data
CLAUSE_FILLER = CxnSpec(
    id="clause",
    kind="base",
    unit="filler",
    fills=CLAUSE_FILLS,
    form=[CONSUME],
    meaning=[[CLAUSE_SPAN, "?x", "?l", "?r"]],
    status="live",
    provenance={"note": "generated on load when a loaded root has a clause slot: the clause as written"},
)


def generated_fillers(specs: Iterable[CxnSpec]) -> list[CxnSpec]:
    """The fillers ``load_store`` generates for the specs it loads: ``[CLAUSE_FILLER]`` when a root of ``specs`` has a
    clause slot, else none."""
    has_clause_slot = any(
        spec.unit == "root" and any(isinstance(p, dict) and p.get("fills") == CLAUSE_FILLS for p in spec.form)
        for spec in specs
    )
    return [CLAUSE_FILLER] if has_clause_slot else []


def composes(specs: Iterable[CxnSpec]) -> bool:
    """True when any of ``specs`` is a filler that holds a clause slot (a form with a slot that fills a clause): the
    grammar composes, and the engine searches it with the slot-order node test. No bonfire before 2.1 holds one."""
    return any(
        spec.unit == "filler" and any(isinstance(p, dict) and p.get("fills") == CLAUSE_FILLS for p in spec.form)
        for spec in specs
    )


def load_store(store: ConstructionStore, engine: RuntimeEngine) -> int:
    """Load the bonfire's grammar into ``engine`` (replacing it): the staged drafts first (ascending precedence), each
    compiled at ``DRAFT_SCORE``, below any score a live row can have, whatever its own score and counters say (rank is
    the score, and the load order only breaks a tie: a draft loaded first at its own score could still outrank a live
    row), then the frame parts of the live cards and taught phrases (``loaded_frame_parts``) and the generated fillers
    of every root loaded (``generated_fillers``: none for a grammar with no clause slot), then ``store.load_order()``.
    Loaded below every live row, a part never takes a sentence a card or phrase reads completely. The LIVE rows decide
    whether the grammar composes (``composes``: the slot-order node test): a staged draft never changes how anyone
    else's turn is searched, and a teacher's reads through their own composing draft are searched without the node test
    until it is confirmed (every analysis stays reachable: search budget, not readings). Returns the number of specs
    loaded."""
    order = store.load_order()
    staged = sorted(store.staged(), key=precedence)
    parts = loaded_frame_parts(order)
    generated = generated_fillers([*staged, *order])
    _warn_taken_ids(generated, [*staged, *parts, *order])
    _warn_reserved_slots(store.all())
    cxns = [cxn for spec in staged for cxn in compiled(spec, draft=True)]
    cxns += [cxn for spec in [*parts, *generated, *order] for cxn in compiled(spec)]
    engine.load(cxns, slot_order=composes(order))
    return len(staged) + len(parts) + len(generated) + len(order)


def _warn_taken_ids(generated: Sequence[CxnSpec], loaded: Sequence[CxnSpec]) -> None:
    """One warning per generated filler whose id a loaded row also holds: the row loads later and replaces the filler
    in the engine. Nothing is refused or dropped (a store that holds the id still opens and loads); the warning names
    the id, never the row's form."""
    taken = {spec.id for spec in loaded}
    for filler in generated:
        if filler.id in taken:
            log.warning(
                "a loaded construction has the id %r, the generated clause filler's: it replaces that filler in the "
                "engine, so a clause slot reads only through a clause filler of the store's own",
                filler.id,
            )


def _warn_reserved_slots(rows: Iterable[CxnSpec]) -> None:
    """One warning per stored filler with a reserved slot name (``l`` or ``r``: it silently binds the span's edge). The
    row still loads: a bonfire that holds one must not stop opening."""
    for spec in rows:
        if reserved_slots(spec):
            log.warning(
                "the stored filler %r has a slot named %s, reserved for the span's edges: it still loads, but the slot "
                "binds the edge, not a filled phrase",
                spec.id,
                " and ".join(repr(n) for n in dict.fromkeys(reserved_slots(spec))),
            )


__all__ = [
    "CLAUSE_FILLER",
    "OUTCOMES",
    "ConstructionStore",
    "Outcome",
    "StoreError",
    "base_ids",
    "composes",
    "generated_fillers",
    "load_store",
    "loaded_frame_parts",
]
