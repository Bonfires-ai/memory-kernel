"""A grammar file and the door that installs it whole (2.1, spec 1a and 1d).

A grammar file is JSON, ``kernel-grammar/v1``: a ``name``, ``rows`` (constructions as the store holds them: ``id``,
``unit``, ``fills``, ``form``, ``meaning``, ``answer_shape``, and an optional ``score`` and, for a root, an optional
``affordance``, the words that raise its frame; never ``kind``, ``status``, ``standing`` or ``provenance``: the door
sets them; never ``as_written``: the kernel sets it on a frame part it compiles) and ``examples`` (a sentence, and the
tool steps its FIRST reading must give, in order, under the file's own scores: at least one step). ``parse_grammar``
reads the file's shape; every row, its affordance included, is checked by ``install``.

``install`` checks the whole file before it writes anything and reports every problem at once (``GrammarError``): the
rows compile by the kernel's own compiler, no filler slot is named ``l`` or ``r``, a stated affordance is one the frame
parts take (``frames.affordance_problems``), each tool a meaning calls is in the registry with that many arguments, ids
are unique and are none the kernel holds or makes itself (a base construction's, the generated clause filler's, a
catalogue card's, one in the frame parts' namespace ``frame.``) nor held by a row this grammar did not install (one
case is no clash: a row of this grammar that teach has modified since, a live ``taught`` row that names the grammar and
holds the id or records it took that id's place, ``replaced``: it stays, the file's row for it is left out and the
result names it, ``kept_taught``), there is at least one example and every example reads as stated. The examples are
read on ``engine`` through a scratch store and runtime holding the LIVE rows the bonfire's store would hold after the
install (a modified row in its file row's place; never a staged draft: a draft is no part of the grammar for anyone
but its teacher). An example whose reading applied a taught row this grammar keeps is not judged against its stated
steps (the row reads what its teacher made it read): it is named in ``Installed.through_taught`` and the install goes
on. When the install keeps taught rows and an example fails, the failing examples are read once more, on a scratch
store with the FILE's rows in the kept rows' places (a form modify can leave an example unread by the grammar as it
will be): one that reads as stated there through a row the install keeps taught is named too (``through_taught``);
any other failing example refuses the file, and with kept taught rows its line names them. ``Installed.examples`` is
the number of examples judged; an install that judges none says so (``Installed.note``). Whatever happens there, the
engine is loaded with the bonfire's own store again before ``install`` returns or raises. Then ONE ``store.update``:
the file's rows put, this grammar's other installed rows removed. A row that stays keeps its standing. A refused file
writes nothing and leaves the engine as it was.

Two errors, never confused: ``GrammarError`` is a refused FILE (a live engine that answered the scratch grammar's load
or an example's read with a refusal is one of its problems: a file whose rows the engine cannot take is a bad file);
``RuntimeEngineError`` is an engine FAULT during the door's work (a timeout, a dead process, a transport failure, a
start refused, a runaway search), raised as the engine raised it, with nothing written (but in one case, which its
message says: the write landed and the engine could not then load the new grammar).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from memory_kernel.runtime.api import Runtime
from memory_kernel.runtime.constructions import CompileError, CxnSpec, Standing, compile_spec, reserved_slots, spec_id
from memory_kernel.runtime.engine import Refused, RuntimeEngine, RuntimeEngineError
from memory_kernel.runtime.executor import VALUE_SPECS, Turn
from memory_kernel.runtime.frames import FRAME_PREFIX, affordance_problems
from memory_kernel.runtime.lexicon import Lexicon, QuestionMap
from memory_kernel.runtime.program import Registry
from memory_kernel.runtime.store import CLAUSE_FILLER, ConstructionStore, base_ids, load_store
from memory_kernel.runtime.teach import GRAMMAR, GRAMMAR_HASH, REPLACED

SCHEMA = "kernel-grammar/v1"
INSTALLED_KIND = "hand"  # an installed row's kind
# the provenance keys GRAMMAR (the grammar's name) and GRAMMAR_HASH (the file's hash) are teach's: a modify keeps them

#: what the door sets: a row of the file may not state them
_DOOR_FIELDS = ("kind", "status", "standing", "provenance")
#: what the kernel sets on a row it compiles itself (a frame part made from a stated word): never an author's
_KERNEL_FIELDS = ("as_written",)
#: ``Installed.note`` when no example was judged
NONE_JUDGED = (
    "no example was judged: each one read through a taught row the grammar keeps, or through the file's row in such a "
    "row's place (through_taught)"
)
#: the fixed turn every example is read on
_SPEAKER, _ADDRESSEE, _CHAT, _AT = "grammar-check", "kernel", "grammar-check", "2026-01-01T00:00:00+00:00"
_MAX_RESUMES = 100


class GrammarError(ValueError):
    """The grammar file, or installing it, is refused. ``problems`` holds every problem found, never only the first."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class Example:
    text: str
    steps: tuple[tuple[str, dict[str, str]], ...]  # (tool, args), in order


@dataclass(frozen=True)
class GrammarFile:
    name: str
    sha256: str  # of the file's bytes
    rows: tuple[dict[str, Any], ...]  # as written: id, unit, fills, form, meaning, answer_shape?, score?
    examples: tuple[Example, ...]


@dataclass(frozen=True)
class Installed:
    name: str
    sha256: str
    added: int
    replaced: int
    removed: int
    kept_taught: tuple[str, ...]  # ids the file holds that teach has since modified: left as they are
    examples: int  # the examples JUDGED (read, and read as stated): the file's examples but those in through_taught
    unexercised: int  # rows put that no judged example's reading applied
    #: the texts of the examples whose reading applied a taught row this grammar keeps (one teach modified since), on
    #: the grammar as it will be, or, for an example that failed there, on the file's rows in the kept rows' places:
    #: named, not judged against their stated steps
    through_taught: tuple[str, ...] = ()
    #: None, or a plain line: ``NONE_JUDGED`` when no example was judged
    note: str | None = None


# -- the file ---------------------------------------------------------------------------------------------------------


def parse_grammar(raw: bytes, source: str) -> GrammarFile:
    """The grammar in ``raw`` (``source`` names it in a refusal). ``GrammarError`` for a file that is not
    ``kernel-grammar/v1``, naming every problem of its shape; the rows themselves are checked by ``install``."""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise GrammarError([f"{source}: not UTF-8 ({exc})"]) from exc
    except json.JSONDecodeError as exc:
        raise GrammarError([f"{source}: not JSON ({exc})"]) from exc
    if not isinstance(doc, dict):
        raise GrammarError([f"{source}: a grammar is a JSON object"])
    problems: list[str] = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"{source}: schema must be {SCHEMA!r}")
    name = doc.get("name")
    if not isinstance(name, str) or not name.strip():
        problems.append(f"{source}: name must be a non-empty string")
    if unknown := sorted(set(doc) - {"schema", "name", "rows", "examples"}):
        problems.append(f"{source}: unknown fields {unknown}")
    rows = doc.get("rows")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        problems.append(f"{source}: rows must be a list of objects")
        rows = []
    examples = _examples(doc.get("examples"), source, problems)
    if problems:
        raise GrammarError(problems)
    return GrammarFile(name, hashlib.sha256(raw).hexdigest(), tuple(rows), examples)


def _examples(raw: object, source: str, problems: list[str]) -> tuple[Example, ...]:
    if not isinstance(raw, list):
        problems.append(f"{source}: examples must be a list")
        return ()
    out: list[Example] = []
    for n, item in enumerate(raw):
        where = f"{source}: example {n}"
        if not isinstance(item, dict) or set(item) != {"text", "steps"}:
            problems.append(f"{where} must be an object of text and steps")
            continue
        steps = item["steps"]
        if not isinstance(item["text"], str) or not item["text"].strip() or not isinstance(steps, list):
            problems.append(f"{where}: text must be a non-empty string and steps a list")
            continue
        if not steps:  # it would pass on any reading, an unread sentence's included
            problems.append(f"{where}: an example states at least one step")
            continue
        parsed: list[tuple[str, dict[str, str]]] = []
        for step in steps:
            args = step.get("args") if isinstance(step, dict) else None
            if (
                not isinstance(step, dict)
                or set(step) != {"tool", "args"}
                or not isinstance(step["tool"], str)
                or not isinstance(args, dict)
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in args.items())
            ):
                problems.append(f"{where}: a step is an object of a tool name and args (text values)")
                break
            parsed.append((step["tool"], dict(args)))
        else:
            out.append(Example(item["text"], tuple(parsed)))
    return tuple(out)


# -- the door ---------------------------------------------------------------------------------------------------------


def install(
    store: ConstructionStore,
    engine: RuntimeEngine,
    grammar: GrammarFile,
    *,
    registry: Registry,
    cards: Iterable[CxnSpec] = (),
) -> Installed:
    """Install ``grammar`` into ``store``, all of it or none of it (see the module docstring). ``cards`` are the
    bonfire's catalogue cards: the examples are read against the whole grammar. ``GrammarError`` names every problem.

    Nothing else may read on ``engine`` while this runs: the self-test loads a scratch grammar into it, so a runtime
    serving turns on the same engine would read them against that grammar. The examples are read with an empty lexicon
    (no rolesets) and an empty graph. The engine ends as it was before the call, decided by whether it was started
    (``RuntimeEngine.started``), not by its process: a started engine (running, or torn down by a fault, which its next
    call would start again) ends running and holding the bonfire's grammar as the store now stands (the new one after a
    write); one that was not started ends not started (the door stops only the engine it started itself). An error
    raised by the self-test or the write is never masked by that restore. ``install`` may be called from a running
    event loop (the examples are then read on a worker thread).

    A live engine that refuses the scratch grammar or an example's read (it answered, and said no) is a problem of the
    file: one problem line, in the ``GrammarError``. Any other ``RuntimeEngineError`` while the door works (a timeout,
    the engine process dead, a transport failure, a start refused, ``Runaway``), the k-th example's read included,
    comes out of ``install`` as that same exception, never as a ``GrammarError``: the store is not written, and the
    engine is restored as above where it can be. One path raises ``RuntimeEngineError`` with the rows WRITTEN: the
    write landed and the engine could not then load the bonfire's new grammar (the restore failed); its message says
    the rows are written, and the store holds them."""
    problems: list[str] = []
    cards = tuple(cards)
    effective = Registry({**registry.specs, **VALUE_SPECS})
    stored = {s.id: s for s in store.all()}
    live = {i: s for i, s in stored.items() if s.status == "live"}  # a staged draft is no part of the install
    modified = _modified(live.values(), grammar.name)
    seen: set[str] = set()
    specs: list[CxnSpec] = []
    #: the file's own row for each id the install keeps taught (never put): read in the kept row's place when an example
    #: fails (``_self_test``)
    kept: dict[str, CxnSpec] = {}
    uncompiled = clashes = 0  # rows the examples cannot be read as installed for
    for n, row in enumerate(grammar.rows):
        label = row["id"] if isinstance(row.get("id"), str) else f"row {n}"
        problems += [f"{label}: the door sets {key}; a row may not state it" for key in _DOOR_FIELDS if key in row]
        problems += [
            f"{label}: {key} is the kernel's (it sets it on a frame part it compiles), never a file's"
            for key in _KERNEL_FIELDS
            if key in row
        ]
        spec = _spec(row, grammar, label, problems)
        if spec is None:
            uncompiled += 1
            continue
        if reserved := reserved_slots(spec):
            problems.append(f"{label}: a filler's slot may not be named {' or '.join(sorted(set(reserved)))}")
        if spec.affordance is not None:
            problems += [f"{label}: {p}" for p in affordance_problems(spec.affordance)]
        problems += _tool_problems(spec, effective)
        clash = _clash(spec, grammar.name, stored, seen, {c.id for c in cards})
        if clash is not None:
            problems.append(f"{label}: {clash}")
            clashes += 1
        elif (spec.id in live and live[spec.id].kind == "taught") or (spec.id not in stored and spec.id in modified):
            kept[spec.id] = spec  # a row of this grammar that teach has modified: it stays
        else:
            specs.append(replace(spec, standing=stored[spec.id].standing) if spec.id in stored else spec)
        seen.add(spec.id)
    if not grammar.examples:
        problems.append("the file has no example: at least one is required")
    puts = specs
    put_ids = {p.id for p in puts}
    removes = [
        s.id for s in stored.values() if _ours(s, grammar.name) and s.kind == INSTALLED_KIND and s.id not in seen
    ]
    readable = not (uncompiled or clashes)
    # the live taught rows of this grammar the install keeps (the door never removes one): an example read through one
    # is named, not judged
    taught = frozenset(s.id for s in live.values() if s.kind == "taught" and _ours(s, grammar.name))
    if grammar.examples and not readable:
        problems.append(f"the examples were not read: {uncompiled} rows do not compile, {clashes} ids clash")
    was_started = engine.started
    written = False
    error: BaseException | None = None
    try:
        applied: set[str] = set()
        through: list[str] = []
        if grammar.examples and readable:
            if not was_started:
                engine.start()  # an engine that cannot start is the caller's error, not a problem of the file
            found, applied, through = _self_test(
                _scratch_rows(stored, removes, put_ids, puts, grammar.name),
                kept,
                grammar,
                engine,
                cards,
                registry,
                taught,
            )
            problems += found
        if problems:
            raise GrammarError(problems)
        replaced = sum(1 for s in puts if s.id in stored)
        store.update(puts=puts, removes=removes)
        written = True
        judged = len(grammar.examples) - len(through)
        return Installed(
            name=grammar.name,
            sha256=grammar.sha256,
            added=len(puts) - replaced,
            replaced=replaced,
            removed=len(removes),
            kept_taught=tuple(kept),
            examples=judged,
            unexercised=sum(1 for s in puts if s.id not in applied),
            through_taught=tuple(through),
            note=None if judged else NONE_JUDGED,
        )
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            _restore(store, engine, was_started)
        except Exception as exc:
            if error is not None:
                error.add_note(f"the engine could not be restored afterwards: {exc}")
            elif written:
                raise RuntimeEngineError(
                    f"the rows of grammar {grammar.name!r} ARE written, but the engine could not load them: {exc}"
                ) from exc
            else:
                raise


def _restore(store: ConstructionStore, engine: RuntimeEngine, was_started: bool) -> None:
    """The engine as it was before the call, holding the store as it now stands: loaded when it was started (running,
    or torn down by a fault before the call; started again first when it has no process: the self-test stopped it, or
    a fault tore it down), stopped when it was not (the engine the door started, also when a fault tore it down: a
    later call does not start it again; an engine never started is left as it is)."""
    if not was_started:
        engine.stop()
        return
    if engine.pid is None:
        engine.start()
    load_store(store, engine)


def _ours(row: CxnSpec, name: str) -> bool:
    """A stored row this grammar installed: its provenance names the grammar and it is a ``hand`` row (the door's own) or
    a ``taught`` one (the door's, modified by teach since)."""
    return row.provenance.get(GRAMMAR) == name and row.kind in (INSTALLED_KIND, "taught")


def _modified(rows: Iterable[CxnSpec], name: str) -> set[str]:
    """The ids of this grammar's rows that teach has modified: a modify (``teach``, 2.1) puts a ``taught`` row under a
    new id in the installed row's place, keeping the grammar's name in its provenance and recording the row it took the
    place of in ``replaced`` (each earlier modify's record carried along)."""
    return {str(record["id"]) for row in rows if row.kind == "taught" and _ours(row, name) for record in _replaced(row)}


def _replaced(row: CxnSpec) -> list[Mapping[str, Any]]:
    """``row``'s records of the rows it took the place of (``REPLACED``) that name an id."""
    return [r for r in row.provenance.get(REPLACED, []) if isinstance(r, Mapping) and isinstance(r.get("id"), str)]


def _scratch_rows(
    stored: Mapping[str, CxnSpec], removes: Sequence[str], put_ids: set[str], puts: Sequence[CxnSpec], name: str
) -> list[CxnSpec]:
    """The bonfire's LIVE rows as the install would leave them, for the self-test (a staged draft is no part of the
    grammar for anyone but its teacher: never in it): every row of this grammar at the file's own score (``won`` =
    ``lost`` = 0: whether a file installs depends on the file, not on what picks have moved since). The write keeps a
    kept row's counters; the scratch read does not."""
    kept = [s for s in stored.values() if s.status == "live" and s.id not in removes and s.id not in put_ids]
    kept = [replace(s, standing=replace(s.standing, won=0, lost=0)) if _ours(s, name) else s for s in kept]
    return kept + [replace(p, standing=replace(p.standing, won=0, lost=0)) for p in puts]


def _in_their_places(after: Sequence[CxnSpec], kept: Mapping[str, CxnSpec], name: str) -> list[CxnSpec]:
    """``after`` (the scratch rows) with the file's own rows ``kept`` (by id: the ids the install keeps taught) in the
    places of the taught rows of this grammar that stand there: the taught row that holds the id, or that records it
    took that id's place (``REPLACED``)."""
    standing = {
        s.id
        for s in after
        if s.kind == "taught"
        and _ours(s, name)
        and (s.id in kept or any(record["id"] in kept for record in _replaced(s)))
    }
    return [*(s for s in after if s.id not in standing), *kept.values()]


def _spec(row: Mapping[str, Any], grammar: GrammarFile, label: str, problems: list[str]) -> CxnSpec | None:
    """The row the door would store, or None (with the reason in ``problems``) when it does not compile. What the door
    or the kernel sets is left out (its problem is said by ``install``)."""
    written = {k: v for k, v in row.items() if k not in _DOOR_FIELDS and k not in _KERNEL_FIELDS}
    provenance = {GRAMMAR: grammar.name, GRAMMAR_HASH: grammar.sha256}
    try:
        spec = CxnSpec.from_json({**written, "kind": INSTALLED_KIND, "status": "live", "provenance": provenance})
        compile_spec(spec)
    except CompileError as exc:
        problems.append(f"{label}: does not compile ({exc})")
        return None
    return spec


def _tool_problems(spec: CxnSpec, registry: Registry) -> list[str]:
    """Each predicate of the meaning is an op of the registry (a tool of the catalogue or a value op) with that many
    arguments: the test ``validate`` makes of a program, applied to one row alone."""
    out = []
    for predicate in spec.meaning:
        name, args = predicate[0], len(predicate) - 1
        if name == "answer":
            continue
        op = registry.specs.get(name)
        if op is None:
            out.append(f"{spec.id}: the meaning calls {name!r}, which is not in the catalogue")
            continue
        fixed = op.arity - 1 if op.variadic else op.arity
        if args < fixed or (args != fixed and not op.variadic):
            out.append(
                f"{spec.id}: {name!r} takes {'at least ' if op.variadic else ''}{fixed} arguments, the meaning gives {args}"
            )
    return out


def _clash(spec: CxnSpec, name: str, stored: Mapping[str, CxnSpec], seen: set[str], cards: set[str]) -> str | None:
    """Why ``spec``'s id may not be installed, or None: it is in the file twice, the kernel holds or makes a row of
    that id itself (a base construction, the generated clause filler, a catalogue card the door is handed, a frame part
    in ``FRAME_PREFIX``'s namespace), or a stored row this grammar did not install holds it."""
    if spec.id in seen:
        return "the id appears twice in the file"
    if spec.id in base_ids():
        return "the id names a base construction"
    if spec.id == CLAUSE_FILLER.id:
        return "the id is the generated clause filler's (the kernel makes that row on load)"
    if spec.id in cards:
        return "the id is a catalogue card's (the catalogue's rows are attached, never stored)"
    if spec.id.partition(".")[0] == FRAME_PREFIX:
        return f"the id is in the frame parts' namespace ({FRAME_PREFIX}.): the kernel generates those rows"
    held = stored.get(spec.id)
    if held is not None and not _ours(held, name):
        return f"the id is held by a stored {held.kind} row that grammar {name!r} did not install"
    return None


# -- the self-test ----------------------------------------------------------------------------------------------------


class _Rows:
    """The scratch store's writer: it only holds rows (nothing is persisted)."""

    def __init__(self, rows: Iterable[CxnSpec]) -> None:
        self.rows = list(rows)

    def phrases(self) -> list[CxnSpec]:
        return self.rows

    def update(
        self, *, puts: Iterable[CxnSpec] = (), removes: Iterable[str] = (), confirmed: Iterable[str] = ()
    ) -> None:
        return None

    def add_uses(self, deltas: Mapping[str, Any]) -> dict[str, Standing]:
        return {}

    def add_choices(self, deltas: Mapping[str, Any]) -> dict[str, Standing]:
        return {}


class _NoGraph:
    async def rows(self, cypher: str, params: Any) -> list[dict]:
        return []


@dataclass(frozen=True)
class _Read:
    """One example read on a scratch grammar (``_walk_examples``)."""

    text: str
    #: the spec ids its first reading applied
    applied: frozenset[str]
    #: its reading applied a row of the walk's ``taught``: named, not judged against its stated steps
    through: bool = False
    #: judged and not as stated: the line that says how (None: as stated, or not judged)
    problem: str | None = None


def _self_test(
    after: Sequence[CxnSpec],
    kept: Mapping[str, CxnSpec],
    grammar: GrammarFile,
    engine: RuntimeEngine,
    cards: Sequence[CxnSpec],
    registry: Registry,
    taught: frozenset[str],
) -> tuple[list[str], set[str], list[str]]:
    """Read the file's examples on ``after`` (the scratch rows): (the problems, the spec ids the JUDGED examples'
    readings applied, the texts of the examples named and not judged, in the file's order). An example whose reading
    applied a row of ``taught`` is named. When the install keeps taught rows (``kept``: the file's own row for each such
    id) and an example fails, the failing examples are read once more on ``after`` with the file's rows in the kept
    rows' places (``_in_their_places``): one that reads as stated there through a kept id is named too (the taught row
    in that place made it fail: a form modify no longer reads it); any other refuses the file, its line naming the kept
    rows. That second scratch load is made only then."""
    refused, reads = _read_examples(after, engine, cards, registry, grammar.examples, taught)
    if refused is not None:
        return [refused], set(), []
    named = {k for k, r in enumerate(reads) if r.through}
    failing = {k: r.problem for k, r in enumerate(reads) if r.problem is not None}
    if failing and kept:
        again_refused, again = _read_examples(
            _in_their_places(after, kept, grammar.name),
            engine,
            cards,
            registry,
            [grammar.examples[k] for k in failing],
        )
        if again_refused is not None:
            return [*failing.values(), again_refused], set(), []
        held = ", ".join(kept)
        for k, read in zip(list(failing), again, strict=True):
            if read.problem is None and not read.applied.isdisjoint(kept):
                named.add(k)
                del failing[k]
            else:
                there = "reads as stated, through none of them" if read.problem is None else "does not read as stated"
                failing[k] += (
                    f" (the grammar keeps the taught rows that took the place of the file's {held}: read with the "
                    f"file's rows there, it {there})"
                )
    applied = {i for k, read in enumerate(reads) if k not in named for i in read.applied}
    return list(failing.values()), applied, [reads[k].text for k in sorted(named)]


def _read_examples(
    after: Sequence[CxnSpec],
    engine: RuntimeEngine,
    cards: Sequence[CxnSpec],
    registry: Registry,
    examples: Sequence[Example],
    taught: frozenset[str] = frozenset(),
) -> tuple[str | None, list[_Read]]:
    """Read each example on ``engine`` through a scratch store holding ``after`` and a runtime over it: (the engine's
    refusal, None when it refused nothing; what each example read, ``_walk_examples``). The runtime's first plan loads
    the scratch grammar into the engine (``install`` restores the engine). It reads the primary analysis alone
    (``analyses=1``: the door checks the first reading's steps, and every engine call of the self-test is then one
    whose failure the plan would not do without) and raises the engine's errors as they are (``raise_engine_errors``):
    a live engine that refuses the scratch grammar or a read is a problem of the file (``_walk_examples``); any other
    ``RuntimeEngineError`` is a fault and leaves here as it is."""
    runtime = Runtime(
        "grammar-self-test",
        engine,
        _NoGraph(),  # type: ignore[arg-type]
        registry,
        store=ConstructionStore(_Rows(after)),
        lexicon=Lexicon([], QuestionMap.load()),
        record_standing=False,
        cards=tuple(cards),
        analyses=1,
        raise_engine_errors=True,
    )
    return _run(_walk_examples(runtime, examples, taught))


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run(coro)``, on a worker thread with its own loop when a loop is already running here."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _walk_examples(
    runtime: Runtime, examples: Sequence[Example], taught: frozenset[str] = frozenset()
) -> tuple[str | None, list[_Read]]:
    """Plan and walk each example on ``runtime`` (one that raises its engine errors): (the engine's refusal, None when
    it refused nothing; what each example read, in order). An example whose reading applied a row of ``taught`` (a
    taught row the grammar keeps reads what its teacher made it read) is named and not judged against its stated
    steps; every other is judged. A live engine's refusal of the scratch grammar (the first plan loads it) or of an
    example's read (``Refused``) ends the walk with one problem, the engine's own words; any other
    ``RuntimeEngineError`` (a fault, at whichever example) is not caught here."""
    reads: list[_Read] = []
    for example in examples:
        turn = Turn(example.text, _SPEAKER, _ADDRESSEE, (_SPEAKER, _ADDRESSEE), _CHAT, None, at=_AT)
        try:
            plan = await runtime.plan(turn, window=0)
        except Refused as exc:  # the engine answered and said no: the file's problem, said once
            return f"the engine refused the grammar the examples are read on ({exc})", reads
        here = frozenset(spec_id(name) for name in (plan.reading or {}).get("applied", ()))
        if here & taught:
            reads.append(_Read(example.text, here, through=True))
            continue
        got: list[tuple[str, dict[str, Any]]] = []
        for _ in range(_MAX_RESUMES):
            if plan.status != "needs_tools":
                break
            got += [(s["tool"], s["args"]) for s in plan.tool_steps]
            results = [
                {"step_id": s["step_id"], "tool": s["tool"], "args": s["args"], "verdict": "committed", "value": None}
                for s in plan.tool_steps
            ]
            plan = await runtime.resume(plan.plan_id, results)
        want = [(tool, dict(args)) for tool, args in example.steps]
        problem = (
            None
            if got == want
            else (
                f"example {example.text!r}: the first reading gives {_show(got)} (it ends {plan.status}), "
                f"the file states {_show(want)}"
            )
        )
        reads.append(_Read(example.text, here, problem=problem))
    return None, reads


def _show(steps: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    return json.dumps([{"tool": tool, "args": args} for tool, args in steps], ensure_ascii=False)


__all__ = [
    "GRAMMAR",
    "GRAMMAR_HASH",
    "INSTALLED_KIND",
    "NONE_JUDGED",
    "SCHEMA",
    "Example",
    "GrammarError",
    "GrammarFile",
    "Installed",
    "install",
    "parse_grammar",
]
