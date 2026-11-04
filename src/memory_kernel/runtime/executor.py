"""Program executor: value and memory operations run here, tool steps suspend the run.

The kernel never executes a tool. ``execute`` runs a program's value and memory operations in dependency order and stops
at the first tool step whose inputs are bound, returning it as a :class:`ToolStep`. The agent runtime runs the tool and
calls ``execute`` again with ``resumed`` (``step_id`` -> result); the run replays from the start, substituting the
results it already has, and continues to the next tool step or to the end. ``prior`` (the previous execution's
``values``) lets the replay reuse the outputs of data ops that already completed instead of running them again.

Control structure follows the spike meaning IR: ``procedure(?p, name)`` names a procedure, ``seq(?p, i, ?step)`` orders
its steps (a step is a tool op's result variable or a nested procedure; ``seq`` order is a contract: the first
unresolved step of a procedure blocks every later step of it, while tool ops outside any procedure may still run), and ``distribute(?p, ?set)`` repeats ``?p``
once per member of ``?set`` with the member bound in place of ``?set``. Distribution is explicit only: a set that no
``distribute`` scopes reaches its readers whole. Inside a distributed procedure, the data ops that depend on the set or
on its tool steps' results, and feed its tool steps, also run once per member; a variable written inside the region
reads, from outside it, as a list in member order.
"""

from __future__ import annotations

import asyncio
import heapq
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from itertools import product
from typing import Any, Protocol

from memory_kernel.runtime.constraints import check_constraint
from memory_kernel.runtime.normalise import NPCut, marked_lemma
from memory_kernel.runtime.policy import DEFAULT_POLICY, ConfirmationPolicy
from memory_kernel.runtime.program import VARIADIC, Const, OpSpec, Program, Registry, ToolParam, Var, validate

# Op argument values of ``deixis`` (turn roles), not natural-language words.
_DEIXIS_ROLES = ("speaker", "addressee")
#: the ``window`` keyword's rules text (the keyword glossary reads it)
WINDOW_RULES = (
    "window(referent ← index): the referent of the turn `index` turns back in the chat's turn window (latest first, "
    "0 = the latest): the entity that turn's answer named, as a Referent. The window holds the chat's last plans; a "
    "turn no root construction read (an unread turn) is not in it; plans leave it 15 minutes after their last use or "
    "past 256, and when the chat's latest plan has left it the window is empty. A turn that did not answer (a repair, "
    "tools still pending) is in it, with no referent. Refused before the program runs when the window holds no turn "
    "`index` back, or when that turn's answer named no entity or more than one (an ambiguous reference is never "
    "resolved by guessing). find-entity returns a Referent as it is, with no query (no other value passes through): it "
    "is never a name looked up again. Reads nothing from the graph."
)
#: the ``either`` keyword's rules text (the keyword glossary reads it)
EITHER_RULES = (
    "either(value ← first, second): the first of two results unless it is empty, else the second. Empty is nothing "
    "(None), or an empty list, set, map or string; anything else, a 0 or a false included, is a result. Both inputs "
    "are computed before it runs (a program runs every op whose inputs are bound): it chooses between two paths, it "
    "never skips one. A wh root uses it to answer from the subject's events first and from its state lines only when "
    "the events give nothing."
)


def empty(value: Any) -> bool:
    """``either``'s emptiness: None, or an empty str, list, tuple, dict or set."""
    return value is None or (isinstance(value, (str, list, tuple, dict, set, frozenset)) and not value)


# Ops without outputs that only shape the order of tool steps; they never run and are not traced.
_STRUCTURAL = frozenset({"seq", "distribute"})

VALUE_SPECS: Mapping[str, OpSpec] = {
    spec.name: spec
    for spec in (
        OpSpec("name-span", ("left", "right"), ("text",), "value"),
        OpSpec("lemma-span", ("left", "right"), ("lemma",), "value"),
        OpSpec("within", ("events", "interval"), ("events",), "value"),
        OpSpec("const", ("value",), ("constant",), "value"),
        OpSpec("either", ("first", "second"), ("value",), "value"),
        OpSpec("gather", (f"value{VARIADIC}",), ("values",), "value"),
        OpSpec("deixis", ("role",), ("person",), "value"),
        OpSpec("everyone", (), ("people",), "value"),
        OpSpec("window", ("index",), ("referent",), "value"),
        OpSpec("definite", ("referent",), (), "value"),
        OpSpec("indefinite", ("referent",), (), "value"),
        OpSpec("plural", ("referent",), (), "value"),
        OpSpec("procedure", ("name",), ("procedure",), "control"),
        OpSpec("seq", ("procedure", "index", "step"), (), "control"),
        OpSpec("distribute", ("procedure", "set"), (), "control"),
        # a reference to a taught construction: replaced by its live program before validation (``teach.expand``)
        OpSpec("invoke", ("construction", "slot…"), ("step",), "control"),
    )
}


class SpanText(Protocol):
    """The turn's text as the runtime grammar read it: ``normalise(text).spans`` (``runtime.normalise.TokenSpans``)."""

    #: the normalised text the engine comprehended (slot offsets index it)
    normalised: str

    def cut(self, left: int, right: int) -> str:
        """The original surface text for normalised offsets ``[left, right)`` (the NP cut: leading determiners
        dropped)."""
        ...

    def np(self, left: int, right: int) -> NPCut:
        """The NP cut and whether a quantifying determiner was dropped from its head (``NPCut.quantified``: a
        constraint no keyword applies, which an answer must not ignore)."""
        ...


class MemoryOps(Protocol):
    async def run(self, op: str, args: Mapping[str, Any]) -> Any:
        """Run one memory operation (read-only) and return its output value."""
        ...


@dataclass(frozen=True)
class Turn:
    text: str
    speaker: str | None
    addressee: str | None
    participants: tuple[str, ...]
    chat_id: str
    message_id: str | None
    #: when the turn was said: an ISO 8601 timestamp (an offset-less one is read as UTC); ``within`` needs it
    at: str | None = None
    #: the chat's turn window, latest first: per earlier turn, the entities its answer named (``window``); filled by
    #: the runtime, never by the caller
    window: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True)
class Referent:
    """An entity the turn window resolved (``window``'s output): ``find-entity`` returns it as it is, with no query."""

    uuid: str
    name: str | None
    kinds: tuple[str, ...] = ()

    @classmethod
    def of(cls, entity: Mapping[str, Any]) -> Referent:
        return cls(str(entity["uuid"]), entity.get("name"), tuple(entity.get("kinds") or ()))


@dataclass(frozen=True)
class TraceStep:
    op: str
    args: dict[str, Any]
    out: Any
    ms: float
    ok: bool
    reused: bool = False  # output taken from ``prior`` values, the op did not run again


class Committed(Protocol):
    """Tool results already committed in the world (by tool and arguments), from before this plan: a step that matches
    one is resolved with its value and never issued."""

    def holds(self, tool: str, args: Mapping[str, Any]) -> bool: ...

    def take(self, tool: str, args: Mapping[str, Any]) -> tuple[bool, Any]:
        """Use up the entry for ``tool(args)``: ``(True, value)``, or ``(False, None)`` when none is left."""
        ...


@dataclass(frozen=True)
class ToolStep:
    step_id: str
    tool: str
    args: dict[str, Any]
    side_effect: bool | None
    needs_confirmation: bool
    #: the step's actor (the turn's speaker): the policy's consent check compares it with the speaker
    actor: str | None
    #: the lexical (WordNet) check of each argument whose parameter declares a ``constraint``: ``{"param", "constraint",
    #: "result"}`` with result pass / fail / unknown. Advice for the game's Resolve, which is authoritative: never a block
    checks: tuple[dict[str, str], ...] = ()


@dataclass
class Execution:
    values: dict[str, Any]
    trace: list[TraceStep]
    pending: tuple[ToolStep, ...]
    done: bool
    problems: list[str] = field(default_factory=list)


# Date-bearing keys of entities, events and messages, most specific first (property names, not vocabulary).
DATE_KEYS = ("happened_at", "valid_at", "told_at")
#: ``within``'s interval: a unit (``D`` a calendar day, ``W`` an ISO week, Monday to Sunday) and a signed offset from
#: the turn's own day or week (``D-1`` the day before, ``W-1`` the week before, ``D0`` the turn's day), in any case; days
#: and weeks are counted at the turn's fixed UTC offset (the offset its ``at`` carries, UTC when it carries none)
_INTERVAL = re.compile(r"([dw])([+-]?\d{1,4})", re.IGNORECASE)
_FRACTION = re.compile(r"(\.\d{6})\d+")
#: the ``within`` keyword's rules text (the keyword glossary reads it)
WITHIN_RULES = (
    "within(events ← events, interval): the events, in their order, whose date falls in the interval. An item's date "
    f"is its first non-empty {', '.join(DATE_KEYS)}; an undated or unreadable item is dropped. The interval (a time "
    "filler's constant) is `D<offset>` (a calendar day) or `W<offset>` (an ISO week, Monday to Sunday), a signed offset from the "
    "turn's own day or week (`D0` the turn's day, `D-1` the day before, `W-1` the week before), counted at the fixed "
    "UTC offset of the turn's time `at` (UTC when it carries none). A date with no time is that whole day; a timestamp "
    "with no offset is read at the turn's offset. Refused before the program runs when the turn has no time. Fails on "
    "an interval that is not D/W<offset> or a turn time that is not ISO 8601. Reads nothing from the graph."
)


def item_date(item: Any) -> str | None:
    """An item's date as stored: its first non-empty ``DATE_KEYS`` property, else None."""
    if isinstance(item, Mapping):
        for key in DATE_KEYS:
            if item.get(key):
                return str(item[key])
    return None


def interval(text: str, at: str) -> tuple[datetime, datetime]:
    """``[start, end)`` of the interval ``text`` names, at the fixed UTC offset of the turn's time ``at``.
    ``ValueError`` for a ``text`` that is no interval or an ``at`` that is no ISO timestamp."""
    match = _INTERVAL.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"within: {text!r} is not an interval (D<offset> or W<offset>, e.g. D-1, W-1)")
    now = _timestamp(at, UTC)
    if now is None:
        raise ValueError(f"within: the turn's time {at!r} is not an ISO 8601 timestamp")
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    unit, offset = match.group(1).upper(), int(match.group(2))
    if unit == "D":
        start = day + timedelta(days=offset)
        return start, start + timedelta(days=1)
    start = day - timedelta(days=day.weekday()) + timedelta(weeks=offset)
    return start, start + timedelta(weeks=1)


def within(events: Any, text: str, at: str) -> list[Any]:
    """The events (in their order) whose date (``item_date``) falls in ``interval(text, at)``. A date without a time
    is that whole day at the turn's fixed UTC offset; a timestamp without an offset is read at that offset; an undated
    or unreadable item is dropped."""
    start, end = interval(text, at)
    items = events if isinstance(events, (list, tuple)) else [] if events is None else [events]
    kept = []
    for item in items:
        stamp = item_date(item)
        if stamp is None:
            continue
        if len(stamp) == 10:
            try:
                if start.date() <= date.fromisoformat(stamp) < end.date():
                    kept.append(item)
            except ValueError:
                pass
            continue
        moment = _timestamp(stamp, start.tzinfo)
        if moment is not None and start <= moment < end:
            kept.append(item)
    return kept


def _timestamp(text: str, zone: Any) -> datetime | None:
    """An ISO 8601 timestamp (a Neo4j ``toString``: ``Z``, nanoseconds, a ``[zone]`` suffix); naive ones in ``zone``."""
    text = _FRACTION.sub(r"\1", text.split("[", 1)[0].strip())
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=zone)


class _Unbound:
    def __repr__(self) -> str:
        return "<unbound>"


_UNBOUND = _Unbound()
Assignment = dict[int, int]  # distribute op index -> member index


def _effective_registry(program: Program, registry: Registry) -> tuple[Registry, list[str]]:
    problems = [
        f"op {name!r} redefines a built-in value op"
        for name, spec in registry.specs.items()
        if name in VALUE_SPECS and spec != VALUE_SPECS[name]
    ]
    for i, op in enumerate(program.ops):
        spec = registry.specs.get(op.name)
        if spec is not None and spec.kind in ("value", "control") and op.name not in VALUE_SPECS:
            problems.append(f"unknown value op {op.name!r} (op {i}): value and control ops are built in")
    return Registry({**registry.specs, **VALUE_SPECS}), problems


def _turn_problems(program: Program, turn: Turn) -> list[str]:
    """Refuse, before anything runs, a program that needs a turn role the turn does not have."""
    problems: list[str] = []
    for i, op in enumerate(program.ops):
        if op.name == "deixis" and len(op.args) == 2:
            role = op.args[1]
            if not isinstance(role, Const) or role.value not in _DEIXIS_ROLES:
                problems.append(f"deixis (op {i}) needs a role in {_DEIXIS_ROLES}, got {role!r}")
            elif getattr(turn, str(role.value)) is None:
                problems.append(f"deixis (op {i}) refers to the {role.value}, but the turn has no {role.value}")
        elif op.name == "everyone" and not turn.participants:
            problems.append(f"everyone (op {i}) needs the turn's participants, but the turn has none")
        elif op.name == "within" and not turn.at:
            problems.append(f"within (op {i}) needs the turn's time, but the turn has none")
        elif op.name == "window" and len(op.args) == 2:
            problems += _window_problems(i, op.args[1], turn)
        elif op.name == "plural":  # teaching reads it and drops it: only a turn's own program still holds one
            problems.append(
                f"plural (op {i}): 'them' refers to the results of earlier steps of a taught phrase, and a turn has "
                "none"
            )
    return problems


def _window_problems(i: int, index: Any, turn: Turn) -> list[str]:
    """``window``'s refusals (``WINDOW_RULES``): no such turn back, or a turn whose answer named not exactly one."""
    if not isinstance(index, Const) or not isinstance(index.value, int) or index.value < 0:
        return [f"window (op {i}) needs a non-negative integer index, got {index!r}"]
    back = index.value
    if back >= len(turn.window):
        return [f"window (op {i}) refers to the turn {back} back, but the chat's turn window holds {len(turn.window)}"]
    named = len(turn.window[back])
    if named != 1:
        what = "no entity" if named == 0 else f"{named} entities"
        return [f"window (op {i}): the turn {back} back answered with {what}, so the reference is not resolved"]
    return []


class _Shape:
    """Static analysis of a validated program: writers, procedure tree, distribution regions, order."""

    def __init__(self, program: Program, specs: Mapping[str, OpSpec]) -> None:
        self.ops = program.ops
        self.specs = [specs[op.name] for op in program.ops]
        self.problems: list[str] = []
        self.writer: dict[str, int] = {}
        for i, (op, spec) in enumerate(zip(self.ops, self.specs, strict=True)):
            for arg in op.args[: len(spec.outputs)]:
                if isinstance(arg, Var):
                    if arg.name in self.writer:
                        self.problems.append(f"variable {arg.name} is written by more than one op")
                    self.writer.setdefault(arg.name, i)

        self.tools = [i for i, s in enumerate(self.specs) if s.kind == "tool"]
        self.deps: dict[int, list[int]] = {}
        for i, op in enumerate(self.ops):
            if op.name in _STRUCTURAL:
                continue
            n_out = len(self.specs[i].outputs)
            self.deps[i] = sorted({self.writer[a.name] for a in op.args[n_out:] if isinstance(a, Var)})
        self.order = self._topological()
        self.data = [i for i in self.order if self.specs[i].kind != "tool"]

        self.steps: dict[str, list[tuple[int, int, str]]] = {}  # proc var -> (index, seq op, step var)
        self.axes_of_proc: dict[str, list[int]] = {}
        self.axis_set: dict[int, str] = {}
        child_vars: set[str] = set()
        for i, op in enumerate(self.ops):
            if op.name == "seq":
                proc, index, step = op.args
                if not (isinstance(proc, Var) and self._is_procedure(proc) and isinstance(step, Var)):
                    self.problems.append(f"seq (op {i}) must order a step of a procedure variable")
                elif not isinstance(index, Const) or not isinstance(index.value, int):
                    self.problems.append(f"seq (op {i}) needs an integer index, got {index!r}")
                elif not self._is_step_writer(self.writer[step.name]):
                    writer = self.ops[self.writer[step.name]].name
                    self.problems.append(f"seq (op {i}) orders {step.name}, written by {writer!r}: not a step")
                else:
                    self.steps.setdefault(proc.name, []).append((index.value, i, step.name))
                    child_vars.add(step.name)
            elif op.name == "distribute":
                proc, members = op.args
                if isinstance(proc, Var) and self._is_procedure(proc) and isinstance(members, Var):
                    self.axes_of_proc.setdefault(proc.name, []).append(i)
                    self.axis_set[i] = members.name
                else:
                    self.problems.append(f"distribute (op {i}) needs a procedure variable and a set variable")
        for entries in self.steps.values():
            entries.sort()
        procedures = [op.args[0].name for op in self.ops if op.name == "procedure" and isinstance(op.args[0], Var)]
        self.roots = [p for p in procedures if p not in child_vars]

        self.op_axes: dict[int, tuple[int, ...]] = {i: () for i in range(len(self.ops))}
        self.tree_tools: set[int] = set()
        reached: set[str] = set()
        for root in self.roots:
            chain: list[tuple[int, str, int]] = []
            self._place_tools(root, (), (), reached, chain)
            self._check_chain(chain)
        if set(procedures) - reached:
            self.problems.append("procedure steps form a cycle")
        for axis, set_var in self.axis_set.items():
            for i in self._region_data(axis, set_var):
                self.op_axes[i] = tuple(sorted({*self.op_axes[i], axis}))
        for axis, set_var in self.axis_set.items():
            if self.op_axes[self.writer[set_var]]:
                self.problems.append(f"distribute (op {axis}) over a set computed inside a distributed procedure")

    def _is_procedure(self, arg: Var) -> bool:
        return self.ops[self.writer[arg.name]].name == "procedure"

    def _is_step_writer(self, i: int) -> bool:
        return self.ops[i].name == "procedure" or self.specs[i].kind == "tool"

    def _check_chain(self, chain: list[tuple[int, str, int]]) -> None:
        """A step of a procedure may not wait on a later step of the same procedure tree (seq order is a contract)."""
        first: dict[int, int] = {}
        for pos, (tool, _, _) in enumerate(chain):
            first.setdefault(tool, pos)
        for pos, (tool, proc, index) in enumerate(chain):
            if first[tool] != pos:
                continue
            seen: set[int] = set()
            stack = list(self.deps[tool])
            while stack:
                d = stack.pop()
                if d in seen:
                    continue
                seen.add(d)
                stack.extend(self.deps[d])
                if d in first and first[d] > pos:
                    self.problems.append(
                        f"step {index} of procedure {proc} ({self.ops[tool].name}) waits on a later step "
                        f"({self.ops[d].name})"
                    )

    def _topological(self) -> list[int]:
        users: dict[int, list[int]] = {i: [] for i in self.deps}
        missing = {i: len(d) for i, d in self.deps.items()}
        for i, deps in self.deps.items():
            for d in deps:
                users[d].append(i)
        ready = [i for i, n in missing.items() if n == 0]
        heapq.heapify(ready)
        order: list[int] = []
        while ready:
            i = heapq.heappop(ready)
            order.append(i)
            for user in users[i]:
                missing[user] -= 1
                if missing[user] == 0:
                    heapq.heappush(ready, user)
        return order

    def _place_tools(
        self,
        proc: str,
        axes: tuple[int, ...],
        path: tuple[str, ...],
        reached: set[str],
        chain: list[tuple[int, str, int]],
    ) -> None:
        if proc in path:
            self.problems.append(f"procedure {proc} contains itself")
            return
        reached.add(proc)
        axes = tuple(sorted({*axes, *self.axes_of_proc.get(proc, ())}))
        for index, _, step in self.steps.get(proc, ()):
            w = self.writer[step]
            if self.ops[w].name == "procedure":
                self._place_tools(step, axes, (*path, proc), reached, chain)
            else:
                self.tree_tools.add(w)
                self.op_axes[w] = tuple(sorted({*self.op_axes[w], *axes}))
                chain.append((w, proc, index))

    def _region_data(self, axis: int, set_var: str) -> set[int]:
        """Data ops that feed a tool step of the distributed procedure and depend, directly or through other ops, on the
        set or on a tool step of that procedure (a ``gather`` of one member's results runs in that member's turn)."""
        reads_set: dict[int, bool] = {}

        def depends(i: int) -> bool:
            if i not in reads_set:
                n_out = len(self.specs[i].outputs)
                direct = any(isinstance(a, Var) and a.name == set_var for a in self.ops[i].args[n_out:])
                in_region = self.specs[i].kind == "tool" and axis in self.op_axes[i]
                reads_set[i] = direct or in_region or any(depends(d) for d in self.deps[i])
            return reads_set[i]

        feeds: set[int] = set()
        stack = [t for t in self.tools if axis in self.op_axes[t]]
        while stack:
            for d in self.deps[stack.pop()]:
                if d not in feeds:
                    feeds.add(d)
                    stack.append(d)
        return {i for i in feeds if self.specs[i].kind != "tool" and depends(i)}


Entry = tuple[int, Assignment, str]  # tool op index, assignment, step_id


class _Run:
    def __init__(self, shape: _Shape, memory: MemoryOps, turn: Turn, spans: SpanText, prior: Mapping[str, Any]) -> None:
        self.shape = shape
        self.prior = prior
        self.memory = memory
        self.turn = turn
        self.spans = spans
        self.bound: dict[tuple[str, tuple[tuple[int, int], ...]], Any] = {}
        self.ran: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
        self.trace: list[TraceStep] = []
        self.problems: list[str] = []

    def members(self, axis: int) -> list[Any] | _Unbound:
        value = self.bound.get((self.shape.axis_set[axis], ()), _UNBOUND)
        if isinstance(value, _Unbound):
            return value
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return list(value)
        return [value]

    def _key(self, op: int, asg: Assignment) -> tuple[tuple[int, int], ...]:
        return tuple((a, asg[a]) for a in self.shape.op_axes[op])

    def lookup(self, var: str, asg: Assignment) -> Any:
        for axis, member in asg.items():
            if self.shape.axis_set[axis] == var:
                members = self.members(axis)
                return _UNBOUND if isinstance(members, _Unbound) else members[member]
        w = self.shape.writer[var]
        axes = self.shape.op_axes[w]
        missing = [a for a in axes if a not in asg]
        if not missing:
            return self.bound.get((var, self._key(w, asg)), _UNBOUND)
        lists = [self.members(a) for a in missing]
        if any(isinstance(m, _Unbound) for m in lists):
            return _UNBOUND
        gathered = []
        for combo in product(*(range(len(m)) for m in lists if isinstance(m, list))):
            value = self.bound.get((var, self._key(w, {**asg, **dict(zip(missing, combo, strict=True))})), _UNBOUND)
            if isinstance(value, _Unbound):
                return _UNBOUND
            gathered.append(value)
        return gathered

    def inputs(self, i: int, asg: Assignment) -> dict[str, Any] | None:
        """The op instance's inputs by name, or None while one is unbound. A variadic tail (``OpSpec.variadic``) is
        one list, in argument order, under its name without the ``…``."""
        op, spec = self.shape.ops[i], self.shape.specs[i]
        values = []
        for arg in op.args[len(spec.outputs) :]:
            value = self.lookup(arg.name, asg) if isinstance(arg, Var) else arg.value
            if isinstance(value, _Unbound):
                return None
            values.append(value)
        if not spec.variadic:
            return dict(zip(spec.inputs, values, strict=True))
        *fixed, tail = spec.inputs
        n = len(fixed)
        return {**dict(zip(fixed, values[:n], strict=True)), tail.removesuffix(VARIADIC): values[n:]}

    def bind(self, i: int, asg: Assignment, out: Any) -> None:
        op, spec = self.shape.ops[i], self.shape.specs[i]
        outs = op.args[: len(spec.outputs)]
        values = (out,) if len(outs) == 1 else tuple(out) if outs else ()
        if len(values) != len(outs):
            raise ValueError(f"{op.name} returned {len(values)} values for {len(outs)} outputs")
        for arg, value in zip(outs, values, strict=True):
            if isinstance(arg, Var):  # validate refuses constants in output positions
                self.bound[(arg.name, self._key(i, asg))] = value

    def reused(self, i: int, asg: Assignment) -> Any:
        """The op instance's output from ``prior`` (as ``bind`` takes it), or unbound when it must run."""
        op, spec = self.shape.ops[i], self.shape.specs[i]
        outs = [a.name for a in op.args[: len(spec.outputs)] if isinstance(a, Var)]
        if not outs or any(name not in self.prior for name in outs):
            return _UNBOUND
        axes = self.shape.op_axes[i]
        flat = 0
        for a in axes:  # position of this instance in the member-order list ``values`` exports
            members = self.members(a)
            if isinstance(members, _Unbound):
                return _UNBOUND
            flat = flat * len(members) + asg[a]
        values = []
        for name in outs:
            value = self.prior[name]
            if axes:
                if not isinstance(value, list) or flat >= len(value):
                    return _UNBOUND
                value = value[flat]
            values.append(value)
        return values[0] if len(values) == 1 else tuple(values)

    def instances(self, i: int) -> list[Assignment] | None:
        axes = self.shape.op_axes[i]
        lists = [self.members(a) for a in axes]
        if any(isinstance(m, _Unbound) for m in lists):
            return None
        ranges = [range(len(m)) for m in lists if isinstance(m, list)]
        return [dict(zip(axes, combo, strict=True)) for combo in product(*ranges)]

    async def value(self, name: str, args: Mapping[str, Any]) -> Any:
        if name == "name-span":
            left, right = args["left"], args["right"]
            if not isinstance(left, int) or not isinstance(right, int):
                raise ValueError(f"name-span offsets must be integers, got {left!r}, {right!r}")
            return self.spans.cut(left, right)
        if name == "lemma-span":
            left, right = args["left"], args["right"]
            if not isinstance(left, int) or not isinstance(right, int):
                raise ValueError(f"lemma-span offsets must be integers, got {left!r}, {right!r}")
            return marked_lemma(self.spans.normalised, left, right)
        if name == "within":
            return within(args["events"], str(args["interval"]), str(self.turn.at))
        if name == "const":
            return args["value"]
        if name == "either":
            return args["second"] if empty(args["first"]) else args["first"]
        if name == "gather":
            return list(args["value"])
        if name == "deixis":
            return getattr(self.turn, args["role"])
        if name == "everyone":
            return tuple(self.turn.participants)
        if name == "window":  # checked before the run (_window_problems): exactly one entity
            return Referent.of(self.turn.window[args["index"]][0])
        if name == "procedure":
            return args["name"]
        if name in ("definite", "indefinite", "plural"):
            return None
        return await self.memory.run(name, args)

    async def run_ready_data(self) -> bool:
        """Run every data op instance whose inputs are bound; False when one failed."""
        progress = True
        while progress:
            progress = False
            for i in self.shape.data:
                for asg in self.instances(i) or ():
                    key = (i, self._key(i, asg))
                    if key in self.ran:
                        continue
                    args = self.inputs(i, asg)
                    if args is None:
                        continue
                    name = self.shape.ops[i].name
                    prior = self.reused(i, asg)
                    if not isinstance(prior, _Unbound):
                        self.bind(i, asg, prior)
                        self.trace.append(TraceStep(name, args, prior, 0.0, True, reused=True))
                        self.ran.add(key)
                        progress = True
                        continue
                    start = time.perf_counter()
                    try:
                        out = await self.value(name, args)
                        self.bind(i, asg, out)
                    except Exception as exc:  # noqa: BLE001 — any op failure ends the run with a diagnostic
                        ms = (time.perf_counter() - start) * 1000
                        self.trace.append(TraceStep(name, args, None, ms, False))
                        self.problems.append(f"{name} (op {i}) failed: {exc}")
                        return False
                    self.trace.append(TraceStep(name, args, out, (time.perf_counter() - start) * 1000, True))
                    self.ran.add(key)
                    progress = True
        return True

    def tool_sequence(self) -> tuple[list[list[Entry]], list[Entry], bool]:
        """One chain of tool instances per root procedure, in ``seq`` order (member-major under distribute), the
        tools outside any procedure, and whether a chain stops at a distributed procedure whose set is unbound."""
        shape = self.shape
        sequence: list[Entry] = []
        seen: dict[int, int] = {}

        def emit(i: int, asg: Assignment) -> None:
            sequence.append((i, asg, f"{i}:{seen.get(i, 0)}"))
            seen[i] = seen.get(i, 0) + 1

        def visit(proc: str, asg: Assignment, path: tuple[str, ...]) -> bool:
            if proc in path:
                return True
            new_axes = [a for a in shape.axes_of_proc.get(proc, ()) if a not in asg]
            lists = [self.members(a) for a in new_axes]
            if any(isinstance(m, _Unbound) for m in lists):
                return False
            for combo in product(*(range(len(m)) for m in lists if isinstance(m, list))):
                sub = {**asg, **dict(zip(new_axes, combo, strict=True))}
                for _, _, step in shape.steps.get(proc, ()):
                    w = shape.writer.get(step)
                    if w is None:
                        continue
                    if shape.ops[w].name == "procedure":
                        if not visit(step, sub, (*path, proc)):
                            return False
                    elif shape.specs[w].kind == "tool":
                        emit(w, sub)
            return True

        chains: list[list[Entry]] = []
        blocked = False
        for root in shape.roots:  # every root, so predicate order never decides whether a program runs
            sequence = []
            blocked = not visit(root, {}, ()) or blocked
            chains.append(sequence)
        sequence = []
        for i in shape.order:
            if shape.specs[i].kind == "tool" and i not in shape.tree_tools:
                emit(i, {})
        return chains, sequence, blocked

    def values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for var in self.shape.writer:
            value = self.lookup(var, {})
            if not isinstance(value, _Unbound):
                out[var] = value
        return out


def _holds_entity(value: Any) -> bool:
    items = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    return any(isinstance(item, Mapping) for item in items)


def _kind_problems(
    shape: _Shape,
    run: _Run,
    sequence: list[Entry],
    resolved: set[str],
    results: Mapping[str, Any],
    committed: Committed | None,
) -> list[str]:
    """The store-kind mismatches (``constraints.check_constraint`` on a resolved memory entity against a parameter's
    ``kind``) of every tool step still to issue whose inputs are bound: the one check that blocks. A step that waits
    on an earlier tool's result is checked when its inputs bind."""
    out: list[str] = []
    for i, asg, step_id in sequence:
        if step_id in resolved or step_id in results:
            continue
        args = run.inputs(i, asg)
        if args is None or (committed is not None and committed.holds(shape.ops[i].name, args)):
            continue
        for param in shape.specs[i].params:
            value = args.get(param.name)
            if param.kind is None or not _holds_entity(value):
                continue
            verdict, why = check_constraint(value, param)
            if verdict == "fail" and why is not None and why not in out:
                out.append(why)
    return out


def _lexical_checks(params: tuple[ToolParam, ...], args: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    return tuple(
        {"param": p.name, "constraint": p.constraint, "result": check_constraint(args.get(p.name), p)[0]}
        for p in params
        if p.constraint is not None
    )


async def execute(
    program: Program,
    registry: Registry,
    memory: MemoryOps,
    *,
    turn: Turn,
    spans: SpanText,
    resumed: Mapping[str, Any] | None = None,
    prior: Mapping[str, Any] | None = None,
    committed: Committed | None = None,
    policy: ConfirmationPolicy = DEFAULT_POLICY,
    id_prefix: str = "",
) -> Execution:
    """Run ``program`` up to its first unresolved tool step (or to the end); ``resumed`` maps step_id -> result and
    ``prior`` is the previous execution's ``values`` (completed data ops are reused, not run again). A step with no
    result in ``resumed`` whose tool and arguments match an entry of ``committed`` is resolved with the entry's value
    (trace ``reused``) and never issued. A step id in a problem message is written ``id_prefix`` + the local id (the
    api exposes ``<plan_id>:<i>:<occurrence>``; the ids of ``pending`` and ``resumed`` stay local)."""
    effective, problems = _effective_registry(program, registry)
    problems += validate(program, effective)
    if not problems:
        problems += _turn_problems(program, turn)
    shape: _Shape | None = None
    if not problems:
        shape = _Shape(program, effective.specs)
        problems += shape.problems
    if problems or shape is None:
        return Execution(values={}, trace=[], pending=(), done=False, problems=problems)

    results = dict(resumed or {})
    run = _Run(shape, memory, turn, spans, prior or {})
    pending: tuple[ToolStep, ...] = ()
    resolved: set[str] = set()
    sequence: list[Entry] = []
    blocked = False
    while await run.run_ready_data():
        chains, free, blocked = run.tool_sequence()
        sequence = [entry for chain in chains for entry in chain] + free
        # The first unresolved step of each procedure chain blocks the rest of that chain; tools outside any
        # procedure stay eligible.
        candidates: list[Entry] = []
        for chain in chains:
            head = next((e for e in chain if e[2] not in resolved), None)
            if head is not None:
                candidates.append(head)
        candidates += [e for e in free if e[2] not in resolved]
        nxt: tuple[int, Assignment, str, dict[str, Any]] | None = None
        for i, asg, step_id in candidates:
            args = run.inputs(i, asg)
            if args is not None:
                nxt = (i, asg, step_id, args)
                break
        if nxt is None:
            break
        i, asg, step_id, args = nxt
        tool = program.ops[i].name
        if step_id in results:
            run.bind(i, asg, results[step_id])
            run.trace.append(TraceStep(tool, args, results[step_id], 0.0, True))
            resolved.add(step_id)
            continue
        if committed is not None:
            found, value = committed.take(tool, args)
            if found:
                run.bind(i, asg, value)
                run.trace.append(TraceStep(tool, args, value, 0.0, True, reused=True))
                resolved.add(step_id)
                continue
        refused = _kind_problems(shape, run, sequence, resolved, results, committed)
        if refused:  # no tool step is issued past an entity of the wrong store kind (G3)
            run.problems += refused
            break
        declared = shape.specs[i].side_effect
        side_effect = declared if isinstance(declared, bool) else None
        pending = (
            ToolStep(
                step_id,
                tool,
                args,
                side_effect,
                needs_confirmation=policy.needs_confirmation(shape.specs[i], actor=turn.speaker, speaker=turn.speaker),
                actor=turn.speaker,
                checks=(
                    await asyncio.to_thread(_lexical_checks, shape.specs[i].params, args)  # a spaCy parse: off the loop
                    if any(p.constraint is not None for p in shape.specs[i].params)
                    else ()
                ),
            ),
        )
        break

    if not pending and not run.problems:
        for i, _, step_id in sequence:
            if step_id not in resolved:
                run.problems.append(
                    f"tool step {id_prefix}{step_id} ({program.ops[i].name}) never had its inputs bound"
                )
        if blocked:
            run.problems.append("a distributed procedure's set was never bound")
        for i in shape.data:
            todo = run.instances(i)
            if todo is None or any((i, run._key(i, asg)) not in run.ran for asg in todo):
                run.problems.append(f"{program.ops[i].name} (op {i}) never had its inputs bound")
    if not blocked and not run.problems:
        known = {step_id for _, _, step_id in sequence}
        run.problems += [
            f"resumed step {id_prefix}{s} is not a step of this program" for s in sorted(set(results) - known)
        ]
    return Execution(
        values=run.values(),
        trace=run.trace,
        pending=pending if not run.problems else (),
        done=not pending and not run.problems,
        problems=run.problems,
    )


__all__ = [
    "DATE_KEYS",
    "Committed",
    "EITHER_RULES",
    "VALUE_SPECS",
    "WINDOW_RULES",
    "WITHIN_RULES",
    "Execution",
    "MemoryOps",
    "Referent",
    "SpanText",
    "ToolStep",
    "TraceStep",
    "Turn",
    "execute",
    "interval",
    "item_date",
    "within",
]
