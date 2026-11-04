"""Program model, operation registry and validator for the runtime grammar."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

OpKind = Literal["memory", "value", "tool", "control"]
#: where a tool's step runs (catalogue ``x-runner``): the office's service, or the agent itself
Runner = Literal["office", "agent"]
RUNNERS: tuple[Runner, ...] = ("office", "agent")
ANSWER_PREDICATE = "answer"
#: ends the name of a variadic last input (``OpSpec.variadic``)
VARIADIC = "…"


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Const:
    value: str | int


Arg = Var | Const


@dataclass(frozen=True)
class Op:
    name: str
    args: tuple[Arg, ...]


class MeaningError(ValueError):
    """A meaning (or its JSON form) is malformed."""


def _canonical_int(symbol: str) -> int | None:
    try:
        value = int(symbol)
    except ValueError:
        return None
    return value if str(value) == symbol else None


def _parse_arg(symbol: str | int) -> Arg:
    if isinstance(symbol, bool) or not isinstance(symbol, (str, int)):
        raise MeaningError(f"unsupported argument {symbol!r}")
    if isinstance(symbol, int):  # a slot offset the engine bound (a Reading's meaning)
        return Const(symbol)
    if symbol.startswith("?"):
        return Var(symbol)
    as_int = _canonical_int(symbol)
    return Const(as_int if as_int is not None else symbol)


def _json_arg(item: object) -> Arg:
    if isinstance(item, bool) or not isinstance(item, (str, int)):
        raise MeaningError(f"unsupported argument {item!r}")
    if isinstance(item, str) and item.startswith("?"):
        return Var(item)
    return Const(item)


@dataclass(frozen=True)
class Program:
    ops: tuple[Op, ...]
    answer: Var | None

    @classmethod
    def from_meaning(cls, meaning: Sequence[Sequence[str | int]]) -> Program:
        """Build a program from FCG meaning predicates; ``(answer ?v)`` names the result. Integer symbols (the slot
        offsets of a Reading) are constants; a numeric string such as ``"11"`` is read as an int as well."""
        ops: list[Op] = []
        answer: Var | None = None
        for predicate in meaning:
            if not predicate:
                raise MeaningError("empty predicate")
            name, *symbols = predicate
            if not isinstance(name, str):
                raise MeaningError(f"predicate name must be a string, got {name!r}")
            if name == ANSWER_PREDICATE:
                if len(symbols) != 1 or not isinstance(symbols[0], str) or not symbols[0].startswith("?"):
                    raise MeaningError(f"answer needs exactly one variable, got {symbols!r}")
                if answer is not None:
                    raise MeaningError("more than one answer predicate")
                answer = Var(symbols[0])
                continue
            ops.append(Op(name, tuple(_parse_arg(s) for s in symbols)))
        return cls(ops=tuple(ops), answer=answer)

    def to_json(self) -> list[list]:
        """Meaning-predicate form, ``answer`` last; integer constants stay ints."""
        out: list[list] = [[op.name, *(a.name if isinstance(a, Var) else a.value for a in op.args)] for op in self.ops]
        if self.answer is not None:
            out.append([ANSWER_PREDICATE, self.answer.name])
        return out

    @classmethod
    def from_json(cls, data: list[list]) -> Program:
        """Inverse of :meth:`to_json`: strings stay strings, ints stay ints."""
        ops: list[Op] = []
        answer: Var | None = None
        for predicate in data:
            if not predicate or not isinstance(predicate[0], str):
                raise MeaningError("empty or unnamed predicate")
            name, *items = predicate
            if name == ANSWER_PREDICATE:
                arg = _json_arg(items[0]) if len(items) == 1 else None
                if not isinstance(arg, Var):
                    raise MeaningError(f"answer needs exactly one variable, got {items!r}")
                if answer is not None:
                    raise MeaningError("more than one answer predicate")
                answer = arg
                continue
            ops.append(Op(name, tuple(_json_arg(i) for i in items)))
        return cls(ops=tuple(ops), answer=answer)


class _Unset:
    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


@dataclass(frozen=True)
class OpSpec:
    """The first ``len(outputs)`` args of an op are its outputs, the rest its inputs."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    kind: OpKind
    side_effect: bool | None | _Unset = _UNSET
    #: a tool's declared state cells (``ToolSpec.reads`` / ``writes``); empty for every other kind
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    #: a tool's typed parameters (``ToolSpec.params``, parallel to ``inputs``); empty for every other kind
    params: tuple[ToolParam, ...] = ()
    #: a tool's consent (``ToolSpec.consent``); ``confirm`` for every other kind
    consent: Literal["confirm", "speaker_turn"] = "confirm"
    #: the parameter a tool's result stands for (``ToolSpec.returns``); ``None`` = an untyped result
    returns: str | None = None
    #: where a tool's step runs (``ToolSpec.runner``); ``office`` for every other kind
    runner: Runner = "office"

    def __post_init__(self) -> None:
        if isinstance(self.side_effect, _Unset):
            # Tools are unknown-effect unless declared; every other kind is pure.
            object.__setattr__(self, "side_effect", None if self.kind == "tool" else False)

    @property
    def arity(self) -> int:
        return len(self.outputs) + len(self.inputs)

    @property
    def variadic(self) -> bool:
        """The last input is declared ``name…``: it takes the rest of the op's arguments, zero or more (``invoke``'s
        slots, ``gather``'s values). A tool's inputs are its catalogue parameters, never variadic."""
        return self.kind != "tool" and bool(self.inputs) and self.inputs[-1].endswith(VARIADIC)


@dataclass(frozen=True)
class ToolParam:
    """One typed tool argument. ``kind`` is a store kind label (Person, Organisation, Artifact, Place); ``constraint`` a
    common-noun concept lemma (checked at plan time); ``many`` means the argument takes a set."""

    name: str
    role: str | None = None
    kind: str | None = None
    constraint: str | None = None
    many: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """Kernel-side view of a catalogue tool. ``side_effect`` is required: the catalogue declares it. ``reads`` /
    ``writes`` name the state cells the tool declares it reads and writes (Declare -> Resolve -> Commit): the kernel
    surfaces them in a plan's licence and never enforces them (the agent runtime's resolver does). ``forms`` are the
    clause patterns the catalogue generates hand cards from."""

    name: str
    params: tuple[ToolParam, ...]
    side_effect: bool
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    consent: Literal["confirm", "speaker_turn"] = "confirm"
    forms: tuple[str, ...] = ()
    #: the name of the parameter this tool's result stands for (catalogue ``x-returns``); ``None`` = untyped result
    returns: str | None = None
    #: where the tool's step runs (catalogue ``x-runner``): the office (default) or the agent
    runner: Runner = "office"

    @classmethod
    def simple(cls, name: str, args: Sequence[str], side_effect: bool, **rest) -> ToolSpec:
        """A tool whose arguments carry only names (no role, kind or constraint)."""
        return cls(name, tuple(ToolParam(a) for a in args), side_effect, **rest)

    @property
    def args(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)


@dataclass(frozen=True)
class Registry:
    specs: Mapping[str, OpSpec]

    def __post_init__(self) -> None:
        if ANSWER_PREDICATE in self.specs:
            raise ValueError(f"{ANSWER_PREDICATE!r} is reserved and cannot be an op")

    def with_tools(self, catalogue: Sequence[ToolSpec]) -> Registry:
        merged = dict(self.specs)
        for tool in catalogue:
            existing = merged.get(tool.name)
            if existing is not None and existing.kind != "tool":
                raise ValueError(f"tool {tool.name!r} collides with the {existing.kind} op of that name")
            merged[tool.name] = OpSpec(
                name=tool.name,
                inputs=tool.args,
                outputs=("result",),
                kind="tool",
                side_effect=tool.side_effect,
                reads=tuple(tool.reads),
                writes=tuple(tool.writes),
                params=tuple(tool.params),
                consent=tool.consent,
                returns=tool.returns,
                runner=tool.runner,
            )
        return Registry(merged)


def _split_args(op: Op, spec: OpSpec) -> tuple[list[Var], list[Var]]:
    n_out = len(spec.outputs)
    outs = [a for a in op.args[:n_out] if isinstance(a, Var)]
    ins = [a for a in op.args[n_out:] if isinstance(a, Var)]
    return outs, ins


def validate(program: Program, registry: Registry) -> list[str]:
    """Return problems: unknown op, arity mismatch, unwritten variable read, cycle."""
    problems: list[str] = []
    writers: dict[str, list[int]] = {}
    reads: list[list[str]] = []
    checked: list[bool] = []
    for i, op in enumerate(program.ops):
        spec = registry.specs.get(op.name)
        if spec is None:
            problems.append(f"unknown op {op.name!r} (op {i})")
            reads.append([])
            checked.append(False)
            continue
        fixed = spec.arity - 1 if spec.variadic else spec.arity  # a variadic tail takes zero or more arguments
        if len(op.args) < fixed or (len(op.args) != fixed and not spec.variadic):
            least = "at least " if spec.variadic else ""
            problems.append(f"arity mismatch for {op.name!r} (op {i}): expected {least}{fixed}, got {len(op.args)}")
            reads.append([])
            checked.append(False)
            continue
        outs, ins = _split_args(op, spec)
        for pos, arg in enumerate(op.args[: len(spec.outputs)]):
            if isinstance(arg, Const):
                problems.append(f"constant {arg.value!r} in output position {pos} of {op.name!r} (op {i})")
        for var in outs:
            writers.setdefault(var.name, []).append(i)
        reads.append([v.name for v in ins])
        checked.append(True)

    for i, names in enumerate(reads):
        for name in names:
            if name not in writers:
                problems.append(f"variable {name} read by {program.ops[i].name!r} (op {i}) is never written")
    if program.answer is not None and program.answer.name not in writers:
        problems.append(f"answer variable {program.answer.name} is never written")

    # Dependency graph: an op depends on every op writing a variable it reads.
    deps: dict[int, set[int]] = {i: set() for i in range(len(program.ops)) if checked[i]}
    for i in deps:
        for name in reads[i]:
            deps[i].update(writers.get(name, []))
    state: dict[int, int] = {}

    def has_cycle(node: int) -> bool:
        state[node] = 1
        for dep in deps.get(node, ()):
            if state.get(dep) == 1 or (dep not in state and has_cycle(dep)):
                return True
        state[node] = 2
        return False

    if any(i not in state and has_cycle(i) for i in deps):
        problems.append("cycle in variable dependencies")
    return problems
