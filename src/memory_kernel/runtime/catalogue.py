"""The per-bonfire tool catalogue (``BonfireMeta.catalogue_json``, absent until the first PUT): typed tools and the
hand cards their forms make.

Every tool must declare ``x-side-effect`` and annotate every ``input_schema`` property in ``x-args``; a missing
annotation is refused (no inference). A form such as ``"play {pattern} on {source}"`` generates one live hand card whose
meaning is the one-step procedure calling the tool with the form's slots bound by name. ``x-runner`` says where the
tool's step runs: ``office`` (the default) or ``agent``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from memory_kernel.runtime.constructions import CompileError, CxnSpec, compile_spec
from memory_kernel.runtime.program import RUNNERS, ToolParam, ToolSpec
from memory_kernel.runtime.teach import _parse_form

SCHEMA_VERSION = "kernel-catalogue/v1"
_CONSENTS = ("confirm", "speaker_turn")
_ARG_KEYS = {"role", "kind", "constraint", "many"}


class CatalogueError(ValueError):
    """The catalogue file is malformed or leaves something unannotated."""


@dataclass(frozen=True)
class Catalogue:
    tools: tuple[ToolSpec, ...]
    cards: tuple[CxnSpec, ...]
    sha256: str


def _opt_str(value: object, where: str) -> str | None:
    if value is not None and not (isinstance(value, str) and value.strip()):
        raise CatalogueError(f"{where} must be a non-empty string")
    return value


def _params(tool: Mapping, name: str) -> tuple[ToolParam, ...]:
    schema = tool.get("input_schema")
    if not isinstance(schema, Mapping):
        raise CatalogueError(f"tool {name!r}: input_schema must be an object")
    props = schema.get("properties", {})
    if not isinstance(props, Mapping):
        raise CatalogueError(f"tool {name!r}: input_schema.properties must be an object")
    annotations = tool.get("x-args", {})
    if not isinstance(annotations, Mapping):
        raise CatalogueError(f"tool {name!r}: x-args must be an object")
    seen: dict[str, str] = {}
    for arg in props:
        if arg.lower() in seen:
            raise CatalogueError(
                f"tool {name!r}: arguments {seen[arg.lower()]!r} and {arg!r} collide (case-insensitive)"
            )
        seen[arg.lower()] = arg
        if arg not in annotations:
            raise CatalogueError(f"tool {name!r}: argument {arg!r} has no x-args annotation")
    for arg in annotations:
        if arg not in props:
            raise CatalogueError(f"tool {name!r}: x-args names {arg!r}, which is not an input_schema property")
    out = []
    for arg in props:
        note = annotations[arg]
        if not isinstance(note, Mapping) or set(note) - _ARG_KEYS:
            raise CatalogueError(f"tool {name!r}: x-args.{arg} must be an object of {sorted(_ARG_KEYS)}")
        where = f"tool {name!r}: x-args.{arg}"
        role = _opt_str(note.get("role"), f"{where}.role")
        if role is None:
            raise CatalogueError(f"{where} needs a role")
        many = note.get("many", False)
        if not isinstance(many, bool):
            raise CatalogueError(f"{where}.many must be a boolean")
        kind = _opt_str(note.get("kind"), f"{where}.kind")
        constraint = _opt_str(note.get("constraint"), f"{where}.constraint")
        out.append(ToolParam(arg, role=role, kind=kind, constraint=constraint, many=many))
    return tuple(out)


def _returns(row: Mapping, name: str, params: tuple[ToolParam, ...]) -> str | None:
    """The parameter the tool's result stands for (``"x-returns": {"of": "<param>"}``), or ``None``."""
    if "x-returns" not in row:
        return None
    note = row["x-returns"]
    if not isinstance(note, Mapping) or set(note) != {"of"} or not isinstance(note["of"], str):
        raise CatalogueError(f'tool {name!r}: x-returns must be an object {{"of": "<parameter>"}}')
    target = next((p for p in params if p.name == note["of"]), None)
    if target is None:
        raise CatalogueError(f"tool {name!r}: x-returns names {note['of']!r}, which is not a parameter of the tool")
    if target.many:
        raise CatalogueError(f"tool {name!r}: x-returns cannot stand for {target.name!r}, a many parameter")
    return target.name


def _check_name(name: str) -> None:
    """A tool's name makes its cards' ids (``card.<name>.<n>``): it must be one a construction id can hold."""
    try:
        CxnSpec(id=f"card.{name.lower()}.0", kind="hand", form=[], meaning=[])
    except CompileError as exc:
        raise CatalogueError(f"tool {name!r}: its name cannot be part of a construction id ({exc})") from exc


def _cards(name: str, params: tuple[ToolParam, ...], forms: tuple[str, ...]) -> list[CxnSpec]:
    cards = []
    names = {p.name.lower(): p.name for p in params}
    for n, text in enumerate(forms):
        parts, _written, problems = _parse_form(text)
        slots = [p["slot"] for p in parts if isinstance(p, dict)]
        problems += [f"form {text!r}: slot {{{s}}} is not a parameter of the tool" for s in slots if s not in names]
        problems += [f"form {text!r}: parameter {p!r} is not bound by a slot" for p in names if p not in slots]
        if problems:
            raise CatalogueError(f"tool {name!r}: " + "; ".join(problems))
        # slot names are lowercased by the form parser; the tool is called with the parameters in declared order
        step = [name, "?card-step1", *(f"?{p.name.lower()}" for p in params)]
        meaning = [["procedure", "?card-proc", name], ["seq", "?card-proc", "1", "?card-step1"], step]
        cards.append(
            CxnSpec(id=f"card.{name.lower()}.{n}", kind="hand", unit="root", form=parts, meaning=meaning, status="live")
        )
    return cards


def _loads(tools: list[ToolSpec], cards: list[CxnSpec]) -> None:
    """The one "would this load?" gate: the runtime's registry takes the tools and every card compiles. Anything that
    would fail the runtime's load is a ``CatalogueError`` naming the tool."""
    from memory_kernel.runtime.api import default_registry

    try:
        default_registry(tools)
    except ValueError as exc:  # a keyword's name, a reserved one
        raise CatalogueError(str(exc)) from exc
    except Exception as exc:  # anything else the registry raises is still a refusal, never a 500
        raise CatalogueError(
            f"tool(s) {', '.join(repr(t.name) for t in tools)}: the registry does not take them ({exc})"
        ) from exc
    for card in cards:
        try:
            compile_spec(card)
        except Exception as exc:
            tool = card.meaning[-1][0]
            raise CatalogueError(f"tool {tool!r}: its card {card.id} does not compile ({exc})") from exc


def load_catalogue(path: Path) -> Catalogue:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CatalogueError(f"{path}: cannot be read ({exc})") from exc
    return parse_catalogue(raw, str(path))


def parse_catalogue(raw: bytes, where: str) -> Catalogue:
    """The catalogue in ``raw`` (``where`` names it in a refusal). ``CatalogueError`` for anything wrong with it."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise CatalogueError(f"{where}: not UTF-8 ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise CatalogueError(f"{where}: not JSON ({exc})") from exc
    if not isinstance(data, Mapping) or data.get("schema_version") != SCHEMA_VERSION:
        raise CatalogueError(f"{where}: schema_version must be {SCHEMA_VERSION!r}")
    rows = data.get("tools")
    if not isinstance(rows, list):
        raise CatalogueError(f"{where}: tools must be a list")
    tools: list[ToolSpec] = []
    cards: list[CxnSpec] = []
    for row in rows:
        name = row.get("name") if isinstance(row, Mapping) else None
        if not isinstance(name, str) or not name:
            raise CatalogueError("a tool needs a name")
        if any(t.name.lower() == name.lower() for t in tools):
            raise CatalogueError(f"tool {name!r} appears twice (names are case-insensitive: they make the card ids)")
        _check_name(name)
        side_effect = row.get("x-side-effect")
        if not isinstance(side_effect, bool):
            raise CatalogueError(f"tool {name!r}: x-side-effect is required and must be true or false")
        consent = row.get("x-consent", "confirm")
        if consent not in _CONSENTS:
            raise CatalogueError(f"tool {name!r}: x-consent must be one of {list(_CONSENTS)}")
        runner = row.get("x-runner", "office")
        if not isinstance(runner, str) or runner not in RUNNERS:
            raise CatalogueError(f"tool {name!r}: x-runner must be one of {list(RUNNERS)}")
        forms = row.get("x-forms", [])
        if not isinstance(forms, list) or not all(isinstance(f, str) for f in forms):
            raise CatalogueError(f"tool {name!r}: x-forms must be a list of strings")
        params = _params(row, name)
        returns = _returns(row, name, params)
        tools.append(
            ToolSpec(name, params, side_effect, consent=consent, forms=tuple(forms), returns=returns, runner=runner)
        )
        cards += _cards(name, params, tuple(forms))
    _loads(tools, cards)
    return Catalogue(tuple(tools), tuple(cards), hashlib.sha256(raw).hexdigest())
