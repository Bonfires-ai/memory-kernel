"""Construction specs and their compiler: JSON rows → the runtime grammar's FCG constructions.

A ``CxnSpec`` describes one construction as data (the shipped base set, mined rows, taught verbs, hand files):

* ``form``: literal parts written NATURALLY ("Where does", "work?") alternating with slots
  (``{"slot": "x", "fills": "np"}``; two slots may be adjacent when one of them is a verb slot, whose marks anchor it). The compiler normalises the literals with ``runtime/normalise.py`` (the text the
  engine reads is normalised: lowercased, verbs as marked lemmas "⟪work⟫", punctuation split off, wrapped in the
  sentinels ⟦ ⟧), so authors never write normalised text (nor the marks: a literal holding one is refused). The
  whole form is normalised at once, each slot stood in for by ``STAND_IN``, so a verb is lemmatised in its sentence
  context as the utterance will be ("works?" alone is tagged a noun and stays "works").
* ``meaning``: FCG meaning predicates (``Program.from_meaning`` must accept them).
* ``unit``:
  - ``"root"``: a clause-level construction. It spans the whole normalised text, sentinel to sentinel, and FCG's goal
    tests check that it does.
  - ``"part"`` (default): any other phrasal construction. It compiles like a root without the sentinels; only the
    label differs. A part never has a slot at its edge (only a root anchors an edge slot, on a sentinel), and its
    outer edges are token edges, so it never matches inside a word: its first anchor opens on a space and its last
    closes on one, unless the edge is a verb mark (every normalised token has a space on each side, the sentinels
    being tokens; a marked verb's marks bound it). "archive" is ``" archive "``: never "archives", never inside
    "⟪archive⟫". A part of ONE word also reads that word as written, lowercased, as a marked verb (an alternate,
    ``_marked_as_written``): a word alone has no sentence to be tagged in ("archive" alone is a noun, a verb in
    "please archive it"; "means" alone is "⟪mean⟫", the tagger keeps "⟪means⟫" in "file X means make ...").
  - ``"filler"``: fills a slot of kind ``fills``. A filler is the only unit that may use ``CONSUME`` (``.{1,200}``),
    which consumes its part of the slot's text, or ``TOKEN`` (``[^ ]{1,200}``: one non-space run, so exactly one
    normalised token; the verb filler's primitive, engine-pinned: a ``CONSUME`` verb filler spans "give⟫ the report
    ⟪say"). A filler must consume its whole span, or text stays in the root and no reading with a slot is ever
    complete. A filler may instead hold slots of its own ("before {Y}", "{P} 's {X}", "my {X}"): its literals anchor
    inside the slot's span and its slots are filled by other fillers (a nested slot named ``x`` refers to what the
    filler's phrase refers to: "my {X}" is the X).
* slot kinds (``fills``): ``np``, ``name``, ``clause``, ``time`` and ``verb``. A verb slot is written between the verb
  marks the normaliser puts around every verb ("⟪publish⟫"): the compiler writes the marks into the neighbouring
  anchors (``" ⟪"`` … ``"⟫ "``), so the slot binds the lemma alone and a verb slot may sit next to an np slot ("when
  did {X} {V} {Y}?"). While a form is normalised a verb slot is stood in for by ``VERB_STAND_IN`` (a verb in every
  probed question form), every other slot by ``STAND_IN``; a literal that normalises to a marked verb in a form with a
  verb slot is refused (the stand-in context would lie: "when did Zed Zed Zed" tags "did" a verb).

Compiled shapes (what the runtime engine reads; probed on the pinned FCG Go):

* root / part, construction ``<id>``: one conditional unit ``?<id>-unit`` holding the meaning and one ``sequence`` per
  literal (anchor ``k`` spans ``?a<k>-l`` .. ``?a<k>-r``). A slot ``s`` between anchors ``k`` and ``k+1`` becomes the
  subunit ``?slot-<s>`` with ``referent ?s``, ``slot <fills>`` and ``boundaries (?a<k>-r ?a<k+1>-l)``; the meaning must
  name the slot's referent ``?s``. A literal followed by a slot gains a trailing space, one preceded by a slot a leading
  one. A root's first anchor starts with ``SENTINEL_START`` and its last ends with ``SENTINEL_END``; a slot at an edge
  of any form anchors on that edge's sentinel ("brew {X}" → ``"⟦ brew "`` … ``" ⟧"``).
* filler, construction ``<id>``: matches a slot unit of its kind that is not yet filled, with ONE ``sequence`` over the
  slot's edges ``?l`` .. ``?r`` (chained sequences never apply on the engine: probed). Its literals and ``CONSUME`` /
  ``TOKEN`` parts are joined into that one pattern (``["the", CONSUME]`` → ``"the .{1,200}"``), so a filler binds only
  ``?x`` (the slot's referent), ``?l`` and ``?r``: the "the" filler's meaning is ``(definite ?x) (name-span ?x ?l ?r)``,
  and its span includes the determiner (the NP cut drops it, ``runtime/normalise``). A filler with slots instead has one
  ``sequence`` per literal (anchor ``k`` spans ``?b<k>-l`` .. ``?b<k>-r``, or ``?l`` / ``?r`` at the span's edge) and
  contributes one subunit per slot between them (``?slot-<s>``, as in a root).

Literals are normalised IN CONTEXT: a root's or part's form is normalised once per np rendering (``np_renderings``:
the bare stand-in, then the stand-in after the shipped definite determiner), because a determiner can change how the
sentence is tagged ("host Zed" tags "host" a proper noun, "host the Zed" a verb). Where the normalised literals differ,
the spec compiles to ALTERNATE constructions ``<id>#v0``, ``<id>#v1``… with one meaning and one precedence; ``spec_id``
maps an applied name back to its spec. The bare rendering is authoritative; a further rendering the compiler refuses
(a context that misreads the form) adds no alternate. A one-word part's marked reading as written is one more alternate
when it differs (the base editor part "means": "⟪mean⟫" and "⟪means⟫"). A filler is normalised bare only. No base
root or filler has alternates.

Sequence patterns are written with ``ensure_ascii=False``: an escaped ``\\uXXXX`` (a curly apostrophe, a sentinel)
never matches.

Refused (``CompileError``) — FCG authoring traps (spike findings; FCG Go's sequence patterns honour only part of
regex): ``+``, ``(``, ``)``, ``\\``, ``^`` and ``"`` in a literal (they silently match nothing or cannot be escaped
without a backslash), and a sentinel; ``CONSUME`` or any ``.{`` pattern outside a filler; a feature named ``category``
(FCG routes it through the categorial network and never matches it); two adjacent slots; a slot whose referent the
meaning never names; in a part, a slot at an edge; in a filler, an ``answer``, a ``?b<n>`` meaning variable (it binds
only ``?x``, ``?l``, ``?r``), a verb slot, or slots mixed with ``CONSUME`` / ``TOKEN``.

``precedence(spec)`` orders specs for loading: ascending, so the engine's recency tie-break (the last-added
construction wins) lands on the highest precedence: kind first, then, among roots and among fillers, the more
specific (more literal characters) loads later ("did anyone {V} {Y}?" over "did {X} {V} {Y}?"), before standing (net
uses, then confirmations) is consulted, so standing never lets a general form overtake a more specific one. Construction
scores are one constant: precedence is the load order, never a score.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

from memory_kernel.runtime.normalise import SENTINEL_END, SENTINEL_START, VERB_CLOSE, VERB_OPEN, Normalised, normalise
from memory_kernel.runtime.program import MeaningError, Program

if TYPE_CHECKING:
    import pyfcg

Kind = Literal["base", "mined", "taught", "hand"]
Status = Literal["staged", "live"]
Unit = Literal["root", "filler", "part"]
Fills = Literal["np", "name", "clause", "time", "verb"]

#: the consuming pattern, written as its own form part of a filler (Task 4 finding: the only probed pattern that
#: both applies and binds a filler's span; ``.*``, ``[?]*``, ``[^ ].*`` never apply)
CONSUME = ".{1,200}"
#: one non-space run (exactly one normalised token), written as its own form part of a filler: the narrowest pattern
#: that lets a verb filler bind one marked token (the negated class applies on the pinned engine; probed)
TOKEN = "[^ ]{1,200}"
#: the slot kind written between the verb marks
VERB = "verb"
#: stands in for each slot (and ``CONSUME``) while a form is normalised; tagged PROPN in the probed question forms
STAND_IN = "Zed"
#: stands in for a verb slot while a form is normalised: a verb in every probed question form ("when did Zed make Zed ?"
#: normalises to "when do zed ⟪make⟫ zed ?"; with ``STAND_IN`` "did" itself is tagged the verb)
VERB_STAND_IN = "make"
#: every construction's score: precedence is the load order (``precedence``), never a score
SCORE = 0.5

KIND_RANK: Mapping[str, int] = {"base": 0, "mined": 1, "hand": 2, "taught": 3}
SLOT_KEYS = frozenset({"slot", "fills"})
#: the keys a slot may also carry: the kind and constraint of the tool parameter it is bound to (teaching records them;
#: the compiler ignores them, ``runtime/constraints`` checks them at plan time)
SLOT_NOTES = frozenset({"kind", "constraint"})
FORBIDDEN_FEATURE = "category"
#: the variables a filler binds
FILLER_VARS = ("?x", "?l", "?r")

_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SLOT_NAME = re.compile(r"[a-z][a-z0-9_]*")
_CHAIN_VAR = re.compile(r"\?b\d+")
#: characters a literal may not hold: FCG Go's sequence matcher cannot take them (no escape works without a
#: backslash; a double quote is written as an escape inside the pattern string), or the normaliser's sentinels
_UNSUPPORTED = frozenset('+()\\^"') | {SENTINEL_START, SENTINEL_END}
#: an authored literal may not hold the verb marks either: the compiler writes them (a verb in a literal compiles to
#: its marked lemma), an author never does
_UNAUTHORED = _UNSUPPORTED | {VERB_OPEN, VERB_CLOSE}
#: characters it honours as regex syntax: written as a one-character class (``?`` → ``[?]``)
_ESCAPED = frozenset(".?|*[]{}$")


class CompileError(ValueError):
    """A construction spec is malformed or uses something the runtime engine cannot match."""


STANDING_KEYS_ORDER = ("confirmed", "uses_ok", "uses_failed", "uses_world")
STANDING_KEYS = frozenset(STANDING_KEYS_ORDER)


@dataclass(frozen=True)
class Standing:
    confirmed: int = 0
    uses_ok: int = 0
    uses_failed: int = 0
    #: uses the room caused (a precondition, a declined step, a tool error): not evidence against the construction, so
    #: never in precedence
    uses_world: int = 0


FormPart = str | dict[str, str]


@dataclass(frozen=True)
class CxnSpec:
    id: str
    kind: Kind
    form: list[FormPart]
    meaning: list[list[str]]
    answer_shape: str | None = None
    standing: Standing = field(default_factory=Standing)
    status: Status = "staged"
    provenance: dict[str, Any] = field(default_factory=dict)
    unit: Unit = "part"
    #: the slot kind a filler fills (fillers only)
    fills: Fills | None = None

    def __post_init__(self) -> None:
        _check_fields(self)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "unit": self.unit,
            "fills": self.fills,
            "form": [dict(p) if isinstance(p, dict) else p for p in self.form],
            "meaning": [list(p) for p in self.meaning],
            "answer_shape": self.answer_shape,
            "standing": {
                "confirmed": self.standing.confirmed,
                "uses_ok": self.standing.uses_ok,
                "uses_failed": self.standing.uses_failed,
                "uses_world": self.standing.uses_world,
            },
            "status": self.status,
            "provenance": json.loads(json.dumps(self.provenance)),
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> CxnSpec:
        """A spec from its JSON row. ``CompileError`` (never ``TypeError``) for a missing, unknown or mistyped field:
        values are type-checked before they are copied, never coerced (a string is not a list of characters)."""
        if not isinstance(row, Mapping):
            raise CompileError(f"a spec row is a JSON object, got {type(row).__name__}")
        known = {"id", "kind", "unit", "fills", "form", "meaning", "answer_shape", "standing", "status", "provenance"}
        unknown = set(row) - known
        if unknown:
            raise CompileError(f"unknown spec fields {sorted(unknown)}")
        for name in ("id", "kind", "form", "meaning"):
            if name not in row:
                raise CompileError(f"spec is missing {name!r}")
        form, meaning = row["form"], row["meaning"]
        if not isinstance(form, list):
            raise CompileError(f"form must be a list of literal parts and slots, got {form!r}")
        if not isinstance(meaning, list) or not all(isinstance(p, list) for p in meaning):
            raise CompileError(f"meaning must be a list of predicate lists, got {meaning!r}")
        standing = row.get("standing")
        standing = {} if standing is None else standing
        if not isinstance(standing, Mapping) or set(standing) - STANDING_KEYS:
            raise CompileError(f"standing must be an object of {'/'.join(STANDING_KEYS_ORDER)}, got {standing!r}")
        provenance = row.get("provenance")
        provenance = {} if provenance is None else provenance
        if not isinstance(provenance, Mapping):
            raise CompileError(f"provenance must be an object, got {provenance!r}")
        return cls(
            id=row["id"],
            kind=row["kind"],
            form=[dict(p) if isinstance(p, Mapping) else p for p in form],
            meaning=[list(p) for p in meaning],
            answer_shape=row.get("answer_shape"),
            standing=Standing(**{k: standing.get(k, 0) for k in STANDING_KEYS_ORDER}),
            status=row.get("status", "staged"),
            provenance=dict(provenance),
            unit=row.get("unit", "part"),
            fills=row.get("fills"),
        )


def _check_fields(spec: CxnSpec) -> None:
    if not isinstance(spec.id, str) or not _ID.fullmatch(spec.id):
        raise CompileError(f"id {spec.id!r} must match {_ID.pattern} (it names the FCG construction)")
    for name, value, allowed in (
        ("kind", spec.kind, get_args(Kind)),
        ("status", spec.status, get_args(Status)),
        ("unit", spec.unit, get_args(Unit)),
    ):
        if not isinstance(value, str) or value not in allowed:
            raise CompileError(f"{spec.id}: {name} {value!r} is not one of {list(allowed)}")
    if spec.unit == "filler":
        if not isinstance(spec.fills, str) or spec.fills not in get_args(Fills):
            raise CompileError(f"{spec.id}: a filler needs fills, one of {list(get_args(Fills))}; got {spec.fills!r}")
    elif spec.fills is not None:
        raise CompileError(f"{spec.id}: only a filler has fills (this unit is {spec.unit!r}); slots name their kind")
    if spec.answer_shape is not None and not isinstance(spec.answer_shape, str):
        raise CompileError(f"{spec.id}: answer_shape must be a string or null")
    if not isinstance(spec.form, list):
        raise CompileError(f"{spec.id}: form must be a list of literal parts and slots")
    if not isinstance(spec.meaning, list) or not all(
        isinstance(p, list) and all(isinstance(s, str) for s in p) for p in spec.meaning
    ):
        raise CompileError(f"{spec.id}: meaning must be a list of predicate lists of strings")
    if not isinstance(spec.provenance, dict):
        raise CompileError(f"{spec.id}: provenance must be an object")
    s = spec.standing
    if not isinstance(s, Standing) or any(
        isinstance(v, bool) or not isinstance(v, int) for v in (s.confirmed, s.uses_ok, s.uses_failed, s.uses_world)
    ):
        raise CompileError(f"{spec.id}: standing counts must be integers, got {s!r}")


def _specificity(spec: CxnSpec) -> int:
    """A root's or a filler's literal characters (spaces aside); 0 for a part."""
    if spec.unit == "part":
        return 0
    return sum(len(p.replace(" ", "")) for p in spec.form if isinstance(p, str) and p not in _PATTERNS)


def precedence(spec: CxnSpec) -> tuple[int, int, int, int, str]:
    """Ascending load order: kind (base < mined < hand < taught), then a root's or filler's specificity (literal
    characters), then uses_ok − uses_failed, then confirmed, then id. Specificity comes before standing: a well-used
    general construction ("brew {X}") never overtakes a more specific one of its kind ("brew {X} for {R}") and eats its
    text; standing orders competitors of equal specificity."""
    s = spec.standing
    return (KIND_RANK[spec.kind], _specificity(spec), s.uses_ok - s.uses_failed, s.confirmed, spec.id)


#: joins a spec id and its alternate's number in a construction name (``<id>#v<n>``); never in a spec id
ALTERNATE = "#v"


@dataclass(frozen=True)
class CompiledCxn:
    id: str
    unit: Unit
    #: one construction per in-context normalisation of the form: ``(<id>,)``, or alternates ``<id>#v0``, ``<id>#v1``…
    cxns: tuple[pyfcg.Construction, ...]

    @property
    def cxn(self) -> pyfcg.Construction:
        """The one construction of a spec without alternates (a spec with alternates is loaded through ``cxns``)."""
        if len(self.cxns) != 1:
            raise CompileError(f"{self.id} compiles to {len(self.cxns)} alternates: load cxns")
        return self.cxns[0]


def spec_id(name: str) -> str:
    """The spec id an applied construction name belongs to (``<id>#v<n>`` → ``<id>``; a plain id is itself)."""
    return name.split(ALTERNATE, 1)[0]


@dataclass(frozen=True)
class _Slot:
    name: str
    fills: str


@dataclass(frozen=True)
class _Consume:
    pattern: str = CONSUME


#: the filler-only consuming parts, written verbatim into the filler's pattern
_PATTERNS = frozenset({CONSUME, TOKEN})


_Part = str | _Slot | _Consume


def compile_spec(spec: CxnSpec) -> CompiledCxn:
    """Compile one spec to its ``pyfcg.Construction``(s): one per distinct in-context normalisation of its literals
    (``normalised_variants``), named by the spec id, or ``<id>#v<n>`` when there are alternates (``spec_id`` maps an
    applied name back to the spec)."""
    import pyfcg

    _check_fields(spec)
    meaning = _check_meaning(spec)
    parts = _parse_form(spec)
    _check_variables(spec, parts, meaning)
    variants = _variants(spec, parts)
    cxns = []
    for n, literals in enumerate(variants):
        if spec.unit == "filler":
            conditional, contributing = _filler_poles(spec, parts, literals, meaning)
        else:
            conditional, contributing = _phrase_poles(spec, parts, literals, meaning)
        for pole in (conditional, contributing):
            _refuse_category(spec, pole)
        name = spec.id if len(variants) == 1 else f"{spec.id}{ALTERNATE}{n}"
        cxns.append(
            pyfcg.Construction(
                name,
                conditional_pole=conditional,
                contributing_pole=contributing,
                attributes={"score": SCORE, "label": spec.unit},
            )
        )
    return CompiledCxn(id=spec.id, unit=spec.unit, cxns=tuple(cxns))


def _check_meaning(spec: CxnSpec) -> list[list[str]]:
    meaning = spec.meaning
    if not meaning:
        raise CompileError(f"{spec.id}: meaning must be a non-empty list of predicates")
    for predicate in meaning:
        if not predicate or not all(isinstance(s, str) for s in predicate):
            raise CompileError(f"{spec.id}: meaning predicate {predicate!r} must be a non-empty list of strings")
    try:
        Program.from_meaning(meaning)
    except MeaningError as exc:
        raise CompileError(f"{spec.id}: meaning refused: {exc}") from exc
    return [list(p) for p in meaning]


def _check_variables(spec: CxnSpec, parts: list[_Part], meaning: list[list[str]]) -> None:
    symbols = {s for predicate in meaning for s in predicate[1:]}
    if spec.unit == "filler":
        if any(predicate[0] == "answer" for predicate in meaning):
            raise CompileError(f"{spec.id}: a filler names no answer (the construction whose slot it fills does)")
        chained = sorted(s for s in symbols if _CHAIN_VAR.fullmatch(s))
        if chained:
            raise CompileError(f"{spec.id}: a filler binds only {', '.join(FILLER_VARS)}; {chained} are never bound")
    for part in parts:
        if isinstance(part, _Slot) and f"?{part.name}" not in symbols:
            raise CompileError(f"{spec.id}: the meaning never names slot {part.name!r}'s referent ?{part.name}")


def _parse_form(spec: CxnSpec) -> list[_Part]:
    if not spec.form:
        raise CompileError(f"{spec.id}: form must be a non-empty list of literal parts and slots")
    parts: list[_Part] = []
    names: set[str] = set()
    for part in spec.form:
        if isinstance(part, str):
            if part in _PATTERNS:
                if spec.unit != "filler":
                    raise CompileError(f"{spec.id}: the consuming pattern {part} is allowed in a filler only")
                parts.append(_Consume(part))
                continue
            if ".{" in part or "]{" in part:
                raise CompileError(
                    f"{spec.id}: pattern in literal {part!r}: only a filler may consume, with {CONSUME} or {TOKEN} as "
                    "its own part"
                )
            bad = sorted(set(part) & _UNAUTHORED)
            if bad:
                raise CompileError(f"{spec.id}: unsupported pattern characters {bad} in literal {part!r}")
            if not part.strip():
                raise CompileError(f"{spec.id}: empty literal part")
            parts.append(part)
        elif isinstance(part, dict):
            if FORBIDDEN_FEATURE in part:
                raise CompileError(
                    f"{spec.id}: a feature named {FORBIDDEN_FEATURE!r} is never matched (FCG routes it through the "
                    "categorial network): use 'fills'"
                )
            if not SLOT_KEYS <= set(part) <= SLOT_KEYS | SLOT_NOTES or any(
                not isinstance(part[k], str) or not part[k].strip() for k in SLOT_NOTES & set(part)
            ):
                raise CompileError(
                    f"{spec.id}: a slot is {{'slot': name, 'fills': kind}} plus optionally a non-empty 'kind' and "
                    f"'constraint', got {part!r}"
                )
            name, fills = part["slot"], part["fills"]
            if not isinstance(name, str) or not _SLOT_NAME.fullmatch(name):
                raise CompileError(f"{spec.id}: slot name {name!r} must match {_SLOT_NAME.pattern}")
            if not isinstance(fills, str) or fills not in get_args(Fills):
                raise CompileError(f"{spec.id}: slot {name!r} fills {fills!r}, not one of {list(get_args(Fills))}")
            if name in names:
                raise CompileError(f"{spec.id}: slot {name!r} appears twice")
            if spec.unit == "filler" and fills == VERB:
                raise CompileError(f"{spec.id}: a verb slot inside a filler is not supported")
            names.add(name)
            parts.append(_Slot(name, fills))
        else:
            raise CompileError(f"{spec.id}: form part {part!r} is neither a literal nor a slot")
    for before, after in zip(parts, parts[1:], strict=False):
        if isinstance(before, str) and isinstance(after, str):
            raise CompileError(f"{spec.id}: literal parts {before!r} and {after!r} must be one part")
        if not isinstance(before, str) and not isinstance(after, str):
            if isinstance(before, _Slot) and isinstance(after, _Slot) and VERB in (before.fills, after.fills):
                continue  # the verb marks anchor the pair
            what = "slots" if isinstance(before, _Slot) else "consuming parts"
            raise CompileError(f"{spec.id}: adjacent {what} need a literal anchor between them")
    if spec.unit == "part" and any(isinstance(parts[i], _Slot) for i in (0, -1)):
        raise CompileError(f"{spec.id}: a part never has a slot at its edge (only a root anchors one, on a sentinel)")
    if spec.unit == "filler" and names and any(isinstance(p, _Consume) for p in parts):
        raise CompileError(f"{spec.id}: a filler either consumes its span or holds slots, not both")
    return parts


def np_renderings() -> tuple[str, ...]:
    """The texts an np slot is rendered as while a form is normalised in context: the bare stand-in, then the stand-in
    after the determiner of the shipped definite np filler (a ``[literal, CONSUME]`` filler whose meaning marks
    ``definite``): no determiner is written in code (rule 18)."""
    return _np_renderings()


@functools.cache
def _np_renderings() -> tuple[str, ...]:
    from memory_kernel.runtime.base import base_specs  # the shipped data; base imports this module

    for spec in base_specs():
        if (
            spec.unit == "filler"
            and spec.fills == "np"
            and len(spec.form) == 2
            and isinstance(spec.form[0], str)
            and spec.form[1] == CONSUME
            and any(p[:1] == ["definite"] for p in spec.meaning)
        ):
            return (STAND_IN, f"{spec.form[0]} {STAND_IN}")
    return (STAND_IN,)


def _rendered(spec: CxnSpec, parts: list[_Part]) -> tuple[str, ...]:
    """The np renderings a form is normalised under: all of them for a root or part with an np slot, the bare one
    otherwise (a filler's literals sit inside a slot, where a determiner-NP context is never real)."""
    if spec.unit == "filler" or not any(isinstance(p, _Slot) and p.fills == "np" for p in parts):
        return (STAND_IN,)
    return np_renderings()


def _variants(spec: CxnSpec, parts: list[_Part]) -> list[dict[int, str]]:
    """The distinct normalised literals of the form, one per np rendering (bare first): a verb is lemmatised and
    marked in its sentence context, which a determiner can change ("host Zed" tags "host" a proper noun, "host the
    Zed" a verb). The bare rendering is authoritative (its refusal is the spec's); a further rendering the compiler
    refuses is a context that misreads the form ("which the Zed did …" tags "did" a verb) and adds no alternate."""
    variants: list[dict[int, str]] = []
    for n, rendering in enumerate(_rendered(spec, parts)):
        try:
            literals = _normalised_literals(spec, parts, rendering)
        except CompileError:
            if n == 0:
                raise
            continue
        if literals not in variants:
            variants.append(literals)
    marked = _marked_as_written(spec, parts)
    if marked is not None and marked not in variants:
        variants.append(marked)
    return variants


def _marked_as_written(spec: CxnSpec, parts: list[_Part]) -> dict[int, str] | None:
    """A one-word part's second reading: the word as written, lowercased, as a marked verb. A word alone has no
    sentence to tag it in, and the turn's own sentence decides: "archive" alone is a noun and stays plain, but
    "please archive it" holds the verb "⟪archive⟫"; "means" alone lemmatises to "⟪mean⟫", but the tagger keeps
    "⟪means⟫" in "file X means make ...". None for any other form, and for a word that is not one token as written."""
    if spec.unit != "part" or len(parts) != 1 or not isinstance(parts[0], str):
        return None
    word = parts[0].strip().lower()
    if not word or any(ch.isspace() for ch in word) or not any(ch.isalnum() for ch in word):
        return None
    return {0: f"{VERB_OPEN}{word}{VERB_CLOSE}"}


def _rendering(parts: list[_Part], np_text: str) -> tuple[Normalised, list[tuple[int, int]]]:
    """The form rendered (each np slot as ``np_text``, a verb slot as ``VERB_STAND_IN``, any other slot and ``CONSUME``
    as ``STAND_IN``) and normalised, with each part's character range in the rendered text."""
    pieces: list[str] = []
    ranges: list[tuple[int, int]] = []
    at = 0
    for part in parts:
        text = (
            part
            if isinstance(part, str)
            else VERB_STAND_IN
            if isinstance(part, _Slot) and part.fills == VERB
            else np_text
            if isinstance(part, _Slot) and part.fills == "np"
            else STAND_IN
        )
        if pieces:
            at += 1
        ranges.append((at, at + len(text)))
        pieces.append(text)
        at += len(text)
    return normalised_rendering(" ".join(pieces)), ranges


def literal_tokens(spec: CxnSpec) -> list[tuple[str, str]]:
    """The tokens of the form's literals as the bare np rendering normalises them, in form order, each
    ``(normalised text, POS)``: "send {X} to {Y}" is ``("⟪send⟫", "VERB")``, ``("to", "ADP")``. Slots and consuming
    parts give none (their stand-ins are not the form's words). ``CompileError`` for a form that does not parse."""
    parts = _parse_form(spec)
    norm, ranges = _rendering(parts, STAND_IN)
    tokens = list(zip(norm.spans.tokens, norm.spans.pos, strict=True))
    return [
        (norm.text[t[0] : t[1]], pos)
        for part, (start, end) in zip(parts, ranges, strict=True)
        if isinstance(part, str)
        for t, pos in tokens
        if t[2] < end and t[3] > start
    ]


def _normalised_literals(spec: CxnSpec, parts: list[_Part], np_text: str = STAND_IN) -> dict[int, str]:
    """Normalise the whole form once (``_rendering``) and cut each literal's normalised text."""
    norm, ranges = _rendering(parts, np_text)
    has_verb_slot = any(isinstance(p, _Slot) and p.fills == VERB for p in parts)
    literals: dict[int, str] = {}
    for index, (part, (start, end)) in enumerate(zip(parts, ranges, strict=True)):
        tokens = [t for t in norm.spans.tokens if t[2] < end and t[3] > start]
        if any(t[2] < start or t[3] > end for t in tokens):
            raise CompileError(f"{spec.id}: a token straddles the edge of form part {index} ({part!r})")
        if not tokens:
            raise CompileError(f"{spec.id}: form part {index} ({part!r}) normalises to nothing")
        if not isinstance(part, str):
            if len(tokens) != len(np_text.split() if isinstance(part, _Slot) and part.fills == "np" else [part]):
                raise CompileError(f"{spec.id}: the stand-in for form part {index} did not stay its own tokens")
            if isinstance(part, _Slot) and part.fills == VERB and not norm.text[tokens[0][0]].startswith(VERB_OPEN):
                raise CompileError(f"{spec.id}: the verb slot's stand-in is not read as a verb in this form")
            continue
        text = norm.text[tokens[0][0] : tokens[-1][1]]
        if has_verb_slot and VERB_OPEN in text:
            raise CompileError(
                f"{spec.id}: literal {part!r} normalises to a marked verb ({text!r}) beside a verb slot: the verb is "
                "the slot's, and a literal verb there means the stand-in context misread the form"
            )
        bad = sorted(set(text) & _UNSUPPORTED)
        if bad:  # the normaliser never adds these, but a lemma could
            raise CompileError(f"{spec.id}: unsupported pattern characters {bad} in normalised literal {text!r}")
        literals[index] = text
    return literals


@functools.lru_cache(maxsize=16384)
def normalised_rendering(text: str) -> Normalised:
    """``normalise(text)`` for a rendering of a form (its slots stood in for as ``renderings`` and the compiler render
    them), cached for the process: compiling the grammar (``load_store``, at start and on every reload) parses each
    form once, and every later reader of the form reuses that parse: its shape (``normalised_form``: a teach's
    competition and duplicate check, the do-support guard, frame triggers) and a teach's reads of its own renderings.
    Sized for two renderings of each of the 4096 specs ``store.compiled`` keeps, well above a bonfire's grammar. A
    turn's text is never kept here (it is read with ``normalise``)."""
    return normalise(text)


def normalised_form(spec: CxnSpec) -> list[FormPart]:
    """The spec's form with each literal replaced by its normalised text, as the compiler cuts it for the first (bare)
    np rendering (slots and consuming parts as written): what a reader of the form's shape, such as the do-support
    guard, compares tokens against. ``normalised_forms`` gives every alternate."""
    return normalised_forms(spec)[0]


def normalised_forms(spec: CxnSpec) -> list[list[FormPart]]:
    """The spec's form, normalised once per alternate (see ``compile_spec``)."""
    parts = _parse_form(spec)
    return [
        [literals[i] if isinstance(part, str) else spec.form[i] for i, part in enumerate(parts)]
        for literals in _variants(spec, parts)
    ]


def renderings(spec: CxnSpec) -> list[str]:
    """Sample texts of the form, one per np rendering (bare first): literals as authored, each np slot rendered as in
    ``np_renderings``, a verb slot as ``VERB_STAND_IN``, any other slot as ``STAND_IN``. A form with no np slot has one."""
    parts = _parse_form(spec)
    texts = []
    for rendering in _rendered(spec, parts):
        pieces = [
            p
            if isinstance(p, str)
            else VERB_STAND_IN
            if isinstance(p, _Slot) and p.fills == VERB
            else rendering
            if isinstance(p, _Slot) and p.fills == "np"
            else STAND_IN
            for p in parts
        ]
        texts.append(" ".join(pieces))
    return texts


def _escape(text: str) -> str:
    return "".join(f"[{ch}]" if ch in _ESCAPED else ch for ch in text)


def _pattern(text: str) -> str:
    """The sequence pattern as FCG reads it: a quoted string, non-ASCII kept as written (never ``\\uXXXX``)."""
    return json.dumps(text, ensure_ascii=False)


def _spaced(parts: list[_Part], index: int, text: str) -> str:
    """A literal next to a slot or a consumed span carries the joining space on that side."""
    if index > 0 and not isinstance(parts[index - 1], str):
        text = " " + text
    if index + 1 < len(parts) and not isinstance(parts[index + 1], str):
        text = text + " "
    return text


def _anchors(parts: list[_Part], literals: dict[int, str], edges: tuple[str, str] | None) -> list[str | _Slot]:
    """Anchor texts (escaped) alternating with slots. A verb slot's marks join the neighbouring anchors with no space
    toward the slot (``" ⟪"`` … ``"⟫ "``); an anchor next to any other slot carries the joining space on that side;
    ``edges`` (a root's sentinels) open and close the list."""
    items: list[str | _Slot] = [edges[0]] if edges else []
    for index, part in enumerate(parts):
        if isinstance(part, _Slot):
            items += [VERB_OPEN, part, VERB_CLOSE] if part.fills == VERB else [part]
        elif isinstance(part, str):
            items.append(_escape(literals[index]))
        else:  # refused in _parse_form: only a filler consumes
            raise CompileError(f"the consuming pattern {part.pattern} is allowed in a filler only")
    if edges:
        items.append(edges[1])
    anchors: list[str | _Slot] = []
    for item in items:
        if isinstance(item, str) and anchors and isinstance(anchors[-1], str):
            anchors[-1] = f"{anchors[-1]} {item}"
        else:
            anchors.append(item)
    for k, anchor in enumerate(anchors):
        if isinstance(anchor, str):
            before = anchors[k - 1] if k > 0 else None
            after = anchors[k + 1] if k + 1 < len(anchors) else None
            if isinstance(before, _Slot) and before.fills != VERB:
                anchor = " " + anchor
            if isinstance(after, _Slot) and after.fills != VERB:
                anchor = anchor + " "
            anchors[k] = anchor
    return anchors


def _sequences(
    anchors: list[str | _Slot], prefix: str, edges: tuple[str, str] | None = None
) -> tuple[list[list[str]], list[list]]:
    """One ``sequence`` per anchor (``?<prefix><k>-l`` .. ``-r``; with ``edges``, a slot at an edge is bounded by that
    edge variable) and one slot unit per slot, bounded by the anchors around it."""
    form: list[list[str]] = []
    slot_units: list[list] = []
    k = 0
    bounds: list[str | _Slot] = []  # anchor edges and slots, in order
    for index, anchor in enumerate(anchors):
        if isinstance(anchor, _Slot):
            bounds.append(anchor)
            continue
        left = edges[0] if edges and index == 0 else f"?{prefix}{k}-l"
        right = edges[1] if edges and index == len(anchors) - 1 else f"?{prefix}{k}-r"
        k += 1
        form.append(["sequence", _pattern(anchor), left, right])
        bounds += [left, right]
    if edges:
        bounds = [edges[0], *bounds] if isinstance(anchors[0], _Slot) else bounds
        bounds = [*bounds, edges[1]] if isinstance(anchors[-1], _Slot) else bounds
    for index, bound in enumerate(bounds):
        if isinstance(bound, _Slot):
            left, right = bounds[index - 1], bounds[index + 1]
            unit = {"referent": f"?{bound.name}", "slot": bound.fills, "boundaries": [left, right]}
            slot_units.append([f"?slot-{bound.name}", unit])
    return form, slot_units


def _token_edges(anchors: list[str | _Slot]) -> list[str | _Slot]:
    """A part's outer edges on token edges: its first anchor opens on a space unless it opens on a verb mark, its last
    closes on one unless it closes on a verb mark (every token of the normalised text has a space on each side, the
    sentinels being tokens too, and a marked verb's marks bound it). A part has no slot at its edge, so both are
    literal anchors."""
    out = list(anchors)
    if isinstance(out[0], str) and not out[0].startswith((" ", VERB_OPEN)):
        out[0] = " " + out[0]
    if isinstance(out[-1], str) and not out[-1].endswith((" ", VERB_CLOSE)):
        out[-1] = out[-1] + " "
    return out


def _phrase_poles(spec: CxnSpec, parts: list[_Part], literals: dict[int, str], meaning: list) -> tuple[list, list]:
    """Anchors are the literals (escaped, spaced) and the verb marks, with a sentinel on each edge of a root; slot
    units sit between consecutive anchors."""
    edges = (SENTINEL_START, SENTINEL_END) if spec.unit == "root" else None
    anchors = _anchors(parts, literals, edges)
    if spec.unit == "part":
        anchors = _token_edges(anchors)
    form, slot_units = _sequences(anchors, "a")
    top = f"?{spec.id}-unit"
    conditional = [[top, {"#meaning": meaning}, {"#form": form}]]
    contributing = [
        [top, {"subunits": [u[0] for u in slot_units], "boundaries": [form[0][2], form[-1][3]]}],
        *slot_units,
    ]
    return conditional, contributing


def _filler_poles(spec: CxnSpec, parts: list[_Part], literals: dict[int, str], meaning: list) -> tuple[list, list]:
    """One sequence over the slot's edges: literals (escaped, spaced) and ``CONSUME`` / ``TOKEN`` joined into one
    pattern; or, for a filler holding slots, one sequence per literal inside the span and one subunit per slot."""
    if spec.fills is None:  # checked in _check_fields
        raise CompileError(f"{spec.id}: a filler needs fills")
    lock: dict[str, Any] = {"slot": spec.fills, "boundaries": ["?l", "?r"], "footprints": ["not", "filled"]}
    if any(isinstance(part, _Slot) for part in parts):
        form, slot_units = _sequences(_anchors(parts, literals, None), "b", ("?l", "?r"))
        conditional = [["?x-unit", {"referent": "?x", "#meaning": meaning}, {**lock, "#form": form}]]
        contributing = [["?x-unit", {"footprints": ["filled"], "subunits": [u[0] for u in slot_units]}], *slot_units]
        return conditional, contributing
    pattern = "".join(
        part.pattern if isinstance(part, _Consume) else _escape(_spaced(parts, index, literals[index]))
        for index, part in enumerate(parts)
    )
    lock["#form"] = [["sequence", _pattern(pattern), "?l", "?r"]]
    conditional = [["?x-unit", {"referent": "?x", "#meaning": meaning}, lock]]
    contributing = [["?x-unit", {"footprints": ["filled"]}]]
    return conditional, contributing


def _refuse_category(spec: CxnSpec, pole: list) -> None:
    for unit in pole:
        for features in unit[1:]:
            if isinstance(features, dict) and FORBIDDEN_FEATURE in features:
                raise CompileError(f"{spec.id}: a feature named {FORBIDDEN_FEATURE!r} is never matched by FCG")


__all__ = [
    "ALTERNATE",
    "CONSUME",
    "KIND_RANK",
    "SENTINEL_END",
    "SENTINEL_START",
    "STAND_IN",
    "TOKEN",
    "VERB",
    "VERB_CLOSE",
    "VERB_OPEN",
    "VERB_STAND_IN",
    "CompileError",
    "CompiledCxn",
    "CxnSpec",
    "Standing",
    "compile_spec",
    "literal_tokens",
    "normalised_form",
    "normalised_forms",
    "normalised_rendering",
    "np_renderings",
    "precedence",
    "renderings",
    "spec_id",
]
