"""The base construction set: the runtime grammar every bonfire starts with (English: ``data/runtime-grammar/en``).

``base.json`` is ``{"auxiliaries": [...], "constructions": [...]}``. The constructions hold the closed-class grammar as
``CxnSpec`` rows (``kind="base"``, ``status="live"``): the clause types (wh, polar), the wh and degree words (who,
what, which N, where, when, how many, how long), the noun-phrase fillers (names and bare or plural nouns, the / a / an
+ noun, possessives, my / your + noun), deixis (I, me, you), time (yesterday, last week, before / after X), the generic
operations (count, the first / the latest, did anyone) and the one verb filler; and two kinds of row that afford and
never run (``runtime/frames``): the editor's frame parts ("teach", "mean") and the reply roots ("yes", "no").
``auxiliaries`` lists the auxiliary-capable lemmas (closed class) the do-support guard reads. Closed-class words live
only there (repo rule 18); a question such as "where does Kira Dace work?" reads into a program from these rows and the
shared question map alone, with no mining.

Roots compose: a wh or polar root names a verb slot and np slots; the verb filler binds the lemma of one marked verb
token (``lemma-span``), the program resolves it through the lexicon (``frames-of``, then ``wh-role`` with the root's wh
label) and reads events of those frames. The base set has the lowest standing: mined, hand and taught constructions
load after it and specialise it. It changes only with a kernel release.

``reading_guard`` refuses the complete-but-wrong readings of a do-support mis-tag: spaCy sometimes tags the auxiliary
"did" a VERB and the main verb a noun ("how many reports did Kira Dace publish?" normalises to "… ⟪do⟫ kira dace
publish ?"), and a subject-shaped root (``who {V} {Y}?``, ``how many {N} {V} {Y}?``: no auxiliary literal before its
verb slot) then reads the auxiliary as its verb. Such a reading is refused with a reason, and the turn falls back.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable
from importlib import resources
from typing import TYPE_CHECKING

from memory_kernel.runtime.constructions import (
    VERB,
    CompileError,
    CxnSpec,
    compile_spec,
    normalised_form,
    precedence,
    spec_id,
)
from memory_kernel.runtime.normalise import SENTINEL_END, SENTINEL_START, marked_lemma

if TYPE_CHECKING:
    from memory_kernel.runtime.engine import Reading, RuntimeEngine
    from memory_kernel.runtime.executor import SpanText

#: the shipped rows, relative to the ``memory_kernel`` package
BASE_FILE = ("data", "runtime-grammar", "en", "base.json")
_KEYS = frozenset({"auxiliaries", "constructions"})


@functools.cache
def _document() -> str:
    return resources.files("memory_kernel").joinpath(*BASE_FILE).read_text(encoding="utf-8")


def _parts() -> tuple[list[str], list]:
    doc = json.loads(_document())
    if not isinstance(doc, dict) or set(doc) != _KEYS:
        raise CompileError(f"{'/'.join(BASE_FILE)} is an object of exactly {sorted(_KEYS)}")
    auxiliaries, rows = doc["auxiliaries"], doc["constructions"]
    if not isinstance(auxiliaries, list) or not all(isinstance(a, str) and a for a in auxiliaries):
        raise CompileError(f"{'/'.join(BASE_FILE)}: auxiliaries is a list of lemmas")
    if not isinstance(rows, list):
        raise CompileError(f"{'/'.join(BASE_FILE)}: constructions is a list of construction rows")
    return auxiliaries, rows


def base_specs() -> list[CxnSpec]:
    """The shipped base rows, parsed strictly (``CxnSpec.from_json``) once, in file order (a fresh list of the same
    frozen specs on every call). ``CompileError`` for a row that is not a live base row, or a repeated id."""
    return list(_base_specs())


@functools.cache
def _base_specs() -> tuple[CxnSpec, ...]:
    specs = [CxnSpec.from_json(row) for row in _parts()[1]]
    seen: set[str] = set()
    for spec in specs:
        if spec.kind != "base" or spec.status != "live":
            raise CompileError(f"{spec.id}: a base row is kind 'base' and status 'live'")
        if spec.id in seen:
            raise CompileError(f"{spec.id}: repeated in {'/'.join(BASE_FILE)}")
        seen.add(spec.id)
    return tuple(specs)


def base_auxiliaries() -> frozenset[str]:
    """The auxiliary-capable lemmas the base data declares (lowercased)."""
    return frozenset(a.lower() for a in _parts()[0])


def load_base(engine: RuntimeEngine) -> int:
    """Compile the base set and load it into ``engine`` (replacing its grammar) in ascending ``precedence``; the number
    of constructions loaded."""
    cxns = [cxn for spec in sorted(base_specs(), key=precedence) for cxn in compile_spec(spec).cxns]
    engine.load(cxns)
    return len(cxns)


@functools.lru_cache(maxsize=1024)
def _subject_shaped(row: str, auxiliaries: frozenset[str]) -> bool:
    """A root whose verb slot no auxiliary literal precedes (derived from the normalised form)."""
    spec = CxnSpec.from_json(json.loads(row))
    if spec.unit != "root":
        return False
    for part in normalised_form(spec):
        if isinstance(part, dict):
            if part.get("fills") == VERB:
                return True
            continue
        tokens = set(part.replace(SENTINEL_START, " ").replace(SENTINEL_END, " ").split())
        if tokens & auxiliaries:
            return False
    return False


def subject_shaped(spec: CxnSpec, auxiliaries: frozenset[str] | None = None) -> bool:
    """True for a root with a verb slot that no auxiliary literal precedes (``who {V} {Y}?``); False for the
    do-support shapes (``what did {X} {V}?``), other units, and roots without a verb slot."""
    aux = base_auxiliaries() if auxiliaries is None else auxiliaries
    return _subject_shaped(json.dumps(spec.to_json(), sort_keys=True), aux)


def reading_guard(reading: Reading, spans: SpanText, specs: Iterable[CxnSpec] | None = None) -> str | None:
    """A reason to refuse ``reading``, or None. Refused: a reading whose root (the first applied construction that is
    a root of ``specs``, by default the base set) is subject-shaped and whose verb slot binds an auxiliary lemma (a
    do-support mis-tag: the auxiliary read as the main verb)."""
    by_id = {s.id: s for s in (base_specs() if specs is None else specs)}
    applied = [spec_id(a) for a in reading.applied]  # an alternate ("<id>#v<n>") is its spec
    root = next((by_id[a] for a in applied if a in by_id and by_id[a].unit == "root"), None)
    auxiliaries = base_auxiliaries()
    if root is None or not subject_shaped(root, auxiliaries):
        return None
    for predicate in reading.meaning:
        if predicate[0] != "lemma-span" or len(predicate) != 4:
            continue
        left, right = predicate[2], predicate[3]
        if not isinstance(left, int) or not isinstance(right, int):
            continue
        lemma = marked_lemma(spans.normalised, left, right)
        if lemma in auxiliaries:
            return (
                f"{root.id}: its verb slot reads the auxiliary {lemma!r} in a subject-shaped root (a do-support "
                "mis-tag: the main verb was not tagged a verb); the turn falls back"
            )
    return None


__all__ = ["BASE_FILE", "base_auxiliaries", "base_specs", "load_base", "reading_guard", "subject_shaped"]
