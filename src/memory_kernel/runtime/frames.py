"""Frames: what read which words of a turn, reported on every plan so the agent can act on it (1.5.1).

A FRAME PART is a construction per anchor word of every live card and taught phrase, generated when the grammar loads
(``frame_parts``; ``store.load_store`` loads them below everything live, after the drafts): the first verb of the root's
form as the compiler normalises it, as its lemma ("please send {X} to {Y}" is "send"), else its first noun or proper
noun ("the {X} report" is "report"), alone, as a ``part`` whose meaning is ``(frame ?frame <name>)``, ``<name>`` being
the card's tool (its ``procedure``'s name) or the phrase's spec id. A form with neither ("{X} please", "who {X}?") gets
no part: it is reached by its full form only. A part has no slot at its edge, so it binds no role: it applies wherever
its word occurs as a whole token ("make onboard josh" reads a partial make frame; "please take JUJ-3 and put it on
Doing" reads take and put; the compiler anchors a part on token edges, so "archive" never reads inside "archives", and a
one-word part also reads its word as a marked verb: "please archive it", "it was archived"), and a card still reads its
own sentences completely (FCG prefers the complete analysis). Two parts on ADJACENT plain words share the space between
them, so only one of them reads (marked verbs carry their own edges and never collide). The engine applies at most one
frame part per search node (``engine``): the parts of a message are applied one after another, never in every order and
subset, and no complete analysis ever holds a part (a part inside a slot's text leaves the filler nothing whole to
consume), so the search for competing analyses leaves them out.
A part's kind is its root's (``hand`` or ``taught``); it is never stored, never drawn into the hand and never records
standing. The editor's parts ("teach", "means": ``(frame ?frame teach_phrase)``) and the reply roots ("yes", "no":
``(reply ?reply yes)``) are base data rows (``data/runtime-grammar/en/base.json``), never Python words. A frame or
reply op never runs: the plan's gate never sees a partial reading, and a reply plan is unread.

``frames_of(readings, norm, specs, registry)`` turns the analyses of a turn (the primary first, then the competing
ones ``RuntimeEngine.comprehend_all`` found) into Frame JSON, one shape from the kernel to the agent::

    {"name": str, "kind": "tool" | "phrase" | "editor" | "reply" | "question", "construction": str,
     "roles": {role: str | None}, "complete": bool, "analysis": int, "composed_with": str | None}

* per frame part applied: ``tool`` (a card's part: its tool's parameters, every role None), ``phrase`` (a taught
  phrase's part: its slots, None) or ``editor`` (a base part: no roles). A word that leads a card and a taught phrase
  ("take {ticket}" and "take {X} for {Y}") has one part per name, and only one of them can consume the word, so the
  applied part reports the frame of EVERY frame part sharing its trigger (``trigger``: the normalised form), in the
  order of ``specs``: both are afforded, and the agent chooses with both frames in view;
* per analysis, its root: a card is a ``tool`` frame (roles keyed by the tool's parameters), a taught root a
  ``phrase``, a ``reply`` root a ``reply`` (named by its reply, no roles), any other root a ``question`` (named by its
  id); each role is the slot's text cut from the turn (``TokenSpans.cut``: the NP cut, as the tool would get it), None
  when the analysis left the slot without boundaries;
* ``complete``: the analysis passed its goal tests; ``analysis``: the reading's index; ``composed_with``: None here
  (a follow-up that composes with an open frame names it).

A frame that repeats one already listed (the same name, kind, roles and completeness: a word said twice, two
triggers of one editor, the same reading through two fillers) is listed once. A message no construction reads gives
no frame: no catch-all ever completes a message (an ``{X}`` root would read every turn and silence every frame).

A FOLLOW-UP composes with the open frames of the turn before (C11): the plan's ``context`` (``context_frames``: at
most ``MAX_CONTEXT`` Frame objects; the caller sends only frames nothing has run for) names them, and a turn that reads
as a bare fragment (nothing complete, and no frame of its own) is read again inside each open frame's own forms
(``compositions``): for each context frame of kind ``tool`` or ``phrase``, each live root it names (its tool's cards,
or the phrase), each slot whose role it leaves open, that root ``render``-ed with the turn's text in the slot and the
frame's bound roles in the others ("JUJ-3" against an open ``take`` frame is "take JUJ-3"; a root with another slot
still open is never rendered); after every open role, a frame whose ONE bound role is a one-slot root's is rendered
with the turn's text in place of that value (a frame the agent held back: "take the new one" reads completely, the
agent asks "which ticket?", and "JUJ-3" is "take JUJ-3"; a frame with two roles bound is never re-filled). The runtime
reads them in that order; the first that reads completely through a root of that frame is the plan's reading, its
frames of that frame's name carrying ``composed_with``. A turn with a frame of its own ("make onboard josh") is never
put in an open slot: it reads on its own.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from memory_kernel.runtime.constructions import CompileError, CxnSpec, literal_tokens, normalised_forms, spec_id
from memory_kernel.runtime.normalise import PROPER_NOUN_POS, VERB_CLOSE, VERB_OPEN

if TYPE_CHECKING:
    from memory_kernel.runtime.engine import Reading
    from memory_kernel.runtime.normalise import Normalised
    from memory_kernel.runtime.program import Registry

#: the meaning op of a frame part: ``(frame ?frame <name>)``
FRAME_OP = "frame"
#: the meaning op of a reply root: ``(reply ?reply <reply>)``
REPLY_OP = "reply"
#: the procedure op a card's meaning names its tool with
PROCEDURE_OP = "procedure"
#: the first component of every frame part's id (``frame.<name>.<word>``)
FRAME_PREFIX = "frame"
#: the root kinds that get frame parts: catalogue cards and taught phrases
FRAMED_KINDS = frozenset({"hand", "taught"})
#: the POS a frame part anchors on when its form holds no marked verb: a noun or a proper noun (a tag test, no word
#: list)
ANCHOR_POS = frozenset({"NOUN", PROPER_NOUN_POS})
#: frame kind of a frame part, by the part's spec kind
PART_KIND: Mapping[str, str] = {"hand": "tool", "taught": "phrase", "base": "editor"}
#: frame kind of a root, by the root's spec kind (any other root is a question; a reply root is a reply)
ROOT_KIND: Mapping[str, str] = {"hand": "tool", "taught": "phrase"}
#: every frame kind (the Frame JSON's ``kind``)
FRAME_KINDS: tuple[str, ...] = ("tool", "phrase", "editor", "reply", "question")
#: the frame kinds a follow-up composes with: a tool's or a taught phrase's frame with a role still open
COMPOSING_KINDS = frozenset({"tool", "phrase"})
#: the most open frames a plan's context holds (each is read again in its forms: the context bounds the reads)
MAX_CONTEXT = 8
#: a context frame's keys: the Frame JSON's (``name``, ``kind`` and ``roles`` required)
_REQUIRED = ("name", "kind", "roles")
_OPTIONAL: Mapping[str, tuple[type, ...]] = {
    "construction": (str,),
    "complete": (bool,),
    "analysis": (int,),
    "composed_with": (str, type(None)),
}
_WORDY = re.compile(r"[^\W_]")
_SLUG = re.compile(r"[^a-z0-9]+")
#: FCG's renamed unit names: the unit of construction ``<id>`` (``<id>-unit-<n>``) and slot unit ``slot-<name>-<n>``
_HOLDER = re.compile(r"^(.+)-unit(?:-\d+)?$")
_SLOT_UNIT = re.compile(r"^slot-(.+?)(?:-\d+)?$")


def frame_parts(specs: Sequence[CxnSpec]) -> list[CxnSpec]:
    """One frame part per (name, anchor word) of every LIVE root of kind ``hand`` or ``taught`` in ``specs``
    (``CxnSpec(id="frame.<name>.<word>", kind=<the root's kind>, unit="part", form=[word], meaning=[["frame",
    "?frame", name]], status="live")``), deduplicated by id, in the order of ``specs``. The word is ``anchor``'s: the
    form's first marked verb, as its lemma ("please send {X} to {Y}" -> "send"), else its first noun or proper noun
    ("{X} for review" -> "review"); a root with neither, or whose word holds no letter or digit, gets none. ``name`` is
    a card's tool (its procedure's name; the spec id when it names none) or a taught phrase's spec id. The compiler
    anchors a part on whole tokens (``constructions``: "archive" never reads inside "archives", and a one-word part also
    reads its word as a marked verb, "please archive it")."""
    out: dict[str, CxnSpec] = {}
    for spec in specs:
        if spec.unit != "root" or spec.status != "live" or spec.kind not in FRAMED_KINDS:
            continue
        word = anchor(spec)
        slug = _SLUG.sub("-", (word or "").lower()).strip("-")
        if word is None or not _WORDY.search(word) or not slug:
            continue
        name = frame_name(spec)
        part_id = f"{FRAME_PREFIX}.{name.lower()}.{slug}"
        if part_id in out:
            continue
        try:
            out[part_id] = CxnSpec(
                id=part_id,
                kind=spec.kind,
                unit="part",
                form=[word],
                meaning=[[FRAME_OP, "?frame", name]],
                status="live",
            )
        except CompileError:  # a name no construction id can hold
            continue
    return list(out.values())


def anchor(spec: CxnSpec) -> str | None:
    """The word a root's frame part reads, from its form's literals as the compiler normalises them (the bare np
    rendering, ``constructions.literal_tokens``: never a slot's stand-in): the first marked verb, as its lemma (the
    normaliser writes every inflection of a verb so), else the first token tagged ``ANCHOR_POS``, as normalised
    (lowercased); None when the form holds neither, or does not compile."""
    try:
        tokens = literal_tokens(spec)
    except CompileError:
        return None
    verb = next((t[1:-1] for t, _ in tokens if len(t) > 2 and t[0] == VERB_OPEN and t[-1] == VERB_CLOSE), None)
    return verb if verb is not None else next((t for t, pos in tokens if pos in ANCHOR_POS), None)


def frame_name(spec: CxnSpec) -> str:
    """What a root's frame is named: a taught phrase's spec id; a card's tool (the name its procedure op carries),
    else the spec id."""
    if spec.kind == "taught":
        return spec.id
    return next((str(p[2]) for p in spec.meaning if p[:1] == [PROCEDURE_OP] and len(p) > 2), spec.id)


def is_frame_part(spec: CxnSpec) -> bool:
    return spec.unit == "part" and len(spec.meaning) == 1 and spec.meaning[0][:1] == [FRAME_OP]


def trigger(part: CxnSpec) -> frozenset[str]:
    """The texts a frame part matches: its form normalised as the compiler normalises it, one per alternate (the
    alternates the engine matches: "Take" and "take" are one trigger, "archive" is the plain "archive" and the marked
    "⟪archive⟫", "means" the marked "⟪mean⟫" and "⟪means⟫"); the casefolded literals when the form does not normalise.
    Two parts whose triggers meet read the same word, and only one of them can consume it."""
    return _trigger(json.dumps(part.to_json(), sort_keys=True, ensure_ascii=False))


@functools.lru_cache(maxsize=1024)
def _trigger(row: str) -> frozenset[str]:
    spec = CxnSpec.from_json(json.loads(row))
    try:
        return frozenset(" ".join(p for p in form if isinstance(p, str)) for form in normalised_forms(spec))
    except CompileError:
        return frozenset({" ".join(p.casefold() for p in spec.form if isinstance(p, str))})


def reply_of(spec: CxnSpec | None) -> str | None:
    """The reply a reply root reads ("yes", "no"), None for any other construction."""
    if spec is None or spec.unit != "root":
        return None
    return next((str(p[2]) for p in spec.meaning if p[:1] == [REPLY_OP] and len(p) > 2), None)


def root_of(reading: Reading | None, specs: Mapping[str, CxnSpec]) -> CxnSpec | None:
    """The first applied construction of ``reading`` that is a root (an alternate's name maps to its spec)."""
    for name in reading.applied if reading is not None else ():
        spec = specs.get(spec_id(name))
        if spec is not None and spec.unit == "root":
            return spec
    return None


def frames_of(
    readings: Sequence[Reading], norm: Normalised, specs: Mapping[str, CxnSpec], registry: Registry
) -> list[dict[str, Any]]:
    """The Frame JSON of ``readings`` (analysis ``k`` is ``readings[k]``), read on ``norm``; ``specs`` holds every
    construction the engine may apply, frame parts included (see the module docstring)."""
    frames: list[dict[str, Any]] = []
    seen: set[str] = set()
    parts = [s for s in specs.values() if is_frame_part(s)]
    for index, reading in enumerate(readings):
        root = root_of(reading, specs)
        for name in reading.applied:
            spec = specs.get(spec_id(name))
            if spec is None:
                continue
            if is_frame_part(spec):  # every part sharing the word: only one of them could consume it
                word = trigger(spec)
                found = [_part_frame(p, specs, registry) for p in parts if p is spec or trigger(p) & word]
            elif spec is root:
                found = [_root_frame(spec, reading, norm, registry)]
            else:
                continue
            for frame in found:
                frame.update(complete=bool(reading.complete), analysis=index, composed_with=None)
                key = json.dumps([frame["name"], frame["kind"], frame["roles"], frame["complete"]], sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    frames.append(frame)
    return frames


def surface_left(reading: Reading | None, norm: Normalised) -> list[str]:
    """The turn's text no construction consumed in ``reading``, as written: the original surface of each piece of
    ``reading.left`` (from its first token's first character to its last token's last); the whole turn when nothing
    read it."""
    if reading is None:
        text = norm.spans.original.strip()
        return [text] if text else []
    out: list[str] = []
    for left, right in reading.left_spans:
        tokens = [t for t in norm.spans.tokens if t[0] < right and t[1] > left]
        if tokens:
            out.append(norm.spans.original[tokens[0][2] : tokens[-1][3]])
    return out


def render(spec: CxnSpec, values: Mapping[str, str]) -> str | None:
    """``spec``'s form as a text: its literals as written and each slot's value from ``values`` (by slot name), joined
    by single spaces as ``constructions.renderings`` joins them; None when a slot has no value (absent or blank)."""
    pieces: list[str] = []
    for part in spec.form:
        if isinstance(part, str):
            pieces.append(part.strip())
            continue
        value = values.get(part["slot"])
        if value is None or not value.strip():
            return None
        pieces.append(value.strip())
    return " ".join(p for p in pieces if p)


def context_frames(context: Any) -> tuple[dict[str, Any], ...]:
    """The open frames a plan reads a follow-up against: a list of at most ``MAX_CONTEXT`` Frame objects, each with a
    non-empty ``name``, a ``kind`` of ``FRAME_KINDS`` and ``roles`` (role -> text or None); ``construction``,
    ``complete``, ``analysis`` and ``composed_with`` may come along with their Frame types, any other key is refused.
    Returns copies; ``ValueError`` says what is wrong."""
    if not isinstance(context, (list, tuple)):
        raise ValueError(f"must be a list of at most {MAX_CONTEXT} frames")
    if len(context) > MAX_CONTEXT:
        raise ValueError(f"holds more than {MAX_CONTEXT} frames")
    frames: list[dict[str, Any]] = []
    for index, frame in enumerate(context):
        problem = _frame_problem(frame)
        if problem is not None:
            raise ValueError(f"frame {index} {problem}")
        frames.append({**frame, "roles": dict(frame["roles"])})
    return tuple(frames)


def _frame_problem(frame: Any) -> str | None:
    if not isinstance(frame, Mapping):
        return "is not an object"
    unknown = sorted(set(frame) - {*_REQUIRED, *_OPTIONAL})
    if unknown:
        return f"has unknown keys {unknown} (a frame is {sorted({*_REQUIRED, *_OPTIONAL})})"
    missing = [key for key in _REQUIRED if key not in frame]
    if missing:
        return f"has no {missing}"
    if not isinstance(frame["name"], str) or not frame["name"].strip():
        return "needs a non-empty name"
    if frame["kind"] not in FRAME_KINDS:
        return f"has kind {frame['kind']!r}, not one of {list(FRAME_KINDS)}"
    roles = frame["roles"]
    if not isinstance(roles, Mapping) or not all(
        isinstance(k, str) and (v is None or isinstance(v, str)) for k, v in roles.items()
    ):
        return "needs roles as an object of role -> text or null"
    for key, types in _OPTIONAL.items():
        value = frame.get(key)
        if key in frame and (not isinstance(value, types) or (key == "analysis" and isinstance(value, bool))):
            return f"has a {key} of the wrong type"
    return None


def compositions(
    context: Sequence[Mapping[str, Any]], roots: Sequence[CxnSpec], text: str, registry: Registry
) -> list[tuple[str, str]]:
    """The texts a follow-up ``text`` is read again as, in order, each with the name of the frame it composes with,
    in two passes over the frames of ``context`` of kind ``tool`` or ``phrase`` (in order) and each live root of
    ``roots`` a frame names (in order: a tool's cards, or the phrase itself):

    1. each slot whose role the frame leaves open (None or absent): the root rendered with ``text`` in that slot and
       the frame's bound roles in the others; a root another of whose slots is open too is not rendered;
    2. then, a root of ONE slot whose role is the only one the frame has bound: the root rendered with ``text`` in
       place of that value. A frame comes back as context only when nothing ran for it, so a bound frame is one the
       agent held back (it asked "which ticket?" after "take the new one", a complete reading: a ticket may be named
       by its title); the answer replaces the value. A frame with two or more roles bound is never re-filled: which
       value an answer replaces is the agent's to say, not the kernel's.

    Every open role is tried before any bound one. Each text is listed once."""
    fragment = text.strip()
    if not fragment:
        return []
    framed: list[tuple[str, Mapping[str, Any], CxnSpec, dict[str, str]]] = []
    for frame in context:
        kind, name, bound = frame.get("kind"), frame.get("name"), frame.get("roles") or {}
        if kind not in COMPOSING_KINDS:
            continue
        for root in roots:
            if root.unit != "root" or root.status != "live" or ROOT_KIND.get(root.kind) != kind:
                continue
            if frame_name(root) == name:
                framed.append((str(name), bound, root, _slot_roles(root, registry)))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(name: str, rendered: str | None) -> None:
        if rendered is not None and rendered not in seen:
            seen.add(rendered)
            out.append((name, rendered))

    for name, bound, root, role in framed:  # 1. the open roles
        for slot in _slots(root):
            if bound.get(role[slot]) is not None:
                continue
            values = {s: v for s in _slots(root) if s != slot and (v := bound.get(role[s])) is not None}
            add(name, render(root, {**values, slot: fragment}))
    for name, bound, root, role in framed:  # 2. the one bound role of a frame held back
        slots = _slots(root)
        if len(slots) == 1 and {r for r, v in bound.items() if v is not None} == {role[slots[0]]}:
            add(name, render(root, {slots[0]: fragment}))
    return out


def _slot_roles(root: CxnSpec, registry: Registry) -> dict[str, str]:
    """Slot -> the role its frame names it by: a card's slots are its tool's parameters, lowercased (the role is the
    parameter as the catalogue writes it); any other root's roles are its slots."""
    kind = ROOT_KIND.get(root.kind, "question")
    params = {p.lower(): p for p in _params(frame_name(root), registry)} if kind == "tool" else {}
    return {slot: params.get(slot, slot) for slot in _slots(root)}


def _params(name: str, registry: Registry) -> list[str]:
    spec = registry.specs.get(name)
    return [p.name for p in spec.params] if spec is not None and spec.kind == "tool" else []


def _slots(spec: CxnSpec) -> list[str]:
    return [p["slot"] for p in spec.form if isinstance(p, dict)]


def _part_frame(part: CxnSpec, specs: Mapping[str, CxnSpec], registry: Registry) -> dict[str, Any]:
    name = str(part.meaning[0][2])
    kind = PART_KIND.get(part.kind, "tool")
    if kind == "tool":
        roles: dict[str, str | None] = dict.fromkeys(_params(name, registry))
    elif kind == "phrase":
        phrase = specs.get(name)
        roles = dict.fromkeys(_slots(phrase)) if phrase is not None else {}
    else:
        roles = {}
    return {"name": name, "kind": kind, "construction": part.id, "roles": roles}


def _root_frame(root: CxnSpec, reading: Reading, norm: Normalised, registry: Registry) -> dict[str, Any]:
    reply = reply_of(root)
    if reply is not None:
        return {"name": reply, "kind": "reply", "construction": root.id, "roles": {}}
    kind = ROOT_KIND.get(root.kind, "question")
    name = frame_name(root) if kind == "tool" else root.id
    spans: dict[str, tuple[int, int]] = {}
    for holder, unit, left, right in reading.slots:
        owner, slot = _HOLDER.match(holder), _SLOT_UNIT.match(unit)
        if owner is not None and slot is not None and spec_id(owner.group(1)) == root.id:
            spans.setdefault(slot.group(1), (left, right))
    roles = {role: _cut(norm, spans.get(slot)) for slot, role in _slot_roles(root, registry).items()}
    return {"name": name, "kind": kind, "construction": root.id, "roles": roles}


def _cut(norm: Normalised, span: tuple[int, int] | None) -> str | None:
    if span is None:
        return None
    try:
        return norm.spans.cut(*span)
    except ValueError:
        return None


__all__ = [
    "ANCHOR_POS",
    "COMPOSING_KINDS",
    "FRAME_KINDS",
    "FRAME_OP",
    "FRAME_PREFIX",
    "MAX_CONTEXT",
    "REPLY_OP",
    "anchor",
    "compositions",
    "context_frames",
    "frame_name",
    "frame_parts",
    "frames_of",
    "is_frame_part",
    "render",
    "reply_of",
    "root_of",
    "surface_left",
    "trigger",
]
