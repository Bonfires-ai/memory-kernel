"""Teaching: a user defines a construction in chat; it is a draft for them in that chat until they confirm it.

``teach(store, engine, TeachRequest(forms=["brew {X} for {R}", "brew {X}"], body=["take water", "take herb",
"make {X} from them", "give {X} to {R}"], defaults={"R": <the speaker's pronoun>}, ...), registry=...)`` compiles the
definition the way spike 2 did, with no LLM:

* each form becomes one ROOT construction (``kind="taught"``, sentinel-anchored, its slots ``np`` slots filled by the
  existing fillers, so an indefinite X reads through the indefinite filler). A slot written ``{Y:clause}`` is a CLAUSE
  slot instead: only the clause filler fills it (``store.CLAUSE_FILLER``, generated on load), and its value is the
  clause as written ("no one owns it", "the team is small": no determiner dropped, no quantifier read). ``{Y:np}`` is
  ``{Y}``; any other kind is refused. The body still writes the slot ``{Y}``. A slot's kind is no part of a form's
  skeleton, so a clause twin of a live np form competes with it (taught with ``replace``, it takes its place);
  it is never its duplicate;
* each body clause is COMPREHENDED by the live grammar, with each slot written as its own name ("make X") and a slot
  the form leaves out written as its default's text (the grammar reads the default, the code never does).
  A span whose text is a slot's name becomes that slot's variable; any other span, or verb lemma, becomes a ``const``
  (the body's offsets index the body, not the turn); determiner markers are dropped. Every slot written in a clause
  must come back as a span of its own, or the clause is refused ("take I" reads I as the speaker);
* a reference in a clause ("it", "them": a ``window``, with ``plural`` for "them") is an earlier step's result,
  resolved here against the parameter it fills, and neither op is stored. The steps it may refer to are the earlier
  clauses of exactly one catalogue tool (an ``invoke`` clause produces nothing a reference may use); a step fits a
  parameter with no kind, or one whose kind is the kind of the tool's ``returns`` parameter; where no step of that
  kind comes before, an untyped result fits it too (1.5.2, C13: a tool with no ``returns``, every agent tool and
  ``think``, gives an untyped result, which may fill any parameter that takes a string, kinded or not; the tool checks
  the value when it runs; a reader that is no tool parameter, such as an ``invoke``'s slot, is a parameter with no kind
  that takes one value). "it" is the one fitting step's output variable (where several fit and every one is untyped,
  its tool declaring no ``returns``, the nearest of them, as a structured step's "it" is the step just before: "web
  search for X, think: pick one, write it" writes what think gave), and a ``gather`` of that one into a parameter that
  takes several values (``many``), so a reference into a ``many`` parameter is always a list; "them" fills a ``many``
  parameter with a ``gather`` of every fitting step's output, in body order (one fitting step gives a list of one).
  A step that runs once per member (a clause ``distribute_everyone`` distributed: "give tea to everyone") has several
  results, and 1.5 flattens nothing: "it" or "them" that would take it is refused;
* a slot that fills the ``returns`` parameter of a step's tool means that step's result in the later clauses, not the
  slot's text: in "make {X} from them", "give {X} to {R}" give gets what make returned. The first step that returns
  a slot gets its text (a later one gets the earlier step's result); a slot two steps return means the latest one's
  result; a constant never does, and neither does a step that runs once per member (the slot keeps its text). Without
  ``returns`` in the catalogue no slot is rebound, "it" and "them" still bind untyped results where the parameter has
  no kind, and a body 1.4 accepted compiles unchanged;
* a body entry may instead be a STRUCTURED STEP (1.5.2, ``_call_step``): ``{"call": <tool>, "args": {<param>:
  <value>}}``, for any catalogue tool, one with no forms too (an agent's tool, ``think``). A value is a slot (``"{X}"``:
  its variable, or its default's text where the form leaves it out), ``"it"`` (the result of the step just before: it
  must produce one that fits, C13 included), ``"them"`` (as in a clause) or a literal (a ``const``); a value holding a
  brace that is not one slot is refused. Every parameter the tool declares must be given (a tool step takes all of
  them, as a form binds all of them). It compiles to the ops a read clause of the tool gives (``procedure`` + ``seq``
  + the tool op, its constants as ``const`` ops), so a slot filling the tool's ``returns`` parameter rebinds as in a
  clause; nothing is read;
* the clause programs are composed in order under one enclosing procedure (``procedure`` + ``seq``), each a step;
  a set a clause reads whole (``everyone``) distributes over that clause's procedure (``distribute``), so "brew tea
  for everyone" brews once per participant and the next step runs once; a set every reader of which is a tool
  parameter that takes several values (``many``) stays one list, and a set read into both kinds is refused: say it in
  two clauses (``distribute_everyone``, ``everyone_conflicts``). The readers are those of the expanded program, in a
  clause as in a live turn: an ``everyone`` that fills an invoked construction's slot is decided by what that slot
  feeds ("stock everyone", where "stock {R}" makes tea from R, stays one list), and teaching checks the expanded
  program for the conflict;
* a clause read through a TAUGHT root (live, or a usable draft) is stored by reference, never copied:
  ``["invoke", "?t-s<n>", <its id>, <slot args in its slot order>]``. ``expand`` replaces the reference with that
  construction's current program at plan time (a replaced construction resolves to the row a confirm put in its place,
  its ``replaced`` record's ``successor``), so a construction taught again changes every body that invokes it. Teaching
  validates the expanded program; a body that would invoke the draft itself (through a row it replaces) is refused.

A FILLER (2.1: ``fills``, the slot kind it fills; None is a root, taught exactly as before) is taught the same way, with
these differences. Each form becomes a filler of that kind (``unit="filler"``), its slots of any kind the compiler
accepts (what the compiler refuses is refused with its reason). It refers to its own whole span: its meaning opens with
the op that names the words in a slot of its kind (``clause-span``, ``lemma-span`` for a verb, else ``name-span``) over
``?x ?l ?r``, and its body names that span ``{X}``; a slot named X, L or R is refused (X is the span, l and r its
edges). It replaces and competes with no live row of the store (a rival filler is what entrenchment settles), so
``replace`` with ``fills`` is refused; a duplicate of a live filler of its kind is refused as a root's is, at confirm
too, and a form with a base filler's kind and skeleton ("it" as an np filler) competes with that base row and is
refused, as a root competing with a base root is. It is read back inside a live root, not alone: its rendering in the
first slot of its kind of a live root (``_read_inside``, within ``READ_BACK_DEADLINE_MS`` for the whole draft), where
any complete analysis that applies both passes; with no such root the teach is refused before anything is read (``no
live root has a <kind> slot``), and the draft names the root it read inside (``Draft.inside``). It states no score, so
it starts at the neutral one (``constructions.entrenched``). A staged filler that composes changes no one's search:
the live rows decide whether the bonfire's grammar composes (``store.load_store``). The read back tries the bonfire's
own roots before the base set's.

A root may STATE ITS AFFORDANCE (2.1, ``affordance``): the words that raise its frame, one frame part each
(``frames.frame_parts``; ``frames.affordance_problems`` says what a word must be), or ``[]`` for none; unstated (None)
it is derived from the form as before, and the row has no ``affordance`` key. A filler raises no frame and states none.

A teach may MODIFY a live construction (2.1, ``modify``: its id), giving only the parts that change: a form (one, with
the row's slots by name and in order, since a body invoking the row passes them by position), a body (the meaning),
an affordance; an empty ``forms`` keeps the form as stored (its parts as they are, named by a taught row's text or, for
any other row, by its slots in capitals, ``{H:clause}``, which a new body writes), an empty ``body`` the meaning, None
the affordance. The row's unit and the kind it fills are kept. Modifiable: ``taught`` and ``mined`` rows, and ``hand``
rows a grammar installed (``GRAMMAR`` in the provenance); a base row, a catalogue card, any other hand row, a
draft and a part are refused, and so are a modify with ``replace``, one giving nothing, and one that changes nothing.
The result is checked whole as a new construction of that unit and kind is (competition with every live row but the
one it modifies, the compiler, duplicates, base rivals, the read back with the grammar as it will be, the row it
modifies gone), and staged as a draft under a new id (the scheme every draft's rows take) with ``provenance["modifies"]``
and ``replaces`` naming the row: until the confirm everyone else reads the row as it was. It is ``taught``; it keeps
the row's stated ``score`` and, for an installed row, the grammar's provenance keys (a later install of that grammar
leaves it, ``kept_taught``). ``confirm`` refuses it when the row is no longer live (another modify took its place), and
otherwise removes the row with the replace record (its ``successor`` the modify, the row's own records carried along, so
a body invoking it, or a row it once replaced, runs the new one) and gives the new row the row's standing as it stands
then (uses, confirmations, choices; the confirm counted), or a fresh one when the form's skeleton changed.

Refused (a ``Draft`` with problems, nothing staged, the engine's grammar as it was): a malformed request; a form with no
literal holding a letter or digit ("{X}!"); a body clause that does not read completely, reads through no root
construction, reads through a draft not usable here (someone else's, or another chat's), is refused by the do-support
guard (``base.reading_guard``), is a question (it names an answer), holds a reference with no step before it (it would
read the turn window, a referent resolved at use time: a body's referents are slots, constants or earlier steps'
results), holds a reference that is not exactly what the rules above give ("it" with two fitting steps, not all
untyped: say which, or write the step as a structured step; no fitting step; "them" into a parameter that takes one
value; a step that runs once per member), holds a quantified noun phrase (no keyword applies the quantifier), or is
not one procedure; an ``everyone`` read into both kinds of parameter;
a program ``validate`` refuses against the registry (the keyword set plus the catalogue's tools: teaching never adds an
op); a step that runs in the agent (``x-runner: agent``) after an office step that waits for the board's approval, in
the expanded program (C12: nothing could run it once the board approves); a form the compiler refuses; a form
competing with a live construction (the same normalised literal skeleton) unless ``replace`` is set, and with a base
or catalogue (hand) construction even then (an installed row, a hand row a grammar installed, is changed by a modify:
the refusal says so). A draft EVERY form of which is already live as it is (a live taught or
catalogue root with the same normalised form and the same meaning up to the names of its slots and variables: a
DUPLICATE, ``_duplicates``) is refused naming those constructions, and nothing
else is said: teaching it again would change nothing. A draft only some of whose forms are live as they are changes
the others (hum taught again with replace and a new default for its short form): it goes through competition as any
draft does.

Teaching is ADMISSION (1.5.1): a teach reads its body clauses and its forms' own renderings, a handful of reads, and
never searches the inventory for overlaps (the 1.5.0 sweep rendered every live root and read thousands of texts).

* COMPETES: before the draft loads, each form's own renderings (``constructions.renderings``: bare, and with a
  determiner for a form with an np slot) are read as the live grammar stands, and the draft reports their frames
  (``frames.frames_of``: what read which words, bound and open roles), per form text (``Draft.competes``): the parse
  information the agent shows the teacher ("look at the office" also reads as the look card, place "office"). A form
  nothing reads competes with nothing.
* SELF-READ: the grammar is loaded as it will stand once the draft is confirmed (the other drafts, the frame parts,
  the live rows but those the draft replaces, then the draft live and confirmed on top of everything), and each
  form's own renderings must read completely through its spec.

Wider overlaps are settled at read time, not here: a broad phrase ("who {X}?" next to the base who-be-x) is admitted,
and where it and another construction both read a turn, both reach the agent as competing analyses (frames from
``RuntimeEngine.comprehend_all``), the outcome of the call moving their standing. The engine is reloaded afterwards
either way (``load_store``).

A draft is staged in the store (``status="staged"``, provenance: teacher, chat, draft id, the request, the rows it
replaces: its same-skeleton rivals, taught with ``replace``) and the engine reloads with drafts loaded below everything
live (``store.load_store``: first, and at a score below any live row's, ``DRAFT_SCORE``, whatever the draft's own score
and counters say), so a draft reads only where nothing live does. ``admissible`` tells a caller whether a
reading may be used by a user in a chat: not when it applied someone else's draft. ``confirm`` (the teacher only)
reads nothing: it makes the draft live bonfire-wide, removes the rows it replaces (as recorded at teach time; recorded
again in ``provenance["replaced"]``: which row, whose, replaced by whom, with each removed row's own records carried
along, and an installed row's grammar keys given to the row that takes its place) and reloads the engine. A draft whose body
read through the teacher's own drafts (``provenance["reads_staged"]``) is confirmed only after they are: a confirmed
construction never carries an unconfirmed one to everyone; and a draft whose skeleton a construction confirmed since
holds is refused (teach it again with ``replace``).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, get_args

from memory_kernel.runtime.base import base_specs, reading_guard
from memory_kernel.runtime.constructions import (
    CONSUME,
    FILLER_VARS,
    RESERVED_FILLER_SLOTS,
    STAND_IN,
    TOKEN,
    VERB,
    VERB_STAND_IN,
    CompileError,
    CxnSpec,
    Fills,
    compile_spec,
    normalised_form,
    normalised_forms,
    normalised_rendering,
    precedence,
    renderings,
    spec_id,
)
from memory_kernel.runtime.executor import VALUE_SPECS
from memory_kernel.runtime.frames import CLAUSE_FILLS, CLAUSE_SPAN, affordance_problems, frames_of
from memory_kernel.runtime.normalise import marked_lemma, normalise
from memory_kernel.runtime.policy import DEFAULT_POLICY
from memory_kernel.runtime.program import (
    ANSWER_PREDICATE,
    VARIADIC,
    MeaningError,
    OpSpec,
    Program,
    Registry,
    ToolParam,
    validate,
)
from memory_kernel.runtime.store import (
    REPLACED,
    SUCCESSOR,
    ConstructionStore,
    compiled,
    composes,
    generated_fillers,
    load_store,
    loaded_frame_parts,
)

if TYPE_CHECKING:
    from memory_kernel.runtime.engine import Reading, RuntimeEngine

#: provenance keys of a taught row
TEACHER = "teacher"
CHAT = "chat_id"
DRAFT = "draft_id"
FORM = "form"
REQUEST = "request"
REPLACES = "replaces"
#: ``REPLACED`` (a confirmed row's record of the rows it took the place of) and its key ``SUCCESSOR`` (the row of the
#: confirmed draft that took that row's place: the modify, else the one with its skeleton; None: none has it) are the
#: store's (``store.ConstructionStore.successor`` follows them, for teach, the resolver and recorded choices alike)
CONFIRMED_BY = "confirmed_by"
#: the staged rows (drafts) the body read through: the draft is confirmed only after they are
READS_STAGED = "reads_staged"
#: the kinds a replace may remove (catalogue and base rows are never removed by chat teaching)
REPLACEABLE = frozenset({"taught", "mined"})
#: the kinds of live root a draft's form may duplicate (a taught phrase, a catalogue card)
DUPLICABLE = frozenset({"taught", "hand"})
#: a confirmed row's ``provenance`` key: what the confirm leaves to look at (rows left invoking a removed construction)
WARNINGS = "warnings"
#: a modify's ``provenance`` key: the id of the live row it takes the place of (also in ``replaces``)
MODIFIES = "modifies"
#: the install door's provenance keys (``grammar.install``, which takes them from here): the name of the grammar that
#: installed a row and its file's hash. A modified installed row keeps both, so a later install of that grammar leaves it
GRAMMAR = "grammar"
GRAMMAR_HASH = "grammar_sha256"
#: the kinds a modify may name (a ``hand`` row only when a grammar installed it: ``GRAMMAR`` in its provenance);
#: a base row, a catalogue card and any other hand row change elsewhere
MODIFIABLE = frozenset({"taught", "mined", "hand"})
#: what teach says of an affordance for a filler
_NO_FILLER_AFFORDANCE = "a filler raises no frame: only a root states an affordance"

#: the reference keyword: a body clause read through a taught construction stores ``["invoke", step, id, *slot args]``
INVOKE_OP = "invoke"
#: how deep references may nest (a reference inside an invoked construction is one level deeper)
INVOKE_DEPTH = 8
#: how long a filler's read back may search, across every root it is read inside (``_read_inside``): each root's
#: continued search gets what is left, and no root is read once it has passed; each root's first search is not bounded
#: by it, so a filler's refused teach may answer later than 2 s (``mk serve``'s teach route says so). Plans wait behind
#: a teach for ``teach_wait_ms`` (15 s), and one root's read takes about 10 ms on the argument grammar: 2 s leaves room
#: for some 200 roots and keeps a teach that reads in none short
READ_BACK_DEADLINE_MS = 2000
#: the step variable of the reference of body clause ``n`` (a hyphen: no slot referent can take it)
INVOKE_STEP = "?t-s{n}"
#: the prefix of an invoked construction's internal variables, per expansion (a hyphen: no slot can take it)
_INVOKE_FRESH = "?t-i{n}-"

#: the slot kind of a taught slot written ``{Name}`` (filled by the existing np fillers)
SLOT_FILLS = "np"
#: the slot kinds a root's form may write after a colon: ``{Name:np}`` (``{Name}``) and ``{Name:clause}`` (a clause kept
#: as written: the clause filler, ``store.CLAUSE_FILLER``, fills it). A filler's form may write any kind: the compiler
#: judges it
SLOT_KINDS = frozenset({SLOT_FILLS, CLAUSE_FILLS})
#: the slot kinds a taught filler may fill (the compiler's)
_FILLS: dict[str, Fills] = {kind: kind for kind in get_args(Fills)}
#: what a taught filler's body names its own whole span (its referent ``?x``, between the edges ``?l`` and ``?r`` the
#: engine binds): a filler's slot may take neither this name nor the edges' (``RESERVED_FILLER_SLOTS``)
FILLER_SPAN = "X"
_RESERVED = RESERVED_FILLER_SLOTS | {FILLER_SPAN.lower()}
#: the op a taught filler names its own span with, by the kind it fills (any other kind: ``name-span``): the op the
#: generated clause filler and the base verb filler name a slot's words with, so a slot holds the same words whoever
#: fills it
_FILLER_SPAN_OP = {CLAUSE_FILLS: CLAUSE_SPAN, VERB: "lemma-span"}
#: the variable of a taught construction's enclosing procedure (a hyphen: no slot referent can take it)
PROCEDURE_VAR = "?t-proc"
#: the plural marker of a reference ("them"): read with its ``window`` (several results), then dropped
PLURAL_OP = "plural"
#: ops that only mark a body span (its determiner, or a reference as plural): dropped once read (the span itself
#: becomes a slot, a constant or an earlier step's result)
_MARKERS = frozenset({"definite", "indefinite", PLURAL_OP})
#: the turn-window keyword (``executor.WINDOW_RULES``): in a body, a reference to an earlier step's result, resolved
#: at teach time and never stored; refused where no step comes before it (a use-time referent could feed an unrelated
#: earlier answer into a side-effecting step)
WINDOW_OP = "window"
#: span ops whose offsets index the text that was read
_SPAN_OPS = frozenset({"name-span", CLAUSE_SPAN, "lemma-span"})
_SLOT = re.compile(r"\{([^{}]*)\}")
_SLOT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_SLUG = re.compile(r"[^a-z0-9]+")
#: a letter or a digit (a structural test: no word list)
_WORDY = re.compile(r"[^\W_]")
#: stands in for every slot in a skeleton
_ANY_SLOT = "\x00"
#: a filler's parts that consume its span: a skeleton stands one placeholder in for them, as for a slot
_CONSUMING = frozenset({CONSUME, TOKEN})


class TeachError(ValueError):
    """A confirm that cannot go through: no such draft, not its teacher, or a competing form went live meanwhile."""


class NotTeacher(TeachError):
    """A confirm by someone who is not the draft's teacher (``mk serve`` answers it with 403)."""


@dataclass(frozen=True)
class TeachRequest:
    #: the construction's forms, slots written ``{Name}`` ("brew {X} for {R}")
    forms: list[str]
    #: the steps its program runs, in order: clauses, slots written as in the forms ("give {X} to {R}"), or structured
    #: steps ``{"call": <tool>, "args": {<param>: <value>}}`` (1.5.2)
    body: list[str | dict[str, Any]]
    #: the text standing in for a slot a form leaves out (slot name -> text), read by the grammar
    defaults: dict[str, str]
    teacher: str
    chat_id: str
    #: replace the live constructions the forms compete with
    replace: bool = False
    #: None: the forms are ROOTS (a whole turn), taught exactly as before fillers. A slot kind (``np``, ``name``,
    #: ``clause``, ``time``, ``verb``): the forms are FILLERS of that kind, each read inside a slot of another
    #: construction ("{H:clause} even though {D:clause}"). A filler refers to its own whole span, which its body names
    #: ``{X}``
    fills: str | None = None
    #: a root's AFFORDANCE (2.1): the words that raise its frame (``CxnSpec.affordance``). None: derived from the form, as
    #: before; ``[]``: none (reached by its full form only). A filler raises no frame: it states none
    affordance: list[str] | None = None
    #: the id of a LIVE construction this teach MODIFIES (2.1): an empty ``forms`` keeps its form, an empty ``body`` its
    #: meaning, ``affordance`` None its affordance; at least one of the three is given. The result is checked whole as a
    #: new construction of the row's unit and kind, staged as a draft, and on ``confirm`` takes the row's place
    modify: str | None = None


@dataclass(frozen=True)
class Draft:
    #: "" when refused
    draft_id: str
    #: the staged specs, one per form ([] when refused)
    specs: list[CxnSpec] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    #: form text -> the frames (Frame JSON, ``frames.frames_of``) the live grammar reads that form's own renderings as,
    #: before the draft loads ([] when nothing reads them; {} when refused)
    competes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: a filler's form text -> the id of the live root it was read inside (``_read_inside``); {} for a root
    inside: dict[str, str] = field(default_factory=dict)


def admissible(reading: Reading, store: ConstructionStore, chat_id: str, user: str) -> bool:
    """False when ``reading`` applied a staged construction (a draft) whose teacher or chat is not ``user`` in
    ``chat_id``: a draft is usable only by its teacher, in the chat where it was taught, until it is confirmed."""
    for name in reading.applied:
        spec = store.get(spec_id(name))
        if (
            spec is not None
            and spec.status == "staged"
            and (spec.provenance.get(TEACHER), spec.provenance.get(CHAT)) != (user, chat_id)
        ):
            return False
    return True


def teach(store: ConstructionStore, engine: RuntimeEngine, req: TeachRequest, *, registry: Registry) -> Draft:
    """Compile ``req`` against the live grammar and stage it (see the module docstring). ``engine`` must hold the
    store's grammar (``load_store``); ``registry`` is the keyword set plus the catalogue's tools."""
    problems = _request_problems(req)
    if problems:
        return Draft("", [], problems)
    # a modify keeps the unit and the kind its row fills; anything else is what the request says
    target: CxnSpec | None = None
    if req.modify is not None:
        target, problems = _target(store, req)
        if target is None:
            return Draft("", [], problems)
        fills = target.fills
    else:
        fills = _FILLS[req.fills] if req.fills is not None else None  # checked: a kind the compiler knows
    if fills is not None and req.affordance is not None:
        return Draft("", [], [_NO_FILLER_AFFORDANCE])
    kept_form = target is not None and not req.forms  # the form as stored (its parts as they are, never re-parsed)
    kept_meaning = target is not None and not req.body
    if target is not None and kept_form:
        text, form, written = _kept_form(target)
        texts, parsed = [text], [(form, written, [])]
    else:
        texts = list(req.forms)
        parsed = [_parse_form(text, fills=fills) for text in texts]
    problems = [p for _, _, form_problems in parsed for p in form_problems]
    if target is not None and not kept_form:  # an invoking body passes the slots by position: they stay as they are
        problems += _slot_problems(target, texts[0], parsed[0][0])
    names = {name: name.lower() for _, written, _ in parsed for name in written}  # written name -> slot name
    if fills is not None:  # X is the filler's own span and l, r its edges: no slot may take their names
        reserved = RESERVED_FILLER_SLOTS if kept_meaning else _RESERVED  # a kept meaning names no {X}
        problems += [
            f"form {text!r}: a filler's slot may not be named {n}: {{{FILLER_SPAN}}} is the filler's own span, and l "
            "and r are its edges"
            for text, (_, written, _) in zip(texts, parsed, strict=True)
            for n in written
            if n.lower() in reserved
        ]
    for slot, count in Counter(names.values()).items():
        if count > 1:
            problems.append(f"slots {sorted(n for n in names if names[n] == slot)} differ only in case")
    # a filler's body may name its own span; no form writes it, and it takes no default
    spans = {FILLER_SPAN} if fills is not None else set()
    used = {name for entry in req.body for name in _entry_slots(entry)}
    problems += [f"body slot {{{n}}} is in no form" for n in sorted(used - set(names) - spans)]
    problems += [f"default for {{{n}}}, a slot of no form" for n in sorted(set(req.defaults) - set(names))]
    for text, (_, written, _) in zip(texts, parsed, strict=True):
        problems += [
            f"form {text!r} leaves out {{{n}}}, which has no default"
            for n in sorted(set(names) - set(written))
            if n not in req.defaults
        ]
    if problems:
        return Draft("", [], problems)

    draft_id = _draft_id(req)
    name = _slug(next((p for p in parsed[0][0] if isinstance(p, str) and p not in _CONSUMING), "")) or "taught"
    # a modify keeps every stored field it does not name: the row's stated score (an installed row's file score: it is
    # still that grammar's row), its answer shape (a wh root's answers are narrowed by it) and, unless it states one,
    # its affordance
    score = target.score if target is not None else None
    answer_shape = target.answer_shape if target is not None else None
    affordance = target.affordance if target is not None and req.affordance is None else req.affordance
    shells = [
        CxnSpec(
            id=f"{name}.{draft_id}.{k}",
            kind="taught",
            unit="root",
            form=form,
            meaning=[["answer", "?_"]],
            answer_shape=answer_shape,
            score=score,
            affordance=affordance,
        )
        if fills is None
        else CxnSpec(
            id=f"{name}.{draft_id}.{k}",
            kind="taught",
            unit="filler",
            fills=fills,
            form=form,
            meaning=[["const", "?x", "-"]],
            answer_shape=answer_shape,
            score=score,
        )
        for k, (form, _, _) in enumerate(parsed)
    ]
    modifies = target.id if target is not None else None
    if fills is None:
        replaces, competing = _competition(store, shells, texts, req.replace, modifies=modifies)
    else:
        # a filler's form the compiler refuses, with its reason (a slot kind it does not know, a verb slot)
        refused = [
            f"form {text!r}: {why}" for shell, text in zip(shells, texts, strict=True) if (why := _refusal(shell))
        ]
        if refused:
            return Draft("", [], refused)
        # it competes with no live row of the store (a rival filler is what entrenchment settles), only with its draft's
        # other forms and with a base filler of its kind (a taught row would win the base row's tie for every speaker,
        # and a base row records no choices: teaching never changes the base set); a modify replaces its row only
        replaces = [modifies] if modifies is not None else []
        competing = [
            *(_competition(store, shells, texts, req.replace, against_live=False)[1] if modifies is None else []),
            *_base_rivals(shells, texts),
        ]
        # it is read back inside a live root with a slot it fills: with none, nothing is read at all
        if not any(_has_slot(root, fills) for root in _live_roots(store)):
            return Draft("", [], [*competing, f"no live root has a {fills} slot"])

    roots = {s.id for s in store.known_specs() if s.unit == "root"}
    effective = Registry({**registry.specs, **VALUE_SPECS})
    body_names = {**names, FILLER_SPAN: FILLER_SPAN.lower()} if fills is not None else names
    specs: list[CxnSpec] = []
    through: set[str] = set()
    built: list[tuple[CxnSpec, str, list, list[str], list[list[str]]]] = []
    for shell, text, (form, written, _) in zip(shells, texts, parsed, strict=True):
        if target is not None and kept_meaning:  # the meaning as stored (a filler's names its span already)
            meaning, form_problems = [list(p) for p in target.meaning], []
        else:
            meaning, form_problems = _program(
                store, engine, req, set(written) | spans, body_names, roots, name, through, effective.specs
            )
            if fills is not None:  # it refers to its own whole span, named as the words in a slot of its kind are
                meaning = [[_FILLER_SPAN_OP.get(fills, "name-span"), *FILLER_VARS], *meaning]
        problems += [f"form {text!r}: {p}" for p in form_problems]
        if not form_problems:
            built.append((shell, text, form, written, meaning))
    # a draft whose every form is already live as it is changes nothing: it names those constructions and says nothing
    # else (its same-skeleton competition is the duplicate itself); a modify's own row is the one it replaces
    duplicates = _duplicates(
        store,
        [(replace(shell, meaning=meaning), text) for shell, text, _, _, meaning in built],
        exclude=frozenset(replaces) if modifies is not None else frozenset(),
    )
    if duplicates and len(duplicates) == len(texts):
        return Draft("", [], duplicates)
    if competing or problems:
        return Draft("", [], [*competing, *problems])

    # The references resolve as they will once the draft is confirmed: a row it replaces resolves to its form of the
    # same skeleton (a modified row to the draft), so a body that would invoke the draft itself is a cycle, refused here.
    successors = (
        {modifies: shells[0].id}
        if modifies is not None
        else {
            old: shell.id
            for old in replaces
            if (row := store.get(old)) is not None
            for shell in shells
            if _skeleton(row) == _skeleton(shell)
        }
    )
    resolve = resolver(
        store,
        req.chat_id,
        req.teacher,
        successors=successors,
        own={shell.id: replace(shell, meaning=meaning) for shell, _, _, _, meaning in built},
    )
    for shell, text, form, written, meaning in built:
        expanded, _, expand_problems = expand(meaning, resolve, root=shell.id)
        if expand_problems:
            problems += [f"form {text!r}: {p}" for p in expand_problems]
            continue
        slots = [f"?{names[n]}" for n in written]
        if fills is not None:  # a filler's span edges are bound by the engine at use time
            slots += FILLER_VARS[1:]
        try:  # the slots are written at use time by their fillers: stand-in writers for the check
            program = Program.from_meaning([*expanded, *(["const", v, "slot"] for v in slots)])
        except MeaningError as exc:
            problems.append(f"form {text!r}: program refused: {exc}")
            continue
        problems += [f"form {text!r}: {p}" for p in validate(program, effective)]
        # an "everyone" a clause reads, through an invoked construction too, into both kinds of parameter
        problems += [f"form {text!r}: {p}" for p in everyone_conflicts(expanded, effective.specs)]
        # an agent step after a step that waits for the board, through an invoked construction too (C12)
        problems += [f"form {text!r}: {p}" for p in _after_approval(expanded, effective.specs)]
        provenance: dict[str, Any] = {
            TEACHER: req.teacher,
            CHAT: req.chat_id,
            DRAFT: draft_id,
            FORM: text,
            REQUEST: {
                "forms": list(req.forms),
                "body": json.loads(json.dumps(req.body)),  # a structured step is copied, never shared
                "defaults": dict(req.defaults),
                "replace": req.replace,
                # a request with none of the later fields is what it was
                **({"fills": req.fills} if req.fills is not None else {}),
                **({"affordance": list(req.affordance)} if req.affordance is not None else {}),
                **({"modify": req.modify} if req.modify is not None else {}),
            },
        }
        if target is not None:  # an installed row stays its grammar's (``grammar.install`` leaves it: ``kept_taught``)
            provenance[MODIFIES] = target.id
            provenance.update({k: target.provenance[k] for k in (GRAMMAR, GRAMMAR_HASH) if k in target.provenance})
        spec = replace(
            shell,
            # a modify of its affordance alone keeps the form exactly as stored (its slots' notes too); a new meaning or
            # a new form has the notes of what the meaning now binds
            form=target.form
            if target is not None and kept_form and kept_meaning
            else _with_slot_notes(form, expanded, effective),
            meaning=meaning,
            status="staged",
            provenance=provenance,
        )
        if target is not None and _keeps_standing(
            target, spec
        ):  # staged as it will be live (``confirm`` takes it again)
            spec = replace(spec, standing=target.standing)
        try:
            compile_spec(spec)
        except CompileError as exc:
            problems.append(f"form {text!r}: {exc}")
            continue
        specs.append(spec)
    if problems:
        return Draft("", [], problems)
    if target is not None and _unchanged(target, specs[0]):
        return Draft("", [], [f"modify {target.id}: nothing changes (its form, meaning and affordance are as stored)"])

    # What the live grammar reads the forms' own uses as now (the engine still holds the store's grammar), then the
    # draft as it would read once confirmed: its own renderings must read through it, or a filler must read inside a
    # live root (the self-read). The engine's grammar is restored (or gains the staged draft) whatever happens.
    competes = _competes(store, engine, specs, registry, req.chat_id, req.teacher)
    try:
        problems, inside = _self_read(store, engine, specs, replaces, req.chat_id, req.teacher)
        if problems:
            return Draft("", [], problems)
        extra: dict[str, Any] = {READS_STAGED: sorted(through)} if through else {}
        if replaces:
            extra[REPLACES] = replaces
        specs = [replace(spec, provenance={**spec.provenance, **extra}) for spec in specs]
        store.update(puts=specs)
    finally:
        load_store(store, engine)
    return Draft(draft_id, specs, [], competes, inside)


def confirm(store: ConstructionStore, engine: RuntimeEngine, draft_id: str, by: str) -> list[CxnSpec]:
    """The draft's teacher confirms it: its specs go live bonfire-wide (``confirmed`` + 1), the live rows it replaces
    (``provenance["replaces"]``, as recorded at teach time) are removed (recorded in ``provenance["replaced"]``) and
    the engine reloads (``load_store``). Nothing is read: a confirm checks the store only. ``TeachError`` for an
    unknown draft, a confirmer who is not its teacher, a draft that reads through drafts not yet confirmed, or a form
    competing with a construction that went live since."""
    with store._lock:  # the rechecks and the write see one state of the store
        drafts = [s for s in store.staged() if s.provenance.get(DRAFT) == draft_id]
        if not draft_id or not drafts:
            raise TeachError(f"no staged draft {draft_id!r}")
        teacher = drafts[0].provenance.get(TEACHER)
        if by != teacher:
            raise NotTeacher(f"draft {draft_id} is confirmed by its teacher only ({teacher!r}), not {by!r}")
        waiting = sorted(
            {r for s in drafts for r in s.provenance.get(READS_STAGED, [])} & {s.id for s in store.staged()}
        )
        if waiting:
            raise TeachError(
                f"draft {draft_id} reads through drafts not yet confirmed {waiting}: confirm them first (a confirmed "
                "construction never carries an unconfirmed one to everyone)"
            )
        replaces = sorted({r for s in drafts for r in s.provenance.get(REPLACES, [])})
        gone = {r for r in replaces if (row := store.get(r)) is not None and row.status == "live"}
        # a modified row takes one successor: a modify whose row went (another modify of it confirmed since) is refused
        modified = sorted({str(m) for s in drafts if (m := s.provenance.get(MODIFIES)) is not None})
        for cxn_id in modified:
            if cxn_id not in gone:
                after = store.live_successor(cxn_id)
                tail = f": it was replaced by {after} (modify that one)" if after is not None else ""
                raise TeachError(f"draft {draft_id} modifies {cxn_id}, which is no longer live{tail}")
        for spec in drafts:
            if spec.unit != "root":  # a filler's rivals are entrenchment's to settle; a duplicate live since is refused
                twins = _duplicates(store, [(spec, str(spec.provenance.get(FORM, spec.id)))], exclude=frozenset(gone))
                if twins:
                    raise TeachError(f"{spec.id}: {twins[0]} (confirmed since it was taught)")
                continue
            rivals = [r.id for r in _live_roots(store) if r.id not in gone and _same_skeleton(r, spec)]
            if rivals:
                raise TeachError(f"{spec.id} now competes with live {rivals}: teach it again with replace")
        replaced = [
            spec
            for spec in (store.get(r) for r in replaces)
            if spec is not None and spec.status == "live" and (spec.kind in REPLACEABLE or spec.id in modified)
        ]
        record = [
            {
                "id": r.id,
                "kind": r.kind,
                "teacher": r.provenance.get(TEACHER),
                "by": by,
                SUCCESSOR: _successor_among(drafts, r),
            }
            for r in replaced
        ]
        # a replaced row's own records go along (a modify's and a replace's alike), so a body invoking a row it once
        # replaced still finds its successor (``resolver``) and a later install still knows the grammar's row
        # (``grammar.install``: ``kept_taught``); a row with none of its own is recorded as before
        carried = [rec for r in replaced for rec in r.provenance.get(REPLACED, []) if isinstance(rec, Mapping)]
        warnings = _orphan_warnings(store, record, {s.id for s in drafts})
        was = {r.id: r for r in replaced}
        # the row taking the place of a grammar's row is that grammar's from then on (a later install leaves it)
        heirs = {rec[SUCCESSOR]: r for rec, r in zip(record, replaced, strict=True) if r.provenance.get(GRAMMAR)}
        live = []
        for spec in drafts:
            provenance = {**spec.provenance, CONFIRMED_BY: by}
            provenance.pop(WARNINGS, None)
            if warnings:
                provenance[WARNINGS] = warnings
            if record or carried:
                provenance[REPLACED] = [*record, *carried]
            heir = heirs.get(spec.id)
            if heir is not None:
                provenance.update({k: heir.provenance[k] for k in (GRAMMAR, GRAMMAR_HASH) if k in heir.provenance})
            # a modify keeps its row's standing as the row stands now, unless the form's skeleton changed
            modifies = spec.provenance.get(MODIFIES)
            old = was.get(modifies) if isinstance(modifies, str) else None
            base = old.standing if old is not None and _keeps_standing(old, spec) else spec.standing
            standing = replace(base, confirmed=base.confirmed + 1)
            live.append(replace(spec, status="live", standing=standing, provenance=provenance))
        store.update(puts=live, removes=sorted(r.id for r in replaced), replacing_taught=True)
        load_store(store, engine)
    return live


def _orphan_warnings(store: ConstructionStore, record: Sequence[Mapping[str, Any]], drafts: set[str]) -> list[str]:
    """One warning per removed row that has no successor (a replace by generalisation a 1.5.0 draft recorded) and that
    other rows still invoke, directly or through the ids it once replaced: those rows repair at plan time until they
    are taught again."""
    rows = store.all()
    out: list[str] = []
    for gone in (r["id"] for r in record if r.get(SUCCESSOR) is None):
        aliases = {gone}
        grew = True
        while grew:  # an earlier replace left ``old -> ... -> gone``: invokers of ``old`` reach ``gone`` too
            reached = {
                rec["id"]
                for row in rows
                for rec in row.provenance.get(REPLACED, [])
                if isinstance(rec, Mapping) and rec.get(SUCCESSOR) in aliases
            }
            grew = not reached <= aliases
            aliases |= reached
        invokers = sorted(
            {
                row.id
                for row in rows
                if row.id not in aliases and row.id not in drafts
                for predicate in row.meaning
                if predicate and predicate[0] == INVOKE_OP and len(predicate) > 2 and predicate[2] in aliases
            }
        )
        if invokers:
            out.append(f"{gone} was removed with no successor: {', '.join(invokers)} invoke it and will repair")
    return out


# -- the request ------------------------------------------------------------------------------------------------------


def _request_problems(req: TeachRequest) -> list[str]:
    problems = []
    for label, value in (("teacher", req.teacher), ("chat_id", req.chat_id)):
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{label} must be a non-empty string")
    if req.fills is not None and (not isinstance(req.fills, str) or req.fills not in _FILLS):
        problems.append(f"fills is the slot kind a filler fills, one of {sorted(_FILLS)}; got {req.fills!r}")
    modifying = req.modify is not None
    if modifying and (not isinstance(req.modify, str) or not req.modify.strip()):
        problems.append(f"modify names a live construction by its id (a non-empty string); got {req.modify!r}")
    if modifying and req.replace:
        problems.append("modify takes the place of the construction it names: teach it without replace")
    elif req.fills is not None and req.replace:
        problems.append(
            "a filler cannot replace: it competes with no live row, so there is none for it to take the place of "
            "(teach it without replace)"
        )
    if req.affordance is not None:
        problems += affordance_problems(req.affordance)
    for label, items in (("forms", req.forms), ("body", req.body)):
        # a modify may leave either out: what it leaves out is kept as stored
        if not isinstance(items, Sequence) or isinstance(items, str) or (not items and not modifying):
            of = "strings" if label == "forms" else "clauses or calls"
            problems.append(f"{label} must be a {'' if modifying else 'non-empty '}list of {of}")
        elif label == "body":
            for entry in items:
                if isinstance(entry, Mapping):
                    problems += _call_shape(entry)
                elif not (isinstance(entry, str) and entry.strip()):
                    problems.append("body holds an empty entry, or one that is neither a clause nor a call")
        elif not all(isinstance(t, str) and t.strip() for t in items):
            problems.append(f"{label} holds an empty or non-string entry")
    if not isinstance(req.defaults, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in req.defaults.items()
    ):
        problems.append("defaults must map slot names to non-empty texts")
    listed = all(isinstance(v, Sequence) and not isinstance(v, str) for v in (req.forms, req.body))
    if modifying and listed and len(req.forms) > 1:
        problems.append("a modify changes one construction: give one form, or none to keep its form")
    if modifying and listed and not req.forms and not req.body and req.affordance is None:
        problems.append(
            "a modify gives nothing to change: give a form, a body or an affordance (an empty list of words is none)"
        )
    return problems


def _target(store: ConstructionStore, req: TeachRequest) -> tuple[CxnSpec | None, list[str]]:
    """The live row ``req`` modifies, or None and the problem: a base row, a catalogue card, a hand row no grammar
    installed (each changes elsewhere), a draft, a part, a row gone (its successor named) or never stored, or a
    ``fills`` that would make it another unit."""
    cxn_id = str(req.modify)
    where = f"modify {cxn_id}"
    row = store.get(cxn_id)
    if row is None:
        if any(s.id == cxn_id for s in base_specs()):
            return None, [f"{where}: a base construction changes only with a kernel release, never by teaching"]
        if any(s.id == cxn_id for s in store.live()):  # an attached catalogue card: live here, never stored
            return None, [f"{where}: a catalogue card changes when its catalogue is installed again, never by teaching"]
        after = store.live_successor(cxn_id)
        if after is not None:
            return None, [f"{where}: it was replaced by {after} (modify that one)"]
        return None, [f"{where}: no stored construction has this id"]
    if row.status != "live":
        return None, [
            f"{where}: it is a draft staged by {row.provenance.get(TEACHER)!r}, not a live construction (a draft changes "
            "when its teacher teaches it again)"
        ]
    if row.unit not in ("root", "filler"):
        return None, [f"{where}: it is a {row.unit}; teach makes and modifies roots and fillers"]
    if row.kind not in MODIFIABLE or (row.kind == "hand" and not row.provenance.get(GRAMMAR)):
        return None, [f"{where}: a hand row that no grammar installed changes where it was written, never by teaching"]
    if req.fills is not None and req.fills != row.fills:
        what = "a root" if row.fills is None else f"a filler of {row.fills}"
        return None, [f"{where}: it is {what}; a modify keeps what it is (leave fills out)"]
    return row, []


def _bare(form: Sequence[Any]) -> list[Any]:
    """A form's parts without the slots' notes (``_with_slot_notes``): its literals and its slots' names and kinds."""
    return [{"slot": p["slot"], "fills": p["fills"]} if isinstance(p, dict) else p for p in form]


def _kept_form(target: CxnSpec) -> tuple[str, list[Any], list[str]]:
    """A modify that keeps ``target``'s form: the text it is named by, its parts (notes left out) and the slot names a
    new body writes. A taught row's is the text it was taught with; any other row's is written from its parts, each slot
    in capitals (``{H:clause}`` for slot ``h``; ``{Y}`` for an np slot), a consumed span as its pattern."""
    bare = _bare(target.form)
    taught = target.provenance.get(FORM) if target.kind == "taught" else None
    if isinstance(taught, str):
        parts, written, problems = _parse_form(taught, fills=target.fills)
        if not problems and parts == bare:
            return taught, bare, written
    pieces: list[str] = []
    written = []
    for part in bare:
        if not isinstance(part, dict):
            pieces.append(str(part))
            continue
        slot = part["slot"].upper()
        written.append(slot)
        pieces.append(f"{{{slot}}}" if part["fills"] == SLOT_FILLS else f"{{{slot}:{part['fills']}}}")
    return " ".join(pieces), bare, written


def _slot_problems(target: CxnSpec, text: str, form: Sequence[Any]) -> list[str]:
    """A modify's new form must hold ``target``'s slots, by name and in order: a body that invokes the row passes them
    by position (``expand``), and its kept meaning names them."""
    old = _slots(target)
    new = [p["slot"] for p in form if isinstance(p, dict)]
    if new == old:
        return []
    return [
        f"form {text!r}: a modify keeps the slots of {target.id} in their order ({', '.join(old) or 'none'}); this "
        f"form has ({', '.join(new) or 'none'})"
    ]


def _form_skeleton(spec: CxnSpec) -> tuple[str, ...]:
    """A root's or a filler's skeleton as its own unit normalises it: its normalised literals, every slot and every
    consumed span (``CONSUME``, ``TOKEN``) one placeholder. ``CompileError`` for a form that does not normalise."""
    return tuple(_ANY_SLOT if isinstance(p, dict) or p in _CONSUMING else p for p in normalised_form(spec))


def _keeps_standing(old: CxnSpec, new: CxnSpec) -> bool:
    """Whether a modify keeps the old row's standing: the new form's skeleton is the old one's (a form as stored, or
    one that normalises to the same literals around the same slots); a form that does not normalise is a new one."""
    if _bare(old.form) == _bare(new.form):
        return True
    try:
        return _form_skeleton(old) == _form_skeleton(new)
    except CompileError:
        return False


def _unchanged(old: CxnSpec, new: CxnSpec) -> bool:
    """A modify whose result is its row as stored: the same form, the same meaning up to names, the same affordance."""
    return (
        _bare(old.form) == _bare(new.form) and _canonical(old) == _canonical(new) and old.affordance == new.affordance
    )


def _successor_among(drafts: Sequence[CxnSpec], row: CxnSpec) -> str | None:
    """The draft that takes ``row``'s place: the one that modifies it, else the one with its skeleton (a replace)."""
    modifying = next((d.id for d in drafts if d.provenance.get(MODIFIES) == row.id), None)
    if modifying is not None:
        return modifying
    return next((d.id for d in drafts if _skeleton(d) == _skeleton(row)), None)


def _parse_form(text: str, *, fills: str | None = None) -> tuple[list[Any], list[str], list[str]]:
    """A form's ``CxnSpec`` parts (literals stripped, ``{Name}`` and ``{Name:np}`` → an np slot, ``{Name:clause}`` → a
    clause slot; for a root, any other kind after the colon is a problem), its written slot names, problems. A
    filler's form (``fills`` set) writes any kind after the colon: the compiler judges it (``_refusal``)."""
    parts: list = []
    written: list[str] = []
    problems: list[str] = []
    at = 0
    for match in [*_SLOT.finditer(text), None]:
        literal = text[at : match.start() if match else len(text)].strip()
        if "{" in literal or "}" in literal:
            problems.append(f"form {text!r}: a brace outside a {{Name}} slot")
        if literal:
            parts.append(literal)
        if match is None:
            break
        inside = match.group(1).strip()
        name, colon, kind = inside.partition(":")
        name, kind = name.strip(), kind.strip()
        if not _SLOT_NAME.fullmatch(name):
            problems.append(f"form {text!r}: slot {{{inside}}} must be a name ({_SLOT_NAME.pattern})")
        elif colon and kind not in SLOT_KINDS and fills is None:
            problems.append(
                f"form {text!r}: slot {{{inside}}} has an unknown kind {kind!r}: a slot is {{Name}}, {{Name:np}} or "
                "{Name:clause}"
            )
        elif name in written:
            problems.append(f"form {text!r}: slot {{{name}}} appears twice")
        else:
            parts.append({"slot": name.lower(), "fills": kind if colon else SLOT_FILLS})
            written.append(name)
        at = match.end()
    if not any(isinstance(p, str) and _WORDY.search(p) for p in parts):
        problems.append(
            f"form {text!r} has no literal with a letter or digit: a form of slots and punctuation would read almost "
            "every turn"
        )
    return parts, written, problems


def _with_slot_notes(form: list, meaning: list[list[str]], registry: Registry) -> list:
    """``form`` with each slot that feeds a tool parameter carrying that parameter's kind and constraint (``SLOT_FILLS``
    stays the filler kind). A slot feeding several parameters keeps an attribute only when they all agree on it; the
    check at plan time reads the parameters themselves, so this record is what the form shows, not what enforces."""
    bound: dict[str, list[ToolParam]] = {}
    for op, *args in meaning:
        spec = registry.specs.get(op)
        if spec is not None and spec.kind == "tool":
            for param, arg in zip(spec.params, args[len(spec.outputs) :], strict=False):
                if arg.startswith("?"):
                    bound.setdefault(arg[1:], []).append(param)
    out: list = []
    for part in form:
        notes: dict[str, str] = {}
        for attr in ("kind", "constraint"):
            values = {getattr(p, attr) for p in bound.get(part["slot"], ())} if isinstance(part, dict) else set()
            if len(values) == 1 and None not in values:
                notes[attr] = values.pop()
        out.append({**part, **notes} if notes else part)
    return out


def _call_shape(entry: Mapping[str, Any]) -> list[str]:
    """Why a structured step is not ``{"call": <tool>, "args"?: {<param>: <value>}}`` (names and values non-empty
    strings); [] when it is."""
    shape = 'a structured step is {"call": "<tool>", "args": {"<param>": "<value>"}}'
    if set(entry) - {"call", "args"} or not isinstance(entry.get("call"), str) or not entry["call"].strip():
        return [f"{shape}, got {dict(entry)!r}: it needs a call naming the tool, and nothing but call and args"]
    args = entry.get("args", {})
    if not isinstance(args, Mapping) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v.strip() for k, v in args.items()
    ):
        return [f"{shape}, got {dict(entry)!r}: its args map parameter names to non-empty strings"]
    return []


def _clause_slots(clause: str) -> list[str]:
    return [m.group(1).strip() for m in _SLOT.finditer(clause)]


def _entry_slots(entry: str | Mapping[str, Any]) -> list[str]:
    """The slot names a body entry writes: a clause's ``{Name}``s, a structured step's slot values."""
    if isinstance(entry, str):
        return _clause_slots(entry)
    args = entry.get("args", {})
    values = args.values() if isinstance(args, Mapping) else ()
    return [m.group(1).strip() for v in values if isinstance(v, str) and (m := _SLOT.fullmatch(v.strip()))]


def _draft_id(req: TeachRequest) -> str:
    keyed: list[Any] = [req.teacher, req.chat_id, list(req.forms), list(req.body), sorted(req.defaults.items())]
    keyed.append(req.replace)
    if req.fills is not None:  # a root's draft id is what it was before fillers
        keyed.append(req.fills)
    if req.affordance is not None:  # a request with none of the 2.1 fields keys as it did
        keyed.append({"affordance": list(req.affordance)})
    if req.modify is not None:
        keyed.append({"modify": req.modify})
    key = json.dumps(
        keyed,
        ensure_ascii=False,
        sort_keys=True,  # a structured step's keys in any order are one step (a body of clauses has no keys)
    )
    return "t" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")


# -- competition ------------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=4096)
def _skeleton_of(form: str) -> tuple[str, ...]:
    shell = CxnSpec(id="skeleton", kind="taught", unit="root", form=json.loads(form), meaning=[["answer", "?_"]])
    return tuple(_ANY_SLOT if isinstance(p, dict) else p for p in normalised_form(shell))


def _skeleton(spec: CxnSpec) -> tuple[str, ...]:
    """The normalised literal skeleton of a root's form: its normalised literals, every slot one placeholder."""
    return _skeleton_of(json.dumps(spec.form, ensure_ascii=False))


def _shape(spec: CxnSpec) -> tuple[bool, ...]:
    """Where a form's slots are, part by part: two forms with one skeleton have one shape, so comparing shapes first
    spares cutting the skeleton of every live root whose shape differs (its parse is the compiler's: ``load_store``
    parsed every live form, ``normalised_rendering``)."""
    return tuple(isinstance(p, dict) for p in spec.form)


def _same_skeleton(a: CxnSpec, b: CxnSpec) -> bool:
    return _shape(a) == _shape(b) and _skeleton(a) == _skeleton(b)


def _live_roots(store: ConstructionStore) -> list[CxnSpec]:
    return [s for s in [*base_specs(), *store.live()] if s.unit == "root"]


def _competition(
    store: ConstructionStore,
    shells: list[CxnSpec],
    texts: list[str],
    replacing: bool,
    *,
    against_live: bool = True,
    modifies: str | None = None,
) -> tuple[list[str], list[str]]:
    """The live store rows the forms replace, and problems: a form the compiler cannot normalise; two forms of the
    draft with one skeleton; a form competing with a base root (always) or a live row (unless ``replacing``). With
    ``against_live`` False (a filler's forms) no live row is compared: the forms replace nothing. A modify (``modifies``,
    the live row it names) replaces that row, never compared, and competes with every other one (it takes no other's
    place)."""
    problems: list[str] = []
    skeletons: dict[tuple[str, ...], str] = {}
    for shell, text in zip(shells, texts, strict=True):
        try:
            skeleton = _skeleton(shell)
        except CompileError as exc:
            problems.append(f"form {text!r}: {exc}")
            continue
        if skeleton in skeletons:
            problems.append(f"forms {skeletons[skeleton]!r} and {text!r} have one skeleton: they would compete")
        skeletons.setdefault(skeleton, text)
    replaces: list[str] = [modifies] if modifies is not None else []
    if not against_live:
        return replaces, problems
    shapes = {_shape(shell) for shell in shells}
    for live in _live_roots(store):
        text = skeletons.get(_skeleton(live)) if _shape(live) in shapes and live.id != modifies else None
        if text is None:
            continue
        if live.kind == "base":
            problems.append(
                f"form {text!r} competes with the base construction {live.id}: base constructions change only with "
                "a kernel release, never by teaching"
            )
        elif live.kind not in REPLACEABLE and GRAMMAR in live.provenance:
            problems.append(
                f"form {text!r} competes with the installed row {live.id} of grammar {live.provenance[GRAMMAR]!r}: an "
                f"installed row is changed by a modify, never replaced (modify {live.id})"
            )
        elif live.kind not in REPLACEABLE:
            problems.append(
                f"form {text!r} competes with the catalogue construction {live.id}: a catalogue row is never replaced "
                "by chat teaching"
            )
        elif modifies is not None:
            problems.append(
                f"form {text!r} competes with the live construction {live.id}: a modify takes the place of {modifies} "
                "only"
            )
        elif not replacing:
            problems.append(f"form {text!r} competes with the live construction {live.id}: teach it with replace")
        else:
            replaces.append(live.id)
    return sorted(replaces), problems


def _duplicates(
    store: ConstructionStore, drafted: Sequence[tuple[CxnSpec, str]], *, exclude: frozenset[str] = frozenset()
) -> list[str]:
    """One problem per drafted form (spec, its text) that is already live as it is: a live taught or catalogue (or
    installed) row of the same unit (and, for a filler, the same kind) with the same meaning up to the names of its
    slots and variables (``_canonical``), the same slot kinds in form order (a clause twin of an np form changes what
    its slot reads: it competes, it is no duplicate) and the same normalised form (the shape, then the skeleton: only
    a row whose meaning matches is normalised). Rows in ``exclude`` (the row a modify takes the place of) are none."""
    units = {(spec.unit, spec.fills) for spec, _ in drafted}
    rows = [
        (row, _canonical(row))
        for row in store.live()
        if (row.unit, row.fills) in units and row.kind in DUPLICABLE and row.id not in exclude
    ]
    out: list[str] = []
    for spec, text in drafted:
        meaning = _canonical(spec)
        for row, theirs in rows:
            if (row.unit, row.fills) != (spec.unit, spec.fills):
                continue
            try:
                same = theirs == meaning and _slot_kinds(row) == _slot_kinds(spec) and _same_skeleton(row, spec)
            except CompileError:  # a form that does not normalise duplicates nothing (competition says why)
                same = False
            if same:
                out.append(f"form {text!r} is already the live construction {row.id}")
                break
    return out


def _slot_kinds(spec: CxnSpec) -> tuple[str | None, ...]:
    """The kind of each slot of a form, in form order."""
    return tuple(p.get("fills") for p in spec.form if isinstance(p, dict))


def _has_slot(spec: CxnSpec, kind: str) -> bool:
    return kind in _slot_kinds(spec)


def _base_rivals(shells: Sequence[CxnSpec], texts: Sequence[str]) -> list[str]:
    """One problem per filler form with the skeleton of a base filler of its kind (``_form_skeleton``: ``np-it`` for
    "it", ``np-my`` for "my {Y}", ``np-the`` for "the {Y}": a slot stands where the base row consumes its span), worded
    as a root's competition with a base root is. A base filler that only consumes (``np-bare``, the verb filler) has
    a skeleton of no literal, which no taught form has."""
    problems: list[str] = []
    for shell, text in zip(shells, texts, strict=True):
        try:
            mine = _form_skeleton(shell)
        except CompileError:  # the compiler's refusal is said already (``_refusal``)
            continue
        for row_id, fills, skeleton in _base_filler_skeletons():
            if fills == shell.fills and skeleton == mine:
                problems.append(
                    f"form {text!r} competes with the base construction {row_id}: base constructions change only with "
                    "a kernel release, never by teaching"
                )
    return problems


@functools.cache
def _base_filler_skeletons() -> tuple[tuple[str, str | None, tuple[str, ...]], ...]:
    """Each base filler's (id, kind, ``_form_skeleton``), in the base set's order (the base set ships with the kernel)."""
    out: list[tuple[str, str | None, tuple[str, ...]]] = []
    for row in base_specs():
        if row.unit != "filler":
            continue
        try:
            out.append((row.id, row.fills, _form_skeleton(row)))
        except CompileError:
            continue
    return tuple(out)


def _refusal(shell: CxnSpec) -> str | None:
    """The compiler's reason for refusing a filler's form (``constructions.normalised_forms`` parses and normalises it
    as the compiler does), else None."""
    try:
        normalised_forms(shell)
    except CompileError as exc:
        return str(exc)
    return None


def _canonical(spec: CxnSpec) -> tuple[tuple[str, ...], ...]:
    """A root's meaning up to names: every variable renamed by first appearance, its slots' first (in form order, so
    "file {X}" and "file {Y}" are one and "give {X} to {Y}" is not "give {Y} to {X}" over the same body); constants
    compare as text."""
    names = {f"?{slot}": f"?s{k}" for k, slot in enumerate(_slots(spec))}

    def name(arg: Any) -> str:
        return names.setdefault(str(arg), f"?v{len(names)}") if _is_var(arg) else str(arg)

    return tuple((str(p[0]), *(name(a) for a in p[1:])) for p in spec.meaning if p)


def _competes(
    store: ConstructionStore,
    engine: RuntimeEngine,
    specs: Sequence[CxnSpec],
    registry: Registry,
    chat_id: str,
    teacher: str,
) -> dict[str, list[dict[str, Any]]]:
    """Each form's text -> the frames (``frames.frames_of``) of what the live grammar reads the form's own renderings
    as, BEFORE the draft loads (the engine holds the store's grammar): the parse information the agent shows the
    teacher. One read per rendering (its primary analysis); a reading through a draft the teacher may not use here, and
    a frame of the draft's own rows (an earlier staging of the same request), count as nothing."""
    known = {s.id: s for s in [*store.known_specs(), *loaded_frame_parts(store.load_order())]}
    mine = {s.id for s in specs}
    out: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        frames: list[dict[str, Any]] = []
        for rendering in renderings(spec):
            norm = normalised_rendering(rendering)
            reading = engine.comprehend(norm.text)
            if reading is None or not admissible(reading, store, chat_id, teacher):
                continue
            for frame in frames_of([reading], norm, known, registry):
                if frame["construction"] not in mine and frame not in frames:
                    frames.append(frame)
        out[str(spec.provenance.get(FORM, spec.id))] = frames
    return out


def _self_read(
    store: ConstructionStore,
    engine: RuntimeEngine,
    specs: Sequence[CxnSpec],
    replaces: Sequence[str],
    chat_id: str,
    teacher: str,
) -> tuple[list[str], dict[str, str]]:
    """Load the grammar as it will stand once the draft is confirmed (the other drafts, at ``DRAFT_SCORE``, the frame
    parts and the generated fillers, as ``load_store`` loads them, the live rows but those the draft replaces, then the
    draft's specs
    live and confirmed ON TOP of everything, so they apply wherever they can) and read each form's own renderings
    (bare, and with a determiner for an np slot): each must read completely through its spec. It is searched as that
    grammar will be: composing when its live rows, the draft's included, compose (``store.load_store``). A filler is
    read inside a live root instead (``_read_inside``), every form of the draft within one ``READ_BACK_DEADLINE_MS``.
    Returns the problems and, per filler form, the root it read inside. The engine is left holding that grammar (the
    caller restores it)."""
    mine = {s.id for s in specs}
    live = [s for s in store.load_order() if s.id not in replaces and s.id not in mine]
    confirmed = sorted(
        (replace(s, status="live", standing=replace(s.standing, confirmed=s.standing.confirmed + 1)) for s in specs),
        key=precedence,
    )
    others = sorted((s for s in store.staged() if s.id not in mine), key=precedence)
    fillers = generated_fillers([*others, *live, *confirmed])
    order = [*loaded_frame_parts([*live, *confirmed]), *fillers, *live, *confirmed]
    cxns = [cxn for spec in others for cxn in compiled(spec, draft=True)]  # below every live row, as ``load_store``
    engine.load([*cxns, *(cxn for spec in order for cxn in compiled(spec))], slot_order=composes([*live, *confirmed]))
    problems: list[str] = []
    inside: dict[str, str] = {}
    ends = time.monotonic() + READ_BACK_DEADLINE_MS / 1000
    for spec in specs:
        if spec.unit == "filler" and spec.fills is not None:
            # the bonfire's own roots first, then the base set's, each in load order: a filler made for the bonfire's
            # grammar is read there before the deadline goes on base roots, and the draft names that root
            roots = [s for s in live if s.unit == "root" and _has_slot(s, spec.fills)]
            roots = [*(s for s in roots if s.kind != "base"), *(s for s in roots if s.kind == "base")]
            root, problem = _read_inside(store, engine, spec, roots, chat_id, teacher, ends)
            if root is not None:
                inside[str(spec.provenance.get(FORM, spec.id))] = root
            if problem is not None:
                problems.append(problem)
            continue
        for rendering in renderings(spec):
            reading = engine.comprehend(normalised_rendering(rendering).text)
            if reading is None or not reading.complete or spec.id not in {spec_id(a) for a in reading.applied}:
                applied = None if reading is None else list(reading.applied)
                text = spec.provenance.get(FORM, spec.id)
                problems.append(f"form {text!r} does not read its own use {rendering!r} (read: {applied})")
    return problems, inside


def _read_inside(
    store: ConstructionStore,
    engine: RuntimeEngine,
    spec: CxnSpec,
    roots: Sequence[CxnSpec],
    chat_id: str,
    teacher: str,
    ends: float,
) -> tuple[str | None, str | None]:
    """A filler's read back: its own rendering (``renderings``: its slots as their stand-ins) placed in a live root's
    first slot of its kind (``_placed``), in ``roots``' order (``_self_read``: the bonfire's own roots, then the base
    set's, each in load order), until one reads it: every complete
    analysis of that text (``RuntimeEngine.comprehend_every``, under ``Runtime.alternatives``' cap and node cap) is
    searched, and the root passes when one that this teacher may use in this chat (``admissible``) applies both the
    root and the filler (at a neutral score a draft need not be the first reading). The read back ends at ``ends``
    (``time.monotonic``: ``READ_BACK_DEADLINE_MS`` for the whole draft): each root's continued search gets what is left
    of it (at most ``Runtime.alternatives``' deadline), and a root reached after it is not read. Returns (the root it
    read inside, None), or (None, the problem): what each root read, and where the deadline or a bound cut the search
    short before any reading applied the filler."""
    from memory_kernel.runtime.api import (  # the runtime imports this module
        ALTERNATIVES_DEADLINE_MS,
        ALTERNATIVES_NODE_CAP,
        ALTERNATIVES_RAW,
    )

    kind = str(spec.fills)
    own = renderings(spec)[0]  # a filler is rendered bare only
    tried: list[str] = []
    for k, root in enumerate(roots):
        left = int((ends - time.monotonic()) * 1000)
        if left < 1:
            unread = ", ".join(r.id for r in roots[k:])
            tried.append(f"the read back's {READ_BACK_DEADLINE_MS} ms cut it short before {unread} could be read")
            break
        text = _placed(root, kind, own)
        every, cut = engine.comprehend_every(
            normalise(text).text,
            cap=ALTERNATIVES_RAW,
            node_cap=ALTERNATIVES_NODE_CAP,
            deadline_ms=min(ALTERNATIVES_DEADLINE_MS, left),
        )
        for reading in every:
            applied = {spec_id(name) for name in reading.applied}
            if reading.complete and {root.id, spec.id} <= applied and admissible(reading, store, chat_id, teacher):
                return root.id, None
        if cut:
            why = "its search was cut short before any reading applied it"
        else:
            why = f"no reading applies it (read: {[list(r.applied) for r in every[:1]]})"
        tried.append(f"{root.id} as {text!r}: {why}")
    form = spec.provenance.get(FORM, spec.id)
    return None, f"form {form!r} reads inside no live root with a {kind} slot ({'; '.join(tried)})"


def _placed(root: CxnSpec, kind: str, text: str) -> str:
    """``root``'s form rendered as ``renderings`` renders it bare (literals as authored, a verb slot as
    ``VERB_STAND_IN``, any other slot as ``STAND_IN``), its first slot of ``kind`` holding ``text``."""
    pieces: list[str] = []
    placed = False
    for part in root.form:
        if isinstance(part, str):
            pieces.append(part)
        elif not placed and part.get("fills") == kind:
            pieces.append(text)
            placed = True
        else:
            pieces.append(VERB_STAND_IN if part.get("fills") == VERB else STAND_IN)
    return " ".join(pieces)


# -- the program ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Step:
    """A body clause already compiled, as a later clause's reference ("it", "them") sees it."""

    #: the clause as read (its slots written as their names)
    clause: str
    #: its tool op's output variable; None when the clause produces nothing a reference may use (an ``invoke``
    #: clause, or one that is not exactly one tool op)
    out: str | None = None
    #: the result's kind: the kind of the tool's ``returns`` parameter (None: an untyped result, or a kind-less one)
    kind: str | None = None
    #: the slot variable that fills the tool's ``returns`` parameter: later clauses read this step's result for it
    slot: str | None = None
    #: the clause runs once per member (``distribute_everyone`` distributed it): it has several results, so a
    #: reference to it is refused and its slot is not rebound
    several: bool = False
    #: the tool declares no ``returns`` (every agent tool and ``think``): an untyped result, which a sentence's "it"
    #: may take as the nearest of several (C13); a kind-less ``returns`` parameter is not this
    untyped: bool = False


def _program(
    store: ConstructionStore,
    engine: RuntimeEngine,
    req: TeachRequest,
    present: set[str],
    names: dict[str, str],
    roots: set[str],
    procedure: str,
    through: set[str],
    specs: Mapping[str, OpSpec],
) -> tuple[list[list[str]], list[str]]:
    """The taught meaning for a form whose slots are ``present``: one procedure whose steps are the body's entries (a
    clause read by the grammar, or a structured step)."""
    meaning: list[list[str]] = [["procedure", PROCEDURE_VAR, procedure]]
    problems: list[str] = []
    earlier: list[_Step] = []  # the steps so far, in body order: what a reference may stand for
    produced: dict[str, str] = {}  # a slot variable -> the result of the latest step that returned it
    for index, entry in enumerate(req.body, 1):
        if isinstance(entry, Mapping):
            where, label = "body step", _label(entry)
            ops, root, clause_problems, step = _call_step(
                entry, label, present, names, req.defaults, f"?c{index}-", specs, earlier, produced
            )
        else:
            where, label = "body clause", entry
            written = Counter(n for n in _clause_slots(entry) if n in present)
            text = _SLOT.sub(lambda m: _fill(m.group(1).strip(), present, req.defaults), entry)
            ops, root, clause_problems, step = _clause(
                store,
                engine,
                req,
                text,
                written,
                names,
                roots,
                f"?c{index}-",
                INVOKE_STEP.format(n=index),
                through,
                specs,
                earlier,
                produced,
            )
        if clause_problems or step is None:
            problems += [f"{where} {label!r}: {p}" for p in clause_problems]
            continue
        meaning += [["seq", PROCEDURE_VAR, str(index), root], *ops]
        earlier.append(step)
        if step.slot is not None and step.out is not None:
            produced[step.slot] = step.out
    return meaning, problems


def _fill(name: str, present: set[str], defaults: dict[str, str]) -> str:
    return name if name in present else defaults[name]


def _label(entry: Mapping[str, Any]) -> str:
    """A structured step as its problems name it: the step's JSON."""
    return json.dumps(dict(entry), ensure_ascii=False)


#: the references a structured step's value may be (exactly these words; anything else is a literal)
_IT, _THEM = "it", "them"


def _call_step(
    entry: Mapping[str, Any],
    label: str,
    present: set[str],
    names: Mapping[str, str],
    defaults: Mapping[str, str],
    prefix: str,
    specs: Mapping[str, OpSpec],
    earlier: Sequence[_Step],
    produced: Mapping[str, str],
) -> tuple[list[list[str]], str, list[str], _Step | None]:
    """A structured body step ``{"call": <tool>, "args": {<param>: <value>}}`` compiled to what ``_clause`` gives for a
    read clause of that tool: its ops (``procedure`` + ``seq`` + the tool op, the parameters in declared order, each
    literal a ``const`` op, variables under ``prefix``), its procedure variable, problems, and the step for the later
    entries (None when refused). Nothing is read. ``"{X}"`` is the slot's variable (an earlier step's result where one
    returned it: ``produced``), or its default's text where the form leaves it out; ``"it"`` is the result of the step
    just before, which must produce one that fits the parameter (C13: an untyped result fits any); ``"them"`` is what
    it is in a clause (``_reference``); a reference into a ``many`` parameter is a list. Every parameter the tool
    declares must be given."""
    tool = str(entry["call"])
    args: Mapping[str, str] = entry.get("args", {})
    spec = specs.get(tool)
    if spec is None or spec.kind != "tool":
        return [], "", [f"{tool!r} is no tool of the catalogue"], None
    declared = {p.name for p in spec.params}
    problems = [
        f"{tool!r} has no parameter {a!r} (its parameters: {sorted(declared)})" for a in args if a not in declared
    ]
    problems += [
        f"{tool!r} needs its parameter {p.name!r} (a structured step gives every parameter the tool declares)"
        for p in spec.params
        if p.name not in args
    ]
    if problems:
        return [], "", problems, None
    proc, out = f"{prefix}proc", f"{prefix}step1"
    values: list[str] = []
    consts: list[list[str]] = []
    gathers: list[list[str]] = []
    slot_of: dict[str, str] = {}  # a parameter -> the slot variable its value names
    for k, param in enumerate(spec.params, 1):
        var = f"{prefix}a{k}"
        value = args[param.name]
        word = value.strip()
        slot = _SLOT.fullmatch(word)
        if slot is not None:
            name = slot.group(1).strip()
            if name in present:
                slot_of[param.name] = f"?{names[name]}"
                values.append(produced.get(slot_of[param.name], slot_of[param.name]))
                continue
            if name not in defaults:  # "body slot {name} is in no form" said it already
                problems.append(f"{{{name}}} is no slot of the forms and has no default")
            else:  # the form leaves the slot out: its default's text, as written (no reference)
                consts.append(["const", var, defaults[name]])
                values.append(var)
            continue
        if "{" in word or "}" in word:
            problems.append(
                f"{param.name!r}: {value!r} is not a value: a value is one slot {{Name}}, 'it', 'them', or a literal "
                "with no braces"
            )
            continue
        if word in (_IT, _THEM):
            outs, problem = _call_reference(word, param, earlier)
            if problem is not None:
                problems.append(problem)
                continue
            if word == _THEM or param.many:  # a reference into a many parameter is always a list
                gathers.append(["gather", var, *outs])
                values.append(var)
            else:
                values.append(outs[0])
            continue
        consts.append(["const", var, value])
        values.append(var)
    if problems:
        return [], "", problems, None
    ops = [["procedure", proc, tool], ["seq", proc, "1", out], [tool, out, *values], *consts, *gathers]
    returns = next((p for p in spec.params if p.name == spec.returns), None)
    if returns is None:  # no returns: an untyped result, and no slot is rebound
        return ops, proc, [], _Step(label, out, untyped=True)
    return ops, proc, [], _Step(label, out, returns.kind, slot_of.get(returns.name))


def _call_reference(word: str, param: ToolParam, earlier: Sequence[_Step]) -> tuple[list[str], str | None]:
    """What a structured step's ``"it"`` or ``"them"`` filling ``param`` stands for: the output variables, or a problem.
    "it" is the step just before (its result must exist and fit ``param``, ``_fits``); "them" is what it is in a
    clause (``_reference``)."""
    if word == _THEM:
        return _reference(f"'them' for {param.name!r}", True, [param], earlier)
    if not earlier:
        return [], f"'it' for {param.name!r}: no step comes before it, so it has no earlier result to stand for"
    before = earlier[-1]
    if before.out is None:
        return [], f"'it' for {param.name!r} is the result of the step before, and {before.clause!r} produces none"
    if before.several:
        return [], f"'it' for {param.name!r}: {before.clause!r} runs once per member, so it has several results"
    if not _fits(before, param):
        return [], (
            f"'it' for {param.name!r} is the result of {before.clause!r}, a {before.kind}, and {param.name!r} takes a "
            f"{param.kind}"
        )
    return [before.out], None


def _clause(
    store: ConstructionStore,
    engine: RuntimeEngine,
    req: TeachRequest,
    text: str,
    written: Counter[str],
    names: dict[str, str],
    roots: set[str],
    prefix: str,
    step: str,
    through: set[str],
    specs: Mapping[str, OpSpec],
    earlier: Sequence[_Step],
    produced: Mapping[str, str],
) -> tuple[list[list[str]], str, list[str], _Step | None]:
    """One body clause comprehended by the live grammar: its ops (variables renamed with ``prefix``, spans turned into
    slot variables or constants, references into ``earlier`` steps' results), its procedure variable, problems, and
    the clause as a step for the later clauses (None when refused). The staged rows it read through are added to
    ``through``. A clause whose root is a taught construction is stored as a reference to it (``invoke``, its result
    ``step``): the root's own ops are left out, its slot arguments kept, so the clause runs that construction's
    current program. A slot an earlier step returned (``produced``) reads that step's result."""
    norm = normalise(text)
    reading = engine.comprehend(norm.text)
    if reading is None or not reading.complete:
        return [], "", [f"{text!r} does not read in the live grammar"], None
    applied = {spec_id(name) for name in reading.applied}
    if not roots & applied:
        return [], "", [f"{text!r} reads through no root construction ({list(reading.applied)})"], None
    if not admissible(reading, store, req.chat_id, req.teacher):
        return [], "", [f"{text!r} reads through a draft not usable here"], None
    guarded = reading_guard(reading, norm.spans, store.known_specs())
    if guarded is not None:
        return [], "", [f"{text!r}: {guarded}"], None
    if any(p[0] == ANSWER_PREDICATE for p in reading.meaning):
        return [], "", [f"{text!r} is a question (it names an answer), not a step"], None
    through.update(a for a in applied if (row := store.get(a)) is not None and row.status == "staged")
    meaning = list(reading.meaning)
    invoked = next(
        (row for a in applied if (row := store.get(a)) is not None and row.unit == "root" and row.kind == "taught"),
        None,
    )
    binding: dict[str, str] = {}
    if invoked is not None:
        found = _match(invoked.meaning, meaning)
        if found is None:
            return [], "", [f"{text!r}: the program of {invoked.id} does not show in its reading"], None
        binding, matched = found
        meaning = [p for i, p in enumerate(meaning) if i not in matched]
    plural = {str(p[1]) for p in meaning if len(p) == 2 and p[0] == PLURAL_OP}
    slot_of: dict[str, str] = {}
    seen: Counter[str] = Counter()
    raw: list[list[Any]] = []
    references: list[str] = []  # the variables of the clause's references ("it", "them"), resolved below
    if invoked is not None:
        raw.append([INVOKE_OP, step, invoked.id, *(binding[f"?{slot}"] for slot in _slots(invoked))])
    for predicate in meaning:
        op, args = str(predicate[0]), list(predicate[1:])
        if op == WINDOW_OP and args:
            references.append(str(args[0]))
            continue
        if op in _MARKERS:
            continue
        if op in _SPAN_OPS and len(args) == 3 and all(isinstance(a, int) for a in args[1:]):
            var, left, right = args
            if op == "name-span":
                cut = norm.spans.np(left, right)
                if cut.quantified:
                    return (
                        [],
                        "",
                        [f"{text!r}: {cut.text!r} is quantified, and no keyword applies the quantifier"],
                        None,
                    )
                span = cut.text
            elif op == CLAUSE_SPAN:  # a clause slot of the root read through: its text as written, nothing read
                span = norm.spans.surface(left, right)
            else:
                raw.append(["const", var, marked_lemma(norm.text, left, right)])
                continue
            if span in written:
                slot_of[str(var)] = f"?{names[span]}"
                seen[span] += 1
                continue
            raw.append(["const", var, span])
            continue
        raw.append([op, *args])
    problems = [
        f"slot {{{n}}} does not read as a noun phrase of its own in {text!r}"
        for n in sorted(written)
        if seen[n] < written[n]
    ]
    slots = _slots(invoked) if invoked is not None else []
    stands_for: dict[str, str] = {}  # "it": the reference's variable -> the one producer's output variable
    gathers: list[tuple[str, list[str]]] = []  # "them": the reference's variable, gathered from these outputs
    for var in references:
        params = _filled(raw, var, specs, slots)
        outs, problem = _reference(text, var in plural, params, earlier)
        if problem is not None:
            problems.append(problem)
        elif var in plural or (params and all(p.many for p in params)):  # a many parameter always gets a list
            gathers.append((var, outs))
        else:
            stands_for[var] = outs[0]

    def rename(arg: Any) -> str:
        if isinstance(arg, str) and arg.startswith("?"):
            if arg in stands_for:
                return stands_for[arg]
            if arg in slot_of:
                return produced.get(slot_of[arg], slot_of[arg])
            return prefix + arg[1:]
        return str(arg)

    ops = [
        [op, args[0], args[1], *(rename(a) for a in args[2:])] if op == INVOKE_OP else [op, *(rename(a) for a in args)]
        for op, *args in raw
    ]
    ops += [["gather", rename(var), *outs] for var, outs in gathers]  # the outputs are earlier clauses': not renamed
    tops = _tops(ops)
    if len(tops) != 1:
        problems.append(f"{text!r} reads as {len(tops)} procedures, not one step")
        return [], "", problems, None
    # an invoked construction's slot is decided by what it feeds there, as a live turn's expanded reading is; a
    # conflict is refused on the form's expanded program (teach)
    expanded = expand(ops, resolver(store, req.chat_id, req.teacher))[0] if invoked is not None else None
    ops, _ = distribute_everyone(ops, specs, expanded=expanded)
    several = any(p[0] == "distribute" for p in ops)
    return ops, tops[0], problems, _as_step(text, raw, specs, invoked is None, rename, slot_of, several)


def _filled(
    raw: Sequence[Sequence[Any]], var: str, specs: Mapping[str, OpSpec], slots: Sequence[str]
) -> list[ToolParam]:
    """The parameters ``var`` fills in a clause's ops: a tool's catalogue parameter; any other reader (an ``invoke``'s
    slot, named from ``slots`` when given, a value or memory op's input) is a parameter of that name with no kind that
    takes one value."""
    params: list[ToolParam] = []
    for op, *args in raw:
        if op == INVOKE_OP and slots:  # [invoke, step, construction id, *slot args in slot order]
            params += [ToolParam(slot) for slot, arg in zip(slots, args[2:], strict=False) if arg == var]
            continue
        spec = specs.get(str(op))
        if spec is None:  # an unknown op: validate refuses it
            continue
        for k, arg in enumerate(args[len(spec.outputs) :]):
            if arg != var:
                continue
            if spec.kind == "tool" and k < len(spec.params):
                params.append(spec.params[k])
            else:
                name = spec.inputs[min(k, len(spec.inputs) - 1)] if spec.inputs else str(op)
                params.append(ToolParam(name.removesuffix(VARIADIC)))
    return params


def _reference(
    text: str, plural: bool, params: Sequence[ToolParam], earlier: Sequence[_Step]
) -> tuple[list[str], str | None]:
    """What a reference in a body clause stands for: the output variables of the earlier steps it refers to, or a
    problem. A step fits a parameter with no kind, or one whose kind its result has; where no earlier step has that
    kind, an untyped result fits it (C13), so a body 1.5.1 accepted binds as it did. Singular ("it"): the one fitting
    step; where several fit and every one is untyped (its tool declares no ``returns``: an agent tool, ``think``), the
    nearest of them, as a structured step's "it" is the step just before. Plural ("them"): every fitting step, in body
    order, into a parameter that takes several values (``many``). A step that runs once per member has several
    results: a reference to it is refused (1.5 flattens nothing)."""
    if not earlier:
        return [], (
            f"{text!r} refers to an earlier turn's answer (the turn window): a taught body's referent must be a slot or "
            "a constant, or an earlier step's result of this phrase, never a referent resolved at use time"
        )
    one = next((p.name for p in params if not p.many), None)
    if plural and one is not None:
        return [], f"{text!r}: a plural reference fills only a parameter that takes several values ({one!r} takes one)"
    # the steps of the parameters' kind first; where none comes before, an untyped result fits too (C13)
    typed = [s for s in earlier if s.out is not None and all(p.kind is None or p.kind == s.kind for p in params)]
    fits = typed or [s for s in earlier if s.out is not None and s.kind is None]
    if not fits:
        param = params[0].name if params else ""
        return [], (
            f"{text!r} refers to an earlier step's result, and no earlier step of this phrase produces one that fits "
            f"{param!r}"
        )
    if not plural and len(fits) > 1 and all(s.untyped for s in fits):  # C13: the nearest untyped result
        fits = fits[-1:]
    if not plural and len(fits) > 1:
        *rest, last = (repr(s.clause) for s in fits)
        return [], (
            f"{text!r}: it could be the result of {', '.join(rest)} or {last}; say which (use a slot of the form, or "
            "'them' for all), or write the step as a structured step, whose 'it' is the result of the step before"
        )
    several = next((s for s in fits if s.several), None)
    if several is not None:
        return [], f"{text!r}: {several.clause!r} runs once per member, so it has several results"
    return [s.out for s in fits if s.out is not None], None


def _fits(step: _Step, param: ToolParam) -> bool:
    """Whether ``step``'s result may fill ``param``: the parameter has no kind, the result has its kind, or the result
    is untyped (C13: an untyped result fills any parameter that takes a string; the tool checks the value when it
    runs)."""
    return param.kind is None or step.kind is None or step.kind == param.kind


def _after_approval(meaning: Sequence[Sequence[Any]], specs: Mapping[str, OpSpec]) -> list[str]:
    """C12: one problem per step that runs in the agent (``x-runner: agent``) after an office step that waits for the
    board's approval (``DEFAULT_POLICY``: a side effect, unknown or one the speaker's turn does not consent to), in the
    order the steps run (``meaning`` expanded): the board's approval runs the office's step, and nothing would run the
    agent's after it."""
    board: str | None = None
    out: list[str] = []
    for predicate in meaning:
        spec = specs.get(str(predicate[0])) if predicate else None
        if spec is None or spec.kind != "tool":
            continue
        if board is not None and spec.runner == "agent":
            out.append(
                f"{spec.name!r} runs in the agent and would follow {board!r}, which waits for the board's approval: "
                "nothing could run it once the board approves (put the agent's steps first, or teach two phrases)"
            )
        elif (
            board is None
            and spec.runner == "office"
            and DEFAULT_POLICY.needs_confirmation(spec, actor="a", speaker="a")
        ):
            board = spec.name  # the speaker is the step's actor: only the board's consent is left
    return out


def _as_step(
    text: str,
    raw: Sequence[Sequence[Any]],
    specs: Mapping[str, OpSpec],
    may_produce: bool,
    rename: Callable[[Any], str],
    slot_of: Mapping[str, str],
    several: bool,
) -> _Step:
    """The clause as a step for the later clauses' references: a clause of exactly one tool op (and no ``invoke``)
    produces its output; the slot filling its tool's ``returns`` parameter is then this step's result, unless the
    clause runs once per member (``several``: the slot keeps its text, as in 1.4)."""
    tools = [p for p in raw if (spec := specs.get(str(p[0]))) is not None and spec.kind == "tool"]
    if not may_produce or len(tools) != 1:
        return _Step(text, several=several)
    op, *args = tools[0]
    spec = specs[str(op)]
    names = [p.name for p in spec.params]
    if not args or spec.returns not in names:  # no returns: an untyped result, and no slot is rebound
        return _Step(text, rename(args[0]) if args else None, several=several, untyped=True)
    k = names.index(spec.returns)
    at = len(spec.outputs) + k
    filled = args[at] if at < len(args) else None
    slot = slot_of.get(filled) if isinstance(filled, str) and not several else None
    return _Step(text, rename(args[0]), spec.params[k].kind, slot, several)


def _tops(meaning: Sequence[Sequence[Any]]) -> list[str]:
    """The procedure variables (``procedure`` or ``invoke``) that are no step of another procedure, sorted."""
    procedures = {p[1] for p in meaning if len(p) > 1 and p[0] in ("procedure", INVOKE_OP)}
    steps = {p[3] for p in meaning if len(p) == 4 and p[0] == "seq"}
    return sorted(procedures - steps)


def distribute_everyone(
    meaning: Sequence[Sequence[Any]],
    specs: Mapping[str, OpSpec],
    *,
    expanded: Sequence[Sequence[Any]] | None = None,
) -> tuple[list[list[Any]], list[str]]:
    """``meaning`` with each ``everyone`` set that no ``distribute`` scopes yet distributed over the meaning's one top
    procedure, and its problems (``everyone_conflicts``): "give tea to everyone" gives once per participant, in a body
    clause and in a live turn alike. A set whose every reader is a tool parameter that takes several values (``many``)
    stays one list (so does a set nothing reads); a set read into both kinds is not distributed and is a problem; a
    meaning with no single top procedure (a question) is returned as it is. ``specs``: the keyword set plus the
    catalogue's tools. ``expanded``: ``meaning`` with its references (``invoke``) expanded, whose readers decide (by
    default ``meaning``'s own): a body clause that invokes a construction is decided by what the invoked program feeds,
    as a live turn's expanded reading is ("stock everyone", where "stock {R}" makes tea from R, stays one list)."""
    out = [list(p) for p in meaning]
    sets = _everyone_readers(out if expanded is None else expanded, specs)
    problems = [_both(many, one) for many, one in sets.values() if many is not None and one is not None]
    tops = _tops(out)
    if len(tops) != 1:
        return out, problems
    own = {p[1] for p in out if len(p) > 1 and p[0] == "everyone"}
    scoped = {p[2] for p in out if len(p) == 3 and p[0] == "distribute"}
    distributed = [
        s for s, (many, one) in sets.items() if s in own and s not in scoped and one is not None and many is None
    ]
    return out + [["distribute", tops[0], s] for s in distributed], problems


def everyone_conflicts(meaning: Sequence[Sequence[Any]], specs: Mapping[str, OpSpec]) -> list[str]:
    """One problem per ``everyone`` set read both into a parameter that takes several values (``many``: the whole set)
    and into one that takes one (a member at a time): the set cannot be both, so the clause must be said as two."""
    sets = _everyone_readers(meaning, specs).values()
    return [_both(many, one) for many, one in sets if many is not None and one is not None]


def _both(many: str, one: str) -> str:
    return (
        f"'everyone' fills both a parameter that takes several values ({many!r}) and one that takes one ({one!r}): "
        "say it in two clauses"
    )


def _everyone_readers(
    meaning: Sequence[Sequence[Any]], specs: Mapping[str, OpSpec]
) -> dict[str, tuple[str | None, str | None]]:
    """Each ``everyone`` set variable -> the first parameter reading it that takes several values, and the first that
    takes one (any reader that is no tool parameter takes one); None when there is none. A ``distribute`` and the
    markers read nothing."""
    readers = [p for p in meaning if p and p[0] != "distribute" and p[0] not in _MARKERS]
    out: dict[str, tuple[str | None, str | None]] = {}
    for p in meaning:
        if len(p) > 1 and p[0] == "everyone":
            params = _filled(readers, str(p[1]), specs, ())
            many = next((q.name for q in params if q.many), None)
            out[str(p[1])] = (many, next((q.name for q in params if not q.many), None))
    return out


def _slots(spec: CxnSpec) -> list[str]:
    """A construction's slot names in form order: the order of an ``invoke``'s slot arguments."""
    return [part["slot"] for part in spec.form if isinstance(part, dict)]


def _is_var(symbol: Any) -> bool:
    return isinstance(symbol, str) and symbol.startswith("?")


def _match(
    pattern: Sequence[Sequence[Any]], meaning: Sequence[Sequence[Any]]
) -> tuple[dict[str, str], set[int]] | None:
    """Where ``pattern`` (a construction's stored meaning) shows in ``meaning`` (a reading, its variables renamed by the
    engine): the variable renaming (one-to-one) and the indices of the predicates it covers, else None. Constants
    compare as text (the engine may hand an index back as an int)."""
    order = sorted(range(len(pattern)), key=lambda i: -sum(not _is_var(a) for a in pattern[i]))
    binding: dict[str, str] = {}
    bound: set[str] = set()
    taken: set[int] = set()

    def walk(k: int) -> bool:
        if k == len(order):
            return True
        want = pattern[order[k]]
        for j, have in enumerate(meaning):
            if j in taken or len(have) != len(want) or have[0] != want[0]:
                continue
            added: list[str] = []
            fits = True
            for a, b in zip(want[1:], have[1:], strict=True):
                if _is_var(a) != _is_var(b):
                    fits = False
                elif not _is_var(a):
                    fits = str(a) == str(b)
                elif a in binding:
                    fits = binding[a] == b
                elif b in bound:
                    fits = False
                else:
                    binding[a] = b
                    bound.add(b)
                    added.append(a)
                if not fits:
                    break
            if fits:
                taken.add(j)
                if walk(k + 1):
                    return True
                taken.discard(j)
            for a in added:
                bound.discard(binding.pop(a))
        return False

    return (dict(binding), set(taken)) if walk(0) else None


Resolve = Callable[[str], "tuple[CxnSpec | None, str | None]"]


def resolver(
    store: ConstructionStore,
    chat_id: str,
    user: str,
    *,
    successors: Mapping[str, str] | None = None,
    own: Mapping[str, CxnSpec] | None = None,
) -> Resolve:
    """Resolve an ``invoke``'s construction id to the construction that runs: the row itself when it is a taught root,
    live or staged by ``user`` in ``chat_id``; a row a confirmed replace removed resolves to its ``successor`` (the
    confirmed row of the same skeleton), followed along a chain of replaces. ``successors`` and ``own`` stand for a
    draft being taught (the rows it would replace, and its own specs). Returns (spec, None) or (None, the reason)."""
    successors = successors or {}
    own = own or {}

    def resolve(ref: str) -> tuple[CxnSpec | None, str | None]:
        gone = f"invokes {ref}, which is no longer live"
        seen: set[str] = set()
        at = ref
        while at not in seen:
            seen.add(at)
            if at in successors:
                at = successors[at]
                continue
            if at in own:
                return own[at], None
            row = store.get(at)
            if row is not None:
                usable = row.status == "live" or (row.provenance.get(TEACHER), row.provenance.get(CHAT)) == (
                    user,
                    chat_id,
                )
                if row.kind != "taught" or row.unit != "root" or not usable:
                    return None, gone
                return row, None
            after = store.successor(at)
            if after is None:
                return None, gone
            at = after
        return None, gone

    return resolve


def expand(
    meaning: Sequence[Sequence[Any]], resolve: Resolve, *, root: str | None = None
) -> tuple[list[list[Any]], list[str], list[str]]:
    """``meaning`` with every ``invoke`` replaced by the invoked construction's program (``resolve``): its procedure
    variable becomes the step, its slot variables the arguments, every other variable fresh; references inside it are
    expanded in turn, at most ``INVOKE_DEPTH`` deep. Returns the expanded meaning, the ids invoked (each expansion
    once), and problems: a construction no longer live, a slot count that differs, a reference that would invoke a
    construction from inside itself (``root`` is the construction ``meaning`` belongs to, when it is one), or one
    nested too deep. A failed reference is left out of the meaning."""
    out: list[list[Any]] = []
    used: list[str] = []
    problems: list[str] = []
    count = 0

    def walk(predicates: Sequence[Sequence[Any]], stack: list[str]) -> None:
        nonlocal count
        for predicate in predicates:
            if not predicate or predicate[0] != INVOKE_OP:
                out.append(list(predicate))
                continue
            if len(predicate) < 3 or not _is_var(predicate[1]):
                problems.append(f"invoke {list(predicate)!r} needs a step variable and a construction")
                continue
            step, ref, args = predicate[1], str(predicate[2]), list(predicate[3:])
            spec, why = resolve(ref)
            if spec is None:
                problems.append(why or f"invokes {ref}, which is no longer live")
                continue
            if spec.id in stack:
                problems.append(f"{spec.id} would invoke itself ({' -> '.join([*stack, spec.id])})")
                continue
            if len(stack) - (root is not None) >= INVOKE_DEPTH:
                problems.append(f"invokes {spec.id} nested deeper than {INVOKE_DEPTH} references")
                continue
            slots = [f"?{slot}" for slot in _slots(spec)]
            if len(slots) != len(args):
                problems.append(f"invokes {spec.id} with {len(args)} arguments for its {len(slots)} slots")
                continue
            used.append(spec.id)
            count += 1
            fresh = _INVOKE_FRESH.format(n=count)
            names = {PROCEDURE_VAR: step, **dict(zip(slots, args, strict=True))}
            body = [[p[0], *(names.get(a, fresh + a[1:]) if _is_var(a) else a for a in p[1:])] for p in spec.meaning]
            walk(body, [*stack, spec.id])

    walk(meaning, [root] if root is not None else [])
    return out, used, problems


__all__ = [
    "CHAT",
    "CONFIRMED_BY",
    "INVOKE_DEPTH",
    "INVOKE_OP",
    "DRAFT",
    "GRAMMAR",
    "GRAMMAR_HASH",
    "MODIFIABLE",
    "MODIFIES",
    "READS_STAGED",
    "REPLACEABLE",
    "REPLACED",
    "REPLACES",
    "SUCCESSOR",
    "TEACHER",
    "Draft",
    "NotTeacher",
    "TeachError",
    "TeachRequest",
    "admissible",
    "confirm",
    "distribute_everyone",
    "everyone_conflicts",
    "expand",
    "resolver",
    "teach",
]
