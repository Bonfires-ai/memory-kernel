"""The runtime grammar's contract: ``Runtime.plan(turn) -> TurnPlan`` and ``Runtime.resume(plan_id, results)``.

``plan`` reads a chat turn with the bonfire's runtime grammar (the base set, the store's live rows and the drafts the
speaker may use: ``store.load_store``) and decides what happens to it:

* **unread**: no root construction read the turn (or it read through a draft not usable by this speaker in this chat):
  the agent runtime runs the turn exactly as today (spec decision 12). The hand is empty.
* **repair**: a root read the turn but the program must not run or did not give an answer. The gate
  (``executable``) refuses a reading that is not complete (no construction read the whole turn), that the do-support
  guard refuses (``base.reading_guard``), whose program ``validate`` refuses, that carries a constraint no keyword
  applies (an op output nothing reads and that is not the answer: ``?other``, ``?noun``, ``?owner``, ``?anchor``), or
  that names a quantified noun phrase ("every team"). After execution: an executor problem (a refused turn role, the
  turn window without the referent, ``within`` without the turn's time, a failed op), a name that names nobody in the
  graph (any ``find-entity`` that found nothing: never an answer built from another entity), a verb that names no
  frame (``frames-of`` found nothing: never a polar "no"), or an empty answer (for
  every shape but a polar one, whose empty evidence is a real "no"); an empty result also carries a ``recall`` result
  (the fallback ``/context`` operation) in ``memory_results["recall"]``. On resume, any tool verdict but ``committed``
  (the repair keeps the program's licence and carries ``failure``).
* **needs_tools**: the program stopped at a tool step; ``tool_steps`` holds it (the kernel never runs a tool) and the
  agent runtime calls ``resume`` with its result. ``licence`` names every tool the program uses, with the state cells
  each declares (``ToolSpec.reads`` / ``writes``: surfaced, not enforced): ``[]`` for an executable plan that uses
  none, ``None`` for an unread plan or a repair whose program uses no tool (no licence: the turn falls back with
  today's tools); a repair of a program that names a tool keeps that program's licence.
* **answered**: the program ran to the end. A list-of-entities answer is narrowed by kind where its shape allows: a
  ``who`` answer keeps the Person entities and a ``what`` answer the others, when the answer holds both (an object
  root over a ditransitive frame returns every role filler: "what did Ana sell?" -> the thing, not the buyer).

Every plan, whatever its status, carries ``frames`` (``runtime.frames``): what read which words of the turn, from
the primary analysis and, when it is complete, the competing ones (``RuntimeEngine.comprehend_all``: one per other
root construction, ``ANALYSES`` in all by default, ``Runtime(analyses=)``; an analysis through a draft not usable
here is left out), and ``left``: the text the primary analysis left unconsumed, as written. ``Runtime(analyses=1)``
reads the primary analysis alone: no competing analyses and no bound role read on its own (``_slot_frames``). When
the competing analyses cannot be read (the engine refused or failed the n-best request), the primary's plan stands on
its own frames and ``reading["alternatives"]`` says why.
A partial reading (a frame part read "make" in "make onboard josh") is unread with its frames: the kernel never runs
it. A reply root ("yes", "no", either with a final "." or "!", or any run of them) is unread with its reply frame: it
answers the turn before, and nothing runs.

``plan(..., context=[Frame])`` names the open frames of the turn before (at most ``frames.MAX_CONTEXT``; the caller
sends only frames whose effect has not run): a turn that reads as a bare fragment (nothing complete, and no frame of its
own: no frame part, no root in any usable analysis) is read again inside each open frame's forms
(``frames.compositions``, the fragment without its final marks, which end the turn's sentence and would land mid-way
in a rendering: "JUJ-3" or "JUJ-3." against an open ``take`` frame is read as "take JUJ-3"; after every open role, a frame
whose one bound role is a one-slot root's, held back by the agent, is read with the fragment in place of that value:
"take the new one" then "JUJ-3" is "take JUJ-3"), and the first rendering that
reads completely through a root of that frame, through nothing unusable here, is the plan's reading: it runs as that
root's reading would (its spans are the rendered text's; ``turn.text`` stays the turn's), its frames of the open
frame's name carry ``composed_with`` (that name; any other analysis of the rendering does not), ``reading["composed"]``
is ``{"with": <name>, "text": <the rendering>}``, and ``reading["ms"]`` (the trace's ``comprehension_ms``) is the wall
ms of every read the plan took: the turn's own and each rendering's. A turn with a frame of its own ("make onboard
josh", "yes please take it") is never composed: binding stays conservative, the whole message never fills an open slot;
nor is a complete turn, nor one whose own reading was refused because a slot took its final mark ("take JUJ-3?": a
root of its own read it); a composition the engine fails leaves the turn's own reading, and ``reading["composed"]`` says
why.

A reference to a taught construction (``invoke``, stored by teaching) is expanded first into that construction's live
program (``teach.expand``: a replaced construction resolves to the row that replaced it); the expanded program is what
is validated and runs, its step ids index it, and every construction invoked records the turn's outcome with those
applied. A reference that cannot expand (the construction is no longer live, a cycle, nested deeper than
``INVOKE_DEPTH``) makes the turn a repair: ``invokes <id>, which is no longer live``. A program that held a reference
is tool-bearing (ruling R4): its repair's licence is that of what did expand, ``[]`` when none of it names a tool,
never ``None``.

A final mark belongs to the turn's sentence or to a root's literal, never to a slot: a reading of the turn in which a
slot reaches into its final marks (its trailing POS ``PUNCT`` tokens, a closing quote or bracket excepted) is never
used, and no competing analysis whose slot does reaches its frames. A turn with no final mark (its last token is not
POS ``PUNCT``) that draws no root is read again ONCE with " ?" appended, and that reading is used only when a root's
literal consumes the "?" ("onboarding is too slow because" never gives the reason "?"; "take" alone is unread with
its frame part open, never take(ticket="?")). A turn whose final marks hold no question mark ("." / "!" / "..." /
"!!"), refused so or drawing no complete root, is read again ONCE without them, never retried, and that reading is
used when it is complete and no slot of it takes a mark ("onboarding is too slow." reads as "onboarding is too slow"
does, "... because of the docs." gives the reason "docs", "yes!!" is the reply "yes"): its spans are that text's
(``reading["normalised"]`` shows it) and ``turn.text`` stays as written. A question mark the speaker wrote ("?", the
full-width one, the emoji, any whose Unicode name says so: ``_question_mark``) is never stripped, so a slot never takes
it either: questions are roots ending in "?". A mark inside closing quotes or brackets is the quoted text's ('take
"JUJ-3?"' reads with the ticket '"JUJ-3?"'). Each turn gets ONE
``StoreMemory`` (its queries are the trace's ``queries``) composed with the
lexicon ops. The applied constructions' standing records the turn's outcome once (``store.record_uses``) when the
turn ends: answered = ``ok``; a repair the room caused (a verdict ``precondition_failed``, ``declined`` or
``tool_error``) = ``world``, which moves ``uses_world`` only and never precedence; any other repair (a misreading, an
empty result, an unresolved name, a constraint or store-kind fail) = ``failed``. A turn waiting on tools records when
its last resume ends; a resume that raises ``ResumeError`` records nothing. A standing write that fails is a diagnostic,
never a failed turn. A plan that expires while tools are pending records nothing: the agent runtime dropping a plan is
no evidence about its constructions. The empty-result checks run before the tool branch: a name that names nobody, or
a verb that names no frame, is a repair with no tool step.

Plans live in memory per ``Runtime``: the most recent 256, each for 15 minutes after its last use. A resume never
re-runs a completed memory op: the plan keeps its execution's values and every tool result so far, and ``execute``
replays with them (``prior`` / ``resumed``). An unknown or expired ``plan_id`` is ``UnknownPlan`` (its message says
whether the plan expired, was evicted or was never made here); a resume that
names a step that is not pending, or a plan that waits on no tool, is ``ResumeError`` (nothing changes, and the plan's
clock is not reset). A step is exposed as ``<plan_id>:<i>:<occurrence>`` and a resume names it so (another plan's step is
not pending). ``retry(plan_id)`` re-issues the failed step of a plan that carries a ``failure`` (back to
``needs_tools``, the same ``step_id``; earlier committed results stay; the retry records its own outcome on the
plan's constructions); any other plan is ``ResumeError``.
``plan(..., committed=[{tool, args, value}])`` names effects already done: a tool step with the same tool and
canonical-JSON args is resolved with the value (trace ``reused``) and never issued, each entry once. A chat's turns are planned and resumed one at a time (an asyncio lock per ``chat_id``, held from
arrival until the plan is kept), so a turn's window holds every earlier turn of its chat.

The TURN WINDOW of a turn is the last ``window`` plans of its ``chat_id`` that drew a construction (unread turns are
not in it), latest first: per plan, the entities its answer named (none for a plan that did not answer); when the
chat's latest plan has expired or been evicted, the window is empty. The base
deixis fillers "it" and "that one" read it through the ``window`` keyword (``executor.WINDOW_RULES``): "what did
Milo publish?" ... "who gave it?". ``Runtime.window(chat_id)`` lists it (answers and entities).

``alternatives(turn)`` returns every distinct complete analysis of the turn (``RuntimeEngine.comprehend_every``,
under a count, a node cap and a deadline of its own), the first being ``plan``'s reading, each a plan walked with
``resume`` as ``plan``'s is and its path (the construction in each slot); its plans never record standing and are never
in the turn window.

``teach`` and ``confirm`` delegate to ``runtime.teach`` (with this runtime's registry). Every engine access (a
comprehension, a teach, a confirm) holds one lock: a teach's self-read loads its draft live for a moment, and no turn
may be read through it meanwhile. ``stop`` stops the engine only (plans, the window and resumes stay); the next engine
access starts it again, on the engine ``use_engine`` gave it (``mk serve`` moves an idle runtime to a free port).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import logging
import re
import threading
import time
import unicodedata
import uuid
import weakref
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any, Literal, NamedTuple

from memory_kernel.graph.g1_names import PERSON
from memory_kernel.graph.g1_nlp import parser_note
from memory_kernel.runtime.base import reading_guard
from memory_kernel.runtime.constructions import CxnSpec, spec_id
from memory_kernel.runtime.engine import Reading, RuntimeEngine, RuntimeEngineError
from memory_kernel.runtime.executor import VALUE_SPECS, Execution, SpanText, ToolStep, Turn, execute
from memory_kernel.runtime.frames import (
    compositions,
    context_frames,
    frame_name,
    frames_of,
    reply_of,
    surface_left,
)
from memory_kernel.runtime.lexicon import (
    LEXICON_SPECS,
    CompositeMemory,
    Lexicon,
    LexiconOps,
)
from memory_kernel.runtime.normalise import VERB_OPEN, Normalised, TokenSpans, normalise
from memory_kernel.runtime.ops_memory import MEMORY_SPECS, Reader, StoreMemory
from memory_kernel.runtime.policy import DEFAULT_POLICY, ConfirmationPolicy
from memory_kernel.runtime.program import ANSWER_PREDICATE, MeaningError, Program, Registry, ToolSpec, Var, validate
from memory_kernel.runtime.store import ConstructionStore, Outcome, load_store, loaded_frame_parts
from memory_kernel.runtime.teach import (
    INVOKE_OP,
    TEACHER,
    Draft,
    TeachRequest,
    admissible,
    distribute_everyone,
    expand,
    resolver,
)
from memory_kernel.runtime.teach import confirm as confirm_draft
from memory_kernel.runtime.teach import teach as teach_draft

Status = Literal["answered", "needs_tools", "unread", "repair"]
#: how a tool step ended, as the game reports it: ``committed`` ran, ``precondition_failed`` was refused by the world's
#: rules, ``tool_error`` broke, ``declined`` was turned down by the speaker or the game
Verdict = Literal["committed", "precondition_failed", "tool_error", "declined"]
VERDICTS: tuple[Verdict, ...] = ("committed", "precondition_failed", "tool_error", "declined")
#: the verdicts a repair the room caused carries: the room lacked something or refused, so the repair is a ``world`` use
WORLD_VERDICTS: frozenset[str] = frozenset({"precondition_failed", "declined", "tool_error"})


def _outcome_of(plan: TurnPlan) -> Outcome:
    """A turn's outcome for its applied constructions: answered is ``ok``; a repair with a verdict failure the room
    caused is ``world``; every other repair (a misreading, an empty result, an unresolved name, a constraint or
    store-kind fail) is ``failed``."""
    if plan.status == "answered":
        return "ok"
    if plan.failure is not None and plan.failure.get("kind") in WORLD_VERDICTS:
        return "world"
    return "failed"


#: plans kept per runtime, and for how long after their last use (seconds)
log = logging.getLogger("memory_kernel.runtime.api")
MAX_PLANS = 256
PLAN_TTL = 15 * 60
#: plans of the chat the turn window holds by default
DEFAULT_WINDOW = 5
#: the POS of a final mark ("?", "."): a turn whose last token has another POS gets one retry with " ?" appended
FINAL_MARK_POS = "PUNCT"
#: the spaCy morph ``PunctType`` of a mark that is half of a pair (a quote, a bracket): a closing one is not a final
#: mark, it stays with its opener
PAIRED_MARK_TYPES = frozenset({"Quot", "Brck"})
#: appended to a turn with no final mark that drew no root (used only when a root's literal consumes it)
RETRY_MARK = " ?"
#: answer shapes (labels of the contract, as the base rows write them) whose empty answer is still an answer: a polar
#: question's empty evidence is "no"
EMPTY_IS_AN_ANSWER = frozenset({"polar"})
#: answer shape -> (store kind label, whether the answer keeps that kind or the others): who-vs-what at answer time
SHAPE_KIND: Mapping[str, tuple[str, bool]] = {"who": (PERSON, True), "what": (PERSON, False)}
#: how many expired or evicted plan ids a runtime remembers, to tell them from ids it never made
FORGOTTEN_IDS = 4096
#: FCG renames a construction's variables per application ("?noun" comes back as "?noun-4")
_APPLICATION_SUFFIX = re.compile(r"-\d+$")
#: how many analyses of a turn its frames report (the primary and the competing ones): ``comprehend_all``'s ``n``;
#: the default of ``Runtime(analyses=)``
ANALYSES = 3
#: the frame kinds a bound role can afford when read on its own (``Runtime._slot_frames``)
INNER_KINDS = frozenset({"tool", "phrase", "editor"})
#: ``Runtime.alternatives``: how many alternatives one call returns at most (counted after distinctness, ``through``
#: and ``maximal``), how many analyses its search takes at most before those filters, and the search's node cap and
#: deadline (``RuntimeEngine.comprehend_every``)
ALTERNATIVES = 16
ALTERNATIVES_RAW = 256
ALTERNATIVES_NODE_CAP = 5000
ALTERNATIVES_DEADLINE_MS = 10_000
#: the generated clause filler's id (``store.CLAUSE_FILLER``), as a path names it
GENERATED = "clause"
#: a slot unit's FCG name: ``slot-<slot name>-<application number>`` (``constructions``: the subunit ``?slot-<s>``)
_SLOT_UNIT = re.compile(r"slot-(.+)-\d+")


class _Reads(NamedTuple):
    """What ``Runtime._read`` made of one text."""

    norm: Normalised
    #: the primary reading (None when the engine returned none)
    reading: Reading | None
    #: the analyses frames come from: the primary first, then the competing ones
    readings: list[Reading]
    ms: float
    retried: bool
    #: why the competing analyses were not read (None when they were, or were not asked for)
    alternatives: str | None
    #: the turn's own reading was refused (a slot took a final mark) and nothing replaced it: a root of its own read
    #: the turn, so it is never composed into an open frame
    refused: bool = False


class UnknownPlan(LookupError):
    """No plan with this id: never made by this runtime, or expired (15 minutes after its last use), or evicted (past
    the most recent 256). The message says which, while the runtime still remembers the id (``FORGOTTEN_IDS``)."""


class ResumeError(ValueError):
    """A resume that cannot apply: the plan waits on no tool, or a result names a step that is not pending."""


def default_registry(tools: Iterable[ToolSpec] = ()) -> Registry:
    """The keyword set (value, memory and lexicon ops) plus the catalogue's tools."""
    return Registry({**VALUE_SPECS, **MEMORY_SPECS, **LEXICON_SPECS}).with_tools(list(tools))


@dataclass(frozen=True)
class TurnPlan:
    """What the runtime grammar made of one turn (JSON-ready: ``to_json``)."""

    plan_id: str
    bonfire_id: str
    #: the turn as given (text, speaker, addressee, participants, chat_id, message_id, at)
    turn: dict[str, Any]
    status: Status
    #: the program's ops as meaning predicates (``Program.to_json``); [] when nothing read the turn
    program: list[list[Any]]
    answer_shape: str | None
    #: the answer's value (narrowed by kind where the shape allows); None unless answered
    answer: Any
    #: every value the program computed so far (variable -> value), plus ``recall`` on an empty-result repair
    memory_results: dict[str, Any]
    #: the pending tool step(s): step_id, tool, args, side_effect, needs_confirmation, actor (the turn's speaker)
    tool_steps: list[dict[str, Any]]
    #: the tools the program uses: tool, side_effect, needs_confirmation, reads, writes; [] for an executable plan
    #: that uses none; None (JSON null) for an unread plan or a repair whose program uses no tool: the turn falls back with
    #: today's tools (decision 12)
    licence: list[dict[str, Any]] | None
    #: ``steps`` (the executor's trace), ``queries`` (op, cypher, params of every graph read), timings
    trace: dict[str, Any]
    diagnostics: list[str]
    #: the reading: applied construction names, comprehension ms, complete, the normalised text, the root, retried
    reading: dict[str, Any]
    #: the drawn constructions: id, kind, unit, maker, standing, the keyword rules text of their meaning
    hand: list[dict[str, Any]] = field(default_factory=list)
    #: set when a resume verdict is not ``committed``: step_id, tool, kind (the verdict), detail; the plan is a repair
    #: that keeps the program's licence
    failure: dict[str, Any] | None = None
    #: what read which words, on every path (``runtime.frames``): one Frame per frame part and per root of each of up
    #: to the runtime's ``analyses`` (``ANALYSES``); [] when nothing read the turn
    frames: list[dict[str, Any]] = field(default_factory=list)
    #: the turn's text no construction consumed in the primary analysis, as written ([] once it read completely)
    left: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class Alternative:
    """One analysis of a turn (``Runtime.alternatives``)."""

    #: walked with ``Runtime.resume`` exactly as a plan from ``Runtime.plan``; it never records standing and is never
    #: in the chat's turn window
    plan: TurnPlan
    #: (slot address, construction spec id): ``("", root id)`` first, then every filled slot, leftmost first, addressed
    #: by slot names from the root (``"s"``, ``"s/h"``, ``"s/d/h"``); a slot left to the generated filler names
    #: ``GENERATED``
    path: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Alternatives:
    found: tuple[Alternative, ...]
    #: a bound ended the search (its node cap, its deadline, or ``ALTERNATIVES_RAW`` analyses: there may be analyses
    #: it did not reach), or more than ``limit`` alternatives were left after the filters (those past it are dropped)
    cut_short: bool


def jsonable(value: Any) -> Any:
    """``value`` with tuples and sets as lists, mapping keys as strings and anything else unknown as its ``str``."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if is_dataclass(value) and not isinstance(value, type):  # a Referent
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    return str(value)


def _plan_trace(comprehension_ms: Any) -> dict[str, Any]:
    """A fresh plan's trace; ``parser`` is present only when turn parsing fell back to the small model."""
    trace: dict[str, Any] = {"steps": [], "queries": [], "comprehension_ms": comprehension_ms, "execution_ms": None}
    if (note := parser_note("turn")) is not None:
        trace["parser"] = note
    return trace


def unread_plan(
    bonfire_id: str,
    turn: Turn,
    diagnostics: Sequence[str],
    info: Mapping[str, Any] | None = None,
    *,
    frames: Sequence[Mapping[str, Any]] = (),
    left: Sequence[str] = (),
) -> TurnPlan:
    """An ``unread`` plan: nothing read the turn completely (or the runtime could not read it now), so no program runs
    and the agent runtime takes the turn (no licence: ``licence`` None, JSON null). ``info`` is the reading's trace
    (none by default); ``frames`` and ``left`` are what did read, and what was left (none by default)."""
    reading = (
        dict(info)
        if info is not None
        else {
            "applied": [],
            "ms": None,
            "complete": False,
            "normalised": None,
            "root": None,
            "retried": False,
        }
    )
    return TurnPlan(
        plan_id=uuid.uuid4().hex,
        bonfire_id=bonfire_id,
        turn=_turn_json(turn),
        status="unread",
        program=[],
        answer_shape=None,
        answer=None,
        memory_results={},
        tool_steps=[],
        licence=None,
        trace=_plan_trace(reading.get("ms")),
        diagnostics=list(diagnostics),
        reading=reading,
        hand=[],
        frames=[dict(f) for f in frames],
        left=list(left),
    )


def _snapshot(record: _Plan) -> dict[str, Any]:
    """Every field of ``record`` as it stands, its mutable containers copied: what a failed retry puts back."""
    return {
        f.name: copy.copy(value) if isinstance(value, (dict, list)) else value
        for f in fields(record)
        for value in [getattr(record, f.name)]
    }


@dataclass
class _Plan:
    """A plan and what its resume needs: the turn (with its window), the spans, the program, the values so far."""

    plan: TurnPlan
    seq: int
    touched: float
    turn: Turn
    drew: bool
    spans: SpanText | None = None
    program: Program | None = None
    values: dict[str, Any] = field(default_factory=dict)
    resumed: dict[str, Any] = field(default_factory=dict)
    #: the effects committed before this plan (``Runtime.plan(committed=...)``): ``{tool, args, value}``
    committed: tuple[dict[str, Any], ...] = ()
    pending: tuple[ToolStep, ...] = ()
    uses: frozenset[str] = frozenset()  # the spec ids the turn applied: each records the turn's outcome once
    #: the reading held a reference (``invoke``): tool-bearing for the licence rule, whatever expanded (ruling R4)
    invokes: bool = False
    recorded: bool = False
    #: False for an alternative's plan (``Runtime.alternatives``): its outcome is never recorded, on any resume or retry
    standing: bool = True
    busy: bool = False
    referent: tuple[dict[str, Any], ...] = ()
    entities: list[dict[str, Any]] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _ChatLock:
    """One chat's turn lock and how many of its turns hold or await it (``Runtime._chat_lock``)."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turns: int = 0


class Runtime:
    """One bonfire's runtime grammar: ``plan`` / ``resume`` / ``teach`` / ``confirm``.

    ``engine`` is the runtime engine (started here on first use when it is not running); ``memory`` is the graph
    ``Reader`` each turn's ``StoreMemory`` queries, scoped by ``group_id`` (default: the bonfire id); ``registry`` is
    the keyword set plus the catalogue's tools (default: ``default_registry()``). ``recall`` is the fallback
    ``/context`` callable an empty-result repair runs (given the turn's ``speaker`` when it takes that keyword);
    ``lexicon`` is the bonfire's rolesets (``load_rolesets``) over the shipped question map. ``record_standing`` False
    never writes standing (``mk plan``); ``store`` is the bonfire's ``ConstructionStore`` (``mk plan`` passes a
    read-only one). ``cards`` are the catalogue's generated cards (``parse_catalogue``): live beside the store's rows,
    loaded into the engine with them, never persisted. ``analyses`` is how many analyses a plan's frames report
    (``ANALYSES``, the primary and the competing ones); 1 reads the primary analysis alone: no ``comprehend_all`` and
    no read of the bound roles on their own (``_slot_frames``), the plan's frames are the primary's.
    ``raise_engine_errors`` True: ``plan`` raises the ``RuntimeEngineError`` that would leave the turn unread (the
    grammar's load or the turn's read failed), as it was raised, instead of planning the turn as unread with a
    ``runtime engine:`` diagnostic, so its type says whether a live engine refused or the engine faulted (the install
    door's self-test). The reads a plan does without (the competing analyses, the bound roles on their own, a
    composition with the turn before) are done without as before: a caller that must see every engine error reads one
    analysis, with no context."""

    def __init__(
        self,
        bonfire_id: str,
        engine: RuntimeEngine,
        memory: Reader,
        registry: Registry | None = None,
        *,
        store: ConstructionStore,
        lexicon: Lexicon,
        group_id: str | None = None,
        recall: Callable[[str], Awaitable[dict]] | None = None,
        max_plans: int = MAX_PLANS,
        ttl: float = PLAN_TTL,
        clock: Callable[[], float] = time.monotonic,
        record_standing: bool = True,
        policy: ConfirmationPolicy = DEFAULT_POLICY,
        cards: Sequence[CxnSpec] = (),
        analyses: int = ANALYSES,
        raise_engine_errors: bool = False,
    ) -> None:
        if isinstance(analyses, bool) or not isinstance(analyses, int) or analyses < 1:
            raise ValueError(f"analyses must be a positive integer, got {analyses!r}")
        self.analyses = analyses
        self.raise_engine_errors = raise_engine_errors
        self.engine = engine
        self.reader = memory
        self.registry = registry if registry is not None else default_registry()
        self.bonfire_id = bonfire_id
        self.group_id = group_id or self.bonfire_id
        self.recall = recall
        self._recall_speaker = _takes_speaker(recall)
        self.record_standing = record_standing
        self.lexicon = lexicon
        self.store = store
        self.policy = policy
        #: the catalogue's generated cards: loaded with the store's rows, never persisted
        self.cards = tuple(cards)
        self.store.attach_cards(self.cards)
        self.max_plans = max_plans
        self.ttl = ttl
        self.clock = clock
        self._specs = {**self.registry.specs, **VALUE_SPECS}
        self._plans: OrderedDict[str, _Plan] = OrderedDict()
        self._forgotten: OrderedDict[str, str] = OrderedDict()  # plan id -> why it left ("expired" / "evicted")
        self._plans_lock = threading.Lock()
        self._grammar = threading.RLock()  # every engine access: comprehend, teach, confirm, (re)load
        self._started = False
        self._seq = 0
        self._latest: dict[str, int] = {}  # chat_id -> seq of its latest plan that drew a construction
        #: per event loop, one lock per chat_id (created lazily): a chat's turns are planned and resumed one at a time
        self._chat_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _ChatLock]] = (
            weakref.WeakKeyDictionary()
        )

    # -- lifecycle ----------------------------------------------------------------------------------------------------

    def start(self) -> None:
        """Start the engine when it is not running and load the bonfire's grammar (``load_store``). Idempotent. A load
        that fails stops the engine again (by pid)."""
        with self._grammar:
            if self._started:
                return
            if self.engine.pid is None:
                self.engine.start()
            try:
                load_store(self.store, self.engine)
            except BaseException:
                self.engine.stop()
                raise
            self._started = True

    def stop(self) -> None:
        """Stop the engine (by pid) and forget that the grammar was loaded: the next engine access starts it again
        (``start``). Plans, the turn window and resumes are untouched (a resume never uses the engine)."""
        with self._grammar:
            self.engine.stop()
            self._started = False

    def use_engine(self, engine: RuntimeEngine) -> None:
        """Run on ``engine`` from the next ``start`` on (``mk serve`` hands a stopped runtime a free port). Only while
        stopped: ``RuntimeError`` otherwise."""
        with self._grammar:
            if self._started or self.engine.pid is not None:
                raise RuntimeError("use_engine needs a stopped runtime (stop() first)")
            self.engine = engine

    async def _ensure_started(self) -> None:
        if not self._started:
            await asyncio.to_thread(self.start)

    # -- teaching (runtime.teach) --------------------------------------------------------------------------------------

    async def teach(self, req: TeachRequest) -> Draft:
        """``runtime.teach.teach`` with this runtime's store, engine and registry (under the engine lock)."""
        await self._ensure_started()
        return await asyncio.to_thread(self._locked, teach_draft, self.store, self.engine, req, registry=self.registry)

    async def confirm(self, draft_id: str, by: str) -> list[CxnSpec]:
        """``runtime.teach.confirm`` with this runtime's store and engine (under the engine lock)."""
        await self._ensure_started()
        return await asyncio.to_thread(self._locked, confirm_draft, self.store, self.engine, draft_id, by)

    def _locked(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._grammar:
            return fn(*args, **kwargs)

    # -- plan ---------------------------------------------------------------------------------------------------------

    async def plan(
        self,
        turn: Turn,
        *,
        window: int = DEFAULT_WINDOW,
        committed: Sequence[Mapping[str, Any]] = (),
        context: Sequence[Mapping[str, Any]] = (),
    ) -> TurnPlan:
        """Read ``turn`` and run what may run (see the module docstring); ``window`` = how many of the chat's plans
        the turn window holds; ``committed`` = effects already done in the world, ``{tool, args, value}`` each: a
        tool step with the same tool and (canonical JSON) arguments is resolved with that value and never issued;
        ``context`` = the open frames of the turn before (Frame JSON, ``frames.context_frames``): a turn that reads
        as a bare fragment (nothing complete, no frame of its own) composes with them."""
        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise ValueError(f"window must be a non-negative integer, got {window!r}")
        done = committed_entries(committed)
        try:
            open_frames = context_frames(context)
        except ValueError as exc:
            raise ValueError(f"context {exc}") from exc
        async with self._chat_lock(turn.chat_id):  # one turn of a chat at a time: the window sees every earlier one
            return await self._plan(turn, window, done, open_frames)

    async def _plan(
        self,
        turn: Turn,
        window: int,
        committed: tuple[dict[str, Any], ...] = (),
        context: Sequence[Mapping[str, Any]] = (),
    ) -> TurnPlan:
        plan_id = uuid.uuid4().hex
        turn = replace(turn, window=())  # the window is the runtime's, never the caller's
        try:
            await self._ensure_started()
            own, composed, composition, spent = await asyncio.to_thread(self._comprehend, turn, context)
        except RuntimeEngineError as exc:
            if self.raise_engine_errors:
                raise
            info = {"applied": [], "ms": None, "complete": False, "normalised": None, "root": None, "retried": False}
            # nothing read the turn, so nothing consumed it: the whole turn is left, as on every other unread path
            left = [turn.text.strip()] if turn.text.strip() else []
            unread = self._unread(plan_id, turn, info, [f"runtime engine: {exc}"], left=left)
            return self._keep(_Plan(unread, 0, 0, turn, False))

        # a composed reading is the plan's reading: its spans are the rendered text's (the gate, the program and a
        # resume cut from them); the turn had no frame of its own (``_comprehend`` composes only a bare fragment)
        norm, reading, readings, ms, retried, alternatives, _ = composed if composed is not None else own
        specs = self._all_specs()
        root = _root(reading, specs)
        # what read which words, on every path: an analysis through a draft not usable here affords nothing
        frames = frames_of(self._usable(readings, turn), norm, {**specs, **self._frame_parts()}, self.registry)
        if composed is not None and composition is not None:
            frames = _composed_frames(frames, composition["with"])
        if self.analyses > 1:  # one analysis: the primary's frames alone, no further read
            frames = [*frames, *await asyncio.to_thread(self._slot_frames, frames, norm, turn)]
        left = surface_left(reading, norm)
        info = {
            "applied": list(reading.applied) if reading is not None else [],
            # a plan that read renderings too costs every read: the turn's own and each rendering's (wall ms)
            "ms": spent if spent is not None else reading.ms if reading is not None else ms,
            "complete": bool(reading is not None and reading.complete),
            "normalised": norm.text,
            "root": root.id if root is not None else None,
            "retried": retried,
        }
        if alternatives is not None:  # the competing analyses could not be read: the primary's frames stand alone
            info["alternatives"] = alternatives
        if composition is not None:  # read inside an open frame of the turn before (or why that failed)
            info["composed"] = composition
        return await self._planned(plan_id, turn, norm, reading, root, specs, info, frames, left, window, committed)

    async def _planned(
        self,
        plan_id: str,
        turn: Turn,
        norm: Normalised,
        reading: Reading | None,
        root: CxnSpec | None,
        specs: dict[str, CxnSpec],
        info: dict[str, Any],
        frames: list[dict[str, Any]],
        left: list[str],
        window: int,
        committed: tuple[dict[str, Any], ...],
        *,
        alternative: bool = False,
    ) -> TurnPlan:
        """The plan of a turn read as ``reading`` (through ``root``, one of ``specs``: the constructions as they were
        when the turn was read, the one snapshot its hand and its gate see) on ``norm``: unread, a repair, or its
        program run as far as it may, kept for ``resume``. ``alternative`` (``Runtime.alternatives``): the plan never
        records standing and never draws into the chat's turn window."""
        if reading is None or root is None:
            unread = self._unread(plan_id, turn, info, ["no construction read the turn"], frames, left)
            return self._keep(_Plan(unread, 0, 0, turn, False))
        if not admissible(reading, self.store, turn.chat_id, turn.speaker or ""):
            diagnostic = "the turn reads through a draft not usable here (someone else's, or another chat's)"
            return self._keep(_Plan(self._unread(plan_id, turn, info, [diagnostic], frames, left), 0, 0, turn, False))
        if reply_of(root) is not None:  # a bare reply answers the turn before: its frame says so, nothing runs
            diagnostic = f"the turn is a reply ({reply_of(root)}): it answers the turn before"
            return self._keep(_Plan(self._unread(plan_id, turn, info, [diagnostic], frames, left), 0, 0, turn, False))

        hand = _hand(reading.applied, specs, self._specs)
        invokes = any(p and p[0] == INVOKE_OP for p in reading.meaning)
        # a reference (``invoke``) runs the invoked construction's live program: the expanded program is what is
        # validated and runs, and every construction invoked records the turn's outcome too
        meaning, invoked, invoke_problems = expand(
            reading.meaning, resolver(self.store, turn.chat_id, turn.speaker or "")
        )
        # a live "everyone" distributes over the turn's procedure as a taught one does (unless it fills a many param;
        # one that fills both kinds is a repair: say it in two clauses)
        distributed, conflicts = distribute_everyone(meaning, self._specs)
        reading = replace(reading, meaning=distributed)
        uses = frozenset(spec_id(name) for name in reading.applied) | frozenset(invoked)
        program, problems = self._gate(reading, norm, specs, root)
        if invoke_problems:  # the gate's program (what did expand) keeps its licence; the repair says what dangled
            problems = invoke_problems
        problems = [*problems, *conflicts]
        turn = replace(turn, window=self._referents(turn.chat_id, window))
        base = TurnPlan(
            plan_id=plan_id,
            bonfire_id=self.bonfire_id,
            turn=_turn_json(turn),
            status="repair",
            program=program.to_json() if program is not None else [],
            answer_shape=root.answer_shape,
            answer=None,
            memory_results={},
            tool_steps=[],
            licence=_repair_licence(program, self.registry, self.policy, turn.speaker, invokes=invokes),
            trace=_plan_trace(info["ms"]),
            diagnostics=problems,
            reading=info,
            hand=hand,
            frames=frames,
            left=left,
        )
        record = _Plan(
            base,
            0,
            0,
            turn,
            not alternative,
            spans=norm.spans,
            program=program,
            committed=committed,
            uses=uses,
            invokes=invokes,
            standing=not alternative,
        )
        if problems or program is None:
            await self._end(record)
            return self._keep(record)
        memory, store_memory = self._memory(turn)
        started = time.perf_counter()
        ex = await execute(
            program,
            self.registry,
            memory,
            turn=turn,
            spans=norm.spans,
            committed=_CommittedSteps(committed),
            policy=self.policy,
            id_prefix=f"{base.plan_id}:",
        )
        await self._outcome(record, ex, memory, store_memory, (time.perf_counter() - started) * 1000)
        return self._keep(record)

    def _comprehend(
        self, turn: Turn, context: Sequence[Mapping[str, Any]]
    ) -> tuple[_Reads, _Reads | None, dict[str, Any] | None, float | None]:
        """Read the turn (``_read``). A turn that reads as a bare fragment (nothing complete, and no frame of its own:
        ``frames_of`` its usable analyses is empty), given ``context``, is read again as each text
        ``frames.compositions`` renders (no retry mark), in order, until one reads completely, through a root of the
        open frame it was rendered from and nothing unusable here; a turn with a frame of its own is never put into an
        open frame's slot (C11: the fragment fills the slot, never a message with its own verb), nor one whose own
        reading was refused for a slot holding its final mark (``_Reads.refused``: a root read it). Returns the turn's
        own reads, the composed reads (None when nothing composed), what goes into ``reading["composed"]``:
        ``{"with", "text"}``, or ``{"error"}`` when the engine failed a composition (the turn's own reading then
        stands), or None; and the wall ms of every read when a rendering was read (the turn's own, each rendering's,
        the failed ones too), else None."""
        with self._grammar:
            own = self._read(turn.text)
            if not context or own.refused or (own.reading is not None and own.reading.complete):
                return own, None, None, None
            specs = self._all_specs()
            if frames_of(self._usable(own.readings, turn), own.norm, {**specs, **self._frame_parts()}, self.registry):
                return own, None, None, None
            spent, rendered = own.ms, False
            # the fragment without the turn's final mark: the mark ends the turn's sentence, and in a rendering it
            # would land mid-sentence ("JUJ-3." against give(person=Milo) is "give JUJ-3 to Milo", ticket "JUJ-3")
            fragment = _without_final_mark(own.norm)
            for name, text in compositions(context, self.store.live(), fragment, self.registry):
                started, rendered = time.perf_counter(), True
                try:
                    read = self._read(text, retry_mark=False)
                except RuntimeEngineError as exc:
                    spent += (time.perf_counter() - started) * 1000
                    return own, None, {"error": f"runtime engine: {exc}"}, spent
                spent += read.ms
                root = _root(read.reading, specs)
                if read.reading is None or not read.reading.complete or root is None or frame_name(root) != name:
                    continue
                if admissible(read.reading, self.store, turn.chat_id, turn.speaker or ""):
                    return own, read, {"with": name, "text": text}, spent
            return own, None, None, spent if rendered else None

    def _slot_frames(self, frames: Sequence[Mapping[str, Any]], norm: Normalised, turn: Turn) -> list[dict[str, Any]]:
        """The frames read inside the bound roles of complete frames: a slot's filler can swallow another verb's clause
        ("search about the monkey situation and write note about it": search's query holds ⟪write⟫), so a bound role
        is read on its own through the same grammar and the tool, phrase and editor frames found there follow the
        frames given (C5: an affordance is not an action; the model sees both). One level deep, and only when the
        turn holds two marked verbs or more (a role of a one-verb turn holds none). An engine failure keeps what was
        found."""
        if norm.text.count(VERB_OPEN) < 2:
            return []
        specs = {**self._all_specs(), **self._frame_parts()}
        seen = {json.dumps([f["name"], f["kind"], f["roles"]], sort_keys=True) for f in frames}
        found: list[dict[str, Any]] = []
        for frame in frames:
            if not frame.get("complete"):
                continue
            for value in (frame.get("roles") or {}).values():
                if not isinstance(value, str) or not value.strip():
                    continue
                try:
                    read = self._read(value, retry_mark=False)
                except RuntimeEngineError:
                    return found
                if VERB_OPEN not in read.norm.text:
                    continue
                for inner in frames_of(self._usable(read.readings, turn), read.norm, specs, self.registry):
                    key = json.dumps([inner["name"], inner["kind"], inner["roles"]], sort_keys=True)
                    if inner["kind"] in INNER_KINDS and key not in seen:
                        seen.add(key)
                        found.append({**inner, "analysis": frame.get("analysis", 0), "composed_with": None})
        return found

    def _usable(self, readings: Sequence[Reading], turn: Turn) -> list[Reading]:
        """The analyses that afford something here: an analysis through a draft not usable here affords nothing."""
        return [r for r in readings if admissible(r, self.store, turn.chat_id, turn.speaker or "")]

    def _read(self, text: str, *, retry_mark: bool = True, analyses: int | None = None) -> _Reads:
        """Normalise and comprehend ``text``; with ``retry_mark`` (the turn's own read) the reading is the turn's
        (``_turn_reading``: a final mark is the sentence's or a root literal's, never a slot's). Returns the text read,
        the primary reading, the analyses its frames come from (the primary first; a complete primary adds the
        competing analyses ``comprehend_all`` finds on the same text, one per other root, up to ``analyses`` in all,
        by default the runtime's, on the turn's own read none whose slot takes a final mark; a partial primary has
        none, nor does a read of one analysis, which never asks for them), the wall ms (every read included), whether
        the reading used came from the ``RETRY_MARK`` retry, and why the competing analyses were not read (None when
        they were, or were not asked for). The n-best never costs the primary its plan: when ``comprehend_all``
        fails, the primary reading stands alone."""
        n = self.analyses if analyses is None else analyses
        with self._grammar:
            started = time.perf_counter()
            norm = normalise(text)
            reading = self.engine.comprehend(norm.text)
            retried = refused = False
            specs = self._all_specs()
            if retry_mark:
                norm, reading, retried, refused = self._turn_reading(text, norm, reading, specs)
            readings = [reading] if reading is not None else []
            unread: str | None = None
            if reading is not None and reading.complete and n > 1:
                primary = _root(reading, specs)
                try:
                    others = self.engine.comprehend_all(norm.text, n)
                except RuntimeEngineError as exc:
                    others, unread = [], f"runtime engine: {exc}"
                if retry_mark:  # the turn's own read: no analysis whose slot takes a final mark
                    marks = _final_marks(norm)
                    others = [r for r in others if not _takes_mark(r, norm, marks)]
                readings += [r for r in others if _root(r, specs) is not primary][: n - 1]
            ms = (time.perf_counter() - started) * 1000
            return _Reads(norm, reading, readings, ms, retried, unread, refused)

    def _turn_reading(
        self, text: str, norm: Normalised, reading: Reading | None, specs: Mapping[str, CxnSpec]
    ) -> tuple[Normalised, Reading | None, bool, bool]:
        """The turn's own reading of ``text``, from its as-written ``norm`` and ``reading``: the text read, the
        reading (None when none is accepted), whether it came from the retry, and whether the as-written reading was
        refused with nothing to replace it. A final mark belongs to the sentence or to a root's literal, never to a
        slot:

        * a reading in which a slot reaches into the turn's final marks (``_takes_mark``) is never accepted;
        * a turn with no final mark (its last token is not POS ``FINAL_MARK_POS``) that draws no root is read once
          more with ``RETRY_MARK`` appended; that reading is used when it draws a root and no slot of it takes the
          appended "?" (a root's literal consumes it), else the as-written one stands;
        * a turn whose final marks hold no question mark (``_asks``), whose as-written reading is refused or draws no
          complete root, is read once more as ``_without_final_mark`` gives it (never retried: the speaker wrote a
          mark, and not a question mark); that reading is used when it is complete, draws a root and itself takes no
          final mark (its spans are its text's, ``turn.text`` stays as written). Otherwise the as-written reading
          stands, or none when it was refused (the turn is unread with no frames, and never composed: a root of its own
          read it).

        A question mark the speaker wrote ("?", the full-width one, the emoji: ``_question_mark``) is never stripped:
        questions are roots that end in "?"."""
        if not _final_mark(norm):  # no final marks: no slot can take one, so the as-written reading is never refused
            if _root(reading, specs) is None:
                again = normalise(text + RETRY_MARK)
                retry = self.engine.comprehend(again.text)
                if _root(retry, specs) is not None and not _takes_mark(retry, again, _final_marks(again)):
                    return again, retry, True, False
            return norm, reading, False, False
        marks = _final_marks(norm)
        refused = _takes_mark(reading, norm, marks)
        if refused or not _complete_root(reading, specs):
            stripped = _without_final_mark(norm) if marks and not _asks(norm, marks) else ""
            if stripped:
                bare = normalise(stripped)
                again = self.engine.comprehend(bare.text)
                if _complete_root(again, specs) and not _takes_mark(again, bare, _final_marks(bare)):
                    return bare, again, False, False
        return norm, None if refused else reading, False, refused

    def _gate(
        self, reading: Reading, norm: Normalised, specs: Mapping[str, CxnSpec], root: CxnSpec
    ) -> tuple[Program | None, list[str]]:
        """``executable``: the program, and the reasons it must not run ([] when it may)."""
        return executable(reading, norm.spans, [*specs.values()], self.registry, root=root)

    def _memory(self, turn: Turn) -> tuple[CompositeMemory, StoreMemory]:
        """ONE StoreMemory for the turn (its queries are the trace's), composed with the lexicon ops. A ``recall`` that
        takes a ``speaker`` keyword gets the turn's speaker (first-person words resolve to them)."""
        recall = self.recall
        if recall is not None and self._recall_speaker:
            base, speaker = recall, turn.speaker

            async def with_speaker(text: str) -> dict:
                return await base(text, speaker=speaker)  # type: ignore[call-arg]

            recall = with_speaker
        store_memory = StoreMemory(self.reader, self.group_id, recall=recall)
        memory = CompositeMemory([(MEMORY_SPECS, store_memory), (LEXICON_SPECS, LexiconOps(self.lexicon))])
        return memory, store_memory

    async def _outcome(
        self, record: _Plan, ex: Execution, memory: CompositeMemory, store_memory: StoreMemory, ms: float
    ) -> None:
        """Set the plan's status, answer, tool steps and trace from an execution (plan or resume)."""
        program = record.program
        if program is None:  # only executed plans get here
            raise RuntimeError("an executed plan has a program")
        record.values = dict(ex.values)
        record.queries += store_memory.queries
        plan = record.plan
        shape = plan.answer_shape
        diagnostics: list[str] = []
        results: dict[str, Any] = dict(ex.values)
        steps = [asdict(step) for step in ex.trace]
        answer: Any = None
        status: Status
        # a step is exposed as ``<plan_id>:<local id>`` (the executor's ``<i>:<occurrence>``): ids never cross plans
        pending = tuple(replace(step, step_id=f"{plan.plan_id}:{step.step_id}") for step in ex.pending)
        unresolved = _unresolved(ex)  # checked BEFORE the tool branch: a nobody never reaches a tool step
        if ex.problems:
            status, diagnostics = "repair", list(ex.problems)
        elif unresolved:
            status, diagnostics = "repair", unresolved
        elif ex.pending:
            status = "needs_tools"
        else:
            answer = ex.values.get(program.answer.name) if program.answer is not None else None
            if program.answer is not None and _empty(answer) and shape not in EMPTY_IS_AN_ANSWER:
                status, diagnostics = "repair", ["the program found nothing for the turn"]
            else:
                status = "answered"
        if status == "repair" and not ex.problems:  # an empty result: the fallback recall rides along
            answer = None
            recalled = await _recall(memory, plan.turn["text"], steps)
            if isinstance(recalled, str):
                diagnostics.append(recalled)
            else:
                results["recall"] = recalled
        elif status == "answered":
            answer, narrowed = _narrow(answer, shape)
            if narrowed is not None:
                steps.append({"op": "narrow", "args": {"shape": shape}, "out": narrowed, "ms": 0.0, "ok": True})
        record.plan = replace(
            plan,
            status=status,
            answer=jsonable(answer),
            memory_results=jsonable(results),
            tool_steps=[jsonable(asdict(step)) for step in pending] if status == "needs_tools" else [],
            licence=_licence(program, self.registry, self.policy, record.turn.speaker)
            if status in ("answered", "needs_tools")
            else _repair_licence(program, self.registry, self.policy, record.turn.speaker, invokes=record.invokes),
            trace={**plan.trace, "steps": jsonable(steps), "queries": jsonable(record.queries), "execution_ms": ms},
            diagnostics=diagnostics,
        )
        record.pending = pending if status == "needs_tools" else ()
        if status == "answered":
            record.referent = tuple(_entities(answer))
        record.entities = _resolved_entities(ex, answer)
        if status != "needs_tools":
            await self._end(record)

    async def _end(self, record: _Plan) -> None:
        """Record the turn's outcome on its applied constructions, once (``store.record_uses``): the outcome of the
        plan as it now stands (``_outcome_of``)."""
        if record.recorded or not record.uses or not self.record_standing or not record.standing:
            return
        record.recorded = True
        outcome = _outcome_of(record.plan)
        known = {s.id for s in self.store.known_specs()}
        # a row a confirm removed since the reading is skipped
        uses: dict[str, Outcome] = {name: outcome for name in record.uses if name in known}
        try:
            # ``recorded`` is set above, before the thread hop: the write completes in its thread even if this task
            # is cancelled, and a plan that says it was not recorded would be written again
            await asyncio.to_thread(self.store.record_uses, uses)
        except Exception as exc:  # a construction a confirm replaced since the reading; a failed graph write (Neo4j
            # unavailable, transient, driver errors): a stats write never fails a turn
            log.warning("bonfire %s: standing not recorded: %s", self.bonfire_id, type(exc).__name__)
            record.plan = replace(record.plan, diagnostics=[*record.plan.diagnostics, f"standing not recorded: {exc}"])

    def _unread(
        self,
        plan_id: str,
        turn: Turn,
        info: dict[str, Any],
        diagnostics: list[str],
        frames: Sequence[Mapping[str, Any]] = (),
        left: Sequence[str] = (),
    ) -> TurnPlan:
        plan = unread_plan(self.bonfire_id, turn, diagnostics, info, frames=frames, left=left)
        return replace(plan, plan_id=plan_id)

    # -- alternatives -------------------------------------------------------------------------------------------------

    async def alternatives(
        self, turn: Turn, *, through: str | None = None, limit: int = ALTERNATIVES, maximal: bool = True
    ) -> Alternatives:
        """Every distinct complete analysis of ``turn``, each a plan walked with ``resume`` exactly as one from
        ``plan``, with its path (``Alternative``): the first ``limit`` of them.

        The turn is read as ``plan`` reads it (its final marks included: the text read is the one ``plan``'s reading
        used), the primary analysis alone, then every complete analysis of that text is searched for
        (``RuntimeEngine.comprehend_every``: up to ``ALTERNATIVES_RAW`` analyses, ``ALTERNATIVES_NODE_CAP`` nodes,
        ``ALTERNATIVES_DEADLINE_MS``), the first being ``plan``'s reading. Kept, in the search's order: a complete
        analysis through a root (through ``through`` when one is named: a root's spec id, ``ValueError`` for anything
        else), whose slot takes no final mark, that applies no draft ``teach.admissible`` refuses for this speaker in
        this chat; two analyses whose programs are equal, variables renamed in order of appearance, are one (the
        earlier kept). ``maximal`` then drops an analysis that leaves to the generated filler a piece that another kept
        analysis reads with another filler in the same slot (same address, same boundaries), never the first. Of what
        is left the first ``limit`` become plans (at most ``limit`` kept per call). ``found[0]`` is the reading
        ``plan`` gives when ``plan`` reads the turn completely with a reading this speaker may use (and through
        ``through`` when one is named); otherwise ``found`` holds what the wider search finds, in its order, and may be
        empty. Plan's reading is the search's analysis with its program, put first wherever the search found it (the
        deadline bounds the wider search only; the first search is plan's, bounded as plan's is); a search that does
        not reproduce it raises ``RuntimeEngineError``. ``cut_short``: a bound ended the search (its node cap, its
        deadline, or ``ALTERNATIVES_RAW`` analyses), or more than ``limit`` alternatives were left. An analysis whose
        slots and fillers do not pair one to one has no path: it is left out, with one WARNING per call naming the
        construction ids of each one left out (never the turn's text); when it is plan's own reading the call raises
        ``RuntimeEngineError`` (a reading with no path cannot be credited).

        A path names the construction in each slot by the analysis's own structure (``RuntimeEngine.comprehend_every``
        says which slot unit each construction filled), whatever order the search applied them in, in any grammar.
        Each plan is read with a window of 0, its frames are its own analysis's, it never records standing and it is
        never in the chat's turn window. ``alternatives`` raises ``RuntimeEngineError`` when the engine fails, and
        then keeps no plan."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit must be a positive integer, got {limit!r}")
        if through is not None:
            spec = self._all_specs().get(through)
            if spec is None or spec.unit != "root":
                raise ValueError(f"through must name a root construction, got {through!r}")
        async with self._chat_lock(turn.chat_id):
            turn = replace(turn, window=())  # the window is the runtime's, never the caller's
            await self._ensure_started()
            own, every, cut = await asyncio.to_thread(self._every, turn)
            norm, specs = own.norm, self._all_specs()
            marks = _final_marks(norm)

            def usable(reading: Reading) -> bool:
                return admissible(reading, self.store, turn.chat_id, turn.speaker or "")

            def rooted(reading: Reading, root_id: str | None) -> bool:
                root = _root(reading, specs)
                return root is not None and (root_id is None or root.id == root_id)

            primary, first = own.reading, None
            if primary is not None and primary.complete and rooted(primary, through) and usable(primary):
                # plan's own reading comes first: the search's analysis with its program, wherever it was found
                key, root_id = _program_key(primary.meaning), getattr(_root(primary, specs), "id", None)
                same = (r for r in every if _program_key(r.meaning) == key and rooted(r, root_id) and usable(r))
                first = next(same, None)
                if first is None:
                    raise RuntimeEngineError("the runtime engine's search did not reproduce plan's reading of the turn")
            candidates = [*([first] if first is not None else []), *(r for r in every if r is not first)]
            kept: list[tuple[Reading, CxnSpec, tuple[tuple[str, str, tuple[int, int] | None], ...]]] = []
            seen: set[str] = set()
            unpaired: list[list[str]] = []
            for reading in candidates:
                # plan's own reading takes no final mark (``_turn_reading``): only the search's others are checked
                if not (reading.complete and rooted(reading, through) and usable(reading)) or (
                    reading is not first and _takes_mark(reading, norm, marks)
                ):
                    continue
                root = _root(reading, specs)
                key = _program_key(reading.meaning)
                if root is None:  # never: rooted
                    continue
                if key in seen:
                    continue
                seen.add(key)
                try:
                    path = _path(reading)
                except RuntimeError as exc:
                    if reading is first:  # the caller could not credit plan's own reading with no path
                        raise RuntimeEngineError(f"plan's reading of the turn has no path: {exc}") from exc
                    unpaired.append([spec_id(name) for name in reading.applied])
                    continue
                kept.append((reading, root, path))
            if unpaired:
                log.warning(
                    "bonfire %s: %d alternative(s) left out, their slots and fillers do not pair: %s",
                    self.bonfire_id,
                    len(unpaired),
                    unpaired,
                )
            if maximal:
                read = {(address, span) for _, _, path in kept for address, cxn, span in path if cxn != GENERATED}
                kept = [
                    k
                    for i, k in enumerate(kept)
                    if i == 0 or not any(cxn == GENERATED and (address, span) in read for address, cxn, span in k[2])
                ]
            cut = cut or len(kept) > limit
            found: list[Alternative] = []
            for reading, root, path in kept[:limit]:
                info = {
                    "applied": list(reading.applied),
                    "ms": reading.ms,
                    "complete": True,
                    "normalised": norm.text,
                    "root": root.id,
                    "retried": own.retried,
                }
                frames = frames_of([reading], norm, {**specs, **self._frame_parts()}, self.registry)
                left = surface_left(reading, norm)
                plan = await self._planned(
                    uuid.uuid4().hex, turn, norm, reading, root, specs, info, frames, left, 0, (), alternative=True
                )
                found.append(Alternative(plan=plan, path=tuple((address, cxn) for address, cxn, _ in path)))
            return Alternatives(found=tuple(found), cut_short=cut)

    def _every(self, turn: Turn) -> tuple[_Reads, list[Reading], bool]:
        """The turn's own read (``_read``, the primary analysis alone: which text ``plan`` reads) and every complete
        analysis of that text, under one hold of the engine."""
        with self._grammar:
            own = self._read(turn.text, analyses=1)
            every, cut = self.engine.comprehend_every(
                own.norm.text,
                cap=ALTERNATIVES_RAW,
                node_cap=ALTERNATIVES_NODE_CAP,
                deadline_ms=ALTERNATIVES_DEADLINE_MS,
            )
            return own, every, cut

    # -- resume -------------------------------------------------------------------------------------------------------

    async def resume(self, plan_id: str, results: Sequence[Mapping[str, Any]]) -> TurnPlan:
        """Continue a ``needs_tools`` plan with its pending tool step's result(s) ``{"step_id", "tool", "args",
        "verdict", "value", "detail"}``; ``tool`` and ``args`` must be the pending step's own. Any verdict but
        ``committed`` ends the turn as a repair that keeps the program's licence and carries ``failure``; otherwise
        the program replays with every tool result so far and the values it already computed (no completed memory op
        runs again)."""
        async with self._chat_lock(self._get(plan_id).turn.chat_id):
            return await self._resume(plan_id, results)

    async def _resume(self, plan_id: str, results: Sequence[Mapping[str, Any]]) -> TurnPlan:
        record = self._get(plan_id)  # again, under the chat's lock: it may have expired or ended while waiting
        if record.plan.status != "needs_tools" or record.program is None or record.spans is None:
            raise ResumeError(f"plan {plan_id} is {record.plan.status}: it waits on no tool step")
        if record.busy:
            raise ResumeError(f"plan {plan_id} is being resumed already")
        parsed = _results(results)
        pending = {step.step_id: step for step in record.pending}
        unknown = sorted(r["step_id"] for r in parsed if r["step_id"] not in pending)
        if unknown:
            raise ResumeError(f"plan {plan_id}: step(s) {unknown} are not pending (pending: {sorted(pending)})")
        for r in parsed:
            step = pending[r["step_id"]]
            if r["tool"] != step.tool or _canonical(r["args"]) != _canonical(jsonable(step.args)):
                raise ResumeError(f"plan {plan_id}: the result for step {r['step_id']} is not for {step.tool}(args)")
        failed = [r for r in parsed if r["verdict"] != "committed"]
        if failed:
            diagnostics = [
                f"tool step {r['step_id']} ({r['tool']}) {r['verdict']}: {r['detail'] if r['detail'] is not None else repr(r['value'])}"
                for r in failed
            ]
            first = failed[0]
            record.pending = ()
            record.plan = replace(
                record.plan,
                status="repair",
                tool_steps=[],
                licence=_licence(record.program, self.registry, self.policy, record.turn.speaker),
                diagnostics=diagnostics,
                failure={
                    "step_id": first["step_id"],
                    "tool": first["tool"],
                    "kind": first["verdict"],
                    "detail": first["detail"],
                },
            )
            await self._end(record)
            return self._keep(record)
        prefix = f"{plan_id}:"
        record.resumed.update({r["step_id"].removeprefix(prefix): r["value"] for r in parsed})
        await self._replay(record)
        return self._keep(record)

    async def retry(self, plan_id: str) -> TurnPlan:
        """Re-issue the step a verdict refused, on a plan that carries a ``failure``: the plan goes back to
        ``needs_tools`` with that step pending under its same ``step_id`` (every earlier committed result stays; the
        failed step's refusal is dropped). A plan with no ``failure`` is ``ResumeError``."""
        async with self._chat_lock(self._get(plan_id).turn.chat_id):
            record = self._get(plan_id)
            failure = record.plan.failure
            if failure is None or record.program is None or record.spans is None:
                raise ResumeError(f"plan {plan_id} is {record.plan.status}: it has no failed step to retry")
            if record.busy:
                raise ResumeError(f"plan {plan_id} is being resumed already")
            kept = _snapshot(record)
            record.plan, record.recorded = replace(record.plan, failure=None), False  # a retry is a new attempt
            try:
                await self._replay(record)
            except BaseException:
                for name, value in kept.items():  # a failed replay leaves the whole plan as it was
                    setattr(record, name, value)
                raise
            return self._keep(record)

    async def _replay(self, record: _Plan) -> None:
        """Run the plan's program again with every result so far and the values already computed."""
        if record.program is None or record.spans is None:
            raise RuntimeError("a replayed plan has a program")
        record.busy = True
        try:
            memory, store_memory = self._memory(record.turn)
            started = time.perf_counter()
            ex = await execute(
                record.program,
                self.registry,
                memory,
                turn=record.turn,
                spans=record.spans,
                resumed=dict(record.resumed),
                prior=record.values,  # this plan's own values only: prior is trusted
                committed=_CommittedSteps(record.committed),
                policy=self.policy,
                id_prefix=f"{record.plan.plan_id}:",
            )
            await self._outcome(record, ex, memory, store_memory, (time.perf_counter() - started) * 1000)
        finally:
            record.busy = False

    # -- plans and the turn window ------------------------------------------------------------------------------------

    def window(self, chat_id: str, size: int = DEFAULT_WINDOW) -> list[dict[str, Any]]:
        """The chat's turn window, latest first: per plan, its id, status, answer and the entities it resolved."""
        return [
            {
                "plan_id": r.plan.plan_id,
                "status": r.plan.status,
                "answer": r.plan.answer,
                "entities": jsonable(r.entities),
            }
            for r in self._window(chat_id, size)
        ]

    def _window(self, chat_id: str, size: int) -> list[_Plan]:
        with self._plans_lock:
            self._purge()
            drawn = [r for r in self._plans.values() if r.drew and r.turn.chat_id == chat_id]
            latest = self._latest.get(chat_id)
        drawn.sort(key=lambda r: r.seq)
        if not size or not drawn or drawn[-1].seq != latest:
            return []  # the chat's latest plan has expired or been evicted: no older plan stands in for it
        return list(reversed(drawn[-size:]))

    def _referents(self, chat_id: str, size: int) -> tuple[tuple[dict[str, Any], ...], ...]:
        return tuple(r.referent for r in self._window(chat_id, size))

    def _keep(self, record: _Plan) -> TurnPlan:
        with self._plans_lock:
            self._purge()
            if record.plan.plan_id not in self._plans:
                self._seq += 1
                record.seq = self._seq
                if record.drew:
                    self._latest[record.turn.chat_id] = record.seq
            record.touched = self.clock()
            self._plans[record.plan.plan_id] = record
            self._plans.move_to_end(record.plan.plan_id)
            while len(self._plans) > self.max_plans:
                self._forget(self._plans.popitem(last=False)[0], "evicted")
        return record.plan

    def _get(self, plan_id: str) -> _Plan:
        with self._plans_lock:
            self._purge()
            record = self._plans.get(plan_id)
            if record is None:
                why = self._forgotten.get(plan_id)
                if why == "expired":
                    raise UnknownPlan(f"plan {plan_id!r} expired {self.ttl:g} s after its last use")
                if why == "evicted":
                    raise UnknownPlan(f"plan {plan_id!r} was evicted: this runtime keeps its {self.max_plans} latest")
                raise UnknownPlan(f"no plan {plan_id!r}: this runtime never made it (or forgot it long ago)")
            return record  # touched only when a resume is accepted (``_keep``)

    def _purge(self) -> None:
        """Expire the plans past ``ttl``; then forget the latest-plan seq and the turn lock of every chat with no plan
        left (a lock only while no turn of the chat holds or awaits it), so neither grows with every chat ever seen."""
        now = self.clock()
        for plan_id in [p for p, r in self._plans.items() if now - r.touched > self.ttl]:
            del self._plans[plan_id]
            self._forget(plan_id, "expired")
        chats = {r.turn.chat_id for r in self._plans.values()}
        for chat_id in [c for c in self._latest if c not in chats]:
            del self._latest[chat_id]
        for locks in list(self._chat_locks.values()):
            for chat_id in [c for c, held in locks.items() if c not in chats and not held.turns]:
                del locks[chat_id]

    def _forget(self, plan_id: str, why: str) -> None:
        """Remember why ``plan_id`` left (bounded: ``FORGOTTEN_IDS``), so ``UnknownPlan`` can say so."""
        self._forgotten[plan_id] = why
        self._forgotten.move_to_end(plan_id)
        while len(self._forgotten) > FORGOTTEN_IDS:
            self._forgotten.popitem(last=False)

    @contextlib.asynccontextmanager
    async def _chat_lock(self, chat_id: str) -> AsyncIterator[None]:
        """Hold ``chat_id``'s turn lock (one turn of a chat at a time, per event loop); counted while held or awaited,
        so ``_purge`` never drops a lock a turn still uses."""
        locks = self._chat_locks.setdefault(asyncio.get_running_loop(), {})
        held = locks.setdefault(chat_id, _ChatLock())
        held.turns += 1
        try:
            async with held.lock:
                yield
        finally:
            held.turns -= 1

    def _all_specs(self) -> dict[str, CxnSpec]:
        """Every construction the engine may apply: the base set, every stored row (drafts included) and the cards."""
        return {s.id: s for s in self.store.known_specs()}

    def _frame_parts(self) -> dict[str, CxnSpec]:
        """The frame parts ``load_store`` loads for the live cards and taught phrases (never stored)."""
        return {s.id: s for s in loaded_frame_parts(self.store.load_order())}


# -- the gate ---------------------------------------------------------------------------------------------------------


def executable(
    reading: Reading,
    spans: SpanText,
    specs: Sequence[CxnSpec],
    registry: Registry,
    *,
    root: CxnSpec | None = None,
) -> tuple[Program | None, list[str]]:
    """The program a reading names and why it must not run ([] when it may). It may run only when the reading is
    complete, a root construction applied, the do-support guard passes, ``validate`` passes, no op output is left
    unapplied (every output is read by another op or is the answer; a tool's result is exempt: its effect is the
    point), and no name slot holds a quantified noun phrase."""
    by_id = {s.id: s for s in specs}
    root = root or _root(reading, by_id)
    if root is None:
        return None, ["no construction read the turn"]
    if not reading.complete:
        return None, [f"no construction read the whole turn ({root.id} read part of it)"]
    guarded = reading_guard(reading, spans, specs)
    if guarded is not None:
        return None, [guarded]
    try:
        program = Program.from_meaning(reading.meaning)
    except MeaningError as exc:
        return None, [f"the reading's meaning is malformed: {exc}"]
    effective = Registry({**registry.specs, **VALUE_SPECS})
    problems = validate(program, effective)
    if problems:
        return program, problems
    for var, op in dangling(program, effective):
        problems.append(f"{_APPLICATION_SUFFIX.sub('', var)} ({op}) is never applied: no keyword applies it yet")
    for predicate in reading.meaning:
        if predicate[0] == "name-span" and len(predicate) == 4:
            left, right = predicate[2], predicate[3]
            if isinstance(left, int) and isinstance(right, int):
                try:
                    cut = spans.np(left, right)
                except ValueError as exc:
                    problems.append(f"name-span [{left}, {right}): {exc}")
                    continue
                if cut.quantified:
                    phrase = spans.normalised[left:right]
                    problems.append(f"{phrase!r} is quantified: no keyword applies the quantifier yet")
    return program, problems


def dangling(program: Program, registry: Registry) -> list[tuple[str, str]]:
    """(variable, op) for every output no op reads and that is not the answer; tool results are exempt. The program
    must pass ``validate`` against ``registry`` first."""
    specs = registry.specs
    read = {arg.name for op in program.ops for arg in op.args[len(specs[op.name].outputs) :] if isinstance(arg, Var)}
    if program.answer is not None:
        read.add(program.answer.name)
    out: list[tuple[str, str]] = []
    for op in program.ops:
        spec = specs[op.name]
        if spec.kind == "tool":
            continue
        out += [(a.name, op.name) for a in op.args[: len(spec.outputs)] if isinstance(a, Var) and a.name not in read]
    return out


# -- helpers ----------------------------------------------------------------------------------------------------------


def _takes_speaker(fn: Callable[..., Any] | None) -> bool:
    """Whether a ``recall`` callable takes a ``speaker`` keyword (``mk serve``'s does; a plain ``recall(text)`` not)."""
    if fn is None:
        return False
    try:
        return "speaker" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _root(reading: Reading | None, specs: Mapping[str, CxnSpec]) -> CxnSpec | None:
    """The first applied construction that is a root (an alternate's name maps to its spec)."""
    if reading is None:
        return None
    for name in reading.applied:
        spec = specs.get(spec_id(name))
        if spec is not None and spec.unit == "root":
            return spec
    return None


def _path(reading: Reading) -> tuple[tuple[str, str, tuple[int, int] | None], ...]:
    """A complete reading's constructions by the slot each filled: ``(address, spec id, boundaries)``, the root first
    (address ``""``, no boundaries), then every slot in tree order, a slot's children leftmost first, addressed by the
    slot names from the root (``"s"``, ``"s/h"``, ``"s/d/h"``). ``Reading.slots`` gives the tree (each slot unit with
    the unit holding it; a slot held by no slot unit is the root's); ``Reading.fills`` (``comprehend_every``) the slot
    unit each applied construction filled, so a slot is paired with its own filler whatever order the search applied
    them in. A reading whose constructions and slots do not pair one to one (no fill record, a slot filled twice or
    never, a fill naming no slot of the reading, not exactly one construction filling none) is a ``RuntimeError``."""
    problem = f"the applied constructions {list(reading.applied)} and the slots {reading.slots} do not pair"
    if not reading.applied or len(reading.fills) != len(reading.applied):
        raise RuntimeError(f"{problem}: no fill record for each construction ({reading.fills})")
    filler: dict[str, str] = {}
    roots = [name for name, slot in zip(reading.applied, reading.fills, strict=True) if not slot]
    for name, slot in zip(reading.applied, reading.fills, strict=True):
        if slot in filler:
            raise RuntimeError(f"{problem}: {slot} is filled twice")
        if slot:
            filler[slot] = name
    held = {slot for _holder, slot, _left, _right in reading.slots}
    if len(roots) != 1 or set(filler) != held:
        raise RuntimeError(f"{problem}: fills {reading.fills}")
    children: dict[str, list[tuple[int, str, int]]] = {}
    for holder, slot, left, right in reading.slots:
        children.setdefault(holder if holder in held else "", []).append((left, slot, right))
    path: list[tuple[str, str, tuple[int, int] | None]] = [("", spec_id(roots[0]), None)]

    def visit(unit: str, prefix: str) -> None:
        for left, slot, right in sorted(children.get(unit, ())):
            match = _SLOT_UNIT.fullmatch(slot)
            address = f"{prefix}{match.group(1) if match else slot}"
            path.append((address, spec_id(filler[slot]), (left, right)))
            visit(slot, address + "/")

    visit("", "")
    return tuple(path)


def _program_key(meaning: Sequence[Sequence[Any]]) -> str:
    """A reading's program up to the names of its variables and the order of its predicates, as JSON: two analyses with
    one key are one. Each variable is coloured by the predicates it occurs in (its positions, the constants, the
    colours of the other variables), refined until the colours stop splitting; the predicates are sorted with each
    variable masked by its colour, the variables renamed ``?v0``, ``?v1``... in that order, and the renamed predicates
    sorted again. The key is a renaming of the program itself, so two different programs never share one."""
    try:
        program = Program.from_meaning(meaning).to_json()
    except MeaningError:
        program = [list(p) for p in meaning]

    def variable(a: Any) -> bool:
        return isinstance(a, str) and a.startswith("?")

    def masked(p: Sequence[Any], colour: Mapping[str, str]) -> str:
        return json.dumps([["?", colour[a]] if variable(a) else a for a in p])

    colour = {a: "" for p in program for a in p if variable(a)}
    while True:
        signature = {
            v: json.dumps(
                [
                    colour[v],
                    sorted(
                        json.dumps([masked(p, colour), [i for i, a in enumerate(p) if a == v]])
                        for p in program
                        if v in p
                    ),
                ]
            )
            for v in colour
        }
        ranks = {sig: str(k) for k, sig in enumerate(sorted(set(signature.values())))}
        refined = {v: ranks[signature[v]] for v in colour}
        split = len(set(refined.values())) > len(set(colour.values()))
        colour = refined
        if not split:
            break
    names: dict[str, str] = {}
    for p in sorted(program, key=lambda p: masked(p, colour)):
        for a in p:
            if variable(a) and a not in names:
                names[a] = f"?v{len(names)}"
    return json.dumps(sorted(json.dumps([names[a] if variable(a) else a for a in p]) for p in program))


def _composed_frames(composed: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """A composed plan's frames: the composed reading's, those of the open frame's ``name`` ``composed_with`` it; any
    other analysis of the rendering (another root's) was not read through the open frame, and stays uncomposed."""
    return [{**frame, "composed_with": name} if frame["name"] == name else frame for frame in composed]


def _final_mark(norm: Normalised) -> bool:
    pos = norm.spans.pos
    return bool(pos) and pos[-1] == FINAL_MARK_POS


def _final_marks(norm: Normalised) -> range:
    """The indices of ``norm``'s final marks, the tokens ``_without_final_mark`` drops: its trailing tokens of POS
    ``FINAL_MARK_POS`` (a retry mark included), up to a closing quote or bracket (``_paired_mark``), which stays with
    its opener; every token when nothing but marks is in it; none when its tokens carry no POS."""
    spans = norm.spans
    count = len(spans.tokens)
    if not count or len(spans.pos) < count:
        return range(count, count)
    if all(pos == FINAL_MARK_POS for pos in spans.pos[:count]):
        return range(count)
    start = count
    while spans.pos[start - 1] == FINAL_MARK_POS and not _paired_mark(spans, start - 1):
        start -= 1
    return range(start, count)


def _without_final_mark(norm: Normalised) -> str:
    """The text ``norm`` was read from, without its final marks (``_final_marks``: its trailing tokens of POS
    ``FINAL_MARK_POS``, a retry mark included), as written. A closing quote or bracket (morph ``PunctType`` in
    ``PAIRED_MARK_TYPES``) is no final mark: it stays with its opener ('"JUJ-3".' is '"JUJ-3"'). '' when nothing but
    marks is in it: a turn of marks alone fills no slot ("?" is no ticket)."""
    spans = norm.spans
    if not spans.tokens or len(spans.pos) < len(spans.tokens):  # no POS to tell a mark by: the text as written
        return spans.original.strip()
    start = _final_marks(norm).start
    return spans.original[: spans.tokens[start - 1][3]].strip() if start else ""


def _takes_mark(reading: Reading | None, norm: Normalised, marks: range) -> bool:
    """Whether a slot of ``reading`` reaches into ``marks``, ``norm``'s final marks (``_final_marks``): a slot unit's
    normalised boundaries (``Reading.slots``, every slot of the analysis, a filler's own and an unfilled one included)
    overlap the span from the first final mark's start to the last one's end."""
    if reading is None or not marks:
        return False
    first, last = norm.spans.tokens[marks.start][0], norm.spans.tokens[marks[-1]][1]
    return any(left < last and right > first for _holder, _slot, left, right in reading.slots)


def _asks(norm: Normalised, marks: range) -> bool:
    """Whether the final marks ``marks`` of ``norm`` hold a question mark as written (``_question_mark``)."""
    spans = norm.spans
    return any(_question_mark(ch) for i in marks for ch in spans.original[spans.tokens[i][2] : spans.tokens[i][3]])


def _question_mark(ch: str) -> bool:
    """A question mark in any script: a punctuation or symbol character (category ``P*`` or ``So``) whose Unicode
    name holds QUESTION and MARK, or INTERROBANG: "?", FULLWIDTH QUESTION MARK (U+FF1F), the emoji BLACK and WHITE
    QUESTION MARK ORNAMENT (U+2753, U+2754), INVERTED QUESTION MARK (U+00BF), ARABIC QUESTION MARK (U+061F), DOUBLE
    QUESTION MARK (U+2047), QUESTION EXCLAMATION MARK (U+2048), EXCLAMATION QUESTION MARK (U+2049), INTERROBANG
    (U+203D), and the like. A math relation with a question mark above it (``Sm``) and the invisible TAG QUESTION MARK
    (``Cf``) are not marks the speaker wrote."""
    name = unicodedata.name(ch, "")
    marked = ("QUESTION" in name and "MARK" in name) or "INTERROBANG" in name
    category = unicodedata.category(ch)
    return marked and (category.startswith("P") or category == "So")


def _complete_root(reading: Reading | None, specs: Mapping[str, CxnSpec]) -> bool:
    """A complete reading that draws a root."""
    return reading is not None and reading.complete and _root(reading, specs) is not None


def _paired_mark(spans: TokenSpans, index: int) -> bool:
    morph = spans.morph[index] if index < len(spans.morph) else ""
    features = dict(f.split("=", 1) for f in morph.split("|") if "=" in f)
    return features.get("PunctType") in PAIRED_MARK_TYPES


def _turn_json(turn: Turn) -> dict[str, Any]:
    return {
        "text": turn.text,
        "speaker": turn.speaker,
        "addressee": turn.addressee,
        "participants": list(turn.participants),
        "chat_id": turn.chat_id,
        "message_id": turn.message_id,
        "at": turn.at,
    }


def _unresolved(ex: Execution) -> list[str]:
    """Empty results that make any answer or tool step meaningless: a name that names nobody (find-entity found
    nothing), a verb that names no frame (frames-of found nothing)."""
    out: list[str] = []
    for step in ex.trace:
        if step.ok and step.out == [] and step.op == "find-entity":
            out.append(f"find-entity: {step.args.get('name')!r} names nobody in the graph")
        elif step.ok and step.out == [] and step.op == "frames-of":
            out.append(f"the verb {step.args.get('lemma')!r} names no frame")
    return out


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, (str, list, tuple, dict, set)) and not value)


def _is_entity(item: Any) -> bool:
    return isinstance(item, Mapping) and isinstance(item.get("uuid"), str) and "kinds" in item


def _entities(value: Any) -> list[dict[str, Any]]:
    items = value if isinstance(value, (list, tuple)) else [value]
    return [dict(i) for i in items if _is_entity(i)]


def _resolved_entities(ex: Execution, answer: Any) -> list[dict[str, Any]]:
    """The entities a plan resolved (every find-entity output) and its answer named, first seen first."""
    seen: dict[str, dict[str, Any]] = {}
    for step in ex.trace:
        if step.op == "find-entity" and step.ok:
            for entity in _entities(step.out):
                seen.setdefault(entity["uuid"], entity)
    for entity in _entities(answer):
        seen.setdefault(entity["uuid"], entity)
    return list(seen.values())


def _narrow(answer: Any, shape: str | None) -> tuple[Any, list[str] | None]:
    """Narrow a list of entities by the kind the shape asks for, when the list holds that kind and others; else the
    answer as it is. Returns the answer and the uuids dropped (None when nothing was dropped)."""
    rule = SHAPE_KIND.get(shape or "")
    if rule is None or not isinstance(answer, list) or not answer or not all(_is_entity(e) for e in answer):
        return answer, None
    label, keep = rule
    fold = label.casefold()

    def has(entity: Mapping[str, Any]) -> bool:
        return fold in {str(k).casefold() for k in entity.get("kinds") or ()}

    kept = [e for e in answer if has(e) == keep or not e.get("kinds")]
    if not kept or len(kept) == len(answer):
        return answer, None
    return kept, [e["uuid"] for e in answer if e not in kept]


async def _recall(memory: CompositeMemory, text: str, steps: list[dict[str, Any]]) -> Any:
    """The fallback ``recall`` op over the turn's text (traced); a diagnostic string when it cannot run."""
    started = time.perf_counter()
    try:
        out = await memory.run("recall", {"text": text})
    except Exception as exc:  # noqa: BLE001 — the repair stands without it; the reason is a diagnostic
        steps.append({"op": "recall", "args": {"text": text}, "out": None, "ms": 0.0, "ok": False})
        return f"recall unavailable: {exc}"
    ms = (time.perf_counter() - started) * 1000
    steps.append({"op": "recall", "args": {"text": text}, "out": out, "ms": ms, "ok": True})
    return out


def _repair_licence(
    program: Program | None,
    registry: Registry,
    policy: ConfirmationPolicy,
    speaker: str | None,
    *,
    invokes: bool = False,
) -> list[dict[str, Any]] | None:
    """A repair's licence: the program's own when it names a tool; None (today's path takes the turn) otherwise. A
    program that held a reference (``invoke``) is tool-bearing whatever expanded (ruling R4): its licence is that of
    what did expand, ``[]`` when none of it names a tool, never None (an effectful turn never reaches today's tools)."""
    licence = _licence(program, registry, policy, speaker) if program is not None else []
    return licence if invokes else licence or None


def _licence(
    program: Program, registry: Registry, policy: ConfirmationPolicy, speaker: str | None
) -> list[dict[str, Any]]:
    """The licence rows of the tools the program names. An op the registry no longer knows (a catalogue dropped a tool
    a taught row copied inline) is still listed, its effect unknown: the game confirms (R6)."""
    out: list[dict[str, Any]] = []
    for op in program.ops:
        spec = registry.specs.get(op.name)
        if any(row["tool"] == op.name for row in out):
            continue
        if spec is None and op.name not in VALUE_SPECS:
            out.append({"tool": op.name, "side_effect": None, "needs_confirmation": True, "reads": [], "writes": []})
            continue
        if spec is None or spec.kind != "tool":
            continue
        side_effect = spec.side_effect if isinstance(spec.side_effect, bool) else None
        out.append(
            {
                "tool": op.name,
                "side_effect": side_effect,
                "needs_confirmation": policy.needs_confirmation(spec, actor=speaker, speaker=speaker),
                "reads": list(spec.reads),
                "writes": list(spec.writes),
            }
        )
    return out


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def committed_entries(committed: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """The ``committed`` input of ``Runtime.plan``: ``{tool: str, args: object, value}`` each."""
    shape = "committed is a list of {tool: str, args: object, value}"
    if isinstance(committed, (str, bytes, Mapping)) or not isinstance(committed, Sequence):
        raise ValueError(f"{shape}, got {committed!r}")
    out: list[dict[str, Any]] = []
    for row in committed:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("tool"), str)
            or not isinstance(row.get("args"), Mapping)
        ):
            raise ValueError(f"{shape}, got {row!r}")
        if "value" not in row:
            raise ValueError(f"{shape}: value is required (null when the effect produced none), got {row!r}")
        out.append({"tool": row["tool"], "args": jsonable(row["args"]), "value": row["value"]})
    return tuple(out)


class _CommittedSteps:
    """``executor.Committed`` over ``Runtime.plan``'s committed entries: matched on tool and canonical-JSON args; each
    entry resolves one step (a second identical step is a second effect)."""

    def __init__(self, entries: Sequence[Mapping[str, Any]]) -> None:
        self._left = [(e["tool"], _canonical(e["args"]), e["value"]) for e in entries]

    def _find(self, tool: str, args: Mapping[str, Any]) -> int | None:
        key = _canonical(jsonable(args))
        return next((n for n, (t, a, _) in enumerate(self._left) if t == tool and a == key), None)

    def holds(self, tool: str, args: Mapping[str, Any]) -> bool:
        return self._find(tool, args) is not None

    def take(self, tool: str, args: Mapping[str, Any]) -> tuple[bool, Any]:
        n = self._find(tool, args)
        return (False, None) if n is None else (True, self._left.pop(n)[2])


_RESULT_SHAPE = (
    "{step_id: str, tool: str, args: object, verdict: one of " + ", ".join(VERDICTS) + ", value, detail?: str}"
)


def _results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(results, (str, bytes, Mapping)) or not isinstance(results, Sequence) or not results:
        raise ResumeError(f"results must be a non-empty list of {_RESULT_SHAPE}")
    parsed: list[dict[str, Any]] = []
    for row in results:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("step_id"), str)
            or not isinstance(row.get("tool"), str)
            or not isinstance(row.get("args"), Mapping)
            or row.get("verdict") not in VERDICTS
            or not (row.get("detail") is None or isinstance(row.get("detail"), str))
        ):
            raise ResumeError(f"a result is {_RESULT_SHAPE}, got {row!r}")
        parsed.append(
            {
                "step_id": row["step_id"],
                "tool": row["tool"],
                "args": jsonable(row["args"]),
                "verdict": row["verdict"],
                "value": row.get("value"),
                "detail": row.get("detail"),
            }
        )
    ids = [r["step_id"] for r in parsed]
    if len(set(ids)) != len(ids):
        raise ResumeError(f"a step's result is given twice: {ids}")
    return parsed


def render_op(predicate: Sequence[Any], specs: Mapping[str, Any]) -> str:
    """One meaning predicate as keyword rules text: ``op(outputs ← inputs)`` (``answer(?v)``)."""
    name, args = str(predicate[0]), [str(a) for a in predicate[1:]]
    spec = specs.get(name)
    if name == ANSWER_PREDICATE or spec is None or not spec.outputs:
        return f"{name}({', '.join(args)})"
    n = len(spec.outputs)
    return f"{name}({', '.join(args[:n])} ← {', '.join(args[n:])})"


def _hand(applied: Sequence[str], specs: Mapping[str, CxnSpec], op_specs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The drawn constructions, in application order, one row per spec."""
    hand: list[dict[str, Any]] = []
    for name in applied:
        spec = specs.get(spec_id(name))
        if spec is None or any(row["id"] == spec.id for row in hand):
            continue
        s = spec.standing
        hand.append(
            {
                "id": spec.id,
                "kind": spec.kind,
                "unit": spec.unit,
                "maker": spec.provenance.get(TEACHER) or spec.kind,
                "standing": {
                    "uses_ok": s.uses_ok,
                    "uses_failed": s.uses_failed,
                    "uses_world": s.uses_world,
                    "confirmed": s.confirmed,
                },
                "rules": [render_op(p, op_specs) for p in spec.meaning],
            }
        )
    return hand


__all__ = [
    "committed_entries",
    "ALTERNATIVES",
    "ALTERNATIVES_DEADLINE_MS",
    "ALTERNATIVES_NODE_CAP",
    "ALTERNATIVES_RAW",
    "ANALYSES",
    "DEFAULT_WINDOW",
    "GENERATED",
    "MAX_PLANS",
    "PLAN_TTL",
    "VERDICTS",
    "Alternative",
    "Alternatives",
    "ResumeError",
    "Runtime",
    "TurnPlan",
    "UnknownPlan",
    "Verdict",
    "dangling",
    "default_registry",
    "executable",
    "jsonable",
    "render_op",
    "unread_plan",
]
