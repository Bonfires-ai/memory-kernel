"""The runtime grammar's own FCG Go engine: a separate process on its own port, one grammar, deterministic readings.

The extraction engine (``pipeline/engine.py``) is not touched. This engine runs the same pinned binary, resolved
through the ``mk setup`` receipt of the data dir (``setup_runtime.managed_path(data_dir, "fcg-go")``) and checked
against the ``runtime-pins.json`` sha256 by ``pipeline.engine.Engine.launch``, which also owns the launch (own
session, ``/proc`` ownership, readiness) and the teardown (SIGTERM, then SIGKILL, by pid). Port: ``MK_RUNTIME_FCG_PORT``,
default 9711. ``<out>/engine.json`` names the engine and the process that started it (``owner``): a later instance
reclaims (kills) the engine only once that owner is gone; a port held by anything else is ``PortBusy``.

ADDRESSING. pyfcg addresses its engine through module globals: ``pyfcg.routes.get`` / ``post`` rebuild the URL from
``pyfcg.fcg_go_bridge.server_address`` / ``server_port`` on every call, and a ``Grammar`` or agent carries only a
grammar id, so there is no per-grammar or per-agent address. A process holding the extraction engine (whose capture
calls ``pyfcg.init``) and this one would route one engine's calls to the other through those globals, and a lock taken
here would not help, because the extraction side never takes it. So this engine never uses pyfcg's transport: it
speaks pyfcg's HTTP routes itself, over ``http.client`` to its own port, with the payloads pyfcg would send. It never
reads or writes the bridge globals, never instantiates ``pyfcg.Grammar`` (its ``__init__`` registers through the
globals) and deep-copies pyfcg's default grammar settings (``Grammar`` aliases them by reference). From pyfcg it uses
the ``Construction`` objects it is given and two pure helpers.

DETERMINISM. Shuffle is off (``shuffle-cxns-before-application`` false, re-asserted after every clear) and the engine's
id counters are reset before every comprehension, so the same text on the same constructions gives the same meaning,
variable names included. An instance lock makes reset + comprehend + status, and clear + re-add, atomic per engine.
Determinism wins over availability: a transport failure (timeout, refused or broken connection, oversized answer), a
dead process, or a ``load`` / ``add`` that fails part-way tears the engine down (it is dirty: its grammar is no longer
known). The next call restarts it and reloads the last COMPLETE construction set; ``comprehend`` is refused until a
``load`` has completed. An engine that answers with an error status is not dirty (the request did not change it).

READINGS. ``comprehend`` reads through ``/fcg-comprehend-all`` with the search bounded at its first solution (an
``:around`` installed on this engine only): that is the solution ``/fcg-comprehend`` returns (best-first, same search,
probed with competing roots in both add orders), and the route also names the applied constructions. The same
``:around`` keeps the solution node, and one REPL call then reads its status: ``Reading.complete`` is True only when
the node passed the grammar's goal tests (FCG's ``succeeded`` status: no construction left to apply and no form left
in the root). A failed search still hands back its partial meaning ("what where does X work?" reads as where-work with
"what " left in the root; a lone verb construction reads as a frame): such a Reading has ``complete=False`` and is
never executed. The same status read gives the text still in the root (``Reading.left`` and ``left_spans``) and
every slot unit's boundaries under the unit holding it (``Reading.slots``). ``comprehend_all(text, n)`` asks for up to
``n`` analyses through the Lisp channel (the HTTP route ignores ``n``): the first solution, then the same search
continued solution by solution (``next-cip-solution`` on its processor) without frame parts (no complete analysis
holds one), keeping one node per root construction until ``n`` roots, the search runs out, ``ALL_SOLUTIONS_CAP`` more
solutions or ``deadline_ms`` (checked at every node of the continued search, which then ends inside the engine): the
competing analyses, counted by root, so a root's filler paths ("the report" through np-the and np-bare in each of two
slots is four solutions) never crowd another root out. FRAME PARTS (``part``-labelled constructions) are searched
linearly: a node gets at most one part-created child (``_FRAME_PARTS_LISP``), so a message's parts apply one after
another, never in every order and subset (ten clauses with a verb each were 530 nodes and a second a read). A slot
filler must consume its slot's text (a ``.{1,n}`` sequence over the slot's boundaries): a filler that only binds the
boundaries leaves the slot text in the root, and no slot reading could ever be complete.

THE UNIT GRAMMAR. ``load(cxns, tiers=...)`` loads the CLEVR-style unit grammar into
its own hashed grammar (``UNIT_GRAMMAR_ID``) beside the text grammar above, and ``comprehend_units`` reads one turn's
units with it. The search is pinned (``unit_configuration``): depth-first, label tiers in order with the ``lex`` tier
applied sequentially, trigger-hashed supply, and the kernel's own Lisp extensions
(``data/runtime-grammar/fcg/kernel-ext.lisp``, sent with every unit grammar the engine is given and pinned to the FCG Go
image ``KERNEL_EXT_IMAGE``: ``load(tiers=)`` refuses any other fcg-go pin before it touches the engine, and the text
grammar never needs them, so it boots on any pinned image): an in-engine budget (``node_cap`` nodes, ``deadline_ms``
per request) that empties the search queue when it trips, a canonical order of independent applications, the trigger
hash, and the no-strings-in-root goal test. A turn with a word no construction can consume is refused before any
search, the word named (``Reading.refused``). The comprehension request waits ``deadline_ms`` + 5 s; past that the
search ran away: the engine is killed by its pid, restarted with both grammars reloaded, and ``Runaway`` is raised.
"""

from __future__ import annotations

import contextlib
import copy
import http.client
import json
import os
import signal
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from memory_kernel.pipeline.engine import (
    BODY_CAP,
    DEAD_PARSER,
    PORT_RANGE,
    PROTECTED_PORTS,
    RESET_IDS_FORM,
    Engine,
    EngineConfig,
    Stop,
    port_listening,
    proc_identity,
    proc_state,
    same_process,
)
from memory_kernel.runtime.constructions import ALTERNATE, spec_id
from memory_kernel.runtime.normalise import SENTINEL_END, SENTINEL_START

if TYPE_CHECKING:
    import pyfcg

DEFAULT_PORT = 9711
PORT_ENV = "MK_RUNTIME_FCG_PORT"
GRAMMAR_ID = "mk-runtime-grammar"
#: the runtime grammar's feature types on top of pyfcg's defaults: slot anchors are offsets (``boundaries``)
FEATURE_TYPES: list[list[str]] = [["boundaries", "sequence"]]
#: ``<data_dir>/runtime-fcg/<port>``: the default ``out`` (the capture slot's orphan scan reads the state files here)
RUNTIME_DIR = "runtime-fcg"
#: ``<out>/engine.json``: the running engine's port and ``/proc`` identity, written once the process is proven ours,
#: removed at stop by the instance that wrote it
STATE_FILE = "engine.json"
#: how long a reclaimed orphan gets after SIGTERM (then SIGKILL) to be gone
ORPHAN_GRACE_S = 5.0
#: ``comprehend-all`` stops at its first solution (an EXPLICIT ``:n`` from a caller still wins) and keeps what it
#: returned (meanings, solution nodes, processor) for the status read that follows under the same lock
_FIRST_SOLUTION_LISP = (
    "(progn"
    " (defparameter fcg-go::*runtime-last* nil)"
    " (defmethod fcg::comprehend-all :around"
    "     ((utterance t) &rest args &key ((:n n) nil n-supplied-p) &allow-other-keys)"
    "   (declare (ignorable n))"
    "   (let ((vals (multiple-value-list"
    "                (if n-supplied-p (call-next-method) (apply #'call-next-method utterance :n 1 args)))))"
    "     (setf fcg-go::*runtime-last* vals)"
    "     (values-list vals)))"
    " t)"
)
#: the text grammar's node test for frame parts (a ``part``-labelled construction: ``frames.frame_parts``, the editor's
#: rows), first in its ``node-tests`` (``FRAME_PARTS_TEST``, set by ``_configure``). A frame part applies wherever its
#: word occurs and binds nothing, so a message with k of them would be searched in every order and subset of them (2^k
#: nodes: ten clauses cost a second a read). By default (``fcg-go::*runtime-parts*`` ``:one-per-node``) a node gets at most one child
#: created by a part, the first one created: the parts still apply one after another down that child, the others
#: rejected at this node apply further down, so the best partial node still holds every part. Bound to ``:none`` (the
#: competing analyses' search, ``_ALL_LISP``) a part-created child is rejected: no complete analysis holds a part (a part
#: inside a slot's text leaves the filler nothing whole to consume). ``fcg-go::*runtime-stop*``, when bound (an internal
#: real time), is a deadline checked at every node: past it the search queue is emptied, as the kernel's ``:mk-budget``
#: does, so the search ends inside one ``next-cip-solution`` call.
_FRAME_PARTS_LISP = (
    "(progn"
    " (defparameter fcg-go::*runtime-parts* :one-per-node)"
    " (defparameter fcg-go::*runtime-stop* nil)"
    " (defun fcg-go::runtime-part-p (c)"
    '   (and c (string-equal (format nil "~a" (or (fcg::attr-val c :label) "")) "part")))'
    " (defun fcg-go::runtime-holds-part-p (node)"
    "   (some #'fcg-go::runtime-part-p (fcg::applied-constructions node)))"
    " (defmethod fcg::cip-node-test ((node fcg::cip-node) (mode (eql :mk-runtime-parts)))"
    "   (cond ((and fcg-go::*runtime-stop* (> (get-internal-real-time) fcg-go::*runtime-stop*))"
    "          (push 'fcg::mk-budget-exhausted (fcg::statuses node))"
    "          (setf (fcg::queue (fcg::cip node)) nil)"
    "          (when (fcg::parent node) (setf (fcg::fully-expanded? (fcg::parent node)) t))"
    "          nil)"
    "         ((not (fcg-go::runtime-part-p (first (fcg::applied-constructions node)))) t)"
    "         ((eq fcg-go::*runtime-parts* :none) nil)"
    "         (t (let ((parent (fcg::parent node)))"
    "              (notany (lambda (s)"
    "                        (and (not (eq s node)) (< (fcg::created-at s) (fcg::created-at node))"
    "                             (fcg-go::runtime-part-p (first (fcg::applied-constructions s)))))"
    "                      (when parent (fcg::children parent)))))))"
    " t)"
)
#: the frame-part node test, put before pyfcg's default node tests of the text grammar (re-asserted after every clear)
FRAME_PARTS_TEST = ":mk-runtime-parts"
_RESET_LISP = f"(progn (setf fcg-go::*runtime-last* nil) {RESET_IDS_FORM})"
#: what one solution node says, as ``info``: ``("complete" | "incomplete" (left right) | "none" pieces slots)`` -- its
#: goal-test status; the boundaries of the one top unit of the analysis (no other unit lists it as a subunit; not the
#: root), else "none"; the ``(string left right)`` sequences still in the root (the text nothing consumed); and per
#: slot unit ``(holder slot left right)``: the unit that lists it as a subunit and the slot unit, FCG's unit names
#: lowercased, with the slot's boundaries. ``BODY`` is replaced by the form that uses ``info``.
_NODE_LISP = """
(labels ((nm (s) (if (symbolp s) (symbol-name s) ""))
         (feat (u name) (second (find name (fcg::unit-body u) :key (lambda (f) (nm (first f))) :test #'string=)))
         (pair (b) (and (consp b) (= (length b) 2) (every #'integerp b)))
         (info (node)
           (let* ((app (fcg::cipn-car node))
                  (units (when app (fcg::left-pole-structure (fcg::car-resulting-cfs app))))
                  (subs (mapcar #'nm (loop for u in units append (copy-list (feat u "SUBUNITS")))))
                  (root (find "ROOT" units :key (lambda (u) (nm (fcg::unit-name u))) :test #'string=))
                  (tops (remove-if (lambda (u) (or (eq u root) (member (nm (fcg::unit-name u)) subs :test #'string=)))
                                   units))
                  (b (when (= (length tops) 1) (feat (first tops) "BOUNDARIES"))))
             (list (if (member "SUCCEEDED" (fcg::statuses node) :key #'nm :test #'string=) "complete" "incomplete")
                   (if (pair b) b "none")
                   (loop for p in (when root (copy-list (feat root "FORM")))
                         when (and (consp p) (string= (nm (first p)) "SEQUENCE") (stringp (second p))
                                   (integerp (third p)) (integerp (fourth p)))
                           collect (list (second p) (third p) (fourth p)))
                   (loop for u in units
                         append (loop for s in (copy-list (feat u "SUBUNITS"))
                                      for su = (find (nm s) units :key (lambda (x) (nm (fcg::unit-name x)))
                                                                  :test #'string=)
                                      for sb = (when su (feat su "BOUNDARIES"))
                                      when (and su (feat su "SLOT") (pair sb))
                                        collect (list (string-downcase (nm (fcg::unit-name u)))
                                                      (string-downcase (nm (fcg::unit-name su)))
                                                      (first sb) (second sb))))))))
  BODY)
"""
#: ``info`` of the kept solution node (``("none")`` when there is none)
_STATUS_LISP = _NODE_LISP.replace(
    "BODY",
    "(let* ((nodes (second fcg-go::*runtime-last*)) (node (if (listp nodes) (first nodes) nodes)))"
    ' (if (null node) (list "none") (info node)))',
)
#: the label a root construction carries (``constructions.compile_spec``: the spec's unit)
ROOT_LABEL = "root"
#: how many solutions past the first ``comprehend_all`` examines at most, however few distinct roots they hold (a root
#: with three np slots over "the ..." text is eight filler paths; three such roots fit)
ALL_SOLUTIONS_CAP = 32
#: up to ``N`` solution nodes of ``TEXT``, one per root construction (its name up to the alternate mark): the first
#: solution (an explicit ``:n 1`` passes the first-solution bound), then, when it succeeded, the same search continued
#: (``next-cip-solution`` on its processor: the solutions ``:n`` would list, in the same order, with the same meanings)
#: without frame parts (the queued nodes holding one are dropped, and ``*runtime-parts*`` ``:none`` rejects every
#: part-created child: no complete analysis holds a part) until ``N`` roots, a node that did not succeed or none (the
#: search ran out), ``CAP`` more solutions or ``DEADLINE`` ms (``*runtime-stop*``: checked at every node of the search,
#: not only between solutions). Per node: its meaning, the applied construction names (application order), their
#: labels, and its ``info``
_ALL_LISP = _NODE_LISP.replace(
    "BODY",
    "(labels ((cxns (node) (reverse (fcg::applied-constructions node)))"
    '         (lab (c) (string-downcase (format nil "~a" (or (fcg::attr-val c :label) ""))))'
    "         (ok (node) (member \"SUCCEEDED\" (fcg::statuses node) :key #'nm :test #'string=))"
    "         (root-of (node)"
    "           (let ((r (loop for c in (cxns node) when (string= (lab c) ROOT_LABEL)"
    "                          return (string-downcase (nm (fcg::name c))))))"
    "             (when r (subseq r 0 (or (search ALTERNATE r) (length r))))))"
    "         (row (node m)"
    "           (list m (mapcar (lambda (c) (string-downcase (nm (fcg::name c)))) (cxns node))"
    "                 (mapcar #'lab (cxns node)) (info node))))"
    " (let* ((vals (multiple-value-list"
    f"         (fcg:comprehend-all TEXT :cxn-inventory fcg-go::{GRAMMAR_ID} :n 1 :silent t)))"
    "        (nodes (second vals))"
    "        (first-node (if (listp nodes) (first nodes) nodes))"
    "        (stop (+ (get-internal-real-time) (round (* DEADLINE internal-time-units-per-second) 1000)))"
    "        (roots (when first-node (list (root-of first-node))))"
    "        (out (when first-node (list (row first-node (first (first vals)))))))"
    "   (when (and first-node (ok first-node))"
    "     (let ((fcg-go::*runtime-parts* :none) (fcg-go::*runtime-stop* stop))"
    "       (setf (fcg::queue (third vals)) (remove-if #'fcg-go::runtime-holds-part-p (fcg::queue (third vals))))"
    "       (loop repeat CAP"
    "             while (and (< (length roots) N) (< (get-internal-real-time) stop))"
    "             do (let ((node (fcg::next-cip-solution (third vals) :notify nil)))"
    "                  (unless (and node (ok node)) (return))"
    "                  (let ((r (root-of node)))"
    "                    (unless (member r roots :test #'equal)"
    "                      (push r roots)"
    "                      (push (row node (fcg::extract-meanings"
    "                                        (fcg::left-pole-structure (fcg::car-resulting-cfs (fcg::cipn-car node)))))"
    "                            out)))))))"
    "   (reverse out)))",
)
#: stripped from both ends of a piece of text still in the root: the joining spaces and the sentinels
_EDGES = f" {SENTINEL_START}{SENTINEL_END}"


def _pieces(raw: Any) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    """The text still in the root, from the status read's ``(string left right)`` sequences: each piece with its
    joining spaces and sentinels stripped (a sentinel-only piece is dropped), and its normalised offsets."""
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for piece in raw if isinstance(raw, list) else ():
        if not (isinstance(piece, list) and len(piece) == 3 and isinstance(piece[0], str)):
            continue
        text, left, right = piece
        if not isinstance(left, int) or not isinstance(right, int):
            continue
        core = text.strip(_EDGES)
        if not core:
            continue
        start = left + len(text) - len(text.lstrip(_EDGES))
        texts.append(core)
        spans.append((start, start + len(core)))
    return tuple(texts), tuple(spans)


def _slot_units(raw: Any) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (s[0], s[1], s[2], s[3])
        for s in (raw if isinstance(raw, list) else ())
        if isinstance(s, list)
        and len(s) == 4
        and all(isinstance(v, str) for v in s[:2])
        and all(isinstance(v, int) for v in s[2:])
    )


def _reading(meaning: list[Any], applied: tuple[str, ...], ms: float, status: list[Any]) -> Reading:
    """A text-grammar ``Reading`` from a meaning, the applied names (application order) and a node's ``info``."""
    span = status[1] if len(status) > 1 and isinstance(status[1], list) else None
    left, left_spans = _pieces(status[2] if len(status) > 2 else None)
    return Reading(
        meaning=meaning,
        applied=applied,
        ms=round(ms, 3),
        complete=status[0] == "complete",
        boundaries=(span[0], span[1]) if span else None,
        left=left,
        left_spans=left_spans,
        slots=_slot_units(status[3] if len(status) > 3 else None),
    )


# -- the unit grammar ---------------------------------------------------------------------------------------------------

UNIT_GRAMMAR_ID = "mk-runtime-units"
#: the unit grammar's feature types on top of pyfcg's defaults (its form holds predicates only, no regex sequences)
UNIT_FEATURE_TYPES: list[list[str]] = [
    ["form", "set-of-predicates"],
    ["meaning", "set-of-predicates"],
    ["args", "sequence"],
    ["subunits", "set"],
    ["footprints", "set"],
    ["superunits", "set"],
    ["bounds", "sequence"],
]
#: the committed tier: applied sequentially (the first application is kept and the node closed)
SEQUENTIAL_TIERS = ("lex",)
#: the open-class and closed-class word classes a unit may carry
WORD_CLASSES = ("noun", "verb", "adj", "num", "prep", "det", "name", "pron")
#: a closed-class literal: its string only, no class
LITERAL = "lit"
#: an ambiguous word: every class it may have, each with its lemma (only the ``alex`` tier consumes it)
AMBIGUOUS = "acat"
#: the lemma predicate of a word class (an ambiguous word's classes keep their lemmas apart); det, name and pron
#: send no lemma
LEMMA_PREDICATES = {"noun": "nlemma", "verb": "vlemma", "adj": "lemma", "num": "lemma", "prep": "lemma"}
#: the search budget (settings: ``MK_RUNTIME_NODE_CAP``, ``MK_RUNTIME_DEADLINE_MS``). The node cap does the refusing;
#: the deadline is a loose backstop, because wall time varies with machine load
DEFAULT_NODE_CAP = 150
DEFAULT_DEADLINE_MS = 1000
NODE_CAP_ENV = "MK_RUNTIME_NODE_CAP"
DEADLINE_ENV = "MK_RUNTIME_DEADLINE_MS"
#: the comprehension request waits the deadline plus this; past it the search ran away (kill by pid, restart)
CLIENT_SLACK_S = 5.0
#: the kernel's Lisp extensions to FCG Go (package data), their version and the image (sha256[:12]) they are pinned to
KERNEL_EXT_FILE = ("data", "runtime-grammar", "fcg", "kernel-ext.lisp")
KERNEL_EXT_VERSION = 1
KERNEL_EXT_IMAGE = "2cc1f08127cf"


def unit_configuration(tiers: Sequence[str], node_cap: int) -> dict[str, Any]:
    """The unit grammar's search, pinned: a heuristic-search
    processor run depth-first with one construction per expansion, label tiers in ``tiers`` order with the ``lex`` tier
    applied sequentially, trigger-hashed supply, the kernel's budget and canonical-order node tests (no duplicate
    check), the no-strings-in-root goal test, raw form with ``meets`` only, shuffle off."""
    return {
        "construction-inventory-processor-mode": ":heuristic-search",
        "search-algorithm": ":depth-first",
        "node-expansion-mode": ":default",
        "cxn-supplier-mode": ":hashed-ordered-by-label",
        "hash-mode": ":mk-trigger",
        "parse-order": list(tiers),
        "cxn-sets-with-sequential-application": [t for t in tiers if t in SEQUENTIAL_TIERS],
        "node-tests": [":mk-budget", ":mk-canonical"],
        "max-nr-of-nodes": node_cap,
        "parse-goal-tests": [
            ":no-applicable-cxns",
            ":mk-no-strings-in-root",
            ":connected-semantic-network",
            ":connected-structure",
        ],
        "de-render-mode": ":de-render-raw",
        "render-mode": ":render-string-meets-precedes",
        "form-predicates": ["meets"],
        "shuffle-cxns-before-application": False,
    }


def kernel_ext_source() -> str:
    """The text of ``kernel-ext.lisp``: one PROGN, evaluated in package FCG, answering T."""
    return resources.files("memory_kernel").joinpath(*KERNEL_EXT_FILE).read_text(encoding="utf-8")


def search_budget(environ: Mapping[str, str] = os.environ) -> tuple[int, int]:
    """``(node_cap, deadline_ms)``: ``MK_RUNTIME_NODE_CAP`` and ``MK_RUNTIME_DEADLINE_MS``, else 150 and 1000."""

    def setting(name: str, default: int) -> int:
        raw = environ.get(name)
        return int(raw) if raw else default

    return setting(NODE_CAP_ENV, DEFAULT_NODE_CAP), setting(DEADLINE_ENV, DEFAULT_DEADLINE_MS)


class RuntimeEngineError(RuntimeError):
    """The runtime engine is not set up, not running, died, or refused a request (the one error callers see)."""


class PortBusy(RuntimeEngineError):
    """The engine's port is held by something this instance may not reclaim (a foreign process, or the engine of an
    owner that is still alive): ``mk serve`` tries its next free port."""


class _Refused(RuntimeEngineError):
    """The engine answered with an error status: the request reached it and did not change it (not dirty)."""


class _Timeout(RuntimeEngineError):
    """A request got no answer within its timeout."""


class Runaway(RuntimeEngineError):
    """A unit comprehension got no answer within the deadline plus ``CLIENT_SLACK_S``: the search ran away inside the
    engine. The engine was killed by its pid and restarted (both grammars reloaded) before this was raised."""


@dataclass(frozen=True)
class Reading:
    #: the engine's meaning predicates; slot offsets (``name-span ?x L R``) arrive as integers
    meaning: list[list[str | int]]
    #: construction names in application order (the engine lists the most recent first)
    applied: tuple[str, ...]
    #: wall time of the comprehension request
    ms: float
    #: the solution node passed the grammar's goal tests; False = a partial analysis, never executed
    complete: bool
    #: text grammar: normalised offsets the analysis's one top unit spans (for the trace), None when there is no
    #: single top unit; always None for the unit grammar
    boundaries: tuple[int, int] | None = None
    #: unit grammar: the search nodes created (rejected ones included); 0 for a static refusal
    nodes: int = 0
    #: unit grammar: the static refusal, naming every word no construction can consume (no search ran), else None
    refused: str | None = None
    #: unit grammar: the budget limit that ended the search, ``"nodes"`` (the node cap) or ``"time"`` (the deadline)
    budget: str | None = None
    #: the turn's strings still in the root (words no construction consumed in this analysis); text grammar: the
    #: normalised pieces, their joining spaces and sentinels stripped (empty for a complete reading)
    left: tuple[str, ...] = ()
    #: text grammar: the normalised offsets of each piece of ``left``
    left_spans: tuple[tuple[int, int], ...] = ()
    #: text grammar: every slot unit of the analysis as ``(holder, slot, left, right)``: the unit that lists it as a
    #: subunit and the slot unit (FCG's unit names, lowercased: ``"card.take.0-unit-1"``, ``"slot-ticket-1"``) and
    #: the slot's normalised boundaries
    slots: tuple[tuple[str, str, int, int], ...] = ()


def runtime_port(environ: Mapping[str, str] = os.environ) -> int:
    """``MK_RUNTIME_FCG_PORT``, else 9711."""
    raw = environ.get(PORT_ENV)
    return int(raw) if raw else DEFAULT_PORT


def runtime_binary(data_dir: Path) -> Path | None:
    """The pinned FCG Go binary the ``mk setup`` receipt of ``data_dir`` records, None when it has none on disk."""
    from memory_kernel import setup_runtime

    return setup_runtime.managed_path(data_dir, "fcg-go")


def _unsendable(ch: str) -> bool:
    """FCG Go refuses a surrogate escape, and would count a non-BMP character as two offsets."""
    code = ord(ch)
    return code > 0xFFFF or 0xD800 <= code <= 0xDFFF


def _guarded(body: str) -> str:
    """A REPL form that answers ``("ERROR" message)`` instead of failing the request."""
    return f'(handler-case (progn {body}\n) (error (e) (list "ERROR" (format nil "~a" e))))'


def _lisp_string(text: str) -> str:
    """A Lisp string literal: only backslash and double quote are escaped (BMP characters pass through)."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _quoted(value: object) -> str | None:
    """The string a pyfcg JSON value names when it is a quoted string (``'"do"'`` is the Lisp string do), else None."""
    if not isinstance(value, str) or len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return None
    try:
        text = json.loads(value)
    except ValueError:
        return None
    return text if isinstance(text, str) else None


_UNIT_FIELDS = ("string", "cat", "lemmas", "start", "end")


@dataclass(frozen=True)
class _Word:
    string: str
    cat: str
    lemmas: tuple  # str per lemma; (class, lemma) pairs for an ambiguous word
    start: int
    end: int


def _words(units: Iterable[Any]) -> list[_Word]:
    """Validate a turn's units (``comprehend_units`` documents the shape); a malformed unit is a ValueError."""
    words = []
    for k, unit in enumerate(units, 1):
        try:
            fields = [unit[f] if isinstance(unit, Mapping) else getattr(unit, f) for f in _UNIT_FIELDS]
        except (KeyError, AttributeError) as exc:
            raise ValueError(f"unit {k} lacks {exc}: a unit has {', '.join(_UNIT_FIELDS)}") from None
        string, cat, lemmas, start, end = fields
        if not isinstance(string, str) or not string:
            raise ValueError(f"unit {k}: string must be a non-empty str, got {string!r}")
        if cat not in (*WORD_CLASSES, LITERAL, AMBIGUOUS):
            raise ValueError(f"unit {k} {string!r}: cat {cat!r} is none of {(*WORD_CLASSES, LITERAL, AMBIGUOUS)}")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)) or not 0 <= start <= end:
            raise ValueError(
                f"unit {k} {string!r}: offsets must be ints with 0 <= start <= end, got {start!r}, {end!r}"
            )
        lemmas = tuple(lemmas or ())
        if cat == AMBIGUOUS:
            if not lemmas or any(
                not isinstance(p, Sequence) or isinstance(p, str) or len(p) != 2 or p[0] not in WORD_CLASSES
                for p in lemmas
            ):
                raise ValueError(f"unit {k} {string!r}: an acat unit's lemmas are (class, lemma) pairs, got {lemmas!r}")
            lemmas = tuple((str(c), lemma) for c, lemma in lemmas)
        if any(not isinstance(lemma, str) for lemma in (p[1] if cat == AMBIGUOUS else p for p in lemmas)):
            raise ValueError(f"unit {k} {string!r}: lemmas must be strings, got {lemmas!r}")
        words.append(_Word(string, cat, lemmas, start, end))
    if not words:
        raise ValueError("no units: a turn has at least one word")
    return words


def _word_unsendable(word: _Word) -> bool:
    texts = [word.string, *(p[1] if word.cat == AMBIGUOUS else p for p in word.lemmas)]
    return any(_unsendable(ch) for text in texts for ch in text)


def _raw_form(words: Sequence[_Word]) -> str:
    """The turn as FCG's raw form: per unit ``(string w s) (span w start end)``, its class (``cat``, or ``acat`` per
    class of an ambiguous word) and lemma predicates; ``(meets w w')`` between neighbours, no ``precedes``. Units are
    named w1 .. wn in token order (the canonical-order node test reads the token index from the name)."""
    preds: list[str] = []
    for k, word in enumerate(words, 1):
        name = f"w{k}"
        preds += [f"(string {name} {_lisp_string(word.string)})", f"(span {name} {word.start} {word.end})"]
        if word.cat == AMBIGUOUS:
            seen: set[str] = set()
            for cls, lemma in word.lemmas:
                if cls not in seen:
                    seen.add(cls)
                    preds.append(f"(acat {name} {_lisp_string(cls)})")
                if lemma and cls in LEMMA_PREDICATES:
                    preds.append(f"({LEMMA_PREDICATES[cls]} {name} {_lisp_string(lemma)})")
        elif word.cat in WORD_CLASSES:
            preds.append(f"(cat {name} {_lisp_string(word.cat)})")
            if word.cat in LEMMA_PREDICATES:
                preds += [f"({LEMMA_PREDICATES[word.cat]} {name} {_lisp_string(m)})" for m in word.lemmas if m]
    preds += [f"(meets w{k} w{k + 1})" for k in range(1, len(words))]
    return "(" + " ".join(preds) + ")"


def _unit_payload(cxn: Any, tiers: tuple[str, ...]) -> dict[str, Any]:
    """One compiled unit construction as the ``/fcg-add-cxn`` payload fields, refused (ValueError) when its label is
    none of the tiers (it would never be offered) or it carries no triggers (it would file under the NIL key)."""
    if isinstance(cxn, Mapping):
        name, cond = cxn.get("name"), cxn.get("conditional-pole")
        contrib, attrs, types = (
            cxn.get("contributing-pole") or [],
            cxn.get("attributes") or {},
            cxn.get("feature-types"),
        )
    else:
        name, cond, contrib = cxn.name, cxn.conditional_pole, cxn.contributing_pole
        attrs, types = cxn.attributes or {}, cxn.feature_types
    if not isinstance(name, str) or not name or not isinstance(cond, list):
        raise ValueError(f"a unit construction needs a name and a conditional pole, got {name!r}")
    if attrs.get("label") not in tiers:
        raise ValueError(f"construction {name}: label {attrs.get('label')!r} is none of the tiers {tiers}")
    triggers = attrs.get("triggers")
    if not isinstance(triggers, list) or not triggers or any(not _quoted(t) for t in triggers):
        raise ValueError(
            f"construction {name} has no triggers: it needs a non-empty list of quoted strings derived from its own "
            f"conditions (no construction files under the NIL hash key), got {triggers!r}"
        )
    return {
        "name": name,
        "conditional-pole": cond,
        "contributing-pole": list(contrib),
        "attributes": dict(attrs),
        "feature-types": list(types or []),
    }


@dataclass(frozen=True)
class _UnitGrammar:
    """The last unit grammar loaded in full, and what its constructions can consume: the literal strings and the
    word classes (``cat:X`` / ``acat:X``) their form predicates name."""

    payloads: tuple[dict[str, Any], ...]
    tiers: tuple[str, ...]
    literals: frozenset[str]
    classes: frozenset[str]

    @classmethod
    def of(cls, payloads: tuple[dict[str, Any], ...], tiers: tuple[str, ...]) -> _UnitGrammar:
        literals, classes = set(), set()
        for payload in payloads:
            for unit in payload["conditional-pole"]:
                for pole in unit[1:] if isinstance(unit, list) else ():
                    for pred in (pole.get("#form") or ()) if isinstance(pole, Mapping) else ():
                        value = _quoted(pred[2]) if isinstance(pred, list) and len(pred) == 3 else None
                        if value is not None and pred[0] == "string":
                            literals.add(value)
                        elif value is not None and pred[0] in ("cat", "acat"):
                            classes.add(f"{pred[0]}:{value}")
        return cls(payloads, tiers, frozenset(literals), frozenset(classes))

    def consumes(self, word: _Word) -> bool:
        """Some construction can consume ``word``: its string is a literal, or a class it carries is consumed."""
        if word.string in self.literals:
            return True
        if word.cat == AMBIGUOUS:
            return any(f"acat:{c}" in self.classes for c, _ in word.lemmas)
        return word.cat in WORD_CLASSES and f"cat:{word.cat}" in self.classes


def _atom(value: Any) -> Any:
    """A meaning argument from the kernel Lisp: ``("s" x)`` is the string x; numbers and symbol names pass."""
    return value[1] if isinstance(value, list) and len(value) == 2 and value[0] == "s" else value


class RuntimeEngine:
    """The runtime grammar's FCG Go process and its one grammar (``GRAMMAR_ID``).

    ``out`` (default ``<data_dir>/runtime-fcg/<port>``) holds the engine's log and ``engine.json``. ``request_timeout``
    bounds one request; ``ready_timeout`` bounds the wait for a fresh process to answer. ``node_cap`` and
    ``deadline_ms`` are the unit grammar's search budget (None: ``search_budget()``, by default 150 nodes and 1000 ms);
    the node cap is applied at ``load``, the deadline at every ``comprehend_units``."""

    def __init__(
        self,
        port: int | None = None,
        data_dir: Path | None = None,
        *,
        out: Path | None = None,
        request_timeout: float = 5.0,
        ready_timeout: float = 30.0,
        node_cap: int | None = None,
        deadline_ms: int | None = None,
    ) -> None:
        from memory_kernel import setup_runtime

        self.port = runtime_port() if port is None else port
        if self.port in PROTECTED_PORTS or self.port not in PORT_RANGE:
            raise ValueError(f"runtime engine port {self.port}: must be a private 97xx port, never {PROTECTED_PORTS}")
        default_cap, default_deadline = search_budget()
        self.node_cap = default_cap if node_cap is None else node_cap
        self.deadline_ms = default_deadline if deadline_ms is None else deadline_ms
        for setting, value in (("node_cap", self.node_cap), ("deadline_ms", self.deadline_ms)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"runtime engine {setting} must be a positive integer, got {value!r}")
        self.data_dir = Path(data_dir) if data_dir is not None else setup_runtime.resolve_data_dir(None, os.environ)
        self.out = Path(out) if out is not None else self.data_dir / RUNTIME_DIR / str(self.port)
        self.request_timeout = request_timeout
        self.ready_timeout = ready_timeout
        self._engine: Engine | None = None
        self._wanted = False  # start() was called and stop() was not: a dirty engine restarts on the next call
        self._complete: list[pyfcg.Construction] | None = None  # the last construction set loaded in full
        self._units: _UnitGrammar | None = None  # the last unit grammar loaded in full
        self._feature_types: list[list[str]] = []
        self._unit_feature_types: list[list[str]] = []
        self._state_pid: int | None = None  # the pid this instance wrote into engine.json
        self._lock = threading.RLock()

    # -- lifecycle ----------------------------------------------------------------------------------------------------

    @property
    def pid(self) -> int | None:
        engine = self._engine
        return engine.process.pid if engine is not None and engine.process is not None else None

    def start(self) -> None:
        """Reclaim an orphan this ``out`` names, launch the pinned binary on ``port``, then register the grammar
        (feature types, shuffle off, first-solution search). Stops the process again when any step fails."""
        with self._lock:
            if self._engine is not None:
                raise RuntimeEngineError(f"runtime engine already started (pid {self.pid})")
            self._boot()
            self._wanted = True

    def stop(self) -> dict | None:
        """SIGTERM, then SIGKILL, the engine's pid (``Engine.teardown``); never a pattern kill. Idempotent: None when
        nothing runs, else the teardown report. A stopped engine is not restarted by a later call."""
        with self._lock:
            self._wanted = False
            return self._teardown()

    def __enter__(self) -> RuntimeEngine:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _boot(self) -> None:
        from memory_kernel import setup_runtime

        binary = runtime_binary(self.data_dir)
        if binary is None:
            raise RuntimeEngineError(
                f"no FCG Go engine binary under {self.data_dir}: run `mk setup --component fcg-go`"
            )
        pinned = setup_runtime.load_pins()["components"]["fcg-go"]["binary_sha256"]
        cfg = EngineConfig(
            port=self.port,
            binary=binary,
            image=Path(os.devnull),  # no grammar image: the runtime grammar is built by adding constructions
            parser_url=DEAD_PARSER,  # no syntactic parser: runtime constructions match the normalised text only
            binary_sha256=pinned,
            image_sha256=None,
        )
        try:
            if self._reclaim():  # an orphan's threads may hold the port a moment after its leader is a zombie
                deadline = time.monotonic() + ORPHAN_GRACE_S
                while port_listening(self.port) and time.monotonic() < deadline:
                    time.sleep(0.1)
            if port_listening(self.port):
                raise PortBusy(f"runtime engine start failed: port {self.port} already listening")
            self._engine = Engine(cfg, self.out)
            self._engine.launch(on_spawn=self._write_state)
            self._wait_ready()
            self._register()
        except BaseException as exc:
            self._teardown()
            if isinstance(exc, Stop | OSError):
                raise RuntimeEngineError(f"runtime engine start failed: {exc}") from exc
            raise

    def _reclaim(self) -> bool:
        """Kill the engine a dead instance left running, when ``<out>/engine.json`` names it and ``/proc`` proves it
        (same pid, start ticks and exe, this user) and the process that started it (``owner``) is gone: SIGTERM, then
        SIGKILL. Anything else is left alone, a live owner's engine included (the start then raises ``PortBusy``).
        True when it killed an orphan."""
        try:
            state = json.loads((self.out / STATE_FILE).read_text())
        except (OSError, ValueError):
            return False
        if not isinstance(state, dict) or not isinstance(state.get("pid"), int):
            return False
        pid = state["pid"]
        identity = proc_identity(pid)
        if identity is None or identity["uid"] != os.getuid() or not same_process(identity, state):
            return False
        owner = state.get("owner")
        owner_pid = owner.get("pid") if isinstance(owner, dict) else None
        if isinstance(owner_pid, int) and same_process(proc_identity(owner_pid), owner):
            return False  # its owner is alive (another runtime, another mk serve or mk plan): never ours to kill
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, sig)
            deadline = time.monotonic() + ORPHAN_GRACE_S
            while time.monotonic() < deadline:
                if proc_state(pid) in (None, "Z", "X") or not same_process(proc_identity(pid), state):
                    return True
                time.sleep(0.1)
        raise RuntimeEngineError(f"orphan runtime engine pid {pid} survived SIGKILL")

    def _teardown(self) -> dict | None:
        engine, self._engine = self._engine, None
        if engine is None:
            return None
        try:
            return engine.teardown()
        finally:
            self._unlink_state()

    def _write_state(self) -> None:
        assert self._engine is not None and self._engine.process is not None
        pid = self._engine.process.pid
        state = {
            "port": self.port,
            "grammar": GRAMMAR_ID,
            **(proc_identity(pid) or {"pid": pid}),
            "owner": proc_identity(os.getpid()) or {"pid": os.getpid()},  # the process that started it
        }
        tmp = self.out / f".{STATE_FILE}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, self.out / STATE_FILE)
        self._state_pid = pid

    def _unlink_state(self) -> None:
        """Remove ``engine.json`` only when it still names the engine this instance wrote it for."""
        pid, self._state_pid = self._state_pid, None
        if pid is None:
            return
        path = self.out / STATE_FILE
        try:
            named = json.loads(path.read_text()).get("pid")
        except (OSError, ValueError, AttributeError):
            return
        if named == pid:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout
        while True:
            try:
                self._request("GET", "/fcg-test-connection", None, timeout=min(2.0, self.request_timeout))
                return
            except RuntimeEngineError:
                if time.monotonic() > deadline:
                    raise RuntimeEngineError(f"runtime engine on port {self.port} never answered") from None
                time.sleep(0.2)

    def _register(self) -> None:
        from pyfcg import grammar as pyfcg_grammar
        from pyfcg.utils import inherit_feature_types

        self._post(
            "/fcg-initialise-grammar",
            {
                "hashed": False,
                "id": GRAMMAR_ID,
                "feature-types": copy.deepcopy(pyfcg_grammar.DEFAULT_FEATURE_TYPES),
                "hierarchy-features": copy.deepcopy(pyfcg_grammar.DEFAULT_HIERARCHY_FEATURES),
                "configuration": copy.deepcopy(pyfcg_grammar.DEFAULT_GRAMMAR_CONFIGURATION),
                "visualization-configuration": copy.deepcopy(pyfcg_grammar.DEFAULT_GRAMMAR_VISUALIZATION_CONFIGURATION),
                "categorial-network": {},
                "package": None,
            },
        )
        self._feature_types = inherit_feature_types(copy.deepcopy(pyfcg_grammar.DEFAULT_FEATURE_TYPES), FEATURE_TYPES)
        self._post("/fcg-set-feature-types", {"feature-types": self._feature_types, "grammar": GRAMMAR_ID})
        if self._lisp(_FIRST_SOLUTION_LISP) is not True:
            raise RuntimeEngineError("runtime engine refused the first-solution search bound")
        if self._lisp(_FRAME_PARTS_LISP) is not True:
            raise RuntimeEngineError("runtime engine refused the frame-part node test")
        self._configure()

    @staticmethod
    def _check_kernel_ext_pin() -> None:
        """A unit grammar needs ``kernel-ext.lisp``, tested against the FCG Go image ``KERNEL_EXT_IMAGE`` only: refuse
        any other fcg-go pin (launch proves the binary's sha256 is the pinned one)."""
        from memory_kernel import setup_runtime

        pinned = setup_runtime.load_pins()["components"]["fcg-go"]["binary_sha256"]
        if not pinned.startswith(KERNEL_EXT_IMAGE):
            raise RuntimeEngineError(
                f"kernel-ext.lisp is pinned to FCG Go {KERNEL_EXT_IMAGE}, the engine pin is {pinned[:12]}: "
                "test the kernel extensions against that image and pin it"
            )

    def _install_kernel_ext(self) -> None:
        """Send ``kernel-ext.lisp`` (it must answer T) and read its version and image pin back."""
        answer = self._lisp(_guarded(kernel_ext_source()), package="fcg")
        if answer is not True:
            raise RuntimeEngineError(f"runtime engine refused kernel-ext.lisp: {str(answer)[:500]}")
        info = self._lisp("(fcg::mk-kernel-ext)", package="fcg")
        if info != ["mk-kernel-ext", KERNEL_EXT_VERSION, KERNEL_EXT_IMAGE]:
            raise RuntimeEngineError(f"kernel-ext.lisp answered {info!r}, expected version {KERNEL_EXT_VERSION}")

    def _configure(self) -> None:
        """Shuffle off, and the frame-part node test (``_FRAME_PARTS_LISP``) before pyfcg's default node tests."""
        from pyfcg import grammar as pyfcg_grammar

        self._post(
            "/fcg-set-grammar-configuration",
            {"key": "shuffle-cxns-before-application", "value": False, "grammar": GRAMMAR_ID},
        )
        defaults = [t for t in pyfcg_grammar.DEFAULT_GRAMMAR_CONFIGURATION["node-tests"] if t != FRAME_PARTS_TEST]
        self._post(
            "/fcg-set-grammar-configuration",
            {"key": "node-tests", "value": [FRAME_PARTS_TEST, *defaults], "grammar": GRAMMAR_ID},
        )

    def _live(self) -> None:
        """The engine is up for a call: refuse when stopped, notice death (tear down, raise), restart a dirty engine
        and reload the last complete construction set."""
        if not self._wanted:
            raise RuntimeEngineError("runtime engine is not running: start() it first")
        engine = self._engine
        if engine is not None:
            rc = engine.process.poll() if engine.process is not None else None
            if rc is None:
                return
            self._teardown()
            raise RuntimeEngineError(f"engine died rc={rc}")
        self._boot()
        self._reload()

    def _reload(self) -> None:
        """Reload the last complete construction set of each grammar into a fresh engine."""
        if self._complete is not None:
            self._send_load(self._complete)
        if self._units is not None:
            self._send_units(self._units)

    # -- grammar ------------------------------------------------------------------------------------------------------

    def load(self, cxns: Sequence[Any], *, tiers: Sequence[str] | None = None) -> None:
        """Clear the grammar and add ``cxns`` in the given order (the caller orders them by ascending precedence).
        A load that fails part-way tears the engine down; the last complete set is reloaded on the next call.

        Without ``tiers``: the text grammar (``pyfcg.Construction`` objects; ``comprehend``). With ``tiers``: the unit
        grammar, searched as ``unit_configuration`` pins it with ``tiers`` as its label order (``comprehend_units``),
        with the kernel extensions installed first; refused, the engine untouched, unless the fcg-go pin is
        ``KERNEL_EXT_IMAGE``.
        Each unit construction is a compiled construction: a mapping with ``name``, ``conditional-pole``,
        ``contributing-pole``, ``attributes`` and optionally ``feature-types`` (the ``/fcg-add-cxn`` fields), or an
        object with pyfcg's attribute names. Its ``attributes`` carry ``label`` (one of ``tiers``) and ``triggers``,
        a non-empty list of quoted strings derived from its own conditions (``'"where"'``, ``'"cat:noun"'``); the load
        refuses one without, before anything reaches the engine. The words the static refusal accepts are read off
        the same constructions: every quoted ``string`` literal, and every ``cat`` / ``acat`` class, of their forms."""
        if tiers is not None:
            self._load_units(cxns, tiers)
            return
        with self._lock:
            self._live()
            cxns = list(cxns)
            self._send_load(cxns)
            self._complete = cxns

    def add(self, cxn: pyfcg.Construction) -> None:
        """Add one construction to the live grammar (after a completed ``load``). A failed add tears it down."""
        with self._lock:
            if self._complete is None:
                raise RuntimeEngineError("no construction set loaded: load() before add()")
            self._live()
            try:
                self._add(cxn)
            except BaseException:
                self._teardown()
                raise
            self._complete.append(cxn)

    def _send_load(self, cxns: Sequence[pyfcg.Construction]) -> None:
        try:
            self._post("/fcg-clear-grammar", {"grammar": GRAMMAR_ID})
            self._configure()
            for cxn in cxns:
                self._add(cxn)
        except BaseException:
            self._teardown()
            raise

    def _add(self, cxn: pyfcg.Construction) -> None:
        from pyfcg.utils import inherit_feature_types

        self._post(
            "/fcg-add-cxn",
            {
                "cxn": {
                    "grammar-id": GRAMMAR_ID,
                    "name": cxn.name,
                    "conditional-pole": cxn.conditional_pole,
                    "contributing-pole": cxn.contributing_pole,
                    "attributes": cxn.attributes,
                    "feature-types": inherit_feature_types(self._feature_types, cxn.feature_types),
                }
            },
        )

    def _load_units(self, cxns: Sequence[Any], tiers: Sequence[str]) -> None:
        tiers = tuple(tiers)
        if not tiers or len(set(tiers)) != len(tiers) or not all(isinstance(t, str) and t for t in tiers):
            raise ValueError(f"tiers must be distinct non-empty labels, got {tiers!r}")
        payloads = tuple(_unit_payload(cxn, tiers) for cxn in cxns)
        grammar = _UnitGrammar.of(payloads, tiers)
        self._check_kernel_ext_pin()  # before the engine is touched: a refused unit grammar leaves the text one running
        with self._lock:
            self._live()
            self._send_units(grammar)
            self._units = grammar

    def _send_units(self, grammar: _UnitGrammar) -> None:
        """Install the kernel extensions (every send: a restarted engine has none) and add the unit grammar."""
        try:
            self._install_kernel_ext()
            self._initialise_units(grammar.tiers)
            for payload in grammar.payloads:
                self._add_unit(payload)
        except BaseException:
            self._teardown()
            raise

    def _initialise_units(self, tiers: tuple[str, ...]) -> None:
        """A fresh hashed grammar ``UNIT_GRAMMAR_ID`` with the pinned search. Every key is set again after the
        initialisation: a hashed construction set's init forces its own supplier and hash mode."""
        from pyfcg import grammar as pyfcg_grammar
        from pyfcg.utils import inherit_feature_types

        config = unit_configuration(tiers, self.node_cap)
        configuration = copy.deepcopy(pyfcg_grammar.DEFAULT_GRAMMAR_CONFIGURATION)
        configuration.update(config)
        types = inherit_feature_types(copy.deepcopy(pyfcg_grammar.DEFAULT_FEATURE_TYPES), UNIT_FEATURE_TYPES)
        self._post(
            "/fcg-initialise-grammar",
            {
                "hashed": True,
                "id": UNIT_GRAMMAR_ID,
                "feature-types": types,
                "hierarchy-features": ["subunits"],
                "configuration": configuration,
                "visualization-configuration": copy.deepcopy(pyfcg_grammar.DEFAULT_GRAMMAR_VISUALIZATION_CONFIGURATION),
                "categorial-network": {},
                "package": None,
            },
        )
        self._post("/fcg-set-feature-types", {"feature-types": types, "grammar": UNIT_GRAMMAR_ID})
        for key, value in config.items():
            self._post("/fcg-set-grammar-configuration", {"key": key, "value": value, "grammar": UNIT_GRAMMAR_ID})
        self._unit_feature_types = types

    def _add_unit(self, payload: Mapping[str, Any]) -> None:
        from pyfcg.utils import inherit_feature_types

        self._post(
            "/fcg-add-cxn",
            {
                "cxn": {
                    "grammar-id": UNIT_GRAMMAR_ID,
                    "name": payload["name"],
                    "conditional-pole": payload["conditional-pole"],
                    "contributing-pole": payload["contributing-pole"],
                    "attributes": payload["attributes"],
                    "feature-types": inherit_feature_types(
                        self._unit_feature_types, payload.get("feature-types") or []
                    ),
                }
            },
        )

    # -- comprehension ------------------------------------------------------------------------------------------------

    def comprehend(self, text: str) -> Reading | None:
        """The first solution's meaning, applied constructions and goal-test status for ``text`` (the normalised
        text), None when the engine finds no meaning at all. Refused until a ``load`` has completed."""
        if any(_unsendable(ch) for ch in text):
            raise ValueError(
                "text outside the Basic Multilingual Plane or with a lone surrogate: comprehend the normaliser's text"
            )
        with self._lock:
            if self._complete is None:
                raise RuntimeEngineError("no construction set loaded: comprehend is refused until a load completes")
            self._live()
            try:
                if self._lisp(_RESET_LISP) != 0:
                    raise RuntimeEngineError("runtime engine id reset did not answer 0")
                started = time.perf_counter()
                result = self._post("/fcg-comprehend-all", {"utterance": text, "grammar": GRAMMAR_ID, "package": None})
                ms = (time.perf_counter() - started) * 1000
                status = self._lisp(_STATUS_LISP)
            except _Refused:
                raise
            except BaseException:
                self._teardown()
                raise
        meanings = result.get("meanings") if isinstance(result, dict) else None
        if not meanings or not meanings[0]:
            return None
        applied = (result.get("applied-cxns") or [None])[0] or []
        if not isinstance(status, list) or not status or status[0] not in ("complete", "incomplete"):
            raise RuntimeEngineError(f"runtime engine status read answered {str(status)[:200]}")
        return _reading(meanings[0], tuple(reversed(applied)), ms, status)

    def comprehend_all(self, text: str, n: int = 3) -> list[Reading]:
        """Up to ``n`` analyses of ``text`` (the normalised text), one per root construction: the first solution, then
        the competing ones the first-solution search stops before, the same search continued through the engine's
        Lisp (its HTTP route ignores ``n``), after the id reset, under the engine lock. Analyses with the same root
        construction (an alternate is its spec) are one, the first kept ("the office" through np-the and through
        np-bare is one look), and they never count toward ``n``: the search goes on past a root's filler paths until
        ``n`` roots, the search runs out, ``ALL_SOLUTIONS_CAP`` more solutions were examined or ``deadline_ms`` passed
        (checked at every node, so the search ends inside the engine). The continued search leaves frame parts out (no
        complete analysis holds one): verbs in a slot's text cost it nothing.
        One ``Reading`` per kept node: its meaning, applied names (application order, as ``comprehend``),
        ``complete`` (the node's goal-test status), ``left`` and ``slots``; ``ms`` is the whole request's. A reading
        with no complete analysis gives its one partial node; no meaning at all gives []. Refused until a ``load``
        completed."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        if any(_unsendable(ch) for ch in text):
            raise ValueError(
                "text outside the Basic Multilingual Plane or with a lone surrogate: comprehend the normaliser's text"
            )
        code = _guarded(
            _ALL_LISP.replace("ROOT_LABEL", _lisp_string(ROOT_LABEL), 1)
            .replace("ALTERNATE", _lisp_string(ALTERNATE), 1)
            .replace("DEADLINE", str(int(self.deadline_ms)), 1)
            .replace("repeat CAP", f"repeat {ALL_SOLUTIONS_CAP}", 1)
            .replace("roots) N)", f"roots) {n})", 1)
            .replace("TEXT", _lisp_string(text), 1)  # last: the turn's text is never searched for a placeholder
        )
        with self._lock:
            if self._complete is None:
                raise RuntimeEngineError("no construction set loaded: comprehend is refused until a load completes")
            self._live()
            try:
                if self._lisp(_RESET_LISP) != 0:
                    raise RuntimeEngineError("runtime engine id reset did not answer 0")
                started = time.perf_counter()
                answer = self._lisp(code)
                ms = (time.perf_counter() - started) * 1000
            except _Refused:
                raise
            except BaseException:
                self._teardown()
                raise
        if isinstance(answer, list) and len(answer) == 2 and answer[0] == "ERROR":
            raise _Refused(f"runtime engine comprehend-all: {answer[1]}")
        if not isinstance(answer, list):
            raise RuntimeEngineError(f"runtime engine comprehend-all answered {str(answer)[:300]}")
        readings: list[Reading] = []
        roots: set[str | None] = set()
        for row in answer:
            if not (isinstance(row, list) and len(row) == 4 and isinstance(row[3], list) and row[3]):
                raise RuntimeEngineError(f"runtime engine comprehend-all answered {str(row)[:300]}")
            meaning, applied, labels, status = row
            if not meaning or status[0] not in ("complete", "incomplete"):
                continue
            root = next((spec_id(a) for a, label in zip(applied, labels, strict=False) if label == ROOT_LABEL), None)
            if root in roots:
                continue
            roots.add(root)
            readings.append(_reading(meaning, tuple(applied), ms, status))
        return readings

    def comprehend_units(self, units: Iterable[Any]) -> Reading:
        """Read one turn's units with the unit grammar under the pinned search and its budget.

        A unit is a mapping or an object with these fields, in turn order:

        - ``string``: the word as the grammar's literals match it (a name run is one unit);
        - ``cat``: a word class (``noun``, ``verb``, ``adj``, ``num``, ``prep``, ``det``, ``name``, ``pron``),
          ``lit`` (a closed-class literal: its string only, no class) or ``acat`` (an ambiguous word);
        - ``lemmas``: the lemma strings of a word class (sent for noun, verb, adj, num and prep; det, name, pron
          and lit send none); for ``acat``, ``(class, lemma)`` pairs, one per class the word may have;
        - ``start``, ``end``: character offsets into the original turn (the raw form's ``span``).

        The engine sees the units as w1 .. wn: a unit's own id is not sent. A turn with a word no construction can
        consume (its string is no literal and no class it carries is consumed, or it cannot be sent: outside the
        Basic Multilingual Plane) is refused before any search: ``refused`` names every such word, ``nodes`` is 0.
        Otherwise the search runs inside the engine under the node cap and the deadline; ``complete`` is the goal
        tests (every word consumed, network and structure connected), ``budget`` names a limit that ended the search,
        and a partial meaning is never to be executed. No answer within the deadline + 5 s: ``Runaway`` (the engine
        was killed by its pid and restarted). Refused until a ``load`` with ``tiers`` has completed."""
        words = _words(units)
        with self._lock:
            grammar = self._units
            if grammar is None:
                raise RuntimeEngineError(
                    "no unit grammar loaded: comprehend_units is refused until load(tiers=) completes"
                )
            self._live()
            dead = [w.string for w in words if _word_unsendable(w) or not grammar.consumes(w)]
            if dead:
                refused = "no construction can consume " + ", ".join(json.dumps(s, ensure_ascii=False) for s in dead)
                return Reading(meaning=[], applied=(), ms=0.0, complete=False, refused=refused)
            code = _guarded(
                f"(fcg::mk-comprehend-units fcg-go::{UNIT_GRAMMAR_ID} '{_raw_form(words)} {self.deadline_ms})"
            )
            timeout = self._client_timeout()
            try:
                started = time.perf_counter()
                answer = self._request(
                    "POST", "/fcg-read-eval-print", {"lisp-code": code, "package": "fcg"}, timeout=timeout
                )
                ms = (time.perf_counter() - started) * 1000
            except _Refused:
                raise
            except _Timeout as exc:
                self._restart_runaway(timeout, exc)
            except BaseException:
                self._teardown()
                raise
        if isinstance(answer, list) and len(answer) == 2 and answer[0] == "ERROR":
            raise _Refused(f"runtime engine unit comprehension: {answer[1]}")
        if not (isinstance(answer, list) and len(answer) == 7):
            raise RuntimeEngineError(f"runtime engine unit comprehension answered {str(answer)[:300]}")
        meaning, applied, statuses, left, _engine_ms, nodes, tripped = answer
        return Reading(
            meaning=[[_atom(a) for a in p] for p in meaning or []],
            applied=tuple(applied or ()),
            ms=round(ms, 3),
            complete="succeeded" in (statuses or ()),
            nodes=int(nodes or 0),
            budget=tripped or None,
            left=tuple(left or ()),
        )

    def _client_timeout(self) -> float:
        return self.deadline_ms / 1000 + CLIENT_SLACK_S

    def _restart_runaway(self, timeout: float, cause: BaseException) -> NoReturn:
        """A search ran away: SIGKILL the engine by its pid, restart it with both grammars reloaded, raise."""
        pid = self.pid
        if pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        self._teardown()
        what = f"no answer within {timeout:.2f} s (deadline {self.deadline_ms} ms + {CLIENT_SLACK_S:g} s)"
        try:
            self._boot()
            self._reload()
        except BaseException as exc:
            raise Runaway(
                f"runtime engine unit comprehension: {what}: engine pid {pid} killed; restart: {exc}"
            ) from exc
        raise Runaway(f"runtime engine unit comprehension: {what}: engine pid {pid} killed and restarted") from cause

    # -- transport: pyfcg's routes on this engine's own port, never through pyfcg's globals ---------------------------

    def _lisp(self, code: str, package: str | None = None) -> Any:
        return self._post("/fcg-read-eval-print", {"lisp-code": code, "package": package})

    def _post(self, endpoint: str, data: Mapping[str, Any]) -> Any:
        return self._request("POST", endpoint, data)

    def _request(
        self, method: str, endpoint: str, data: Mapping[str, Any] | None, *, timeout: float | None = None
    ) -> Any:
        timeout = self.request_timeout if timeout is None else timeout
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            body = None if data is None else json.dumps(data).encode()
            headers = {} if data is None else {"Content-Type": "application/json"}
            connection.request(method, endpoint, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read(BODY_CAP + 1)
        except TimeoutError:
            raise _Timeout(f"runtime engine {endpoint}: no answer within {timeout} s") from None
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeEngineError(f"runtime engine {endpoint}: {type(exc).__name__}: {exc}") from None
        finally:
            connection.close()
        if len(payload) > BODY_CAP:
            raise RuntimeEngineError(f"runtime engine {endpoint}: answer over {BODY_CAP} bytes")
        try:
            value = json.loads(payload) if payload else None
        except ValueError:
            value = None
            if response.status == 200:
                raise RuntimeEngineError(f"runtime engine {endpoint}: not JSON: {payload[:200]!r}") from None
        if response.status != 200 or (isinstance(value, dict) and "error-message" in value):
            message = value.get("error-message") if isinstance(value, dict) else payload[:500]
            raise _Refused(f"runtime engine {endpoint}: HTTP {response.status}: {message}")
        return value


__all__ = [
    "AMBIGUOUS",
    "DEADLINE_ENV",
    "DEFAULT_DEADLINE_MS",
    "DEFAULT_NODE_CAP",
    "DEFAULT_PORT",
    "FEATURE_TYPES",
    "GRAMMAR_ID",
    "KERNEL_EXT_IMAGE",
    "KERNEL_EXT_VERSION",
    "LITERAL",
    "NODE_CAP_ENV",
    "PORT_ENV",
    "RUNTIME_DIR",
    "STATE_FILE",
    "UNIT_GRAMMAR_ID",
    "WORD_CLASSES",
    "Reading",
    "RuntimeEngine",
    "PortBusy",
    "Runaway",
    "RuntimeEngineError",
    "kernel_ext_source",
    "runtime_binary",
    "runtime_port",
    "search_budget",
    "unit_configuration",
]
