"""PyFCG-owned grammar image storage for bonfire/profile grammars."""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from memory_kernel.normalization import stable_id

logger = logging.getLogger(__name__)


#: Set truthy to keep LispWorks' stock MODAL error dialogs for interactive
#: FCG Go debugging. Default OFF: a stack overflow signals STORAGE-CONDITION, which is
#: NOT a subtype of ERROR, so every ``handler-case (error ...)`` in the request path
#: misses it and the condition escapes to the CAPI GUI as a modal that BLOCKS the
#: worker thread until a human clicks OK -- the client meanwhile hangs into its HTTP
#: timeout with zero trace in any captured log (observed live 2026-07-20, and the
#: signature behind the hub-walk gate's unexplained ReadTimeout at degree 150).
FCG_DIALOG_DEBUGGER_ENV = "MK_FCG_DIALOG_DEBUGGER"

#: ``defvar`` (not defparameter): re-ensuring the bridge must not clobber captured
#: history. ``(abort)`` unwinds the worker thread -- the client sees a fast connection
#: abort instead of a silent hang, and the condition text stays readable server-side
#: via ``(car fcg-go::*kernel-condition-log*)``. Verified live 2026-07-20: a forced
#: unbounded recursion returned in ~2s with NO dialog, log entry "Stack overflow
#: (stack size 32394801)." -- identical to the operator-observed dialogs.
#: Probe (2026-07-20, live-verified against a synthetic ``(labels ((r (n) (r
#: (1+ n)))) (r 0))`` overflow): of the three LispWorks backtrace candidates,
#: ``dbg:output-backtrace :brief`` was the first to return a non-empty string
#: from inside the ``*debugger-hook*`` handler -- 75,878 chars, with the
#: recursing frame appearing by line 6 (5 lines of debugger-internal frames,
#: then 995 repeats of the recursing frame). ``(let ((*debug-io* s)) ...)``
#: returned an EMPTY string in this context (rebinding ``*debug-io*`` does not
#: redirect ``output-backtrace``'s implicit destination the way an explicit
#: stream argument does). ``mp:map-process-backtrace`` also worked but is
#: unbounded per-frame output -- it produced an 18MB string for the same
#: synthetic overflow, dangerous at a moment when the stack is nearly full
#: (see the module docstring's stack-depth caution). ``:brief`` to an explicit
#: stream, truncated to 2000 chars, is enough: the repeating recursion frame
#: starts within the first ~20 lines, well inside the truncation window.
_CONDITION_HOOK_SHIM_LISP = (
    "(progn"
    " (defvar fcg-go::*kernel-condition-log* nil)"
    " (setf *debugger-hook*"
    "       (lambda (c h) (declare (ignore h))"
    '         (push (format nil "~a | ~a" c'
    "           (let ((bt (or (ignore-errors (with-output-to-string (s)"
    "                           (dbg:output-backtrace :brief s)))"
    '                         "")))'
    "             (subseq bt 0 (min 2000 (length bt)))))"
    "           fcg-go::*kernel-condition-log*)"
    "         (when (> (length fcg-go::*kernel-condition-log*) 200)"
    "           (setf fcg-go::*kernel-condition-log*"
    "                 (subseq fcg-go::*kernel-condition-log* 0 200)))"
    "         (abort)))"
    " t)"
)


def install_fcg_condition_hook(pyfcg: Any) -> bool:
    """Route conditions that would open a LispWorks modal into a log + thread abort.

    Idempotent; best-effort hardening, never a boot dependency: a failed install
    logs a warning and returns False rather than breaking bridge startup.
    """

    if str(os.environ.get(FCG_DIALOG_DEBUGGER_ENV, "") or "").strip().lower() in {"1", "true", "yes"}:
        return False
    lisp_eval = getattr(getattr(pyfcg, "routes", None), "lisp_eval", None)
    if not callable(lisp_eval):
        return False
    try:
        lisp_eval(_CONDITION_HOOK_SHIM_LISP)
    except Exception:
        logger.warning(
            "FCG condition-hook shim install failed; stack overflows will surface as MODAL "
            "dialogs that block worker threads until dismissed",
            exc_info=True,
        )
        return False
    return True


#: ``(graph-utils:neighbors graph node)`` where ``node`` is a symbol
#: not present in the graph: the T-specialized trampoline method does
#: ``(neighbors graph (gethash node (nodes graph)))``; on a miss ``gethash``
#: yields NIL, NIL is not an integer, dispatch lands on the SAME T method
#: again -- infinite recursion -- "Stack overflow (stack size 32394801)".
#: Verified live 2026-07-20 with exactly::
#:
#:     (let ((g (graph-utils:make-undirected-typed-graph)))
#:       (graph-utils:add-node g 'fcg-go::cat-a)
#:       (length (graph-utils:neighbors g 'fcg-go::missing-category-xyz)))
#:
#: -- connection abort + the condition logged (the condition hook above). This is
#: 41/60 of the card-lane comprehension crashes seen in the overflow probe:
#: utterances whose cxns reference a category never added to the categorial
#: network. Default ON: unknown category -> NIL (no neighbors), the same
#: silent-filter semantics FCG already uses for absent categories in its cxn
#: supplier. Set truthy to fall back to the stock (overflowing) trampoline.
#:
#: The miss arm RECORDS before returning NIL. A silent shim would be un-measurable: a run made
#: with this guard installed (it is installed when the engine is armed) must be able to show
#: whether it fired, so the statements it censored are NOT evidence that the guard is absent.
#: ``fcg_neighbors_guard_misses`` reads the inventory back.
#:
#: CONCURRENCY (recorded, not fixed): the miss arm ``pushnew``es onto a global from
#: whatever FCG-Go worker thread hit the miss, and ``pushnew`` is read-modify-write, not
#: atomic. Two simultaneous misses can therefore lose one entry. The inventory is a
#: BOUNDED, DIAGNOSTIC set of distinct names -- the failure mode is a slightly short list,
#: never corruption and never a control-path effect -- so it is documented rather than
#: locked: a lock in the miss arm would sit on the hot path of the very trampoline this
#: guard exists to keep cheap. Read a zero-length inventory as "nothing recorded", never
#: as "nothing missed".
NEIGHBORS_GUARD_ENV = "MK_FCG_NO_NEIGHBORS_GUARD"

_NEIGHBORS_GUARD_SHIM_LISP = (
    "(progn"
    " (defvar fcg-go::*kernel-neighbors-guard-misses* nil)"
    " (defmethod graph-utils:neighbors"
    "     ((graph graph-utils:undirected-typed-graph) node"
    "      &key edge-type (return-ids? t))"
    "   (let ((id (gethash node (graph-utils::nodes graph))))"
    "     (if id"
    "         (graph-utils:neighbors graph id :edge-type edge-type :return-ids? return-ids?)"
    "         (progn"
    '           (pushnew (format nil "~a" node) fcg-go::*kernel-neighbors-guard-misses*'
    "                    :test #'string=)"
    "           nil))))"
    " t)"
)


#: How many settled analyses a comprehension may collect before the search stops.
#: ONE value, both lanes, no per-lane override -- one grammar, one config.
#:
#: WHY A BOUND EXISTS AT ALL. `backend._maximal_solution_index` can only choose among the
#: solutions the engine returns, so `_comprehend_query` asks `routes.comprehend_all` instead of
#: `routes.comprehend`. Unbounded, that runs the search to EXHAUSTION on every statement -- the
#: same thing `:connected-structure` used to force -- and costs back everything the bound gained.
#: 56 stratified live production statements, 321-card grammar, 5.0 s budget, through
#: `fold_matcher.match_statement`, ONE arm-order-controlled session (position inside a session is
#: worth ~50% of the median, so only these rows compare):
#:
#:   arm                  settled  timeouts   p50     p75    firings  reselections
#:   maximal, N=2          45/56      11     178 ms  1817 ms    52          6
#:   maximal, N=5          44/56      12     189 ms  2057 ms    51          6
#:   stock (pre-bound)       2/56      11     680 ms  3467 ms    47         --
#:
#: Isolated in the same way, collection alone (same goal tests, same first-leaf read, only the
#: transport moving from `comprehend` to an unbounded `comprehend_all`) is p50 84 -> 684 ms.
#:
#: WHY N=2 AND NOT MORE. Reach is what the bound costs, and it is measured: 6 of the 7
#: reselections are at solution index 1 and one is at index 3, so N=2 recovers six -- including
#: the adjudicated `np.memory` loss and every case in both lanes' acceptance -- and N>=4
#: recovers the seventh (a 19-word statement, one extra firing). Smallest N that clears
#: acceptance wins; raising it is a measured trade, not a free one.
#:
#: THE BOUND IS GLOBAL, so it also bounds `card_compiler.comprehend_card_utterance`, which
#: unions applied-cxn rows across solutions. That is deliberate -- a second, lane-local value
#: is the fork this unit refuses -- and the card-probe integration suites are part of the
#: bound's acceptance for exactly that reason.
#:
#: 2 -> 6, RAISED for open categorial slots -- ON MEASUREMENT, and the
#: measurement is that N=2 no longer clears acceptance, so this is the same "smallest N that
#: clears acceptance" rule applied to a changed grammar rather than a relaxation of it.
#: **OPERATOR-VISIBLE: this is a serving bound and it now carries a different
#: number.**
#:
#: WHY OPEN-CLASS SUPPLY FORCES IT. A provisional noun (`fcg.open_class`) can read ANY token,
#: including one an attested lexical already covers, so every such token yields a SECOND
#: settled analysis identical to the first but for which cxn read it. Those duplicates are
#: charged against `n` before the search ever reaches a competing clause reading. Measured on
#: a fixture ("graph record tasks": three
#: attested noun primitives, the middle one ALSO the card's verb, so the clause reading has to
#: win a contest) -- the clause's first appearance in the solution stream:
#:
#:   supply                                      clause at index   N needed
#:   before open-class supply (enumerated only)                          1               2
#:   + provisional noun                                2               3
#:   + provisional noun + nominal compound             4               6
#:
#: The compound's extra cost is ASSOCIATIVITY: `(a b) c` and `a (b c)` settle as two analyses
#: with byte-identical applied-cxn multisets, so every scatter reading doubles. FCG offers no
#: way to make one branching legal without footprint negation, which this dict-shaped pole API
#: cannot express -- verified 2026-08-19, including an attempt to gate the compound's daughters
#: on an extra unit feature (`open-class`), which FCG matched permissively and did not enforce.
#: At N=2 the consequence is not a worse reading but NO reading: `synthetic.record.v1` fired
#: perfectly well and `applied_construct_ids` came back EMPTY.
#:
#: WHY 6 AND NOT 3. 3 clears the provisional noun alone; the compound is what makes a role slot
#: bind "crypto security law" instead of "crypto" (burn-in defect class 3), and dropping it to
#: hold the bound down would trade a measured product defect for a measured search cost. The
#: table above prices that cost: N=5 vs N=2 was settled 44 vs 45 and p50 189 vs 178 ms --
#: within noise on the arm that measured it. On the open-class probe utterances the same shape holds
#: (18 -> 39 ms on the three-token fixture, 15 -> 24 ms on "i am <novel>").
#:
#: NOT RE-MEASURED ON THE 56-STATEMENT BATTERY. That battery predates the open-class
#: supply and the honest statement is that this number is set from the mechanism plus the
#: fixture above, not from a fresh sweep; the sweep belongs with the next reprojection, which
#: rebuilds the population these numbers describe.
#:
#: `None` means unbounded (no shim installed).
COMPREHEND_MAX_SOLUTIONS: int | None = 6

#: `comprehend-all` in this FCG Go build is `(utterance &key cxn-inventory silent n)` -- verified
#: 2026-08-11 through `describe` on the generic -- but pyfcg's `/fcg-comprehend-all` route sends
#: no `n`, so the bound cannot be passed per call. This `:around` supplies the default, and an
#: EXPLICIT `:n` from any caller still wins. Same injection seam as the neighbors guard.
_COMPREHEND_SOLUTION_BOUND_SHIM_LISP = (
    "(progn"
    " (defparameter fcg-go::*kernel-comprehend-all-n* {bound})"
    " (defmethod fcg::comprehend-all :around"
    "     ((utterance t) &rest args &key ((:n n) nil n-supplied-p) &allow-other-keys)"
    "   (declare (ignorable n))"
    "   (if (or n-supplied-p (null fcg-go::*kernel-comprehend-all-n*))"
    "       (call-next-method)"
    "       (apply #'call-next-method utterance :n fcg-go::*kernel-comprehend-all-n* args)))"
    " t)"
)


def install_fcg_comprehend_solution_bound(pyfcg: Any) -> bool:
    """Bound how many solutions ``comprehend-all`` searches for, at :data:`COMPREHEND_MAX_SOLUTIONS`.

    Idempotent (the ``defparameter`` re-asserts the current value on every bridge ensure, so a
    process that changes the constant gets the new bound) and best-effort: a failed install logs a
    warning and returns False. The degradation is a SLOWER search, never a wrong one — every
    solution the bounded search would have found is still found, and the selector still picks the
    maximal one — which is why this is not a boot dependency the way a missing goal-test method
    would be. Returns False without touching the engine when the bound is ``None``.
    """

    if COMPREHEND_MAX_SOLUTIONS is None:
        return False
    lisp_eval = getattr(getattr(pyfcg, "routes", None), "lisp_eval", None)
    if not callable(lisp_eval):
        return False
    try:
        lisp_eval(_COMPREHEND_SOLUTION_BOUND_SHIM_LISP.format(bound=int(COMPREHEND_MAX_SOLUTIONS)))
    except Exception:
        logger.warning(
            "FCG comprehend-all solution bound install failed; every comprehension will now run "
            "the search to EXHAUSTION instead of stopping at %s solutions (measured 2026-08-11: "
            "p50 189ms -> 682ms on the 56-statement fold sample)",
            COMPREHEND_MAX_SOLUTIONS,
            exc_info=True,
        )
        return False
    return True


def install_fcg_neighbors_guard(pyfcg: Any) -> bool:
    """NIL-guard the ``graph-utils:neighbors`` trampoline for an absent category.

    Idempotent; best-effort hardening, never a boot dependency: a failed install
    logs a warning and returns False rather than breaking bridge startup.
    """

    if str(os.environ.get(NEIGHBORS_GUARD_ENV, "") or "").strip().lower() in {"1", "true", "yes"}:
        return False
    lisp_eval = getattr(getattr(pyfcg, "routes", None), "lisp_eval", None)
    if not callable(lisp_eval):
        return False
    try:
        lisp_eval(_NEIGHBORS_GUARD_SHIM_LISP)
    except Exception:
        logger.warning(
            "FCG neighbors-guard shim install failed; a cxn referencing a category never "
            "added to the categorial network will overflow the stack instead of returning "
            "no neighbors",
            exc_info=True,
        )
        return False
    return True


#: Reads the category key off a SAMPLE of the restored inventory's unhashed bucket.
#:
#: The unhashed (``nil``) bucket is where every ARGUMENT-STRUCTURE-CXN lands -- they carry no
#: lemma, so ``:hash-lemma`` cannot key them -- and it is exactly the bucket the supplier is
#: supposed to narrow using the categorial network. THE PROCESSING INVENTORY IS THE ONE THAT
#: COUNTS: ``fcg::processing-cxn-inventory`` holds a table distinct from the grammar's own, and
#: it is the one the supplier reads. Probing the grammar's table instead reports a healthy image
#: for a broken one.
#:
#: A SAMPLE, not a scan: the bucket is homogeneous by construction (all 8027 of pb-en.store's
#: entries carry the identical attribute key set), and a full walk of 22535 constructions costs
#: real time on every load for no extra discrimination.
PROPBANK_SUPPLIER_CONTRACT_LISP = (
    "(let* ((inventory (fcg::processing-cxn-inventory {grammar}))"
    "       (bucket (gethash nil (slot-value inventory (quote fcg::constructions-hash-table))))"
    "       (current 0) (retired 0) (neither 0))"
    "  (dolist (cxn (subseq bucket 0 (min {sample} (length bucket))))"
    "    (let ((attributes (slot-value cxn (quote fcg::attributes))))"
    "      (cond ((assoc {current_key} attributes) (incf current))"
    "            ((assoc {retired_key} attributes) (incf retired))"
    "            (t (incf neither)))))"
    '  (format nil "bucket=~a current=~a retired=~a neither=~a" (length bucket) current retired neither))'
)


def _propbank_conll_block_from_tokens(
    *,
    sentence_index: int,
    tokens: Sequence[str],
    predicate_index: int,
    roleset: str,
    annotations: Mapping[int, str],
    predicate_surface: str = "",
) -> str:
    """The COLUMN half of :func:`_propbank_conll_block`, over a token list that is already fixed.

    Split out for the bonfire ledger (``memory_kernel.grammar.ledger``),
    whose rows carry their own token list — clitic-split at the row's construction, so the block
    must NOT be re-tokenized from the statement text. Every column decision lives here and only
    here, so there is exactly one dialect: lemma and roleset on the predicate row, ``-``
    everywhere else, and the annotation the caller resolved per index (``*`` when absent).

    Trusts its inputs: bounds are the caller's business (:func:`_propbank_conll_block` checks
    them against the frame, the ledger against its row). Pure.
    """

    lemma = _safe_fcg_token(roleset.split(".", 1)[0] or predicate_surface, prefix="lemma")
    return "\n".join(
        _conll_row(
            sentence_index,
            token_index,
            token,
            lemma if token_index == predicate_index else "-",
            roleset if token_index == predicate_index else "-",
            annotations.get(token_index, "*"),
        )
        for token_index, token in enumerate(tokens)
    )


def _propbank_span_annotations(*, token_count: int, spans: Mapping[str, Sequence[int]]) -> dict[int, str]:
    """``{token index: bracket annotation}`` for role -> token-index spans. See
    :func:`_propbank_annotations`, which is this plus the frame reading and the V check."""

    annotations: dict[int, str] = {}
    for role, raw_span in spans.items():
        label = "V" if str(role).lower() == "v" else str(role).upper()
        span = sorted({int(index) for index in raw_span if 0 <= int(index) < token_count})
        if not span:
            continue
        start, end = span[0], span[-1]
        if start == end:
            annotations[start] = f"({label}*)"
            continue
        annotations[start] = f"({label}*"
        for index in range(start + 1, end):
            annotations[index] = "*"
        annotations[end] = "*)"
    return annotations


#: Public aliases for the two halves the BONFIRE LEDGER writes through (`grammar.ledger`,
#: "no second dialect"): the column renderer over a fixed token list and
#: the span -> bracket-annotation map. Same private-import-ratchet reason as above.
propbank_conll_block_from_tokens = _propbank_conll_block_from_tokens
propbank_span_annotations = _propbank_span_annotations


def _conll_row(sentence_index: int, token_index: int, token: str, lemma: str, roleset: str, annotation: str) -> str:
    """Render one CoNLL row in the column layout the PropBank reader expects.

    Column widths are part of the format the reader parses, so the padding is load-bearing rather
    than cosmetic. The token is sanitized; the other fields are written as given. Pure.
    """

    safe_token = _safe_conll_token(token)
    return f"/    {sentence_index:<3} {token_index:<3} {safe_token:<12} /     /      {lemma:<10} {roleset:<12} {annotation}"


def _safe_conll_token(value: str) -> str:
    """Reduce a token to ASCII with no internal whitespace so it cannot break the column layout.

    Accents are decomposed and non-ASCII dropped; runs of whitespace become underscores. LOSSY — a
    fully non-ASCII token reduces to the placeholder ``_``, and never to an empty column. Pure.
    """

    token = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    token = re.sub(r"\s+", "_", token.strip())
    return token or "_"


def _lisp_string(value: str) -> str:
    """Quote a Python string as a Lisp string literal for embedding in a ``lisp_eval`` form.

    Backslashes and double quotes are escaped; nothing else is, so a value carrying other control
    characters is passed through as-is. Pure.
    """

    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _safe_fcg_token(value: Any, *, prefix: str = "sym") -> str:
    """Reduce a value to a lowercase ``[a-z0-9_]`` token the Lisp reader can take bare.

    Accents are decomposed and non-ASCII dropped, then runs of anything else collapse to underscores.
    A value that reduces to nothing, or to something starting with a digit, gets ``prefix`` in front
    — the empty case falling back to the value's stable id, so distinct inputs stay distinct.

    LOSSY and NOT injective in general: two values differing only in punctuation collapse to the same
    token. Deterministic across runs. Pure.
    """

    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()
    if not text:
        text = f"{prefix}_{stable_id(value)}"
    if text[0].isdigit():
        text = f"{prefix}_{text}"
    return text


# ---------------------------------------------------------------------------
# Constructicon card grammar images (GRAMMAR lane).
#
# The per-statement CONTENT-lane builders above (_add_lexical_construct /
# _add_relation_construct / _add_frame_construct / _add_propbank_frame_construct
# and _primitive_sequence_cxn) are UNTOUCHED; everything below is additive
# grammar-lane machinery. The card->cxn shapes live in card_compiler (the ONE
# owner); this seam owns content addressing + the admission gate.
# ---------------------------------------------------------------------------


#: cxn-supply is THE card-lane behaviour, not a flag.
#:
#: RETIRED: an anchor-hash flag. Its docstring said "OFF by
#: default -- first stage only ... no cxn carries an ``anchors`` attribute yet, so ON is
#: RECALL-NEUTRAL". That was true at first and FALSE once a later stage stamped
#: the anchors -- and nobody updated the note, so a measured speedup sat
#: switched off, self-documented as a no-op, for six days.
#:
#: MEASURED 2026-07-26, 200 real production statements, 161-card live grammar, A/B with the
#: SAME 128-primitive lexicon on both arms (the first attempt changed the lexicon and
#: the flag together and was discarded):
#:
#:   flag OFF -> coverage 31.0%, fidelity 69.5%, median 249.2ms, 63 outliers >1s, max 118s
#:   flag ON  -> coverage 30.5%, fidelity 66.0%, median  92.8ms, 37 outliers >1s, max  34s
#:
#: So the ORIGINAL claim survives on the axis that mattered -- recall-neutral (31.0 vs
#: 30.5) -- while the partition is 2.7x on the median and halves the pathological tail.
#: There is no configuration in which OFF is the better choice, so there is no knob.
#: The card lane is the CANONICAL comprehension lane;
#: PropBank-style learning reaches NO serving lane -- it is the bootstrap route's
#: COMPREHENSION SUBSTRATE only (``propbank_substrate.build_propbank_substrate``),
#: built offline, discarded after the run, never published.


__all__ = []
