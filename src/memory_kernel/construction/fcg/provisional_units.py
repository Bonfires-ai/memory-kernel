"""Pre-search provisional nominal supply: the open class, contributed at DE-RENDER.

WHAT THIS REPLACES, AND WHY IT HAD TO MOVE OUT OF THE GRAMMAR. ``fcg.open_class`` supplied
the open class as two CONSTRUCTIONS -- ``gl-open-nominal-cxn`` (one token of any spelling ->
``noun-cxn``) and ``gl-nominal-compound-cxn`` (two adjacent nominals -> one). Both were
correct about WHAT the card lane needs and wrong about WHERE it can come from, and the cost
was measured on 2026-08-20 as a ~1000x comprehension regression.

THE ROOT CAUSE, in one sentence: **a construction that is applicable wherever an uncoerced
token remains makes the ``:no-valid-children`` parse goal test unreachable.** A node settles
only when NO construction can still apply, so with those two in the grammar no analysis could
settle until every token had been coerced and every adjacent pair composed -- ~2N-1 forced
applications for an N-token utterance, in N! x Catalan-many orders. Measured on this tree
(2-card probe grammar, own scratch FCG-Go, ``max-nr-of-nodes`` 250):

    tokens          8      9     10     11     12       13       14
    with scaffold  0.63s  0.88s 1.08s  1.61s  1.75s   23.02s   32.40s

The cliff at 13 is exactly where 2N-1 crosses ``max-search-depth: 25``: past it NOTHING
settles, the search burns its whole node budget, and the caller's entire time budget with it
(the live 17-card deck returned ZERO rows on every statement over 250 characters, at 45s).
Isolating the cap confirms it -- at ``max-search-depth: 40``, nodes held at 250, the same
13/14-token cases fall to 2.14s/2.87s.

The card lane now RUNS at 40 (``max-search-depth`` raised from 25, the ceiling then
remeasured) -- ``grammar_store.CARD_LANE_MAX_SEARCH_DEPTH``, the one owner every
card-family configuration and the serving resolver's FORCED row read. The 25 above was inherited
from the pb-en learner's stock configuration at lane unification, never measured against this
de-render; the development-conversation projection that night (427 of 763 comprehends at a 30 s budget,
288 of them in the 13-16 word band) looked like this cliff at production scale -- and the
re-measured curve (a settle-curve measurement, 48 units x 2
arms on the 98-card development-conversation fold grammar) says it is NOT: every unit pair comprehends in
the same time at depth 25 and 40 (all 48 within noise; identical fired sets), and the 30 s
timeouts survive at 40. On a card grammar of this size the search never reaches depth 25; it is
bounded by the node budget and the clock, i.e. by PER-NODE cost (candidate density -- a unit that
fires nothing is ~50 ms at any length; a 12-word unit with one firing is 15 s), not by depth.
The 2N-1 cliff above was real on the 2-card probe grammar and is not what the fold hits today.
Levers that address per-node cost: fewer provisional units per utterance (the pb-overlay
de-render, one parse instead of width-3 spans), candidate narrowing at the anchor partition, or a
node budget sized to fail fast. The segment ceiling was NOT moved by this measurement (see
``statement_segmentation``).

TWO CANDIDATE REPAIRS WERE BUILT AND MEASURED DEAD, recorded so they are not retried:

* A LEFT-LINEAR COMPOUND (right daughter gated to a single-token span by a FORM constraint,
  which -- unlike the three unit-feature gates ``open_class`` records as permissively matched
  -- the engine does honour, since a compound unit's own form is only the separator). It
  bought nothing: 12.47s at 13 tokens, and removing the compound ENTIRELY still left 10.99s.
  The leaf coercion alone is essentially the whole cost.
* A DEMAND-ANCHORED CHAIN (coerce only the token adjacent to an already-carved unit, via the
  compound's ``(sequence " " ?p-right ?left)`` adjacency trick). Catastrophic in practice, and
  structurally hopeless anyway: every construction application IS a search step, so covering N
  tokens costs O(N) depth however the applicability is shaped.

That last sentence is the whole argument for this module. No construction-shaped supply can be
O(1) in search steps, and O(1) is what the goal test requires. So the supply stops being a
construction: the units are contributed to the INITIAL TRANSIENT STRUCTURE, deterministically,
before the search begins. Coercion was never a search decision -- every token is provisionally
a nominal, and the only real choice (does this NP fill that slot?) belongs to the cards that
consume the units, where it now lives alone.

THE MECHANISM. :data:`PROVISIONAL_UNITS_SHIM_LISP` defines one new ``fcg::de-render`` method,
specialised on the keyword :data:`PROVISIONAL_UNITS_DE_RENDER_MODE`. It calls the stock
``:de-render-sequence-predicates`` method and APPENDS units to the left pole. Installed
through ``routes.lisp_eval`` at bridge start, exactly like the four shims ``grammar_store``
already ships (``ANCHOR_HASH_SHIM_LISP``, the condition hook, the neighbours guard, the
solution bound).

ROOT'S SEQUENCE PREDICATES ARE LEFT INTACT -- that is the load-bearing detail. The provisional
units are contributed BESIDE the root's form, not carved out of it, so every attested lexical,
verb and function-word construction still matches exactly the material it matched before. The
new mode's output is a strict SUPERSET of the stock mode's.

THE LEFT POLE, MEASURED (2026-08-20). FCG-Go's ``de-render`` puts ``(FORM ((SEQUENCE ...)))``
on the LEFT pole, and the left pole is what a construction's conditional units are matched
against during comprehension. Units contributed to the RIGHT pole alone are INERT: the clause
fires, its role variables stay unbound, and no ``provisional-noun`` row reaches the meaning.
Contributed on the LEFT they bind: " i   am   blorptok " went from ``{}`` to
``{'person': '__self_speaker__', 'name': 'blorptok'}`` in 19 ms.

WHAT A UNIT LOOKS LIKE, and why each feature is there::

    (PROV-SPAN-2-1 (CATEGORY NOUN-CXN)        ; what a clause role slot demands
                   (FE-CATEGORY NOUN-CXN)     ; what `lex-categories` reads
                   (BOUNDARIES (9 19))        ; the clause's precedes/sequence facts
                   (REFERENT ?PROV-N-2-1)     ; what the role binds
                   (MEANING ((PROVISIONAL-NOUN ?PROV-N-2-1 9 19))))

The boundaries are the PADDED span -- one space either side of the token, which is what
``gl-open-nominal-cxn``'s padded literal used to carve and therefore what every downstream
offset consumer already speaks. Under ``card_canonical_spacing``'s TRIPLE interior spacing two
adjacent padded spans leave a one-character residue, which is what the clause's ``precedes``
facts need.

SPANS, AND WHY THEY ARE LEFT-LINEAR. Width-1 units alone re-open the NP-head truncation defect
the earlier compound construction closed (measured: ``topic`` bound "crypto" instead of "crypto security law"), so the mode
also contributes every contiguous span of 2..:data:`PROVISIONAL_SPAN_WIDTH` tokens. Each
carries the SAME six-atom ``provisional-compound`` row the compound construction emitted, and
decomposes LEFT-LINEARLY -- span (i, w) names span (i, w-1) and leaf (i+w-1) as its daughters
-- so ``open_class.contained_provisional_fillers``'s recursive containment check reads it
unchanged, and there is exactly ONE analysis per span instead of Catalan-many.

WIDTH IS A MEASURED TRADE, NOT A CONSTANT SOMEONE LIKED. Every span exists unconditionally
now, and ``open_class.provisional_binding_width`` MAXIMISES bound width, so a too-wide ceiling
turns that tie-break into "grab the longest thing". Measured on the two-card probe over
"i am interested in crypto security law":

    width   full-NP latency   what ``topic``/``name`` bound        interest card fires?
      1        0.101s         topic "crypto" (truncated)                  yes
      2        0.417s         topic "crypto security" (truncated)         yes
      4        5.142s         name "in crypto security law"               NO

Width 4 reproduces, exactly, the greedy-literal failure ``open_class``'s module docstring
records as REJECTED -- the span swallowed the card's own connective and the card whose form
reads "<person> interested in <topic>" stopped firing altogether.

THE TABLE ABOVE HAS A HOLE AT ITS OWN SHIPPED VALUE, AND FILLING IT CHANGED THE DESIGN
(2026-08-20, second measurement pass, same two-card probe, now A/B'd against the retired
scaffold on the same fixture rather than against nothing). Width 3 fails the SAME way width 4
does -- 3.105 s, ``probe.interest.v1`` never fires, ``topic`` never binds -- while the
scaffold it replaces did both in 0.125 s. Width 2 kept the card but bound ``name`` to "in
crypto": a filler that spans the OTHER card's connective. So the ceiling was never the real
control. The real control is WHAT A SPAN MAY CROSS:

    7-token fixture, "i am interested in crypto security law"
    arm                       latency   topic bound            interest card
    retired scaffold          0.125s   crypto security law    fires
    unfenced W=1               0.106s   crypto                 fires
    unfenced W=2               0.446s   crypto security        fires  (name "in crypto")
    unfenced W=3               3.105s   --                     NEVER FIRES
    FENCED W=3                 0.286s   crypto security law    fires

:func:`_open_run` is that control: a span may not contain a closed-class form
(``closed_class_anchors.closed_class_anchor_forms``, READ not re-typed). It cuts the unit
count on that fixture from 18 to 10, makes the connective-swallowing filler structurally
impossible at EVERY width, and takes the shipped width from "measurably harmful" to "better
than the mechanism it replaces on both axes". Width 3 rides because with the fence it is the
first width that reaches a real three-word NP; the fence, not the number, is what makes it
safe. RE-ASSESS BOTH with corpus evidence, not by taste.

THE TOKEN CEILING, and the ONE thing this mechanism did NOT fix (measured 2026-08-20). Moving
the supply out of the grammar removes the DEPTH cost and nothing else: the units are all live
in the same analysis at the same time, so a k-role clause card has O(units^k) applicable
matches at every node, where the retired constructions only ever offered as many nominals as
had already been coerced. Measured, a 446-character statement against a 2-role card
(``pb.be.01``) still exhausts the budget at EVERY width -- ~75 units, ~5.6k role pairs, 250
nodes. That is a different defect from the one this module closes, it is bounded by the
INPUT rather than by the grammar, and the serving lane's answer to it is segmentation (the
serving lane feeds <=15-word segments). :data:`PROVISIONAL_UNIT_TOKEN_CEILING` is
the fail-safe for the path that somehow does not segment: above it the mode contributes NO
units and returns the stock structure verbatim, so an unsegmented long utterance degrades to
exactly the behaviour this tree had before this supply existed -- open-class supply is lost for that one
utterance, which is a KNOWN and RECOVERABLE loss, instead of a fresh latency cliff which is
neither. :func:`provisional_units_skipped` reads the count back out of the live engine, so
"the ceiling is firing on real traffic" is an observation rather than an inference.

CONSTITUENT UNITS -- WHY THE SPAN SUPPLY ABOVE IS NO LONGER THE CARD LANE'S
DEFAULT.

The span supply's cost is not the number of UNITS, it is the number of ROLE PAIRS one
expansion of one card offers. A card's clause cxn, applied right after its verb anchor,
materialises every (unit before the verb x unit after the verb) pair as a child node, because
every provisional unit carries ``(CATEGORY NOUN-CXN)`` and therefore satisfies every NP role
slot. On a 16-word unit the span supply offers ~45 units (every 1..3-token open-class run), and
one single 2-NP-slot card then costs, MEASURED in a fresh worker against its own scratch
engine (a single-card burn probe):

    card (NP slots)                                   width 1   width 2   width 3
    pb.bring.01.arg0-np-v-v-arg1-np (2)                4.05 s    16.2 s    30.8 s
    pb.bring.01....-arg1-np-arg2-pp (3)               16.5 s      --      > 30 s
    the same 2-slot card on the 8-word sentence        0.18 s    0.74 s    1.16 s

The fold grammar sums that over 9-17 anchored cards per burner, so the 30 s budget goes. NO
SEARCH PARAMETER BOUNDS IT: depth 25 -> 40 changed nothing on all 48 probe units, and
``max-nr-of-nodes`` 250 -> 50 changed nothing either (both arms in
the settle-curve measurements) -- budgets count EXPANDED nodes and this is
ONE expansion. ``max-nr-of-nodes`` 1 settles every burner in 45-72 ms with zero firings, which
is the proof: the whole cost is the first clause-cxn expansion after the verb anchor.

So the number of provisional units has to become the number of NP CONSTITUENTS (2-5 per
clause) rather than the number of token spans (3N-3). :data:`CONSTITUENT_UNITS_SHIM_LISP` is
that supply and :func:`constituent_units` its mirror: same skeleton as the span mode (stock
de-render first, ROOT'S SEQUENCE PREDICATES INTACT, units appended to the LEFT pole, the same
six features), different LAW -- one unit per base NP of the parse, plus a width-1 leaf per
open-class unattested token no NP covers, and no ``provisional-compound`` rows at all.

THE PARSE IS THE ONE THE PB LANE ALREADY PAYS FOR: ``nlp-tools:get-penelope-syntactic-
analysis`` through the already-repointed ``*penelope-host*`` (``fcg.penelope_host``), ~45 ms
in-cluster. Three facts about it were MEASURED on a scratch engine (2026-09-03) and each one
shaped the design:

* ``get-penelope-noun-chunks`` returns NIL through the engine's Lisp client, so base NPs come
  from the constituent tree (an NP node with no NP node under it). Checked against the
  service's own ``/spacy-api/noun-chunks`` on the burner sentence -- the two agree exactly.
* the call returns the FIRST TREE ONLY: on a three-sentence utterance it answers with sentence
  0's nodes and drops the rest. So the shim RE-CALLS the parser starting one token past the
  last token the previous parse covered -- the parse's own coverage, never a punctuation rule,
  decides where the next segment begins (sentence segmentation is a separate lever).
* the triple-spaced, space-padded engine-facing utterance parses IDENTICALLY to its
  single-spaced form through the Lisp client (the HTTP endpoint itself 500s on it), so the
  shim hands the parser the utterance verbatim and needs no normalisation step.

BOUNDARIES STAY ON SPACE-DELIMITED TOKENS. spaCy splits further than the space character does
("adorable!" is two tokens, "they're" is two), so an NP's spaCy span is mapped OUT to the
enclosing space tokens and padded exactly like a span unit -- ``l = max(0, start-1)``,
``r = min(len, end+1)``. Every downstream reader (``open_class._span_surface``, the containment
predicate, the ``precedes`` arithmetic the card lane's TRIPLE spacing exists for) is therefore
looking at the same shape of offsets it always has. :func:`_align_tokens` is that mapping and
it is pure arithmetic over token LENGTHS, with a hard refusal on any desync.

FALLBACK, NEVER SILENCE: a first parse that is missing, empty or misaligned yields the SPAN
units instead and bumps ``fcg-go::*kernel-constituent-fallbacks*``
(:func:`constituent_fallbacks`, read back like :func:`provisional_units_skipped`).

WHICH MODE THE CARD LANE USES is :func:`card_lane_de_render_mode`. The former
:data:`PROVISIONAL_SPAN_MODE_ENV` override is rejected because spans lack contextual heads.

Pure and engine-free apart from :func:`install_provisional_units_shim`. :func:`provisional_units`
and :func:`constituent_units` are faithful Python MIRRORS of what the two Lisp methods build --
they exist so the shapes are falsifiable without an engine, and
so re-check each against the engine whenever the Lisp side changes: a mirror that
is not re-checked can drift into fiction.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Collection
from typing import Any

#: THE FUNCTION-WORD FENCE, borrowed rather than re-typed.
#: ``closed_class_anchors`` is the declared owner of "this bare form is a function word", it
#: was built for exactly this kind of comprehension-cost judgment (its own docstring prices
#: ``np.the`` at 63.8% of the corpus), and it is a frozen type-level constant rather than a
#: tagger -- which is what a de-render-time gate needs, since there is no context to tag.
from memory_kernel.config.flags import env_flag
from memory_kernel.construction.closed_class_anchors import closed_class_anchor_forms

logger = logging.getLogger(__name__)

__all__ = [
    "CONSTITUENT_UNITS_DE_RENDER_MODE",
    "CONSTITUENT_UNITS_SHIM_LISP",
    "PROVISIONAL_SPAN_MODE_ENV",
    "PROVISIONAL_SPAN_WIDTH",
    "PROVISIONAL_UNITS_DE_RENDER_MODE",
    "PROVISIONAL_UNITS_SHIM_LISP",
    "PROVISIONAL_UNIT_TOKEN_CEILING",
    "card_lane_de_render_mode",
    "constituent_units_shim_lisp",
    "install_provisional_units_shim",
]

#: The card lane's de-render mode. ADDITIVE -- it delegates to the stock mode below and
#: appends, so no stock mode's behaviour moves and the PropBank substrate's leg is untouched.
PROVISIONAL_UNITS_DE_RENDER_MODE = ":de-render-sequence-predicates-with-provisional-units"

#: THE CARD LANE'S MODE -- one provisional unit per NP CONSTITUENT of the
#: parse instead of one per token span. Same skeleton as the span mode above (stock de-render
#: first, root's sequence predicates intact, units appended to the LEFT pole, same six
#: features per unit); only the SUPPLY LAW differs. See CONSTITUENT UNITS in the module
#: docstring for the measurement that made the span supply untenable.
CONSTITUENT_UNITS_DE_RENDER_MODE = ":de-render-sequence-predicates-with-constituent-units"

#: The mode the new one delegates to: the card lane's own former leg, whose sequence-literal
#: protocol every card cxn is written against.
_STOCK_DE_RENDER_MODE = ":de-render-sequence-predicates"

#: Widest provisional constituent, in TOKENS. See WIDTH IS A MEASURED TRADE above -- this
#: number is a measurement, and both directions off it cost something named there.
PROVISIONAL_SPAN_WIDTH = 3

#: Operator override for :data:`PROVISIONAL_SPAN_WIDTH`, read LIVE (not at import) so a
#: measurement sweep can move the ceiling without a rebuild.
PROVISIONAL_SPAN_WIDTH_ENV = "MK_FCG_PROVISIONAL_SPAN_WIDTH"

#: Obsolete A/B override. A truthy value is rejected by :func:`card_lane_de_render_mode`:
#: current card semantics require the contextual evidence supplied by constituent mode.
#: Direct span utilities remain available for nominal supply tests and historical probes.
PROVISIONAL_SPAN_MODE_ENV = "MK_FCG_PROVISIONAL_SPAN_MODE"

#: THE FAIL-SAFE, in TOKENS -- see THE TOKEN CEILING in the module docstring. Above this many
#: tokens the mode contributes nothing and hands back the stock structure, so the O(units^roles)
#: role-binding cost this mechanism does NOT fix cannot become a new latency cliff on an
#: unsegmented input.
#:
#: 24 is chosen between two measurements and is deliberately not tight to either: the serving
#: lane segments at <=15 words, so nothing on the real path should ever reach it, and
#: the exhaustion the sibling measurement found sits at ~26 tokens (~75 units, a 2-role card).
#: Raising it re-opens that; lowering it silently costs open-class supply on inputs that are
#: fine today, which is why :func:`provisional_units_skipped` exists to make either visible.
PROVISIONAL_UNIT_TOKEN_CEILING = 24

#: Operator override for :data:`PROVISIONAL_UNIT_TOKEN_CEILING`, read LIVE for the same reason.
PROVISIONAL_UNIT_TOKEN_CEILING_ENV = "MK_FCG_PROVISIONAL_TOKEN_CEILING"


#: The Lisp side, as a ``str.format`` template. The body carries no braces of its own, so the
#: five fields below are the only substitutions -- and the two MODE names are substitutions
#: rather than literals precisely because a mode-name typo on either side is INVISIBLE at
#: runtime (FCG-Go answers an unknown de-render mode with its default method, not an error),
#: while the two VOCABULARY fields are substitutions so that this file never becomes a second
#: spelling of a list somebody else owns.
PROVISIONAL_UNITS_SHIM_LISP = """(progn
 (defparameter fcg-go::*kernel-provisional-span-width* {width})
 (defparameter fcg-go::*kernel-provisional-token-ceiling* {ceiling})
 (defvar fcg-go::*kernel-provisional-skipped* 0)
 (defun fcg-go::kernel-token-spans (utterance)
   (let ((spans nil) (start nil) (n (length utterance)))
     (dotimes (i n)
       (if (char= (char utterance i) #\\Space)
           (when start (push (cons start i) spans) (setf start nil))
         (unless start (setf start i))))
     (when start (push (cons start n) spans))
     (coerce (nreverse spans) (quote vector))))
 (defparameter fcg-go::*kernel-provisional-closed-class* (list {closed_class}))
 (defparameter fcg-go::*kernel-provisional-attested* (list {attested}))
 (defun fcg-go::kernel-open-run-p (utterance spans i w)
   (loop for k from i to (+ i w -1)
         always (let ((token (string-downcase (subseq utterance (car (aref spans k)) (cdr (aref spans k))))))
                  (notany (lambda (form) (string= form token))
                          fcg-go::*kernel-provisional-closed-class*))))
 (defun fcg-go::kernel-unattested-run-p (utterance spans i w)
   (loop for k from i to (+ i w -1)
         always (let ((token (string-downcase (subseq utterance (car (aref spans k)) (cdr (aref spans k))))))
                  (notany (lambda (form) (string= form token))
                          fcg-go::*kernel-provisional-attested*))))
 (defun fcg-go::kernel-provisional-cfs (utterance)
   (let* ((cfs (fcg::de-render utterance {stock}))
          (spans (fcg-go::kernel-token-spans utterance))
          (n (length spans))
          (w-max (min fcg-go::*kernel-provisional-span-width* n))
          (cat (intern "NOUN-CXN" :fcg-go))
          (k-cat (intern "CATEGORY" :fcg-go))
          (k-fecat (intern "FE-CATEGORY" :fcg-go))
          (k-bnd (intern "BOUNDARIES" :fcg-go))
          (k-ref (intern "REFERENT" :fcg-go))
          (k-mean (intern "MEANING" :fcg-go))
          (p-noun (intern "PROVISIONAL-NOUN" :fcg-go))
          (p-comp (intern "PROVISIONAL-COMPOUND" :fcg-go))
          (len (length utterance))
          (units nil))
     (when (> n fcg-go::*kernel-provisional-token-ceiling*)
       (incf fcg-go::*kernel-provisional-skipped*)
       (return-from fcg-go::kernel-provisional-cfs cfs))
     (loop for w from 1 to w-max do
       (loop for i from 0 to (- n w) do
         (when (and (or (= w 1) (fcg-go::kernel-open-run-p utterance spans i w))
                    (fcg-go::kernel-unattested-run-p utterance spans i w))
           (let* ((l (max 0 (1- (car (aref spans i)))))
                  (r (min len (1+ (cdr (aref spans (+ i w -1))))))
                  (uname (intern (format nil "PROV-SPAN-~a-~a" i w) :fcg-go))
                  (var (intern (format nil "?PROV-N-~a-~a" i w) :fcg-go))
                  (row (if (= w 1)
                           (list p-noun var l r)
                         (list p-comp var l r
                               (intern (format nil "?PROV-N-~a-~a" i (1- w)) :fcg-go)
                               (intern (format nil "?PROV-N-~a-1" (+ i w -1)) :fcg-go)))))
             (push (list uname (list k-cat cat) (list k-fecat cat) (list k-bnd (list l r))
                         (list k-ref var) (list k-mean (list row)))
                   units)))))
     (setf (fcg::left-pole-structure cfs)
           (append (fcg::left-pole-structure cfs) (nreverse units)))
     cfs))
 (defmethod fcg::de-render ((utterance string)
                            (mode (eql {mode}))
                            &key &allow-other-keys)
   (fcg-go::kernel-provisional-cfs utterance))
 t)"""


def _attested_set(attested: Collection[str] | None) -> frozenset[str]:
    """Normalize a caller's surfaces to the case-folded, blank-free set the comparison uses."""

    if not attested:
        return frozenset()
    return frozenset(surface for surface in (str(item).strip().lower() for item in attested) if surface)


def _span_width_ceiling() -> int:
    """The live span-width ceiling: the env override when it parses to >= 1, else the default."""

    return _positive_env(PROVISIONAL_SPAN_WIDTH_ENV, PROVISIONAL_SPAN_WIDTH)


def _token_ceiling() -> int:
    """The live token fail-safe: the env override when it parses to >= 1, else the default."""

    return _positive_env(PROVISIONAL_UNIT_TOKEN_CEILING_ENV, PROVISIONAL_UNIT_TOKEN_CEILING)


def _positive_env(name: str, default: int) -> int:
    """``name``'s value when it parses to a positive int, else ``default``.

    Deliberately FORGIVING where the rest of this tree's env parsing raises: both knobs it
    serves are measurement conveniences with safe defaults, and this one is read on the LIVE
    comprehend path's install, where a typo must not take the bridge down. Pure.
    """

    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed >= 1 else default


def _lisp_string(value: str) -> str:
    """One Lisp string literal for ``value``, with the two characters that can close it escaped.

    A surface is CORPUS TEXT -- it reaches this function from a manifest built out of real
    chat -- so a stray quote or backslash would otherwise terminate the literal early and the
    whole ``progn`` would fail to read. The install is not best-effort, so that failure would
    take the bridge down rather than degrade quietly; escaping is what keeps it impossible.
    """

    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def provisional_units_shim_lisp(width: int | None = None, *, attested: Collection[str] | None = None) -> str:
    """The shim source for ``width`` (default: the live ceiling), ready for ``lisp_eval``.

    ``attested`` is the surface set the engine must NOT contribute provisional nominals over;
    empty (the default) leaves the guard installed but inert, so the emitted structure is
    byte-identical to what it has always been.
    """

    return PROVISIONAL_UNITS_SHIM_LISP.format(
        width=int(width) if width is not None else _span_width_ceiling(),
        ceiling=_token_ceiling(),
        closed_class=" ".join(f'"{form}"' for form in sorted(closed_class_anchor_forms())),
        attested=" ".join(_lisp_string(surface) for surface in sorted(_attested_set(attested))),
        stock=_STOCK_DE_RENDER_MODE,
        mode=PROVISIONAL_UNITS_DE_RENDER_MODE,
    )


#: The CONSTITUENT side, as a ``str.format`` template with the same discipline as the span
#: template above: no braces of its own, and both mode keywords substituted rather than typed.
#:
#: IT DEPENDS ON THE SPAN SHIM, deliberately and hard. ``kernel-token-spans``,
#: ``kernel-open-run-p``, ``kernel-unattested-run-p``, ``*kernel-provisional-attested*``,
#: ``*kernel-provisional-token-ceiling*``, ``*kernel-provisional-skipped*`` and -- for the
#: FALLBACK arm -- ``kernel-provisional-cfs`` itself are all the span shim's, READ here rather
#: than re-spelled (two spellings of one tokenizer is exactly the drift that
#: would make the Python mirror a lie). :func:`install_provisional_units_shim` therefore
#: installs the two together, span first, and is the ONLY installer.
_CONTEXTUAL_VERB_TAGS = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"})
# Penn Treebank tags emitted by the contextual parser; unrecognized labels stay unavailable.
_CONTEXTUAL_NONVERB_TAGS = frozenset(
    [
        "CC",
        "CD",
        "DT",
        "EX",
        "FW",
        "IN",
        "JJ",
        "JJR",
        "JJS",
        "LS",
        "MD",
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "PDT",
        "POS",
        "PRP",
        "PRP$",
        "RB",
        "RBR",
        "RBS",
        "RP",
        "SYM",
        "TO",
        "UH",
        "WDT",
        "WP",
        "WP$",
        "WRB",
        ".",
        ",",
        ":",
        "-LRB-",
        "-RRB-",
        "``",
        "''",
        "HYPH",
        "NFP",
        "ADD",
        "AFX",
        "GW",
    ]
)

CONSTITUENT_UNITS_SHIM_LISP = """(progn
 (defvar fcg-go::*kernel-constituent-fallbacks* 0)
 (defvar fcg-go::*kernel-contextual-head-verbal* 0)
 (defvar fcg-go::*kernel-contextual-head-nonverbal* 0)
 (defvar fcg-go::*kernel-contextual-head-unavailable* 0)
 (defun fcg-go::kernel-parse-nodes (text)
   (handler-case (nlp-tools:get-penelope-syntactic-analysis text)
     (error () nil)))
 (defun fcg-go::kernel-node-value (node key)
   (cdr (assoc key node)))
 (defun fcg-go::kernel-np-node-p (node)
   (and (member "NP" (fcg-go::kernel-node-value node (quote :phrase_types)) :test (function string=)) t))
 (defun fcg-go::kernel-leaf-node-p (node)
   (let ((kind (fcg-go::kernel-node-value node (quote :node_type))))
     (and (stringp kind) (string= kind "leaf"))))
 (defun fcg-go::kernel-all-attested-run-p (utterance spans i w)
   (and fcg-go::*kernel-provisional-attested*
        (loop for k from i to (+ i w -1)
              always (let ((token (string-downcase (subseq utterance (car (aref spans k)) (cdr (aref spans k))))))
                       (some (lambda (form) (string= form token))
                             fcg-go::*kernel-provisional-attested*)))))
 (defun fcg-go::kernel-align-tokens (spans base nodes)
   (let* ((n (length spans))
          (leaves (sort (remove-if-not (function fcg-go::kernel-leaf-node-p) (copy-list nodes))
                        (function <)
                        :key (lambda (nd) (fcg-go::kernel-node-value nd (quote :start)))))
          (table (make-hash-table))
          (j base)
          (consumed 0))
     (dolist (leaf leaves)
       (when (>= j n) (return-from fcg-go::kernel-align-tokens nil))
       (let ((width (length (fcg-go::kernel-node-value leaf (quote :string))))
             (size (- (cdr (aref spans j)) (car (aref spans j)))))
         (incf consumed width)
         (when (> consumed size) (return-from fcg-go::kernel-align-tokens nil))
         (setf (gethash (fcg-go::kernel-node-value leaf (quote :start)) table) j)
         (when (= consumed size) (incf j) (setf consumed 0))))
     (if (and (= consumed 0) (> j base)) (cons table (1- j)) nil)))
 (defun fcg-go::kernel-base-np-spans (nodes)
   (let ((all nil))
     (dolist (node nodes)
       (when (fcg-go::kernel-np-node-p node)
         (pushnew (cons (fcg-go::kernel-node-value node (quote :start))
                        (fcg-go::kernel-node-value node (quote :end)))
                  all :test (function equal))))
     (remove-if (lambda (a)
                  (some (lambda (b)
                          (and (not (equal a b)) (>= (car b) (car a)) (<= (cdr b) (cdr a))))
                        all))
                all)))
 (defun fcg-go::kernel-contextual-segment-heads (utterance spans nodes table base last-covered)
   (let ((groups (make-hash-table)) (heads nil) (known 0) (nonverbal 0)
         (leaves (sort (remove-if-not (function fcg-go::kernel-leaf-node-p) (copy-list nodes))
                       (function <)
                       :key (lambda (nd) (fcg-go::kernel-node-value nd (quote :start))))))
     ;; Validate content as well as lengths; a same-width wrong parse cannot license a head.
     (loop for leaf in leaves for expected from 0 do
       (unless (eql expected (fcg-go::kernel-node-value leaf (quote :start)))
         (return-from fcg-go::kernel-contextual-segment-heads (values nil 0 0)))
       (push leaf (gethash (gethash expected table) groups)))
     (loop for i from base to last-covered do
       (let* ((group (reverse (gethash i groups)))
              (token (subseq utterance (car (aref spans i)) (cdr (aref spans i))))
              (surface (format nil "~{{~a~}}"
                         (mapcar (lambda (nd) (fcg-go::kernel-node-value nd (quote :string))) group))))
         (unless (string= surface token)
           (return-from fcg-go::kernel-contextual-segment-heads (values nil 0 0)))))
     (loop for i from base to last-covered do
       (let* ((group (gethash i groups))
              (tag (and (= (length group) 1)
                        (fcg-go::kernel-node-value (car group) (quote :lex_class)))))
         (when (and (stringp tag)
                    (member tag '({known_tags}) :test (function string=)))
           (incf known)
           (if (member tag '({verb_tags}) :test (function string=))
               (push (list (intern (format nil "CONTEXTUAL-HEAD-~a" i) :fcg-go)
                           (list (intern "CONTEXTUAL-HEAD" :fcg-go) (intern "VERB" :fcg-go))
                           (list (intern "BOUNDARIES" :fcg-go)
                                 (list (max 0 (1- (car (aref spans i))))
                                       (min (length utterance) (1+ (cdr (aref spans i))))))) heads)
               (incf nonverbal)))))
     (values (nreverse heads) known nonverbal t)))
 (defun fcg-go::kernel-constituent-ranges (utterance spans)
   (let ((n (length spans)) (ranges nil) (cursor 0) (guard 0)
         (heads nil) (known 0) (nonverbal 0) (heads-available t))
     (block segments
       (loop while (< cursor n) do
         (incf guard)
         (when (> guard n) (return-from segments))
         (let* ((text (subseq utterance (car (aref spans cursor))))
                (nodes (fcg-go::kernel-parse-nodes text)))
           (when (null nodes)
             (incf fcg-go::*kernel-constituent-fallbacks*)
             (return-from segments))
           (let ((alignment (fcg-go::kernel-align-tokens spans cursor nodes)))
             (when (null alignment)
               (incf fcg-go::*kernel-constituent-fallbacks*)
               (return-from segments))
             (let ((table (car alignment)) (last-covered (cdr alignment)))
               (when heads-available
                 (multiple-value-bind (segment-heads segment-known segment-nonverbal valid)
                     (fcg-go::kernel-contextual-segment-heads utterance spans nodes table cursor last-covered)
                   (setf heads-available valid)
                   (setf heads (append heads segment-heads))
                   (incf known segment-known)
                   (incf nonverbal segment-nonverbal)))
               (dolist (np (fcg-go::kernel-base-np-spans nodes))
                 (let ((i0 (gethash (car np) table))
                       (i1 (gethash (1- (cdr np)) table)))
                   (when (and i0 i1 (<= i0 i1))
                     (pushnew (cons i0 i1) ranges :test (function equal)))))
               (when (< last-covered cursor) (return-from segments))
               (setf cursor (1+ last-covered)))))))
     (incf fcg-go::*kernel-contextual-head-verbal* (length heads))
     (incf fcg-go::*kernel-contextual-head-nonverbal* nonverbal)
     (incf fcg-go::*kernel-contextual-head-unavailable* (- n known))
     (values (sort ranges (function <) :key (function car)) heads)))
 (defun fcg-go::kernel-constituent-cfs (utterance)
   (let* ((cfs (fcg::de-render utterance {stock}))
          (spans (fcg-go::kernel-token-spans utterance))
          (n (length spans))
          (cat (intern "NOUN-CXN" :fcg-go))
          (k-cat (intern "CATEGORY" :fcg-go))
          (k-fecat (intern "FE-CATEGORY" :fcg-go))
          (k-bnd (intern "BOUNDARIES" :fcg-go))
          (k-ref (intern "REFERENT" :fcg-go))
          (k-mean (intern "MEANING" :fcg-go))
          (p-noun (intern "PROVISIONAL-NOUN" :fcg-go))
          (len (length utterance))
          (units nil))
     (when (= n 0) (return-from fcg-go::kernel-constituent-cfs cfs))
     (when (> n fcg-go::*kernel-provisional-token-ceiling*)
       (incf fcg-go::*kernel-provisional-skipped*)
       (incf fcg-go::*kernel-contextual-head-unavailable* n)
       (return-from fcg-go::kernel-constituent-cfs cfs))
     (multiple-value-bind (ranges heads) (fcg-go::kernel-constituent-ranges utterance spans)
       (let ((covered (make-array n :initial-element nil))
           (starts (make-array n :initial-element nil)))
       (when (null ranges)
         (incf fcg-go::*kernel-constituent-fallbacks*)
         (let ((fallback (fcg-go::kernel-provisional-cfs utterance)))
           (setf (fcg::left-pole-structure fallback)
                 (append (fcg::left-pole-structure fallback) heads))
           (return-from fcg-go::kernel-constituent-cfs fallback)))
       (dolist (range ranges)
         (loop for k from (car range) to (cdr range) do (setf (aref covered k) t))
         (let ((w (1+ (- (cdr range) (car range)))))
           (when (or (null (aref starts (car range))) (> w (aref starts (car range))))
             (setf (aref starts (car range)) w))))
       (dotimes (i n)
         (let ((w (aref starts i)))
           (cond
             (w
              (unless (fcg-go::kernel-all-attested-run-p utterance spans i w)
                (let* ((l (max 0 (1- (car (aref spans i)))))
                       (r (min len (1+ (cdr (aref spans (+ i w -1))))))
                       (uname (intern (format nil "PROV-NP-~a-~a" i w) :fcg-go))
                       (var (intern (format nil "?PROV-NP-~a-~a" i w) :fcg-go)))
                  (push (list uname (list k-cat cat) (list k-fecat cat) (list k-bnd (list l r))
                              (list k-ref var) (list k-mean (list (list p-noun var l r))))
                        units))))
             ((and (not (aref covered i))
                   (fcg-go::kernel-open-run-p utterance spans i 1)
                   (fcg-go::kernel-unattested-run-p utterance spans i 1))
              (let* ((l (max 0 (1- (car (aref spans i)))))
                     (r (min len (1+ (cdr (aref spans i)))))
                     (uname (intern (format nil "PROV-SPAN-~a-1" i) :fcg-go))
                     (var (intern (format nil "?PROV-N-~a-1" i) :fcg-go)))
                (push (list uname (list k-cat cat) (list k-fecat cat) (list k-bnd (list l r))
                            (list k-ref var) (list k-mean (list (list p-noun var l r))))
                      units))))))
       (setf (fcg::left-pole-structure cfs)
             (append (fcg::left-pole-structure cfs) (nreverse units) heads))
       cfs))))
 (defmethod fcg::de-render ((utterance string)
                            (mode (eql {mode}))
                            &key &allow-other-keys)
   (fcg-go::kernel-constituent-cfs utterance))
 t)"""


def constituent_units_shim_lisp() -> str:
    """The constituent shim source, ready for ``lisp_eval``.

    Takes no ``width``/``attested`` arguments: this mode has no width (the parse decides the
    constituent) and it READS the attested set the span shim's ``*kernel-provisional-attested*``
    already carries, so a widen that re-installs the span shim moves both modes at once.
    """

    return CONSTITUENT_UNITS_SHIM_LISP.format(
        stock=_STOCK_DE_RENDER_MODE,
        mode=CONSTITUENT_UNITS_DE_RENDER_MODE,
        known_tags=" ".join(_lisp_string(tag) for tag in sorted(_CONTEXTUAL_VERB_TAGS | _CONTEXTUAL_NONVERB_TAGS)),
        verb_tags=" ".join(_lisp_string(tag) for tag in sorted(_CONTEXTUAL_VERB_TAGS)),
    )


def card_lane_de_render_mode() -> str:
    """Resolve the card lane's constituent mode and reject obsolete span overrides live."""

    if env_flag(PROVISIONAL_SPAN_MODE_ENV, default=False):
        raise ValueError(
            f"Unset obsolete {PROVISIONAL_SPAN_MODE_ENV}: card grammars require constituent "
            "de-rendering for contextual verbal head evidence; span mode cannot supply it."
        )
    return CONSTITUENT_UNITS_DE_RENDER_MODE


def install_provisional_units_shim(pyfcg: Any, attested: Collection[str] | None = None) -> bool:
    """Install the de-render method into a live FCG-Go, returning whether it took.

    Idempotent -- ``defmethod`` on the same specialiser replaces rather than accumulates -- and
    called from ``Engine.arm`` (``pipeline.engine``) beside the other five shims, so the
    managed engine has it before the grammar image is loaded.

    NOT best-effort, unlike the condition hook: the card lane's configured de-render mode is
    this one, and FCG-Go answers an unknown mode with its default ``(T T)`` method rather than
    an error. A failed install would therefore comprehend every card utterance against a
    structure with no nominal supply at all -- fast, green, and binding nothing. Raising here
    is the only way that state is ever visible.

    INSTALLS BOTH MODES, span first. The constituent shim is written against the
    span shim's tokenizer, fence, attested set and -- on its fallback arm --
    ``kernel-provisional-cfs`` itself, so the order is load-bearing. The PB overlay's card
    leg also requires this constituent mode to be installed first. Validate the selected
    mode before touching the engine, so an obsolete override cannot partially install shims.
    """

    mode = card_lane_de_render_mode()
    lisp_eval = getattr(getattr(pyfcg, "routes", None), "lisp_eval", None)
    if not callable(lisp_eval):
        raise RuntimeError("pyfcg.routes.lisp_eval is required to install the provisional-units de-render mode")
    surfaces = _attested_set(attested)
    lisp_eval(provisional_units_shim_lisp(attested=surfaces))
    lisp_eval(constituent_units_shim_lisp())
    logger.info(
        "Installed provisional-units (width %s) and constituent-units de-render modes; "
        "card lane resolves to %s (attested-suppression: %d surfaces)",
        _span_width_ceiling(),
        mode,
        len(surfaces),
    )
    return True
