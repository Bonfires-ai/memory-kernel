"""Stop the ENGINE-side search when the Python side gives up on a comprehend.

THE DEFECT. ``grammar_store.fcg_comprehend_timeout`` bounds how long the CLIENT waits. It
never bounded what the ENGINE does: FCG Go is a LispWorks image serving HTTP through
Hunchentoot with a one-thread-per-connection taskmaster, so abandoning the socket leaves
the worker thread expanding nodes for a search whose answer nobody will ever read. Measured
live 2026-08-20 against the running engine: ~11 sockets in ``TCP_CLOSE_WAIT`` (client gone,
server still holding), 16 of 45 threads runnable, ~930% CPU; an earlier incident left 84
minutes of orphan cputime behind. The cost is not the one abandoned statement -- it is that
every statement AFTER it competes for cores with the corpses of the ones before.

WHY A CANCEL AND NOT AN ENGINE-SIDE DEADLINE. Both were on the table. A deadline would have
to be consulted somewhere inside the search, and the only per-node hook FCG offers is
``cip-node-test``, which is dispatched by NAME from the grammar's ``node-tests``
configuration -- a list whose unknown names are silently dropped (``grammar_store``'s
``PARSE_GOAL_TESTS`` docstring: a name no method dispatches on stops the search at depth 1,
fires zero cxns and returns in 23 ms, green and blind), and which has to be replayed onto
the scoped clone as well. The cancel needs none of that: the client already knows the exact
moment it gave up, and the engine can be told.

WHY IT CAN BE DELIVERED AT ALL. Probed 2026-08-21 on a scratch engine: a ``(sleep 15)``
running in one worker did NOT delay a concurrent ``(+ 40 2)`` on the same port -- it
answered in 4 ms. The taskmaster is genuinely concurrent, so a second connection carrying
``lisp_eval`` reaches the image while a comprehend is in flight. That is the whole
mechanism; without it there would be nothing to do but document the leak.

WHAT THE INTERRUPT DOES. ``mp:process-interrupt`` runs a closure in the target thread at
its next safe point; the closure signals an error, which unwinds the worker exactly the way
``grammar_store._CONDITION_HOOK_SHIM_LISP``'s ``(abort)`` already unwinds one on a stack
overflow -- an established, live-verified mode in this image rather than a new one. Measured
on a scratch engine 2026-08-21, an 11-word card comprehension that runs >30 s: interrupted at
t=2.0 s, the worker unwound at t=2.01 s, engine cputime went flat (0.01 s over the next 3 s),
the registry emptied, and a fresh request answered in 3 ms.

THE SPECIALISER IS LOAD-BEARING. ``grammar_store._COMPREHEND_SOLUTION_BOUND_SHIM_LISP``
already owns ``fcg::comprehend-all :around`` on ``((utterance t) ...)``. ``defmethod`` with
the same qualifier and the same specialisers REPLACES; a registration shim written on
``(utterance t)`` would delete the solution bound and silently return every lane to
searching to exhaustion (p50 189 ms -> 682 ms, every test still green). This shim
specialises on ``string``, a different method, and both run -- ``string`` being the more
specific ``:around``, it simply wraps the bound.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

#: Operator escape hatch, same shape as ``grammar_store.NEIGHBORS_GUARD_ENV``: set truthy to
#: leave abandoned searches running, which is the pre-2026-08-21 behaviour. Default OFF
#: (i.e. the cancel IS installed), because the leak is unbounded and its only symptom is a
#: slowly worsening latency floor on every OTHER statement.
COMPREHEND_CANCEL_DISABLED_ENV = "MK_FCG_NO_COMPREHEND_CANCEL"

#: The condition text the interrupt signals. It travels back to whichever client is still
#: attached (in production: nobody -- the socket is already gone), and into
#: ``fcg-go::*kernel-condition-log*`` through the condition hook, so an operator reading that
#: log can tell a cancelled search from a real engine fault.
CANCEL_CONDITION_TEXT = "kernel-comprehend-cancelled"

#: THE SHIM. A registry of the worker threads currently inside a search, plus the interrupt
#: that empties it.
#:
#: THREE ENTRY POINTS, because the lanes do not share one. ``fcg::comprehend`` and
#: ``fcg::comprehend-all`` are what ``backend._comprehend_query`` posts through.
#: ``fcg::next-cip-solution`` is the search iterator itself, and it is the ONLY one of the
#: three that the PropBank frames lane can be caught by: its route resolves to
#: ``fcg-go::comprehend-and-extract-frames``, which is a plain function (verified 2026-08-21:
#: ``fboundp`` T, ``generic-function`` NIL) and therefore cannot be specialised at all. All
#: three were verified generic in the live image the same day.
#:
#: A DEPTH COUNT, not a set, because those three NEST: ``comprehend-all`` calls
#: ``next-cip-solution`` once per solution, and a plain ``remove`` in the inner cleanup would
#: deregister a thread that is still searching. The hash table is keyed by process under a
#: lock, so a lost entry (which for the ``pushnew`` inventory in ``grammar_store``'s neighbors
#: guard is merely a short list) cannot become a missed cancel here.
#:
#: THE SNAPSHOT IS TAKEN BEFORE ANY INTERRUPT. The victim's ``unwind-protect`` cleanup takes
#: the same lock to deregister itself, so a cancel that held the lock across
#: ``process-interrupt`` would block forever on the thread it just stopped.
_CANCEL_SHIM_LISP = f"""
(progn
 (defvar fcg-go::*kernel-search-lock* (mp:make-lock :name "kernel-search-registry"))
 (defvar fcg-go::*kernel-search-processes* (make-hash-table :test (function eq)))
 (defun fcg-go::kernel-search-enter (p)
   (mp:with-lock (fcg-go::*kernel-search-lock*)
     (incf (gethash p fcg-go::*kernel-search-processes* 0))))
 (defun fcg-go::kernel-search-leave (p)
   (mp:with-lock (fcg-go::*kernel-search-lock*)
     (when (<= (decf (gethash p fcg-go::*kernel-search-processes* 0)) 0)
       (remhash p fcg-go::*kernel-search-processes*))))
 (defun fcg-go::kernel-search-snapshot ()
   (mp:with-lock (fcg-go::*kernel-search-lock*)
     (let ((out nil))
       (maphash (lambda (k v) (declare (ignore v)) (push k out))
                fcg-go::*kernel-search-processes*)
       (copy-list out))))
 (defun fcg-go::kernel-cancel-searches ()
   (let ((ps (remove mp:*current-process* (fcg-go::kernel-search-snapshot))))
     (dolist (p ps)
       (ignore-errors
         (mp:process-interrupt p (lambda () (error "{CANCEL_CONDITION_TEXT}")))))
     (length ps)))
 (defmethod fcg::comprehend :around ((utterance string) &rest args &key &allow-other-keys)
   (declare (ignorable args))
   (let ((self mp:*current-process*))
     (fcg-go::kernel-search-enter self)
     (unwind-protect (call-next-method) (fcg-go::kernel-search-leave self))))
 (defmethod fcg::comprehend-all :around ((utterance string) &rest args &key &allow-other-keys)
   (declare (ignorable args))
   (let ((self mp:*current-process*))
     (fcg-go::kernel-search-enter self)
     (unwind-protect (call-next-method) (fcg-go::kernel-search-leave self))))
 (defmethod fcg::next-cip-solution :around ((cip t) &rest args &key &allow-other-keys)
   (declare (ignorable args))
   (let ((self mp:*current-process*))
     (fcg-go::kernel-search-enter self)
     (unwind-protect (call-next-method) (fcg-go::kernel-search-leave self))))
 t)
"""


def _lisp_eval(pyfcg: Any) -> Any:
    """Resolve ``pyfcg.routes.lisp_eval``, or ``None`` when this bridge does not expose one.

    Pure apart from the attribute walk: resolving the callable contacts nothing. A bridge
    without it is a legitimate state (every unit test's fake), never an error.
    """

    lisp_eval = getattr(getattr(pyfcg, "routes", None), "lisp_eval", None)
    return lisp_eval if callable(lisp_eval) else None


def _disabled() -> bool:
    """Whether the operator has turned the cancel off. Reads the environment fresh."""

    return str(os.environ.get(COMPREHEND_CANCEL_DISABLED_ENV, "") or "").strip().lower() in {"1", "true", "yes"}


def install_comprehend_cancel(pyfcg: Any) -> bool:
    """Install the in-flight-search registry and its interrupt. Idempotent, best-effort.

    Idempotent in both halves: ``defvar`` leaves an existing registry (and the searches
    already in it) alone, while ``defmethod`` on the same specialisers replaces rather than
    accumulates, so re-ensuring the bridge is a no-op on a healthy image.

    Best-effort, never a boot dependency: a failed install logs a WARNING and returns False.
    The degradation is exactly the pre-existing behaviour -- abandoned searches keep running
    -- which is bad but not wrong, and refusing to start the bridge over it would trade a
    latency leak for an outage.
    """

    if _disabled():
        return False
    lisp_eval = _lisp_eval(pyfcg)
    if lisp_eval is None:
        return False
    try:
        lisp_eval(_CANCEL_SHIM_LISP)
    except Exception:
        logger.warning(
            "FCG comprehend-cancel shim install failed; a comprehend that exceeds its budget "
            "will leave an ORPHAN search running in the engine (measured 2026-08-20: ~11 "
            "abandoned connections against one engine at ~930%% CPU, and once 84 minutes of "
            "orphan cputime), so every later statement competes with it for cores",
            exc_info=True,
        )
        return False
    return True
