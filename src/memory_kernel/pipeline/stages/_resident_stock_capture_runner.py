"""Stock PropBank request builders: the resident grammar's name and configuration, and two Lisp requests.

Ported from a reference-harness module (unreached top-level definitions pruned; ROOT removed, callers pass root).

* ``GRAMMAR`` / ``CONFIG``: the resident grammar's name and the configuration a readback must show.
* ``stock_deadline_request`` wraps a Lisp body in a local timer (at most 25 s): on the deadline it returns
  ``{"status":"deadline"}``, otherwise the JSON-encoded payload (refused above 8,000,000 characters) or the
  error message.
* ``stock_readback_request`` reads the resident grammar's owner, class, configuration, parser host, search
  count and supplier contract back from the engine.
"""

from __future__ import annotations

GRAMMAR = "memory-kernel-propbank-grammar-1"

CONFIG = {
    "hash-mode": ":HASH-LEMMA",
    "cxn-supplier-mode": ":HASHED-CATEGORIAL-NETWORK",
    "de-render-mode": ":DE-RENDER-CONSTITUENTS-DEPENDENTS",
    "shuffle-cxns-before-application": "NIL",
    "model": "NIL",
    "node-tests": "(:CHECK-DOUBLE-ROLE-ASSIGNMENT)",
    "goal-tests": "NIL",
    "max-nr-of-nodes": "200",
}


def stock_deadline_request(body: str, seconds: float = 25) -> str:
    if not isinstance(body, str) or not body or not 0 < seconds <= 25:
        raise ValueError("invalid local deadline request")
    # The callback accesses active only in the owning thread. A queued callback
    # after unwind sees NIL and cannot throw into a completed request.
    return r"""
(let ((owner mp:*current-process*) (active t) (tag (list :stock-local-deadline))
      (timer nil) (json:*lisp-identifier-name-to-json* #'json:lisp-to-camel-case))
 (catch tag
  (unwind-protect
   (progn
    (setf timer (mp:make-timer
     (lambda () (ignore-errors
      (mp:process-interrupt owner
       (lambda () (when active (throw tag "{\"status\":\"deadline\"}"))))))))
    (mp:schedule-timer-relative timer __SECONDS__)
    (handler-case
     (let ((encoded (json:encode-json-to-string
                    (list (cons :status "ok") (cons :payload __BODY__)))))
      (if (> (length encoded) 8000000)
          "{\"status\":\"error\",\"message\":\"encoded response cap\"}" encoded))
     (error (e) (json:encode-json-to-string
       (list (cons :status "error")
             (cons :message (let ((s (princ-to-string e))) (subseq s 0 (min 2000 (length s))))))))))
   (setf active nil)
   (when timer (mp:unschedule-timer timer)))))
""".replace("__SECONDS__", str(float(seconds))).replace("__BODY__", body)


def stock_readback_request() -> str:
    from memory_kernel.construction.fcg.grammar_store import PROPBANK_SUPPLIER_CONTRACT_LISP

    pairs = " ".join(f'(cons "{key}" (prin1-to-string (fcg::get-configuration g :{key})))' for key in CONFIG)
    supplier = PROPBANK_SUPPLIER_CONTRACT_LISP.format(
        grammar="g", sample=3, current_key=":argst-category", retired_key=":gram-category"
    )
    return stock_deadline_request(
        f'''
(let* ((s (find-symbol "{GRAMMAR.upper()}" :fcg-go))
        (g (and s (boundp s) (symbol-value s)))
        (registry (find-symbol "KERNEL-SEARCH-SNAPSHOT" :fcg-go)))
 (unless (and g registry (fboundp registry)) (error "resident or search registry unavailable"))
 (list (cons :owner (package-name (symbol-package s)))
       (cons :class (symbol-name (type-of g)))
       (cons :config (list {pairs}))
       (cons :parser nlp-tools::*penelope-host*)
       (cons :searches (length (funcall registry)))
       (cons :supplier {supplier})))''',
        5,
    )
