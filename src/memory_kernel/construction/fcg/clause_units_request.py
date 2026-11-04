"""Request builders for the one-request, multi-pass fenced clause-unit comprehend.

Pure Lisp-string builders (no engine I/O). Logic ported from the reference harness
(with the ``passes`` vector fix). ``fenced_request`` returns ``(request, deadline_s)``; the
driver may take ``hashlib.sha256(request.encode()).hexdigest()`` to fill ``phase_a["request_sha256s"]``.
"""

from __future__ import annotations

import json

T_PASS = 5
DEADLINE_CAP = 25.0
ENCODED_CAP = 15_000_000
GRAMMAR = "memory-kernel-propbank-grammar-1"


def _lisp_string(value):
    from memory_kernel.construction.fcg.grammar_store import _lisp_string as lisp_string

    return lisp_string(value)


def _nodes_json(nodes, utf8):
    if utf8:
        return json.dumps(nodes, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    from memory_kernel.construction.mined_affordance_contract import canonical_json

    return canonical_json(nodes)


def lisp_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("invalid Lisp timeout")
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def deadline_request(body, seconds):
    """Copy of ``_resident_stock_capture_runner.stock_deadline_request`` with partial results.

    ``body`` pushes one alist per finished pass onto ``p2-rows``. The encoder runs AFTER the catch and
    after ``active`` is cleared and the timer unscheduled, so a late interrupt cannot cut the encoding.
    """
    if not isinstance(body, str) or not body or not 0 < seconds <= 60:
        raise ValueError("invalid fenced deadline request")
    return (
        r"""
(let ((owner mp:*current-process*) (active t) (tag (list :p2-local-deadline))
      (timer nil) (p2-rows nil) (finished nil) (failure nil)
      (json:*lisp-identifier-name-to-json* #'json:lisp-to-camel-case))
 (catch tag
  (unwind-protect
   (progn
    (setf timer (mp:make-timer
     (lambda () (ignore-errors
      (mp:process-interrupt owner
       (lambda () (when active (throw tag nil))))))))
    (mp:schedule-timer-relative timer __SECONDS__)
    (handler-case (progn __BODY__ (setf finished t))
     (error (e) (setf failure (let ((s (princ-to-string e))) (subseq s 0 (min 2000 (length s))))))))
   (setf active nil)
   (when timer (mp:unschedule-timer timer))))
 (let ((encoded (json:encode-json-to-string
                 (list (cons :status (cond (failure "error") (finished "ok") (t "deadline")))
                       (cons :message failure)
                       (cons :payload (list (cons :n (length p2-rows))
                                            (cons :passes (coerce (reverse p2-rows) 'vector))))))))
  (if (> (length encoded) __CAP__)
      "{\"status\":\"error\",\"message\":\"encoded response cap\"}" encoded)))
""".replace("__SECONDS__", repr(float(seconds)))
        .replace("__BODY__", body)
        .replace("__CAP__", str(ENCODED_CAP))
    )


def _pass_body(text, pass_id, nodes, t_pass, *, grammar, utf8):
    from memory_kernel.construction.fcg.native_analysis import wrap_native_analysis_request

    if not nodes:
        raise ValueError(f"refusing nil nodes for pass {pass_id}")
    timeout = f" :timeout {lisp_number(t_pass)}"
    inner = f"""
(let* ((json:*json-identifier-name-to-lisp* #'json:camel-case-to-lisp)
       (json:*lisp-identifier-name-to-json* #'json:lisp-to-camel-case)
       (text {_lisp_string(text)})
       (nodes (json:decode-json-from-string {_lisp_string(_nodes_json(nodes, utf8))}))
       (t0 (get-internal-real-time))
       (row (if (null nodes)
                (list (cons :status "refused-nil"))
                (handler-case
                    (multiple-value-bind (meaning node frame-set)
                        (fcg-go::comprehend-and-extract-frames text
                          :cxn-inventory {grammar} :syntactic-analysis (copy-tree nodes) :silent t{timeout})
                      (list (cons :status "returned")
                            (cons :meaning-kind (cond ((null meaning) "nil") ((symbolp meaning) (symbol-name meaning))
                                                      ((consp meaning) "list") (t (prin1-to-string (type-of meaning)))))
                            (cons :frame-set-kind (cond ((null frame-set) "nil") ((symbolp frame-set) (symbol-name frame-set))
                                                        (t (prin1-to-string (type-of frame-set)))))
                            (cons :frames (when (and frame-set (not (symbolp frame-set))) (fcg-propbank::frames frame-set)))
                            (cons :native (snapshot meaning node))))
                  (error (e) (list (cons :status "error")
                                   (cons :message (let ((s (princ-to-string e))) (subseq s 0 (min 2000 (length s)))))))))))
  (push (append (list (cons :id {_lisp_string(pass_id)})
                      (cons :seconds (float (/ (- (get-internal-real-time) t0) internal-time-units-per-second))))
                row)
        p2-rows))"""
    return wrap_native_analysis_request(inner)


def order_passes(passes):
    """Smallest first (node count, then id) -- the pass most likely to finish runs first."""
    return sorted(passes, key=lambda p: (len(p["nodes"]), p["id"]))


def passes_request(text, passes, *, t_pass, deadline_s, grammar=GRAMMAR, utf8=False):
    """One request over explicit ``[{"id", "nodes"}]`` passes, run in the given order."""
    if not text.strip() or len(text) > 2000:
        raise ValueError("invalid source unit")
    if not passes:
        raise ValueError("no passes")
    ids = [p["id"] for p in passes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pass ids")
    body = "\n".join(_pass_body(text, p["id"], p["nodes"], t_pass, grammar=grammar, utf8=utf8) for p in passes)
    return deadline_request(body, deadline_s)


def outer_deadline(n_passes, t_pass=T_PASS, cap=DEADLINE_CAP):
    return min(t_pass * n_passes + 1.0, cap)


def fenced_request(job, analysis, *, t_pass=T_PASS, utf8=False):
    passes = order_passes(analysis["passes"])
    deadline = outer_deadline(len(passes), t_pass)
    return passes_request(job["source"]["text"], passes, t_pass=t_pass, deadline_s=deadline, utf8=utf8), deadline
