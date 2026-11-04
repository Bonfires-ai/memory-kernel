"""Private retained-CFS runtime; explicitly installed, never a resident mutation.

The source shapes are the bounded asserted/reported transitive clauses exercised
by warm-query-session/v1. Matching labels yields source candidates, not identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict

from memory_kernel.construction.fcg.native_object_question_layer import (
    native_object_what_cxn_name,
)

SCHEMA = "native-warm-session/v1"
SUPPLIED_SCHEMA = "native-warm-supplied-tree/v1"
SHAPE_VERSION = "object-what-source-candidates/v1"


def check_native_warm_config(contract, grammar_id, max_sources, max_chars, native_seconds):
    native_object_what_cxn_name(contract)
    if (
        contract.subject_binding_mode != "local-source"
        or contract.object_binding_mode != "local-source"
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_@.-]*", grammar_id)
        or type(max_sources) is not int
        or not 1 <= max_sources <= 32
        or type(max_chars) is not int
        or not 1 <= max_chars <= 2000
        or type(native_seconds) not in {int, float}
        or not 0 < native_seconds <= 25
    ):
        raise ValueError("warm session requires bounded configuration and local-source candidate roles")


def native_warm_session_config_digest(
    contract, grammar_id, max_sources, max_chars, native_seconds, *, role_package="FCG-PROPBANK"
):
    check_native_warm_config(contract, grammar_id, max_sources, max_chars, native_seconds)
    if role_package not in {"FCG-PROPBANK", "COMMON-LISP-USER"}:
        raise ValueError("native warm role package differs")
    payload = [SHAPE_VERSION, asdict(contract), grammar_id, max_sources, max_chars, float(native_seconds)]
    if role_package != "FCG-PROPBANK":
        payload.append({"role_package": role_package})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def native_warm_supplied_config_digest(
    contract, grammar_id, max_sources, max_chars, native_seconds, *, role_package="FCG-PROPBANK"
):
    base = native_warm_session_config_digest(
        contract, grammar_id, max_sources, max_chars, native_seconds, role_package=role_package
    )
    return hashlib.sha256((SUPPLIED_SCHEMA + ":" + base).encode()).hexdigest()


def native_warm_session_deadline(body: str, seconds: float) -> str:
    if not body or type(seconds) not in {int, float} or not 0 < seconds <= 25:
        raise ValueError("invalid native session deadline")
    return r"""
(let ((owner mp:*current-process*) (active t) (tag (list :warm-local-deadline))
      (timer nil) (json:*lisp-identifier-name-to-json* #'json:lisp-to-camel-case))
 (catch tag (unwind-protect
  (progn
   (setf timer (mp:make-timer (lambda () (ignore-errors (mp:process-interrupt owner
    (lambda () (when active (throw tag "{\"status\":\"deadline\"}"))))))))
   (mp:schedule-timer-relative timer __SECONDS__)
   (handler-case
    (let ((encoded (json:encode-json-to-string (list (cons :status "ok") (cons :payload __BODY__)))))
     (if (> (length encoded) 8000000) "{\"status\":\"error\",\"message\":\"response cap\"}" encoded))
    (error (e) (json:encode-json-to-string (list (cons :status "error")
      (cons :message (let ((s (princ-to-string e))) (subseq s 0 (min 2000 (length s))))))))))
  (setf active nil) (when timer (mp:unschedule-timer timer)))))
""".replace("__SECONDS__", str(float(seconds))).replace("__BODY__", body)
