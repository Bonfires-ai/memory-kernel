"""The engine's HTTP codec and grammar name: the request body, the reply decoder and ``GRAMMAR``.

Ported from a reference-harness module (unreached top-level definitions pruned). What is left is what
``engine.Engine`` uses:

* ``GRAMMAR`` names the resident PropBank grammar the engine is armed with.
* ``http_body`` builds the ``/fcg-read-eval-print`` body, as ASCII-escaped JSON or as raw UTF-8.
* ``decode_response`` decodes the engine's doubly-encoded reply; given the source's non-BMP characters it
  first repairs CL-JSON's over-width escapes.

The fenced request builders are in ``construction.fcg.clause_units_request`` and the fenced-unit record in
``construction.fcg.clause_units``.
"""

from __future__ import annotations

import json

GRAMMAR = "memory-kernel-propbank-grammar-1"


def http_body(code, *, utf8):
    """The /fcg-read-eval-print body. ASCII-escaped JSON turns a non-BMP character into a surrogate pair
    (``\\ud83c\\udf1f``), which the engine's JSON reader refuses ("No character corresponds to code
    #xD83C"). With ``utf8`` the body is sent as raw UTF-8, as the capture's
    utf8-v4 transport does (the reference harness's utf8 HTTP transport)."""
    if utf8:
        return json.dumps({"lisp-code": code, "package": "FCG-GO"}, ensure_ascii=False).encode("utf-8")
    return json.dumps({"lisp-code": code, "package": "FCG-GO"}).encode()


def decode_response(payload, allowed=None):
    """Decode the engine's doubly-encoded reply; with ``allowed`` (the source's non-BMP characters) first
    repair CL-JSON's over-width ``\\uXXXXX`` escapes, verified against the source (utf8-v4 repair v2)."""
    outer = json.loads(payload)
    if not isinstance(outer, str):
        return outer
    if allowed:
        from memory_kernel.pipeline.stages._unicode_repair import repair_cl_json_astral_known

        outer = repair_cl_json_astral_known(outer, allowed)
    return json.loads(outer)
