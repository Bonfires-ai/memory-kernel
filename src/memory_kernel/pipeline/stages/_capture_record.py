"""Fresh stock detector capture. Raw deadlines are not warm graph receipts.

Ported from a reference-harness module (unreached top-level definitions pruned). Differences: the stock capture runner is
imported from the ported ``_resident_stock_capture_runner`` (no stage imports reference-harness code).
"""

import json

from memory_kernel.construction.fcg.grammar_store import _lisp_string
from memory_kernel.construction.fcg.native_analysis import wrap_native_analysis_request
from memory_kernel.construction.memory_semantic_jev import seal, sha
from memory_kernel.construction.mined_affordance_contract import canonical_json
from memory_kernel.pipeline.stages import _resident_stock_capture_runner as stock

UNIT_SCHEMA = "mk-raw-unit/v1"


def round_native_request(job, seconds=5, *, utf8=False):
    text = job["source"]["text"]
    if not text.strip() or len(text) > 2000 or not job["nodes"]:
        raise ValueError("invalid source unit")
    nodes_json = (
        json.dumps(job["nodes"], ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        if utf8
        else canonical_json(job["nodes"])
    )
    body = f"""
(let* ((json:*json-identifier-name-to-lisp* #'json:camel-case-to-lisp)
        (json:*lisp-identifier-name-to-json* #'json:lisp-to-camel-case)
        (text {_lisp_string(text)})
        (nodes (json:decode-json-from-string {_lisp_string(nodes_json)})))
 (multiple-value-bind (meaning node frame-set)
     (fcg-go::comprehend-and-extract-frames text
       :cxn-inventory {stock.GRAMMAR} :syntactic-analysis (copy-tree nodes) :silent t)
   (list (cons "frames" (when frame-set (fcg-propbank::frames frame-set)))
         (cons "native" (snapshot meaning node)))))"""
    return stock.stock_deadline_request(wrap_native_analysis_request(body), seconds)


def round_capture_record(job, response, request):
    if response == {"status": "deadline"}:
        status = "cutoff"
    elif response.get("status") == "ok" and set(response.get("payload", {})) == {"frames", "native"}:
        status = "framed" if response["payload"]["frames"] else "empty"
    else:
        raise ValueError("native failure is neither empty nor cutoff")
    return seal(
        {
            "schema_version": UNIT_SCHEMA,
            "job": job,
            "response": response,
            "request_sha256": sha(request.encode()),
            "capture_status": status,
            "semantic_acceptance": "not_performed",
        }
    )
