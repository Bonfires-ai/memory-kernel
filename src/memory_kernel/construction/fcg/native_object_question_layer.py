"""Additive object-WH shape for a unary NP subject; never auto-installed.

The explicit discourse contract supplies the predicate interpretation. This is
not general question coverage or a license to infer a roleset from a lemma.
"""

import hashlib
import json
from dataclasses import asdict

from memory_kernel.construction.fcg.native_discourse_layer import (
    NativeDiscourseContract,
    native_discourse_cxn_names,
)


def native_object_what_cxn_name(contract: NativeDiscourseContract) -> str:
    native_discourse_cxn_names(contract)
    payload = json.dumps(["object-what-unary-subject/v1", asdict(contract)], sort_keys=True, separators=(",", ":"))
    return "dqr-object-what-unary-" + hashlib.sha256(payload.encode()).hexdigest()[:20] + "-cxn"
