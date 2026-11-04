"""The receipts a build's ``sanity`` stage writes: no search, no Voyage, no graph read, $0.

A forward append's (``append_receipt``): its structure is proven by construction, by the write's forward gate (rules
1-4) and the index's append gate. The receipt records those results, what the projection minted, reused, routed and
linked, and what the referents stage decided (its counts, its hard cases with Jev's p(yes), the discriminated mints).

A full build's (``full_receipt``, kernel 1.8, D1): the index's results (its message count, recipe tag and stamped keys,
nothing stale or missing). 1.7's blind retrieval check searched the conversation's taxonomy questions; those left with
the taxonomy layer, and retrieval quality is guarded by the key gate and the bench.

Both are written to ``11-sanity-001/sanity.json``, so the seal and the commit are unchanged."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from memory_kernel.pipeline.stages._g1_cards import write_once
from memory_kernel.pipeline.stages.write import SUMMARY, Refused, _out

if TYPE_CHECKING:
    from memory_kernel.pipeline.runner import Ctx

RECEIPT_SCHEMA = "mk-append-receipt/v1"
FULL_RECEIPT_SCHEMA = "mk-build-receipt/v1"
FULL_CHECK = (
    "a full build (the bootstrap, a rebuild) writes the build receipt of its index; no build searches at sanity"
)
#: The index results a full build's receipt carries.
FULL_KEYS: tuple[str, ...] = ("n_messages", "recipe_tag", "keys_stamped_with_tag")


def build_receipt(*, write_summary: Mapping, index_result: Mapping, decisions: Mapping) -> dict[str, Any]:
    block = write_summary.get("append") or {}
    fwd = block.get("forward")
    if not isinstance(fwd, Mapping):
        raise Refused("receipt: the write's summary.json carries no forward result (append.forward)")
    if block.get("passed") is not True:
        raise Refused("receipt: the write gate did not pass (append.passed); no receipt over a failed write")
    if index_result.get("passed") is not True:
        raise Refused("receipt: the index gate did not pass (index.json passed); no receipt over a failed index")
    minted = [{k: m[k] for k in ("uuid", "key", "display", "kind")} for m in fwd["minted"]]
    minted_ids = {m["uuid"] for m in minted}
    return {
        "schema": RECEIPT_SCHEMA,
        "mode": "append",
        "passed": True,
        "gates": {
            "write": {k: block.get(k) for k in ("passed", "episodic", "saga", "created", "inversion")},
            "index": {"passed": True, "append_gate": index_result.get("append_gate")},
        },
        "minted": minted,
        "reused": sorted(set(fwd["touched"]) - minted_ids),
        "routes": dict(fwd["routes"]),
        "same_as": list(fwd["same_as"]),
        "conflicts": list(fwd["conflicts"]),
        "referents": {
            "counts": dict(decisions.get("counts") or {}),
            "hard": list(decisions.get("hard") or ()),
            "discriminated": sorted(m["key"] for m in minted if "#" in m["key"]),
        },
        "full_check": FULL_CHECK,
        "voyage_tokens": 0,
        "voyage_requests": 0,
    }


def append_receipt(ctx: Ctx) -> dict:
    """A forward append's ``sanity``: the receipt, write-once; $0."""
    from memory_kernel.pipeline.stages.index import STAGE_FILES

    target = _out(ctx, "sanity", STAGE_FILES["sanity"])
    summary = json.loads((ctx.path("write") / SUMMARY).read_bytes())
    index_result = json.loads((ctx.path("index") / STAGE_FILES["index"]).read_bytes())
    decisions = json.loads((ctx.path("referents") / "decisions.json").read_bytes())
    with ctx.cost("index", run=target.parent.name, note=f"C2 append receipt {ctx.cfg.sample_id}, $0 (no search)"):
        out = build_receipt(write_summary=summary, index_result=index_result, decisions=decisions)
    write_once(target, out)
    return {k: v for k, v in out.items() if k != "reused"}


def build_full_receipt(*, index_result: Mapping) -> dict[str, Any]:
    """A full build's receipt from its ``index.json``: refused (``Refused``) when a key is stale or missing, or the
    recipe tag is not the freeze's."""
    if index_result.get("stale") or index_result.get("missing"):
        raise Refused(
            f"receipt: the index has {index_result.get('stale')} stale and {index_result.get('missing')} missing keys; "
            "no receipt over a stale index"
        )
    if index_result.get("tag_equals_freeze") is not True:
        raise Refused("receipt: the index's recipe tag is not the freeze's (tag_equals_freeze); no receipt")
    return {
        "schema": FULL_RECEIPT_SCHEMA,
        "mode": "full",
        "passed": True,
        **{k: index_result[k] for k in FULL_KEYS},
        "stale": 0,
        "missing": 0,
    }


def full_receipt(ctx: Ctx) -> dict:
    """A full build's ``sanity``: the receipt of its index, write-once; $0."""
    from memory_kernel.pipeline.stages.index import STAGE_FILES

    target = _out(ctx, "sanity", STAGE_FILES["sanity"])
    index_result = json.loads((ctx.path("index") / STAGE_FILES["index"]).read_bytes())
    with ctx.cost("index", run=target.parent.name, note=f"C2 build receipt {ctx.cfg.sample_id}, $0 (no search)"):
        out = build_full_receipt(index_result=index_result)
    write_once(target, out)
    return out
