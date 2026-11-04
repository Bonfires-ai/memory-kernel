"""One native parser capture, complete sentence partitions, no text heuristics."""

from __future__ import annotations

import hashlib
from copy import deepcopy

from memory_kernel.construction.fcg.grammar_store import _lisp_string
from memory_kernel.construction.fcg.native_frame_transport import source_tokens_from_nodes
from memory_kernel.construction.fcg.native_warm_session_lisp import native_warm_session_deadline
from memory_kernel.construction.memory_discourse_contract import check_discourse_source, discourse_id
from memory_kernel.construction.memory_discourse_native import _native_token_char_spans
from memory_kernel.construction.memory_discourse_units import compile_source_units
from memory_kernel.construction.memory_semantic_jev import check_seal, seal
from memory_kernel.construction.mined_affordance_contract import canonical_json

SYNTAX_SCHEMA = "warm-message-syntax/v1"
MAX_TREES = 31


def capture_warm_syntax(sender, text, *, grammar_id):
    """The sender's host owns runtime attestation, lease and transport deadline."""
    import re

    if (
        type(text) is not str
        or not text.strip()
        or len(text) > 2000
        or "\x00" in text
        or type(grammar_id) is not str
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_@.-]*", grammar_id)
    ):
        raise ValueError("invalid native message syntax request")
    body = f"""
(let* ((resident (find-symbol {_lisp_string(grammar_id.upper())} "FCG-GO"))
       (inventory (and resident (boundp resident) (symbol-value resident)))
       (text {_lisp_string(text)})
       (json:*json-identifier-name-to-lisp* #'json:camel-case-to-lisp)
       (json:*lisp-identifier-name-to-json* #'json:lisp-to-camel-case))
 (unless inventory (error "resident grammar absent"))
 (let* ((model (or (fcg::get-configuration inventory :model) "en_benepar"))
        (parsed (nlp-tools::run-penelope-syntactic-parser text :model model)))
  (list (cons "text" text) (cons "parser" (list nlp-tools::*penelope-host* model))
        (cons "trees" (cdr (assoc :trees parsed))))))"""
    result = sender(native_warm_session_deadline(body, 5))
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ValueError("native whole-message syntax capture failed")
    payload = result["payload"]
    if set(payload) != {"text", "parser", "trees"} or payload["text"] != text:
        raise ValueError("native whole-message source differs")
    capture = seal(
        {
            "schema_version": SYNTAX_SCHEMA,
            **deepcopy(payload),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "parser_calls": 1,
        }
    )
    check_warm_syntax(capture)
    return capture


def check_warm_syntax(capture):
    check_seal(capture, SYNTAX_SCHEMA)
    if (
        set(capture) != {"schema_version", "content_sha256", "text", "text_sha256", "parser", "trees", "parser_calls"}
        or type(capture["text"]) is not str
        or not capture["text"].strip()
        or len(capture["text"]) > 2000
        or "\x00" in capture["text"]
        or capture["text_sha256"] != hashlib.sha256(capture["text"].encode()).hexdigest()
        or type(capture["parser_calls"]) is not int
        or capture["parser_calls"] != 1
        or type(capture["parser"]) is not list
        or len(capture["parser"]) != 2
        or any(type(value) is not str or not value for value in capture["parser"])
        or type(capture["trees"]) is not list
        or not 1 <= len(capture["trees"]) <= MAX_TREES
        or len(canonical_json(capture).encode()) > 131072
    ):
        raise ValueError("native whole-message syntax bounds differ")
    for index, tree in enumerate(capture["trees"]):
        if (
            type(tree) is not dict
            or set(tree) != {"sent_id", "tree"}
            or type(tree["sent_id"]) is not int
            or tree["sent_id"] != index
            or type(tree["tree"]) is not list
            or not tree["tree"]
            or any(type(node) is not dict for node in tree["tree"])
        ):
            raise ValueError("native sentence inventory differs")


def warm_source_units(source, capture):
    """Normalize only token coordinates; node IDs and clause structure are retained."""
    check_discourse_source(source)
    check_warm_syntax(capture)
    if source["text"] != capture["text"]:
        raise ValueError("syntax capture belongs to a different source")
    text = source["text"]
    all_nodes = [node for tree in capture["trees"] for node in tree["tree"]]
    tokens = source_tokens_from_nodes(
        [{key.replace("_", "--"): value for key, value in node.items()} for node in all_nodes], text
    )
    token_spans = _native_token_char_spans(text, tokens, 0)
    ranges, cursor = [], 0
    for tree in capture["trees"]:
        roots = [node for node in tree["tree"] if node.get("parent_constituent") is None]
        if len(roots) != 1:
            raise ValueError("native sentence requires exactly one root")
        start, end = roots[0]["start"], roots[0]["end"]
        if type(start) is not int or type(end) is not int or start != cursor or not start < end <= len(tokens):
            raise ValueError("native sentences do not partition all source tokens")
        ids = [node.get("node_id") for node in tree["tree"]]
        if any(type(value) is not int for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("native sentence node identities differ")
        for node in tree["tree"]:
            if (
                type(node.get("start")) is not int
                or type(node.get("end")) is not int
                or not start <= node["start"] < node["end"] <= end
                or (node.get("parent_constituent") is not None and node["parent_constituent"] not in ids)
                or (node.get("dependency_head") is not None and node["dependency_head"] not in ids)
            ):
                raise ValueError("native sentence node escapes its tree")
        by_id = {node["node_id"]: node for node in tree["tree"]}
        for node in tree["tree"]:
            ancestors = {node["node_id"]}
            while node.get("parent_constituent") is not None:
                parent_id = node["parent_constituent"]
                if type(parent_id) is not int or parent_id in ancestors:
                    raise ValueError("native constituent ancestry cycles")
                parent = by_id[parent_id]
                if not parent["start"] <= node["start"] < node["end"] <= parent["end"]:
                    raise ValueError("native constituent escapes its parent span")
                ancestors.add(parent_id)
                node = parent
        ranges.append((start, end))
        cursor = end
    if cursor != len(tokens):
        raise ValueError("native sentences leave source tokens uncovered")
    root_id = discourse_id("segment", source["content_sha256"], 0, len(text))
    root = {
        "segment_id": root_id,
        "start": 0,
        "end": len(text),
        "parent_id": None,
        "kind": "source",
        "token_indices": list(range(len(tokens))),
        "dependency": None,
    }
    segments = [root]
    if len(ranges) > 1:
        for ordinal, (start, end) in enumerate(ranges):
            begin = 0 if ordinal == 0 else token_spans[start][0]
            stop = len(text) if ordinal == len(ranges) - 1 else token_spans[end][0]
            segments.append(
                {
                    "segment_id": discourse_id("segment", source["content_sha256"], begin, stop),
                    "start": begin,
                    "end": stop,
                    "parent_id": root_id,
                    "kind": "sentence",
                    "token_indices": list(range(start, end)),
                    "dependency": None,
                }
            )
    selected = segments[1:] or segments
    ids = [row["segment_id"] for row in selected]
    syntax = seal(
        {
            "schema_version": "discourse-syntax/v1",
            "source_sha256": source["content_sha256"],
            "parser_stamp": capture["content_sha256"],
            "segments": segments,
            "status": "parsed",
            "diagnostics": [],
            "alternatives": [
                {"alternative_id": discourse_id("segmentation", source["content_sha256"], ids), "segment_ids": ids}
            ],
        }
    )
    units = compile_source_units(source, syntax)
    for unit, tree, (start, end) in zip(units, capture["trees"], ranges, strict=True):
        nodes = deepcopy(tree["tree"])
        for node in nodes:
            node["start"] -= start
            node["end"] -= start
        unit.update(
            text=text[unit["start"] : unit["end"]],
            nodes=nodes,
            parser=deepcopy(capture["parser"]),
            original_token_range=[start, end],
            syntax_sha256=capture["content_sha256"],
        )
        source_tokens_from_nodes(
            [{key.replace("_", "--"): value for key, value in node.items()} for node in nodes], unit["text"]
        )
    return units
