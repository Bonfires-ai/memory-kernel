"""Opt-in, bounded live native sessions with source-qualified answer candidates.

The injected sender must hold the engine's exclusive lease, evaluate only on the
same resident image, enforce a transport timeout, and return the decoded deadline
envelope. This module never starts an engine, loads an image, resolves identity,
or commits world truth. Its records are volatile; rollover is not persistence.
"""

from __future__ import annotations

import re

from memory_kernel.construction.fcg.native_analysis import (
    NIL,
    NativeCons,
    NativeSymbol,
    decode_tagged,
    native_derender_nodes,
    proper_list,
    validate_native_analysis,
)
from memory_kernel.construction.fcg.native_frame_transport import source_tokens_from_nodes


def native_warm_bindings(application: dict, construction: str) -> dict:
    if application["construction"] != construction:
        raise ValueError("unlicensed native session construction")
    results = []
    for field in ("match", "merge"):
        bindings = {}
        for pair in proper_list(decode_tagged(application[field])):
            if (
                not isinstance(pair, NativeCons)
                or not isinstance(pair.car, NativeSymbol)
                or not re.fullmatch(r"\?[A-Z-]+-\d+", pair.car.name)
            ):
                raise ValueError("malformed native binding")
            key = pair.car.name.rsplit("-", 1)[0]
            if key in bindings:
                raise ValueError("duplicate native variable")
            bindings[key] = pair.cdr
        results.append(bindings)
    if not results[0] or results[0] != results[1]:
        raise ValueError("native match/merge bindings differ")
    return results[0]


def warm_derender_nodes(nodes):
    """Use the same native syntax conversion as cold transport and replay."""
    return native_derender_nodes(nodes)


def native_warm_units(receipt: dict):
    """Adapt only known serializer keys; never rewrite native symbols or values."""
    aliases = {
        "node_id": "node--id",
        "node_type": "node--type",
        "phrase_types": "phrase--types",
        "parent_constituent": "parent--constituent",
        "lex_class": "lex--class",
        "dependency_label": "dependency--label",
        "dependency_head": "dependency--head",
        "named_entity_type": "named--entity--type",
    }
    nodes = []
    for node in receipt["nodes"]:
        if not isinstance(node, dict) or set(node) - (set(aliases) | {"start", "end", "string", "lemma"}):
            raise ValueError("unknown parser metadata")
        nodes.append({aliases.get(k, k): v for k, v in node.items()})
    tokens = source_tokens_from_nodes(nodes, receipt["source"]["text"])
    keys = {
        "nodeState": "node-state",
        "nodeType": "node-type",
        "finalUnits": "final-units",
        "nonListUnits": "non-list-units",
        "opaqueCount": "opaque-count",
        "truncatedCount": "truncated-count",
    }
    raw = receipt["native"]
    if set(raw) != set(keys) | {"meaning", "applications"}:
        raise ValueError("unknown native snapshot metadata")
    snapshot = {keys.get(k, k): v for k, v in raw.items()}
    snapshot["final-units"] = [
        {("unit-name" if k == "unitName" else k): v for k, v in u.items()} for u in raw["finalUnits"] or []
    ]
    app_keys = {"chainPosition": "chain-position", "matchBindings": "match-bindings", "mergeBindings": "merge-bindings"}
    snapshot["applications"] = [{app_keys.get(k, k): v for k, v in app.items()} for app in raw["applications"] or []]
    return validate_native_analysis(
        snapshot, nodes=warm_derender_nodes(nodes), token_range=(0, len(tokens)), has_frames=bool(receipt["frames"])
    )


def native_warm_source_evidence(receipt, contract, *, role_package="FCG-PROPBANK"):
    if role_package not in {"FCG-PROPBANK", "COMMON-LISP-USER"}:
        raise ValueError("native source role package differs")
    analysis = native_warm_units(receipt)
    parser = receipt.get("parser")
    if not isinstance(parser, list) or len(parser) != 2 or any(type(v) is not str or not v for v in parser):
        raise ValueError("parser provenance missing")
    if receipt["source"]["kind"] == "question":
        return analysis
    claims = [proper_list(row) for row in proper_list(decode_tagged(receipt["claims"]))]
    apps = receipt["qualification"] or []
    if (
        len(claims) != len(apps)
        or len(claims) > 1
        or receipt["status"] != ("qualified" if claims else "unsupported_source")
    ):
        raise ValueError("source qualification not unique or mismatched")
    pb = "FCG-PROPBANK"
    for claim, app in zip(claims, apps, strict=True):
        if len(claim) != 9 or claim[0] != NativeSymbol("KEYWORD", "WARM-SOURCE-CLAIM"):
            raise ValueError("invalid native source contribution")
        _, scope, sense, event, subject, obj, label, answer, speaker = claim
        if scope not in {"asserted", "reported"} or sense != NativeSymbol(pb, contract.roleset.upper()):
            raise ValueError("source scope or roleset differs")
        bindings = native_warm_bindings(app, "warm-source/" + scope)
        text_variables = {"?LABEL", "?ANSWER"} | ({"?SPEAKER-LABEL"} if scope == "reported" else set())
        unit_variables = {"?ROOT", "?EVENT", "?SUBJECT", "?OBJECT", "?VP", "?PUNCT"}
        if scope == "reported":
            unit_variables |= {"?REPORT", "?REPORT-VP", "?SPEAKER", "?COMPLEMENT", "?MARKER", "?CLAUSE"}
        if set(bindings) != unit_variables | text_variables:
            raise ValueError("source application variable inventory differs")
        needed = {"?EVENT": event, "?SUBJECT": subject, "?OBJECT": obj, "?LABEL": label, "?ANSWER": answer}
        if any(bindings.get(k) != v for k, v in needed.items()):
            raise ValueError("source contribution lacks its exact application witnesses")
        units = analysis.units_by_name
        if any(bindings[key] not in units for key in unit_variables):
            raise ValueError("source references foreign native units")
        if units[bindings["?ROOT"]].parent is not None:
            raise ValueError("source qualification is not rooted")
        event_features = dict(units[event].features)
        expected_head = event if scope == "asserted" else bindings["?REPORT"]
        expected_dependency = NativeSymbol("FCG", "ROOT") if scope == "asserted" else NativeSymbol(pb, "CCOMP")
        if (
            event_features.get(NativeSymbol(pb, "DEPENDENCY-HEAD")) != expected_head
            or event_features.get(NativeSymbol(pb, "DEPENDENCY-LABEL")) != expected_dependency
        ):
            raise ValueError("source clause scope disagrees with native event attachment")
        required = {
            (NativeSymbol(pb, "FRAME"), sense, event),
            (NativeSymbol(pb, "FRAME-ELEMENT"), NativeSymbol(role_package, contract.subject_role), event, subject),
            (NativeSymbol(pb, "FRAME-ELEMENT"), NativeSymbol(role_package, contract.object_role), event, obj),
        }
        if scope == "reported":
            report, reporter, complement = (bindings.get(k) for k in ("?REPORT", "?SPEAKER", "?COMPLEMENT"))
            if (
                any(value not in units for value in (report, reporter, complement))
                or bindings.get("?SPEAKER-LABEL") != speaker
                or dict(units[reporter].features).get(NativeSymbol("COMMON-LISP", "STRING")) != speaker
            ):
                raise ValueError("reported source lacks reporter evidence")
            required |= {
                (NativeSymbol(pb, "FRAME"), NativeSymbol(pb, "SAY.01"), report),
                (NativeSymbol(pb, "FRAME-ELEMENT"), NativeSymbol(role_package, "ARG0"), report, reporter),
                (NativeSymbol(pb, "FRAME-ELEMENT"), NativeSymbol(role_package, "ARG1"), report, complement),
            }
        elif speaker is not NIL:
            raise ValueError("asserted source acquired a reporter")
        if role_package == "COMMON-LISP-USER":
            selected_events = {event} | ({report} if scope == "reported" else set())
            if any(
                row[0] == NativeSymbol(pb, "FRAME-ELEMENT")
                and len(row) == 4
                and row[2] in selected_events
                and (type(row[1]) is not NativeSymbol or row[1].package != role_package)
                for row in analysis.meaning
            ):
                raise ValueError("source frame has a competing role namespace")
        if not required <= set(analysis.meaning):
            raise ValueError("source qualification lacks native predicate/role evidence")
        if (
            dict(units[subject].features).get(NativeSymbol("COMMON-LISP", "STRING")) != label
            or dict(units[obj].features).get(NativeSymbol("COMMON-LISP", "STRING")) != answer
        ):
            raise ValueError("claim text does not match its exact source units")
    return analysis
