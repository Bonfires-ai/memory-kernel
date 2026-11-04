"""Atomic message projection with complete sentence and attachment provenance."""

from copy import deepcopy

from memory_kernel.construction.fcg.native_warm_source_units import warm_source_units
from memory_kernel.construction.memory_discourse_contract import check_discourse_source, discourse_id, discourse_source
from memory_kernel.construction.memory_semantic_jev import seal
from memory_kernel.construction.memory_warm_graph import (
    MAX_RECORD_BYTES,
    MESSAGE_PROFILE_SCHEMAS,
    MESSAGE_RECORD_SCHEMA,
    check_warm_graph_profile,
    project_warm_source,
)
from memory_kernel.construction.mined_affordance_contract import canonical_json

BUNDLE_SCHEMA = "native-warm-message-evidence/v1"


def warm_child_source(parent, text, identifier):
    return discourse_source(
        scope=parent["scope"],
        message_id=identifier,
        order=parent["order"],
        text=text,
        speaker=parent["speaker"],
        session_id=parent["session_id"],
        timestamp=parent["timestamp"],
        addressees=parent["addressees"],
    )


def warm_message_parts(source):
    check_discourse_source(source)
    parts = [("message_text", None, source["text"])]
    parts.extend(("attachment_caption", a["attachment_id"], a["caption"]) for a in source["attachments"])
    result = []
    for kind, attachment_id, text in parts:
        if len(text) > 2000 or "\x00" in text:
            raise ValueError("warm message part exceeds native text bounds")
        part_id = discourse_id("warm-part", source["content_sha256"], kind, attachment_id)
        result.append(
            {"kind": kind, "attachment_id": attachment_id, "source": warm_child_source(source, text, part_id)}
        )
    return result


def warm_message_jobs(source, part, capture):
    jobs = []
    for unit in warm_source_units(part["source"], capture):
        origin = {
            "kind": part["kind"],
            "attachment_id": part["attachment_id"],
            "source_sha256": source["content_sha256"],
            "source_unit_id": unit["unit_id"],
            "span": [unit["start"], unit["end"]],
            "syntax_sha256": capture["content_sha256"],
        }
        child = warm_child_source(source, unit["text"], unit["unit_id"])
        jobs.append({"source": child, "origin": origin, "nodes": unit["nodes"], "parser": unit["parser"]})
    return jobs


def project_warm_message(source, bundle, *, profile):
    check_warm_graph_profile(profile)
    if profile["schema_version"] not in MESSAGE_PROFILE_SCHEMAS or source["scope"] != profile["scope"]:
        raise ValueError("warm message profile differs")
    parts = warm_message_parts(source)
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"schema_version", "parts"}
        or bundle["schema_version"] != BUNDLE_SCHEMA
        or not isinstance(bundle["parts"], list)
        or len(bundle["parts"]) != len(parts)
        or len(canonical_json(bundle).encode()) > MAX_RECORD_BYTES
    ):
        raise ValueError("warm message evidence bounds differ")
    units, statuses = [], []
    for part, evidence in zip(parts, bundle["parts"], strict=True):
        if (
            not isinstance(evidence, dict)
            or set(evidence) != {"kind", "attachment_id", "capture", "receipts"}
            or evidence["kind"] != part["kind"]
            or evidence["attachment_id"] != part["attachment_id"]
            or not isinstance(evidence["receipts"], list)
        ):
            raise ValueError("warm message part binding differs")
        if not part["source"]["text"].strip():
            if evidence["capture"] is not None or evidence["receipts"]:
                raise ValueError("empty message part cannot contain native evidence")
            statuses.append({"kind": part["kind"], "attachment_id": part["attachment_id"], "status": "empty"})
            continue
        jobs = warm_message_jobs(source, part, evidence["capture"])
        if len(evidence["receipts"]) != len(jobs):
            raise ValueError("warm message receipt coverage differs")
        for job, receipt in zip(jobs, evidence["receipts"], strict=True):
            if receipt["nodes"] != job["nodes"] or receipt["parser"] != job["parser"]:
                raise ValueError("unit native tree differs from captured message syntax")
            record = project_warm_source(job["source"], receipt, profile=profile)
            units.append({"origin": job["origin"], "record": record})
        statuses.append({"kind": part["kind"], "attachment_id": part["attachment_id"], "status": "parsed"})
    supported = sum(unit["record"]["status"] != "unsupported_source" for unit in units)
    record = seal(
        {
            "schema_version": MESSAGE_RECORD_SCHEMA,
            "profile_sha256": profile["content_sha256"],
            "source": deepcopy(source),
            "receipt": deepcopy(bundle),
            "units": units,
            "parts": statuses,
            "status": "unsupported_source" if not supported else "qualified" if supported == len(units) else "partial",
            "world_fact_asserted": False,
            "canonical_identity": False,
        }
    )
    if len(canonical_json(record).encode()) > MAX_RECORD_BYTES:
        raise ValueError("warm message record byte bound exceeded")
    return record
