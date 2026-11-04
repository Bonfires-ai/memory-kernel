"""Durable, source-qualified graphs from checked warm-session receipts.

This is a distinct admission adapter, not a legacy compositional admission.
The host pins the resident runtime and leases the sender; hashes establish
integrity, not authentication. Stored native evidence can be replayed offline.
"""

from __future__ import annotations

import json
import re
import sqlite3
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from memory_kernel.construction.fcg.native_analysis import decode_tagged, proper_list
from memory_kernel.construction.fcg.native_discourse_layer import NativeDiscourseContract
from memory_kernel.construction.fcg.native_frame_transport import source_tokens_from_nodes
from memory_kernel.construction.fcg.native_warm_session import native_warm_bindings, native_warm_source_evidence
from memory_kernel.construction.fcg.native_warm_session_lisp import (
    SCHEMA,
    SUPPLIED_SCHEMA,
    native_warm_session_config_digest,
    native_warm_supplied_config_digest,
)
from memory_kernel.construction.memory_discourse_contract import check_discourse_source, discourse_id
from memory_kernel.construction.memory_discourse_native import _native_token_char_spans
from memory_kernel.construction.memory_semantic_algebra import (
    LexicalEmbedding,
    PredicateAtom,
    SemanticGraph,
    SemanticReferent,
    SemanticRoot,
    SemanticWitness,
    UnresolvedContent,
    semantic_algebra_profile,
    semantic_graph_to_wire,
    validate_semantic_graph,
)
from memory_kernel.construction.memory_semantic_jev import check_seal, seal
from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256

PROFILE_SCHEMA = "native-warm-graph-profile/v1"
MESSAGE_PROFILE_SCHEMA = "native-warm-graph-profile/v2"
LEARNED_MESSAGE_PROFILE_SCHEMA = "native-warm-graph-profile/v3"
MESSAGE_PROFILE_SCHEMAS = {MESSAGE_PROFILE_SCHEMA, LEARNED_MESSAGE_PROFILE_SCHEMA}
MESSAGE_RECORD_SCHEMA = "native-warm-message-record/v1"
RECORD_SCHEMA = "native-warm-graph-record/v1"
CHECKPOINT_SCHEMA = "native-warm-graph-checkpoint/v1"
ADAPTER_VERSION = "checked-warm-semantic-projection/v1"
MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_STORE_SOURCES = 4096
ZERO = "0" * 64


def warm_graph_profile(
    *,
    contract,
    scope,
    runtime_sha256,
    grammar_id="memory-kernel-propbank-grammar-1",
    source_units=False,
    source_native_seconds=5,
    role_package="FCG-PROPBANK",
):
    if (
        not isinstance(scope, dict)
        or not scope
        or any(type(k) is not str or not k or type(v) is not str or not v for k, v in scope.items())
        or type(runtime_sha256) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", runtime_sha256)
    ):
        raise ValueError("warm graph requires an explicit scope and host-pinned runtime digest")
    config = native_warm_session_config_digest(contract, grammar_id, 32, 2000, 5, role_package=role_package)
    if type(source_units) is not bool:
        raise ValueError("source_units must be boolean")
    if role_package != "FCG-PROPBANK" and not source_units:
        raise ValueError("learned role namespace requires complete source-message ingestion")
    if (
        type(source_native_seconds) not in {int, float}
        or not 0 < source_native_seconds <= 25
        or (not source_units and source_native_seconds != 5)
    ):
        raise ValueError("invalid source native deadline")
    extra = (
        {}
        if not source_units
        else {
            "source_units": "complete-native-sentences-and-captions/v1",
            "source_config_sha256": native_warm_supplied_config_digest(
                contract, grammar_id, 32, 2000, source_native_seconds, role_package=role_package
            ),
        }
    )
    if source_units and source_native_seconds != 5:
        extra["source_native_seconds"] = float(source_native_seconds)
    schema = MESSAGE_PROFILE_SCHEMA if source_units else PROFILE_SCHEMA
    if role_package != "FCG-PROPBANK":
        schema = LEARNED_MESSAGE_PROFILE_SCHEMA
        extra["role_package"] = role_package
    return seal(
        {
            "schema_version": schema,
            "contract": asdict(contract),
            "scope": deepcopy(scope),
            "runtime_sha256": runtime_sha256,
            "grammar_id": grammar_id,
            "config_sha256": config,
            "adapter": ADAPTER_VERSION,
            "identity_policy": "source_occurrences_only",
            "max_record_bytes": MAX_RECORD_BYTES,
            "max_sources": MAX_STORE_SOURCES,
            **extra,
        }
    )


def check_warm_graph_profile(profile):
    schema = profile.get("schema_version")
    if schema not in {PROFILE_SCHEMA, *MESSAGE_PROFILE_SCHEMAS}:
        raise ValueError("unsupported warm graph profile")
    check_seal(profile, schema)
    expected = warm_graph_profile(
        contract=NativeDiscourseContract(**profile["contract"]),
        scope=profile["scope"],
        runtime_sha256=profile["runtime_sha256"],
        grammar_id=profile["grammar_id"],
        source_units=schema in MESSAGE_PROFILE_SCHEMAS,
        source_native_seconds=profile.get("source_native_seconds", 5),
        role_package=profile.get("role_package", "FCG-PROPBANK"),
    )
    if profile != expected:
        raise ValueError("warm graph profile differs from the supported contract")


def check_warm_receipt(receipt, profile, *, kind):
    check_warm_graph_profile(profile)
    source = receipt["source"]
    supplied = kind == "source" and profile["schema_version"] in MESSAGE_PROFILE_SCHEMAS
    config = profile["source_config_sha256"] if supplied else profile["config_sha256"]
    if (
        receipt["schema"] != (SUPPLIED_SCHEMA if supplied else SCHEMA)
        or receipt["config_sha256"] != config
        or type(receipt["session"]) is not str
        or not re.fullmatch(r"NATIVE-WARM-[0-9A-F]{32}", receipt["session"])
        or set(source) != {"id", "text", "sha256", "kind"}
        or source["kind"] != kind
        or type(source["text"]) is not str
        or not source["text"].strip()
        or len(source["text"]) > 2000
        or "\x00" in source["text"]
        or type(source["id"]) is not str
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", source["id"])
    ):
        raise ValueError("warm receipt source, configuration or session differs")
    import hashlib

    if source["sha256"] != hashlib.sha256(source["text"].encode()).hexdigest():
        raise ValueError("warm receipt text digest differs")
    for key in ("source_count", "parser_calls", "active_sources", "epoch", "identity_preserved"):
        if type(receipt[key]) is not int:
            raise ValueError("warm receipt accounting must be integral")
    if (
        not 1 <= receipt["source_count"] <= 32
        or receipt["parser_calls"] != (0 if supplied else receipt["source_count"])
        or not 1 <= receipt["active_sources"] <= receipt["source_count"]
        or not 0 <= receipt["epoch"] <= 32
        or receipt["identity_preserved"] != 1
    ):
        raise ValueError("warm receipt accounting differs")


def project_warm_source(source, receipt, *, profile):
    """Reproduce the typed graph from exact native witnesses, without a parser."""
    check_discourse_source(source)
    check_warm_receipt(receipt, profile, kind="source")
    if (
        source["scope"] != profile["scope"]
        or source["message_id"] != receipt["source"]["id"]
        or source["text"] != receipt["source"]["text"]
    ):
        raise ValueError("warm admission source binding differs")
    if len(canonical_json(receipt).encode()) > MAX_RECORD_BYTES:
        raise ValueError("warm receipt byte bound exceeded")
    contract = NativeDiscourseContract(**profile["contract"])
    analysis = native_warm_source_evidence(receipt, contract, role_package=profile.get("role_package", "FCG-PROPBANK"))
    evidence_sha = content_sha256(receipt)
    source_sha = source["content_sha256"]
    witnesses = [
        SemanticWitness(discourse_id("witness", source_sha), "source", ((0, len(source["text"])),), evidence_sha)
    ]
    refs, occurrences, nodes = [], [], []
    claims = [proper_list(row) for row in proper_list(decode_tagged(receipt["claims"]))]
    if not claims:
        wid, nid = discourse_id("unresolved-witness", source_sha), discourse_id("unresolved", source_sha)
        witnesses.append(SemanticWitness(wid, "unresolved", ((0, len(source["text"])),), evidence_sha))
        nodes.append(UnresolvedContent(nid, "unsupported_native_source", "proposition", wid))
        root = SemanticRoot(discourse_id("root", source_sha), nid, witnesses[0].witness_id, "unresolved", "unresolved")
        unresolved = ("unsupported_native_source",)
    else:
        _, scope, _sense, event, subject, obj, _label, _answer, _speaker = claims[0]
        application = receipt["qualification"][0]
        bindings = native_warm_bindings(application, "warm-source/" + scope)
        parser_nodes = [{key.replace("_", "--"): value for key, value in row.items()} for row in receipt["nodes"]]
        tokens = source_tokens_from_nodes(parser_nodes, source["text"])
        token_spans = _native_token_char_spans(source["text"], tokens, 0)

        def referent(unit, sort):
            start, end = analysis.units_by_name[unit].token_range
            if not 0 <= start < end <= len(token_spans):
                raise ValueError("native occurrence token span differs")
            spans = ((token_spans[start][0], token_spans[end - 1][1]),)
            oid = discourse_id("occurrence", source_sha, sort, spans)
            if any(row.reference_id == oid for row in refs):
                return oid
            wid = discourse_id("witness", oid)
            witnesses.append(SemanticWitness(wid, "frame" if sort == "event" else "reference", spans, evidence_sha))
            refs.append(SemanticReferent(oid, sort, oid, None, wid))
            occurrences.append(
                {
                    "occurrence_id": oid,
                    "kind": sort,
                    "spans": [list(s) for s in spans],
                    "quote": source["text"][spans[0][0] : spans[0][1]],
                }
            )
            return oid

        def predicate(unit, roleset, roles):
            event_ref = referent(unit, "event")
            frame_id = discourse_id("frame", source_sha, roleset, event_ref, roles)
            wid = next(ref.witness_id for ref in refs if ref.reference_id == event_ref)
            atom = PredicateAtom(
                discourse_id("predicate", frame_id), frame_id, roleset, event_ref, tuple(sorted(roles)), wid
            )
            nodes.append(atom)
            return atom.node_id

        body = predicate(
            event,
            contract.roleset,
            (
                (contract.subject_role.lower(), referent(subject, "entity")),
                (contract.object_role.lower(), referent(obj, "entity")),
            ),
        )
        if scope == "reported":
            reporter = referent(bindings["?SPEAKER"], "entity")
            complement = referent(bindings["?COMPLEMENT"], "proposition")
            owner = predicate(bindings["?REPORT"], "say.01", (("arg0", reporter), ("arg1", complement)))
            wid = discourse_id("scope-witness", source_sha)
            # The qualifier emits one application in its own ordered envelope.
            witnesses.append(
                SemanticWitness(
                    wid,
                    "scope",
                    ((0, len(source["text"])),),
                    content_sha256(application),
                    0,
                    0,
                    (None, application["construction"]),
                )
            )
            embedding = LexicalEmbedding(discourse_id("embedding", source_sha), owner, reporter, body, "reported", wid)
            nodes.append(embedding)
            body = embedding.node_id
        root = SemanticRoot(
            discourse_id("root", source_sha), body, witnesses[0].witness_id, "qualified_description", "explicit"
        )
        unresolved = ()
    semantic = SemanticGraph(
        source_sha,
        len(source["text"]),
        evidence_sha,
        semantic_algebra_profile(profile["content_sha256"])["content_sha256"],
        tuple(refs),
        tuple(witnesses),
        tuple(nodes),
        (root,),
        unresolved,
    )
    validate_semantic_graph(semantic, source=source)
    record = seal(
        {
            "schema_version": RECORD_SCHEMA,
            "profile_sha256": profile["content_sha256"],
            "source": deepcopy(source),
            "receipt": deepcopy(receipt),
            "occurrences": occurrences,
            "semantic": semantic_graph_to_wire(semantic),
            "status": receipt["status"],
            "world_fact_asserted": False,
            "canonical_identity": False,
        }
    )
    if len(canonical_json(record).encode()) > MAX_RECORD_BYTES:
        raise ValueError("warm graph record byte bound exceeded")
    return record


def project_warm_record(source, receipt, *, profile):
    if profile["schema_version"] in MESSAGE_PROFILE_SCHEMAS:
        from memory_kernel.construction.memory_warm_message import project_warm_message

        return project_warm_message(source, receipt, profile=profile)
    return project_warm_source(source, receipt, profile=profile)


class WarmGraphStore:
    """Append-only transactional graph records; readers open SQLite in read-only mode.

    A trusted checkpoint is required for serving, so re-sealing modified records
    cannot silently replace the accepted corpus. Iteration retains one record.
    """

    def __init__(self, path, *, profile, readonly=False, checkpoint=None):
        check_warm_graph_profile(profile)
        self.profile = deepcopy(profile)
        self.readonly = readonly
        path = Path(path).resolve()
        self.db = sqlite3.connect(path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc"), uri=True)
        try:
            if readonly and checkpoint is None:
                raise ValueError("read-only search requires a trusted graph checkpoint")
            if not readonly:
                self.db.execute("PRAGMA synchronous=FULL")
                with self.db:
                    self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    self.db.execute(
                        "CREATE TABLE IF NOT EXISTS records (sequence INTEGER PRIMARY KEY, "
                        "message_id TEXT UNIQUE NOT NULL, source_sha TEXT NOT NULL, "
                        "payload TEXT NOT NULL, head TEXT NOT NULL)"
                    )
                    self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('profile', ?)", (canonical_json(profile),))
            stored = self.db.execute("SELECT value FROM metadata WHERE key='profile'").fetchone()
            if stored is None or stored[0] != canonical_json(profile):
                raise ValueError("stored graph profile differs")
            if readonly:
                self.db.execute("BEGIN")
            current = self.verify()
            if checkpoint is not None and current != checkpoint:
                raise ValueError("stored graph checkpoint differs")
        except BaseException:
            self.db.close()
            raise

    def checkpoint(self):
        row = self.db.execute("SELECT sequence, head FROM records ORDER BY sequence DESC LIMIT 1").fetchone()
        return seal(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "profile_sha256": self.profile["content_sha256"],
                "length": 0 if row is None else row[0],
                "head": ZERO if row is None else row[1],
            }
        )

    def records(self):
        for (payload,) in self.db.execute("SELECT payload FROM records ORDER BY sequence"):
            if len(payload.encode()) > MAX_RECORD_BYTES:
                raise ValueError("stored graph record exceeds byte bound")
            yield json.loads(payload)

    def verify(self):
        head = ZERO
        rows = self.db.execute("SELECT sequence, message_id, source_sha, payload, head FROM records ORDER BY sequence")
        for count, (sequence, message_id, source_sha, payload, stored_head) in enumerate(rows, 1):
            if sequence != count or count > MAX_STORE_SOURCES or len(payload.encode()) > MAX_RECORD_BYTES:
                raise ValueError("stored graph sequence or byte bound differs")
            record = json.loads(payload)
            schema = (
                MESSAGE_RECORD_SCHEMA if self.profile["schema_version"] in MESSAGE_PROFILE_SCHEMAS else RECORD_SCHEMA
            )
            check_seal(record, schema)
            if (
                record != project_warm_record(record["source"], record["receipt"], profile=self.profile)
                or message_id != record["source"]["message_id"]
                or source_sha != record["source"]["content_sha256"]
            ):
                raise ValueError("stored graph projection differs from native witnesses")
            head = content_sha256([head, record["content_sha256"]])
            if head != stored_head:
                raise ValueError("stored graph hash chain differs")
        return self.checkpoint()

    def existing(self, source):
        check_discourse_source(source)
        if source["scope"] != self.profile["scope"]:
            raise ValueError("source scope differs from graph")
        row = self.db.execute(
            "SELECT source_sha, payload FROM records WHERE message_id=?", (source["message_id"],)
        ).fetchone()
        if row is None:
            return None
        if row[0] != source["content_sha256"]:
            raise ValueError("immutable source identifier already committed with different content")
        return json.loads(row[1])

    def append(self, source, receipt):
        if self.readonly:
            raise ValueError("cannot append to a read-only graph")
        record = project_warm_record(source, receipt, profile=self.profile)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.existing(source)
            if existing is not None:
                if existing != record:
                    raise ValueError("source already has different native evidence")
                self.db.rollback()
                return existing
            previous = self.checkpoint()
            if previous["length"] >= MAX_STORE_SOURCES:
                raise ValueError("durable graph source budget exhausted")
            self.db.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                (
                    previous["length"] + 1,
                    source["message_id"],
                    source["content_sha256"],
                    canonical_json(record),
                    content_sha256([previous["head"], record["content_sha256"]]),
                ),
            )
            self.db.commit()
            return record
        except BaseException:
            self.db.rollback()
            raise

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
