"""Bounded typed composition, not a replacement for native semantic authority.

The codec checks structure and source bindings. Only a checked native adapter
can license a graph; journal admission must reproduce that adapter's output.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

from memory_kernel.construction.memory_discourse_contract import check_discourse_source
from memory_kernel.construction.memory_semantic_jev import seal
from memory_kernel.construction.mined_affordance_contract import canonical_json

ALGEBRA_SCHEMA = "discourse-semantic-algebra/v2"
ALGEBRA_PROFILE_SCHEMA = "discourse-semantic-profile/v2"
SCOPE_ADAPTER_VERSION = "native-scope-composition/v1"
DISCOURSE_ADAPTER_VERSION = "native-discourse-composition/v1"
MAX_SEMANTIC_NODES = 2048
MAX_SEMANTIC_REFERENTS = 256
MAX_SEMANTIC_WITNESSES = 512
MAX_SEMANTIC_ROOTS = 128
MAX_SEMANTIC_DEPTH = 16
MAX_SEMANTIC_ALTERNATIVES = 16
MAX_SEMANTIC_VISITS = 8192
MAX_SEMANTIC_BYTES = 4 * 1024 * 1024


class SemanticAlgebraError(ValueError):
    """A source-bound term, type or accessibility region is malformed."""


@dataclass(frozen=True, slots=True)
class SemanticWitness:
    witness_id: str
    kind: Literal[
        "source", "frame", "scope", "question", "condition", "reference", "attachment", "derived", "unresolved"
    ]
    spans: tuple[tuple[int, int], ...]
    evidence_sha256: str
    envelope_index: int | None = None
    chain_position: int | None = None
    construction: tuple[str | None, str] | None = None


@dataclass(frozen=True, slots=True)
class SemanticReferent:
    reference_id: str
    sort: Literal["entity", "event", "attachment", "proposition", "kind"]
    occurrence_id: str | None
    binder_id: str | None
    witness_id: str


@dataclass(frozen=True, slots=True)
class PredicateAtom:
    node_id: str
    frame_id: str
    roleset: str
    event_ref: str
    roles: tuple[tuple[str, str], ...]
    witness_id: str


@dataclass(frozen=True, slots=True)
class Negation:
    node_id: str
    body: str
    witness_id: str


@dataclass(frozen=True, slots=True)
class Modal:
    node_id: str
    body: str
    value: Literal["possible", "necessary", "unknown"]
    witness_id: str


@dataclass(frozen=True, slots=True)
class LexicalEmbedding:
    node_id: str
    owner: str
    holder: str
    body: str
    reading: Literal["reported", "intended"]
    witness_id: str


@dataclass(frozen=True, slots=True)
class Conditional:
    node_id: str
    antecedent: str
    consequent: str
    bindings: tuple[str, ...]
    witness_id: str


@dataclass(frozen=True, slots=True)
class Question:
    node_id: str
    body: str
    bindings: tuple[str, ...]
    mode: Literal["wh", "polar"]
    witness_id: str


@dataclass(frozen=True, slots=True)
class AlternativeSet:
    node_id: str
    choices: tuple[str, ...]
    witness_id: str


@dataclass(frozen=True, slots=True)
class UnresolvedContent:
    node_id: str
    reason: str
    expected_type: Literal["proposition", "question"]
    witness_id: str


SemanticTerm = (
    PredicateAtom | Negation | Modal | LexicalEmbedding | Conditional | Question | AlternativeSet | UnresolvedContent
)


@dataclass(frozen=True, slots=True)
class SemanticRoot:
    root_id: str
    term_id: str
    witness_id: str
    projection_status: Literal["qualified_description", "question", "unresolved"]
    scope_status: Literal["explicit", "partial", "unresolved"]


@dataclass(frozen=True, slots=True)
class SemanticGraph:
    source_sha256: str
    source_length: int
    native_archive_sha256: str
    profile_sha256: str
    referents: tuple[SemanticReferent, ...]
    witnesses: tuple[SemanticWitness, ...]
    nodes: tuple[SemanticTerm, ...]
    roots: tuple[SemanticRoot, ...]
    unresolved: tuple[str, ...]
    # Session provenance only; native adapters leave this unbound.
    admission_version_sha256: str | None = None


_TERM_TAGS = {
    PredicateAtom: "predicate",
    Negation: "negation",
    Modal: "modal",
    LexicalEmbedding: "embedding",
    Conditional: "conditional",
    Question: "question",
    AlternativeSet: "alternatives",
    UnresolvedContent: "unresolved",
}


def _semantic_require_text(value, label, maximum=256):
    if type(value) is not str or not value or len(value) > maximum:
        raise SemanticAlgebraError(label + " must be bounded nonempty text")


def _semantic_require_sha(value, label):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SemanticAlgebraError(label + " must be a SHA-256 digest")


def semantic_algebra_profile(base_profile_sha256: str) -> dict:
    _semantic_require_sha(base_profile_sha256, "base profile")
    return seal(
        {
            "schema_version": ALGEBRA_PROFILE_SCHEMA,
            "base_profile_sha256": base_profile_sha256,
            "algebra_schema": ALGEBRA_SCHEMA,
            "scope_adapter": SCOPE_ADAPTER_VERSION,
            "discourse_adapter": DISCOURSE_ADAPTER_VERSION,
            "limits": {
                "nodes": MAX_SEMANTIC_NODES,
                "referents": MAX_SEMANTIC_REFERENTS,
                "witnesses": MAX_SEMANTIC_WITNESSES,
                "roots": MAX_SEMANTIC_ROOTS,
                "depth": MAX_SEMANTIC_DEPTH,
                "alternatives": MAX_SEMANTIC_ALTERNATIVES,
                "expanded_visits": MAX_SEMANTIC_VISITS,
                "bytes": MAX_SEMANTIC_BYTES,
            },
            "authority": "checked_native_projection_not_world_truth",
        }
    )


def _semantic_children(node):
    if type(node) in {Negation, Modal, Question}:
        return (("body", node.body),)
    if type(node) is LexicalEmbedding:
        return (("owner", node.owner), ("body", node.body))
    if type(node) is Conditional:
        return (("antecedent", node.antecedent), ("consequent", node.consequent))
    if type(node) is AlternativeSet:
        return tuple((f"alternative:{i}", child) for i, child in enumerate(node.choices))
    return ()


def _semantic_references(node):
    if type(node) is PredicateAtom:
        return (("$predicate", node.event_ref), *node.roles)
    if type(node) is LexicalEmbedding:
        return (("$holder", node.holder),)
    return ()


def _semantic_visits(graph):
    """Expand bounded root-relative regions; an embedding owner stays outside its body."""
    nodes = {node.node_id: node for node in graph.nodes}
    stack = [(root, root.term_id, (), (), False) for root in reversed(graph.roots)]
    result = []
    while stack:
        root, node_id, ancestors, regions, ambiguous = stack.pop()
        if node_id in ancestors or len(ancestors) >= MAX_SEMANTIC_DEPTH:
            raise SemanticAlgebraError("semantic cycle or depth bound exceeded")
        if len(result) >= MAX_SEMANTIC_VISITS:
            raise SemanticAlgebraError("semantic expanded-visit bound exceeded")
        node = nodes.get(node_id)
        if node is None:
            raise SemanticAlgebraError("semantic child is absent")
        result.append((root, node, ancestors, regions, ambiguous))
        for branch, child in reversed(_semantic_children(node)):
            next_regions = regions
            if type(node) in {Negation, Modal, Conditional, Question, AlternativeSet} or (
                type(node) is LexicalEmbedding and branch == "body"
            ):
                next_regions = (*regions, (node.node_id, _TERM_TAGS[type(node)], branch, node.witness_id))
            stack.append((root, child, (*ancestors, node_id), next_regions, ambiguous or type(node) is AlternativeSet))
    return result


def _semantic_result_types(graph, visits):
    nodes = {node.node_id: node for node in graph.nodes}
    result = {}
    for _root, node, _ancestors, _regions, _ambiguous in reversed(visits):
        if node.node_id in result:
            continue
        children = [result[child] for _branch, child in _semantic_children(node)]
        if type(node) in {Negation, Modal, Conditional, Question} and any(x != "proposition" for x in children):
            raise SemanticAlgebraError("proposition constructor received a question")
        if type(node) is LexicalEmbedding and type(nodes[node.owner]) is not PredicateAtom:
            raise SemanticAlgebraError("embedding owner must be a native predicate atom")
        if type(node) is AlternativeSet and len(set(children)) != 1:
            raise SemanticAlgebraError("semantic alternatives have different result types")
        result[node.node_id] = (
            "question"
            if type(node) is Question
            else node.expected_type
            if type(node) is UnresolvedContent
            else children[0]
            if type(node) is AlternativeSet
            else "proposition"
        )
    return result


def _semantic_guaranteed_references(visits, *, introduction=False):
    """Intersection across alternatives; closed scopes cannot introduce outer bindings."""
    result = {}
    for _root, node, _ancestors, _regions, _ambiguous in reversed(visits):
        if node.node_id in result:
            continue
        children = [result[child] for _branch, child in _semantic_children(node)]
        own = {ref for _role, ref in _semantic_references(node)}
        if type(node) is AlternativeSet:
            result[node.node_id] = set.intersection(*children)
        elif introduction and type(node) in {Negation, Modal, Conditional, Question}:
            result[node.node_id] = set()
        elif introduction and type(node) is LexicalEmbedding:
            result[node.node_id] = own | children[0]
        else:
            result[node.node_id] = own.union(*children)
    return result


def validate_semantic_graph(graph: SemanticGraph, *, source: Mapping | None = None) -> None:
    """Validate a closed structural algebra; this does not authenticate native evidence."""
    if type(graph) is not SemanticGraph:
        raise SemanticAlgebraError("exact semantic graph type required")
    for value, label in (
        (graph.source_sha256, "source"),
        (graph.native_archive_sha256, "native archive"),
        (graph.profile_sha256, "semantic profile"),
    ):
        _semantic_require_sha(value, label)
    if graph.admission_version_sha256 is not None:
        _semantic_require_sha(graph.admission_version_sha256, "admission version")
    if type(graph.source_length) is not int or not 0 <= graph.source_length <= 65536:
        raise SemanticAlgebraError("semantic source length differs")
    if source is not None:
        check_discourse_source(source)
        if graph.source_sha256 != source["content_sha256"] or graph.source_length != len(source["text"]):
            raise SemanticAlgebraError("semantic graph belongs to another source")
    for rows, expected, maximum, label in (
        (graph.referents, SemanticReferent, MAX_SEMANTIC_REFERENTS, "referents"),
        (graph.witnesses, SemanticWitness, MAX_SEMANTIC_WITNESSES, "witnesses"),
        (graph.roots, SemanticRoot, MAX_SEMANTIC_ROOTS, "roots"),
    ):
        if type(rows) is not tuple or len(rows) > maximum or any(type(row) is not expected for row in rows):
            raise SemanticAlgebraError("semantic " + label + " types or bounds differ")
    if (
        type(graph.nodes) is not tuple
        or len(graph.nodes) > MAX_SEMANTIC_NODES
        or any(type(node) not in _TERM_TAGS for node in graph.nodes)
    ):
        raise SemanticAlgebraError("semantic node type or count differs")
    if not graph.nodes or not graph.roots or not graph.witnesses:
        raise SemanticAlgebraError("source requires a semantic root or explicit unresolved content")
    if type(graph.unresolved) is not tuple or len(graph.unresolved) > 256:
        raise SemanticAlgebraError("semantic unresolved bounds differ")
    for gap in graph.unresolved:
        _semantic_require_text(gap, "semantic gap", 4096)
    witnesses, refs, nodes, roots = {}, {}, {}, {}
    for rows, registry, attribute in (
        (graph.witnesses, witnesses, "witness_id"),
        (graph.referents, refs, "reference_id"),
        (graph.nodes, nodes, "node_id"),
        (graph.roots, roots, "root_id"),
    ):
        for row in rows:
            identifier = getattr(row, attribute)
            _semantic_require_text(identifier, attribute)
            if identifier in registry:
                raise SemanticAlgebraError("duplicate semantic " + attribute)
            registry[identifier] = row
    for witness in graph.witnesses:
        _semantic_require_text(witness.kind, "witness kind")
        if witness.kind not in {
            "source",
            "frame",
            "scope",
            "question",
            "condition",
            "reference",
            "attachment",
            "derived",
            "unresolved",
        }:
            raise SemanticAlgebraError("unsupported semantic witness kind")
        _semantic_require_sha(witness.evidence_sha256, "native evidence")
        if type(witness.spans) is not tuple or len(witness.spans) > 256:
            raise SemanticAlgebraError("semantic witness span bounds differ")
        cursor = 0
        for span in witness.spans:
            if (
                type(span) is not tuple
                or len(span) != 2
                or any(type(x) is not int for x in span)
                or not cursor <= span[0] < span[1] <= graph.source_length
            ):
                raise SemanticAlgebraError("semantic witness spans overlap or escape source")
            cursor = span[1]
        if not witness.spans and graph.source_length and witness.kind not in {"attachment", "derived"}:
            raise SemanticAlgebraError("textual witness lacks a source span")
        if witness.kind == "source" and witness.spans != (((0, graph.source_length),) if graph.source_length else ()):
            raise SemanticAlgebraError("source witness must cover the complete original source")
        for index in (witness.envelope_index, witness.chain_position):
            if index is not None and (type(index) is not int or not 0 <= index <= 65535):
                raise SemanticAlgebraError("native witness position differs")
        if witness.construction is not None:
            if type(witness.construction) is not tuple or len(witness.construction) != 2:
                raise SemanticAlgebraError("native witness construction differs")
            package, name = witness.construction
            if package is not None:
                _semantic_require_text(package, "construction package")
            _semantic_require_text(name, "construction name")
        if witness.kind in {"scope", "question", "condition"} and (
            witness.envelope_index is None or witness.chain_position is None or witness.construction is None
        ):
            raise SemanticAlgebraError("native operator witness lacks application coordinates")
    for ref in graph.referents:
        _semantic_require_text(ref.sort, "referent sort")
        _semantic_require_text(ref.witness_id, "referent witness")
        if ref.binder_id is not None:
            _semantic_require_text(ref.binder_id, "variable binder")
        if ref.sort not in {"entity", "event", "attachment", "proposition", "kind"} or ref.witness_id not in witnesses:
            raise SemanticAlgebraError("semantic referent sort or witness differs")
        if ref.binder_id is None:
            if ref.occurrence_id != ref.reference_id:
                raise SemanticAlgebraError("concrete reference must retain its exact occurrence identity")
            expected_kind = {
                "entity": "reference",
                "event": "frame",
                "attachment": "attachment",
                "proposition": "reference",
                "kind": "reference",
            }[ref.sort]
            if witnesses[ref.witness_id].kind != expected_kind:
                raise SemanticAlgebraError("concrete reference has an incompatible witness kind")
        elif (
            ref.occurrence_id is not None
            or ref.binder_id not in nodes
            or type(nodes[ref.binder_id]) not in {Question, Conditional}
            or ref.reference_id not in nodes[ref.binder_id].bindings
        ):
            raise SemanticAlgebraError("semantic variable lacks its declared binder")
        else:
            binder = nodes[ref.binder_id]
            witness = witnesses[ref.witness_id]
            binder_witness = witnesses.get(binder.witness_id)
            expected_kind = "question" if type(binder) is Question else "condition"
            if (
                binder_witness is None
                or witness.kind != expected_kind
                or (witness.envelope_index, witness.chain_position, witness.construction)
                != (binder_witness.envelope_index, binder_witness.chain_position, binder_witness.construction)
            ):
                raise SemanticAlgebraError("variable witness differs from its binder application")
        if ref.sort == "attachment" and (ref.binder_id is not None or witnesses[ref.witness_id].kind != "attachment"):
            raise SemanticAlgebraError("attachment reference differs from attachment witness")
    for node in graph.nodes:
        _semantic_require_text(node.witness_id, "node witness")
        if node.witness_id not in witnesses:
            raise SemanticAlgebraError("semantic node has no witness")
        allowed_witnesses = {
            PredicateAtom: {"frame"},
            Negation: {"scope"},
            Modal: {"scope"},
            LexicalEmbedding: {"scope"},
            Conditional: {"condition"},
            Question: {"question"},
            AlternativeSet: {"derived"},
            UnresolvedContent: {"unresolved", "derived"},
        }
        if witnesses[node.witness_id].kind not in allowed_witnesses[type(node)]:
            raise SemanticAlgebraError("semantic constructor has an incompatible witness kind")
        if type(node) is PredicateAtom:
            _semantic_require_text(node.frame_id, "native frame")
            _semantic_require_text(node.roleset, "native roleset")
            _semantic_require_text(node.event_ref, "predicate event reference")
            if type(node.roleset) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*\.[0-9]+", node.roleset) is None:
                raise SemanticAlgebraError("predicate requires an exact native roleset")
            if node.event_ref not in refs or refs[node.event_ref].sort != "event":
                raise SemanticAlgebraError("predicate event reference differs")
            if (
                type(node.roles) is not tuple
                or len(node.roles) > 6
                or any(
                    type(pair) is not tuple
                    or len(pair) != 2
                    or type(pair[0]) is not str
                    or re.fullmatch(r"arg[0-5]", pair[0]) is None
                    for pair in node.roles
                )
                or len({role for role, _ref in node.roles}) != len(node.roles)
                or tuple(sorted(node.roles)) != node.roles
            ):
                raise SemanticAlgebraError("predicate role bindings differ")
        elif type(node) is Modal and (
            type(node.value) is not str or node.value not in {"possible", "necessary", "unknown"}
        ):
            raise SemanticAlgebraError("unsupported modal value")
        elif type(node) is LexicalEmbedding and (
            type(node.reading) is not str or node.reading not in {"reported", "intended"}
        ):
            raise SemanticAlgebraError("unsupported embedding reading")
        elif type(node) is UnresolvedContent:
            _semantic_require_text(node.reason, "unresolved reason", 4096)
            if type(node.expected_type) is not str or node.expected_type not in {"proposition", "question"}:
                raise SemanticAlgebraError("unresolved result type differs")
        if type(node) in {Question, Conditional} and (
            type(node.bindings) is not tuple
            or len(node.bindings) > 16
            or any(type(ref) is not str for ref in node.bindings)
            or len(set(node.bindings)) != len(node.bindings)
            or any(ref not in refs or refs[ref].binder_id != node.node_id for ref in node.bindings)
        ):
            raise SemanticAlgebraError("binder ownership or variable count differs")
        if type(node) is Question and (
            type(node.mode) is not str or node.mode not in {"wh", "polar"} or (node.mode == "wh") != bool(node.bindings)
        ):
            raise SemanticAlgebraError("question mode and requested variables differ")
        if type(node) is AlternativeSet and (
            type(node.choices) is not tuple
            or not 2 <= len(node.choices) <= MAX_SEMANTIC_ALTERNATIVES
            or any(type(x) is not str for x in node.choices)
            or len(set(node.choices)) != len(node.choices)
        ):
            raise SemanticAlgebraError("semantic alternatives are duplicate or outside bounds")
        for _branch, child in _semantic_children(node):
            if type(child) is not str or child not in nodes:
                raise SemanticAlgebraError("semantic child is absent")
        for _role, ref in _semantic_references(node):
            if type(ref) is not str or ref not in refs:
                raise SemanticAlgebraError("semantic role reference is absent")
    for root in graph.roots:
        _semantic_require_text(root.term_id, "root term")
        _semantic_require_text(root.witness_id, "root witness")
        _semantic_require_text(root.projection_status, "root projection")
        _semantic_require_text(root.scope_status, "root scope")
        if (
            root.term_id not in nodes
            or root.witness_id not in witnesses
            or root.projection_status not in {"qualified_description", "question", "unresolved"}
            or root.scope_status not in {"explicit", "partial", "unresolved"}
        ):
            raise SemanticAlgebraError("semantic root, witness or projection status differs")
        if root.witness_id != nodes[root.term_id].witness_id and witnesses[root.witness_id].kind != "source":
            raise SemanticAlgebraError("semantic root witness is unrelated to its term or whole source")
        if type(nodes[root.term_id]) is UnresolvedContent and root.projection_status != "unresolved":
            raise SemanticAlgebraError("unresolved root requires an unresolved projection")
    if len(canonical_json(asdict(graph)).encode("utf-8")) > MAX_SEMANTIC_BYTES:
        raise SemanticAlgebraError("semantic graph byte bound exceeded")
    visits = _semantic_visits(graph)
    for root, node, _ancestors, _regions, _ambiguous in visits:
        if root.scope_status == "explicit" and (
            type(node) is UnresolvedContent or (type(node) is Modal and node.value == "unknown")
        ):
            raise SemanticAlgebraError("unresolved content cannot have an explicit root scope")
    if {node.node_id for _root, node, _ancestors, _regions, _ambiguous in visits} != set(nodes):
        raise SemanticAlgebraError("semantic graph contains unreachable terms")
    result_types = _semantic_result_types(graph, visits)
    guaranteed = _semantic_guaranteed_references(visits)
    introductions = _semantic_guaranteed_references(visits, introduction=True)
    for node in graph.nodes:
        if type(node) is LexicalEmbedding and node.holder not in dict(nodes[node.owner].roles).values():
            raise SemanticAlgebraError("embedding holder is not a role of its owner predicate")
        if type(node) is Question and not set(node.bindings) <= guaranteed[node.body]:
            raise SemanticAlgebraError("question variable absent from a body alternative")
        if type(node) is Conditional and not set(node.bindings) <= introductions[node.antecedent]:
            raise SemanticAlgebraError("conditional binding is not introduced in every antecedent alternative")
    for root in graph.roots:
        if (root.projection_status == "question" and result_types[root.term_id] != "question") or (
            root.projection_status == "qualified_description" and result_types[root.term_id] != "proposition"
        ):
            raise SemanticAlgebraError("semantic root projection has another result type")
    bound_uses = {ref.reference_id: [] for ref in graph.referents if ref.binder_id is not None}
    for _root, node, ancestors, regions, _ambiguous in visits:
        for _role, reference in _semantic_references(node):
            binder = refs[reference].binder_id
            if binder is not None:
                if binder not in ancestors:
                    raise SemanticAlgebraError("semantic variable escapes its binder region")
                bound_uses[reference].append(regions)
    for variable, uses in bound_uses.items():
        binder = nodes[refs[variable].binder_id]
        if not uses or (
            type(binder) is Conditional
            and not any(
                any(region[0] == binder.node_id and region[2] == "antecedent" for region in regions) for regions in uses
            )
        ):
            raise SemanticAlgebraError("bound variable is unused or absent from its introducing antecedent")


def semantic_graph_to_wire(graph: SemanticGraph) -> dict:
    validate_semantic_graph(graph)
    body = asdict(graph)
    body["nodes"] = [{"constructor": _TERM_TAGS[type(node)], **asdict(node)} for node in graph.nodes]
    body.update(schema_version=ALGEBRA_SCHEMA, authority="structural_native_projection", world_fact_asserted=False)
    encoded = canonical_json(body)
    if len(encoded.encode("utf-8")) > MAX_SEMANTIC_BYTES:
        raise SemanticAlgebraError("semantic wire byte bound exceeded")
    return seal(json.loads(encoded))
