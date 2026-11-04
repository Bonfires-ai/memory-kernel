"""Shared types for the memory_kernel search package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from memory_kernel.construction import (
    ConstructionTrace,
    GrammarWorkset,
    QueryConstruction,
)

GraphExpansionMode = Literal["off", "episode_bounded", "comparison_graphiti", "comparison_worldview_vector"]


@dataclass(frozen=True, slots=True)
class MemoryKernelConfig:
    """Static knobs for the memory kernel boundary."""

    default_top_k: int = 20
    candidate_limit: int = 100
    surface_limit: int = 12
    embed_query: bool = True
    hydrate_graph: bool = True
    graph_expansion_mode: GraphExpansionMode = "episode_bounded"
    include_temporal_surfaces: bool = True
    normalize_ascii: bool = True
    expected_artifact_families: tuple[str, ...] = ()
    use_fcg: bool = True  # FCG grammar selection is mandatory for host-service search
    fcg_artifact_store: Literal["local", "s3"] = "local"
    fcg_artifact_cache_dir: str = "/tmp/memory-kernel-fcg"
    fcg_strict_startup_health: bool = True
    fcg_health_timeout_s: float = 10.0
    fcg_topic_similarity_threshold: float = 0.3  # min cosine sim for topic grammar match
    fcg_topic_top_k: int = 8
    fcg_comprehension_attempt_limit: int = 3
    fcg_grammar_cache_size: int = 4
    fcg_precision_miss_policy: str = "continue"
    fcg_miss_learning_enabled: bool = True
    fcg_learning_top_evidence_k: int = 3
    fcg_learning_abstract_support_threshold: int = 2
    fcg_learned_overlay_enabled: bool = True
    fcg_learned_overlay_min_score: float = 0.45
    construct_candidate_limit: int = 64
    ecs_search_enabled: bool = False
    context_packet_enabled: bool = False
    # Deictic frameshift-stop for extraction. The kernel library knob; a host application
    # may drive it from its own configuration.
    # Default OFF ⇒ byte-identical extraction.
    deictic_frameshift_stop: bool = False


@dataclass(frozen=True, slots=True)
class SearchScope:
    """Hard filters applied to index and search operations."""

    bonfire_id: str | None = None
    topic_ids: tuple[str, ...] = ()
    episode_ids: tuple[str, ...] = ()
    fact_ids: tuple[str, ...] = ()
    statement_ids: tuple[str, ...] = ()
    artifact_families: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    surface_families: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SurfaceQuery:
    """A deterministic query surface emitted by the grammar planner."""

    family: str
    query_text: str
    tokens: tuple[str, ...]
    lemmas: tuple[str, ...]
    required_terms: tuple[str, ...] = ()
    optional_terms: tuple[str, ...] = ()
    excluded_terms: tuple[str, ...] = ()
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class GrammarQueryPlan:
    """Query interpretation used to drive deterministic search surfaces."""

    query: str
    workset: GrammarWorkset
    tokens: tuple[str, ...]
    lemmas: tuple[str, ...]
    traces: tuple[ConstructionTrace, ...]
    expected_artifact_families: tuple[str, ...]
    surface_queries: tuple[SurfaceQuery, ...]
    temporal: bool = False
    count_hint: bool = False
    list_hint: bool = False
    set_intersection_hint: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def trace(self, kind: str) -> ConstructionTrace | None:
        return next((trace for trace in self.traces if trace.recipe_kind == kind and trace.matched), None)

    def has_construction(self, kind: str) -> bool:
        return self.trace(kind) is not None

    def construction(self, kind: str) -> QueryConstruction | None:
        trace = self.trace(kind)
        if trace is None:
            return None
        return QueryConstruction(
            kind=kind,
            subjects=frozenset(trace.bound_slots.get("subjects", ())),
            actions=frozenset(trace.bound_slots.get("actions", ())),
            slots=frozenset(trace.bound_slots.get("slots", ())),
            operators=frozenset(trace.bound_slots.get("operators", ())),
            operator_frames=frozenset(trace.bound_slots.get("operator_frames", ())),
            connectives=frozenset(trace.bound_slots.get("connectives", ())),
            dialogue_act=frozenset(trace.bound_slots.get("dialogue_act", ())),
            bound_slots=trace.bound_slots,
            trace=trace,
        )

    def diagnostics(self) -> list[dict[str, Any]]:
        return [trace.as_dict() for trace in self.traces]
