"""Construction-grammar boundary for the standalone memory kernel.

The memory kernel owns this import surface. Callers should depend on this module
so the package can swap in a standalone or PyFCG-backed engine later.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from memory_kernel.ports import ConstructionRecipeProviderProtocol


class ConstructionPhase(Enum):
    """Ordered phases for match+merge construction pipeline.

    Recipes execute within their phase. After each phase completes,
    matched recipes merge their bound slots back into the transient
    structure for downstream phases to consume.
    """

    LEXICAL = auto()
    """Entity recognition, possessive detection, temporal anchor resolution."""

    ROLE = auto()
    """Semantic role binding — subject, action, object from event frames."""

    TEMPORAL_DISCOURSE = auto()
    """Temporal scoping, discourse relations, countable event projections."""

    META = auto()
    """Query-type classification — cardinality, set intersection, dialogue act."""


@dataclass
class TransientStructure:
    """Mutable accumulative structure built through phased construction application.

    Starts as a normalized GrammarWorkset (the initial transient structure in FCG
    terms). Each matched construction can merge its bound slots back into this
    structure, enriching it for downstream constructions and for surface-query
    generation. Footprints prevent the same recipe kind from being applied twice.

    This is the core innovation over the current static GrammarWorkset — it enables
    the FCG match+merge pattern where earlier constructions inform later ones.
    """

    workset: GrammarWorkset
    """The initial normalized surface, kept immutable for backward compat."""

    bound_slots: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Accumulated slot bindings from matched constructions, keyed by slot name."""

    antecedent_stack: list[dict[str, Any]] = field(default_factory=list)
    """Ordered list of antecedent contributions from matched constructions."""

    footprints: set[str] = field(default_factory=set)
    """Set of recipe_kind values that have already been applied (prevents reapplication)."""

    applied_recipes: list[ConstructionTrace] = field(default_factory=list)
    """Traces of matched constructions, in application order."""

    rejected_recipes: list[ConstructionTrace] = field(default_factory=list)
    """Traces of constructions that did not match, in application order."""

    def merge(self, trace: ConstructionTrace) -> TransientStructure:
        """Merge a matched construction's bindings back into the transient structure.

        Appends to antecedent_stack, records the trace in applied_recipes,
        and accumulates bound slots (merging new values into existing ones
        for key slots like subjects, actions, and slots).
        """
        if not trace.matched:
            self.rejected_recipes.append(trace)
            return self
        if trace.recipe_kind in self.footprints:
            return self
        self.applied_recipes.append(trace)
        self.footprints.add(trace.recipe_kind)
        self.antecedent_stack.append(trace.as_dict())
        for key, values in trace.bound_slots.items():
            if not values:
                continue
            existing = set(self.bound_slots.get(key, ()))
            existing.update(values)
            self.bound_slots[key] = tuple(sorted(existing))
        return self

    def slot(self, key: str) -> tuple[str, ...]:
        """Return accumulated values for a bound slot, or empty tuple."""
        return self.bound_slots.get(key, ())


@dataclass(frozen=True, slots=True)
class GrammarWorkset:
    """Package-owned normalized surface for construction recipes."""

    source_text: str
    source_kind: str
    tokens: tuple[str, ...]
    lemmas: tuple[str, ...]
    dependency_roles: dict[str, tuple[str, ...]]
    entities: tuple[str, ...]
    lexnames: dict[str, tuple[str, ...]]
    closed_class_operators: tuple[str, ...]
    temporal_cues: tuple[str, ...]
    discourse_cues: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConstructionTrace:
    """Package-owned trace of one construction recipe application."""

    recipe_kind: str = ""
    matched: bool = False
    bound_slots: dict[str, tuple[str, ...]] = field(default_factory=dict)
    artifact_kind: str | None = None
    artifact_preview: str = ""
    source_ids: tuple[str, ...] = ()
    rejection_reason: str | None = None
    trace_id: str = ""

    @classmethod
    def from_trace(cls, trace: Any) -> ConstructionTrace:
        """Copy any trace-shaped object into this frozen type, coercing each field by attribute."""
        return cls(
            recipe_kind=str(getattr(trace, "recipe_kind", "") or ""),
            matched=bool(getattr(trace, "matched", False)),
            bound_slots={
                str(key): tuple(str(value) for value in values)
                for key, values in dict(getattr(trace, "bound_slots", {}) or {}).items()
            },
            artifact_kind=_optional_text(getattr(trace, "artifact_kind", None)),
            artifact_preview=str(getattr(trace, "artifact_preview", "") or ""),
            source_ids=tuple(str(value) for value in getattr(trace, "source_ids", ()) if str(value or "")),
            rejection_reason=_optional_text(getattr(trace, "rejection_reason", None)),
            trace_id=str(getattr(trace, "trace_id", "") or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        """Render the trace as a JSON-safe row.

        When the trace carries no ``trace_id``, a deterministic one is derived from the recipe
        kind, artifact preview and source ids, so the same trace always serializes to the same id.
        """
        return {
            "recipe_kind": self.recipe_kind,
            "matched": self.matched,
            "bound_slots": {key: list(values) for key, values in self.bound_slots.items()},
            "artifact_kind": self.artifact_kind,
            "artifact_preview": self.artifact_preview,
            "source_ids": list(self.source_ids),
            "rejection_reason": self.rejection_reason,
            "trace_id": self.trace_id or _stable_id(self.recipe_kind, self.artifact_preview, *self.source_ids),
        }


@dataclass(frozen=True, slots=True)
class ConstructionRecipe:
    """Package-owned construction recipe facade over the current recipe inventory.

    Each recipe represents an FCG-style construction with:
    - A kind label (e.g. ``"EVENT_PROJECTION"``)
    - An artifact kind for the produced surface (e.g. ``"RoleBoundEventFrame"``)
    - Compatible search lanes for artifact admission
    - A pipeline phase  (LEXICAL → ROLE → TEMPORAL_DISCOURSE → META)
    - An ``apply`` matcher that tests against a GrammarWorkset
    - An optional ``merge`` callback that enriches the TransientStructure after match
    """

    kind: str
    artifact_kind: str
    compatible_lanes: tuple[str, ...]
    _apply: Callable[[GrammarWorkset], ConstructionTrace] = field(repr=False, compare=False)
    phase: ConstructionPhase = ConstructionPhase.ROLE
    """Pipeline phase this recipe executes in. Defaults to ROLE for backward compat."""
    _merge: Callable[[TransientStructure, ConstructionTrace], None] | None = field(
        repr=False, compare=False, default=None
    )
    """Optional merge callback that enriches the transient structure after a match.

    When None, the default merge logic in TransientStructure.merge() is used,
    which simply accumulates bound slots by key.
    """

    @classmethod
    def from_callable(
        cls,
        *,
        kind: str,
        artifact_kind: str,
        compatible_lanes: tuple[str, ...],
        apply: Callable[[GrammarWorkset], Any],
        phase: ConstructionPhase = ConstructionPhase.ROLE,
        merge: Callable[[TransientStructure, ConstructionTrace], None] | None = None,
    ) -> ConstructionRecipe:
        """Build a recipe from a matcher callable.

        The callable may return any trace-shaped object; it is wrapped so applying the recipe
        always yields a ``ConstructionTrace``. ``merge`` is optional — without it the default
        slot-accumulating merge is used.
        """
        return cls(
            kind=kind,
            artifact_kind=artifact_kind,
            compatible_lanes=compatible_lanes,
            phase=phase,
            _apply=lambda workset: ConstructionTrace.from_trace(apply(workset)),
            _merge=merge,
        )

    def apply(self, workset: GrammarWorkset) -> ConstructionTrace:
        """Run this recipe's matcher against a workset and return the resulting trace."""
        return ConstructionTrace.from_trace(self._apply(workset))


@dataclass(frozen=True, slots=True)
class QueryConstruction:
    """Package-owned query construction compiled from a grammar trace."""

    kind: str
    subjects: frozenset[str] = frozenset()
    actions: frozenset[str] = frozenset()
    slots: frozenset[str] = frozenset()
    operators: frozenset[str] = frozenset()
    operator_frames: frozenset[str] = frozenset()
    connectives: frozenset[str] = frozenset()
    dialogue_act: frozenset[str] = frozenset()
    bound_slots: dict[str, tuple[str, ...]] = field(default_factory=dict)
    trace: ConstructionTrace | None = None


@dataclass(frozen=True, slots=True)
class DeterministicMemoryConstructionEngine:
    """Compatibility engine that delegates to shared construction recipes.

    This is the original engine — it runs all recipes independently against
    the same static workset with no merge phase. Prefer
    AccumulatingMemoryConstructionEngine for FCG-style match+merge behaviour.
    """

    recipe_provider: ConstructionRecipeProviderProtocol = field(
        default_factory=lambda: StaticConstructionRecipeProvider(())
    )

    def build_query_traces(self, workset: GrammarWorkset) -> tuple[ConstructionTrace, ...]:
        """Apply every recipe independently to the same workset, in provider order.

        No state flows between recipes here, so a trace never depends on an earlier match.
        """
        return tuple(recipe.apply(workset) for recipe in self.recipe_provider.recipes())


@dataclass(frozen=True, slots=True)
class StaticConstructionRecipeProvider:
    """In-memory recipe provider for tests and adapters."""

    _recipes: tuple[ConstructionRecipe, ...] = ()

    def recipes(self) -> tuple[ConstructionRecipe, ...]:
        """Return the fixed recipe tuple this provider was constructed with."""
        return self._recipes


@dataclass(frozen=True, slots=True)
class TopicGrammarScope:
    """Topic-local grammar context carried behind the kernel boundary."""

    topic_id: str | None = None
    taxonomy_label: str | None = None
    source_episode_id: str | None = None
    statement_ids: tuple[str, ...] = ()
    ontology_terms: tuple[str, ...] = ()
    grammar_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_provenance(self) -> dict[str, Any]:
        """Render the scope back into provenance form.

        Only populated fields are emitted, so the result can be merged into an existing provenance
        mapping without blanking entries it already holds.
        """
        data: dict[str, Any] = {}
        if self.topic_id:
            data["topic_id"] = self.topic_id
        if self.taxonomy_label:
            data["taxonomy_label"] = self.taxonomy_label
        if self.source_episode_id:
            data["source_episode_id"] = self.source_episode_id
        if self.statement_ids:
            data["statement_ids"] = list(self.statement_ids)
        if self.ontology_terms:
            data["ontology_terms"] = list(self.ontology_terms)
        if self.grammar_id:
            data["grammar_id"] = self.grammar_id
        if self.metadata:
            data["topic_scope_metadata"] = dict(self.metadata)
        return data


__all__ = [
    "ConstructionPhase",
    "ConstructionRecipe",
    "ConstructionTrace",
    "DeterministicMemoryConstructionEngine",
    "GrammarWorkset",
    "QueryConstruction",
    "StaticConstructionRecipeProvider",
    "TopicGrammarScope",
    "TransientStructure",
]


def _clean_text(value: Any) -> str:
    """Return the value as a stripped string, with ``None`` becoming ``""``."""
    return str(value or "").strip()


def _optional_text(value: Any) -> str | None:
    """Return the cleaned text, or ``None`` when it is empty."""
    text = _clean_text(value)
    return text or None


def _stable_id(*parts: str) -> str:
    """Return a deterministic 16-hex id over the ``|``-joined parts.

    Stable across processes and runs, so ids derived from the same parts stay comparable.
    """
    return hashlib.sha1("|".join(str(part or "") for part in parts).encode("utf-8")).hexdigest()[:16]
