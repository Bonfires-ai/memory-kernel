"""Protocol boundaries for the memory_kernel package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memory_kernel.construction import ConstructionRecipe, ConstructionTrace, GrammarWorkset


@runtime_checkable
class MemoryConstructionEngineProtocol(Protocol):
    """Construction engine seam for query trace generation."""

    def build_query_traces(self, workset: GrammarWorkset) -> tuple[ConstructionTrace, ...]: ...


@runtime_checkable
class ConstructionRecipeProviderProtocol(Protocol):
    """Provide construction recipes for deterministic trace generation."""

    def recipes(self) -> tuple[ConstructionRecipe, ...]: ...
