"""The ``memory_kernel`` package root.

LAZY BY CONSTRUCTION (PEP 562), like :mod:`memory_kernel.graph` and :mod:`memory_kernel.ingest`. The names below
resolve on first attribute access, not at package-import time, so ``import memory_kernel`` loads this one module and
nothing else from the package (no numpy either).

Why the root and not just the leaves: Python must execute THIS file to reach *any* submodule, so an eager root would
be a tax on every import in the package.

``from memory_kernel import X`` and ``import memory_kernel`` + attribute access both work for every name:
``__getattr__`` imports the owning submodule on demand and caches the value in module globals, so repeat access is a
plain dict hit. ``__all__`` is a literal list of the same names as ``_LAZY_EXPORTS``; the import-smoke test resolves
every name in both. The ``TYPE_CHECKING`` block keeps every name statically visible to type checkers and IDEs, which
a runtime-only ``__getattr__`` would otherwise hide.

A lazy root also makes latent import cycles observable: an eager root can hide one by importing modules in a working
order. ``tests/test_import_smoke.py`` imports every module in a fresh interpreter, so an import cycle cannot silently
return.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .construction import (
        ConstructionPhase,
        TransientStructure,
    )
    from .construction.artifacts import (
        ARTIFACT_FAMILY_BY_LEGACY_KIND,
        GRAPH_VISIBLE_ARTIFACT_FAMILIES,
        SHARED_RECIPE_KINDS,
        TypedArtifact,
    )

#: Public name -> the submodule that defines it: the one table ``__getattr__`` resolves names from.
#: ``__all__`` is a literal list of the same names.
_LAZY_EXPORTS: dict[str, str] = {
    "ARTIFACT_FAMILY_BY_LEGACY_KIND": ".construction.artifacts",
    "ConstructionPhase": ".construction",
    "GRAPH_VISIBLE_ARTIFACT_FAMILIES": ".construction.artifacts",
    "SHARED_RECIPE_KINDS": ".construction.artifacts",
    "TransientStructure": ".construction",
    "TypedArtifact": ".construction.artifacts",
}

__all__ = [
    "ARTIFACT_FAMILY_BY_LEGACY_KIND",
    "ConstructionPhase",
    "GRAPH_VISIBLE_ARTIFACT_FAMILIES",
    "SHARED_RECIPE_KINDS",
    "TransientStructure",
    "TypedArtifact",
]


def __getattr__(name: str) -> Any:
    """Import the owning submodule on first access (PEP 562), then cache it in globals."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module  # noqa: PLC0415 — deferred with the rest of the cost

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
