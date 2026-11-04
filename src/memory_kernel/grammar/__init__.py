"""Grammar utilities for memory_kernel (the spaCy loader, the tense reading).

Every public name is imported eagerly. ``_LAZY_EXPORTS`` and ``__getattr__`` below are the PEP 562 seam (the same
mechanism ``graph/__init__.py`` and ``ingest/__init__.py`` use) for a name whose eager import would close an import
cycle; the table is empty today, so ``__getattr__`` only raises ``AttributeError``.
"""

from typing import Any

from .dependencies import load_spacy_model
from .semantic_operators import (
    SemanticOperatorProfile,
    SemanticOperatorService,
    get_semantic_operator_service,
)

#: Public name -> the submodule that defines it (empty: every public name is eager today).
_LAZY_EXPORTS: dict[str, str] = {}

__all__ = [
    "load_spacy_model",
    "SemanticOperatorProfile",
    "SemanticOperatorService",
    "get_semantic_operator_service",
]


def __getattr__(name: str) -> Any:
    """Import the owning submodule on first access (PEP 562), then cache it in globals."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module  # noqa: PLC0415 — deferred with the rest of the cost

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
