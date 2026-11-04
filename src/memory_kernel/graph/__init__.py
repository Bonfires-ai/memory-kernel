"""Graph integration namespace for memory_kernel.

LAZY BY CONSTRUCTION (PEP 562). This package's public names resolve on first attribute access,
not at package-import time, so importing a LEAF module out of this package costs only that leaf.

The eager version imported the graph runtime and the topic-graph seeder here, which
transitively pulled the whole graph runtime stack including numpy — and Python must execute
this file to reach ANY submodule. That was the entire cost of importing a pure-stdlib leaf out
of this package. Lazy brings it back to the leaf's own cost, with no numpy.

``from memory_kernel.graph import X`` and ``import memory_kernel.graph`` + attribute access
both still work for every name below: ``__getattr__`` imports the owning submodule on demand
and caches the value in module globals, so repeat access is a plain dict hit. The
``TYPE_CHECKING`` block keeps every lazy name statically visible to type checkers and IDEs,
which a runtime-only ``__getattr__`` would otherwise hide.

The cheapness is a contract, not a happy accident: a cold import of a leaf under this package
loads only a handful of ``memory_kernel`` modules and
never numpy. So NOTHING may be imported at the top level of this file — an eager import here is
paid by every importer of every leaf beneath it.
"""

from __future__ import annotations

from typing import Any

#: Public name -> the submodule that defines it: the one table ``__getattr__`` resolves names from.
#: ``__all__`` is a literal list of the same names.
_LAZY_EXPORTS: dict[str, str] = {}

__all__ = []


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
    """Every lazy export plus whatever has already been resolved, so introspection does not force an import."""
    return sorted({*globals(), *_LAZY_EXPORTS})
