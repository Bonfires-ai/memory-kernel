"""One home for the kernel's env-flag READERS.

The idiom — ``os.getenv(name)``, then ``.strip().lower() in {"1","true","yes","on"}`` — lives here once instead of
being repeated in every module that reads a flag. This module is the readers only; it is deliberately NOT a registry of
the ``MK_*`` names, which stay where their meaning lives.

## No import-time env reads

Every function here reads the environment when it is CALLED. A module-level
``FLAG = os.getenv(...)`` freezes the value at import and makes a process's
behaviour depend on import order, which is exactly the bug this module would
otherwise centralise.

## Loud, not silent

Unset **or empty** degrades to ``default``; a *present, non-empty, malformed*
value RAISES. A mistyped deployment knob should fail fast at startup rather than
run for a week on a default nobody chose.

## Empty is unset

An **exported-but-empty** variable (``export FOO=``) is the one input class on which readers tend to disagree. Here
every reader treats it as unset: :func:`env_flag`, :func:`env_flag_unless_disabled`, :func:`env_int` and
:func:`env_float` all return ``default`` for it.

:func:`env_flag_unless_disabled` is kept apart from :func:`env_flag` because of its *vocabulary*: an inverted flag
("on unless explicitly switched off") whose truth set is the complement of a falsy set. Collapsing it into
:func:`env_flag` would flip the value of every string in neither set (``"maybe"``).
"""

from __future__ import annotations

import os

#: The truthy vocabulary.
TRUTHY = frozenset({"1", "true", "yes", "on"})
#: The falsy vocabulary :func:`env_flag_unless_disabled` tests against.
FALSY = frozenset({"0", "false", "no", "off"})


def _raw(name: str) -> str:
    """The env value, stripped. Unset and exported-empty are the same thing here."""

    return str(os.environ.get(name) or "").strip()


def env_flag(name: str, default: bool = False) -> bool:
    """``True`` iff the variable is set to one of :data:`TRUTHY` (case-insensitive).

    Unset or empty yields ``default``. Anything else — including a typo'd
    ``"treu"`` — is ``False``: a flag has no malformed shape to be loud about,
    since every string that is not truthy is meaningfully not-truthy.

    ``default`` is positional-OR-keyword on purpose, so ``env_flag(name, False)`` and
    ``env_flag(name, default=True)`` are both valid.
    """

    raw = _raw(name)
    if not raw:
        return default
    return raw.lower() in TRUTHY


def env_flag_unless_disabled(name: str, default: bool = True) -> bool:
    """Inverted flag: ``False`` only for an explicit :data:`FALSY` value.

    The shape of ``MK_BOUNDARY_SMART_MASK`` in `ingest/episode_boundary` — a
    default-on knob you switch OFF. Not the complement of :func:`env_flag`: a
    value in neither vocabulary (``"maybe"``) is ``True`` here and ``False``
    there, which is why the two stay separate functions. Unset or empty yields
    ``default``.
    """

    raw = _raw(name)
    if not raw:
        return default
    return raw.lower() not in FALSY


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Deployment knob. Unset/empty degrades to ``default``; malformed RAISES.

    ``minimum`` clamps a value that was actually SET — it never raises ``default``.
    """

    raw = _raw(name)
    if not raw:
        return default
    value = int(raw)
    return value if minimum is None else max(minimum, value)


def env_float(name: str, default: float) -> float:
    """Same contract as :func:`env_int`: unset/empty degrades, malformed raises."""

    raw = _raw(name)
    if not raw:
        return default
    return float(raw)


__all__ = [
    "FALSY",
    "TRUTHY",
    "env_flag",
    "env_flag_unless_disabled",
    "env_float",
    "env_int",
]
