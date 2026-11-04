"""Event time for G1 projections.

One resolver for every predicate, occurrence or state: the argm-tmp filler is read against the
message's ``told_at``, and the result carries the shared property names used on Events and on
state fact edges:

- ``told_at`` -- the envelope time, passed through unchanged;
- ``happened_at`` / ``happened_at_grain`` -- the resolved value at its own grain, or the told day
  at day grain when no time expression resolves;
- ``time_source`` -- ``explicit`` (a written calendar date), ``deictic`` (a relative expression
  resolved against ``told_at``) or ``told_default`` (no time expression at all; ``happened_at`` is the told day) or ``unresolved`` (a time
  expression is present but names no date; ``happened_at`` is None, ``valid_at`` is told_at);
- ``valid_at`` -- ``instant_of(resolved)`` when a value resolved, else ``told_at``;
- ``phrase`` -- the in-text rendering of the resolution (``"on 2023-05-07"``), or ``None``.

Resolver chain (shared live code, called only): ``temporal_deixis.resolve_surface`` first; if it
names nothing, the day/month/year clauses of ``relative_resolution_note`` are read back. Anything
else is ``unresolved``: an unresolvable phrase never becomes an invented date.

``MK_TEMPORAL_DEIXIS`` off means ``told_default`` for every event, whatever the text says.
"""

from __future__ import annotations

import re

from memory_kernel.knowledge.temporal_deixis import (
    MONTH_NAMES,
    ResolvedTime,
    anchor_date,
    explicit_date_value,
    instant_of,
    relative_resolution_note,
    resolve_surface,
    temporal_deixis_enabled,
)

_MONTH_ALTERNATION = "|".join(sorted(MONTH_NAMES.values(), key=len, reverse=True))
_MONTH_YEAR = re.compile(rf"^({_MONTH_ALTERNATION})\s+(\d{{4}})$")
_YEAR_ONLY = re.compile(r"^(\d{4})$")
_MONTH_NUMBER = {name: number for number, name in MONTH_NAMES.items()}
_EXPLICIT_MARK = re.compile(rf"\d{{4}}-\d{{2}}-\d{{2}}|\b(?:{_MONTH_ALTERNATION})\b", re.IGNORECASE)


def _told_day(told_at: str) -> str | None:
    day = anchor_date(told_at)
    return day.isoformat() if day is not None else None


def _told_default(told_at: str) -> dict[str, str | None]:
    return {
        "told_at": told_at,
        "happened_at": _told_day(told_at),
        "happened_at_grain": "day" if _told_day(told_at) is not None else None,
        "time_source": "told_default",
        "valid_at": told_at,
        "phrase": None,
    }


def _unresolved(told_at: str) -> dict[str, str | None]:
    """A time expression is present but names no date: keep the told time, invent nothing."""
    return {
        "told_at": told_at,
        "happened_at": None,
        "happened_at_grain": None,
        "time_source": "unresolved",
        "valid_at": told_at,
        "phrase": None,
    }


def _from_resolved(told_at: str, resolved: ResolvedTime, source: str) -> dict[str, str | None]:
    return {
        "told_at": told_at,
        "happened_at": resolved.value,
        "happened_at_grain": resolved.grain,
        "time_source": source,
        "valid_at": instant_of(resolved) or told_at,
        "phrase": resolved.phrase,
    }


def _from_note(text: str, told_at: str) -> ResolvedTime | None:
    """Read the first ``phrase = resolution`` clause the legacy note spells out as a day, month or year.

    Week, weekend and range clauses are skipped: they have no single value at the grains this
    module stores, and inventing one would be the same failure as an invented date.
    """
    note = relative_resolution_note(text, told_at)
    if not note:
        return None
    lowered = text.lower()
    clauses = [clause.split(" = ", 1) for clause in note.split("; ") if " = " in clause]
    clauses.sort(key=lambda pair: 0 if pair[0].strip().lower() in lowered else 1)
    for phrase, resolution in clauses:
        resolution = resolution.strip()
        day = explicit_date_value(resolution)
        if day is not None and re.fullmatch(r"\d{1,2} [A-Za-z]+ \d{4}", resolution):
            value = day.isoformat()
            return ResolvedTime(phrase.strip(), value, "day", f"on {value}")
        month = _MONTH_YEAR.fullmatch(resolution)
        if month is not None:
            value = f"{month.group(2)}-{_MONTH_NUMBER[month.group(1)]:02d}"
            return ResolvedTime(phrase.strip(), value, "month", f"in {value}")
        year = _YEAR_ONLY.fullmatch(resolution)
        if year is not None:
            return ResolvedTime(phrase.strip(), year.group(1), "year", f"in {year.group(1)}")
    return None


def event_time(tmp_text: str | None, told_at: str) -> dict[str, str | None]:
    """The time fields for one predicate, from its argm-tmp filler (or ``None``) and the message's ``told_at``."""
    if not temporal_deixis_enabled() or not tmp_text or not tmp_text.strip():
        return _told_default(told_at)
    resolved = resolve_surface(tmp_text, told_at)
    if resolved is not None:
        explicit = _EXPLICIT_MARK.search(tmp_text) is not None
        return _from_resolved(told_at, resolved, "explicit" if explicit else "deictic")
    fallback = _from_note(tmp_text, told_at)
    if fallback is not None:
        return _from_resolved(told_at, fallback, "deictic")
    return _unresolved(told_at)
