"""Temporal deixis: a relative time expression resolves against the UTTERANCE's own clock.

THE LAW (auto-population). "the runtime clock
resolves temporal deixis ('tomorrow' said 08-11 => 2026-08-12, resolved against
UTTERANCE time, not server-now) ... and date roles bind absolute dates." The
clock that resolves a deictic is the clock of the message that said it — never
the fold's wall clock, which is hours-to-months later on a drain and arbitrary
on a re-fold.

WHAT IT COST TO NOT DO THIS. Measured on the live serving graph (the production graph,
2026-08-11):

    (equal#0)     -[:ARG0]-> (today)      valid_at 2026-07-13T22:02:42Z
    (record.01#0) -[:ARG0]-> (today)      valid_at 2026-07-13T22:02:42Z
    (replace.01#0)-[:ARG0]-> (today)      valid_at 2026-07-13T22:02:42Z
    ... plus one (yesterday)

Four ``today`` nodes and one ``yesterday``, each an ENTITY named after a word
whose meaning is a function of the moment it was said — and the moment is right
there on the edge, in ``valid_at``, unread. Nothing downstream can ever recover
which day those rows mean. Beside them, the one card that declares a date role
(``event.birthday.v1``: ``$person -[:BORN_IN]-> $date``) holds
``dan -[:BORN_IN]-> "August 12, 2026"`` — correct only because a human spelled
the date out, which is the law's exhibit verbatim.

THE CLOSED SET, AND WHY IT IS CLOSED. This is not a temporal parser and must not
become one. It resolves a bounded vocabulary — the same vocabulary
the host procedural normalizer's date lane has resolved since it shipped, which
is why :data:`DEIXIS_PATTERN` and the arithmetic below LIVE here now and that
lane imports them rather than keeping a second copy — plus explicit calendar
dates ("august 12"), which are safe only under the FULL-SURFACE discipline
:func:`resolve_surface` applies and so are deliberately absent from the
in-text pattern the normalizer searches with.

TWO CALLERS, TWO SHAPES, ONE VOCABULARY:

* :func:`resolve_first` — the normalizer's shape. Finds the FIRST deictic
  anywhere in a sentence and hands back its in-text rendering ("on 2023-07-02",
  "the week of 2023-06-26"), because that lane rewrites prose.
* :func:`resolve_surface` — the fold's shape. Judges a role FILLER, whole, and
  hands back a bare absolute value ("2023-07-02"), because a date role binds a
  date and not a phrase. Whole-surface is the safety property: "the party
  tomorrow" is not a date and resolving inside it would author
  ``"the party 2026-08-12"``, which is worse than the surface it replaced.

TIMEZONE, STATED HONESTLY. Statement timestamps are UTC (the host fold
renders ``%Y-%m-%dT%H:%M:%SZ``), and the anchor day is the timestamp's OWN
calendar day with no zone conversion — so a speaker two hours east of UTC saying
"tomorrow" at 01:00 local anchors on the previous UTC day and resolves one day
early. Fixing that needs a per-actor timezone the system does not collect;
recording it here is the honest alternative to pretending it is right.

YEAR DEFAULTING, ALSO STATED HONESTLY. An explicit date with no year takes the
ANCHOR's year, flatly — no nearest-occurrence heuristic, no tense reading. "December
25" said on January 2nd resolves to that same year's December, eleven months out.
Choosing better needs the sentence's tense, which is a parser's job, not this
module's.

FAIL-OPEN, ALWAYS. Every entry point returns ``None`` — meaning "no resolution,
keep the surface" — for an unusable anchor, an unmatched surface, or an
arithmetic that does not come out: the impossible calendar day of "february 30",
and the day off the end of the calendar that "1000000 days ago" names (``date``
spans years 1-9999, and ``timedelta`` refuses an offset too large for a C int at
all). A weird temporal form must never be able to fail a fold; that is the same
bar the emoji and bare-dot cases are held to, and it is enforced at ONE seam
— :func:`_resolve_kind` guards the whole arithmetic table, so a kind added later
inherits the property instead of having to remember it.

ACCEPTED FALSE POSITIVE, NAMED. The explicit day-first
shape reads "10 march" and "5 April" as calendar dates, so a quantity-plus-noun
filler that happens to spell a month resolves to a date it never meant. It is
NOT guarded, deliberately: the only signal available is the entity label, and
``Thing`` is the residual lane's default — label-guarding ``Thing`` would disable
the entire residual seam, which is the lane the live exhibit came through. The
class is pinned by a test so a future change to it is a decision rather than a
drift.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

from memory_kernel.config.flags import env_flag_unless_disabled
from memory_kernel.grammar import lexicon
from memory_kernel.grammar.semantic_operators import get_semantic_operator_service
from memory_kernel.primitives import dedupe as _dedupe

logger = logging.getLogger(__name__)

__all__ = [
    "DEIXIS_PATTERN",
    "MONTH_NAMES",
    "MONTH_NAME_TO_NUMBER",
    "NUMBER_WORD_SCALES",
    "NUMBER_WORD_VALUES",
    "ResolvedTime",
    "SEASONS",
    "TEMPORAL_DEIXIS_ENV",
    "WEEKDAYS",
    "anchor_date",
    "content_with_relative_time",
    "explicit_date_text",
    "explicit_date_value",
    "format_human_day",
    "format_human_month",
    "instant_of",
    "monday_of",
    "next_weekday",
    "next_weekday_including_anchor",
    "parse_timestamp_date",
    "previous_weekday",
    "previous_weekday_including_anchor",
    "relative_number_value",
    "relative_offset_date_value",
    "relative_resolution_note",
    "resolve_surface",
    "safe_date",
    "temporal_deixis_enabled",
    "temporal_text",
    "this_weekend_start",
    "timestamp_to_text",
]

#: Escape hatch, ON by default: this is a law, not a lane. Set to a falsey value
#: to restore the earlier behaviour for a bisect — never as a standing config.
#: Only the FOLD seam consults it; the normalizer's date lane has its own switch
#: (a separate switch) and predates this module.
TEMPORAL_DEIXIS_ENV = "MK_TEMPORAL_DEIXIS"

#: Weekday names in ``date.weekday()`` order (Monday-zero), so ``.index(name)``
#: IS the weekday number.
WEEKDAY_NAMES: tuple[str, ...] = lexicon.weekday_names()

#: The in-text deictic vocabulary. MOVED here from
#: the host procedural normalizer's ``_DATE_PATTERN`` (which now imports it): one
#: vocabulary, two readers. Extending it changes the normalizer's output and so
#: requires bumping that module's ``ALGO_VERSION``.
#:
#: ONE extension since the move: ``days ago`` became
#: ``days? ago``, so the singular "1 day ago" resolves where the inherited
#: pattern left it standing. Purely ADDITIVE — no form that resolved before
#: resolves differently now — but it is still a vocabulary change, so
#: ``ALGO_VERSION`` went 3 -> 4 with it. "a day ago" is deliberately still out:
#: it needs a word-to-number map, which is a parser's job, not this module's.
_OFFSET = "|".join(lexicon.relative_offsets())  # "last|next"
DEIXIS_PATTERN = re.compile(
    r"\b(?:"
    r"(?P<last_night>last night)"
    rf"|(?P<this_part>this (?:{'|'.join(lexicon.dayparts_after_this())}))"
    rf"|(?P<rel_weekend>(?:{_OFFSET}) weekend)"
    rf"|(?P<rel_week>(?:{_OFFSET}) week)"
    rf"|(?P<rel_month>(?:{_OFFSET}) month)"
    rf"|(?P<rel_year>(?:{_OFFSET}) year)"
    rf"|(?P<rel_weekday>(?:{_OFFSET}) (?:{'|'.join(WEEKDAY_NAMES)}))"
    r"|(?P<days_ago>(\d+) days? ago)"
    r"|(?P<yesterday>yesterday)"
    r"|(?P<today>today)"
    r"|(?P<tonight>tonight)"
    r"|(?P<tomorrow>tomorrow)"
    r")\b",
    re.IGNORECASE,
)

#: The named alternatives of :data:`DEIXIS_PATTERN`, in pattern order. ``lastgroup``
#: can point at the inner ``(\d+)`` of ``days_ago``, so the kind is recovered by
#: walking these instead of trusting it.
_DEIXIS_KINDS: tuple[str, ...] = (
    "last_night",
    "this_part",
    "rel_weekend",
    "rel_week",
    "rel_month",
    "rel_year",
    "rel_weekday",
    "days_ago",
    "yesterday",
    "today",
    "tonight",
    "tomorrow",
)

#: Full-surface only (see the module docstring): "AUGUST 12", "Aug 12th, 2026",
#: "12 August", "12th of August 2026". The year is optional and defaults to the
#: anchor's. A bare month name and a bare "month year" ("august 2026") match
#: NEITHER shape — both want a day number — and so resolve to nothing.
#:
#: The DAY-FIRST shape carries the module docstring's named false positive: it
#: cannot tell "10 march" the date from "10 march" the count of marches, and it
#: resolves both. Accepted, not guarded — see the docstring for why the only
#: available guard (the entity label) cannot be used here.
_EXPLICIT_MONTH_FIRST = re.compile(
    r"(?P<month>[a-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:\s*,)?(?:\s+(?P<year>\d{4}))?",
    re.IGNORECASE,
)
_EXPLICIT_DAY_FIRST = re.compile(
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<month>[a-z]+)(?:\s*,)?(?:\s+(?P<year>\d{4}))?",
    re.IGNORECASE,
)

#: Already-absolute shapes. A filler that is already a date resolves to itself,
#: which is a NO-OP — reported as "nothing to do" so the row stays byte-identical.
_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")

#: Leading function words a role filler can carry in front of a date ("on
#: tomorrow", "on August 12", "the week of next week"). Stripped before matching
#: and never re-emitted: the VALUE is the date, not the prepositional phrase.
#: Longest-first where one is a prefix of another, so "the week of " is tried
#: before the bare "the ". Up to TWO come off (see :func:`_strip_affixes`), which
#: is what lets "on the 12th of August" reduce the way "the 12th of August" does.
#: The words are a stripping vocabulary, NOT a licence: "on monday" still
#: resolves to nothing, because a bare weekday is not in the closed set.
_LEADING_FUNCTION_WORDS: tuple[str, ...] = lexicon.leading_function_words()

#: Dayparts a deictic day can trail ("tomorrow morning"). Discarded: the grain
#: this module resolves to is the DAY, and a daypart does not change which day.
_TRAILING_DAYPARTS: tuple[str, ...] = tuple(f" {word}" for word in lexicon.dayparts_after_tomorrow())


@dataclass(frozen=True)
class ResolvedTime:
    """One resolved temporal expression.

    ``surface`` is the matched text verbatim; ``value`` is the absolute rendering
    the fold binds — ``YYYY-MM-DD`` for a day, ``YYYY-Www`` for a week,
    ``YYYY-MM`` for a month, ``YYYY`` for a year, each at ITS OWN precision and
    never a coarser grain wearing a day's shape; ``grain`` names which of those it
    is; ``phrase`` is the in-text rendering the normalizer's date lane
    substitutes, which is prose and so still spells a week as "the week of
    <Monday>".
    """

    surface: str
    value: str
    grain: str
    phrase: str


def temporal_deixis_enabled() -> bool:
    """Whether the fold's deixis seam is armed (default ON — see :data:`TEMPORAL_DEIXIS_ENV`).

    A default-ON knob you switch OFF, so only an explicit falsey value disarms
    it: this is a law, and an unrecognised value must not silently repeal it.
    """

    return env_flag_unless_disabled(TEMPORAL_DEIXIS_ENV, default=True)


def anchor_date(ts: Any) -> date | None:
    """The calendar date of a message timestamp, or ``None`` when there is nothing usable.

    ``datetime`` values give their date; strings are parsed as ISO-8601 with a trailing ``Z``
    accepted. No timezone conversion — see the module docstring's timezone note.

    Returning ``None`` for an unparseable stamp is the designed behaviour: timestamps come from
    external message sources and one malformed value must not abort the batch — resolution simply
    does not fire for that message. The bad value and the parse error are logged at warning level;
    other types yield ``None`` silently.
    """
    if isinstance(ts, datetime):
        return ts.date()
    if isinstance(ts, str) and ts.strip():
        try:
            return datetime.fromisoformat(ts.strip().replace("Z", "+00:00")).date()
        except ValueError as exc:
            # KEEP: per-item isolation — external message
            # timestamp, mirrors episode_boundary.py:244 in the same package.
            logger.warning("Unparseable message timestamp %r: %s", ts, exc)
            return None
    return None


def monday_of(d: date) -> date:
    """The Monday of ``d``'s week; a date that is already a Monday is returned unchanged."""
    return d - timedelta(days=d.weekday())


def _kind_of(match: re.Match[str]) -> str | None:
    """Which named alternative of :data:`DEIXIS_PATTERN` ``match`` matched."""
    for name in _DEIXIS_KINDS:
        if match.group(name):
            return name
    return None


def _resolve_kind(kind: str, phrase: str, base: date) -> ResolvedTime | None:
    """The absolute time a matched deictic ``kind`` names, anchored on ``base`` — never raises.

    The module's fail-open bar, enforced at ONE seam. ``None`` for any kind whose
    arithmetic does not come out: ``days_ago`` with no digits (which the pattern
    makes unreachable), and — the reason this guard is a wrapper and not a
    per-branch ``if`` — any offset that walks off the calendar. ``date`` spans
    years 1-9999, so "1000000 days ago" raises ``OverflowError`` from
    ``timedelta`` arithmetic, and an astronomically large N raises it from the
    ``timedelta`` constructor before any date is touched. Both reach
    :func:`resolve_surface`, :func:`resolve_first`, the normalizer's date lane
    (every ingested message) and the host fold's planner — where an escaping
    exception would kill the whole fold over one weird surface, which is exactly
    what this module promises cannot happen.

    Guarding the TABLE rather than the one branch that was reported is
    deliberate: a kind added later inherits the property instead of having to
    remember it.
    """
    try:
        return _kind_arithmetic(kind, phrase, base)
    except (OverflowError, ValueError) as exc:
        # KEEP: fail-open IS the contract — the return is
        # the same `None` an unmatched surface produces, and the caller's only
        # act on it is to keep the surface it already has. Debug, not warning:
        # the input is ordinary conversation text, and a warning per occurrence
        # would say nothing the caller does not already handle.
        logger.debug("Temporal deixis %r off the calendar at anchor %s: %s", phrase, base, exc)
        return None


def _kind_arithmetic(kind: str, phrase: str, base: date) -> ResolvedTime | None:
    """The arithmetic table :func:`_resolve_kind` guards. May raise; call it through the guard."""
    lower = phrase.lower()
    if kind in {"yesterday", "last_night"}:
        return _day(phrase, base - timedelta(days=1))
    if kind in {"today", "tonight", "this_part"}:
        return _day(phrase, base)
    if kind == "tomorrow":
        return _day(phrase, base + timedelta(days=1))
    if kind == "rel_week":
        shift = -7 if lower.startswith("last") else 7
        target = monday_of(base + timedelta(days=shift))
        # WEEK GRAIN RENDERS AS A WEEK. This used to hand
        # back the Monday's `YYYY-MM-DD`, which mints an entity node named
        # `2026-08-03` — indistinguishable from a day-precision claim, and so a
        # breach of this module's own law that a coarser grain renders at its own
        # precision and never a fake day. ISO 8601 week (`2026-W32`) joins the
        # month (`2026-07`) and year (`2025`) family. The ISO YEAR is not the
        # calendar year near a boundary — 2025-12-29 is `2026-W01` — so the
        # rendering takes both numbers from `isocalendar()` and neither from the
        # date. `phrase`, which is the NORMALIZER's in-text rendering, keeps the
        # Monday it has always had: "the week of 2026-08-03" reads as prose where
        # "the week of 2026-W32" does not, and that lane's output is pinned.
        iso = target.isocalendar()
        return ResolvedTime(
            surface=phrase,
            value=f"{iso.year:04d}-W{iso.week:02d}",
            grain="week",
            phrase=f"the week of {target}",
        )
    if kind == "rel_weekend":
        # A WEEKEND IS NOT AN ISO WEEK, and has no standard coarse notation — so
        # it is resolved DOWN to its Saturday and reported at day grain
        # (choice stated). That is a narrowing to a real day
        # inside the span, not the coarse-grain-wearing-a-day-shape the week
        # rendering above just stopped doing: the value and the grain agree.
        if lower.startswith("last"):
            delta = (base.weekday() - 5) % 7 or 7
            return _day(phrase, base - timedelta(days=delta))
        delta = (5 - base.weekday()) % 7 or 7
        return _day(phrase, base + timedelta(days=delta))
    if kind == "rel_month":
        year, month = base.year, base.month
        month = month - 1 if lower.startswith("last") else month + 1
        if month == 0:
            year, month = year - 1, 12
        elif month == 13:
            year, month = year + 1, 1
        # The month and year branches build their value as a STRING, so unlike
        # the day arithmetic above they cannot raise — they would instead have
        # happily minted "0000-12" off a year-1 anchor and "10000" off a
        # year-9999 one, names of months and years that do not exist. Asked of
        # `date` the same question, the answer is the same `None` an off-calendar
        # day gets: one rule for the whole table. The
        # constructed day is what the rendering then reads, so the check cannot
        # be mistaken for a dead statement and deleted.
        first = date(year, month, 1)
        value = f"{first.year:04d}-{first.month:02d}"
        return ResolvedTime(surface=phrase, value=value, grain="month", phrase=f"in {value}")
    if kind == "rel_year":
        first = date(base.year - 1 if lower.startswith("last") else base.year + 1, 1, 1)
        return ResolvedTime(surface=phrase, value=f"{first.year:04d}", grain="year", phrase=f"in {first.year:04d}")
    if kind == "rel_weekday":
        target_weekday = WEEKDAY_NAMES.index(lower.split()[1])
        if lower.startswith("last"):
            delta = (base.weekday() - target_weekday) % 7 or 7
            return _day(phrase, base - timedelta(days=delta))
        delta = (target_weekday - base.weekday()) % 7 or 7
        return _day(phrase, base + timedelta(days=delta))
    if kind == "days_ago":
        digits = re.search(r"\d+", phrase)
        if digits is None:
            return None
        # N is unbounded by the pattern and the arithmetic is unbounded by the
        # calendar: "1000000 days ago" and an astronomically large "N days ago" both
        # raise OverflowError here, and :func:`_resolve_kind`'s guard is what
        # turns that into the ordinary "no resolution".
        return _day(phrase, base - timedelta(days=int(digits.group(0))))
    return None


def _day(surface: str, value: date) -> ResolvedTime:
    """A day-grain resolution — the shape ten of the twelve deictic kinds produce."""
    return ResolvedTime(surface=surface, value=value.isoformat(), grain="day", phrase=f"on {value}")


#: How each grain's :attr:`ResolvedTime.value` is spelled, as the ``strptime``
#: format that reads it back. Weeks are ISO ("2026-W33") and need the ISO
#: year/week/weekday triple — ``%G-W%V-%u`` with a ``-1`` appended for Monday —
#: because ``%Y`` is the CALENDAR year and disagrees with ``%G`` across a
#: year boundary (2025-12-29 is 2026-W01), which is the same trap the week
#: rendering above documents.
_GRAIN_VALUE_FORMATS: dict[str, tuple[str, str]] = {
    "day": ("{value}", "%Y-%m-%d"),
    "month": ("{value}", "%Y-%m"),
    "week": ("{value}-1", "%G-W%V-%u"),
    "year": ("{value}", "%Y"),
}


def instant_of(resolved: ResolvedTime) -> str | None:
    """The floored UTC instant a resolution names, as a full ISO timestamp, or ``None``.

    A :class:`ResolvedTime` renders at ITS OWN grain — ``"2026-08-16"``,
    ``"2026-07"``, ``"2026-W33"``, ``"2025"`` — which is the honest shape for a
    VALUE but not a timestamp: a consumer that parses timestamps either rejects
    the coarse spellings or, worse, MISREADS one (pydantic reads the bare year
    ``"2025"`` as epoch seconds and yields 1970-01-01T00:33:45Z — silent
    corruption, measured 2026-08-17). So a consumer that needs an instant gets
    the START of the span, floored to midnight: ``"2026-07"`` ->
    ``"2026-07-01T00:00:00Z"``, ``"2026-W33"`` -> its Monday
    ``"2026-08-10T00:00:00Z"``, ``"2025"`` -> ``"2025-01-01T00:00:00Z"``.

    THE INSTANT IS LOSSY AND IS NOT THE VALUE. Flooring a month to its first day
    manufactures a day-precision surface for a month-precision claim — exactly
    what :func:`_kind_arithmetic`'s week branch refuses to do to ``value``. It is
    safe only where the grain travels ALONGSIDE it (``NormalizedStatement``
    carries ``event_ts_grain`` next to ``event_ts_resolved``); a caller that
    stores this instant without the grain has thrown away the precision and this
    function is the wrong one for it.

    ``None`` for an unrecognised grain or a value that does not parse — the same
    fail-open the rest of this module practises, and the same ``None`` an
    unmatched surface produces. Timezone: midnight on the resolved calendar day,
    UTC, per the module docstring's timezone note.
    """

    spelling = _GRAIN_VALUE_FORMATS.get(resolved.grain)
    if spelling is None:
        logger.debug("Temporal grain %r has no instant spelling (value %r)", resolved.grain, resolved.value)
        return None
    template, fmt = spelling
    try:
        start = datetime.strptime(template.format(value=resolved.value), fmt)
    except ValueError:
        logger.debug("Temporal value %r does not read back at grain %r", resolved.value, resolved.grain)
        return None
    return start.strftime("%Y-%m-%dT%H:%M:%SZ")


def _strip_affixes(surface: str) -> str:
    """A role filler reduced to the bare temporal phrase it may be wrapping.

    Leading function words and one trailing daypart come off; surrounding
    whitespace and trailing sentence punctuation go with them. Applied ONLY
    inside :func:`resolve_surface`, whose contract is whole-surface matching.

    UP TO TWO leading words come off. Stripping exactly one
    made "the 12th of August" resolve while "on the 12th of August" — the same
    date, said the way people say it — did not. Two is the bound because two is
    what English stacks here ("on the", "in the", "by the"); this is a fixed
    vocabulary coming off a filler, not a grammar, and it must not grow into one.
    """
    core = surface.strip().strip(".,;:!?").strip()
    for _ in range(2):
        lowered = core.lower()
        for prefix in _LEADING_FUNCTION_WORDS:
            if lowered.startswith(prefix):
                core = core[len(prefix) :].strip()
                break
        else:
            break
    return core


def resolve_surface(surface: str, ts: Any) -> ResolvedTime | None:
    """The absolute time a role filler names WHOLE, or ``None`` when it names none.

    Whole-surface by contract (see the module docstring): "tomorrow", "on August
    12" and "on the 12th of August" resolve; "the party tomorrow" does not,
    because it is not a date and a partial rewrite would author a worse filler
    than the one it replaced.

    A filler that is ALREADY absolute (``2026-08-12``) returns ``None`` — there is
    nothing to resolve, and reporting "no change" is what keeps the caller's row
    byte-identical.
    """
    # The surface REPORTED is the filler as it arrived (trimmed only); the
    # surface MATCHED is the bare temporal phrase inside it. Keeping the arrival
    # form is what makes the marker an audit trail — after the rewrite it is the
    # only record of what licensed the date.
    reported = surface.strip()
    core = _strip_affixes(surface)
    if not core or _ISO_DAY.fullmatch(core):
        return None
    base = anchor_date(ts)
    if base is None:
        return None
    match = DEIXIS_PATTERN.fullmatch(core)
    if match is None:
        # "tomorrow morning" — the daypart does not change WHICH day, so it comes
        # off and the bare deictic is tried once more. Tried second so that
        # "this morning", which the pattern DOES carry whole, is never truncated
        # to a bare "this" that matches nothing.
        lowered = core.lower()
        for daypart in _TRAILING_DAYPARTS:
            if lowered.endswith(daypart):
                match = DEIXIS_PATTERN.fullmatch(core[: -len(daypart)].strip())
                break
    kind = _kind_of(match) if match is not None else None
    resolved = _resolve_kind(kind, core, base) if kind is not None else _resolve_explicit(core, base)
    if resolved is None:
        return None
    return ResolvedTime(reported, resolved.value, resolved.grain, resolved.phrase)


def _resolve_explicit(core: str, base: date) -> ResolvedTime | None:
    """A named calendar date ("august 12", "12th of August 2026"), year defaulting to the anchor's.

    ``None`` for anything that is not one — including a month name the vocabulary
    does not carry and a day number no month has, both of which are ordinary text
    rather than a malformed date.
    """
    for pattern in (_EXPLICIT_MONTH_FIRST, _EXPLICIT_DAY_FIRST):
        match = pattern.fullmatch(core)
        if match is None:
            continue
        month = MONTH_NAME_TO_NUMBER.get(match.group("month").lower())
        if month is None:
            continue
        year = int(match.group("year")) if match.group("year") else base.year
        try:
            value = date(year, month, int(match.group("day")))
        except ValueError:
            # "february 30" — a well-formed phrase naming no day. Fail-open: the
            # surface survives unresolved, exactly as an unmatched one would.
            return None
        return _day(core, value)
    return None


# -----------------------------------------------------------------------------
# Moved here from a retired fact-temporal utility module (this module is the one
# owner of temporal handling): a broader, less closed-set vocabulary than :data:`DEIXIS_PATTERN`
# above (full calendar dates, offset arithmetic off a written anchor, query-side date extraction).
# Its own history is preserved in the individual functions' comments (the ``word2number``
# deletion note on :func:`relative_number_value`, the fail-open ``KEEP`` notes on
# :func:`safe_date` / :func:`parse_timestamp_date`).
# -----------------------------------------------------------------------------

MONTH_NAMES = lexicon.month_names()
WEEKDAYS = {name: index for index, name in enumerate(WEEKDAY_NAMES)}
SEASONS = lexicon.seasons()
MONTH_NAME_TO_NUMBER = lexicon.month_name_to_number()
NUMBER_WORD_VALUES = lexicon.number_word_values()
NUMBER_WORD_SCALES = lexicon.number_word_scales()


def content_with_relative_time(content: str, timestamp: Any) -> str:
    """Append a resolved-relative-time annotation to the content, or return it unchanged.

    The annotation reads ``[resolved relative time: <note>; anchor: <day>]`` and exists so that text
    saying "last Tuesday" still retrieves when a reader asks about the actual date. The content is
    returned untouched when it contains no relative expression, when it already carries such an
    annotation, or when ``timestamp`` gives no day to anchor against — the original wording is never
    rewritten, only extended.
    """
    note = relative_resolution_note(content, timestamp)
    if not note or "[resolved relative time:" in content:
        return content
    anchor = parse_timestamp_date(timestamp)
    if anchor is None:
        return content
    return f"{content} [resolved relative time: {note}; anchor: {format_human_day(anchor)}]"


def temporal_text(timestamp: Any, content: str) -> str | None:
    """The single human-readable time a fact should be filed under, or ``None`` if it has none.

    Preference order: a date computed from an offset phrase in the content ("the week before 4 May
    2023"), then an explicit date written in the content, then — if the content holds a relative
    expression the timestamp can anchor — a ``resolved relative time: …`` note, and finally the raw
    timestamp rendered as text. Dates are formatted as ``D Month YYYY``; the last case is whatever
    string the timestamp stringifies to, so it is not normalized.

    ``None`` means neither the content nor the timestamp carried anything datable.
    """
    offset_date = relative_offset_date_value(content)
    if offset_date:
        return format_human_day(offset_date)
    explicit = explicit_date_text(content)
    if explicit:
        return explicit
    timestamp_text = timestamp_to_text(timestamp)
    note = relative_resolution_note(content, timestamp)
    if note:
        anchor = parse_timestamp_date(timestamp)
        suffix = f"; anchor: {format_human_day(anchor)}" if anchor is not None else ""
        return f"resolved relative time: {note}{suffix}"
    return timestamp_text


def explicit_date_text(text: str) -> str | None:
    """The first explicit date written in the text, rendered ``D Month YYYY``; ``None`` if there is none."""
    parsed = explicit_date_value(text)
    return format_human_day(parsed) if parsed else None


def explicit_date_value(text: str) -> date | None:
    """The first fully-specified calendar date written in the text, as a plain ``date``.

    Recognises ``May 4, 2023`` and ``May 4th 2023``, ``4 May 2023``, and ``2023-05-04``, in that
    order of preference; all three require an explicit four-digit year, so no date is ever inferred
    from context. Matching is case-insensitive and scans anywhere in the text.

    ``None`` when nothing matches, and also when a match is calendrically impossible (``31 February
    2023``) — an unparseable or invalid date is reported as absent, never raised.
    """
    value = str(text or "")
    if not value:
        return None
    month_names = "|".join(MONTH_NAMES.values())
    month_lookup = {name.lower(): month for month, name in MONTH_NAMES.items()}
    month_first = re.search(
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*|\s+)(\d{{4}})\b",
        value,
        re.I,
    )
    if month_first:
        month = month_lookup.get(month_first.group(1).lower())
        if month:
            return safe_date(int(month_first.group(3)), month, int(month_first.group(2)))
    day_first = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})\s+(\d{{4}})\b",
        value,
        re.I,
    )
    if day_first:
        month = month_lookup.get(day_first.group(2).lower())
        if month:
            return safe_date(int(day_first.group(3)), month, int(day_first.group(1)))
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", value)
    if iso:
        return safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    return None


def relative_offset_date_value(text: str) -> date | None:
    """Resolve a phrase that offsets from a date written in the same text, such as "two days before 4 May 2023".

    Day, week, weekend and named-weekday offsets are tried in that order and the first that resolves
    wins. The anchor must itself be an explicit date inside the text — this function never consults
    a statement timestamp, so it is independent of when the text was recorded.

    ``None`` when no offset phrase is present, when its anchor is not an explicit date, or when the
    amount is not a number this module can read.
    """
    value = str(text or "")
    day_offset = _relative_day_offset_date_value(value)
    if day_offset is not None:
        return day_offset
    week_offset = _relative_week_offset_date_value(value)
    if week_offset is not None:
        return week_offset
    weekend_offset = _relative_weekend_offset_date_value(value)
    if weekend_offset is not None:
        return weekend_offset
    return _relative_weekday_offset_date_value(value)


def _relative_day_offset_date_value(value: str) -> date | None:
    """Resolve "<n> days before/after <explicit date>"; ``None`` unless both parts parse."""
    match = re.search(
        r"\b(?P<amount>\d+|[a-z]+(?:[-\s]+[a-z]+){0,3})\s+days?\s+"
        r"(?P<direction>before|after)\s+"
        r"(?P<anchor>[^\].;]+)",
        value,
        flags=re.I,
    )
    if not match:
        return None
    anchor = explicit_date_value(match.group("anchor"))
    amount = relative_number_value(match.group("amount"))
    if anchor is None or amount is None:
        return None
    delta = timedelta(days=amount)
    return anchor - delta if match.group("direction").lower() == "before" else anchor + delta


def _relative_week_offset_date_value(value: str) -> date | None:
    """Resolve "<n> weeks before/after <explicit date>" as a whole-week shift of the anchor day.

    The amount is optional and defaults to one, so "the week before 4 May 2023" resolves. ``None``
    unless the anchor is an explicit date and the amount reads as a number.
    """
    match = re.search(
        r"\b(?:the\s+)?(?:(?P<amount>\d+|[a-z]+(?:[-\s]+[a-z]+){0,3})\s+)?weeks?\s+"
        r"(?P<direction>before|after)\s+"
        r"(?P<anchor>[^\].;]+)",
        value,
        flags=re.I,
    )
    if not match:
        return None
    anchor = explicit_date_value(match.group("anchor"))
    amount = relative_number_value(match.group("amount") or "1")
    if anchor is None or amount is None:
        return None
    delta = timedelta(days=7 * amount)
    return anchor - delta if match.group("direction").lower() == "before" else anchor + delta


def _relative_weekend_offset_date_value(value: str) -> date | None:
    """Resolve "<n> weekends before/after <explicit date>" to that weekend's Saturday.

    The count is optional and defaults to one; counting starts at the nearest Saturday strictly on
    the named side of the anchor, so "the weekend after 4 May 2023" is the Saturday that follows it.
    ``None`` unless the anchor is an explicit date and the amount reads as a number.
    """
    match = re.search(
        r"\b(?:the\s+)?(?:(?P<amount>\d+|[a-z]+(?:[-\s]+[a-z]+){0,3})\s+)?weekends?\s+"
        r"(?P<direction>before|after)\s+"
        r"(?P<anchor>[^\].;]+)",
        value,
        flags=re.I,
    )
    if not match:
        return None
    anchor = explicit_date_value(match.group("anchor"))
    amount = relative_number_value(match.group("amount") or "1")
    if anchor is None or amount is None:
        return None
    if match.group("direction").lower() == "before":
        resolved = previous_weekday(anchor, WEEKDAYS["saturday"])
        return resolved - timedelta(days=7 * (amount - 1))
    resolved = next_weekday(anchor, WEEKDAYS["saturday"])
    return resolved + timedelta(days=7 * (amount - 1))


def _relative_weekday_offset_date_value(value: str) -> date | None:
    """Resolve "<weekday> before/after <explicit date>" to the nearest such weekday, excluding the anchor itself.

    ``None`` unless the anchor is an explicit date.
    """
    match = re.search(
        r"\b(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
        r"(?P<direction>before|after)\s+"
        r"(?P<anchor>[^\].;]+)",
        value,
        flags=re.I,
    )
    if not match:
        return None
    anchor = explicit_date_value(match.group("anchor"))
    if anchor is None:
        return None
    weekday = WEEKDAYS[match.group("weekday").lower()]
    if match.group("direction").lower() == "before":
        return previous_weekday(anchor, weekday)
    return next_weekday(anchor, weekday)


def safe_date(year: int, month: int, day: int) -> date | None:
    """Construct a ``date``, returning ``None`` instead of raising when the combination is impossible.

    Callers use this wherever a year/month/day is assembled from parsed text or offset arithmetic, so
    that an out-of-range day reads as "no date here" rather than aborting an extraction run.
    """
    try:
        return date(year, month, day)
    except ValueError as exc:
        # KEEP: a calendrically-invalid y/m/d combination
        # is reachable from ordinary relative-date arithmetic (day overflow),
        # not data corruption.
        logger.warning("Calendrically invalid date %s-%s-%s: %s", year, month, day, exc)
        return None


def relative_resolution_note(
    text: str,
    timestamp: Any,
    *,
    dedupe: Callable[[list[str]], list[str]] | None = None,
) -> str | None:
    """Spell out every relative time expression in the text against the day ``timestamp`` falls on.

    The result is a ``"; "``-joined list of ``<phrase> = <resolution>`` clauses, deduplicated while
    keeping first-seen order — for example ``yesterday = 3 May 2023; last month = April 2023``. Day
    resolutions render as ``D Month YYYY``, month resolutions as ``Month YYYY``, year and season
    resolutions as a bare year (with the season name).

    Bare weekday mentions are resolved too, looking forward or backward according to the sentence's
    tense; reading that tense goes through the semantic operator service, which degrades from a
    parsed reading to a keyword one when no spaCy model is loaded.

    ``None`` when ``timestamp`` yields no anchor day, or when the text holds no relative expression
    — the two cases are not distinguishable from the return value.

    ``dedupe`` overrides the deduplication used on the assembled clauses.
    """
    anchor = parse_timestamp_date(timestamp)
    if anchor is None:
        return None
    lower = str(text or "").lower()
    notes: list[str] = []
    if re.search(r"\byesterday\b", lower):
        notes.append(f"yesterday = {format_human_day(anchor - timedelta(days=1))}")
    if re.search(r"\b(?:last|previous)\s+(?:night|evening)\b", lower):
        notes.append(f"last night = {format_human_day(anchor - timedelta(days=1))}")
    if re.search(r"\byesterday\s+(?:morning|afternoon|evening|night)\b", lower):
        notes.append(f"yesterday = {format_human_day(anchor - timedelta(days=1))}")
    if re.search(r"\btoday\b", lower):
        notes.append(f"today = {format_human_day(anchor)}")
    if re.search(r"\b(?:this\s+)?(?:morning|afternoon|evening|tonight)\b", lower):
        notes.append(f"today = {format_human_day(anchor)}")
    if re.search(r"\btomorrow\b", lower):
        notes.append(f"tomorrow = {format_human_day(anchor + timedelta(days=1))}")
    if re.search(r"\btomorrow\s+(?:morning|afternoon|evening|night)\b", lower):
        notes.append(f"tomorrow = {format_human_day(anchor + timedelta(days=1))}")
    day_ago = _relative_amount_ago_match(lower, units={"day", "days"})
    if day_ago:
        phrase, amount = day_ago
        notes.append(f"{phrase} = {format_human_day(anchor - timedelta(days=amount))}")
    weeks_ago = _relative_amount_ago_match(lower, units={"week", "weeks"})
    if weeks_ago:
        phrase, amount = weeks_ago
        notes.append(f"{phrase} = {format_human_day(anchor - timedelta(days=7 * amount))}")
    for phrase, _weekday in re.findall(
        r"\b((?:last|previous|next)\s+"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
        lower,
    ):
        direction, day_name = phrase.split()
        weekday_idx = WEEKDAYS[day_name]
        resolved = next_weekday(anchor, weekday_idx) if direction == "next" else previous_weekday(anchor, weekday_idx)
        notes.append(f"{phrase} = {format_human_day(resolved)}")
    notes.extend(_bare_weekday_resolution_notes(lower, anchor))
    if re.search(r"\b(last|previous)\s+week\b|\bweek before\b", lower):
        notes.append(f"last week = the week before {format_human_day(anchor)}")
    if re.search(r"\b(?:last|previous|this\s+past|past)\s+weekend\b|\bweekend before\b", lower):
        saturday = previous_weekday(anchor, WEEKDAYS["saturday"])
        notes.append(f"last weekend = {format_human_day(saturday)} to {format_human_day(saturday + timedelta(days=1))}")
    if re.search(r"\bthis\s+weekend\b", lower):
        saturday = this_weekend_start(anchor)
        notes.append(f"this weekend = {format_human_day(saturday)} to {format_human_day(saturday + timedelta(days=1))}")
    if re.search(r"\bthis month\b", lower):
        notes.append(f"this month = {format_human_month(anchor)}")
    if re.search(r"\bnext month\b", lower):
        year = anchor.year + (1 if anchor.month == 12 else 0)
        month = 1 if anchor.month == 12 else anchor.month + 1
        notes.append(f"next month = {MONTH_NAMES[month]} {year}")
    if re.search(r"\blast month\b", lower):
        year = anchor.year - (1 if anchor.month == 1 else 0)
        month = 12 if anchor.month == 1 else anchor.month - 1
        notes.append(f"last month = {MONTH_NAMES[month]} {year}")
    if re.search(r"\bthis year\b", lower):
        notes.append(f"this year = {anchor.year}")
    if re.search(r"\bnext year\b", lower):
        notes.append(f"next year = {anchor.year + 1}")
    if re.search(r"\b(?:last|previous) year\b", lower):
        notes.append(f"last year = {anchor.year - 1}")
    years_ago = _relative_years_ago_match(lower)
    if years_ago:
        phrase, amount_text = years_ago
        amount = relative_number_value(amount_text)
        if amount:
            notes.append(f"{phrase} = {anchor.year - amount}")
    for direction, season_name in re.findall(
        r"\b(last|previous|this|next)\s+(" + "|".join(SEASONS) + r")\b",
        lower,
    ):
        season_year = anchor.year
        if direction in {"last", "previous"}:
            season_year -= 1
        elif direction == "next":
            season_year += 1
        notes.append(f"{direction} {season_name} = {SEASONS[season_name]} {season_year}")
    if not notes:
        return None
    dedupe_fn = dedupe or _dedupe
    return "; ".join(dedupe_fn(notes))


def _bare_weekday_resolution_notes(text: str, anchor: date) -> list[str]:
    """Resolve weekday names carrying no ``last``/``next`` modifier, relative to ``anchor``.

    A weekday adjacent to a month name or a number is skipped as part of a written date rather than
    a bare reference. Direction is future when the weekday follows a forward modifier ("this Friday",
    "this coming Friday": the one ahead, whatever the tense, since a team says it of a plan in the
    present tense), when the sentence reads as future-tense, or when the word ``next`` appears nearby;
    past otherwise. Either way the anchor day itself is an allowed answer, so "Friday" on a Friday
    resolves to that same day.
    """
    notes: list[str] = []
    month_names = {name.lower() for name in MONTH_NAMES.values()}
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    weekday_positions = [(index, token) for index, token in enumerate(tokens) if token in WEEKDAYS]
    if not weekday_positions:
        return notes
    directional_modifiers = {"last", "previous", "next"}
    forward_modifiers = set(lexicon.weekday_forward_modifiers())
    tense: list[bool] = []  # the sentence's tense, read from its parse only when a bare weekday needs it

    def future_tense() -> bool:
        if not tense:
            tense.append("future" in get_semantic_operator_service().profile(text).tense_aspects)
        return tense[0]

    for index, token in weekday_positions:
        previous_token = tokens[index - 1] if index > 0 else ""
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        if previous_token in directional_modifiers:
            continue
        if previous_token in month_names or next_token in month_names or next_token.isdigit():
            continue
        window = tokens[max(0, index - 6) : min(len(tokens), index + 4)]
        if previous_token in forward_modifiers or "next" in window or future_tense():
            resolved = next_weekday_including_anchor(anchor, WEEKDAYS[token])
        else:
            resolved = previous_weekday_including_anchor(anchor, WEEKDAYS[token])
        notes.append(f"{token} = {format_human_day(resolved)}")
    return notes


@lru_cache(maxsize=4096)
def relative_number_value(value: str) -> int | None:
    """Read the count in a relative time phrase, as digits, spelled-out English, or an indefinite article.

    ``"3"``, ``"three"``, ``"twenty-three"``, ``"one hundred"`` and ``"a"``/``"an"`` (both 1) all
    resolve; hyphens and runs of whitespace are equivalent and case is ignored. ``None`` when the
    value contains any token outside the number vocabulary, so partial parses never leak a wrong
    count.
    """
    clean = re.sub(r"[-\s]+", " ", str(value or "").strip().lower())
    if clean.isdigit():
        return int(clean)
    if clean in {"a", "an"}:
        return 1
    # A `word2number` fallback used to sit here, guarded by a `try/except Exception: return None`.
    # `word2number` is not a declared dependency of this package, so the branch only ever
    # swallowed ModuleNotFoundError and never saw the "not a number word" condition it was
    # kept for. Deleted as dead code: `_number_words_value` below already resolves the
    # number-word vocabulary this module needs ("three", "twenty three", "one hundred", ...).
    return _number_words_value(clean)


def _number_words_value(clean: str) -> int | None:
    """Sum a whitespace-separated run of English number words into an integer.

    Expects already-normalized input: lowercase, single-spaced. ``and`` is ignored as filler; scale
    words multiply what precedes them, and ``thousand`` and above close off a group. ``None`` for an
    empty string or as soon as any token is not a number word — the whole phrase must be readable.
    """
    if not clean:
        return None
    total = 0
    current = 0
    seen = False
    for token in clean.split():
        if token == "and":
            continue
        value = NUMBER_WORD_VALUES.get(token)
        if value is not None:
            current += value
            seen = True
            continue
        scale = NUMBER_WORD_SCALES.get(token)
        if scale is not None:
            current = max(1, current) * scale
            if scale >= 1000:
                total += current
                current = 0
            seen = True
            continue
        return None
    return total + current if seen else None


def _relative_years_ago_match(text: str) -> tuple[str, str] | None:
    """Find a "<amount> year(s) ago" phrase and return it alongside the amount as text, not a number.

    Up to five tokens before the unit are tried as the amount, and the reading with the largest value
    wins, shortest span breaking ties — so "twenty two years ago" is read as 22 rather than 2. The
    caller converts the returned amount text itself.

    ``None`` when the text holds no such phrase or no leading token run parses as a number.
    """
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    for index, token in enumerate(tokens):
        if token not in {"year", "years"}:
            continue
        if index + 1 >= len(tokens) or tokens[index + 1] != "ago":
            continue
        candidates: list[tuple[int, int, str]] = []
        for start in range(index - 1, max(-1, index - 6), -1):
            amount_text = " ".join(tokens[start:index])
            parsed = relative_number_value(amount_text)
            if parsed is not None:
                candidates.append((parsed, index - start, amount_text))
        if candidates:
            parsed, _length, best_amount = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
            del parsed
            return f"{best_amount} {token} ago", best_amount
    return None


def _relative_amount_ago_match(text: str, *, units: set[str]) -> tuple[str, int] | None:
    """Find an "<amount> <unit> ago" phrase for the given unit words, returning the phrase and its count.

    Up to five tokens before the unit are tried as the amount; the longest span that parses wins,
    largest value breaking ties. The phrase is rebuilt from the winning amount text and the unit as
    it appeared, so it can be quoted back in a resolution note.

    ``None`` when the text holds no such phrase or no leading token run parses as a number.
    """
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    for index, token in enumerate(tokens):
        if token not in units:
            continue
        if index + 1 >= len(tokens) or tokens[index + 1] != "ago":
            continue
        candidates: list[tuple[int, int, str]] = []
        for start in range(index - 1, max(-1, index - 6), -1):
            amount_text = " ".join(tokens[start:index])
            parsed = relative_number_value(amount_text)
            if parsed is not None:
                candidates.append((parsed, index - start, amount_text))
        if candidates:
            parsed, _length, best_amount = sorted(candidates, key=lambda item: (-item[1], -item[0]))[0]
            return f"{best_amount} {token} ago", parsed
    return None


def parse_timestamp_date(value: Any) -> date | None:
    """Reduce any timestamp shape to the calendar day it names, as a plain ``date``.

    Accepts a ``datetime``, a ``date``, or text: full ISO-8601 (with ``Z`` treated as ``+00:00``),
    otherwise the leading ten characters as an ISO date. No timezone conversion happens — an aware
    value keeps its own offset's calendar day, so a UTC-evening timestamp and the same instant in a
    later zone can yield different days. The returned ``date`` carries no zone of its own.

    ``None`` for an empty or unparseable value; unparseable input is never raised on.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        # KEEP: interim attempt of a two-stage external
        # timestamp parse; the date-only fallback below is tried next, and
        # its own failure is what gets logged.
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        # KEEP: per-item isolation — external timestamp
        # text, final fallback of the two-stage parse.
        logger.warning("Unparseable timestamp text %r: %s", text, exc)
        return None


def format_human_day(value: date) -> str:
    """Render a day as ``D Month YYYY`` — the one day format every resolution note in this module uses."""
    return f"{value.day} {MONTH_NAMES[value.month]} {value.year}"


def format_human_month(value: date) -> str:
    """Render the month a day falls in as ``Month YYYY``; the day itself is discarded."""
    return f"{MONTH_NAMES[value.month]} {value.year}"


def previous_weekday(anchor: date, weekday: int) -> date:
    """The most recent given weekday strictly before ``anchor``; on a matching anchor it steps back a week.

    ``weekday`` is Monday-zero, as ``date.weekday()`` returns.
    """
    delta = (anchor.weekday() - weekday) % 7
    if delta == 0:
        delta = 7
    return anchor - timedelta(days=delta)


def next_weekday(anchor: date, weekday: int) -> date:
    """The first given weekday strictly after ``anchor``; on a matching anchor it steps forward a week.

    ``weekday`` is Monday-zero, as ``date.weekday()`` returns.
    """
    delta = (weekday - anchor.weekday()) % 7
    if delta == 0:
        delta = 7
    return anchor + timedelta(days=delta)


def previous_weekday_including_anchor(anchor: date, weekday: int) -> date:
    """The given weekday at or before ``anchor`` — returns ``anchor`` itself when it already matches."""
    return anchor - timedelta(days=(anchor.weekday() - weekday) % 7)


def next_weekday_including_anchor(anchor: date, weekday: int) -> date:
    """The given weekday at or after ``anchor`` — returns ``anchor`` itself when it already matches."""
    return anchor + timedelta(days=(weekday - anchor.weekday()) % 7)


def this_weekend_start(anchor: date) -> date:
    """The Saturday of the weekend a speaker on ``anchor`` would call "this weekend".

    Forward-looking on every day except Sunday, when the current weekend's Saturday is yesterday and
    that day is returned instead. On a Saturday the anchor itself is the answer.
    """
    if anchor.weekday() == WEEKDAYS["sunday"]:
        return previous_weekday(anchor, WEEKDAYS["saturday"])
    return next_weekday_including_anchor(anchor, WEEKDAYS["saturday"])


def timestamp_to_text(value: Any) -> str | None:
    """The timestamp's string form, stripped; ``None`` for ``None`` or for anything that strips to empty.

    No parsing or normalization — whatever the value stringifies to is what comes back.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None
