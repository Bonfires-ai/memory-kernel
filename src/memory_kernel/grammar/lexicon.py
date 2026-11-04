"""English content vocabulary (month, weekday and season names, number words, relative-time words, entity type hints),
read once from ``data/runtime-grammar/en/lexicon.json``. Rule 18: no word list lives in Python; the modules that need
one take it from here. Tables derived from the data (name to number) are computed here, not stored twice."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

__all__ = [
    "dayparts_after_this",
    "dayparts_after_tomorrow",
    "leading_function_words",
    "month_name_to_number",
    "month_names",
    "number_word_scales",
    "number_word_values",
    "relative_determiners",
    "relative_offsets",
    "safe_type_hints",
    "seasons",
    "weekday_forward_modifiers",
    "weekday_names",
]

LEXICON_FILE = ("data", "runtime-grammar", "en", "lexicon.json")


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    return json.loads(resources.files("memory_kernel").joinpath(*LEXICON_FILE).read_text(encoding="utf-8"))


def month_names() -> dict[int, str]:
    """Month number (1-12) to its capitalised English name."""
    return {number: name for number, name in enumerate(_data()["month_names"], start=1)}


def month_name_to_number() -> dict[str, int]:
    """Lowercase full names and the listed abbreviations ("sept") to the month number; an abbreviation names the month
    whose name starts with its first three letters."""
    full = {name.lower(): number for number, name in month_names().items()}
    by_prefix = {name[:3]: number for name, number in full.items()}
    return {**full, **{abbr: by_prefix[abbr[:3]] for abbr in _data()["month_abbreviations"]}}


def weekday_names() -> tuple[str, ...]:
    """Lowercase weekday names, Monday first (the index is ``date.weekday()``)."""
    return tuple(_data()["weekdays"])


def seasons() -> dict[str, str]:
    return dict(_data()["seasons"])


def number_word_values() -> dict[str, int]:
    return dict(_data()["number_word_values"])


def number_word_scales() -> dict[str, int]:
    return dict(_data()["number_word_scales"])


def leading_function_words() -> tuple[str, ...]:
    """The prepositions and articles stripped (with their trailing space) before a date filler is read."""
    return tuple(_data()["leading_function_words"])


def dayparts_after_this() -> tuple[str, ...]:
    return tuple(_data()["dayparts_after_this"])


def dayparts_after_tomorrow() -> tuple[str, ...]:
    return tuple(_data()["dayparts_after_tomorrow"])


def relative_determiners() -> tuple[str, ...]:
    """The words that point a week, month or year at the speaker's now ("this", "last", "next")."""
    return tuple(_data()["relative_determiners"])


def relative_offsets() -> tuple[str, ...]:
    """The determiners that name another period than the current one ("last", "next")."""
    return tuple(_data()["relative_offsets"])


def weekday_forward_modifiers() -> tuple[str, ...]:
    """The words before a weekday that point it at the one ahead whatever the tense ("this Friday", "this coming
    Friday")."""
    return tuple(_data()["weekday_forward_modifiers"])


def safe_type_hints() -> tuple[str, ...]:
    return tuple(_data()["safe_type_hints"])
