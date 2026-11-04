"""Agent mentions: a mention of the agent makes it an addressee. At the start or the end of a message the
mention is a vocative and is removed before parsing; in the middle it is part of the sentence and becomes the bare name."""

import re
from collections.abc import Sequence


def _pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w@.])@{re.escape(name)}(?!\w|\.\w)", re.IGNORECASE)


def agent_mentions(text: str, handles: Sequence[str]) -> tuple[str, list[str]]:
    """``(text, mentioned names)``; names are the handles without ``@``, in ``handles`` order."""
    mentioned: list[str] = []
    for handle in handles:
        name = handle.strip().lstrip("@")
        pat = _pattern(name)
        if not pat.search(text):
            continue
        mentioned.append(name)
        lead = re.compile(rf"^\s*{pat.pattern}[\s,:;!]*", re.IGNORECASE)
        trail = re.compile(rf"[\s,;:\-]*{pat.pattern}(?=\s*[.!?]*\s*$)", re.IGNORECASE)
        while True:
            new = trail.sub("", lead.sub("", text, count=1), count=1)
            if new == text:
                break
            text = new
        text = pat.sub(lambda _, bare=name: bare, text).strip()
    if mentioned and not re.search(r"\w", text):
        text = ""  # nothing but the mention and punctuation
    return text, mentioned
