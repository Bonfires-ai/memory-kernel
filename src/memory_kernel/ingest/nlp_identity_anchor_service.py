"""Identity/location anchor extraction for NLP fact building."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class IdentityAnchorDependencies:
    """The one callable this service borrows from the NLP runtime.

    ``dedupe`` must preserve first-seen order — the query lists it returns are consumed as ranked surfaces.
    """

    dedupe: Callable[[list[str]], list[str]]


class IdentityAnchorService:
    """Recovers facts a speaker states about themselves in passing — "my hometown, Springfield".

    The extraction is purely pattern-based: no model, no lexicon beyond a fixed set of place descriptors, so
    it is deterministic and cheap. It only recognises first-person appositives, so a place mentioned about
    someone else is not an anchor.
    """

    def __init__(self, deps: IdentityAnchorDependencies):
        """Bind the service to a dependency bundle that stays live for the service's lifetime."""
        self.deps = deps

    def location_anchors(self, content: str, *, speaker: str) -> list[dict[str, str]]:
        """Extract first-person appositive place anchors from a statement."""
        anchors: list[dict[str, str]] = []
        descriptor_pattern = r"home country|hometown|home town|home city|home state"
        pattern = re.compile(
            rf"\b(?:my|our)\s+(?P<descriptor>{descriptor_pattern})\s*,\s*(?P<value>[A-Z][A-Za-z0-9' -]{{1,60}})",
            re.I,
        )
        for match in pattern.finditer(content or ""):
            descriptor = re.sub(r"\s+", " ", match.group("descriptor").lower()).strip()
            value = self.clean_anchor_value(match.group("value"))
            if not value:
                continue
            readable_descriptor = "hometown" if descriptor in {"hometown", "home town"} else descriptor
            anchors.append(
                {
                    "descriptor": readable_descriptor,
                    "value": value,
                    "content": f"{speaker}'s {readable_descriptor} is {value}.",
                }
            )
        return self.dedupe_anchors(anchors)

    @staticmethod
    def clean_anchor_value(value: str) -> str:
        """Trim a captured place name down to the name itself, or return an empty string to reject it.

        The capture is cut at the first sentence break or coordinating word, then stripped of surrounding
        punctuation and of trailing tokens containing no letters. What remains is rejected — returned as an
        empty string — when it is blank, longer than five words, or entirely lowercase, since a place name in
        this position is expected to be capitalised.
        """
        text = re.split(r"\s*(?:[.;!?]|\band\b|\bbut\b|\bbecause\b)\s*", value or "", maxsplit=1, flags=re.I)[0]
        text = re.sub(r"\s+", " ", text).strip(" ,:-")
        words = text.split()
        while words and not any(char.isalpha() for char in words[-1]):
            words.pop()
        if not words or len(words) > 5:
            return ""
        if not any(word[:1].isupper() for word in words):
            return ""
        return " ".join(words)

    @staticmethod
    def dedupe_anchors(anchors: list[dict[str, str]]) -> list[dict[str, str]]:
        """Drop repeated descriptor-and-value pairs, comparing case-insensitively and keeping the first anchor.

        Two anchors that agree on descriptor and value collapse even if their rendered content differs.
        """
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, str]] = []
        for anchor in anchors:
            key = (anchor["descriptor"].lower(), anchor["value"].lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(anchor)
        return out

    def anchor_queries(self, *, speaker: str, descriptor: str, value: str, fallback: list[str]) -> list[str]:
        """Phrase the questions an identity anchor answers, so the fact is reachable by how it would be asked.

        The generated questions come first and ``fallback`` fills out the rest; the whole list is deduplicated
        and capped at eight. A blank ``value`` simply omits the question that would have named it.
        """
        queries = [
            f"What is {speaker}'s {descriptor}?",
            f"Where is {speaker} from?",
            f"Which place is {speaker}'s {descriptor}?",
        ]
        if value:
            queries.append(f"What is known about {speaker} and {value}?")
        return self.deps.dedupe([*queries, *fallback])[:8]


__all__ = ["IdentityAnchorDependencies", "IdentityAnchorService"]
