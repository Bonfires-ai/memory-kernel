"""Python transcription of the affordance-channel primitives.

Direct, function-for-function port of
the benchmark harness's TypeScript affordance-channel provider — the query-
comprehension-driven boosts and the answer-directive map for the kernel-
native search recipe. Captions (`b:captions`) are not ported.

The directive strings (leg-4 v2) are part of the measured recipe: change them only with a
directive version bump.
"""

from __future__ import annotations

#: The export contract this module never declared. Every name below is consumed across
#: the repo boundary by host applications; declaring ``__all__`` states as API what the boundary
#: already treats as API. Additive only -- nothing wildcard-imports this module, so no caller's
#: resolution changes. NOT a licence to make anything else private: ``__all__`` governs ``import *``
#: alone, so a name's absence from this list says nothing about who may reach it.
__all__ = []


# Leg-4 v2: strings are part of the recipe;
# change them only with a directive version bump.
def answer_directive(wh_slot: str | None) -> str | None:
    """The answering instruction for a question shape, or None when the slot has no directive.

    The returned text is part of the measured recipe, not cosmetic prose: it tells the answerer how to
    shape a reply to a "when"/"list"/"inference"/"how_long"/"how_many" or plain entity question. A None
    return means the caller emits no slot directive at all.
    """
    if wh_slot == "when":
        return (
            "The question asks for a specific date or time. Resolve relative "
            "references in the evidence ('last Saturday', 'the week before') "
            "against that message's own timestamp, then answer with the most "
            "specific absolute date supported."
        )
    if wh_slot == "list":
        return (
            "The question asks for multiple items. Enumerate ALL distinct items "
            "supported by the evidence and the context window; do not stop at the "
            "first. Do not add items of a kind the question did not ask about."
        )
    if wh_slot == "inference":
        return (
            "This is an inference question. Reason from the evidence to a "
            "definite answer (e.g. yes/no or likely/unlikely) with a brief "
            "reason. Do NOT answer 'Not enough information' if the evidence "
            "supports a reasonable inference."
        )
    if wh_slot == "how_long":
        return (
            "Answer with the duration AND its absolute anchor (e.g. 'since "
            "2016'), derived from the evidence timestamps if needed."
        )
    if wh_slot == "how_many":
        return "Answer with a specific quantity or duration."
    if wh_slot in ("who", "where", "what", "which", "how_degree"):  # a degree other than count or length: a fact
        return "Answer with the specific entity or fact, concisely."
    return None


# Rides every slot directive; NEVER emitted alone.
def directive_preamble() -> str:
    """The shared instruction prefixed to every slot directive; never emitted on its own.

    It fixes the two answering defaults the slot directives assume: quote the evidence's own concrete
    wording, and reserve "Not enough information" for evidence that is genuinely irrelevant.
    """
    return (
        "State the concrete fact(s) in the evidence's own words rather than a "
        "vague paraphrase. Answer 'Not enough information' ONLY when the "
        "evidence and context contain nothing relevant to the question."
    )


# ---------- wh_role affordance rows (flag-gated) ----------
