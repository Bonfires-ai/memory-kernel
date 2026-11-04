"""Referent mentions — what an utterance POINTS AT, kept where the meaning is still whole.

`role_bindings_from_meaning` performs the ``?var`` join and then DESTROYS the variable, and
it drops every deictic-headed NP ("my monkey") because such a surface must never mint an
entity through the effect ladder. Both are right for BINDINGS and wrong for REFERENTS: the
variable IS intra-message referent identity, and "my monkey" is exactly the population this
lane exists to surface. So this module runs the same join for a different product, and owns
nothing else — no vocabulary of its own (every set below is REACHED, not declared),
no I/O, no knowledge of who the speaker is. The HOST resolves owners and addresses; the
kernel says WHAT KIND of pointing this is and whether the laws permit a create.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory_kernel.grammar.pronouns import of_token

__all__ = ["MentionAnalysis", "classify_mention"]

#: The clitic that marks an explicit possessor. MORPHOLOGY, not a vocabulary: it is a
#: suffix rule over any surface, and enumerates nothing.
_POSSESSIVE_CLITICS = ("'s", "’s")

#: PTB tags of the pronouns and wh-words that never name anything by themselves ("he", "whose", "when").
_DEICTIC_TAGS = frozenset({"PRP", "PRP$", "WP", "WP$", "WDT", "WRB"})
_POLES = {1: "speaker", 2: "addressee"}


@dataclass(frozen=True)
class MentionAnalysis:
    """What KIND of pointing a surface performs, and who (if anyone) it points through."""

    #: definite | indefinite | possessive | name | pronoun | bare. ``name`` is stamped
    #: HOST-side (only the canon knows which surfaces are people); the kernel never claims it.
    mention: str
    head_surface: str
    #: "" | "speaker" | "addressee" | "explicit" — WHICH pole owns it, never WHO.
    owner_hint: str
    #: The possessor's own surface, non-empty only for ``owner_hint == "explicit"``.
    owner_surface: str


def _pole(token: Any) -> str:
    pronoun = of_token(token)
    return _POLES.get(pronoun.person, "") if pronoun else ""


def classify_mention(mention: Any) -> MentionAnalysis:
    """Classify one mention from its parse: ``mention`` is a spaCy Span (or Doc) over the mention's own tokens.
    Pronoun and possessive detection depends on the tagger's PRP/PRP$ tags.

    ORDER IS THE RULE, and every branch reads the parse:

    1. a single pronoun or wh-word token (PTB ``PRP``, ``PRP$``, ``WP``, ``WP$``, ``WDT``, ``WRB``) is a ``pronoun``; a
       first- or second-person one carries its pole (its Person), which is what makes "I"/"you" resolvable;
    2. a possessive-determiner head (``PRP$``: "my", "your", "his") is a ``possessive`` whose head is the remainder — the
       pole comes from its Person, and a THIRD-person possessive ("his monkey") is a possessive with NO owner:
       cross-message anaphora is out of scope and absence is never a guess;
    3. a leading ``'s``-clitic token is an EXPLICIT possessive naming its possessor, which
       the caller resolves as a mention of its own;
    4. a leading determiner decides definiteness (its ``Definite`` feature: ``Def`` or ``Ind``);
    5. anything else is ``bare`` — the honest statement that nothing marked it.
    """
    text = mention.text.strip()
    if not text:
        return MentionAnalysis(mention="bare", head_surface="", owner_hint="", owner_surface="")
    first_token = mention[0]
    if len(mention) == 1 and first_token.tag_ in _DEICTIC_TAGS:
        return MentionAnalysis(mention="pronoun", head_surface=text, owner_hint=_pole(first_token), owner_surface="")
    first, _, rest = text.partition(" ")
    rest = rest.strip()
    if rest and first_token.tag_ == "PRP$":
        return MentionAnalysis(mention="possessive", head_surface=rest, owner_hint=_pole(first_token), owner_surface="")
    if rest and any(first.casefold().endswith(clitic) for clitic in _POSSESSIVE_CLITICS):
        possessor = first[: -len("'s")].strip()
        return MentionAnalysis(mention="possessive", head_surface=rest, owner_hint="explicit", owner_surface=possessor)
    definite = first_token.morph.get("Definite") if first_token.pos_ == "DET" else []
    if definite == ["Def"]:
        return MentionAnalysis(mention="definite", head_surface=text, owner_hint="", owner_surface="")
    if definite == ["Ind"]:
        return MentionAnalysis(mention="indefinite", head_surface=text, owner_hint="", owner_surface="")
    return MentionAnalysis(mention="bare", head_surface=text, owner_hint="", owner_surface="")
