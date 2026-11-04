"""Function-tag role edge names for the G1 projection.

A role edge is named by the PropBank function tag of the argument it carries, read from the frame
file's ``f=`` attribute (``arg0`` of ``go.02`` -> ``PAG``). ARGM labels take their own suffix
(``argm-tmp`` -> ``TMP``). Anything without a usable tag degrades to ``ARG``; nothing is rejected.
The tag set is published by PropBank, so no hand-written vocabulary lives here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

from memory_kernel.graph.g1_names import AGENT_ROLE, CAUSE_TAG, EXPERIENCER_THEME, STIMULUS_THEME
from memory_kernel.knowledge.propbank_framesets import (
    PropBankRolesetDefinition,
    load_propbank_roleset_index,
)

UNMAPPED_TAG = "ARG"

_CONTINUATION_OR_REFERENCE = re.compile(r"^[cr](arg\d+)$")
_NUMBERED = re.compile(r"^arg\d+$")
_TAG = re.compile(r"^[A-Z]+$")


def _base_numbered_role(role: str) -> str | None:
    """``arg1`` -> ``arg1``; ``carg1``/``rarg1`` -> ``arg1``; anything else -> None."""

    match = _CONTINUATION_OR_REFERENCE.match(role)
    if match:
        return match.group(1)
    return role if _NUMBERED.match(role) else None


def _frame_tag(roleset: PropBankRolesetDefinition | None, base_role: str) -> str:
    if roleset is None:
        return ""
    for definition in roleset.roles:
        if definition.role == base_role:
            tag = definition.function_tag.strip().upper()
            return tag if _TAG.match(tag) else ""
    return ""


def function_tag(
    roleset: str,
    role: str,
    *,
    index: Mapping[str, PropBankRolesetDefinition] | None = None,
) -> str:
    """Role edge name for ``role`` of ``roleset``: a PropBank function tag, else ``ARG``.

    ``index`` is an already-loaded roleset index; when omitted the cached index of the configured
    PropBank frames checkout is used (missing checkout raises, it never silently degrades).
    """

    key = role.strip().lower()
    if key.startswith("argm-"):
        suffix = key.removeprefix("argm-").strip()
        return suffix.upper() if _TAG.match(suffix.upper()) else UNMAPPED_TAG
    base = _base_numbered_role(key)
    if base is None:
        return UNMAPPED_TAG
    frames = index if index is not None else load_propbank_roleset_index()
    return _frame_tag(frames.get(roleset), base) or UNMAPPED_TAG


#: Psych-holder switch: when True, the psych-verb alternation
#: (arg0 VerbNet ``stimulus`` + another core role ``experiencer``) moves the holder to the experiencer; when False,
#: the earlier rule holds everywhere (arg0 unless CAU). Folded into ``g1_constructions.language_version``.
#: Switching it OFF is a procedure, not a one-line edit:
#:   1. set this to False (the graph stays coherent: A == B, keys unchanged, a distinct language_version);
#:   2. the reference harness's invariants gate keys its expected counts on this switch (the False entry is all
#:      zeros): re-run the gate;
#:   3. update every test that asserts ``psych_holder=True`` for ``LANGUAGE_VERSION`` to the new setting;
#:   4. the frozen baseline sha includes the Saga ``language_version``: write a new write-once baseline and gate
#:      later units against it.
R7C_PSYCH_HOLDER: bool = True


def holder_role(themes: Mapping[str, Sequence[str]], tag: Callable[[str], str]) -> str | None:
    """The role whose filler holds a state, from frame data only (``verbnet_role_themes``).

    The psych-verb alternation (stimulus subject, experiencer object: amaze.01, awe.01, please.01): when ``arg0``'s
    VerbNet theme is ``stimulus`` and another core role's is ``experiencer``, the lowest-numbered such role holds.
    In every other case the rule is ``arg0`` unless its PropBank function tag is ``CAU`` (cause/stimulus), else None
    (a bare experiencer theme elsewhere, e.g. show.01 arg2, does not move the holder).
    With ``R7C_PSYCH_HOLDER`` False the alternation is off and only the arg0-unless-CAU rule applies.
    ``tag(role)`` is the role's function tag (``g1_roles.function_tag``)."""

    if R7C_PSYCH_HOLDER and STIMULUS_THEME in themes.get(AGENT_ROLE, ()):
        experiencers = sorted(
            (r for r, ts in themes.items() if r != AGENT_ROLE and _NUMBERED.match(r) and EXPERIENCER_THEME in ts),
            key=lambda r: int(r[3:]),
        )
        if experiencers:
            return experiencers[0]
    return None if tag(AGENT_ROLE) == CAUSE_TAG else AGENT_ROLE


__all__ = ["UNMAPPED_TAG", "function_tag", "holder_role"]
