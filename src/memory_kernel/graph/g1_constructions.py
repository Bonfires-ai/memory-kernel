"""Language constructions as data: how each eventuality class is observed.

Each eventuality label maps to one construction: its PROV-O type and the projection route it takes.
Every observation row carries the construction's ``plan`` so it can be traced to the rule that made it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

from memory_kernel.graph import g1_roles
from memory_kernel.graph.g1_names import (
    AGENT_ROLE,
    CAUSE_TAG,
    EVALUATION,
    EXPERIENCER_THEME,
    LOCATION_TAG,
    OCCURRENCE,
    PLACE,
    PREDICATIVE_TAG,
    SPEECH_ACT,
    STATE,
    STIMULUS_THEME,
    TIME_TAG,
)


def _sha8(*parts: str) -> str:
    return sha256("\x1f".join(parts).encode()).hexdigest()[:8]


@dataclass(frozen=True)
class Construction:
    eventuality: str  # g1_names label
    prov_type: str  # "prov:Activity" | "prov:Communication" | "prov:specializationOf" | "attributed"
    route: str  # "event" | "speech" | "fact_or_summary" | "holder_or_envelope"

    @property
    def plan(self) -> str:
        return f"lang:{self.eventuality}@{_sha8(self.eventuality, self.prov_type, self.route)}"


LANGUAGE_CONSTRUCTIONS: Mapping[str, Construction] = {
    c.eventuality: c
    for c in (
        Construction(OCCURRENCE, "prov:Activity", "event"),
        Construction(SPEECH_ACT, "prov:Communication", "speech"),
        Construction(STATE, "prov:specializationOf", "fact_or_summary"),
        Construction(EVALUATION, "attributed", "holder_or_envelope"),
    )
}


def construction_for(label: str) -> Construction:
    return LANGUAGE_CONSTRUCTIONS[label]


INTRODUCES, PREDICATIVE, NONE = "introduces", "predicative", "none"


@dataclass(frozen=True)
class RoleConstruction:
    """How a filled PropBank role is observed: does its filler introduce a referent, and which entity
    kind does it confer. Keyed by the role's PropBank function tag (``g1_roles.function_tag``)."""

    tag: str
    referent: str  # INTRODUCES | PREDICATIVE (a bare NP ascribes, introduces nothing) | NONE (never an entity)
    kind: str | None = None

    @property
    def plan(self) -> str:
        return f"lang:role:{self.tag}@{_sha8(self.tag, self.referent, self.kind or '')}"


#: Role constructions keyed by PropBank function tag (published inventory): a predicative (PRD) filler
#: introduces a referent only when it is not a bare/indefinite noun phrase; a time-role (TMP) filler never does
#: (g1_time owns it); a location (LOC) filler introduces a referent of kind Place. Every other tag introduces a referent (``DEFAULT_ROLE``).
ROLE_CONSTRUCTIONS: Mapping[str, RoleConstruction] = {
    r.tag: r
    for r in (
        RoleConstruction(PREDICATIVE_TAG, PREDICATIVE),
        RoleConstruction(TIME_TAG, NONE),
        RoleConstruction(LOCATION_TAG, INTRODUCES, PLACE),
    )
}
DEFAULT_ROLE = RoleConstruction("", INTRODUCES)


def role_construction(tag: str | None) -> RoleConstruction:
    return ROLE_CONSTRUCTIONS.get(tag or "", DEFAULT_ROLE)


#: The VerbNet thematic role whose filler holds a state, read per roleset from the frame rolelinks, when arg0
#: is the ``STIMULUS_THEME`` (the psych-verb alternation).
HOLDER_THEME = EXPERIENCER_THEME


def language_version(
    table: Mapping[str, Construction],
    *,
    roles: Mapping[str, RoleConstruction] = ROLE_CONSTRUCTIONS,
    holder_theme: str = HOLDER_THEME,
    psych_holder: bool | None = None,
) -> str:
    """Hash of the language layer. ``psych_holder`` defaults to the live ``g1_roles.R7C_PSYCH_HOLDER`` switch, so
    the psych-holder on and off settings carry distinct versions (off = the earlier holder rule, arg0 unless CAU)."""
    on = g1_roles.R7C_PSYCH_HOLDER if psych_holder is None else psych_holder
    holder = f"holder={holder_theme}|subject={STIMULUS_THEME}" if on else f"holder={AGENT_ROLE}-unless-{CAUSE_TAG}"
    return _sha8(*sorted(c.plan for c in table.values()), *sorted(r.plan for r in roles.values()), holder)


"""The default (psych-holder on) version. Writers call ``language_version(LANGUAGE_CONSTRUCTIONS)`` to follow the switch."""
