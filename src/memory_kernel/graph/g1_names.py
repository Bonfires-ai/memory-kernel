"""PropBank tag and eventuality names used by the G1 projection and renderer.

These are published/mined inventory labels (PropBank role and function tags,
eventuality classes), gathered in one place; not a new vocabulary.
"""

from __future__ import annotations

import re

TIME_ROLE = "argm-tmp"
AGENT_ROLE = "arg0"
CAUSE_TAG = "CAU"
#: PropBank function tag of a predicative argument.
PREDICATIVE_TAG = "PRD"
#: PropBank function tag of a time-role argument (ARGM-TMP).
TIME_TAG = "TMP"
#: PropBank function tag of a location argument (numbered LOC args and ARGM-LOC alike).
LOCATION_TAG = "LOC"
#: PropBank function tag of a proto-patient argument: the bearer a predicative (PRD) argument is said of.
PATIENT_TAG = "PPT"
#: Entity kind a LOC-tagged filler confers.
PLACE = "Place"
#: Entity kinds a referent's evidence can decide (``referent_policy`` structure), next to ``PLACE``.
PERSON = "Person"
ORGANISATION = "Organisation"
ARTIFACT = "Artifact"
KIND_LABELS = (PERSON, ORGANISATION, ARTIFACT, PLACE)
#: WordNet lexicographer file (a noun's class) -> the entity kind it gives (``g1_kinds``).
KIND_OF_LEXNAME = {
    "noun.person": PERSON,
    "noun.group": ORGANISATION,
    "noun.artifact": ARTIFACT,
    "noun.communication": ARTIFACT,
    "noun.location": PLACE,
}
#: WordNet synset a ``noun.group`` noun's first sense must be (or lie below) to give ``ORGANISATION`` ("team",
#: "company"); another group noun ("meeting", "audience") gives no kind.
ORGANISATION_SYNSET = "organization.n.01"
EXPERIENCER_THEME = "experiencer"
STIMULUS_THEME = "stimulus"


def is_core_role(role: str) -> bool:
    """A numbered PropBank argument (``arg0`` .. ``arg9``), not a modifier (``argm-*``)."""
    return re.fullmatch(r"arg\d", role) is not None


OCCURRENCE = "occurrence"
STATE = "state"
EVALUATION = "evaluation"
SPEECH_ACT = "speech_act"
EVENTUALITIES = (OCCURRENCE, STATE, SPEECH_ACT, EVALUATION)
PRIOR = "prior"
CONTEXT = "context"
