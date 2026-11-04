"""Pure authoring of a bounded native discourse proof layer; never auto-installed.

Locks follow dqr001-native-analysis-proof-v1. An explicit caller-supplied
interpretation selects predicate and roles; its review_id is provenance, not
authenticated approval. These forms have not established general WH/NP coverage.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Literal

NATIVE_DISCOURSE_LAYER_VERSION = "dqr-native-discourse-v8"
NATIVE_DISCOURSE_EXTENDED_LAYER_VERSION = "dqr-native-discourse-v9"
NATIVE_DISCOURSE_OBJECT_WH_LAYER_VERSION = "dqr-native-discourse-v10"


@dataclass(frozen=True)
class NativeQuestionSourceContract:
    """Fixed reviewed syntax authority for the additive v9 question shapes."""

    version: Literal["dqr-native-question-source-v1"] = "dqr-native-question-source-v1"
    conditional_review_id: str = "native-conditional-question-source-v1"
    conditional_marker_lemma: Literal["if"] = "if"
    conditional_modal_lemma: Literal["would"] = "would"
    polar_review_id: str = "native-polar-question-source-v1"
    polar_auxiliary_lemma: Literal["do"] = "do"
    requested_dimension: Literal["polarity"] = "polarity"


NATIVE_QUESTION_SOURCE_CONTRACT = NativeQuestionSourceContract()


def native_discourse_question_source_contract(
    discourse_version: str,
) -> NativeQuestionSourceContract | None:
    """Return the fixed additive source contract selected by an exact version."""
    if type(discourse_version) is not str:
        raise ValueError("native discourse requires an exact supported version")
    if discourse_version == NATIVE_DISCOURSE_LAYER_VERSION:
        return None
    if discourse_version == NATIVE_DISCOURSE_OBJECT_WH_LAYER_VERSION:
        return None
    if discourse_version == NATIVE_DISCOURSE_EXTENDED_LAYER_VERSION:
        return NATIVE_QUESTION_SOURCE_CONTRACT
    raise ValueError("unsupported native discourse version")


@dataclass(frozen=True)
class NativeDiscourseContract:
    """Explicit interpretation for one predicate, independent of word order.

    The caller must establish review authority and pin the served interpretation
    separately. This declaration does not infer a roleset from lexical_category.
    """

    review_id: str
    roleset: str
    lexical_category: str
    subject_role: str
    object_role: str
    subject_binding_mode: str = "local-source"
    object_binding_mode: str = "local-source"


@dataclass(frozen=True)
class NativeObjectWhShapeContract:
    """Reviewed opt-in WHAT object question for exactly one member roleset."""

    review_id: str
    roleset: str
    orientation: Literal["object"] = "object"
    wh_lemma: Literal["what"] = "what"

    def __post_init__(self) -> None:
        if (
            type(self.review_id) is not str
            or not self.review_id
            or self.review_id != self.review_id.strip()
            or len(self.review_id) > 4096
            or type(self.roleset) is not str
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*\.[0-9]+", self.roleset) is None
            or self.orientation != "object"
            or self.wh_lemma != "what"
        ):
            raise ValueError("WHAT object question requires exact reviewed roleset and shape")


def _validate_contract(contract: NativeDiscourseContract) -> None:
    if not isinstance(contract, NativeDiscourseContract):
        raise TypeError("native discourse requires an explicit NativeDiscourseContract")
    for key, value in asdict(contract).items():
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"native discourse contract requires nonempty exact {key}")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*\.[0-9]+", contract.roleset):
        raise ValueError("native discourse contract requires an explicit numbered roleset")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_().+-]*", contract.lexical_category):
        raise ValueError("native discourse lexical category must be an explicit native category")
    if any(not re.fullmatch(r"ARG[0-9]+", role) for role in (contract.subject_role, contract.object_role)):
        raise ValueError("native discourse contract requires explicit numbered ARG roles")
    if contract.subject_role == contract.object_role:
        raise ValueError("native discourse subject and object roles must differ")
    if any(
        mode not in {"local-source", "required-nominal-reference"}
        for mode in (contract.subject_binding_mode, contract.object_binding_mode)
    ):
        raise ValueError("native discourse requires an explicit supported reference binding mode")


@dataclass(frozen=True)
class NativeDiscourseCxnDescriptor:
    """Explicit role orientation and subject authority for one authored shape."""

    name: str
    orientation: Literal["subject", "object", "topic", "conditional-subject", "polar"]
    subject_authority: Literal["dependency", "constituent"]
    auxiliary_authority: Literal["dependency", "constituent"] = "dependency"
    bound_shape: Literal["phrase-np", "unary-pronoun-object", "unary-nominal-subject"] = "phrase-np"
    wh_lemma: Literal["who", "what"] = "who"


def native_discourse_cxn_descriptors(
    contract: NativeDiscourseContract,
    *,
    discourse_version: str = NATIVE_DISCOURSE_LAYER_VERSION,
    object_wh_shape: NativeObjectWhShapeContract | None = None,
) -> tuple[NativeDiscourseCxnDescriptor, ...]:
    """Content-bound names prevent a changed interpretation overwriting an old one."""
    _validate_contract(contract)
    source_contract = native_discourse_question_source_contract(discourse_version)
    payload = json.dumps([NATIVE_DISCOURSE_LAYER_VERSION, asdict(contract)], sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
    descriptors = tuple(
        NativeDiscourseCxnDescriptor(f"dqr-{digest}-{kind}-cxn", orientation, authority, auxiliary)
        for kind, orientation, authority, auxiliary in (
            ("subject-wh", "subject", "dependency", "dependency"),
            ("object-wh", "object", "dependency", "dependency"),
            ("topic-question", "topic", "dependency", "dependency"),
            ("object-wh-constituent-subject", "object", "constituent", "dependency"),
            ("object-wh-do-root-constituent", "object", "constituent", "constituent"),
        )
    )
    if contract.object_binding_mode == "required-nominal-reference":
        descriptors += (
            NativeDiscourseCxnDescriptor(
                f"dqr-{digest}-subject-wh-pronominal-object-cxn",
                "subject",
                "dependency",
                bound_shape="unary-pronoun-object",
            ),
        )
    if source_contract is not None:
        extension_payload = json.dumps(
            [discourse_version, asdict(contract), asdict(source_contract)],
            sort_keys=True,
            separators=(",", ":"),
        )
        extension_digest = hashlib.sha256(extension_payload.encode()).hexdigest()[:20]
        descriptors += (
            NativeDiscourseCxnDescriptor(
                f"dqr-{extension_digest}-conditional-subject-wh-cxn",
                "conditional-subject",
                "constituent",
                "constituent",
            ),
            NativeDiscourseCxnDescriptor(
                f"dqr-{extension_digest}-polar-question-cxn",
                "polar",
                "constituent",
                "constituent",
            ),
        )
    if object_wh_shape is not None:
        if (
            discourse_version != NATIVE_DISCOURSE_OBJECT_WH_LAYER_VERSION
            or type(object_wh_shape) is not NativeObjectWhShapeContract
            or object_wh_shape.roleset != contract.roleset
        ):
            raise ValueError("WHAT object shape must match its versioned discourse member")
        shape_payload = json.dumps(
            [discourse_version, asdict(contract), asdict(object_wh_shape)], sort_keys=True, separators=(",", ":")
        )
        shape_digest = hashlib.sha256(shape_payload.encode()).hexdigest()[:20]
        descriptors += (
            NativeDiscourseCxnDescriptor(
                f"dqr-{shape_digest}-object-what-cxn", "object", "dependency", wh_lemma="what"
            ),
            NativeDiscourseCxnDescriptor(
                f"dqr-{shape_digest}-object-what-unary-np-cxn",
                "object",
                "dependency",
                bound_shape="unary-nominal-subject",
                wh_lemma="what",
            ),
        )
    return descriptors


def native_discourse_cxn_names(
    contract: NativeDiscourseContract,
    *,
    discourse_version: str = NATIVE_DISCOURSE_LAYER_VERSION,
    object_wh_shape: NativeObjectWhShapeContract | None = None,
) -> tuple[str, ...]:
    return tuple(
        descriptor.name
        for descriptor in native_discourse_cxn_descriptors(
            contract, discourse_version=discourse_version, object_wh_shape=object_wh_shape
        )
    )
