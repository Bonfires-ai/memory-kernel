"""Typed construction artifact envelopes owned by memory_kernel."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# `value` is an already-extracted `.get()` result (e.g.
# `artifact.get("confidence")`), so `None` is the ordinary "field not present"
# shape, not malformed data. A genuinely
# malformed *present* value still raises loud. `primitives.as_float` IS that
# body; the alias keeps this module's two call sites unchanged, `default=`
# keyword included (`as_float`'s `default` is positional-OR-keyword).
from memory_kernel.primitives import as_float as _float

ARTIFACT_FAMILY_BY_LEGACY_KIND = {
    "role_bound_event_frame": "RoleBoundEventFrame",
    "event_mention": "EventProjection",
    "event_projection": "EventProjection",
    "event_state": "ObservedEvent",
    "aggregate_set": "AggregateSet",
    "construction_recipe": "AggregateSet",
    "temporal_edge": "TemporalRoleBoundEvent",
    "role_edge": "RoleBoundEventFrame",
    "discourse_relation": "DiscourseRelation",
    "dialogue_act": "DialogueAct",
    "mention_cluster": "MentionCluster",
    "assertion_view": "AssertionView",
}

GRAPH_VISIBLE_ARTIFACT_FAMILIES = frozenset(
    {
        "AggregateSet",
        "RoleBoundEventFrame",
        "TemporalRoleBoundEvent",
        "EventProjection",
        "ObservedEvent",
        "DiscourseRelation",
        "DialogueAct",
        "MentionCluster",
        "AssertionView",
    }
)

SHARED_RECIPE_KINDS = frozenset(
    {
        "ATTACHMENT_PERDURANT",
        "CARDINALITY_QUERY",
        "ROLE_BOUND_EVENT_FRAME",
        "EVENT_PROJECTION",
        "COUNTABLE_EVENT_PROJECTION",
        "VISUAL_OBJECT_INVENTORY",
        "TEMPORAL_ROLE_BOUND_EVENT",
        "SET_INTERSECTION",
        "POSSESSIVE_RELATION",
        "OBJECT_TRANSFER_POSSESSION",
        "DISCOURSE_RELATION",
        "DIALOGUE_ACT",
        "MENTION_CLUSTER",
    }
)


@dataclass(frozen=True)
class TypedArtifact:
    """Dictionary-compatible artifact envelope stored inside existing surfaces."""

    kind: str
    signature: str
    statement_ids: tuple[str, ...] = tuple()
    episode_ids: tuple[str, ...] = tuple()
    source_message_ids: tuple[str, ...] = tuple()
    source_fact_ids: tuple[str, ...] = tuple()
    topic_id: str | None = None
    confidence: float = 1.0
    trace_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        artifact: dict[str, Any],
        *,
        episode_ids: tuple[str, ...] = tuple(),
        source_message_ids: tuple[str, ...] = tuple(),
        source_fact_ids: tuple[str, ...] = tuple(),
        topic_id: str | None = None,
    ) -> TypedArtifact:
        """Read a legacy artifact dict into the envelope, filling every field the dict left open.

        The legacy ``kind`` is mapped to an artifact family, a missing ``signature`` is derived structurally, and a
        missing ``trace_id`` is minted from ``(family, signature, statement_ids)`` — so an artifact that carries no
        explicit identity still gets one that is reproducible from its own content. The id-bearing tuples are
        de-duplicated and merged with the keyword-supplied ids, first occurrence winning. Every key that is not an
        envelope field survives verbatim in ``payload``, which additionally gains ``legacy_kind`` and
        ``artifact_family`` unless the input already set them.
        """
        legacy_kind = str(artifact.get("legacy_kind") or artifact.get("kind") or "").strip()
        family = str(
            artifact.get("artifact_family") or ARTIFACT_FAMILY_BY_LEGACY_KIND.get(legacy_kind, legacy_kind) or ""
        ).strip()
        signature = str(artifact.get("signature") or _artifact_signature(artifact)).strip()
        statement_ids = _clean_ids(artifact.get("statement_ids"))
        merged_episode_ids = _clean_ids([*list(artifact.get("episode_ids") or []), *episode_ids])
        merged_message_ids = _clean_ids([*list(artifact.get("source_message_ids") or []), *source_message_ids])
        merged_fact_ids = _clean_ids([*list(artifact.get("source_fact_ids") or []), *source_fact_ids])
        trace_id = str(artifact.get("trace_id") or _stable_id(family or legacy_kind, signature, *statement_ids)).strip()
        payload = {
            key: value
            for key, value in artifact.items()
            if key
            not in {
                "kind",
                "signature",
                "statement_ids",
                "episode_ids",
                "source_message_ids",
                "source_fact_ids",
                "topic_id",
                "confidence",
                "trace_id",
            }
        }
        payload.setdefault("legacy_kind", legacy_kind)
        payload.setdefault("artifact_family", family)
        return cls(
            kind=legacy_kind or family,
            signature=signature,
            statement_ids=tuple(statement_ids),
            episode_ids=tuple(merged_episode_ids),
            source_message_ids=tuple(merged_message_ids),
            source_fact_ids=tuple(merged_fact_ids),
            topic_id=topic_id if topic_id is not None else artifact.get("topic_id"),
            confidence=_float(artifact.get("confidence"), default=1.0),
            trace_id=trace_id,
            payload=payload,
        )

    def as_dict(self) -> dict[str, Any]:
        """Flatten back to the legacy dict shape: envelope fields first, then ``payload`` merged over them.

        A payload key that shadows an envelope key wins, so round-tripping is lossless only for artifacts whose
        payload does not collide with the envelope's own names. An empty ``trace_id`` is minted here rather than
        emitted blank.
        """
        data = {
            "kind": self.kind,
            "signature": self.signature,
            "statement_ids": list(self.statement_ids),
            "episode_ids": list(self.episode_ids),
            "source_message_ids": list(self.source_message_ids),
            "source_fact_ids": list(self.source_fact_ids),
            "topic_id": self.topic_id,
            "confidence": self.confidence,
            "trace_id": self.trace_id or _stable_id(self.kind, self.signature, *self.statement_ids),
        }
        data.update(self.payload)
        return data


def _artifact_signature(artifact: dict[str, Any]) -> str:
    """Derive a structural signature for an artifact that carries none: ``kind|trigger|role|target``, empty parts
    dropped.

    Deriving it from a fixed field order is what makes two artifacts describing the same structure collide, and what
    makes the derived signature reproducible across runs. Returns ``""`` when every part is absent.
    """

    parts = [
        str(artifact.get("kind") or ""),
        str(artifact.get("trigger") or artifact.get("frame_type") or ""),
        str(artifact.get("role") or artifact.get("subject_role") or ""),
        str(artifact.get("target") or artifact.get("answer_role") or ""),
    ]
    return "|".join(part for part in parts if part)


def _clean_text(value: Any) -> str:
    """Stringify and collapse all whitespace runs to single spaces; ``None`` and other falsy values become ``""``."""

    return " ".join(str(value or "").strip().split())


def _clean_values(value: Any) -> list[str]:
    """Cleaned strings from a list, or from a single value wrapped as one; blanks dropped, duplicates kept.

    Only ``list`` is treated as a sequence here — a tuple or set is wrapped whole.
    """

    raw = value if isinstance(value, list) else [value]
    return [_clean_text(item) for item in raw if _clean_text(item)]


def _clean_ids(raw: Any) -> list[str]:
    """Stripped, de-duplicated ids from any sequence or single value, first occurrence winning.

    Order-preserving by contract: envelope id tuples are built by concatenating sources and passing them through
    here, so the earlier source's ordering survives. Ids are only stripped, never case-folded — two spellings that
    differ in case remain two ids.
    """

    if raw is None:
        return []
    values = raw if isinstance(raw, list | tuple | set | frozenset) else [raw]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _stable_id(*parts: str) -> str:
    """A 16-hex-character trace id over ``|``-joined parts (sha1, truncated).

    This is artifact *trace* identity, not the package's canonical content address: it is deliberately a different
    function from :func:`memory_kernel.primitives.stable_id`, which is sha256 rendered UUID-shaped. Ids from the two
    are not interchangeable and must never be compared or substituted for one another.

    Its own contract is the one that binds callers: the argument list of every call site is frozen. These ids are
    persisted trace identity, so reordering parts, adding one, or changing how a part is normalized silently re-keys
    every artifact already written. ``None`` parts hash as the empty string, which means a trailing absent part and
    a trailing empty one collide.
    """

    return hashlib.sha1("|".join(str(part or "") for part in parts).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "ARTIFACT_FAMILY_BY_LEGACY_KIND",
    "GRAPH_VISIBLE_ARTIFACT_FAMILIES",
    "SHARED_RECIPE_KINDS",
    "TypedArtifact",
]
