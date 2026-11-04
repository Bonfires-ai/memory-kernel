"""Pure, bounded decision-contract compilation for mined constructions.

The contract is advisory data, not an identity or graph-authority mechanism.  It
may ask a bounded question about a construction's declared roles and reuse only
that construction's already validated effect primitives.

DRAFT / NOT ACTIVATION-READY: this module validates caller-supplied, already
authorized source evidence structurally; it neither reads a database nor
performs authorization. It intentionally has no provider, persistence, or
activation call site.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class DecisionContractError(ValueError):
    """A proposal does not meet the bounded contract."""


@dataclass(frozen=True)
class SourceScope:
    bonfire_id: str
    profile: str
    conversation_id: str


def source_scope_from_dict(value: Mapping[str, Any]) -> SourceScope:
    required = {"bonfire_id", "profile", "conversation_id"}
    if set(value) != required or any(not isinstance(value[key], str) or not value[key] for key in required):
        raise DecisionContractError("source scope requires nonempty bonfire, profile, and conversation")
    return SourceScope(value["bonfire_id"], value["profile"], value["conversation_id"])


__all__ = ["DecisionContractError", "SourceScope", "source_scope_from_dict"]
