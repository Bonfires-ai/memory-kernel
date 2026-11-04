"""Portable message contracts for memory-kernel ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# envelope fields. Read via
# Message.from_dict, which is the ONE sanctioned reader of these keys
# off historical `metadata` (back-compat for older docs).
_ENVELOPE_FIELDS = (
    "message_id",
    "speaker_fingerprint",
    "display_name",
    "reply_to_message_id",
    "channel",
)


@dataclass
class Message:
    """Portable chat-message DTO mirrored from the host backend's message boundary."""

    content: str
    speaker: str = ""
    timestamp: datetime | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    persona_root: str = ""
    subagent_instance: str = ""
    # envelope
    message_id: str | None = None
    speaker_fingerprint: str | None = None
    display_name: str | None = None
    reply_to_message_id: str | None = None
    channel: str | None = None  # "platform:chat_id[:topic_id]"

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> Message:
        """Build a message from a mapping (fresh payload or historical doc).

        ``content`` falls back from ``content`` to ``text``; ``speaker`` falls
        back from ``speaker`` to ``username``; every missing string field
        becomes ``""`` rather than ``None``, so downstream code never has to
        None-check them. ``metadata`` is copied, so the caller's mapping is
        not aliased.

        Each envelope field reads its typed top-level key FIRST, then falls
        back to ``metadata[<same name>]`` — the ONE sanctioned metadata read,
        kept for historical docs that only ever carried the envelope inside
        ``metadata``.
        """

        metadata = dict(item.get("metadata") or {})
        envelope = {
            name: item.get(name) if item.get(name) is not None else metadata.get(name) for name in _ENVELOPE_FIELDS
        }
        return cls(
            content=str(item.get("content") or item.get("text") or ""),
            speaker=str(item.get("speaker") or item.get("username") or ""),
            timestamp=item.get("timestamp"),
            metadata=metadata,
            persona_root=str(item.get("persona_root") or ""),
            subagent_instance=str(item.get("subagent_instance") or ""),
            **envelope,
        )

    def to_dict(self) -> dict[str, Any]:
        """Flatten this message to a mapping with typed top-level envelope keys.

        No envelope-in-metadata: the five envelope fields are always emitted
        as their own top-level keys, never folded into ``metadata``.
        """

        return {
            "content": self.content,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
            "persona_root": self.persona_root,
            "subagent_instance": self.subagent_instance,
            "message_id": self.message_id,
            "speaker_fingerprint": self.speaker_fingerprint,
            "display_name": self.display_name,
            "reply_to_message_id": self.reply_to_message_id,
            "channel": self.channel,
        }


__all__ = ["Message"]
