"""The confirmation policy (G9): whether a tool step waits for the game's confirmation before it is resolved.

A tool that declares no side effect never confirms. A tool whose consent is ``speaker_turn`` is confirmed by the
speaker's own play turn: the step's actor is the speaker. Every other tool confirms, including one whose effect is
unknown."""

from __future__ import annotations

from dataclasses import dataclass

from memory_kernel.runtime.program import OpSpec, ToolSpec


@dataclass(frozen=True)
class ConfirmationPolicy:
    def needs_confirmation(self, tool: ToolSpec | OpSpec, *, actor: str | None, speaker: str | None) -> bool:
        if tool.side_effect is False:
            return False
        if tool.side_effect is None:  # an unknown effect is true (decision 15, R6): always confirms
            return True
        return not (tool.consent == "speaker_turn" and bool(actor) and actor == speaker)


DEFAULT_POLICY = ConfirmationPolicy()
