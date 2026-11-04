"""Who "you" is. A conversation that carries addressing (any source with addressees) resolves second
person only through a message's own addressees; a conversation without any keeps the 1.0.0 two-party rule.

A full build decides that once for the whole conversation (``is_explicit``). A forward append decides it per message,
bonfire-wide (``is_explicit_at``): a message is explicit when some message at or before it, in any chat, carried
addressees, so a later session never changes an earlier message's "you"."""

from collections.abc import Collection, Iterable, Mapping, Sequence


def is_explicit(items: Iterable[Mapping]) -> bool:
    return any(item.get("addressees") for item in items)


def is_explicit_at(order: int, explicit_since: int | None) -> bool:
    """The message at ``order`` is explicit: the bonfire's first addressed message (``explicit_since``, bonfire order)
    is at or before it."""
    return explicit_since is not None and explicit_since <= order


def addressee_of(
    addressees: Sequence[str], speaker: str, session_speakers: Collection[str], *, explicit: bool
) -> str | None:
    if explicit:
        return addressees[0] if len(addressees) == 1 else None
    others = sorted(set(session_speakers) - {speaker})
    return others[0] if len(others) == 1 else None
