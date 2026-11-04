"""The model-family rule shared by the Jev archive reader, the ledger and the gap sender."""

from __future__ import annotations


def model_family_ok(requested: str, answered: str) -> bool:
    """The answered model is the requested alias itself or a dated snapshot of it (``<alias>-<digits>``)."""
    if answered == requested:
        return True
    suffix = answered.removeprefix(requested + "-") if answered.startswith(requested + "-") else ""
    return suffix.isascii() and suffix.isdigit()
