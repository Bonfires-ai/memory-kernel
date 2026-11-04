"""Exception bases that modules on both sides of an import (``graph`` and ``pipeline``) raise."""

from __future__ import annotations


class BonfireError(ValueError):
    """A bonfire refusal (a ValueError: the CLI reports it and exits 2)."""
