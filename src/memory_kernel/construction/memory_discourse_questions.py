"""Finite discourse hypotheses over exact syntax and receipt-bound Jev evidence.

Facets are model hypotheses, never operator paths or semantic effects. Native
tasks retain the parent source and do not change construction priorities.
"""

from __future__ import annotations

from memory_kernel.construction.memory_discourse_contract import (
    DiscourseContractError,
)


class DiscourseQuestionError(DiscourseContractError):
    """A decision differs from the finite, exact-source discourse request."""


class DiscourseQuestionBudgetExhausted(DiscourseQuestionError):
    """Exact question evidence cannot fit the fixed provider request budget."""
