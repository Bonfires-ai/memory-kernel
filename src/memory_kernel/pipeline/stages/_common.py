"""Shared plumbing of the ported stage bodies.

Ported from the reference harness:
  * ``result`` / ``gate`` / ``cap_check`` / ``ProgrammeCap``  <- the harness's stage helpers
  * ``Gates``-free: the thresholds live next to the stage that checks them.

``cap_check`` reads the programme cap and ledger from the config, not from module state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memory_kernel.pipeline import provenance

if TYPE_CHECKING:
    from memory_kernel.pipeline.runner import Ctx


class ProgrammeCap(RuntimeError):
    """Ledger paid total + the stage's projection would cross the programme pilot cap."""


def cap_check(ctx: Ctx, projection_usd: float) -> dict:
    """Constraint (Money): ledger paid + projection <= cap, before a paid stage that the Jev Meter does not guard."""
    paid = provenance.ledger_summary(ctx.cfg.ledger)["paid_usd"]
    verdict = {"paid_usd": paid, "projection_usd": projection_usd, "cap_usd": ctx.cfg.cap_usd}
    if paid + projection_usd > ctx.cfg.cap_usd:
        raise ProgrammeCap(json.dumps(verdict, sort_keys=True))
    return verdict


def result(stage: str, plan: Any, live: Any = None, gate: dict | None = None) -> dict:
    return {"stage": stage, "plan": plan, "live": live, "gate": gate}


def gate(name: str, passed: bool, values: Mapping) -> dict:
    return {"name": name, "pass": bool(passed), "values": dict(values)}


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_bytes())
