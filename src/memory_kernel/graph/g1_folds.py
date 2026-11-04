"""Folds: the current answer to a Question, a pure function of its live ANSWERS codings (``filler``,
``told_at``, ``uuid``), cached on the Question with the sha of those inputs (C2). Never a node."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

ACCUMULATE, LATEST, COUNT = "accumulate", "latest", "count"
POLICIES = (ACCUMULATE, LATEST, COUNT)


def fold(policy: str, codings: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """``{"fold_answer", "fold_count", "fold_sha"}`` over the live codings (no ``invalidated_by``), oldest first by
    (``told_at``, ``uuid``); ``told_at`` is compared as a STRING, which is chronological only while every value
    shares one UTC offset (all values in the development conversation are ``+00:00``); mixed offsets would misorder ``latest``, so normalise
    to UTC before folding data that has them. accumulate: the distinct fillers in that order; latest: the newest
    coding's filler; count: no filler list (``fold_count`` is the answer). ``fold_sha`` = sha256[:16] of the policy and the inputs."""
    if policy not in POLICIES:
        raise ValueError(f"unknown fold policy {policy!r}; expected one of {POLICIES}")
    inputs = sorted(
        (
            {"uuid": str(c["uuid"]), "filler": c.get("filler"), "told_at": str(c.get("told_at") or "")}
            for c in codings
            if not c.get("invalidated_by")
        ),
        key=lambda r: (r["told_at"], r["uuid"]),
    )
    fillers = [str(r["filler"]) for r in inputs if r["filler"]]
    if policy == ACCUMULATE:
        answer = list(dict.fromkeys(fillers))
    elif policy == LATEST:
        answer = fillers[-1:]
    else:
        answer = []
    text = json.dumps({"policy": policy, "inputs": inputs}, sort_keys=True, ensure_ascii=True)
    return {
        "fold_answer": answer,
        "fold_count": len(inputs),
        "fold_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
    }
