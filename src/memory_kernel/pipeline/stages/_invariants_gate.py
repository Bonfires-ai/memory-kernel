"""Invariant checks on the rows of a written projection.

Ported from a reference-harness module (unreached top-level definitions pruned).

``load_rows`` reads a projection's ``rows.jsonl`` into nodes and edges. ``run_invariants`` runs the seven
``g1_invariants`` row checks on them and, given a reference projection, the episodic-degree budget against it;
without one the budget is recorded as skipped (``NO_REFERENCE``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: ``budget["skipped"]`` when no ``--invariants-reference`` was given.
NO_REFERENCE = "no invariants reference"


def load_rows(path: Path) -> dict[str, Any]:
    nodes, edges = [], []
    for line in (path / "rows.jsonl").read_text().splitlines():
        row = json.loads(line)
        (nodes if "labels_neo4j" in row else edges).append(row)
    return {"nodes": nodes, "edges": edges}


def run_invariants(rows: dict[str, Any], pre_c1: dict[str, Any] | None) -> dict[str, Any]:
    """Every row check on ``rows`` (the B projection) with the budget baseline from ``pre_c1`` (the spec's reference).

    The total is measured on observation edges only; the coding edges are reported beside it. With no
    reference (``pre_c1`` None) the seven row checks still run and the budget is skipped, recorded as such."""
    from memory_kernel.graph import g1_invariants as inv

    checks = (
        inv.derivation_acyclic,
        inv.observations_card_free,
        inv.plans_resolve,
        inv.codings_traceable,
        inv.codings_anchored,
        inv.observations_traceable,
        inv.folds_current,
    )
    if pre_c1 is None:
        found = {c.__name__: c(rows) for c in checks}
        return {
            "violations": {k: len(v) for k, v in found.items()},
            "violation_samples": {k: v[:20] for k, v in found.items() if v},
            "budget": {"skipped": NO_REFERENCE},
        }
    mean, observed = inv.mean_episodic_degree(rows), inv.observation_edge_count(rows)
    base_mean, base_total = inv.mean_episodic_degree(pre_c1), inv.observation_edge_count(pre_c1)
    budget = inv.budget(
        rows,
        baseline_mean=base_mean,
        baseline_total_edges=base_total,
        baseline_node_uuids={n["props"]["uuid"] for n in pre_c1["nodes"]},
    )
    violations = {c.__name__: c(rows) for c in checks}
    violations["budget"] = budget
    return {
        "violations": {k: len(v) for k, v in violations.items()},
        "violation_samples": {k: v[:20] for k, v in violations.items() if v},
        "budget": {
            "mean_episodic_degree": mean,
            "baseline_mean": base_mean,
            "observation_edges": observed,
            "baseline_observation_edges": base_total,
            "coding_edges": len(rows["edges"]) - observed,
        },
    }
