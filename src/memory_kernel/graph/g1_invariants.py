"""Contract invariants as row checks.

Every check takes ``rows`` (``{"nodes": [{"labels_neo4j", "props"}], "edges": [{"type", "source", "target",
"props"}]}``, as projected by ``g1_projection``) and returns a list of violation strings; empty means it holds.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from memory_kernel.graph.g1_folds import ACCUMULATE, fold

Rows = Mapping[str, Any]

DAG_EDGES = frozenset({"DERIVED_FROM", "INVALIDATED_BY", "NEXT_SAGA"})
OBSERVATION_PLAN_PREFIX = "lang:"
#: Node kinds that exist once per bonfire (not per message).
BONFIRE_KINDS = frozenset({"Question", "Plan", "Purpose", "RevisionPass"})
ANCHOR_KINDS = frozenset({"event", "fact_edge", "summary_line"})
#: The coding layer (C2): these edges are reported by the gates, not counted against the edge budget.
CODING_EDGES = frozenset({"ANSWERS", "ANSWERS_VIA", "DERIVED_FROM", "INVALIDATED_BY"})


def _nodes(rows: Rows, label: str) -> list[dict[str, Any]]:
    return [n for n in rows["nodes"] if label in n["labels_neo4j"]]


def _edges(rows: Rows, etype: str) -> list[dict[str, Any]]:
    return [e for e in rows["edges"] if e["type"] == etype]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]


def derivation_acyclic(rows: Rows) -> list[str]:
    """No cycle over ``DERIVED_FROM | INVALIDATED_BY | NEXT_SAGA`` (a self-loop is a cycle)."""
    graph: dict[str, list[str]] = defaultdict(list)
    for e in rows["edges"]:
        if e["type"] in DAG_EDGES:
            graph[e["source"]].append(e["target"])
    state: dict[str, int] = {}  # 1 = on the stack, 2 = done
    out: list[str] = []
    for root in sorted(graph):
        if root in state:
            continue
        state[root] = 1
        stack: list[tuple[str, Iterable[str]]] = [(root, iter(graph[root]))]
        path = [root]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[node] = 2
                stack.pop()
                path.pop()
            elif state.get(nxt) == 1:
                cycle = [*path[path.index(nxt) :], nxt]
                out.append("derivation cycle: " + " -> ".join(cycle))
            elif nxt not in state:
                state[nxt] = 1
                stack.append((nxt, iter(graph.get(nxt, ()))))
                path.append(nxt)
    return out


def observations_card_free(rows: Rows) -> list[str]:
    """Invariant O: Event and RELATES_TO rows carry only language-layer plans, and no Event has ``card``."""
    out: list[str] = []
    for n in _nodes(rows, "Event"):
        p = n["props"]
        plan = p.get("plan")
        if plan is not None and not str(plan).startswith(OBSERVATION_PLAN_PREFIX):
            out.append(f"Event {p['uuid']} has plan {plan!r}")
        if "card" in p:
            out.append(f"Event {p['uuid']} has card {p['card']!r}")
    for e in _edges(rows, "RELATES_TO"):
        plan = e["props"].get("plan")
        if plan is not None and not str(plan).startswith(OBSERVATION_PLAN_PREFIX):
            out.append(f"RELATES_TO {e['props']['uuid']} has plan {plan!r}")
    return out


def plans_resolve(rows: Rows) -> list[str]:
    """Plans are never deleted: every ``ANSWERS.plan`` and every ``ANSWERS_VIA`` source resolves to a Plan node."""
    plans = _nodes(rows, "Plan")
    names = {n["props"].get("name") for n in plans} - {None}
    uuids = {n["props"]["uuid"] for n in plans}
    out = [
        f"ANSWERS {e['props']['uuid']} plan {e['props'].get('plan')!r} resolves to no Plan"
        for e in _edges(rows, "ANSWERS")
        if e["props"].get("plan") not in names
    ]
    out += [
        f"ANSWERS_VIA {e['props']['uuid']} source {e['source']} is not a Plan"
        for e in _edges(rows, "ANSWERS_VIA")
        if e["source"] not in uuids
    ]
    return out


def codings_traceable(rows: Rows) -> list[str]:
    """Each ANSWERS has ``plan`` and ``pass``; ``pass`` is a saga ``ingest_pass`` or a RevisionPass name; its Question
    reaches the Purpose via ``DERIVED_FROM``."""
    passes = {x for n in _nodes(rows, "Saga") for x in _as_list(n["props"].get("ingest_pass"))}
    passes |= {n["props"].get("name") for n in _nodes(rows, "RevisionPass")}
    purposes = {n["props"]["uuid"] for n in _nodes(rows, "Purpose")}
    up: dict[str, list[str]] = defaultdict(list)
    for e in _edges(rows, "DERIVED_FROM"):
        up[e["source"]].append(e["target"])

    reach: dict[str, bool] = {}

    def reaches(start: str) -> bool:
        if start in reach:
            return reach[start]
        seen, todo, found = {start}, [start], False
        while todo and not found:
            for nxt in up.get(todo.pop(), ()):
                if nxt in purposes:
                    found = True
                    break
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        reach[start] = found
        return found

    out: list[str] = []
    for e in _edges(rows, "ANSWERS"):
        p, uid = e["props"], e["props"]["uuid"]
        if not p.get("plan"):
            out.append(f"ANSWERS {uid} has no plan")
        if not p.get("observation") or not p.get("observation_kind"):
            out.append(f"ANSWERS {uid} has no observation anchor")
        if not p.get("pass"):
            out.append(f"ANSWERS {uid} has no pass")
        elif p["pass"] not in passes:
            out.append(f"ANSWERS {uid} pass {p['pass']!r} is neither a saga ingest_pass nor a RevisionPass")
        if not reaches(e["target"]):
            out.append(f"ANSWERS {uid} question {e['target']} does not reach the Purpose via DERIVED_FROM")
    return out


def _lines_parallel(
    p: Mapping[str, Any], ids: str, sources: str, texts: str, identity: str, joined: str | None
) -> list[str]:
    """One Entity row's line lists against its line ids ``ids`` (``codings_anchored``): ``sources`` one per id;
    ``texts``, when present, one per id and re-joining (``"\\n"``) to ``joined``; ``identity``, when present, a
    subset of the ids in line order. A node's own lists (``joined`` given) are paired: texts and identity ids both
    present or both absent (``kg_contract.line_data``)."""
    uid, line_ids = p["uuid"], list(p.get(ids) or [])
    out = [f"Entity {uid} {sources} is not one per {ids}"] if len(p.get(sources) or []) != len(line_ids) else []
    if joined is not None and (p.get(texts) is None) != (p.get(identity) is None):
        out.append(f"Entity {uid} {texts} and {identity} are not both present or both absent")
    if p.get(texts) is not None:
        if len(p[texts]) != len(line_ids):
            out.append(f"Entity {uid} {texts} is not one per {ids}")
        elif joined is not None and "\n".join(p[texts]) != (p.get(joined) or ""):
            out.append(f"Entity {uid} {texts} does not re-join to its {joined}")
    if p.get(identity) is not None:
        held = set(p[identity])
        if [i for i in line_ids if i in held] != list(p[identity]):
            out.append(f"Entity {uid} {identity} is not a subset of {ids} in line order")
    return out


def codings_anchored(rows: Rows) -> list[str]:
    """Every ANSWERS names an existing observation of its kind (C2): an Event node, a fact edge (RELATES_TO with
    ``event_keys``), or a line of an Entity's summary whose id is in ``summary_line_ids`` with the coding's own
    Episodic as its ``summary_sources`` entry. In a forward append a stored Entity's row carries only the lines the
    writer appends to its summary (``summary_append_line_ids`` / ``summary_append_sources``): those anchor too.

    The lines themselves are well formed: an Entity's per-line lists are parallel to its line ids (sources, and the
    texts when present, one per id, the texts re-joining to ``summary``) and its identity ids are a subset of its line
    ids in line order; an appended row's lists likewise (``summary_append`` the texts)."""
    events = {n["props"]["uuid"] for n in _nodes(rows, "Event")}
    facts = {e["props"]["uuid"] for e in _edges(rows, "RELATES_TO") if e["props"].get("event_keys")}
    lines: dict[tuple[str, str], str] = {}
    out: list[str] = []
    for n in _nodes(rows, "Entity"):
        p = n["props"]
        for ids, sources, texts, identity, joined in (
            ("summary_line_ids", "summary_sources", "summary_line_texts", "summary_identity_line_ids", "summary"),
            ("summary_append_line_ids", "summary_append_sources", "summary_append", "summary_append_identity_line_ids", None),
        ):  # fmt: skip
            for lid, src in zip(p.get(ids) or [], p.get(sources) or [], strict=False):
                lines[(p["uuid"], lid)] = src
            out += _lines_parallel(p, ids, sources, texts, identity, joined)
    for e in _edges(rows, "ANSWERS"):
        p, uid = e["props"], e["props"]["uuid"]
        kind, obs = p.get("observation_kind"), p.get("observation")
        if kind not in ANCHOR_KINDS:
            out.append(f"ANSWERS {uid} observation_kind {kind!r}")
        elif kind == "event" and obs not in events:
            out.append(f"ANSWERS {uid} event {obs} is not an Event")
        elif kind == "fact_edge" and obs not in facts:
            out.append(f"ANSWERS {uid} fact edge {obs} is not a fact edge")
        elif kind == "summary_line":
            src = lines.get((str(obs), str(p.get("summary_line"))))
            if src is None:
                out.append(f"ANSWERS {uid} summary line {p.get('summary_line')!r} is not on entity {obs}")
            elif src != e["source"]:
                out.append(f"ANSWERS {uid} summary line source {src} is not the coding's Episodic {e['source']}")
    return out


def observations_traceable(rows: Rows) -> list[str]:
    """Each Episodic has exactly one incoming HAS_EPISODE, from a Saga with a non-empty ``ingest_pass``."""
    sagas = {n["props"]["uuid"]: n["props"] for n in _nodes(rows, "Saga")}
    incoming: dict[str, list[str]] = defaultdict(list)
    for e in _edges(rows, "HAS_EPISODE"):
        incoming[e["target"]].append(e["source"])
    out: list[str] = []
    for n in _nodes(rows, "Episodic"):
        uid = n["props"]["uuid"]
        sources = incoming.get(uid, [])
        if len(sources) != 1:
            out.append(f"Episodic {uid} has {len(sources)} incoming HAS_EPISODE")
        elif sources[0] not in sagas:
            out.append(f"Episodic {uid} HAS_EPISODE source {sources[0]} is not a Saga")
        elif not _as_list(sagas[sources[0]].get("ingest_pass")):
            out.append(f"Episodic {uid} saga {sources[0]} has empty ingest_pass")
    return out


def mean_episodic_degree(rows: Rows) -> float:
    """Mean number of incident edges per Episodic (an edge counts once per Episodic it touches)."""
    eps = {n["props"]["uuid"] for n in _nodes(rows, "Episodic")}
    degree: Counter[str] = Counter()
    for e in rows["edges"]:
        for end in {e["source"], e["target"]} & eps:
            degree[end] += 1
    return sum(degree[u] for u in eps) / len(eps) if eps else 0.0


def observation_edge_count(rows: Rows) -> int:
    return sum(1 for e in rows["edges"] if e["type"] not in CODING_EDGES)


def budget(
    rows: Rows,
    *,
    baseline_mean: float,
    baseline_total_edges: int,
    baseline_node_uuids: Collection[str],
) -> list[str]:
    """Readability budget: mean incident edges per Episodic <= ``baseline_mean``,
    observation edges <= ``baseline_total_edges`` (the coding layer's edges, ``CODING_EDGES``, do not
    count against the total; the gates report them), and no new per-message nodes (a node outside ``baseline_node_uuids`` that is
    not a per-bonfire Question / Plan / Purpose / RevisionPass). All three parameters are required."""
    out: list[str] = []
    mean = mean_episodic_degree(rows)
    if mean > baseline_mean + 1e-9:
        out.append(f"mean incident edges per Episodic {mean:.4f} > baseline {baseline_mean:.4f}")
    observed = observation_edge_count(rows)
    if observed > baseline_total_edges:
        out.append(f"observation edges {observed} > baseline {baseline_total_edges}")
    known = set(baseline_node_uuids)
    out += [
        f"new per-message node {n['props']['uuid']} {sorted(n['labels_neo4j'])}"
        for n in rows["nodes"]
        if n["props"]["uuid"] not in known and not BONFIRE_KINDS & set(n["labels_neo4j"])
    ]
    return out


def folds_current(rows: Rows) -> list[str]:
    """Each Question's cached fold equals ``g1_folds.fold`` over its live ANSWERS (folds read codings only)."""
    by_q: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for e in _edges(rows, "ANSWERS"):
        by_q[e["target"]].append(e["props"])
    out: list[str] = []
    for n in _nodes(rows, "Question"):
        p = n["props"]
        want = fold(p.get("fold") or ACCUMULATE, by_q.get(p["uuid"], []))
        got = {k: p.get(k) for k in want}
        if got != want:
            out.append(f"Question {p['uuid']} fold {got} != {want}")
    return out
