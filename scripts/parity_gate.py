"""Parity gate: dump one Neo4j group to JSON and compare two dumps.

``dump`` reads a group's nodes and edges (read-only Cypher: plain MATCH) and writes
``{nodes: {uuid: {labels, props}}, edges: {"TYPE:uuid": {source, target, props}}}``. An embedding (a list of numbers
under a property named ``*_embedding``) is reduced to ``{"dim": n}``; ``search_key`` and ``search_key_sha`` stay as
they are. The output is deterministic (sorted keys, sorted labels), so equal graphs give byte-equal files.

``compare`` exits 0 only when two dumps are equal after the ignore filters; otherwise it prints ``only_a``,
``only_b`` and ``changed`` counts with up to 3 samples each and exits 1.

The password comes from ``--password-file`` only, never argv, and is never printed.

    parity_gate.py dump --neo4j-uri U --password-file F --group G --out OUT [--user neo4j]
    parity_gate.py compare A B [--ignore-props a,b] [--ignore-prefix kind_ ...] [--ignore-labels L1,L2]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from numbers import Number
from pathlib import Path
from typing import Any

SAMPLES = 3
EMBEDDING_SUFFIX = "_embedding"

NODES_CYPHER = "MATCH (n {group_id: $group}) RETURN labels(n) AS labels, properties(n) AS props"
EDGES_CYPHER = (
    "MATCH (a {group_id: $group})-[r]->(b {group_id: $group}) "
    "RETURN type(r) AS type, properties(r) AS props, a.uuid AS source, b.uuid AS target"
)
EDGE_COUNT_CYPHER = "MATCH ()-[r {group_id: $group}]->() RETURN count(r) AS n"


def _is_vector(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(x, Number) and not isinstance(x, bool) for x in value)
    )


def clean_props(props: Mapping[str, Any]) -> dict[str, Any]:
    """Properties as JSON-safe values; every ``*_embedding`` number list becomes ``{"dim": n}``."""
    out: dict[str, Any] = {}
    for k, v in props.items():
        if k.endswith(EMBEDDING_SUFFIX) and _is_vector(v):
            out[k] = {"dim": len(v)}
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def dump_group(driver: Any, group: str) -> dict[str, Any]:
    """The group's nodes and edges. A row is keyed deterministically or the dump fails: nothing is dropped silently."""
    nodes: dict[str, Any] = {}
    edges: dict[str, Any] = {}
    with driver.session(default_access_mode="READ") as session:
        for rec in session.run(NODES_CYPHER, group=group):
            props = clean_props(rec["props"])
            uid = props.get("uuid")
            if not uid:
                raise SystemExit(f"parity_gate: node without uuid, labels={sorted(rec['labels'])}")
            if str(uid) in nodes:
                raise SystemExit(f"parity_gate: duplicate node uuid {uid}")
            nodes[str(uid)] = {"labels": sorted(rec["labels"]), "props": props}
        for rec in session.run(EDGES_CYPHER, group=group):
            props = clean_props(rec["props"])
            if not props.get("uuid"):
                raise SystemExit(
                    f"parity_gate: edge without uuid, type={rec['type']} source={rec['source']} target={rec['target']}"
                )
            key = f"{rec['type']}:{props['uuid']}"
            if key in edges:
                raise SystemExit(f"parity_gate: duplicate edge key {key}")
            edges[key] = {"source": rec["source"], "target": rec["target"], "props": props}
        total = next(iter(session.run(EDGE_COUNT_CYPHER, group=group)))["n"]
    if total != len(edges):
        raise SystemExit(
            f"parity_gate: group has {total} edges but {len(edges)} were dumped (an endpoint is outside the group)"
        )
    return {"nodes": dict(sorted(nodes.items())), "edges": dict(sorted(edges.items()))}


def write_dump(dump: Mapping[str, Any], out: Path) -> None:
    out.write_text(json.dumps(dump, sort_keys=True, indent=1, default=str) + "\n")


def _filter_props(props: Mapping[str, Any], ignore_props: set[str], prefixes: Sequence[str]) -> dict[str, Any]:
    return {k: v for k, v in props.items() if k not in ignore_props and not any(k.startswith(p) for p in prefixes)}


def normalize(
    dump: Mapping[str, Any],
    ignore_props: Sequence[str] = (),
    ignore_prefixes: Sequence[str] = (),
    ignore_labels: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    """Dump with ignored props/prefixes removed, nodes carrying an ignored label dropped, and edges touching them."""
    drop_props, drop_labels = set(ignore_props), set(ignore_labels)
    nodes: dict[str, Any] = {}
    dropped: set[str] = set()
    for uid, n in dump["nodes"].items():
        if drop_labels & set(n["labels"]):
            dropped.add(uid)
            continue
        nodes[uid] = {"labels": sorted(n["labels"]), "props": _filter_props(n["props"], drop_props, ignore_prefixes)}
    edges = {
        key: {
            "source": e["source"],
            "target": e["target"],
            "props": _filter_props(e["props"], drop_props, ignore_prefixes),
        }
        for key, e in dump["edges"].items()
        if e["source"] not in dropped and e["target"] not in dropped
    }
    return {"nodes": nodes, "edges": edges}


def diff(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    """``only_a`` / ``only_b`` / ``changed`` ids as ``kind:id`` strings, sorted."""
    out: dict[str, list[str]] = {"only_a": [], "only_b": [], "changed": []}
    for kind in ("nodes", "edges"):
        left, right = a[kind], b[kind]
        out["only_a"] += [f"{kind}:{k}" for k in sorted(left.keys() - right.keys())]
        out["only_b"] += [f"{kind}:{k}" for k in sorted(right.keys() - left.keys())]
        out["changed"] += [f"{kind}:{k}" for k in sorted(left.keys() & right.keys()) if left[k] != right[k]]
    return out


def _split(values: Sequence[str]) -> list[str]:
    return [p for v in values for p in v.split(",") if p]


def _cmd_dump(args: argparse.Namespace) -> int:
    from neo4j import GraphDatabase

    password = Path(args.password_file).read_text().strip()
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.user, password))
    try:
        dump = dump_group(driver, args.group)
    finally:
        driver.close()
    write_dump(dump, Path(args.out))
    print(f"wrote {args.out}: {len(dump['nodes'])} nodes, {len(dump['edges'])} edges")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    kw = {
        "ignore_props": _split(args.ignore_props),
        "ignore_prefixes": _split(args.ignore_prefix),
        "ignore_labels": _split(args.ignore_labels),
    }
    a = normalize(json.loads(Path(args.a).read_text()), **kw)
    b = normalize(json.loads(Path(args.b).read_text()), **kw)
    d = diff(a, b)
    if not any(d.values()):
        print("equal")
        return 0
    for name in ("only_a", "only_b", "changed"):
        print(f"{name}: {len(d[name])}")
        for s in d[name][:SAMPLES]:
            print(f"  {s}")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="write one group's nodes and edges as JSON")
    d.add_argument("--neo4j-uri", required=True)
    d.add_argument("--user", default="neo4j")
    d.add_argument("--password-file", required=True)
    d.add_argument("--group", required=True)
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="exit 0 only when two dumps are equal")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--ignore-props", action="append", default=[], help="comma-separated property names")
    c.add_argument("--ignore-prefix", action="append", default=[], help="property-name prefix; repeatable")
    c.add_argument("--ignore-labels", action="append", default=[], help="comma-separated; drops those nodes and edges")
    args = ap.parse_args(argv)
    return {"dump": _cmd_dump, "compare": _cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
