"""parity_gate: compare filters, one-sided/changed reporting, and vectors reduced to their dimension."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "parity_gate.py"
spec = importlib.util.spec_from_file_location("parity_gate", SCRIPT)
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)


def _node(uid, labels, **props):
    return {"labels": labels, "props": {"uuid": uid, **props}}


def _write(tmp_path, name, dump):
    p = tmp_path / name
    p.write_text(json.dumps(dump))
    return str(p)


def _base():
    return {
        "nodes": {
            "e1": _node("e1", ["Entity"], name="x", kind_person=1, run=1, other=1),
            "m1": _node("m1", ["BonfireMeta"], bonfire_id="b"),
        },
        "edges": {
            "MENTIONS:r1": {"source": "e1", "target": "m1", "props": {"uuid": "r1"}},
            "HAS:r2": {"source": "e1", "target": "e1", "props": {"uuid": "r2"}},
        },
    }


def test_compare_ignores_listed_props_and_labels_only(tmp_path, capsys):
    a, b = _base(), _base()
    b["nodes"]["e1"]["props"].update(run=2, kind_person=5)
    pa, pb = _write(tmp_path, "a.json", a), _write(tmp_path, "b.json", b)
    assert pg.main(["compare", pa, pb]) == 1
    assert pg.main(["compare", pa, pb, "--ignore-props", "run", "--ignore-prefix", "kind_"]) == 0
    # an unlisted difference still fails
    b["nodes"]["e1"]["props"]["other"] = 9
    pb = _write(tmp_path, "b.json", b)
    assert pg.main(["compare", pa, pb, "--ignore-props", "run", "--ignore-prefix", "kind_"]) == 1
    # a label-ignored node and its edges vanish from both sides
    b2 = _base()
    del b2["nodes"]["m1"], b2["edges"]["MENTIONS:r1"]
    pb2 = _write(tmp_path, "b2.json", b2)
    assert pg.main(["compare", pa, pb2]) == 1
    assert pg.main(["compare", pa, pb2, "--ignore-labels", "BonfireMeta"]) == 0
    # edges touching a dropped node go too, even if only one side has the node
    assert pg.main(["compare", pa, pa, "--ignore-labels", "Entity,BonfireMeta"]) == 0


def test_compare_reports_one_sided_and_changed(tmp_path, capsys):
    a, b = _base(), _base()
    for i in range(5):
        a["nodes"][f"a{i}"] = _node(f"a{i}", ["Entity"])
    b["nodes"]["b0"] = _node("b0", ["Entity"])
    b["nodes"]["e1"]["props"]["name"] = "y"
    assert pg.main(["compare", _write(tmp_path, "a.json", a), _write(tmp_path, "b.json", b)]) == 1
    out = capsys.readouterr().out
    assert "only_a: 5" in out and "only_b: 1" in out and "changed: 1" in out
    assert out.count("nodes:a") == 3  # three samples at most
    assert pg.main(["compare", _write(tmp_path, "a.json", a), _write(tmp_path, "a2.json", a)]) == 0


class _Session:
    def __init__(self, nodes, edges, edge_total=None):
        self.nodes, self.edges, self.queries = nodes, edges, []
        self.edge_total = len(edges) if edge_total is None else edge_total

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self.queries.append(cypher)
        assert params == {"group": "g"}
        if cypher == pg.NODES_CYPHER:
            return self.nodes
        if cypher == pg.EDGES_CYPHER:
            return self.edges
        assert cypher == pg.EDGE_COUNT_CYPHER
        return [{"n": self.edge_total}]


class _Driver:
    def __init__(self, nodes, edges, edge_total=None):
        self.s = _Session(nodes, edges, edge_total)
        self.modes = []

    def session(self, **kw):
        self.modes.append(kw.get("default_access_mode"))
        return self.s


def test_dump_strips_vectors_to_their_dimension(tmp_path):
    vec = [0.1, 0.2, 0.3]
    nodes = [
        {
            "labels": ["Episodic", "A"],
            "props": {"uuid": "n1", "message_embedding": vec, "search_key": "k", "search_key_sha": "s", "scores": vec},
        },
    ]
    edges = [
        {"type": "RELATES_TO", "props": {"uuid": "r1", "fact_embedding": vec}, "source": "n1", "target": "n1"},
    ]
    drv = _Driver(nodes, edges)
    dump = pg.dump_group(drv, "g")
    props = dump["nodes"]["n1"]["props"]
    assert props["message_embedding"] == {"dim": 3}
    assert props["scores"] == vec  # only *_embedding names are reduced
    assert props["search_key"] == "k" and props["search_key_sha"] == "s"
    assert dump["nodes"]["n1"]["labels"] == ["A", "Episodic"]
    assert dump["edges"]["RELATES_TO:r1"]["props"]["fact_embedding"] == {"dim": 3}
    assert drv.modes and all(m == "READ" for m in drv.modes)
    assert all(q.lstrip().startswith("MATCH") and "CREATE" not in q and "MERGE" not in q for q in drv.s.queries)
    out1, out2 = tmp_path / "1.json", tmp_path / "2.json"
    pg.write_dump(dump, out1)
    pg.write_dump(pg.dump_group(_Driver(list(reversed(nodes)), edges), "g"), out2)
    assert out1.read_bytes() == out2.read_bytes()


def _edge(t, uid=None, s="n1", d="n1"):
    props = {} if uid is None else {"uuid": uid}
    return {"type": t, "props": props, "source": s, "target": d}


def _node_rec(uid=None, labels=("A",)):
    return {"labels": list(labels), "props": {} if uid is None else {"uuid": uid}}


def test_edge_without_uuid_fails_loudly():
    drv = _Driver([_node_rec("n1")], [_edge("REL"), _edge("REL")])
    with pytest.raises(SystemExit, match="REL.*n1.*n1"):
        pg.dump_group(drv, "g")


def test_duplicate_edge_key_fails_loudly():
    drv = _Driver([_node_rec("n1")], [_edge("REL", "r1"), _edge("REL", "r1")])
    with pytest.raises(SystemExit, match="REL:r1"):
        pg.dump_group(drv, "g")


def test_duplicate_node_uuid_fails_loudly():
    with pytest.raises(SystemExit, match="dup1"):
        pg.dump_group(_Driver([_node_rec("dup1"), _node_rec("dup1")], []), "g")


def test_node_without_uuid_fails_loudly():
    with pytest.raises(SystemExit, match="Zed"):
        pg.dump_group(_Driver([_node_rec(None, ("Zed",))], []), "g")


def test_edge_leaving_the_group_fails_loudly():
    drv = _Driver([_node_rec("n1")], [_edge("REL", "r1")], edge_total=2)
    with pytest.raises(SystemExit, match="2.*1|1.*2"):
        pg.dump_group(drv, "g")
