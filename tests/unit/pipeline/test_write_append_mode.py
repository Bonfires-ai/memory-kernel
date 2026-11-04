"""Append mode of the write stage (kernel 1.2, forward chain): ``mk append`` writes ONE session's new rows, the edges
they add onto stored nodes and its ``SAME_AS`` links into the bonfire's group, under the forward gate.

All offline. Neo4j is an in-memory group behind a fake async driver (``Neo4j``) that interprets the statements the
stage and ``kg_writer`` issue (the MERGEs, the typed lookups, the counts) and logs every one; ``kg_writer.write_rows``,
``read_rows``, ``read_group`` and ``graphdb.label_counts`` run for real against it. The projection is hand-made rows of
one forward session, built with the real uuid formulas (so ``own_uuids`` holds): chat k1's bootstrap holds m1 m2
(episode e1) and m3 (e2); the session m4 continues e2 and m5 opens e3. Invented names only.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from memory_kernel.graph.g1_ids import content_uuid, summary_line_id
from memory_kernel.graph.kg_writer import UNION_FIELDS, append_summary
from memory_kernel.pipeline import chain as chain_state
from memory_kernel.pipeline.chain import Chain, ChainError, ChatState, Minted, SameAs
from memory_kernel.pipeline.stages import write

GROUP = "g-ember"
PASS = "mk-conv-ember-001"
CREATED = "2026-10-01T00:00:00+00:00"
TS = {f"m{i}": f"2026-10-0{1 + i // 4}T09:{i:02d}:00+00:00" for i in range(1, 10)}
ORDER = {f"m{i}": i - 1 for i in range(1, 10)}


def ep(m: str) -> str:
    return content_uuid("ep", GROUP, m)


def saga(e: str) -> str:
    return content_uuid("saga", GROUP, e)


def ent(key: str) -> str:
    return content_uuid("ent", GROUP, key)


QUESTION, PURPOSE = content_uuid("q", GROUP, "q1"), content_uuid("purpose", GROUP)
OLD_LINE, NEW_LINE = "[Ana] the kiln is new", "[Ana] the kiln is hot"


# -- rows ---------------------------------------------------------------------------------------------------------------


def episodic(m: str) -> dict:
    props = {"uuid": ep(m), "group_id": GROUP, "name": m, "created_at": CREATED, "message_id": m}
    return {"labels_neo4j": ["Episodic"], "props": props | {"content": f"Ana: {m}", "valid_at": TS[m]}}


def saga_row(e: str, msgs: list[str], prior: tuple[str, ...] = ()) -> dict:
    whole = [*prior, *msgs]
    props = {
        "uuid": saga(e),
        "group_id": GROUP,
        "name": e,
        "created_at": CREATED,
        "summary": f"{TS[whole[0]]} to {TS[msgs[-1]]}; {len(whole)} messages",
        "first_episode_uuid": ep(whole[0]),
        "last_episode_uuid": ep(msgs[-1]),
        "started_at": TS[whole[0]],
        "ended_at": TS[msgs[-1]],
        "message_count": len(whole),
        "ingest_pass": [PASS],
    }
    return {"labels_neo4j": ["Saga"], "props": props}


def entity(key: str, name: str, labels: tuple[str, ...] = ("Entity",), **extra) -> dict:
    props = {"uuid": ent(key), "group_id": GROUP, "name": name, "created_at": CREATED, "summary": "", **extra}
    return {"labels_neo4j": list(labels), "props": props}


def contract(fold_count: int) -> list[dict]:
    base = {"group_id": GROUP, "created_at": CREATED, "ingest_pass": PASS}
    question = {"uuid": QUESTION, "name": "What is fired?", "qid": "q1", "fold_count": fold_count, **base}
    return [
        {"labels_neo4j": ["Purpose"], "props": {"uuid": PURPOSE, "name": "remember the kiln", **base}},
        {"labels_neo4j": ["Question"], "props": question},
    ]


def edge(kind: str, source: str, target: str, **props) -> dict:
    uuid = content_uuid("edge", GROUP, kind, source, target)
    return {"type": kind, "source": source, "target": target, "props": {"uuid": uuid, "group_id": GROUP} | props}


def line_id(line: str, source: str) -> str:
    return summary_line_id(ent("kiln"), source, line)


def bootstrap() -> dict:
    """What the bootstrap wrote: chat k1, m1 m2 (e1), m3 (e2); Ana (a speaker) and the kiln (stored as a Place)."""
    nodes = [episodic(m) for m in ("m1", "m2", "m3")] + [saga_row("e1", ["m1", "m2"]), saga_row("e2", ["m3"])]
    nodes += [
        entity("ana", "Ana", ("Entity", "Person"), name_key="ana", name_embedding=[0.0]),
        entity(
            "kiln",
            "kiln",
            ("Entity", "Place"),
            name_key="kiln",
            name_embedding=[0.0],
            summary=OLD_LINE,
            summary_sources=[ep("m1")],
            summary_line_ids=[line_id(OLD_LINE, ep("m1"))],
        ),
        *contract(1),
    ]
    edges = [
        edge("HAS_EPISODE", saga("e1"), ep("m1")),
        edge("HAS_EPISODE", saga("e1"), ep("m2")),
        edge("HAS_EPISODE", saga("e2"), ep("m3")),
        edge("NEXT_EPISODE", ep("m1"), ep("m2")),
        edge("NEXT_SAGA", saga("e1"), saga("e2")),
        edge("MENTIONS", ep("m1"), ent("ana")),
        edge("MENTIONS", ep("m1"), ent("kiln")),
        edge("MENTIONS", ep("m3"), ent("ana")),
        edge("DERIVED_FROM", QUESTION, PURPOSE),
    ]
    return {"nodes": nodes, "edges": edges}


def source(m: str, session: int) -> dict:
    return {
        "message_id": m,
        "speaker": "Ana",
        "text": f"{m} text",
        "timestamp": TS[m],
        "order": ORDER[m],
        "session_id": f"conv-ember-session_{session}",
    }


#: The chain the bootstrap committed (run 0): two entities, chat k1 with its open episode e2 = m3.
CHAIN = Chain(
    entities={
        ent("ana"): Minted(ent("ana"), "ana", "person", "Ana", ("Ana",), 0, 0),
        ent("kiln"): Minted(ent("kiln"), "kiln", "entity", "kiln", ("kiln",), 0, 0),
    },
    chats={
        "k1": ChatState(
            episodes=2,
            last_episode="e2",
            last_saga=saga("e2"),
            last_order=2,
            roster=["Ana"],
            tail=[source("m3", 1)],
            boundary={"messages": 3, "episode_start": 2, "cursor": 2},
        )
    },
    message_ids={"m1": 0, "m2": 0, "m3": 0},
    messages=3,
    episodes=2,
    sessions=1,
    run=0,
)
SAME_AS_DEE = SameAs(ent("dee"), ent("ana"), "m5", "Ana", "Dee", "name-claim", 1)


def session() -> dict:
    """The projection of the session m4 (continues e2), m5 (opens e3): Ana and the kiln reused (the kiln gets a new
    summary line), the glaze and Dee minted, Dee linked to Ana by her claim in m5."""
    nodes = [episodic("m4"), episodic("m5"), saga_row("e2", ["m4"], prior=("m3",)), saga_row("e3", ["m5"])]
    nodes += [
        entity("ana", "Ana", ("Entity", "Person")),
        entity(
            "kiln",
            "kiln",
            summary_append=[NEW_LINE],
            summary_append_sources=[ep("m4")],
            summary_append_line_ids=[line_id(NEW_LINE, ep("m4"))],
        ),
        entity("glaze", "glaze", name_key="glaze"),
        entity("dee", "Dee", ("Entity", "Person"), name_key="dee"),
        *contract(3),  # the coding aggregates move
    ]
    edges = [
        edge("HAS_EPISODE", saga("e2"), ep("m4")),
        edge("HAS_EPISODE", saga("e3"), ep("m5")),
        edge("NEXT_EPISODE", ep("m3"), ep("m4")),
        edge("NEXT_SAGA", saga("e2"), saga("e3")),
        edge("MENTIONS", ep("m4"), ent("ana")),
        edge("MENTIONS", ep("m4"), ent("kiln")),
        edge("MENTIONS", ep("m5"), ent("ana")),
        edge("MENTIONS", ep("m5"), ent("glaze")),
        edge("MENTIONS", ep("m5"), ent("dee")),
        edge("SAME_AS", ent("dee"), ent("ana"), claim_message="m5", holder="Ana", name="Dee", card="name-claim"),
        edge("DERIVED_FROM", QUESTION, PURPOSE),
    ]
    forward = {
        "minted": [
            Minted(ent("glaze"), "glaze", "entity", "glaze", ("glaze",), 1, 1),
            Minted(ent("dee"), "dee", "person", "Dee", ("Dee",), 1, 1),
        ],
        "routes": {"dee": ent("ana")},
        "same_as": [SAME_AS_DEE],
        "conflicts": [],
        "touched": sorted({ent("ana"), ent("kiln"), ent("glaze"), ent("dee")}),
    }
    external = {ep("m3"): {"labels_neo4j": ["Episodic"], "props": {"uuid": ep("m3")}}}
    return {"nodes": nodes, "edges": edges, "external_endpoints": external, "forward": forward}


# -- the fake Neo4j -----------------------------------------------------------------------------------------------------


class _Result:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    async def single(self):
        return self.rows[0] if self.rows else None

    def __aiter__(self):
        async def gen():
            for r in self.rows:
                yield r

        return gen()


class _Session:
    def __init__(self, db: Neo4j) -> None:
        self.db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def run(self, query: str, **params):
        return _Result(self.db.run(query, params))


class Neo4j:
    """One database: the group's nodes and edges by uuid. ``statements`` logs every statement it was sent. ``foreign``:
    uuid -> the other group that holds it (``kg_writer._foreign`` reads it; empty = a single-group database)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}
        self.foreign: dict[str, str] = {}
        self.statements: list[str] = []

    def session(self, **_kw) -> _Session:
        return _Session(self)

    async def close(self) -> None:
        return None

    def seed(self, rows: dict) -> None:
        for n in rows["nodes"]:
            self.nodes[n["props"]["uuid"]] = copy.deepcopy({"labels_neo4j": n["labels_neo4j"], "props": n["props"]})
        for e in rows["edges"]:
            self.edges[e["props"]["uuid"]] = copy.deepcopy(e)

    def counts(self) -> dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}

    def run(self, q: str, p: dict) -> list[dict]:  # noqa: PLR0911 (one branch per statement shape)
        self.statements.append(q)
        if q.startswith("UNWIND $rows AS r MERGE (n:"):
            return self._merge_nodes(q, p["rows"])
        if q.startswith("UNWIND $rows AS r MATCH (a:"):
            return self._merge_edges(q, p["rows"])
        if "<> $g" in q:  # kg_writer._foreign: the rows another group holds
            return [{"u": u, "g": self.foreign[u]} for u in p["ids"] if u in self.foreign]
        if m := re.match(r"MATCH \(n:(\w+) \{group_id:\$g\}\) WHERE n.uuid IN \$ids RETURN n.uuid AS u$", q):
            return [{"u": u} for u in p["ids"] if u in self.nodes and m[1] in self.nodes[u]["labels_neo4j"]]
        if "AS summary_line_ids" in q:
            fields = (
                "summary",
                "summary_sources",
                "summary_line_ids",
                "summary_line_texts",
                "summary_identity_line_ids",
            )
            hits = [self.nodes[u]["props"] for u in p["ids"] if u in self.nodes]
            return [{"u": h["uuid"], **{f: h.get(f) for f in fields}} for h in hits]
        if "name_embedding IS NOT NULL" in q:
            hits = [u for u in p["ids"] if u in self.nodes]
            return [{"u": u} for u in hits if self.nodes[u]["props"].get("name_embedding") is not None]
        if "fact_embedding IS NOT NULL" in q:
            hits = [u for u in p["ids"] if u in self.edges]
            return [{"u": u} for u in hits if self.edges[u]["props"].get("fact_embedding") is not None]
        if m := re.match(r"MATCH \(n:(\w+) \{group_id:\$g\}\) WHERE n.uuid IN \$ids RETURN labels", q):
            hits = [self.nodes[u] for u in p["ids"] if u in self.nodes and m[1] in self.nodes[u]["labels_neo4j"]]
            return [{"l": list(n["labels_neo4j"]), "p": copy.deepcopy(n["props"])} for n in hits]
        if m := re.match(r"MATCH \(a\)-\[r:(\w+) \{group_id:\$g\}\]->\(b\) WHERE r.uuid IN \$ids", q):
            hits = [self.edges[u] for u in p["ids"] if u in self.edges and self.edges[u]["type"] == m[1]]
            return [{"t": e["type"], "s": e["source"], "d": e["target"], "p": copy.deepcopy(e["props"])} for e in hits]
        if m := re.match(r"MATCH \(n:(\w+) \{group_id:\$g\}\) RETURN labels\(n\) AS l", q):
            hits = [n for n in self.nodes.values() if m[1] in n["labels_neo4j"]]
            return [{"l": list(n["labels_neo4j"]), "p": copy.deepcopy(n["props"])} for n in hits]
        if q.startswith("MATCH (a)-[r {group_id:$g}]->(b) RETURN type(r)"):
            return [{"t": e["type"], "s": e["source"], "d": e["target"], "p": e["props"]} for e in self.edges.values()]
        if m := re.match(r"MATCH \(n:(\w+) \{group_id:\$g\}\) RETURN count\(n\) AS c", q):
            return [{"c": sum(m[1] in n["labels_neo4j"] for n in self.nodes.values())}]
        raise AssertionError(f"the fake Neo4j does not know this statement: {q}")

    def _merge_nodes(self, q: str, rows: list[dict]) -> list[dict]:
        first = re.search(r"MERGE \(n:(\w+) ", q)[1]
        extra = re.search(r" SET n:([\w:]+) WITH", q)
        labels = [first, *(extra[1].split(":") if extra else [])]
        unions = [f for f in UNION_FIELDS if f"coalesce(n.{f}" in q]
        created = 0
        for r in rows:
            have = self.nodes.get(r["uuid"])
            if have is None:
                self.nodes[r["uuid"]] = {"labels_neo4j": list(labels), "props": copy.deepcopy(r["props"])}
                created += 1
                continue
            props = have["props"]
            props.update(copy.deepcopy(r["update"]))
            for f in unions:
                if r["props"].get(f) is not None:
                    old = list(props.get(f) or [])
                    props[f] = old + [x for x in r["props"][f] if x not in old]
            if "n.first_episode_uuid = coalesce" in q:
                props["first_episode_uuid"] = props.get("first_episode_uuid") or r["props"].get("first_episode_uuid")
            have["labels_neo4j"] += [label for label in labels if label not in have["labels_neo4j"]]
        return [{"c": created}]

    def _merge_edges(self, q: str, rows: list[dict]) -> list[dict]:
        etype = re.search(r"MERGE \(a\)-\[e:(\w+) ", q)[1]
        ends = re.search(r"MATCH \(a:(\w+) .* MATCH \(b:(\w+) ", q)
        unions = [f for f in UNION_FIELDS if f"coalesce(e.{f}" in q]
        matched = created = 0
        for r in rows:
            a, b = self.nodes.get(r["source"]), self.nodes.get(r["target"])
            if not (a and ends[1] in a["labels_neo4j"] and b and ends[2] in b["labels_neo4j"]):
                continue
            matched += 1
            have = self.edges.get(r["uuid"])
            if have is None:
                self.edges[r["uuid"]] = {"type": etype, "source": r["source"], "target": r["target"]} | {
                    "props": copy.deepcopy(r["props"])
                }
                created += 1
                continue
            have["props"].update(copy.deepcopy(r["update"]))
            for f in unions:
                if r["props"].get(f) is not None:
                    old = list(have["props"].get(f) or [])
                    have["props"][f] = old + [x for x in r["props"][f] if x not in old]
        return [{"n": matched, "c": created}]


class Emb:
    model, tokens, requests = "m", 0, 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> Neo4j:
    """The fake database behind ``graphdb.driver``; what the write reports about the rows besides the gate (the
    fresh path's own business) is stubbed."""
    from memory_kernel.graph import kg_contract, kg_writer

    st = Neo4j()

    def a(value):
        async def call(*_a, **_k):
            return value() if callable(value) else value

        return call

    async def saga_passes(_driver, group):
        assert group == GROUP
        return [list(n["props"].get("ingest_pass") or []) for n in st.nodes.values() if "Saga" in n["labels_neo4j"]]

    snapshot = lambda: {"nodes": {GROUP: len(st.nodes), "other": 7}, "edges": {GROUP: len(st.edges)}}  # noqa: E731
    monkeypatch.setattr(write.graphdb, "driver", lambda _cfg: st)
    monkeypatch.setattr(write.graphdb, "group_counts", a(st.counts))
    monkeypatch.setattr(write.graphdb, "saga_passes", saga_passes)
    monkeypatch.setattr(write.graphdb, "snapshot", a(snapshot))
    monkeypatch.setattr(write.graphdb, "foreign_edges", a(0))
    monkeypatch.setattr(write.graphdb, "cypher_dups", a({}))
    monkeypatch.setattr(kg_writer, "ensure_indexes", a({}))
    monkeypatch.setattr(kg_writer, "validate_rows", lambda *_a, **_k: [])
    monkeypatch.setattr(kg_contract, "validate_rows", lambda *_a, **_k: [])
    monkeypatch.setattr(write, "invariants", lambda _rows, _ref: {})
    monkeypatch.setattr(write, "cap_check", lambda _ctx, _usd: {})
    monkeypatch.setattr(write, "projection_inputs", lambda _ctx: {})
    monkeypatch.setattr(write, "frozen_sha", lambda _rows, **_kw: "f")
    monkeypatch.setattr(write.embed, "embedder", lambda _cfg, _kind: None)
    # the stand-in vector cache: every text has a vector, nothing is sent to Voyage
    monkeypatch.setattr(write, "run_vectors", lambda _ctx, texts, _cost: {t: [float(len(t)), 1.0] for t in texts})
    monkeypatch.setattr(write.embed, "CappedEmbed", lambda _inner, **_cfg: Emb())
    monkeypatch.setattr(write.embed, "record", lambda _emb: {})
    for name in ("shape", "envelope_census"):
        monkeypatch.setattr(write.base, name, lambda _n, _e: {})
    monkeypatch.setattr(write.base, "graph_report", lambda _n, _e: {"event_names": {}})
    monkeypatch.setattr(write.base, "row_set_hash", lambda _n, _e: "r")
    monkeypatch.setattr(write.base, "content_hash", lambda _n, _e: "c")
    return st


def ctx_for(
    tmp_path,
    *,
    mode: str = "append",
    messages: int = 2,
    opened=("e3",),
    forward: bool = True,
    chain: Chain = CHAIN,
    inversion: dict | None = None,
    strict: bool = False,
):
    """A stage context of the session run at ``tmp_path``: its forward input (the ``chain``'s run and totals) and its
    boundary's continuation (``opened``); ``strict`` = ``strict_db_snapshot``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pinned = None
    if forward:
        pinned = tmp_path / "forward.json"
        totals = {"messages": chain.messages, "episodes": chain.episodes}
        pinned.write_text(json.dumps({"chat": "k1", "chain_run": chain.run, "totals": totals}))
    (tmp_path / "boundary").mkdir(exist_ok=True)
    continuation = {"opened": list(opened), "inversion": inversion}
    applied = {"episode_by_message": {"m4": "e2", "m5": "e3"}, "continuation": continuation}
    (tmp_path / "boundary" / "applied-boundaries.json").write_text(json.dumps(applied))
    cost = SimpleNamespace(add=lambda *_a, **_k: None, note="")
    return SimpleNamespace(
        cfg=SimpleNamespace(
            group=GROUP,
            write_mode=mode,
            ingest_pass=PASS,
            sample_id="conv-ember",
            root=tmp_path,
            invariants_reference=None,
            concurrent_group_prefix=None,
            strict_db_snapshot=strict,
            forward_input=pinned,
        ),
        counts=SimpleNamespace(messages=messages),
        path=lambda key: tmp_path / key,
        cost=lambda *_a, **_k: contextlib.nullcontext(cost),
    )


def append(
    monkeypatch, tmp_path, rows: dict | None = None, *, chain: Chain = CHAIN, messages=("m4", "m5"), **kw
) -> dict:
    """``run_write`` of the session (or ``rows`` of the session ``messages``) against ``chain``."""
    rows = copy.deepcopy(rows if rows is not None else session())
    records = [{"message_id": m, "text": f"{m} text"} for m in messages]
    data = {"records": records, "chain": chain, "run": chain.run + 1, "conversation_of": {m: "k1" for m in messages}}
    (tmp_path / "select").mkdir(parents=True, exist_ok=True)
    selected = [{**source(m, 2), "scope": {"conversation_id": "conv-ember"}} for m in messages]
    (tmp_path / "select" / "sources.json").write_text(json.dumps(selected))
    kwargs = {"ingest_pass": PASS}
    monkeypatch.setattr(write, "project", lambda _ctx, **_k: (data, kwargs, rows))
    return write.run_write(ctx_for(tmp_path, messages=len(messages), chain=chain, **kw))


# -- the group-state rules (unchanged) -----------------------------------------------------------------------------------


def test_append_accepts_a_non_empty_group_and_refuses_an_empty_one():
    write._check_counts(GROUP, {"nodes": 5, "edges": 3}, rewrite=False, append=True)
    write._check_counts(GROUP, {"nodes": 5, "edges": 3}, rewrite=True, append=True)
    for rewrite in (False, True):
        with pytest.raises(write.Refused, match="append to 'g-ember' but the group is empty"):
            write._check_counts(GROUP, {"nodes": 0, "edges": 0}, rewrite=rewrite, append=True)


def test_fresh_rules_are_unchanged():
    write._check_counts(GROUP, {"nodes": 0, "edges": 0}, rewrite=False)
    with pytest.raises(write.Refused, match="is not empty"):
        write._check_counts(GROUP, {"nodes": 1, "edges": 0}, rewrite=False)
    write._check_counts(GROUP, {"nodes": 1, "edges": 0}, rewrite=True)
    with pytest.raises(write.Refused, match="the first write is missing"):
        write._check_counts(GROUP, {"nodes": 0, "edges": 0}, rewrite=True)


def test_the_group_must_belong_to_this_bonfire():
    write.check_owner(GROUP, [[PASS], ["mk-earlier", PASS]], PASS)
    with pytest.raises(write.Refused, match="has no Saga"):
        write.check_owner(GROUP, [], PASS)
    with pytest.raises(write.Refused, match=r"does not belong to this bonfire: 1 of the 2 distinct ingest-pass lists"):
        write.check_owner(GROUP, [[PASS], ["mk-conv-other-001"]], PASS)
    with pytest.raises(write.Refused, match=r"<none>"):
        write.check_owner(GROUP, [[]], PASS)


# -- the forward append over the fake database ---------------------------------------------------------------------------


FACT = edge("RELATES_TO", ent("ana"), ent("kiln"))["props"]["uuid"]


def fact(m: str, key_text: str, plan: str) -> dict:
    """The fact "Ana fires the kiln" as message ``m`` states it: the same edge (endpoints, name, normalized fact,
    force) whatever words carried it, so a restatement MERGEs onto the stored one."""
    props = {
        "uuid": FACT,
        "group_id": GROUP,
        "created_at": CREATED,
        "name": "FIRES",
        "fact": "Ana fires the kiln",
        "episodes": [ep(m)],
        "event_keys": [f"{m}#0"],
        "valid_at": TS[m],
        "reference_time": TS[m],
        "source_uuid": ent("ana"),
        "target_uuid": ent("kiln"),
        "key_text": key_text,
        "plan": plan,
        "told_at": TS[m],
    }
    return {"type": "RELATES_TO", "source": ent("ana"), "target": ent("kiln"), "props": props}


def test_the_chain_commit_keeps_the_first_route_and_refuses_a_stray_one():
    folded = chain_state._fold_forward(
        replace(CHAIN, routes={"dee": ent("kiln")}), {"run": 1, "routes": {"dee": ent("ana")}}
    )
    assert folded[1] == {"dee": ent("kiln")}  # the first route for a name wins
    with pytest.raises(ChainError, match="lead to no entity of the chain"):
        chain_state._fold_forward(CHAIN, {"run": 1, "routes": {"cy": ent("cy")}})
    with pytest.raises(ChainError, match="minted at run 5, not 1"):
        chain_state._fold_forward(CHAIN, {"run": 5})


def test_summary_lines_append_once_by_line_id():
    """Lines append once by id. A stored node without per-line data (built before 1.4.1, not backfilled) gets none
    (its parallel lists stay all-absent, never partial); one with it, or a new node, grows texts and identity ids."""
    stored = {"summary": "a", "summary_sources": ["s1"], "summary_line_ids": ["l1"]}
    got = append_summary(stored, ["a", "b", "b"], ["s1", "s2", "s2"], ["l1", "l2", "l2"], ["l2"])
    assert got == {"summary": "a\nb", "summary_sources": ["s1", "s2"], "summary_line_ids": ["l1", "l2"]}
    assert append_summary({**stored, **got}, ["b"], ["s2"], ["l2"], []) == {}  # twice: the same node
    held = stored | {"summary_line_texts": ["a"], "summary_identity_line_ids": ["l1"]}
    got = append_summary(held, ["a", "b\nc", "d"], ["s1", "s2", "s3"], ["l1", "l2", "l3"], ["l1", "l3"])
    assert got["summary"] == "a\nb\nc\nd" and got["summary_line_ids"] == ["l1", "l2", "l3"]
    assert got["summary_line_texts"] == ["a", "b\nc", "d"] and got["summary_identity_line_ids"] == ["l1", "l3"]
    assert append_summary({}, ["c"], ["s3"], ["l3"], ["l3"]) == {"summary": "c", "summary_sources": ["s3"]} | {
        "summary_line_ids": ["l3"],
        "summary_line_texts": ["c"],
        "summary_identity_line_ids": ["l3"],
    }
    # desynced since its backfill (a 1.4.0 append grew the ids, not the texts) or texts without identity ids: no line
    # data grows on it (the backfill repairs it), the lines still append
    grown = held | {"summary": "a\nx", "summary_sources": ["s1", "s9"], "summary_line_ids": ["l1", "l9"]}
    unpaired = {k: v for k, v in held.items() if k != "summary_identity_line_ids"}
    for broken in (grown, unpaired):
        got = append_summary(broken, ["d"], ["s3"], ["l3"], ["l3"])
        assert got["summary_line_ids"][-1] == "l3" and got["summary"].endswith("\nd")
        assert "summary_line_texts" not in got and "summary_identity_line_ids" not in got


def test_create_only_fields_are_never_set_on_match():
    """A stored row keeps its mint order, its saga start and the times it was first written with (a fact a later
    session states again)."""
    from memory_kernel.graph.kg_contract import BASE_RELATES_TO, BASE_SAGA
    from memory_kernel.graph.kg_writer import _split

    fact = {"uuid": "f", "fact": "Ana likes pears", "episodes": ["e"], "told_at": "t2", "happened_at": "t2"}
    assert _split(fact, BASE_RELATES_TO)[1] == {}  # the episodes union apart, nothing is set on match
    started = saga_row("e2", ["m4"], prior=("m3",))["props"]
    assert "started_at" not in _split(started, BASE_SAGA)[1] and "ended_at" in _split(started, BASE_SAGA)[1]
    assert _split({"uuid": "x", "minted_order": [1, 0]}, frozenset({"uuid"}))[1] == {}


def test_an_append_needs_its_forward_input(db, monkeypatch, tmp_path):
    db.seed(bootstrap())
    with pytest.raises(write.Refused, match="an append write needs its forward input"):
        append(monkeypatch, tmp_path, forward=False)
    assert not [q for q in db.statements if "MERGE" in q]


def test_append_refuses_an_empty_group_and_a_foreign_one_before_writing(db, monkeypatch, tmp_path):
    with pytest.raises(write.Refused, match="the group is empty"):
        append(monkeypatch, tmp_path)
    db.seed(bootstrap())
    with pytest.raises(write.Refused, match="does not belong to this bonfire"):
        db.nodes[saga("e1")]["props"]["ingest_pass"] = ["mk-conv-other-001"]
        append(monkeypatch, tmp_path)
    assert not [q for q in db.statements if "MERGE" in q] and not (tmp_path / "write").exists()


def test_fresh_mode_is_unchanged(db, monkeypatch, tmp_path):
    rows = bootstrap()
    for n in rows["nodes"]:
        n["props"].pop("name_embedding", None)
    rows["external_endpoints"] = {}
    out = append(monkeypatch, tmp_path, rows, mode="fresh", forward=False)
    assert "append" not in out and "write_mode" not in out
    assert out["write"]["nodes_created"] == len(rows["nodes"]) and not any(
        "minted_order" in n["props"] for n in db.nodes.values()
    )
    with pytest.raises(write.Refused, match="is not empty"):  # a fresh first write still refuses a taken group
        append(monkeypatch, tmp_path / "again", rows, mode="fresh", forward=False)


def test_the_delete_path_and_prune_orphans_are_retired(tmp_path):
    from memory_kernel import cli
    from memory_kernel.pipeline import graphdb
    from memory_kernel.pipeline.config import BuildConfig

    assert not hasattr(graphdb, "prune_rows") and not hasattr(write, "orphans")
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["append", "fire", "session.json", "--prune-orphans"])
    cfg = BuildConfig(
        sample_id="conv-ember",
        conversation=tmp_path / "c.json",
        group=GROUP,
        purpose="remember the kiln",
        workdir=tmp_path / "w",
        root=tmp_path,
    )
    doc = cfg.to_json() | {"prune_orphans": False}  # an older bootstrap's build-config.json
    assert BuildConfig.from_json(doc, secrets={}) == cfg


# -- NEXT_SAGA from the chain (Review Focus 4) ---------------------------------------------------------------------------


def test_chain_sagas_feeds_the_chats_own_last_saga():
    two = replace(
        CHAIN,
        chats={
            **CHAIN.chats,
            "k2": ChatState(1, "e9", saga("e9"), 3, ["Bo"], [source("m4", 2)], {"messages": 1, "episode_start": 0}),
        },
    )
    (row,) = write.chain_sagas(two, "k1", GROUP, PASS)
    assert (row["uuid"], row["name"], row["episodes"]) == (saga("e2"), "e2", [ep("m3")])
    assert row["prior"] == [("m3", ep("m3"), "Ana", TS["m3"])] and row["ingest_pass"] == [PASS]
    assert write.chain_sagas(two, "k2", GROUP, PASS)[0]["uuid"] == saga("e9")
    assert write.chain_sagas(two, "k3", GROUP, PASS) == []  # a new chat starts its own chain
    short = replace(CHAIN, chats={"k1": replace(CHAIN.chats["k1"], boundary={"messages": 3, "episode_start": 1})})
    with pytest.raises(write.Refused, match="does not reach back to its open episode"):
        write.chain_sagas(short, "k1", GROUP, PASS)


# -- the driver's preflight -------------------------------------------------------------------------------------------


def _cfg(tmp_path, mode: str):
    from memory_kernel.pipeline.config import BuildConfig

    return BuildConfig(
        sample_id="conv-ember",
        conversation=tmp_path / "c.json",
        group=GROUP,
        purpose="remember the kiln",
        workdir=tmp_path / "w",
        root=tmp_path,
        neo4j_password="pw",
        write_mode=mode,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("counts", "refusal"),
    [
        ({"nodes": 5, "edges": 2, "saga_passes": [["mk-conv-ember-001"]]}, None),
        ({"nodes": 0, "edges": 0, "saga_passes": []}, "the group is empty"),
        ({"nodes": 5, "edges": 2, "saga_passes": [["mk-conv-other-001"]]}, "does not belong to this bonfire"),
    ],
)
def test_neo4j_preflight_in_append_mode_needs_the_bonfires_own_group(tmp_path, monkeypatch, counts, refusal):
    from memory_kernel.pipeline import runner

    async def probe(_cfg):
        return counts

    monkeypatch.setattr(runner, "_neo4j_probe", probe)
    cfg = _cfg(tmp_path, "append")
    got = runner.neo4j_preflight(cfg, {"write", "rewrite", "index", "sanity", "seal"})
    if refusal is None:
        assert got == []
    else:
        assert len(got) == 1 and refusal in got[0] and got[0].startswith("neo4j: append to 'g-ember'")


def test_neo4j_preflight_fresh_mode_is_unchanged(tmp_path, monkeypatch):
    from memory_kernel.pipeline import runner

    async def probe(_cfg):
        return {"nodes": 5, "edges": 2}

    monkeypatch.setattr(runner, "_neo4j_probe", probe)
    got = runner.neo4j_preflight(_cfg(tmp_path, "fresh"), {"write", "index"})
    assert len(got) == 1 and "is not empty" in got[0]


def test_the_probe_reads_saga_passes_only_in_append_mode(tmp_path, monkeypatch):
    from memory_kernel.pipeline import graphdb, runner
    from memory_kernel.serve import answer

    calls: list[str] = []

    async def counts(_d, _g):
        calls.append("counts")
        return {"nodes": 1, "edges": 0}

    async def passes(_d, _g):
        calls.append("passes")
        return [[PASS]]

    async def noop(*_a):
        return None

    monkeypatch.setattr(graphdb, "group_counts", counts)
    monkeypatch.setattr(graphdb, "saga_passes", passes)
    monkeypatch.setattr(graphdb, "driver", lambda _c: SimpleNamespace(verify_connectivity=noop, close=noop))
    monkeypatch.setattr(answer, "ReadOnlyDriver", lambda raw: raw)
    assert asyncio.run(runner._neo4j_probe(_cfg(tmp_path, "fresh"))) == {"nodes": 1, "edges": 0}
    assert calls == ["counts"]
    got = asyncio.run(runner._neo4j_probe(_cfg(tmp_path, "append")))
    assert got == {"nodes": 1, "edges": 0, "saga_passes": [[PASS]]}


def test_saga_passes_reads_lists_strings_and_missing_passes():
    from memory_kernel.pipeline import graphdb

    class Result:
        def __aiter__(self):
            async def gen():
                for p in (["a", "b"], "c", None):
                    yield {"p": p}

            return gen()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def run(self, query, **params):
            assert query.startswith("MATCH (s:Saga {group_id:$g})") and params == {"g": GROUP}
            return Result()

    got = asyncio.run(graphdb.saga_passes(SimpleNamespace(session=Session), GROUP))
    assert got == [["a", "b"], ["c"], []]


def test_a_discriminated_instance_is_not_a_duplicate_name():
    """An append's referents mint a specific instance as ``kind#<8 hex>`` under its kind's name: the build_dry
    duplicate-name check counts real dedup failures only."""
    from memory_kernel.pipeline.stages import _g1_build

    def entity(uuid, name, key):
        return {"labels_neo4j": ["Entity"], "props": {"uuid": uuid, "name": name, "name_key": key}}

    nodes = [entity("a1", "answer", "answer"), entity("a2", "answer", "answer#0123abcd")]
    report = _g1_build.graph_report(nodes, [])
    assert report["entity_name_dups_excl_events"] == 0 and report["entity_instances_excl_events"] == 1
    nodes.append(entity("g1", "graph", None))
    nodes.append(entity("g2", "graph", "graph"))
    report = _g1_build.graph_report(nodes, [])
    assert report["entity_name_dups_excl_events"] == 1 and report["entity_name_dup_names_excl_events"] == ["graph"]
    # a later append that reuses both a stored instance and its kind: reused rows carry no name_key, the stored key tells
    reused = [entity("k1", "Kernel", None), entity("k2", "Kernel", None)]
    report = _g1_build.graph_report(reused, [], stored_keys={"k1": "kernel#1f5432dd", "k2": "kernel"})
    assert report["entity_name_dups_excl_events"] == 0 and report["entity_instances_excl_events"] == 1
