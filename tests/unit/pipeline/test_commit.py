"""Kernel 2.0: one Neo4j write transaction per append is the commit (``pipeline.commit``).

In graph mode (``BuildConfig.graph_commit`` with an append) the write, rewrite and index stages run as rolled-back
transactions that record their vectors (``embed.RecordingEmbed``), and ``commit_append`` replays them
(``embed.ReplayEmbed``) in one guarded transaction: nothing half-written is ever visible, a retry never doubles a
count nor changes a decision, and the committed keys are the dry pass's.

The bonfire is a real one on the test Neo4j, at $0: a two-chat bootstrap (three sessions) projected by the real
``g1_projection.project`` (structure policy), stamped and written with its Roleset / Route / BonfireMeta rows
(``committed_run`` 0) and its search keys; then one appended session of chat k2 projected against the chain the 2.0
reader gives (``state_read.read_chain``). It mints Gil, claims "my name is Mira" (a ``SAME_AS`` and the route
``mira`` to Ana, a stored entity), asks the new roleset ``own.01`` and adds the agent handle ``@scribe``. The stages
run for real over that run dir; only the projection call, the message vectors, the money caps and Voyage are stood
in (``FakeVoyage``: a deterministic vector per text). Invented names only."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.graph.g1_names import EVENTUALITIES
from memory_kernel.graph.g1_projection import project
from memory_kernel.graph.kg_writer import (
    KEY_EMBED_DIM,
    UNION_FIELDS,
    ensure_constraints,
    ensure_indexes,
    read_group,
    write_rows,
)
from memory_kernel.graph.state import meta_props, meta_row, roleset_rows, route_rows, stamp_state
from memory_kernel.graph.state_read import read_chain, read_meta
from memory_kernel.graph.tx import transaction
from memory_kernel.knowledge.temporal_deixis import TEMPORAL_DEIXIS_ENV
from memory_kernel.pipeline import chain as ch
from memory_kernel.pipeline import commit, embed, graphdb
from memory_kernel.pipeline.bonfire import session_date
from memory_kernel.pipeline.embed import message_texts
from memory_kernel.pipeline.runner import STAGE_DIRS
from memory_kernel.pipeline.stages import index as index_stage
from memory_kernel.pipeline.stages import write
from memory_kernel.pipeline.stages._conversation import FIRST_SESSION, build_prepared, build_sources, split_conversation
from memory_kernel.pipeline.stages._g1_build import CREATED_AT
from memory_kernel.search import kg_index

SAMPLE = "conv-kiln"
PASS = "mk-conv-kiln-001"
POLICY = "structure"
RENDER = "legacy"
SCOPE = {"bonfire_id": "bf-kiln", "conversation_id": SAMPLE}
MODEL = {"name": "fixture-parser", "version": "0"}
LABEL = {"like.01": "occurrence", "meet.01": "occurrence", "be.01": "state", "own.01": "state"}


def me(speaker: str) -> dict:
    return {"text": "I", "span": [0], "referent": {"status": "deixis", "text": speaker}}


def r(text: str, span: list[int]) -> dict:
    return {"text": text, "span": span}


def my_name(name: str) -> tuple:
    return (f"my name is {name}", "be.01", [2], {"arg1": r("my name", [0, 1]), "arg2": r(name, [3])})


def likes(who: str, what: str) -> tuple:
    return (f"{who} likes {what}", "like.01", [1], {"arg0": r(who, [0]), "arg1": r(what, [2])})


# session -> (chat, start, [(message id, speaker, addressees, (text, roleset, anchor, roles))])
BOOT = {
    1: ("k1", "2026-06-03T09:00:00+00:00", [
        ("b1", "Ana", [], ("I like apples", "like.01", [1], {"arg0": me("Ana"), "arg1": r("apples", [2])})),
        ("b2", "Bo", ["Ana"], likes("Mira", "pears")),
        ("b3", "Eli", ["helper"], likes("Eli", "figs")),
    ]),
    2: ("k2", "2026-06-04T10:00:00+00:00", [("b4", "Dov", ["scribe"], likes("Bo", "plums"))]),
    3: ("k1", "2026-06-05T11:00:00+00:00", [
        ("b5", "Bo", [], my_name("Dee")),
        ("b6", "Ana", [], ("Dee met Cy", "meet.01", [1], {"arg0": r("Dee", [0]), "arg1": r("Cy", [2])})),
        ("b7", "Eli", [], ("I like plums", "like.01", [1], {"arg0": me("Eli"), "arg1": r("plums", [2])})),
    ]),
}  # fmt: skip
EPISODES0 = {"b1": "ep-1", "b2": "ep-1", "b3": "ep-2", "b4": "ep-3", "b5": "ep-4", "b6": "ep-4", "b7": "ep-4"}
NUMBER, CHAT, START, HANDLES = 4, "k2", "2026-06-06T12:00:00+00:00", ["@scribe"]
APPEND = [
    ("a1", "Ana", [], my_name("Mira")),
    ("a2", "Dov", ["Ana"], likes("Eli", "pears")),
    ("a3", "Dov", [], ("Gil owns plums", "own.01", [1], {"arg0": r("Gil", [0]), "arg1": r("plums", [2])})),
]  # fmt: skip
EPISODES1 = {"a1": "ep-5", "a2": "ep-5", "a3": "ep-6"}
OPENED = ["ep-5", "ep-6"]


def p_yes(top: str) -> dict:
    return {e: (0.9 if e == top else 0.05) for e in EVENTUALITIES}


ROLESETS0 = {
    "like.01": {"gloss": "like (roles: arg0 = liker, arg1 = thing liked)", "gloss_source": "frames",
                "examples": ["Ana: I like apples"], "label": "occurrence", "p_yes": p_yes("occurrence"),
                "compiled_sha256": "1a" * 32},
    "be.01": {"gloss": "be (roles: arg1 = topic, arg2 = comment)", "gloss_source": "frames",
              "examples": ["Bo: my name is Dee"], "label": "state", "p_yes": p_yes("state"),
              "compiled_sha256": "2b" * 32},
    "meet.01": {"gloss": "meet (roles: arg0 = one party, arg1 = other party)", "gloss_source": "frames",
                "examples": ["Ana: Dee met Cy"]},
}  # fmt: skip
NEW_ROLESETS = {  # the run's events stage ``new_rolesets`` (Task 5's row shape)
    "own.01": {"gloss": "own (roles: arg0 = owner, arg1 = possession)", "gloss_source": "frames",
               "examples": ["Dov: Gil owns plums"], "label": "state", "p_yes": p_yes("state"),
               "compiled_sha256": "3c" * 32},
}  # fmt: skip


def vector(text: str) -> list[float]:
    """The stand-in Voyage vector of ``text``: deterministic, ``KEY_EMBED_DIM`` wide."""
    digest = hashlib.sha256(text.encode()).digest()
    return [round(b / 255, 6) for b in digest] * (KEY_EMBED_DIM // len(digest))


class FakeVoyage:
    """The stand-in Voyage embedder; ``texts`` logs every text it was asked for."""

    model, tokens, requests = "voyage-3", 0, 0

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.texts += list(texts)
        return [vector(t) for t in texts]


def turns(start: str, spec: list) -> list[dict]:
    t0 = datetime.fromisoformat(start)
    return [
        {
            "speaker": speaker,
            "dia_id": mid,
            "text": body[0],
            "timestamp": (t0 + timedelta(minutes=i)).isoformat(),
            **({"addressees": addressees} if addressees else {}),
        }
        for i, (mid, speaker, addressees, body) in enumerate(spec)
    ]


def graph_records(sources: list[dict], spec: dict) -> list[dict]:
    return [
        {
            "unit": f"{s['order']:04d}/unit-00.json",
            "message_id": s["message_id"],
            "session_id": s["session_id"],
            "speaker": s["speaker"],
            "text": s["text"],
            "timestamp": s["timestamp"],
            "capture_status": "captured",
            "addressees": s["addressees"],
            "events": [
                {
                    "roleset": spec[s["message_id"]][1],
                    "anchor": spec[s["message_id"]][2],
                    "roles": spec[s["message_id"]][3],
                }
            ],
        }  # fmt: skip
        for s in sources
    ]


def eventuality(records: list[dict]) -> dict:
    out = {}
    for rec in records:
        label = LABEL[rec["events"][0]["roleset"]]
        out[f"{rec['unit']}#0"] = {"label": label, "p": {label: 1.0}}
    return out


# -- the live test Neo4j -----------------------------------------------------------------------------------------------


def neo(target, fn) -> Any:
    """``fn(driver)`` on a driver of the test Neo4j, in its own event loop."""
    from neo4j import AsyncGraphDatabase

    async def go():
        driver = AsyncGraphDatabase.driver(target.uri, auth=(target.user, target.password))
        try:
            return await fn(driver)
        finally:
            await driver.close()

    return asyncio.run(go())


async def snapshot(driver, group: str) -> str:
    """The group, every node and edge with its labels and props, as canonical JSON."""
    nodes, edges = await read_group(driver, group)
    nodes = sorted(([sorted(n["labels_neo4j"]), n["props"]] for n in nodes), key=lambda n: n[1]["uuid"])
    edges = sorted(([e["type"], e["source"], e["target"], e["props"]] for e in edges), key=lambda e: e[3]["uuid"])
    return json.dumps([nodes, edges], sort_keys=True)


async def seed(driver, group: str) -> tuple[ch.Chain, list[dict]]:
    """The bootstrap, committed as a 2.0 bonfire: its rows stamped (run 0), its Roleset / Route rows, its BonfireMeta
    at ``committed_run`` 0 and every message's search key. Returns ``(the chain read back, its select rows)``."""
    await ensure_indexes(driver)
    await ensure_constraints(driver)
    conv: dict = {"agent_handles": ["@helper"]}
    chat_of: dict[str, str] = {}
    spec: dict = {}
    for n, (chat, start, msgs) in BOOT.items():
        conv[f"session_{n}_date_time"] = session_date(datetime.fromisoformat(start))
        conv[f"session_{n}_conversation"] = chat
        conv[f"session_{n}"] = turns(start, msgs)
        chat_of |= {m[0]: chat for m in msgs}
        spec |= {m[0]: m[3] for m in msgs}
    conv_only = split_conversation({"sample_id": SAMPLE, "conversation": conv, "qa": []})[0]
    sources = build_sources(build_prepared(conv_only), SCOPE)
    records = graph_records(sources, spec)
    rows = project(
        records, eventuality=eventuality(records), episodes=EPISODES0, group_id=group, created_at=CREATED_AT,
        ingest_pass=PASS, conversation_of=chat_of, referent_policy=POLICY,
    )  # fmt: skip
    counts = Counter(chat_of.values())
    boot = ch.from_bootstrap(
        copy.deepcopy(rows), sources, EPISODES0, chat_of, ROLESETS0,
        walks={c: ch.closed_boundary(k) for c, k in counts.items()}, sessions=len(BOOT),
        agent_handles=conv["agent_handles"],
    )  # fmt: skip
    stamp_state(
        rows, run=0, records=sources, chat_of=chat_of, vectors={t: vector(t) for t in message_texts(sources).values()}
    )
    route_nodes, route_edges = route_rows(group, boot.routes, run=0)
    meta = meta_props(
        bonfire_id=SAMPLE,
        build_config={
            "sample_id": SAMPLE,
            "group": group,
            "ingest_pass": PASS,
            "referent_policy": POLICY,
            "boundary_policy": "segmenter",
        },
        recipe_tag=kg_index.recipe_tag(RENDER),
        extraction_model=MODEL,
        agent_handles=boot.agent_handles,
        agent_handle_runs=[0] * len(boot.agent_handles),
        chat_tagged=True,
        explicit_since=boot.explicit_since,
        sessions=boot.sessions,
        committed_run=0,  # the run before the append's
        generation="gen-seed-1",
    )
    whole = {
        "nodes": [*rows["nodes"], *roleset_rows(group, ROLESETS0, run=0), *route_nodes, meta_row(group, meta)],
        "edges": [*rows["edges"], *route_edges],
        "external_endpoints": rows["external_endpoints"],
    }
    await write_rows(driver, whole, group_id=group, embed=FakeVoyage())
    msgs, facts, mentions = await kg_index.read_key_rows(driver, group)
    keys = kg_index.keys_from_graph(msgs, facts, mentions, key_render=RENDER)
    await kg_index.persist_keys(driver, group, keys, FakeVoyage(), key_render=RENDER)
    return await read_chain(driver, group), sources


@dataclass
class Fire:
    """One append's run over the seeded bonfire: the run dir, its stage context and what it was projected from."""

    target: Any
    group: str
    chain: ch.Chain  # the committed chain the run started from (run 0)
    view: ch.Chain  # the chain as the run's forward input pins it
    run_root: Path
    ctx: Any
    data: dict
    rows: dict  # the projection of the session (unstamped)
    voyage: FakeVoyage
    generation: str  # the BonfireMeta generation the run started from

    def stage(self, key: str) -> Path:
        return self.run_root / STAGE_DIRS[key]

    def json(self, key: str, name: str) -> Any:
        return json.loads((self.stage(key) / name).read_bytes())

    def snapshot(self) -> str:
        return neo(self.target, lambda d: snapshot(d, self.group))

    def meta(self) -> dict:
        return neo(self.target, lambda d: read_meta(d, self.group))

    def dry(self) -> tuple[dict, dict, dict]:
        """The write, rewrite and index stages of the run, in graph mode."""
        summary = write.run_write(self.ctx)
        rerun = write.rewrite(self.ctx, True)
        indexed = index_stage.index(self.ctx, True)
        return summary, rerun, indexed

    def commit_input(self) -> commit.CommitInput:
        return commit.commit_input(
            self.run_root, group=self.group, run=self.chain.run + 1, chain=self.chain, generation=self.generation
        )

    def commit(self, ci: commit.CommitInput) -> dict:
        return neo(self.target, lambda d: commit.commit_append(d, ci))


@pytest.fixture
def fire(neo4j_target, tmp_path, monkeypatch) -> Fire:
    return make_fire(neo4j_target, tmp_path, monkeypatch)


def make_fire(neo4j_target, tmp_path: Path, monkeypatch, *, graph_commit: bool = True) -> Fire:
    """Seed the bonfire into the target's group, then lay out the append's run dir and stage context."""
    monkeypatch.delenv(TEMPORAL_DEIXIS_ENV, raising=False)
    group = neo4j_target.group
    chain, _ = neo(neo4j_target, lambda d: seed(d, group))
    generation = neo(neo4j_target, lambda d: read_meta(d, group))["generation"]  # read at run start

    pinned = json.loads(json.dumps(ch.projection_view(chain)))  # what forward.json pins
    view = ch.from_projection_view(pinned)
    handles = list(dict.fromkeys([*chain.agent_handles, *HANDLES]))
    key = f"session_{NUMBER}"
    session_turns = turns(START, APPEND)
    conv = {
        FIRST_SESSION: NUMBER,
        key: session_turns,
        f"{key}_date_time": session_date(datetime.fromisoformat(session_turns[0]["timestamp"])),
        f"{key}_conversation": CHAT,
        "agent_handles": handles,
    }
    prepared = build_prepared(split_conversation({"sample_id": SAMPLE, "conversation": conv})[0])
    sources = build_sources(prepared, SCOPE, ch.next_order(chain))
    records = graph_records(sources, {m[0]: m[3] for m in APPEND})
    chat_of = {s["message_id"]: CHAT for s in sources}
    run = view.run + 1
    rows = project(
        records, eventuality=eventuality(records), episodes=EPISODES1, group_id=group, created_at=CREATED_AT,
        ingest_pass=PASS, conversation_of=chat_of, chain=view, run=run,
        existing_sagas=write.chain_sagas(view, CHAT, group, PASS), referent_policy=POLICY, referee=None,
    )  # fmt: skip

    run_dir = tmp_path / "runs" / f"{run:04d}"
    run_root = run_dir / SAMPLE

    def put(key: str, name: str, doc: Any) -> None:
        (run_root / STAGE_DIRS[key]).mkdir(parents=True, exist_ok=True)
        (run_root / STAGE_DIRS[key] / name).write_text(json.dumps(doc))

    put("select", "sources.json", sources)
    before = chain.chats[CHAT].boundary["messages"]
    continuation = {
        "chat": CHAT,
        **ch.closed_boundary(before + len(sources)),
        "opened": OPENED,
        "last_episode": OPENED[-1],
        "inversion": None,
    }
    put("boundary", "applied-boundaries.json", {"episode_by_message": EPISODES1, "continuation": continuation})
    put("events", "eventuality.json", {"new_rolesets": NEW_ROLESETS})
    totals = {"messages": chain.messages, "episodes": chain.episodes}
    forward = {"chat": CHAT, "chain_run": chain.run, "totals": totals, "chain": pinned}
    (run_dir / "forward.json").write_text(json.dumps(forward))
    (run_dir / "conversation.json").write_text(json.dumps({"sample_id": SAMPLE, "conversation": conv}))
    start = {"event": "driver-start", "recipe": {"name": "mk-build", "extraction_model": MODEL}}
    (run_root / "driver-events.jsonl").write_text(json.dumps(start) + "\n")

    cfg = SimpleNamespace(
        group=group,
        sample_id=SAMPLE,
        ingest_pass=PASS,
        write_mode="append",
        graph_commit=graph_commit,
        root=run_root,
        forward_input=run_dir / "forward.json",
        neo4j_uri=neo4j_target.uri,
        neo4j_user=neo4j_target.user,
        neo4j_password=neo4j_target.password,
        invariants_reference=None,
        concurrent_group_prefix=None,
        strict_db_snapshot=False,
        recipe_manifest=None,
        bonfire_index_record=None,
        referent_policy=POLICY,
    )
    cost = SimpleNamespace(add=lambda *_a, **_k: None, note="")
    ctx = SimpleNamespace(
        cfg=cfg,
        counts=SimpleNamespace(messages=len(sources)),
        path=lambda key: run_root / STAGE_DIRS[key],
        cost=lambda *_a, **_k: contextlib.nullcontext(cost),
    )
    data = {"records": records, "chain": view, "run": run, "conversation_of": chat_of}
    voyage = FakeVoyage()
    monkeypatch.setattr(write, "project", lambda _ctx, **_kw: (data, {"ingest_pass": PASS}, copy.deepcopy(rows)))
    monkeypatch.setattr(write, "run_vectors", lambda _ctx, texts, _cost: {t: vector(t) for t in texts})
    monkeypatch.setattr(write, "invariants", lambda _rows, _ref: {})
    monkeypatch.setattr(write, "cap_check", lambda _ctx, _usd: {})
    monkeypatch.setattr(index_stage, "cap_check", lambda _ctx, _usd: {})
    monkeypatch.setattr(write, "projection_inputs", lambda _ctx: {})
    monkeypatch.setattr(embed, "embedder", lambda _cfg, _kind: voyage)
    return Fire(neo4j_target, group, chain, view, run_root, ctx, data, rows, voyage, generation)


# -- the embedders -----------------------------------------------------------------------------------------------------


def test_replay_embed_raises_on_a_miss(tmp_path):
    inner = FakeVoyage()
    rec = embed.RecordingEmbed(inner)
    assert rec(["a", "b"]) == [vector("a"), vector("b")]
    assert rec.vectors == {"a": vector("a"), "b": vector("b")} and (rec.model, rec.tokens) == ("voyage-3", 0)
    rec.save(tmp_path / "v.json")
    recorded = json.loads((tmp_path / "v.json").read_text())
    assert recorded == rec.vectors

    replay = embed.ReplayEmbed(recorded)
    assert replay(["b", "a", "b"]) == [vector("b"), vector("a"), vector("b")]
    assert (replay.tokens, replay.requests) == (0, 0)
    with pytest.raises(embed.EmbeddingMiss, match="1 document texts"):
        replay(["a", "c"])
    assert inner.texts == ["a", "b"]  # the replay never reached an embedder
    assert embed.record(replay)["kind"] == "replay"
    assert embed.record(rec) == embed.record(inner)  # a recording reports what it wraps


# -- the dry stages ----------------------------------------------------------------------------------------------------


@pytest.mark.needs_neo4j
def test_forward_gate_accepts_this_runs_type_rows(fire):
    g = fire.group
    summary = write.run_write(fire.ctx)
    assert summary["dry"] is True and summary["append"]["passed"]

    sent = fire.json("write", write.COMMIT_ROWS)
    labels = {n["props"]["uuid"]: n["labels_neo4j"] for n in sent["nodes"]}
    roleset, route = content_uuid("roleset", g, "own.01"), content_uuid("route", g, "mira")
    assert labels[roleset] == ["Roleset"] and labels[route] == ["Route"]
    (routes_to,) = [e for e in sent["edges"] if e["type"] == "ROUTES_TO"]
    ana = fire.rows["forward"]["routes"]["mira"]
    assert (routes_to["source"], routes_to["target"]) == (route, ana) and ana in fire.chain.entities
    assert ana in labels or ana in sent["external_endpoints"]  # a stored target the rows do not carry
    assert content_uuid("route", g, "dee") not in labels  # the bootstrap's route is the chain's: never sent again

    # the type rows pass rule 1 because their uuids join own_uuids; without them the gate refuses
    rows = copy.deepcopy(fire.rows)
    rows["forward"]["routes"] = {**rows["forward"]["routes"], "dee": ana}  # an already-routed key, re-routed
    typed, own_types = write.with_type_rows(fire.ctx, fire.data, rows)
    assert content_uuid("route", g, "dee") not in {n["props"]["uuid"] for n in typed["nodes"]}  # no second ROUTES_TO
    own = write.own_uuids(rows["forward"], fire.data["records"], OPENED, g)
    with pytest.raises(write.Refused, match="rule 1: Roleset"):
        write.forward_gate(typed, [], [], fire.view, own, chat=CHAT)
    write.forward_gate(typed, [], [], fire.view, own | own_types, chat=CHAT)
    assert own_types == {roleset, route}


@pytest.mark.needs_neo4j
def test_dry_write_leaves_the_group_unchanged(fire):
    before = fire.snapshot()
    summary, rerun, indexed = fire.dry()
    assert fire.snapshot() == before
    assert fire.meta()["committed_run"] == 0

    assert summary["dry"] is True and summary["append"]["passed"] and summary["append"]["created"]["nodes"]
    assert summary["append"]["episodic"] == {"got": 10, "expected": 10, "passed": True}  # inside its transaction
    assert (fire.stage("write") / write.WRITE_VECTORS).is_file()
    # the rewrite replays the rows twice in one transaction: the second creates and embeds nothing
    assert rerun["dry"] is True and rerun["gate"]["pass"]
    assert rerun["first"]["nodes_created"] == summary["write"]["nodes_created"]
    assert not any(rerun["write"][k] for k in ("nodes_created", "edges_created", "node_texts_embedded"))
    # the index persisted the session's keys in its own transaction, under 1.8's gate
    assert indexed["passed"] and indexed["dry"] is True and indexed["keys_stamped_with_tag"] == 10
    keys = fire.json("index", index_stage.KEYS)
    scope = summary["append"]["forward"]["episodic"]
    assert set(keys["keys"]) == set(scope) and set(keys["vectors"]) == set(keys["keys"].values())
    assert keys["key_render"] == RENDER and keys["recipe_tag"] == kg_index.recipe_tag(RENDER)


# -- the commit --------------------------------------------------------------------------------------------------------


@pytest.mark.needs_neo4j
def test_commit_writes_everything_or_nothing(fire):
    summary, _, _ = fire.dry()
    ci = fire.commit_input()
    before = fire.snapshot()
    broken = replace(ci, expected={**ci.expected, "Saga": ci.expected["Saga"] + 1})  # step 6 fails
    with pytest.raises(commit.CommitRefused, match="Saga"):
        fire.commit(broken)
    assert fire.snapshot() == before and fire.meta()["committed_run"] == 0

    out = fire.commit(ci)
    assert out["guard"] == {"previous": 0, "committed_run": 1} and out["already_committed"] is False
    assert out["created"] == {k: summary["append"]["created"][k] for k in ("nodes", "edges")}  # what the dry pass wrote
    assert out["keys"]["embedded"] == len(ci.scope)
    assert neo(fire.target, lambda d: graphdb.label_counts(d, fire.group)) == ci.expected == {"Episodic": 10, "Saga": 6}
    meta = fire.meta()
    assert (meta["committed_run"], meta["sessions"], meta["explicit_since"]) == (1, NUMBER, 1)
    assert (meta["agent_handles"], meta["agent_handle_runs"]) == (["@helper", "@scribe"], [0, 1])
    assert json.loads(meta["extraction_model_json"]) == MODEL and meta["committed_at"]

    after = neo(fire.target, lambda d: read_chain(d, fire.group))  # the next append's chain
    assert (after.run, after.messages, after.episodes, after.sessions) == (1, 10, 6, NUMBER)
    assert after.routes["mira"] == fire.rows["forward"]["routes"]["mira"] and after.rolesets["own.01"]["run"] == 1
    assert {m.key for m in after.entities.values() if m.run == 1} >= {"gil"}


async def set_meta(driver, group: str, **props) -> None:
    async with driver.session() as s:
        await (await s.run("MATCH (m:BonfireMeta {group_id: $g}) SET m += $p", g=group, p=props)).consume()


@pytest.mark.needs_neo4j
def test_commit_refused_when_committed_run_moved(fire):
    fire.dry()
    ci = fire.commit_input()
    fire.commit(ci)  # the replacement worker committed run 1 (its own session)
    after = fire.snapshot()
    other = replace(ci, scope=[content_uuid("ep", fire.group, "z1")])  # the stale worker's session of run 1
    with pytest.raises(commit.StaleCommit, match="committed run is 1, not 0"):
        fire.commit(other)
    assert fire.snapshot() == after

    neo(fire.target, lambda d: set_meta(d, fire.group, committed_run=2))  # and the bonfire moved on
    moved = fire.snapshot()
    with pytest.raises(commit.StaleCommit, match="committed run is 2, not 0"):
        fire.commit(ci)
    assert fire.snapshot() == moved


@pytest.mark.needs_neo4j
def test_a_commit_sent_twice_is_already_committed(fire):
    """A lost acknowledgement or a duplicate job sends the same commit again: it writes nothing and says so."""
    fire.dry()
    ci = fire.commit_input()
    assert fire.commit(ci)["already_committed"] is False
    once = fire.snapshot()
    again = fire.commit(ci)
    assert again["already_committed"] is True and again["created"] == {"nodes": 0, "edges": 0}
    assert again["guard"] == {"previous": 0, "committed_run": 1}
    assert fire.snapshot() == once


@pytest.mark.needs_neo4j
def test_a_re_created_bonfire_is_stale(fire):
    """A bonfire emptied and seeded again while the run was in flight is another generation, at the same
    ``committed_run``: the guard refuses it."""
    fire.dry()
    ci = fire.commit_input()
    assert ci.generation == "gen-seed-1"
    neo(fire.target, lambda d: set_meta(d, fire.group, generation="gen-seed-2"))  # re-seeded meanwhile
    before = fire.snapshot()
    assert fire.meta()["committed_run"] == ci.run - 1
    with pytest.raises(commit.StaleCommit, match="re-created"):
        fire.commit(ci)
    assert fire.snapshot() == before


@pytest.mark.needs_neo4j
def test_an_overlapping_commit_of_the_same_run_is_stale(fire):
    """The guard locks the BonfireMeta node before it reads ``committed_run``: a commit that starts while another
    commit of its run holds the guard waits, then finds the run moved."""
    fire.dry()
    ci = fire.commit_input()

    async def race(driver):
        async with transaction(driver, commit=True) as tx:
            await commit.guard(tx, ci)  # the other worker, its guard taken
            late = asyncio.create_task(commit.commit_append(driver, ci))
            await asyncio.sleep(1.0)
            assert not late.done()  # waiting on the meta node
        with pytest.raises(commit.StaleCommit):
            await late
        return await graphdb.label_counts(driver, ci.group)

    assert neo(fire.target, race) == {"Episodic": 7, "Saga": 4}  # the late commit wrote nothing
    assert fire.meta()["committed_run"] == 1


@pytest.mark.needs_neo4j
def test_commit_makes_no_embedding_call(fire, monkeypatch):
    from memory_kernel.graph import kg_writer

    fire.dry()
    ci = fire.commit_input()
    monkeypatch.setattr(embed, "embedder", lambda *_a, **_k: pytest.fail("the commit builds no embedder"))
    monkeypatch.setattr(kg_writer.VoyageEmbedder, "__call__", lambda *_a, **_k: pytest.fail("the commit embeds"))
    served: list[str] = []

    class Spy(embed.ReplayEmbed):
        def __call__(self, texts):
            served.extend(texts)
            return super().__call__(texts)

    monkeypatch.setattr(embed, "ReplayEmbed", Spy)

    before = fire.snapshot()
    gil = next(n["props"] for n in ci.rows["nodes"] if n["props"].get("name_key") == "gil")
    short = replace(ci, write_vectors={t: v for t, v in ci.write_vectors.items() if t != gil["name"]})
    with pytest.raises(embed.EmbeddingMiss):  # a text the dry pass did not record is never embedded here
        fire.commit(short)
    assert fire.snapshot() == before

    served.clear()
    fire.commit(ci)
    # every vector the commit stored is a replay of exactly what the dry write and the dry index recorded
    assert sorted(set(served)) == sorted(set(ci.write_vectors) | set(ci.key_vectors))

    async def stored(driver):
        async with driver.session() as s:
            ent = await (
                await s.run(
                    "MATCH (n:Entity {uuid: $u, group_id: $g}) RETURN n.name_embedding AS v", u=gil["uuid"], g=ci.group
                )
            ).single()
            eps = await s.run(
                "MATCH (m:Episodic {group_id: $g}) WHERE m.uuid IN $ids RETURN m.search_key AS k, "
                "m.search_key_embedding AS v",
                g=ci.group,
                ids=ci.scope,
            )
            return ent["v"], [(r["k"], r["v"]) async for r in eps]

    name_vector, keyed = neo(fire.target, stored)
    assert name_vector == ci.write_vectors[gil["name"]]
    assert len(keyed) == len(ci.scope) and all(v == ci.key_vectors[k] for k, v in keyed)


@pytest.mark.needs_neo4j
def test_create_only_decision_fields_survive_a_later_merge(fire):
    fire.dry()
    ci = fire.commit_input()
    fire.commit(ci)
    before = json.loads(fire.snapshot())

    rows = copy.deepcopy(ci.rows)  # a later run re-sends the same rows with every decision moved
    moved: Counter[str] = Counter()
    for n in rows["nodes"]:
        p = n["props"]
        if "Event" in n["labels_neo4j"]:
            p |= {"eventuality": "habit", "eventuality_source": "later", "run": 2}
            moved["event"] += 1
        if p.get("mint_kind") is not None:
            p |= {"mint_kind": "person" if p["mint_kind"] != "person" else "entity", "ground_legacy": True, "run": 2}
            p["minted_order"] = [2, 0]
            moved["minted"] += 1
        if "Episodic" in n["labels_neo4j"]:
            p |= {"order": 99, "chat": "k9", "session": 99, "run": 2}
            moved["episodic"] += 1
        if "Roleset" in n["labels_neo4j"]:
            p |= {"gloss": "own (rewritten)", "examples": ["rewritten"], "first_run": 2, "label": "occurrence"}
            moved["roleset"] += 1
    for e in rows["edges"]:
        e["props"]["run"] = 2
        moved["edge"] += 1
    assert all(moved[k] for k in ("event", "minted", "episodic", "roleset", "edge")), moved
    fire.commit(replace(ci, run=2, rows=rows))

    after = json.loads(fire.snapshot())
    meta_uuid = content_uuid("meta", fire.group)
    strip = lambda snap: [[n for n in snap[0] if n[1]["uuid"] != meta_uuid], snap[1]]  # noqa: E731
    assert strip(after) == strip(before)  # no decision, roleset or route moved
    assert sum(e[0] == "ROUTES_TO" for e in after[1]) == len(fire.chain.routes) + 1  # one per routed key
    assert fire.meta()["committed_run"] == 2


@pytest.mark.needs_neo4j
def test_retry_after_a_crash_commits_once_with_no_duplicates(fire, monkeypatch):
    fire.dry()
    ci = fire.commit_input()
    before = fire.snapshot()
    real = kg_index.persist_keys

    async def killed(*args, **kwargs):
        await real(*args, **kwargs)
        raise RuntimeError("worker killed")

    monkeypatch.setattr(kg_index, "persist_keys", killed)
    with pytest.raises(RuntimeError, match="worker killed"):
        fire.commit(ci)
    assert fire.snapshot() == before and fire.meta()["committed_run"] == 0  # the crash left nothing

    monkeypatch.setattr(kg_index, "persist_keys", real)
    fire.commit(ci)  # the replayed session takes the same run

    async def dupes(driver):
        async with driver.session() as s:
            nodes = await s.run(
                "MATCH (n {group_id: $g}) WITH n.uuid AS u, count(*) AS c WHERE c > 1 RETURN u", g=ci.group
            )
            edges = await s.run(
                "MATCH ()-[e {group_id: $g}]->() WITH e.uuid AS u, count(*) AS c WHERE c > 1 RETURN u", g=ci.group
            )
            return [r["u"] async for r in nodes] + [r["u"] async for r in edges]

    assert neo(fire.target, dupes) == []
    assert neo(fire.target, lambda d: graphdb.label_counts(d, ci.group)) == ci.expected
    nodes, edges = json.loads(fire.snapshot())
    for props in [n[1] for n in nodes] + [e[3] for e in edges]:
        for field in (*UNION_FIELDS, "summary_line_ids"):  # grown lists: nothing appended twice
            held = props.get(field) or []
            assert len(held) == len(set(held)), (props["uuid"], field)
    assert fire.meta()["committed_run"] == 1

    once = fire.snapshot()
    assert fire.commit(ci)["already_committed"] is True  # a replay after the commit landed writes nothing
    assert fire.snapshot() == once


@pytest.mark.needs_neo4j
def test_commit_keys_equal_the_dry_index_keys(fire):
    fire.dry()
    keys = fire.json("index", index_stage.KEYS)
    ci = fire.commit_input()
    assert (ci.keys, ci.key_render, ci.key_vectors) == (keys["keys"], keys["key_render"], keys["vectors"])

    before = fire.snapshot()
    drifted = {u: f"{k} (drift)" for u, k in ci.keys.items()}
    drift = replace(ci, keys=drifted, key_vectors={**ci.key_vectors, **{k: vector(k) for k in drifted.values()}})
    with pytest.raises(commit.CommitRefused, match="key drift"):
        fire.commit(drift)
    assert fire.snapshot() == before

    fire.commit(ci)

    async def stored(driver):
        async with driver.session() as s:
            got = await s.run(
                "MATCH (m:Episodic {group_id: $g}) WHERE m.uuid IN $ids RETURN m.uuid AS u, m.search_key AS k, "
                "m.search_key_sha AS sha, m.search_key_recipe AS recipe, m.search_key_embedding AS v",
                g=ci.group,
                ids=ci.scope,
            )
            return {r["u"]: dict(r) async for r in got}

    held = neo(fire.target, stored)
    assert {u: h["k"] for u, h in held.items()} == keys["keys"]
    assert {u: h["sha"] for u, h in held.items()} == fire.json("index", "index.json")["key_sha256"]
    assert all(h["recipe"] == keys["recipe_tag"] and h["v"] == keys["vectors"][h["k"]] for h in held.values())


@pytest.mark.needs_neo4j
def test_commit_input_refuses_a_failed_dry_stage(fire):
    fire.dry()
    fire.commit_input()  # the run as it passed is committable
    with pytest.raises(commit.CommitRefused, match="at run 1, not 0"):  # the chain is not the run's predecessor
        commit.commit_input(
            fire.run_root, group=fire.group, run=1, chain=replace(fire.chain, run=1), generation=fire.generation
        )

    def refusal(key: str, name: str, change) -> str:
        path = fire.stage(key) / name
        kept = path.read_bytes()
        doc = json.loads(kept)
        change(doc)
        path.write_text(json.dumps(doc))
        try:
            with pytest.raises(commit.CommitRefused) as caught:
                fire.commit_input()
            return str(caught.value)
        finally:
            path.write_bytes(kept)

    # a write that failed its gate after its summary.json landed (run_write writes it before it raises)
    assert "append gate failed" in refusal("write", write.SUMMARY, lambda d: d["append"].update(passed=False))
    invalid = lambda d: d.update(validation_errors=1, validation_error_sample=["bad row"])  # noqa: E731
    assert "readback validation errors" in refusal("write", write.SUMMARY, invalid)
    # a rewrite that was not idempotent, or whose readback lost a row
    assert "dry rewrite" in refusal("write", write.RERUN, lambda d: d["idempotent"].update(passed=False))
    assert "dry rewrite" in refusal("write", write.RERUN, lambda d: d["readback"]["missing_nodes"].append("u1"))
    # an index that failed its gate
    assert "dry index" in refusal("index", "index.json", lambda d: d.update(passed=False))
    # a session that holds no messages: the already-committed check must never be vacuous
    empty = lambda d: d["append"]["forward"].update(episodic=[])  # noqa: E731
    assert "holds no messages" in refusal("write", write.SUMMARY, empty)
    fire.commit_input()


@pytest.mark.needs_neo4j
def test_commit_checks_refuse_with_the_group_unchanged(fire, monkeypatch):
    from memory_kernel.graph import kg_writer

    fire.dry()
    ci = fire.commit_input()
    before = fire.snapshot()
    real_write, real_read, real_tag = kg_writer.write_rows, kg_writer.read_rows, kg_index.recipe_tag

    passes: list[dict] = []

    async def moving(*args, **kwargs):  # step 3: a writer whose second pass still creates a row
        out = await real_write(*args, **kwargs)
        passes.append(out)
        return {**out, "nodes_created": 1} if len(passes) == 2 else out

    monkeypatch.setattr(kg_writer, "write_rows", moving)
    with pytest.raises(commit.CommitRefused, match="second time were not a no-op"):
        fire.commit(ci)
    assert fire.snapshot() == before
    monkeypatch.setattr(kg_writer, "write_rows", real_write)

    async def short(*args, **kwargs):  # step 6: a readback that misses a written row
        nodes, edges = await real_read(*args, **kwargs)
        return nodes[1:], edges

    monkeypatch.setattr(kg_writer, "read_rows", short)
    with pytest.raises(commit.CommitRefused, match="readback misses"):
        fire.commit(ci)
    assert fire.snapshot() == before
    monkeypatch.setattr(kg_writer, "read_rows", real_read)

    # step 6: the session keyed under another recipe than the bonfire's stored keys
    monkeypatch.setattr(kg_index, "recipe_tag", lambda render: f"{real_tag(render)}|other")
    with pytest.raises(commit.CommitRefused, match="Episodic keys carry"):
        fire.commit(ci)
    assert fire.snapshot() == before
    monkeypatch.setattr(kg_index, "recipe_tag", real_tag)
    assert fire.commit(ci)["already_committed"] is False


def test_graph_commit_rides_the_build_config_json(tmp_path):
    from memory_kernel.errors import BonfireError
    from memory_kernel.pipeline.config import BuildConfig

    base = {
        "sample_id": "conv-kiln",
        "conversation": tmp_path / "c.json",
        "group": "g",
        "workdir": tmp_path,
        "root": tmp_path,
    }
    assert "graph_commit" not in BuildConfig(**base).to_json()  # a 1.8 config is byte-identical
    doc = BuildConfig(**base, write_mode="append", graph_commit=True).to_json()
    assert doc["graph_commit"] is True
    assert BuildConfig.from_json(json.loads(json.dumps(doc)), secrets={}).graph_commit is True
    with pytest.raises(BonfireError, match="graph_commit with strict_db_snapshot"):  # its stages take no snapshot
        BuildConfig(**base, write_mode="append", graph_commit=True, strict_db_snapshot=True)
