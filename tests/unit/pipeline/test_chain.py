"""Chain state (kernel 1.2, Task 1): the initial chain of a bootstrap (kernel 2.0 seeds its graph from it).

Fixture: ``project()`` arguments built by hand (the style of ``test_identity_projection``), two chats that interleave.
Invented names only. Records, in source order (message, chat, speaker, addressees):

    0  D1:1  k1  Bo   []        "Bo likes pears"
    1  D2:1  k2  Dov  [Ana]     "Dov likes pears"     (Ana is first named here, as an addressee: no MENTIONS edge)
    2  D1:2  k1  Ana  []        "Ana met Cy"
    3  D3:1  k1  Ana  []        "Ana likes pears"

Episodes (one counter across chats, in the order their first messages come): ep-1 = D1:1 D1:2 (k1), ep-2 = D2:1 (k2),
ep-3 = D3:1 (k1).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from memory_kernel.grammar.dependencies import wordnet_ready
from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.graph.g1_projection import project
from memory_kernel.pipeline import chain as chain_mod
from memory_kernel.pipeline.chain import (
    TAIL,
    ChainError,
    ChatState,
    Minted,
    bootstrap_chain,
    from_bootstrap,
)
from unit.graph._rows import (  # noqa: E402
    BOUNDARY_ROWS,
    CHATS,
    EPISODES,
    GROUP,
    MESSAGES,
    ROLESETS,
    WALKS,
    bootstrap_state,
    graph_records,
    projection,
    saga,
    sources,
)

# -- from_bootstrap ----------------------------------------------------------------------------------------------------


def test_entities_are_minted_in_the_order_of_their_first_mention():
    state = bootstrap_state()
    by_key = {m.key: m for m in state.entities.values()}
    assert set(by_key) == {"bo", "dov", "ana", "cy"}
    # Ana is first named by Dov's addressee (message 1: no MENTIONS edge), before her own message 2
    assert {k: m.order for k, m in by_key.items()} == {"bo": 0, "dov": 1, "ana": 1, "cy": 2}
    assert all(m.run == 0 for m in state.entities.values())
    for speaker in ("Bo", "Dov", "Ana"):
        uuid = content_uuid("ent", GROUP, speaker.lower())
        assert state.entities[uuid].key == speaker.lower() and state.entities[uuid].display == speaker
        assert state.entities[uuid].kind == "person" and state.entities[uuid].surfaces == (speaker,)
    cy = by_key["cy"]
    assert (cy.uuid, cy.kind, cy.display) == (content_uuid("ent", GROUP, "cy"), "entity", "Cy")


def test_per_chat_counters_tail_and_flags(monkeypatch):
    state = bootstrap_state()
    assert set(state.chats) == {"k1", "k2"}
    k1, k2 = state.chats["k1"], state.chats["k2"]
    assert (k1.episodes, k1.last_episode, k1.last_saga, k1.last_order) == (2, "ep-3", saga("ep-3"), 3)
    assert (k2.episodes, k2.last_episode, k2.last_saga, k2.last_order) == (1, "ep-2", saga("ep-2"), 1)
    assert state.explicit_since == 1  # the bonfire's first addressed message (D2:1, order 1), any chat
    assert state.sessions == 3 and state.agent_handles == []  # the sessions the rows name; no handles given
    handled = from_bootstrap(projection(), sources(), EPISODES, CHATS, ROLESETS, walks=WALKS, sessions=4,
                             agent_handles=["@loom", "@loom", "@dock"])  # fmt: skip
    assert (handled.sessions, handled.agent_handles) == (4, ["@loom", "@dock"])  # an empty 4th session counts
    # the tail is the open episode (k1's ep-3 starts at its third message), at least TAIL rows
    assert [r["message_id"] for r in k1.tail] == ["D3:1"] and k1.boundary == WALKS["k1"]
    assert [r["message_id"] for r in k2.tail] == ["D2:1"] and k2.boundary == WALKS["k2"]
    assert k1.roster == ["Bo", "Ana"] and k2.roster == ["Dov", "Ana"]
    assert (state.messages, state.episodes, state.run) == (4, 3, 0)
    assert state.message_ids == dict.fromkeys(CHATS, 0)
    assert state.same_as == []
    assert state.rolesets == {
        "like.01": {"gloss": "like (roles: arg0 = liker)", "examples": ["Bo: Bo likes pears"], "run": 0},
        "meet.01": {"gloss": "meet (roles: arg0 = meeter)", "examples": ["Ana: Ana met Cy"], "run": 0},
    }
    monkeypatch.setattr(chain_mod, "TAIL", 2)
    short = bootstrap_state().chats["k1"]
    assert [r["message_id"] for r in short.tail] == ["D1:2", "D3:1"]
    monkeypatch.setattr(chain_mod, "TAIL", 1)
    open_since_start = {**WALKS, "k1": {"messages": 3, "episode_start": 0, "cursor": 0}}
    long = from_bootstrap(projection(), sources(), EPISODES, CHATS, ROLESETS, walks=open_since_start).chats["k1"]
    assert [r["message_id"] for r in long.tail] == ["D1:1", "D1:2", "D3:1"]  # an open episode is kept whole


def test_the_tail_is_a_copy_of_the_select_rows():
    rows = sources()
    state = from_bootstrap(projection(), rows, EPISODES, CHATS, ROLESETS, walks=WALKS)
    rows[3]["text"] = "changed"
    assert state.chats["k1"].tail[0]["text"] == "Ana likes pears"


def test_entities_and_routes_are_the_projections_first_mint():
    """The chain takes the projection's ``forward`` as it is: a fuzzy merge the projection made is a route; an entity
    the first mint misses, or one minted after run 0, is refused."""
    records = graph_records()
    # D3:1 names "the Marigold Potttery Studio", D1:2 "the Marigold Pottery Studio": one node (fuzzy), two keys
    records[2]["events"][0]["roles"]["arg1"] = {"text": "the Marigold Pottery Studio", "span": [2, 3, 4, 5]}
    records[3]["events"][0]["roles"]["arg1"] = {"text": "the Marigold Potttery Studio", "span": [2, 3, 4, 5]}
    rows = projection(records)
    state = from_bootstrap(rows, sources(), EPISODES, CHATS, ROLESETS, walks=WALKS)
    (studio,) = [m for m in state.entities.values() if "studio" in m.key]
    other = ({"marigold pottery studio", "marigold potttery studio"} - {studio.key}).pop()
    assert state.routes == {other: studio.uuid} == rows["forward"]["routes"]
    assert list(state.entities.values()) == rows["forward"]["minted"]

    missing = {**rows, "forward": {**rows["forward"], "minted": rows["forward"]["minted"][1:]}}
    with pytest.raises(ChainError, match="not in the projection's first mint"):
        from_bootstrap(missing, sources(), EPISODES, CHATS, ROLESETS, walks=WALKS)
    late = [replace(m, run=1) for m in rows["forward"]["minted"]]
    with pytest.raises(ChainError, match="not minted at run 0"):
        from_bootstrap({**rows, "forward": {**rows["forward"], "minted": late}}, sources(), EPISODES, CHATS,
                       ROLESETS, walks=WALKS)  # fmt: skip
    with pytest.raises(ChainError, match="carries no forward"):
        from_bootstrap({k: v for k, v in rows.items() if k != "forward"}, sources(), EPISODES, CHATS, ROLESETS,
                       walks=WALKS)  # fmt: skip


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_identity_aliases_become_routes():
    records = [
        {
            "unit": "0000/unit-00.json",
            "message_id": "m1",
            "session_id": "s1",
            "speaker": "Ana",
            "text": "my name is Dee",
            "timestamp": "2026-10-14T00:00:00Z",
            "capture_status": "captured",
            "events": [
                {
                    "roleset": "be.01",
                    "anchor": [2],
                    "roles": {"arg1": {"text": "my name", "span": [0, 1]}, "arg2": {"text": "Dee", "span": [3]}},
                }
            ],
        },
        {
            "unit": "0000/unit-01.json",
            "message_id": "m2",
            "session_id": "s1",
            "speaker": "Bo",
            "text": "Dee likes pears",
            "timestamp": "2026-10-14T00:01:00Z",
            "capture_status": "captured",
            "events": [
                {
                    "roleset": "like.01",
                    "anchor": [1],
                    "roles": {"arg0": {"text": "Dee", "span": [0]}, "arg1": {"text": "pears", "span": [2]}},
                }
            ],
        },
    ]
    eventuality = {
        "0000/unit-00.json#0": {"label": "state", "p": {"state": 1.0}},
        "0000/unit-01.json#0": {"label": "occurrence", "p": {"occurrence": 1.0}},
    }
    rows = project(
        records,
        eventuality=eventuality,
        episodes={"m1": "ep-1", "m2": "ep-1"},
        group_id=GROUP,
        created_at="2026-10-14T00:00:00Z",
        ingest_pass="pass-1",
    )
    rows_in = [
        {"message_id": "m1", "order": 0, "speaker": "Ana", "addressees": [], "session_id": "s1"},
        {"message_id": "m2", "order": 1, "speaker": "Bo", "addressees": [], "session_id": "s1"},
    ]
    one_chat = {"m1": "k", "m2": "k"}
    walks = {"k": {"messages": 2, "episode_start": 0, "cursor": 0}}
    state = from_bootstrap(rows, rows_in, {"m1": "ep-1", "m2": "ep-1"}, one_chat, {}, walks=walks)
    ana = content_uuid("ent", GROUP, "ana")
    assert state.routes == {"dee": ana}
    assert state.entities[ana].surfaces == ("Ana", "Dee")
    assert "dee" not in {m.key for m in state.entities.values()}


def test_a_bootstrap_whose_inputs_disagree_is_refused():
    with pytest.raises(ChainError, match="different messages"):
        from_bootstrap(projection(), sources(), {**EPISODES, "D9:9": "ep-9"}, CHATS, ROLESETS, walks=WALKS)
    with pytest.raises(ChainError, match="Saga of episode 'ep-x' is not in the rows"):
        from_bootstrap(projection(), sources(), {**EPISODES, "D3:1": "ep-x"}, CHATS, ROLESETS, walks=WALKS)
    short = {**WALKS, "k1": {"messages": 2, "episode_start": 0, "cursor": 0}}
    with pytest.raises(ChainError, match="chat 'k1': the boundary walk does not cover its 3 messages"):
        from_bootstrap(projection(), sources(), EPISODES, CHATS, ROLESETS, walks=short)


# -- storage -----------------------------------------------------------------------------------------------------------


# -- reading a bootstrap root ------------------------------------------------------------------------------------------


def fake_root(tmp_path: Path) -> Path:
    from memory_kernel.pipeline import runner
    from memory_kernel.pipeline.config import BuildConfig

    cfg = BuildConfig(
        sample_id="conv-ember",
        conversation=tmp_path / "ignored.json",
        group=GROUP,
        purpose="remember the pottery",
        workdir=tmp_path / "boot",
        root=tmp_path,
    )
    runner.write_config(cfg)
    select = runner.path(cfg, "select")
    select.mkdir(parents=True)
    (select / "sources.json").write_text(json.dumps(sources()))
    conv = {f"session_{n}": [] for n in (1, 2, 3)} | {"agent_handles": []}
    (select / f"{cfg.stem}-conversation-only.json").write_text(json.dumps({"conversation": conv}))
    events = runner.path(cfg, "events")
    events.mkdir(parents=True)
    (events / "eventuality.json").write_text(json.dumps({"rolesets": ROLESETS}))
    boundary = runner.path(cfg, "boundary")
    boundary.mkdir(parents=True)
    (boundary / "applied-boundaries.json").write_text(json.dumps({"rows": BOUNDARY_ROWS}))
    return cfg.conv_root


def test_bootstrap_chain_reads_the_root_and_projects_again(tmp_path, monkeypatch):
    from memory_kernel.pipeline.stages import write

    root = fake_root(tmp_path)
    seen = []

    def project_again(ctx):
        seen.append(ctx.cfg)
        return {"episodes": EPISODES, "conversation_of": CHATS}, {}, projection()

    monkeypatch.setattr(write, "project", project_again)
    assert bootstrap_chain(root) == bootstrap_state()
    cfg = seen[0]
    assert cfg.conv_root == root  # rooted at the bootstrap itself, wherever it was built
    assert cfg.conversation == root / "01-select-001" / f"{cfg.stem}-conversation-only.json"


def test_bootstrap_chain_refuses_a_root_that_misses_an_input(tmp_path, monkeypatch):
    from memory_kernel.pipeline.stages import write

    root = fake_root(tmp_path)
    monkeypatch.setattr(
        write, "project", lambda _ctx: ({"episodes": EPISODES, "conversation_of": CHATS}, {}, projection())
    )
    (root / "10-events-001" / "eventuality.json").unlink()
    with pytest.raises(ChainError, match="cannot derive the chain state"):
        bootstrap_chain(root)
    with pytest.raises(ChainError, match="no readable logs/mk/build-config.json"):
        bootstrap_chain(tmp_path / "nowhere")


# -- the real path: write.project -> from_bootstrap on a small root --------------------------------------------------------


def spacy_ready() -> bool:
    try:
        import spacy

        return spacy.util.is_package("en_core_web_sm")
    except ImportError:
        return False


NEEDS_NLP = pytest.mark.skipif(
    not (spacy_ready() and wordnet_ready()),
    reason="the projection needs the spaCy model en_core_web_sm and the WordNet corpus (mk setup --component ...)",
)

FRAMES = """<frameset><predicate lemma="{lemma}"><roleset id="{lemma}.01" name="{lemma}"><roles>
<role n="0" descr="agent"/><role n="1" descr="theme"/></roles></roleset></predicate></frameset>"""


def projectable_root(tmp_path: Path) -> Path:
    """A sealed-enough bootstrap root for ``write.project``: the stage outputs it reads (graph, eventuality,
    boundaries), a conversation-only copy, select sources and a two-frame PropBank dir.
    The conversation: sessions 1 and 3 in chat k1, session 2 in chat k2, the messages of ``MESSAGES``."""
    from memory_kernel.construction.mined_affordance_contract import content_sha256
    from memory_kernel.pipeline import runner
    from memory_kernel.pipeline.config import BuildConfig
    from memory_kernel.pipeline.stages._conversation import build_prepared, build_sources, split_conversation

    frames = tmp_path / "frames" / "frames"
    frames.mkdir(parents=True)
    for lemma in ("like", "meet"):
        (frames / f"{lemma}.xml").write_text(FRAMES.format(lemma=lemma))
    sample = "conv-ember"
    sessions = {"D1:1": 1, "D1:2": 1, "D2:1": 2, "D3:1": 3}
    chat_of_session = {1: "k1", 2: "k2", 3: "k1"}
    conv: dict = {}
    for n in (1, 2, 3):
        conv[f"session_{n}_date_time"] = f"9:0{n} am on 3 June, 2026"
        conv[f"session_{n}_conversation"] = chat_of_session[n]
        conv[f"session_{n}"] = [
            {
                "speaker": m[3],
                "dia_id": m[0],
                "text": m[5],
                "timestamp": f"2026-06-03T09:0{n}:00+00:00",
                **({"addressees": m[4]} if m[4] else {}),
            }
            for m in MESSAGES
            if sessions[m[0]] == n
        ]
    conv_only = split_conversation({"sample_id": sample, "conversation": conv, "qa": []})[0]
    rows = build_sources(build_prepared(conv_only), {"bonfire_id": "bf-ember", "conversation_id": sample})
    cfg = BuildConfig(
        sample_id=sample,
        conversation=tmp_path / "ignored.json",
        group=GROUP,
        purpose="remember the pottery",
        workdir=tmp_path / "boot",
        root=tmp_path,
        frame_dirs=(frames,),
        ingest_pass="pass-1",
    )
    runner.write_config(cfg)
    put = lambda key, name, doc: (  # noqa: E731
        runner.path(cfg, key).mkdir(parents=True, exist_ok=True),
        (runner.path(cfg, key) / name).write_text(json.dumps(doc)),
    )
    put("select", "sources.json", rows)
    put("select", f"{cfg.stem}-conversation-only.json", conv_only)
    by_id = {m[0]: m for m in MESSAGES}
    records = [
        {
            "unit": f"0000/unit-{i:02d}.json",
            "message_id": r["message_id"],
            "session_id": r["session_id"],
            "speaker": r["speaker"],
            "text": r["text"],
            "timestamp": r["timestamp"],
            "capture_status": "captured",
            "addressees": r["addressees"],
            "events": [
                {
                    "roleset": by_id[r["message_id"]][6],
                    "anchor": by_id[r["message_id"]][7],
                    "roles": {k: {"text": t, "span": sp} for k, (t, sp) in by_id[r["message_id"]][8].items()},
                }
            ],
        }
        for i, r in enumerate(rows)
    ]
    put("graph_a", "graph.json", {"records": records})
    predicates = {
        f"{r['unit']}#0": {"excluded": False, "label": "occurrence", "p_yes": {"occurrence": 1.0}} for r in records
    }
    ctx_doc = {"predicates": predicates, "distribution_predicates": {"occurrence": len(records)}}
    put("ctx", "eventuality.json", ctx_doc | {"content_sha256": content_sha256(ctx_doc)})
    # the boundary stage's episode map: one episode per chat run, ids in order of first message
    episodes = {"D1:1": "ep-1", "D1:2": "ep-1", "D2:1": "ep-2", "D3:1": "ep-3"}
    applied = {"episode_by_message": {r["message_id"]: episodes[r["message_id"]] for r in rows}, "rows": BOUNDARY_ROWS}
    put("boundary", "applied-boundaries.json", applied | {"content_sha256": content_sha256(applied)})
    put("events", "eventuality.json", {"rolesets": ROLESETS})
    return cfg.conv_root


@NEEDS_NLP
def test_the_real_projection_path_gives_a_chain(tmp_path):
    """No patching: ``write.project`` -> ``from_bootstrap`` on a small root, as ``mk bonfire init`` runs it."""
    root = projectable_root(tmp_path)
    state = bootstrap_chain(root)
    assert state.messages == 4 and set(state.message_ids) == set(CHATS) and set(state.chats) == {"k1", "k2"}
    assert (state.episodes, state.run) == (3, 0)
    keys = {m.key for m in state.entities.values()}
    assert {"bo", "dov", "ana", "cy"} <= keys
    k1 = state.chats["k1"]
    assert (k1.episodes, k1.last_episode, k1.last_order) == (2, "ep-3", 3)
    assert k1.last_saga == saga("ep-3") and [r["message_id"] for r in k1.tail] == ["D3:1"]
    assert k1.boundary == WALKS["k1"]
    assert state.explicit_since == 2 and state.sessions == 3
    assert set(state.rolesets) == {"like.01", "meet.01"}
    # the evidence holds exactly the first and last mention of each ground row, as select rows' five fields
    ends = {m for g in state.ground.values() for m in (g.first_mid, g.last_mid)}
    assert ends and set(state.evidence) == ends
    sources = {r["message_id"]: r for r in json.loads((root / "01-select-001" / "sources.json").read_text())}
    assert all(row == chain_mod.evidence_row(sources[m]) for m, row in state.evidence.items())
    assert set(next(iter(state.evidence.values()))) == set(chain_mod.EVIDENCE_FIELDS)
    # the data keys it reads from the projection's loaded inputs
    from memory_kernel.pipeline.runner import Ctx
    from memory_kernel.pipeline.stages import write

    data, _, rows = write.project(Ctx(cfg=chain_mod._bootstrap_config(root), concurrency=1))
    assert (
        set(data["episodes"]) == set(CHATS) == set(data["conversation_of"]) and data["conversation_of"]["D2:1"] == "k2"
    )
    assert {n["props"]["name_key"] for n in rows["nodes"] if "name_key" in n["props"]} == keys


@NEEDS_NLP
def test_an_append_run_projects_against_the_chain_its_forward_input_pins(tmp_path):
    """``write.project`` of a run whose forward input pins a chain view resolves against it (``chain=``): the
    bootstrap's own messages, projected again against the bootstrap's chain, mint nothing and re-key nothing."""
    from memory_kernel.pipeline.runner import Ctx
    from memory_kernel.pipeline.stages import write

    root = projectable_root(tmp_path)
    state = bootstrap_chain(root)
    view = chain_mod.projection_view(state)
    assert chain_mod.from_projection_view(json.loads(json.dumps(view))).entities == state.entities
    pinned = tmp_path / "forward.json"
    chat = next(iter(state.chats))
    pinned.write_text(json.dumps({"chain": view, "chat": chat, "explicit_since": state.explicit_since}))
    cfg = replace(chain_mod._bootstrap_config(root), forward_input=pinned)
    data, _, rows = write.project(Ctx(cfg=cfg, concurrency=1), referee=None)  # no referents stage ran here
    assert rows["forward"]["minted"] == [] and rows["forward"]["routes"] == {}
    # the write's forward gate knows the row kinds a real projection sends (Events, facts, sagas):
    # rule 1 passes with the run's own uuids, and rule 2 finds no reused entity row that would change what the full
    # build stored
    episodes = sorted(set(data["episodes"].values()))
    own = write.own_uuids(rows["forward"], data["records"], episodes, GROUP)
    write.forward_gate(rows, [], [], data["chain"], own, chat=chat)
    kinds = {label for n in rows["nodes"] for label in n["labels_neo4j"]}
    assert {"Event", "Saga", "Episodic"} <= kinds and {"RELATES_TO", "MENTIONS"} <= {e["type"] for e in rows["edges"]}
    _, _, full = write.project(Ctx(cfg=chain_mod._bootstrap_config(root), concurrency=1))
    stored = {n["props"]["uuid"]: n for n in full["nodes"]}
    reused = [n for n in rows["nodes"] if n["props"]["uuid"] in state.entities]
    assert reused and all(write._frozen_changes(n, stored[n["props"]["uuid"]]) == [] for n in reused)
    entities = {n["props"]["uuid"] for n in rows["nodes"] if "Entity" in n["labels_neo4j"]} - {
        n["props"]["uuid"] for n in rows["nodes"] if "Event" in n["labels_neo4j"]
    }
    assert entities and entities <= set(state.entities)
    assert not [n for n in rows["nodes"] if n["props"]["uuid"] in entities and "name_key" in n["props"]]
    # the mint run follows the pinned chain (its last run + 1), never the run directory: a replay mints the same
    other = tmp_path / "other"
    other.mkdir()
    (other / "forward.json").write_text(json.dumps({"chain": view, "chat": chat, "run": 99}))
    data, _, again = write.project(
        Ctx(cfg=replace(cfg, forward_input=other / "forward.json"), concurrency=1), referee=None
    )
    assert data["run"] == state.run + 1 and again["forward"] == rows["forward"]
    pinned.write_text(json.dumps({"run": 7}))
    with pytest.raises(write.Refused, match="carries no chain view"):
        write.project(Ctx(cfg=cfg, concurrency=1))


def test_the_dataclasses_are_the_documented_shapes():
    assert [f for f in Minted.__dataclass_fields__] == ["uuid", "key", "kind", "display", "surfaces", "run", "order"]
    assert [f for f in ChatState.__dataclass_fields__] == [
        "episodes",
        "last_episode",
        "last_saga",
        "last_order",
        "roster",
        "tail",
        "boundary",
    ]
    assert TAIL == 1
