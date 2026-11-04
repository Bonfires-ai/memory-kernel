"""One session per append run (kernel 2.0 keeps the pure parts): ``bonfire.plan_session`` / ``forward_input`` over a Chain
view and a continued boundary walk build the wires a cumulative full build would.

The reference is a cumulative full build of every session so far (``build_sources`` + ``walk`` over the whole
conversation, its chain = ``chain.from_bootstrap``); the session-only run starts from the full build of the sessions
before it, so no folder bonfire and no ``chain.advance`` (1.8) is involved: the 2.0 commit folds the same state in the
graph. Wires are compared as built, never sent: the boundary's (a fixed answer per message id), graph_picks' and ctx's.
Offline: no stage child, no engine, no Neo4j, no network."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_kernel.construction.mined_affordance_contract import canonical_json
from memory_kernel.graph.addressing import addressee_of
from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.pipeline import bonfire, chain, ledger, llm, runner
from memory_kernel.pipeline.config import BuildConfig
from memory_kernel.pipeline.stages import _antecedent_pick as ap
from memory_kernel.pipeline.stages import _boundary_walk as ub
from memory_kernel.pipeline.stages import _clause_choice as cc
from memory_kernel.pipeline.stages import _eventuality_ctx as cx
from memory_kernel.pipeline.stages import ingest
from memory_kernel.pipeline.stages._conversation import (
    build_prepared,
    build_sources,
    session_conversations,
    split_conversation,
)
from memory_kernel.pipeline.stages._g1_graph import _others, deixis_referent, event_rows, explicit_flags

SAMPLE, GROUP = "conv-loom", "g-loom"


def turn(n: int, i: int, at: str, text: str, who: str, **kw) -> dict:
    return {"speaker": who, "dia_id": f"D{n}:{i}", "text": text, "timestamp": at, **kw}


#: n -> (chat, turns). Two chats, interleaved: weave A1, dock B1, weave A2 (same UTC day as A1: its first message's
#: window reaches back into A1), dock B2 (the next day: a hard cut), weave A3 (the next day too). A1 carries an
#: addressed message, so the whole conversation resolves "you" through addressees.
SESSIONS: dict[int, tuple[str, list[dict]]] = {
    1: (
        "weave",
        [
            turn(1, 1, "2026-03-02T09:00:00+00:00", "The new warp is on the loom.", "Ines"),
            turn(1, 2, "2026-03-02T09:05:00+00:00", "Is it the indigo one?", "Oren", addressees=["Ines"]),
            turn(1, 3, "2026-03-02T09:12:00+00:00", "Yes, forty threads to the inch.", "Ines"),
        ],
    ),
    2: (
        "dock",
        [
            turn(2, 1, "2026-03-02T09:20:00+00:00", "The ferry leaves at noon today.", "Pell"),
            turn(2, 2, "2026-03-02T09:30:00+00:00", "Then the crates go on first.", "Quill"),
        ],
    ),
    3: (
        "weave",
        [
            turn(3, 1, "2026-03-02T11:00:00+00:00", "You said the indigo would bleed.", "Oren"),
            turn(3, 2, "2026-03-02T11:04:00+00:00", "You washed it twice, so it holds.", "Ines"),
            turn(3, 3, "2026-03-02T11:09:00+00:00", "Then the shawl can ship on Friday.", "Oren"),
        ],
    ),
    4: (
        "dock",
        [
            turn(4, 1, "2026-03-03T08:00:00+00:00", "You loaded the crates in the rain.", "Quill"),
            turn(4, 2, "2026-03-03T08:06:00+00:00", "The tarp kept them dry enough.", "Pell"),
        ],
    ),
    5: (
        "weave",
        [
            turn(5, 1, "2026-03-03T10:00:00+00:00", "The shawl shipped this morning.", "Ines"),
            turn(5, 2, "2026-03-03T10:03:00+00:00", "You should weave a second one.", "Oren"),
        ],
    ),
}


def item(n: int, sessions: dict = SESSIONS, handles: tuple[str, ...] = ()) -> dict:
    conv: dict = {"agent_handles": list(handles)} if handles else {}
    for k in range(1, n + 1):
        chat, turns = sessions[k]
        conv |= {
            f"session_{k}": copy.deepcopy(turns),
            f"session_{k}_date_time": "9:00 am on 2 March, 2026",
            f"session_{k}_conversation": chat,
        }
    return {"sample_id": SAMPLE, "conversation": conv, "qa": []}


def session_doc(n: int, sessions: dict = SESSIONS) -> dict:
    chat, turns = sessions[n]
    messages = [
        {
            "message_id": t["dia_id"],
            "speaker": t["speaker"],
            "text": t["text"],
            "timestamp": t["timestamp"],
            **({"addressees": t["addressees"]} if "addressees" in t else {}),
        }
        for t in turns
    ]
    return {"conversation_id": chat, "messages": messages}


def decide(step: dict, history: list) -> dict:
    """A fixed answer per message, whatever else is in the input (the archive replays it)."""
    shift = int(hashlib.sha256(step["message_id"].encode()).hexdigest(), 16) % 3 == 0
    return {
        "index": step["index"],
        "message_id": step["message_id"],
        "choice": "shift_topic" if shift else "continue_active",
    }


class Full:
    """The cumulative full build of sessions 1..n: its select rows and its boundary walk."""

    def __init__(self, n: int, cfg: BuildConfig, sessions: dict = SESSIONS, answer=None, handles=()) -> None:
        self.n, self.handles = n, tuple(handles)
        doc = item(n, sessions, self.handles)
        self.sources = build_sources(
            build_prepared(split_conversation(doc)[0]), {"bonfire_id": cfg.bonfire_id, "conversation_id": SAMPLE}
        )
        self.chats = session_conversations(doc["conversation"], SAMPLE)
        self.walk = ub.walk(
            self.sources,
            answer or decide,
            episode_prefix=str(cfg.episode_prefix),
            window_prefix=cfg.boundary_window_prefix,
            conversations=self.chats,
        )
        self.conversation_of = {s["message_id"]: self.chats[s["session_id"]] for s in self.sources}

    def of_session(self, n: int) -> list[dict]:
        return [s for s in self.sources if s["session_id"] == f"{SAMPLE}-session_{n}"]

    def rows_of(self, ids: list[str]) -> list[dict]:
        by_id = {r["message_id"]: r for r in self.walk["rows"]}
        return [by_id[m] for m in ids]

    def chain(self) -> chain.Chain:
        """``from_bootstrap`` of this build (a projection of only its Episodic and Saga rows: no entities)."""
        episodes = self.walk["episode_by_message"]
        nodes = [
            {
                "labels_neo4j": ["Saga"],
                "props": {"uuid": content_uuid("saga", GROUP, ep), "name": ep, "group_id": GROUP},
            }
            for ep in sorted(set(episodes.values()))
        ] + [
            {
                "labels_neo4j": ["Episodic"],
                "props": {"uuid": content_uuid("ep", GROUP, m), "message_id": m, "group_id": GROUP},
            }
            for m in episodes
        ]
        rows = {"nodes": nodes, "edges": [], "forward": {"minted": [], "routes": {}}}  # nothing minted
        walks = ub.walk_states(self.walk["rows"], self.conversation_of)
        return chain.from_bootstrap(
            rows,
            self.sources,
            episodes,
            self.conversation_of,
            {},
            walks=walks,
            sessions=self.n,  # the input's session lists, an empty one included (bootstrap_chain reads them so)
            agent_handles=self.handles,
        )


def cfg_of(tmp_path: Path) -> BuildConfig:
    return BuildConfig(
        sample_id=SAMPLE,
        conversation=tmp_path / "boot.json",
        group=GROUP,
        purpose="remember the weaving",
        workdir=tmp_path / "boot",
        root=tmp_path,
    )


@pytest.fixture
def world(tmp_path):
    """``make(n)``: the state of a bonfire holding sessions 1..n (the real ``from_bootstrap`` of that full build) and the
    build config it appends under."""

    def make(n: int, sessions: dict = SESSIONS, answer=None, handles=()) -> SimpleNamespace:
        base = cfg_of(tmp_path)
        return SimpleNamespace(base=base, state=Full(n, base, sessions, answer, handles).chain(), n=n)

    return make


def plan_of(w: SimpleNamespace, doc: dict, state=None):
    return bonfire.plan_session(doc, state or w.state, w.base, sample_id=SAMPLE, chat_tagged=True)


def step(
    w: SimpleNamespace, n: int, sessions: dict = SESSIONS, answer=None, tmp: Path | None = None
) -> SimpleNamespace:
    """One session-only run of session ``n`` on the state of sessions 1..n-1: its plan, forward input and boundary walk."""
    w = SimpleNamespace(base=w.base, state=Full(n - 1, w.base, sessions, answer or None).chain())
    plan = plan_of(w, session_doc(n, sessions))
    forward = bonfire.forward_input(w.state, plan, voyage_cache=(tmp or Path("/nonexistent")) / "cache.json")
    assert set(chain.from_projection_view(forward["chain"]).chats) == set(w.state.chats)  # the Chain view the run pins
    walked = ub.walk(
        plan.sources,
        answer or decide,
        episode_prefix=str(w.base.episode_prefix),
        window_prefix=w.base.boundary_window_prefix,
        continuation=forward["boundary"],
    )
    return SimpleNamespace(plan=plan, forward=forward, walked=walked, state=w.state)


def canonical(value) -> bytes:
    return canonical_json(value).encode()


@pytest.mark.parametrize("first", ["continue_active", "shift_topic"])
def test_the_first_message_continues_the_last_episode_or_opens_the_next(world, first):
    """Weave's A2 starts on A1's UTC day: its first message is a paid step whose answer either continues A1's last
    episode (its id goes on) or opens episode ``<prefix><bonfire episode count>``; later new episodes count on."""

    def answer(step_: dict, history: list) -> dict:
        choice = first if step_["message_id"] == "D3:1" else "continue_active"
        if step_["message_id"] == "D3:3":
            choice = "shift_topic"
        return {"index": step_["index"], "message_id": step_["message_id"], "choice": choice}

    w = world(2, answer=answer)
    ran = step(w, 3, answer=answer)
    before, last = ran.state.episodes, ran.state.chats["weave"].last_episode
    got = ran.walked["episode_by_message"]
    prefix = w.base.episode_prefix
    if first == "continue_active":
        assert got == {"D3:1": last, "D3:2": last, "D3:3": f"{prefix}{before}"}
    else:
        assert got == {"D3:1": f"{prefix}{before}", "D3:2": f"{prefix}{before}", "D3:3": f"{prefix}{before + 1}"}
    end = ran.walked["continuation"]
    assert end["opened"] == sorted(set(got.values()) - {last})
    assert (end["messages"], end["episode_start"], end["last_episode"]) == (6, 5, got["D3:3"])
    assert got == {m: Full(3, w.base, answer=answer).walk["episode_by_message"][m] for m in got}


def test_graph_picks_wires_equal_the_full_builds(world, tmp_path, monkeypatch):
    """graph_picks' wire carries the message's source sha (its pair) and its own units; the referent context is the
    message's earlier units and, only when ``PREVIOUS_MESSAGE_ANTECEDENTS`` is on, the previous message of the SAME
    session. Both are equal for every session-only message, with the flag on or off."""
    w = world(2)
    full = Full(5, w.base)
    monkeypatch.setattr(ap, "antecedent_candidates", lambda *_a, **_k: [{"text": "the warp", "span": [1]}])
    monkeypatch.setattr(cc, "PREVIOUS_MESSAGE_ANTECEDENTS", True)

    def capture(rows: list[dict], where: Path) -> Path:
        for s in rows:
            unit = where / f"{s['order']:04d}" / "unit-00.json"
            unit.parent.mkdir(parents=True)
            leaf = {"node_type": "leaf", "start": 0, "string": s["text"].split()[0], "node_id": 1}
            unit.write_text(json.dumps({"job": {"nodes": [leaf]}}))
        return where

    def wires(rows: list[dict], capture_dir: Path, ids: list[str]) -> list:
        by_order = {s["order"]: s for s in rows}
        out = []
        for s in rows:
            if s["message_id"] not in ids:
                continue
            context = cc._referent_context(capture_dir, f"{s['order']:04d}/unit-00.json", by_order)
            unit = {
                "unit_id": 0,
                "text": s["text"],
                "record_sha256": "r" * 64,
                "tokens": [{"index": 0, "text": "it"}],
                "parsed_evidence": {
                    "native_frame_candidates": [
                        {"frameElements": [{"indices": [0], "referent": {"status": "candidate"}}]}
                    ]
                },
            }
            pair = {"content_sha256": s["content_sha256"], "body": {"units": [unit]}}
            compiled = ap.pick_questions(pair, lambda _u, c=context: ([], c), batcher=llm.QuestionBatcher())
            out.append([compiled["batches"], [c["where"] for c in context]])
        return out

    full_capture = capture(full.sources, tmp_path / "full")
    for n in (3, 4, 5):
        ran = step(w, n, tmp=tmp_path)
        ids = [s["message_id"] for s in ran.plan.sources]
        got = wires(ran.plan.sources, capture(ran.plan.sources, tmp_path / f"s{n}"), ids)
        want = wires(full.sources, full_capture, ids)
        assert canonical(got) == canonical(want)
        assert got[0][1] == []  # the session's first message has no previous message of its session


def ctx_wires(sources: list[dict], ids: list[str], explicit: dict[str, bool]) -> dict[str, list]:
    """ctx's wires for ``ids``: graph_a's per-message records ("you" by ``addressee_of`` under ``explicit``, the
    per-message flags ``build_graph`` uses), then ``message_states`` and ``compile_message``."""
    speakers = _others(sources)
    records = []
    for s in sources:
        tokens = dict(enumerate(s["text"].split()))
        frame = {
            "frameName": "say.01",
            "frameEvokingElement": {"indices": [1]},
            "frameElements": [{"feRole": "arg0", "indices": [0]}],
        }
        other = addressee_of(
            s.get("addressees", []), s["speaker"], speakers[s["session_id"]], explicit=explicit[s["message_id"]]
        )
        events = event_rows([frame], tokens, speaker=s["speaker"], other=other, cards={}, tags={0: "PRP"})
        records.append(
            {
                **{k: s[k] for k in ("message_id", "session_id", "speaker", "text")},
                "unit": f"{s['order']:04d}/unit-00.json",
                "events": events,
            }
        )
    analysed = {f"{r['unit']}#0": {"excluded": False, "word": r["text"].split()[1]} for r in records}
    states = cx.message_states(records, analysed, {"say.01": {"gloss": "say (roles: arg0 = sayer)"}})
    batcher = llm.QuestionBatcher()
    return {mid: cx.compile_message(mid, state, batcher=batcher)["batches"] for mid, state in states if mid in ids}


def append_flags(ran: SimpleNamespace) -> dict[str, bool]:
    """graph_a's flags in the append run (``explicit_flags`` from the forward input)."""
    return explicit_flags(ran.plan.sources, append=True, explicit_since=ran.forward["explicit_since"])


def test_ctx_wires_equal_the_full_builds_once_the_history_is_addressed(world):
    """ctx's wire carries the predicates' role texts, whose "you" referent depends on whether the message is in
    explicit mode. A1 (order 1) carried addressees, so every appended message is explicit in the append (bonfire-wide,
    per message) as in the full build (conversation-wide): every wire is equal."""
    w = world(2)
    full = Full(5, w.base)
    want_flags = explicit_flags(full.sources)
    for n in (3, 4, 5):
        ran = step(w, n)
        ids = [s["message_id"] for s in ran.plan.sources]
        assert ran.forward["explicit_since"] == 1
        got = ctx_wires(ran.plan.sources, ids, append_flags(ran))
        assert got and canonical(got) == canonical(ctx_wires(full.sources, ids, want_flags))


#: The history carries no addressees; weave's next session addresses its THIRD message first.
UNADDRESSED = {
    1: (
        "weave",
        [
            turn(1, 1, "2026-03-02T09:00:00+00:00", "The new warp is on the loom.", "Ines"),
            turn(1, 2, "2026-03-02T09:05:00+00:00", "Is it the indigo one?", "Oren"),
        ],
    ),
    2: (
        "weave",
        [
            turn(2, 1, "2026-03-03T09:00:00+00:00", "You dyed the skeins last night.", "Oren"),
            turn(2, 2, "2026-03-03T09:04:00+00:00", "You hung them by the stove.", "Ines"),
            turn(2, 3, "2026-03-03T09:09:00+00:00", "You can take them down now.", "Oren", addressees=["Ines"]),
            turn(2, 4, "2026-03-03T09:12:00+00:00", "You fold them for the market.", "Ines"),
        ],
    ),
    3: ("weave", [turn(3, 1, "2026-03-04T09:00:00+00:00", "You sold four skeins today.", "Oren")]),
}


def test_a_session_whose_kth_message_is_first_addressed_differs_only_before_it(world):
    """The history has no addressees; the appended session's 3rd message is the bonfire's first addressed one. The
    full build is conversation-wide (every message explicit, so "you" without addressees resolves to nothing); the
    append is per message: messages 1..2 keep the two-party "you" (they are before the first addressed message), so
    exactly their ctx wires differ; from message 3 on, and in every later session, they are equal."""
    w = world(1, UNADDRESSED)
    assert w.state.explicit_since is None
    full = Full(3, w.base, UNADDRESSED)
    want_flags = explicit_flags(full.sources)
    assert set(want_flags.values()) == {True}
    ran = step(w, 2, sessions=UNADDRESSED)
    ids = [s["message_id"] for s in ran.plan.sources]
    assert ran.forward["explicit_since"] is None
    assert Full(2, w.base, UNADDRESSED).chain().explicit_since == ran.plan.sources[2]["order"]
    got, want = ctx_wires(ran.plan.sources, ids, append_flags(ran)), ctx_wires(full.sources, ids, want_flags)
    assert [m for m in ids if canonical(got[m]) != canonical(want[m])] == ["D2:1", "D2:2"]
    later = step(w, 3, sessions=UNADDRESSED)
    assert later.forward["explicit_since"] == ran.plan.sources[2]["order"]
    assert canonical(ctx_wires(later.plan.sources, ["D3:1"], append_flags(later))) == canonical(
        ctx_wires(full.sources, ["D3:1"], want_flags)
    )


def test_tail_is_pinned_at_one():
    """No per-message stage reads a message of another session (graph_picks' and ctx's previous message are
    same-session; graph_picks does not read it at all), so the only cross-session reader is
    the boundary walk: the chat's last message (day change, later part) plus the open episode (window and archived
    part), which the tail always keeps (``chain.tail_start``)."""
    assert chain.TAIL == 1
    assert cc.PREVIOUS_MESSAGE_ANTECEDENTS is False
    assert chain.tail_start({"messages": 10, "episode_start": 4, "cursor": 7}) == 4
    assert chain.tail_start({"messages": 10, "episode_start": 9, "cursor": 9}) == 9


#: A one-message session on a new UTC day for weave, then a new chat's first one-message session: neither has a
#: same-day step (no Jev call), so the boundary's stop-rule denominator is 0.
ONE_MESSAGE = {
    **{k: SESSIONS[k] for k in (1, 2)},
    3: ("weave", [turn(3, 1, "2026-03-05T09:00:00+00:00", "The second shawl is on the loom.", "Ines")]),
    4: ("yard", [turn(4, 1, "2026-03-05T10:00:00+00:00", "The timber came in this morning.", "Sela")]),
}


def run_boundary_stage(w: SimpleNamespace, n: int, sessions: dict, tmp_path: Path) -> dict:
    """The real ``select`` then ``boundary`` stage (live) of session ``n`` in a seeded run directory, with an inherited
    ``strict_gates`` and a sender that refuses every call (nothing may be sent)."""
    plan = plan_of(w, session_doc(n, sessions))
    run_root = tmp_path / "runs" / f"{n:04d}" / SAMPLE
    run_dir = run_root.parent
    run_dir.mkdir(parents=True)
    (run_dir / bonfire.RUN_CONVERSATION).write_text(json.dumps(plan.conversation))
    forward = bonfire.forward_input(w.state, plan, voyage_cache=run_dir / "cache.json")
    (run_dir / bonfire.FORWARD_INPUT).write_text(json.dumps(forward))
    shard = tmp_path / "shard"  # the bonfire's one shard (a hand-made, sealed one)
    ledger.make_execution(shard, round_id="r-loom", capture_sha="c" * 64, g0_sha="d" * 64)
    cfg = bonfire.run_config(
        w.base,
        run_root,
        conversation=run_dir / bonfire.RUN_CONVERSATION,
        forward_input=run_dir / bonfire.FORWARD_INPUT,
        new_messages=plan.new_messages,
        limit_base_usd=0,
        ceiling_usd=None,
        index_record=tmp_path / "index.json",
        shard_dir=shard,
        capture_cache=None,
        strict_gates=True,
    )
    ctx = runner.Ctx(cfg=cfg, concurrency=1)
    ctx.__dict__["sender"] = llm.RefusingSender()
    ingest.select(ctx)
    out = ingest.boundary(ctx, live=True)
    return {"out": out, "sent": ctx.sender.misses}


def test_an_appends_stop_rules_are_advisory_and_need_ten_steps_to_shift():
    two_shifts = [{"decision_origin": "jev_finite_choice", "choice": "shift_topic"}] * 2
    assert ub.check_stop_rules([], 0) == {
        "same_day_arrivals": 0,
        "answered": 0,
        "unresolved": 0,
        "shift_topic": 0,
        "unresolved_share": 0.0,
        "shift_share": 0.0,
        "fired": False,
    }
    assert ub.check_stop_rules(two_shifts, 2)["fired"] is True  # a full build
    assert ub.check_stop_rules(two_shifts, 2, min_shift_steps=ub.APPEND_MIN_SHIFT_STEPS)["fired"] is False
    ten = [{"decision_origin": "jev_finite_choice", "choice": "shift_topic"}] * 10
    assert ub.check_stop_rules(ten, 10, min_shift_steps=ub.APPEND_MIN_SHIFT_STEPS)["fired"] is True


def test_a_fired_stop_rule_never_stops_an_append_even_with_strict_gates(world, monkeypatch, tmp_path):
    real = ub.check_stop_rules
    monkeypatch.setattr(ub, "check_stop_rules", lambda *a, **k: {**real(*a, **k), "fired": True})
    w = world(2, ONE_MESSAGE)
    got = run_boundary_stage(w, 3, ONE_MESSAGE, tmp_path)  # strict_gates inherited: no StopRule, the gate still passes
    assert got["out"]["gate"]["pass"] is True and got["out"]["gate"]["values"]["stop_rules"]["fired"] is True


def test_an_unsplit_id_holding_a_hash_is_a_duplicate(world):
    """I2: the chain's exact ids count, not only the turn ids of split parts (``<id>#k``)."""
    from dataclasses import replace

    w = world(2)
    state = replace(w.state, message_ids={**w.state.message_ids, "loom#7": 0})
    doc = {**session_doc(3), "messages": [{**session_doc(3)["messages"][0], "message_id": "loom#7"}]}
    with pytest.raises(bonfire.SessionDuplicate):
        plan_of(w, doc, state)
    split = replace(w.state, message_ids={**w.state.message_ids, "D3:1#1": 0, "D3:1#2": 0})  # a split turn's parts
    with pytest.raises(bonfire.BonfireError, match="session overlaps"):
        plan_of(w, session_doc(3), split)


@pytest.mark.parametrize(
    ("word", "tag", "expected"),
    [
        ("mine", "NN", None),
        ("US", "NNP", None),
        ("I", "PRP", "Ana"),
        ("myself", "PRP", "Ana"),
        ("us", "PRP", "Ana and others"),
        ("ours", "PRP", "Ana and others"),
        ("you", "PRP", "Milo"),
        ("yourself", "PRP", "Milo"),
        ("it", "PRP", None),
        ("she", "PRP", None),
    ],
)
def test_deixis_is_the_pronouns_person(word, tag, expected):
    row = deixis_referent(word, tag, speaker="Ana", other="Milo")
    assert (row["text"] if row else None) == expected


@pytest.mark.parametrize(
    ("word", "tag", "asked"),
    [("it", "PRP", True), ("itself", "PRP", True), ("they", "PRP", True), ("theirs", "PRP", True),
     ("she", "PRP", False), ("him", "PRP", False), ("we", "PRP", False), ("it", "NN", False), ("that", "DT", False)],
)  # fmt: skip
def test_only_ungendered_third_person_pronouns_are_asked_about(word, tag, asked):
    from memory_kernel.pipeline.stages._referents import pronoun_roles

    leaf = {"node_type": "leaf", "start": 0, "string": word, "lex_class": tag}
    frame = {"frameElements": [{"indices": [0]}]}
    assert bool(pronoun_roles(frame, [leaf])) is asked
    if (
        tag == "PRP"
    ):  # a token whose morph has Person but no Gender agrees with the leaf (the table row's gender fills in)
        from types import SimpleNamespace

        from memory_kernel.grammar import pronouns

        person = str(pronouns.of_word(word).person)
        token = SimpleNamespace(tag_=tag, lower_=word, morph={"Person": [person]}, children=[])
        assert pronouns.of_token(token).ungendered_third is pronouns.of_leaf(word, tag).ungendered_third
