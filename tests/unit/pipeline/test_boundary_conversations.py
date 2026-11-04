"""The boundary walk runs one independent walk per conversation (chat): sessions of two chats may overlap in time."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_kernel.construction.mined_affordance_contract import content_sha256
from memory_kernel.pipeline.config import ConversationError, load_conversation
from memory_kernel.pipeline.stages import _boundary_walk as ub
from memory_kernel.pipeline.stages._conversation import (
    build_prepared,
    build_sources,
    session_conversations,
    split_conversation,
)
from memory_kernel.pipeline.stages._g1_build import check_episode_map


def turn(session: int, i: int, at: str, text: str, who: str = "Ana") -> dict:
    return {"speaker": who, "dia_id": f"D{session}:{i}", "text": text, "timestamp": f"2026-05-01T{at}:00+00:00"}


def two_chats(tagged: bool = True) -> dict:
    """c1 10:00-11:00, c2 10:30-10:45, c1 again at 12:00 (sessions interleaved in input order)."""
    c = {
        "session_1_date_time": "10:00 am on 1 May, 2026",
        "session_1": [
            turn(1, 1, "10:00", "Shall we repaint the shed this weekend?"),
            turn(1, 2, "10:20", "Yes, the green one from the old store.", "Bo"),
            turn(1, 3, "11:00", "I will buy two cans tomorrow."),
        ],
        "session_2_date_time": "10:30 am on 1 May, 2026",
        "session_2": [
            turn(2, 1, "10:30", "Is the choir rehearsal still on Friday?", "Cy"),
            turn(2, 2, "10:45", "Yes, seven in the evening.", "Di"),
        ],
        "session_3_date_time": "12:00 pm on 1 May, 2026",
        "session_3": [
            turn(3, 1, "12:00", "The store closes early on Saturday.", "Bo"),
            turn(3, 2, "12:05", "Then I go on Friday."),
        ],
    }
    if tagged:
        c.update(session_1_conversation="c1", session_2_conversation="c2", session_3_conversation="c1")
    return {"sample_id": "s1", "conversation": c, "qa": []}


def sources_of(item: dict) -> list[dict]:
    return build_sources(build_prepared(split_conversation(item)[0]), {"bonfire_id": "b1", "conversation_id": "s1"})


def conversation_map(item: dict) -> dict[str, str]:
    return session_conversations(item["conversation"], item["sample_id"])


def recording_walk(sources: list[dict], conversations, shift_every: int = 0) -> tuple[dict, list[dict]]:
    steps: list[dict] = []

    def decide(step: dict, history: list) -> dict:
        steps.append(step)
        shift = bool(shift_every) and len(steps) % shift_every == 0
        return {
            "index": step["index"],
            "message_id": step["message_id"],
            "choice": "shift_topic" if shift else "continue_active",
        }

    result = ub.walk(sources, decide, episode_prefix="ep-", window_prefix="w", conversations=conversations)
    return result, steps


# -- the walk --------------------------------------------------------------------------------------------------------


def test_interleaved_chats_walk_apart():
    item = two_chats()
    sources = sources_of(item)
    conversations = conversation_map(item)
    plan = ub.plan(sources, episode_prefix="ep-", window_prefix="w", conversations=conversations)
    assert plan["arrivals"] == 7
    assert plan["same_day_requests"] == 5  # 7 arrivals minus the first message of each chat
    result, steps = recording_walk(sources, conversations)
    conv_of = {s["message_id"]: conversations[s["session_id"]] for s in sources}
    episodes: dict[str, set[str]] = {}
    for message_id, episode in result["episode_by_message"].items():
        episodes.setdefault(episode, set()).add(conv_of[message_id])
    assert all(len(convs) == 1 for convs in episodes.values())  # no episode spans two chats
    assert len(episodes) == result["episodes"] == 2  # all-continue: one episode per chat, distinct ids
    first_c2 = next(r for r in result["rows"] if conv_of[r["message_id"]] == "c2")
    assert first_c2["choice"] == "initial" and first_c2["request_sha256"] is None  # no paid call
    assert first_c2["episode_id"] != result["episode_by_message"]["D1:1"]
    # rows come back in global source order, with global indices
    assert [r["message_id"] for r in result["rows"]] == [s["message_id"] for s in sources]
    assert [r["index"] for r in result["rows"]] == list(range(len(sources)))
    # a wire carries messages of its own chat only
    ids = {s["message_id"]: conv_of[s["message_id"]] for s in sources}
    for step in steps:
        own = ids[step["message_id"]]
        text = json.dumps(step["wire"])
        assert all(f'"{m}"' not in text for m, c in ids.items() if c != own)


def test_episode_ids_unique_and_ordered_by_first_message():
    item = two_chats()
    sources = sources_of(item)
    result, _ = recording_walk(sources, conversation_map(item), shift_every=2)
    first_seen: list[str] = []
    for r in result["rows"]:
        if r["episode_id"] not in first_seen:
            first_seen.append(r["episode_id"])
    assert first_seen == [f"ep-{i}" for i in range(result["episodes"])]
    assert check_episode_map(
        sealed(result["episode_by_message"]),
        set(result["episode_by_message"]),
        messages=len(sources),
        episodes=result["episodes"],
        rank={s["message_id"]: s["order"] for s in sources},
        conversation_of={s["message_id"]: conversation_map(item)[s["session_id"]] for s in sources},
    )


def test_untagged_overlap_still_refused():
    """The partition is opt-in: without session_N_conversation the input is one chat, and it goes back in time."""
    item = two_chats(tagged=False)
    sources = sources_of(item)
    with pytest.raises(ValueError, match="reverse chronological order"):
        ub.plan(sources, episode_prefix="ep-", window_prefix="w", conversations=conversation_map(item))
    with pytest.raises(ValueError, match="reverse chronological order"):
        ub.plan(sources, episode_prefix="ep-", window_prefix="w")


# -- single conversation: byte-identical to 14b27ae0 -------------------------------------------------------------------

#: sha256 of the wire digests, walk result and plan of ``single_chat`` computed at kernel 14b27ae0 (before the partition).
SINGLE_CHAT_DIGEST = "9edaa100924a8d64c66220188dc7ebb1130ad305d5e0b64318e8a8eb0fa3d06a"


def single_chat(conversation: str | None = None) -> dict:
    def t(i, s, at, text, who="Ana"):
        return {"speaker": who, "dia_id": f"D{s}:{i}", "text": text, "timestamp": at}

    c = {
        "session_1_date_time": "10:00 am on 1 May, 2026",
        "session_1": [
            t(1, 1, "2026-05-01T10:00:00+00:00", "We should plan the garden this weekend."),
            t(2, 1, "2026-05-01T10:05:00+00:00", "Sure, tomatoes and basil again?", "Bo"),
            t(3, 1, "2026-05-01T10:20:00+00:00", "A long note about the soil. " * 40),
            t(4, 1, "2026-05-01T10:40:00+00:00", "Also my sister moved to Lisbon.", "Bo"),
        ],
        "session_2_date_time": "9:00 am on 2 May, 2026",
        "session_2": [
            t(1, 2, "2026-05-02T09:00:00+00:00", "Morning! The seeds arrived."),
            t(2, 2, "2026-05-02T09:10:00+00:00", "Great, let us start the trays.", "Bo"),
            t(3, 2, "2026-05-02T11:00:00+00:00", "Unrelated: the car needs service."),
        ],
    }
    if conversation is not None:
        c.update(session_1_conversation=conversation, session_2_conversation=conversation)
    return {"sample_id": "s1", "conversation": c, "qa": []}


def single_chat_digest(item: dict, conversations) -> str:
    sources = sources_of(item)
    wires: list[list[str]] = []

    def decide(step: dict, history: list) -> dict:
        wires.append([step["request_sha256"], step["content_sha256"]])
        shift = len(wires) % 3 == 0
        return {
            "index": step["index"],
            "message_id": step["message_id"],
            "choice": "shift_topic" if shift else "continue_active",
        }

    kw = {} if conversations is None else {"conversations": conversations}
    res = ub.walk(sources, decide, episode_prefix="ep-", window_prefix="w", **kw)
    plan = ub.plan(sources, episode_prefix="ep-", window_prefix="w", **kw)
    return hashlib.sha256(json.dumps({"wires": wires, "walk": res, "plan": plan}, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("conversation", [None, "only-chat"])
def test_single_conversation_is_byte_identical(conversation):
    item = single_chat(conversation)
    assert single_chat_digest(item, None if conversation is None else conversation_map(item)) == SINGLE_CHAT_DIGEST
    assert single_chat_digest(item, conversation_map(item)) == SINGLE_CHAT_DIGEST


# -- the conversation input ------------------------------------------------------------------------------------------


def test_session_conversations_default_to_one():
    item = two_chats(tagged=False)
    assert conversation_map(item) == {f"s1-session_{i}": "s1" for i in (1, 2, 3)}
    assert conversation_map(two_chats()) == {"s1-session_1": "c1", "s1-session_2": "c2", "s1-session_3": "c1"}


def test_conversation_key_loads(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(two_chats()))
    item, _ = load_conversation(p, None)
    assert item["conversation"]["session_2_conversation"] == "c2"


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("session_1_conversation", "", "non-empty string"),
        ("session_1_conversation", 7, "non-empty string"),
        ("session_9_conversation", "c1", "no session_9"),
    ],
)
def test_bad_conversation_key_refused(tmp_path: Path, key, value, match):
    item = two_chats()
    item["conversation"][key] = value
    p = tmp_path / "c.json"
    p.write_text(json.dumps(item))
    with pytest.raises(ConversationError, match=match):
        load_conversation(p, None)


def test_a_partly_tagged_input_is_refused_naming_the_untagged_sessions(tmp_path: Path):
    item = two_chats()
    del item["conversation"]["session_2_conversation"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(item))
    with pytest.raises(ConversationError, match=r"\['session_2'\] do not: tag every session or none"):
        load_conversation(p, None)
    p.write_text(json.dumps(two_chats(tagged=False)))  # none tagged: one conversation, as 1.0.0
    assert load_conversation(p, None)[0]["sample_id"] == "s1"


# -- the episode map check -------------------------------------------------------------------------------------------


def sealed(mapping: dict) -> dict:
    body = {"episode_by_message": mapping}
    return {**body, "content_sha256": content_sha256(body)}


RANK = {"a1": 0, "b1": 1, "a2": 2, "b2": 3, "a3": 4}
CONV = {"a1": "c1", "a2": "c1", "a3": "c1", "b1": "c2", "b2": "c2"}


def test_episode_contiguous_within_its_chat_passes():
    mapping = {"a1": "e0", "b1": "e1", "a2": "e0", "b2": "e1", "a3": "e2"}
    assert check_episode_map(sealed(mapping), set(mapping), messages=5, episodes=3, rank=RANK, conversation_of=CONV)
    with pytest.raises(ValueError, match="not contiguous"):  # without the chats it is interleaved in global order
        check_episode_map(sealed(mapping), set(mapping), messages=5, episodes=3, rank=RANK)


def test_episode_split_within_its_chat_fails():
    mapping = {"a1": "e0", "b1": "e1", "a2": "e2", "b2": "e1", "a3": "e0"}
    with pytest.raises(ValueError, match="e0 is not contiguous"):
        check_episode_map(sealed(mapping), set(mapping), messages=5, episodes=3, rank=RANK, conversation_of=CONV)


def test_episode_spanning_chats_fails():
    mapping = {"a1": "e0", "b1": "e0", "a2": "e0", "b2": "e0", "a3": "e0"}
    with pytest.raises(ValueError, match="spans conversations"):
        check_episode_map(sealed(mapping), set(mapping), messages=5, episodes=1, rank=RANK, conversation_of=CONV)


# -- the stage passes the partition ----------------------------------------------------------------------------------


def test_boundary_stage_plans_per_conversation(tmp_path: Path):
    from memory_kernel.pipeline.stages import ingest

    item = two_chats()
    conv_path = tmp_path / "c.json"
    conv_path.write_text(json.dumps(item))
    select = tmp_path / "select"
    select.mkdir()
    (select / "sources.json").write_text(json.dumps(sources_of(item)))
    plans: dict = {}
    cfg = SimpleNamespace(
        sample_id="s1",
        conversation=conv_path,
        episode_prefix="ep-",
        boundary_window_prefix="w",
        boundary_replay_archive=None,
    )
    ctx = SimpleNamespace(
        cfg=cfg,
        messages=7,
        path=lambda key: {"select": select, "boundary": tmp_path / "boundary"}[key],
        save_plan=lambda name, plan: plans.update({name: plan}),
    )
    out = ingest.boundary(ctx, live=False)
    assert out["plan"]["same_day_requests"] == 5
    assert out["plan"]["conversations"] == 2


# -- the wire limit: an over-limit window keeps its most recent messages that fit -------------------------------------


def test_an_over_limit_window_keeps_its_most_recent_messages_and_a_window_within_is_unchanged():
    """One long same-day episode: the frozen compiler's window can reach 32,704 bytes, the ledger refuses a wire over
    31,744. ``boundary_wire`` then keeps the most recent messages that fit (a later working-window start, the same
    incoming message, deterministic); a step within the limit is the frozen step as it was."""
    from memory_kernel.construction.memory_jev_wire_ids import MAX_WIRE_BYTES
    from memory_kernel.construction.mined_affordance_contract import canonical_json

    words = ["kiln", "glaze", "ember", "shed", "choir", "lantern", "harbour", "copper"]
    turns = [
        turn(1, i, f"10:{i:02d}", " ".join(words[(i + k) % 8] for k in range(300 + 37 * (i % 11))))
        for i in range(1, 22)
    ]
    sources = sources_of(
        {"sample_id": "s1", "conversation": {"session_1_date_time": "10:00 am on 1 May, 2026", "session_1": turns}}
    )
    timestamps, scope = ub.timestamp_rows(sources), ub.window_scope(sources)
    trimmed = within = 0
    for index in range(1, len(sources)):
        history = [{"index": p, "choice": "continue_active"} for p in range(1, index)]
        args = {
            "index": index,
            "history": history,
            "previous_cursor": 0,
            "scope": scope,
            "window_prefix": "full-active",
        }
        frozen = ub.boundary_wire(sources, timestamps, fit=False, **args)
        step = ub.boundary_wire(sources, timestamps, **args)
        assert len(canonical_json(step["wire"]).encode()) <= MAX_WIRE_BYTES
        if len(canonical_json(frozen["wire"]).encode()) <= MAX_WIRE_BYTES:
            assert step == frozen  # within the limit: the frozen step, byte for byte
            within += 1
            continue
        trimmed += 1
        start = step["working_window_start_order"]
        assert step["window_trimmed"] and step["message_id"] == frozen["message_id"]  # the same incoming message
        assert start > frozen["working_window_start_order"]  # its earlier messages left out
        assert step == ub.boundary_wire(sources, timestamps, **args)  # same input, same wire
        one_earlier = ub.boundary_wire(sources, timestamps, **{**args, "previous_cursor": start - 1})
        assert one_earlier["working_window_start_order"] == start  # one more earlier message would not fit
    assert trimmed and within
    plan = ub.plan(sources, episode_prefix="ep-", window_prefix="full-active")
    assert not plan["wires_within_limit"] and "window_trimmed" not in plan  # the plan keeps today's figures


def test_an_arrival_that_cannot_fit_cuts_the_episode_and_the_walk_completes():
    """A long same-day episode carries the ids of its earlier messages in every step (``capacity``), so at some arrival
    not even the prior message plus the incoming one fits: the frozen compiler refuses that step
    (``window_capacity_pending``) or makes a wire the ledger refuses, either of which stopped the build. The walk then opens a new episode there with no call
    (``window_capacity_cut``) and goes on, deterministically."""
    from memory_kernel.construction.memory_jev_wire_ids import MAX_WIRE_BYTES
    from memory_kernel.construction.mined_affordance_contract import canonical_json
    from memory_kernel.pipeline.stages._boundary_frozen import FullIngestError

    turns = [
        {
            "speaker": "Ana",
            "dia_id": f"talk-{i:04d}-" + "segment" * 28,  # a long source id: the archived ids grow ~210 bytes a message
            "text": f"Point {i} of the talk about the kiln.",
            "timestamp": f"2026-05-01T{10 + i // 60:02d}:{i % 60:02d}:00+00:00",
        }
        for i in range(1, 200)
    ]
    sources = sources_of(
        {"sample_id": "s1", "conversation": {"session_1_date_time": "10:00 am on 1 May, 2026", "session_1": turns}}
    )
    sizes: list[int] = []

    def decide(step: dict, history: list) -> dict:
        sizes.append(len(canonical_json(step["wire"]).encode()))
        return {"index": step["index"], "message_id": step["message_id"], "choice": "continue_active"}

    walked = ub.walk(sources, decide, episode_prefix="ep-", window_prefix="full-active")
    cuts = [r for r in walked["rows"] if r["decision_origin"] == ub.CAPACITY_ORIGIN]
    assert cuts and all(r["choice"] == "shift_topic" and r["request_sha256"] is None for r in cuts)
    assert walked["episodes"] == len(cuts) + 1 and max(sizes) <= MAX_WIRE_BYTES
    first = cuts[0]["index"]  # today this step is refused: the frozen compiler raises, or the ledger refuses its wire
    try:
        frozen = ub.boundary_wire(
            sources,
            ub.timestamp_rows(sources),
            index=first,
            history=[{"index": p, "choice": "continue_active"} for p in range(1, first)],
            previous_cursor=walked["rows"][first - 1]["working_window_start_order"],
            scope=ub.window_scope(sources),
            window_prefix="full-active",
            fit=False,
        )
    except FullIngestError as exc:
        assert str(exc).startswith("window_capacity_pending")
    else:
        assert len(canonical_json(frozen["wire"]).encode()) > MAX_WIRE_BYTES
    again = ub.walk(sources, decide, episode_prefix="ep-", window_prefix="full-active")
    assert again == walked  # same input, same cuts
    assert ub.plan(sources, episode_prefix="ep-", window_prefix="full-active")["capacity_cuts"] >= 1
