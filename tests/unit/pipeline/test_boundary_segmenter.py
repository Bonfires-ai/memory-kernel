"""The ``segmenter`` boundary policy: each session is a hard boundary, cut inside by the embedding segmenter."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.pipeline import bonfire, chain, embed
from memory_kernel.pipeline.stages import write
from memory_kernel.pipeline.stages._boundary_segment import run_segment_boundary

_START = datetime(2026, 3, 2, 9, tzinfo=UTC)
_TOPICS = ("lanterns", "orchards", "ferries")


class _FakeVoyage:
    """Stands in for the Voyage client inside ``TextVoyageCache``: one unit vector per topic word in the text."""

    def __init__(self) -> None:
        self.calls = 0
        self.tokens = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.tokens += sum(len(t.split()) for t in texts)
        return [[1.0 if topic in text else 0.0 for topic in _TOPICS] for text in texts]


def _session(session_id: str, topics: list[str], first: int) -> list[dict]:
    rows = []
    for i, topic in enumerate(topics):
        n = first + i
        rows.append(
            {
                "message_id": f"m-{n}",
                "order": n,
                "session_id": session_id,
                "speaker": "Ilse" if i % 2 else "Oren",
                "text": f"more about {topic}, note {n}",
                "timestamp": (_START + timedelta(seconds=30 * n)).isoformat(),
                "attachments": [],
                "scope": {"bonfire_id": "b-test", "conversation_id": "c-test"},
            }
        )
    return rows


def _sources() -> list[dict]:
    short = _session("c-test-session_1", ["ferries"] * 30, 0)
    long = _session("c-test-session_2", ["lanterns"] * 200 + ["orchards"] * 200, 30)
    tiny = _session("c-test-session_3", ["ferries"] * 5, 430)  # too short to cut: the segmenter embeds none of it
    return short + long + tiny


def _cache(path, fake):
    cache = embed.TextVoyageCache(path, None)
    cache.inner = fake
    return cache


def _run(sources, out, cache):
    return run_segment_boundary(
        sources,
        out,
        cache=cache,
        episode_prefix="ep-",
        conversations=None,
        first_episode=5,
        unit_guard=lambda spend: None,
    )


def test_segmenter_boundary_maps_every_message_once(tmp_path):
    sources = _sources()
    cache = _cache(tmp_path / "voyage.json", _FakeVoyage())
    summary = _run(sources, tmp_path / "out", cache)
    applied = json.loads((tmp_path / "out" / "applied-boundaries.json").read_bytes())

    rows = applied["rows"]
    assert [r["message_id"] for r in rows] == [s["message_id"] for s in sources]
    assert applied["episode_by_message"] == {r["message_id"]: r["episode_id"] for r in rows}  # every message once
    session_of = {s["message_id"]: s["session_id"] for s in sources}
    members: dict[str, list[str]] = {}
    for r in rows:
        members.setdefault(r["episode_id"], []).append(r["message_id"])
    assert all(len({session_of[m] for m in ms}) == 1 for ms in members.values())  # never across sessions
    assert all(len(ms) <= 150 for ms in members.values())
    assert list(members) == [f"ep-{5 + k}" for k in range(len(members))]  # numbered from first_episode
    assert members["ep-5"] == [s["message_id"] for s in sources[:30]]

    assert [r["index"] for r in rows] == list(range(len(sources)))
    assert all(r["request_sha256"] is None and r["boundary_step_sha256"] is None for r in rows)
    assert all(r["boundary_history_row"] is None for r in rows)
    assert [r["choice"] for r in rows if r["choice"] != "continue_active"].count("initial") == 3
    opened = [r["episode_id"] for r in rows if r["choice"] in {"initial", "segment_cut"}]
    assert opened == list(members)
    assert applied["schema_version"] == "mk-utc-day-applied-boundaries/v1"

    assert summary["policy"] == "segmenter"
    assert summary["stop_rules"] == {"not_applicable": True, "fired": False}
    assert summary["episodes"] == len(members)
    assert summary["per_session"] == {
        "c-test-session_1": 1,
        "c-test-session_2": len(members) - 2,
        "c-test-session_3": 1,
    }
    assert set(summary["cut_reasons"]) <= {"time_gap", "shift", "centroid", "ceiling"}
    assert summary["applied_sha256"] == applied["content_sha256"]
    # the boundary embeds exactly the strings the events stage embeds, so events later hits the cache
    assert set(json.loads((tmp_path / "voyage.json").read_text())) == set(embed.message_texts(sources).values())


def test_segmenter_resume_sends_nothing(tmp_path):
    sources = _sources()
    _run(sources, tmp_path / "first", _cache(tmp_path / "voyage.json", _FakeVoyage()))
    again = _FakeVoyage()
    _run(sources, tmp_path / "second", _cache(tmp_path / "voyage.json", again))

    assert again.calls == 0
    first = (tmp_path / "first" / "applied-boundaries.json").read_bytes()
    assert (tmp_path / "second" / "applied-boundaries.json").read_bytes() == first


def test_segmenter_append_opens_new_sagas_and_continues_numbering(tmp_path):
    """A segmenter bonfire's chain is closed after every run: an appended session never grows the chat's last Saga (no
    ``prior``; its first segment links from it by NEXT_SAGA) and its episodes continue the bonfire's numbering."""
    group, chat, prefix = "g-test", "c-test", "ep-"
    boot = _session("c-test-session_1", ["ferries"] * 30, 0) + _session("c-test-session_2", ["orchards"] * 30, 30)
    cache = _cache(tmp_path / "voyage.json", _FakeVoyage())
    kw = {"cache": cache, "episode_prefix": prefix, "conversations": None, "unit_guard": lambda spend: None}
    run_segment_boundary(boot, tmp_path / "boot", first_episode=0, **kw)
    applied = json.loads((tmp_path / "boot" / "applied-boundaries.json").read_bytes())
    assert sorted(set(applied["episode_by_message"].values())) == ["ep-0", "ep-1"]

    closed = {"messages": 60, "episode_start": 60, "cursor": 60, "policy": "segmenter"}
    walks = chain.boundary_walks(applied["rows"], dict.fromkeys(applied["episode_by_message"], chat), "segmenter")
    assert walks == {chat: closed}  # no open episode: the bootstrap's state is closed
    state = chain.Chain(
        chats={
            chat: chain.ChatState(
                episodes=2,
                last_episode="ep-1",
                last_saga=content_uuid("saga", group, "ep-1"),
                last_order=59,
                roster=["Oren", "Ilse"],
                tail=boot[chain.tail_start(closed) :],
                boundary=closed,
            )
        },
        message_ids=dict.fromkeys(applied["episode_by_message"], 0),
        messages=60,
        episodes=2,
        sessions=2,
        ground={},
        evidence={},
    )
    (link,) = write.chain_sagas(state, chat, group, None)
    assert link["prior"] == [] and link["uuid"] == content_uuid("saga", group, "ep-1")  # NEXT_SAGA source, not grown

    plan = SimpleNamespace(chat=chat, session_id="c-test-session_3", number=3, first_order=60, skipped_ids=())
    forward = bonfire.forward_input(state, plan, voyage_cache=tmp_path / "voyage.json")
    session = _session("c-test-session_3", ["lanterns"] * 20, 60)
    run_segment_boundary(
        session,
        tmp_path / "append",
        first_episode=forward["boundary"]["next_episode"],
        continuation=forward["boundary"],
        **kw,
    )
    appended = json.loads((tmp_path / "append" / "applied-boundaries.json").read_bytes())
    # walk numbering is 0-based across the bonfire: after ep-0 and ep-1 the third episode is ep-2
    assert set(appended["episode_by_message"].values()) == {"ep-2"}
    assert appended["continuation"]["opened"] == ["ep-2"]
