"""``NEXT_SAGA`` runs per conversation (chat) (kernel 1.1 C1, Task 5 ruling): with several chats a saga links only from
the previous saga of its own chat; one chat (or no map) is the single chain of 1.0.0, row for row.

Fixture: ``project()`` with message records that carry no events (the saga / episode rows need none), one episode
per two messages, ``ingest_pass`` set (the chain is written only then). Invented names only.
"""

from __future__ import annotations

from memory_kernel.graph.g1_invariants import derivation_acyclic
from memory_kernel.graph.g1_projection import project

GROUP = "g-test"
#: (message id, session, speaker, minute); chat A = sessions s1, s3; chat B = s2 (interleaved in time with A).
MESSAGES = [
    ("a1", "s1", "Ana", 0),
    ("a2", "s1", "Bo", 1),
    ("b1", "s2", "Cy", 2),
    ("b2", "s2", "Di", 3),
    ("a3", "s3", "Ana", 4),
    ("a4", "s3", "Bo", 5),
    ("b3", "s2", "Cy", 6),
    ("b4", "s2", "Di", 7),
]
EPISODES = {"a1": "A-0", "a2": "A-0", "b1": "B-0", "b2": "B-0", "a3": "A-1", "a4": "A-1", "b3": "B-1", "b4": "B-1"}


def records() -> list[dict]:
    return [
        {
            "unit": f"0000/unit-{i:02d}.json",
            "message_id": mid,
            "session_id": session,
            "speaker": speaker,
            "text": "hello there",
            "timestamp": f"2026-10-14T00:{minute:02d}:00Z",
            "capture_status": "captured",
            "events": [],
        }
        for i, (mid, session, speaker, minute) in enumerate(MESSAGES)
    ]


def run(conversation_of=None) -> dict:
    return project(
        records(),
        eventuality={},
        episodes=EPISODES,
        group_id=GROUP,
        created_at="2026-10-14T00:00:00Z",
        ingest_pass="mk-conv-test-001",
        conversation_of=conversation_of,
    )


def chain(out: dict) -> list[tuple[str, str]]:
    name = {n["props"]["uuid"]: n["props"]["name"] for n in out["nodes"] if "Saga" in n["labels_neo4j"]}
    return sorted((name[e["source"]], name[e["target"]]) for e in out["edges"] if e["type"] == "NEXT_SAGA")


def test_one_chat_is_the_single_chain_row_for_row():
    plain = run()
    one_chat = run({m: "conv-test" for m, *_ in MESSAGES})
    assert one_chat == plain
    assert chain(plain) == [("A-0", "B-0"), ("A-1", "B-1"), ("B-0", "A-1")]  # the global chain, by first message


def test_two_chats_chain_apart():
    chat = {m: ("A" if m.startswith("a") else "B") for m, *_ in MESSAGES}
    out = run(chat)
    assert chain(out) == [("A-0", "A-1"), ("B-0", "B-1")]
    assert derivation_acyclic(out) == []
    # nothing else moves: the sagas, episodes and the other edges are those of the single chain
    plain = run()
    others = lambda o: sorted(e["props"]["uuid"] for e in o["edges"] if e["type"] != "NEXT_SAGA")  # noqa: E731
    assert others(out) == others(plain)
    assert sorted(n["props"]["uuid"] for n in out["nodes"]) == sorted(n["props"]["uuid"] for n in plain["nodes"])


# -- a forward append's chain (kernel 1.2, Review Focus 4) ---------------------------------------------------------------


def _forward_state(chat: str, msgs: list[str], last_episode: str, open_from: int) -> object:
    """Chat ``chat``'s chain state after ``msgs`` (its last episode ``last_episode`` opened at walk position
    ``open_from``): the tail holds the open episode."""
    from memory_kernel.graph.g1_ids import content_uuid
    from memory_kernel.pipeline.chain import ChatState

    rows = {r["message_id"]: r for r in records()}
    return ChatState(
        episodes=len({EPISODES[m] for m in msgs}),
        last_episode=last_episode,
        last_saga=content_uuid("saga", GROUP, last_episode),
        last_order=0,
        roster=[],
        tail=[rows[m] for m in msgs[open_from:]],
        boundary={"messages": len(msgs), "episode_start": open_from, "cursor": open_from},
    )


def _session(ids: list[str], chat_state, chat: str, episodes: dict[str, str]) -> dict:
    """``project`` of one session of ``chat`` with its chain's ``existing_sagas`` (``write.chain_sagas``)."""
    from memory_kernel.pipeline.chain import Chain
    from memory_kernel.pipeline.stages.write import chain_sagas

    chain_of = Chain(chats=chat_state)
    return project(
        [r for r in records() if r["message_id"] in ids],
        eventuality={},
        episodes={m: episodes[m] for m in ids},
        group_id=GROUP,
        created_at="2026-10-14T00:00:00Z",
        existing_sagas=chain_sagas(chain_of, chat, GROUP, "mk-conv-test-001"),
        ingest_pass="mk-conv-test-001",
        conversation_of=dict.fromkeys(ids, chat),
    )


def test_interleaved_chats_link_each_new_saga_from_their_own_last_saga():
    """Bootstrap a1 a2 (A-0) b1 b2 (B-0); then the sessions a3 a4 (chat A) and b3 b4 (chat B), one at a time: each
    first new saga links from its OWN chat's last saga, giving exactly the full build's per-chat edges."""
    from memory_kernel.graph.g1_ids import content_uuid

    chat = {m: ("A" if m.startswith("a") else "B") for m, *_ in MESSAGES}
    full = {e["props"]["uuid"] for e in run(chat)["edges"] if e["type"] == "NEXT_SAGA"}
    states = {"A": _forward_state("A", ["a1", "a2"], "A-0", 0), "B": _forward_state("B", ["b1", "b2"], "B-0", 0)}
    a = _session(["a3", "a4"], states, "A", EPISODES)
    b = _session(["b3", "b4"], states, "B", EPISODES)
    s = lambda episode: content_uuid("saga", GROUP, episode)  # noqa: E731
    pairs = lambda o: [(e["source"], e["target"]) for e in o["edges"] if e["type"] == "NEXT_SAGA"]  # noqa: E731
    assert pairs(a) == [(s("A-0"), s("A-1"))] and pairs(b) == [(s("B-0"), s("B-1"))]  # never across chats
    assert a["external_endpoints"][s("A-0")]["labels_neo4j"] == ["Saga"]
    forward = {e["props"]["uuid"] for out in (a, b) for e in out["edges"] if e["type"] == "NEXT_SAGA"}
    assert forward == full


def test_a_session_continuing_its_chats_open_episode_extends_that_saga():
    """a3 continues chat A's open episode A-0 (a1 a2): the saga keeps its start and counts the whole episode, and
    ``NEXT_EPISODE`` runs on from the stored a2; the row is the full build's, row for row."""
    from memory_kernel.graph.g1_ids import content_uuid

    episodes = {**EPISODES, "a3": "A-0", "a4": "A-0"}
    states = {"A": _forward_state("A", ["a1", "a2"], "A-0", 0)}
    out = _session(["a3", "a4"], states, "A", episodes)
    full = project(
        records(),
        eventuality={},
        episodes=episodes,
        group_id=GROUP,
        created_at="2026-10-14T00:00:00Z",
        ingest_pass="mk-conv-test-001",
        conversation_of={m: ("A" if m.startswith("a") else "B") for m, *_ in MESSAGES},
    )
    saga_of = lambda o: next(n for n in o["nodes"] if n["props"]["name"] == "A-0")  # noqa: E731
    assert saga_of(out) == saga_of(full) and saga_of(out)["props"]["message_count"] == 4
    a2_to_a3 = [
        e for e in out["edges"] if e["type"] == "NEXT_EPISODE" and e["target"] == content_uuid("ep", GROUP, "a3")
    ]
    assert [e["source"] for e in a2_to_a3] == [content_uuid("ep", GROUP, "a2")]
    assert chain(out) == []  # a continued saga is already in its chat's chain
