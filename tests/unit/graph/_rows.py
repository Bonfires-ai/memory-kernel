"""Projection helpers shared by the chain and graph-state tests: a hand-made bootstrap (two interleaved chats, four
messages) and the ``project()`` rows, select rows and rolesets it yields. Invented names only."""

from __future__ import annotations

import json

from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.graph.g1_projection import project
from memory_kernel.pipeline.chain import Chain, from_bootstrap, projection_view

GROUP = "g-test"
#: (message, chat, session, speaker, addressees, text, roleset, anchor, roles)
MESSAGES = [
    ("D1:1", "k1", "s1", "Bo", [], "Bo likes pears", "like.01", [1], {"arg0": ("Bo", [0]), "arg1": ("pears", [2])}),
    (
        "D2:1",
        "k2",
        "s2",
        "Dov",
        ["Ana"],
        "Dov likes pears",
        "like.01",
        [1],
        {"arg0": ("Dov", [0]), "arg1": ("pears", [2])},
    ),
    ("D1:2", "k1", "s1", "Ana", [], "Ana met Cy", "meet.01", [1], {"arg0": ("Ana", [0]), "arg1": ("Cy", [2])}),
    ("D3:1", "k1", "s3", "Ana", [], "Ana likes pears", "like.01", [1], {"arg0": ("Ana", [0]), "arg1": ("pears", [2])}),
]
EPISODES = {"D1:1": "ep-1", "D2:1": "ep-2", "D1:2": "ep-1", "D3:1": "ep-3"}
CHATS = {m[0]: m[1] for m in MESSAGES}
#: the boundary rows of that walk (source order): k1 opens ep-1, continues it, then shifts to ep-3; k2 opens ep-2
BOUNDARY_ROWS = [
    {"message_id": "D1:1", "choice": "initial", "working_window_start_order": 0},
    {"message_id": "D2:1", "choice": "initial", "working_window_start_order": 0},
    {"message_id": "D1:2", "choice": "continue_active", "working_window_start_order": 0},
    {"message_id": "D3:1", "choice": "shift_topic", "working_window_start_order": 2},
]
WALKS = {"k1": {"messages": 3, "episode_start": 2, "cursor": 2}, "k2": {"messages": 1, "episode_start": 0, "cursor": 0}}


def graph_records() -> list[dict]:
    return [
        {
            "unit": f"0000/unit-{i:02d}.json",
            "message_id": mid,
            "session_id": sid,
            "speaker": speaker,
            "text": text,
            "timestamp": f"2026-10-14T00:0{i}:00Z",
            "capture_status": "captured",
            "addressees": addressees,
            "events": [
                {"roleset": rs, "anchor": anchor, "roles": {r: {"text": t, "span": s} for r, (t, s) in roles.items()}}
            ],
        }
        for i, (mid, _chat, sid, speaker, addressees, text, rs, anchor, roles) in enumerate(MESSAGES)
    ]


def sources() -> list[dict]:
    """The select rows (``sources.json``): the fields of ``discourse_source`` the chain reads, plus a few."""
    return [
        {
            "message_id": mid,
            "order": i,
            "session_id": sid,
            "speaker": speaker,
            "addressees": addressees,
            "text": text,
            "timestamp": f"2026-10-14T00:0{i}:00Z",
        }
        for i, (mid, _chat, sid, speaker, addressees, text, *_rest) in enumerate(MESSAGES)
    ]


def projection(records: list[dict] | None = None) -> dict:
    records = records or graph_records()
    eventuality = {f"{r['unit']}#0": {"label": "occurrence", "p": {"occurrence": 1.0}} for r in records}
    return project(
        records,
        eventuality=eventuality,
        episodes={r["message_id"]: EPISODES[r["message_id"]] for r in records},
        group_id=GROUP,
        created_at="2026-10-14T00:00:00Z",
        ingest_pass="pass-1",
        conversation_of={r["message_id"]: CHATS[r["message_id"]] for r in records},
    )


ROLESETS = {
    "like.01": {
        "gloss": "like (roles: arg0 = liker)",
        "examples": ["Bo: Bo likes pears"],
        "predicates": 3,
        "label": "state",
    },
    "meet.01": {"gloss": "meet (roles: arg0 = meeter)", "examples": ["Ana: Ana met Cy"], "predicates": 1},
}


def bootstrap_state() -> Chain:
    return from_bootstrap(projection(), sources(), EPISODES, CHATS, ROLESETS, walks=WALKS)


def saga(episode: str) -> str:
    return content_uuid("saga", GROUP, episode)


def view(chain: Chain) -> dict:
    """What the 2.0 state read must reproduce of a 1.8 chain (``tests/fixtures/state/*-view.json``), as canonical
    JSON: its ``projection_view`` plus the fields the view leaves out (``evidence``, ``sessions``, ``messages``,
    ``episodes``, ``message_ids``, ``rolesets``). ``Minted.surfaces`` is blanked on every entity: 2.0 drops it (nothing
    reads it but the 1.8 chain files), so a chain read from the graph carries ``()``."""
    pv = projection_view(chain)
    pv["entities"] = [{**e, "surfaces": []} for e in pv["entities"]]
    doc = {
        "projection_view": pv,
        "evidence": chain.evidence,
        "sessions": chain.sessions,
        "messages": chain.messages,
        "episodes": chain.episodes,
        "message_ids": chain.message_ids,
        "rolesets": chain.rolesets,
    }
    return json.loads(json.dumps(doc, sort_keys=True))


def dump(doc: object) -> str:
    """The fixture bytes of ``doc``: sorted keys, one-space indent, a final newline."""
    return json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=True) + "\n"
