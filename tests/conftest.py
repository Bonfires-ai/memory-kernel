"""Minimal test wiring for the end-state suite.

``pyproject`` sets ``pythonpath = ["src", "tests"]``, which only edits THIS process's ``sys.path``. Tests that spawn
``python -m memory_kernel...`` children need ``src`` on ``PYTHONPATH`` too, so ``child_env`` builds that environment
(``src`` only: no child imports from the repo root any more).

Both the in-process tests and the children run under the blocked env, whoever launches pytest: no real key and a dead
proxy (a Jev, OpenRouter or Voyage request fails fast and never bills), and the Hugging Face hub offline for the
``[parser]`` extra's ``transformers``, should a test load it (no MiniLM any more).
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

BLOCKED_ENV = {
    "OPENROUTER_API_KEY": "replay-invalid",
    "HTTPS_PROXY": "http://127.0.0.1:9",
    "https_proxy": "http://127.0.0.1:9",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


@pytest.fixture(autouse=True)
def _blocked_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in BLOCKED_ENV.items():
        monkeypatch.setenv(name, value)


def chain_of_fake_root(root: Path):
    """The chain of a hand-made bootstrap root (config, ``01-select-001/sources.json``, markers, a shard), from its
    select rows alone: every message id (run 0), its chat (the conversation-only file's session chats, else the sample),
    one episode per session (its boundary walk: the open episode is the chat's last session, the cursor at its start),
    the counters, tail and roster, the session count, addressing start and agent handles (the conversation-only file's,
    else the select rows'). No entities, routes or rolesets; an empty ground and evidence."""
    from memory_kernel.graph.g1_ids import content_uuid
    from memory_kernel.pipeline.chain import Chain, ChatState, tail_start
    from memory_kernel.pipeline.stages._conversation import session_conversations, session_keys

    rows = json.loads((root / "01-select-001" / "sources.json").read_text())
    config = json.loads((root / "logs" / "mk" / "build-config.json").read_text())
    sample, group = config["sample_id"], config["group"]
    chats_of: dict[str, str] = {}
    conv: dict = {}
    for conv_only in (root / "01-select-001").glob("*-conversation-only.json"):
        conv = json.loads(conv_only.read_text())["conversation"]
        chats_of = session_conversations(conv, sample)
    rows = [{"message_id": f"m{i}", "session_id": sample, "speaker": "?", **r} for i, r in enumerate(rows)]
    by_chat: dict[str, list[dict]] = {}
    for r in rows:
        by_chat.setdefault(chats_of.get(r["session_id"], sample), []).append(r)
    chats, episodes = {}, set()
    for chat, msgs in by_chat.items():
        sessions = list(dict.fromkeys(m["session_id"] for m in msgs))
        episodes |= {f"{chat}:{s}" for s in sessions}
        last = f"{chat}:{sessions[-1]}"
        roster = dict.fromkeys(n for m in msgs for n in (m["speaker"], *(m.get("addressees") or ())))
        start = next(i for i, m in enumerate(msgs) if m["session_id"] == sessions[-1])
        walk = {"messages": len(msgs), "episode_start": start, "cursor": start}
        chats[chat] = ChatState(
            episodes=len(sessions),
            last_episode=last,
            last_saga=content_uuid("saga", group, last),
            last_order=msgs[-1]["order"],
            roster=list(roster),
            tail=msgs[tail_start(walk) :],
            boundary=walk,
        )
    return Chain(
        chats=chats,
        message_ids=dict.fromkeys((r["message_id"] for r in rows), 0),
        messages=len(rows),
        episodes=len(episodes),
        sessions=len(list(session_keys(conv))) if conv else len({r["session_id"] for r in rows}),
        explicit_since=min((r["order"] for r in rows if r.get("addressees")), default=None),
        agent_handles=list(conv.get("agent_handles") or []),
        ground={},  # no entities, so an empty ground (as ``from_bootstrap`` gives a bootstrap): never None
        evidence={},  # none with no ground rows; None exactly when ground is
    )


@pytest.fixture(autouse=True)
def _hand_made_bootstraps_get_a_select_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """``bonfire.init`` derives the chain state with ``chain.bootstrap_chain``: the real projection of the root's stage
    outputs. A root that has them (``10-events-001/eventuality.json``) goes through the REAL function. A hand-made root
    (config, sources, markers, a shard: no stage outputs to project) gets ``chain_of_fake_root``, a chain of its select
    rows, never an empty one. The tests of ``chain`` hold the real function from collection time."""
    from memory_kernel.pipeline import chain

    real = chain.bootstrap_chain

    def bootstrap_chain(root: Path):
        if (Path(root) / "10-events-001" / "eventuality.json").is_file():
            return real(root)
        return chain_of_fake_root(Path(root))

    monkeypatch.setattr(chain, "bootstrap_chain", bootstrap_chain)


@pytest.fixture(scope="session")
def child_env() -> dict[str, str]:
    """``os.environ`` plus the blocked env, with ``src`` on ``PYTHONPATH``."""

    env = {**os.environ, **BLOCKED_ENV}
    env["PYTHONPATH"] = str(SRC)
    return env


def _turn_reads_on_trf() -> bool:
    """True when the ``turn`` purpose resolves to an installed ``en_core_web_trf`` (not an override, not a fallback to sm)."""
    import importlib.util

    from memory_kernel.grammar.dependencies import spacy_model

    return spacy_model("turn") == "en_core_web_trf" and importlib.util.find_spec("en_core_web_trf") is not None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """``@pytest.mark.trf``: the test pins a reading of the trf parser; skipped where turns would read on sm (trf not
    installed, or ``MK_TURN_SPACY_MODEL`` set to another model). release-check.yml runs them."""
    if not any(item.get_closest_marker("trf") for item in items):
        return
    if not _turn_reads_on_trf():
        skip = pytest.mark.skip(reason="the turn parser is not en_core_web_trf here (not installed, or overridden)")
        for item in items:
            if item.get_closest_marker("trf"):
                item.add_marker(skip)


class Neo4jTarget(NamedTuple):
    uri: str
    user: str
    password: str
    group: str


@pytest.fixture
def neo4j_target():
    """A live Neo4j test group ``mk20-test-<8 hex>``, deleted afterwards. Skips without ``MK_TEST_NEO4J_URI``; it
    ignores ``MK_TEST_NEO4J_GROUP``, since every test makes its own group."""
    uri = os.environ.get("MK_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("MK_TEST_NEO4J_URI not set")
    from neo4j import GraphDatabase

    target = Neo4jTarget(
        uri,
        os.environ.get("MK_TEST_NEO4J_USER", "neo4j"),
        os.environ.get("MK_TEST_NEO4J_PASSWORD", ""),
        f"mk20-test-{secrets.token_hex(4)}",
    )
    try:
        yield target
    finally:
        with GraphDatabase.driver(target.uri, auth=(target.user, target.password)) as driver:
            driver.execute_query("MATCH (n {group_id: $g}) DETACH DELETE n", g=target.group)
