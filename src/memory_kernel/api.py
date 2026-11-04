"""The library entry points a service calls without knowing how a bonfire is built.

* ``append_session(bonfire_id, session, ops=, scratch_root=)``: one newly closed session in, the bonfire grown
  (``pipeline.graph_append.append``): one run of that session alone on the bonfire's graph state, committed in one Neo4j
  write transaction. It is idempotent on the session's message ids: a session whose ids are all in the bonfire already
  is a ``"duplicate"`` and nothing runs (a retried job); a session that shares only SOME ids with it is refused, never
  merged, unless ``on_built="skip"``: then only its new messages are appended (``skipped_ids`` names the others).
* ``context(bonfire_id, question)``: a question in, the kernel's context block out, from the bonfire's graph in Neo4j
  (the bonfire is its ``BonfireMeta`` node; no directory is read or written).

``append_session`` blocks for the whole build (a stage child per stage); ``context`` runs its own event loop
(``acontext`` is the coroutine, for a caller already inside one). Secrets come from the process environment
(``cli.resolve_secrets``) unless ``serve_cfg_overrides`` passes them.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_kernel.pipeline import bonfire
from memory_kernel.pipeline.bonfire import AppendBusy, BonfireError
from memory_kernel.serve import answer

#: The run label of the bootstrap root (a recorded run is labelled by its number, ``"0001"`` ...).
BOOTSTRAP_RUN = "bootstrap"


@dataclass(frozen=True)
class SessionInput:
    """One closed session of one chat. ``messages``: ``{message_id, speaker, text, timestamp}`` each (``timestamp``
    ISO 8601 with a timezone), optionally ``addressees``; ``agent_handles`` are added to the conversation's."""

    conversation_id: str
    messages: list[dict[str, Any]]
    agent_handles: list[str] | None = None


class HandlesFixed(BonfireError):
    """Compatibility only, NEVER raised since kernel 1.2: a forward append never prepares a built turn again, so a
    session's agent handles apply to that session alone. Kept because the staging kernel worker classifies kernel
    errors by class name."""


def _default_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"


def append_session(
    bonfire_id: str,
    session: SessionInput,
    *,
    ops: Any,
    scratch_root: Path | str,
    owner: str | None = None,
    lease_ttl_s: float = 3600,
    on_built: bonfire.OnBuilt = "refuse",
    overrides: Mapping[str, Any] | None = None,
    **build_opts: Any,
) -> dict[str, Any]:
    """Grow the bonfire ``bonfire_id`` (kernel 2.0: its state lives in its Neo4j group) by one closed session
    (``graph_append.append``). ``ops`` is the ``KernelOps`` holding the bonfire's lease and spend, ``scratch_root`` the
    directory the run is built in (deleted afterwards), ``owner`` names this writer in the lease (default: unique per
    call) and ``lease_ttl_s`` is the lease's time to live (renewed every third of it). ``overrides`` are the build config
    fields set over the bonfire's own: the secrets, the Neo4j address, the host's install paths. ``build_opts``:
    ``secret_sources``, ``runner``, ``allow_parser_change`` (an append whose extraction model differs from the
    bonfire's recorded one is refused without it). ``on_built`` is what a session with SOME message ids built already
    gets: ``"refuse"`` (the default) refuses it, ``"skip"`` drops those ids before anything is built or spent and
    appends the rest.

    Returns ``{status, run, sessions, new_messages, skipped_ids, result}``. ``status`` is:

    * ``"duplicate"``: every message id is built already (nothing runs; ``run`` and ``result`` are None);
    * ``"empty"``: every message is blank or markup only, nothing to build (nothing runs);
    * ``"appended"``: the run sealed and its commit landed;
    * ``"stopped"``: it ran but stopped before ``seal`` (nothing committed, the spend settled). Send the same session
      again, with the same ``on_built``, BEFORE any later session of this bonfire: sessions are ordered as they reach
      the graph, and the lease refuses any other session (``PendingSession``) until this one lands.

    Refused (``BonfireError``): no such bonfire, a malformed session, ``AppendBusy`` (retriable) while another append
    of the bonfire holds its lease, ``PendingSession``, the spend ceiling, and every refusal of the plan and the run."""
    from memory_kernel.pipeline import graph_append

    return graph_append.append(
        bonfire_id,
        session_doc(session),
        ops=ops,
        scratch_root=scratch_root,
        owner=owner or _default_owner(),
        lease_ttl_s=lease_ttl_s,
        on_built=on_built,
        overrides=overrides,
        **build_opts,
    )


def session_doc(session: SessionInput) -> dict[str, Any]:
    """The session file ``bonfire.append`` takes (``bonfire.read_session``): the session as given."""
    doc: dict[str, Any] = {"conversation_id": session.conversation_id, "messages": session.messages}
    if session.agent_handles is not None:
        doc["agent_handles"] = session.agent_handles
    return json.loads(json.dumps(doc))


# -- context -----------------------------------------------------------------------------------------------------------


def run_label(committed_run: int) -> str:
    """A committed run's label: ``"bootstrap"`` for run 0, else ``f"{n:04d}"``."""
    return BOOTSTRAP_RUN if int(committed_run) == 0 else f"{int(committed_run):04d}"


def serve_config(
    bonfire_id: str, meta: Mapping[str, Any] | None = None, *, overrides: Mapping[str, Any] | None = None
) -> answer.ServeConfig:
    """The ``ServeConfig`` of bonfire ``bonfire_id`` (its group): the secrets from the process environment
    (``cli.resolve_secrets``), the bonfire's ``query_cache_key`` from its ``BonfireMeta`` ``meta``, then ``overrides``
    (the Neo4j target comes from the service config, not from the bonfire)."""
    from memory_kernel.cli import resolve_secrets

    kw: dict[str, Any] = {
        "group": bonfire_id,
        "query_cache_key": str((meta or {}).get("query_cache_key") or ""),
        **resolve_secrets(env_file=None, container=None).values,
    }
    return answer.ServeConfig(**(kw | dict(overrides or {})))


async def bonfire_config(
    bonfire_id: str, overrides: Mapping[str, Any] | None = None
) -> tuple[answer.ServeConfig, dict[str, Any]]:
    """``(ServeConfig, BonfireMeta properties)`` of bonfire ``bonfire_id``. The bonfire is its ``BonfireMeta`` node in
    the Neo4j that ``overrides`` name: ``BonfireError`` when the group has none (a 1.8 group, a seed in progress, a
    typo). Nothing is read from or written to disk."""
    from memory_kernel.serve.registry import BonfireRegistry

    driver = answer.open_driver(serve_config(bonfire_id, overrides=overrides))
    try:
        meta = await BonfireRegistry(driver).meta(bonfire_id)
    finally:
        await driver.close()
    if meta is None:
        raise BonfireError(f"{bonfire_id!r} is not a bonfire here: its group has no BonfireMeta")
    return serve_config(bonfire_id, meta, overrides=overrides), meta


async def acontext(
    bonfire_id: str,
    question: str,
    *,
    serve_cfg_overrides: Mapping[str, Any] | None = None,
    speaker: str | None = None,
) -> dict[str, Any]:
    """``context`` as a coroutine."""
    cfg, meta = await bonfire_config(bonfire_id, serve_cfg_overrides)
    resolved = answer.resolve_first_person(question, speaker)
    async with answer.Server(cfg) as srv:
        out = await srv.context(resolved)
    return {
        **out,
        "engine": "kernel",
        "run": run_label(meta["committed_run"]),
        **({"resolved_query": resolved} if resolved != question else {}),
    }


def context(
    bonfire_id: str,
    question: str,
    *,
    serve_cfg_overrides: Mapping[str, Any] | None = None,
    speaker: str | None = None,
) -> dict[str, Any]:
    """The kernel's context block for ``question`` from bonfire ``bonfire_id``'s committed graph (``Server.context``'s
    ``frame``, ``text``, ``message_ids``, ``tokens``, ``retrieved`` and ``degraded``: False) plus ``engine: "kernel"``
    and ``run``. The frame is the wh rules' ``shape`` alone. Cache-only unless ``serve_cfg_overrides`` sets
    ``allow_paid``; the Neo4j target (``neo4j_uri``, ``neo4j_user``) is in the overrides too. ``speaker`` (the asker,
    as the bonfire keys people) resolves the question's first-person words (``answer.resolve_first_person``); the
    result then carries ``resolved_query``."""
    return asyncio.run(acontext(bonfire_id, question, serve_cfg_overrides=serve_cfg_overrides, speaker=speaker))


async def _bonfire_exists(group: str, overrides: Mapping[str, Any]) -> bool:
    from neo4j import AsyncGraphDatabase

    from memory_kernel.cli import resolve_secrets
    from memory_kernel.graph.state_read import read_meta

    password = (
        overrides.get("neo4j_password") or resolve_secrets(env_file=None, container=None).values["neo4j_password"]
    )
    if password is None:
        raise BonfireError("no Neo4j password configured (NEO4J_PASSWORD or the overrides' neo4j_password)")
    uri = overrides.get("neo4j_uri") or "bolt://localhost:7687"
    driver = AsyncGraphDatabase.driver(uri, auth=(overrides.get("neo4j_user") or "neo4j", password))
    try:
        return await read_meta(driver, group) is not None
    finally:
        await driver.close()


def bonfire_exists(bonfire_id: str, *, overrides: Mapping[str, Any] | None = None) -> bool:
    """Whether the group ``bonfire_id`` holds a ``BonfireMeta``: a seeded 2.0 bonfire (kernel 2.0 names the group by the
    bonfire id). ``overrides`` carry the Neo4j address (``neo4j_uri``, ``neo4j_user``, ``neo4j_password``) as an append's
    do; the password defaults from the environment."""
    return asyncio.run(_bonfire_exists(bonfire_id, dict(overrides or {})))


__all__ = [
    "BOOTSTRAP_RUN",
    "AppendBusy",
    "HandlesFixed",
    "SessionInput",
    "acontext",
    "bonfire_config",
    "append_session",
    "bonfire_exists",
    "context",
    "run_label",
]
