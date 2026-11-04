"""``project_u2`` wiring: load the build's inputs and project the G1 event graph through the roleset-index modules.

Ported from a reference-harness module (unreached top-level definitions pruned). Differences: the module `INPUTS` (rebound
by the reference harness's build driver) is the `inputs` argument of `load_inputs`; `base.EXPECTED_MESSAGES` /
`EXPECTED_EPISODES` / `base.GROUP` and the `FRAME_DIRS` default of `frame_index` / `UPSTREAM_FRAMES` are arguments
(`expected_*`, `group`, `frame_dirs`). The reference script was a dry run (no Neo4j); here ``stages.write`` calls
``project_u2`` and writes its rows. Kernel 1.8 reads no Jev1 labels, alignments or card inventory: the projection
carries no codings, topic labels or cards.

It reuses the input checks of ``_g1_build`` (imported as ``base``) and projects with:

- ONE tag source: the PropBank roleset index of the upstream checkout (``frame_dirs[0]``, never the merged annex dirs)
  feeds ``project(roleset_index=...)``; role edge names, edge direction (CAU) and the referring filter all read it
  through ``g1_roles.function_tag``.
- VerbNet classes come from the same checkout (``verbnet_classes(roleset, frame_dirs[0])``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from memory_kernel.graph.g1_projection import message_rank, project
from memory_kernel.knowledge.propbank_framesets import (
    load_propbank_roleset_index,
    verbnet_classes,
    verbnet_role_themes,
)
from memory_kernel.pipeline.stages import _g1_build as base
from memory_kernel.pipeline.stages._eventuality_ctx import frame_index


def load_inputs(
    root: Path,
    inputs: dict[str, str],
    *,
    frame_dirs,
    expected_messages: int,
    expected_episodes: int,
    session_conversations: Mapping[str, str] | None = None,
) -> dict:
    """``inputs``: graph / eventuality / episode_map, root-relative path strings (``stages.write.projection_inputs``).
    ``session_conversations`` (``session_id`` -> conversation id, ``_conversation.session_conversations``) checks
    the episode map per conversation (``check_episode_map``) and is returned as ``conversation_of`` (message id ->
    conversation; None without it), which ``project_u2`` hands to ``project`` (one ``NEXT_SAGA`` chain per chat)."""
    records = json.loads((root / inputs["graph"]).read_text())["records"]
    conversation_of = (
        {r["message_id"]: session_conversations[r["session_id"]] for r in records}
        if session_conversations is not None
        else None
    )
    episodes = base.check_episode_map(
        json.loads((root / inputs["episode_map"]).read_text()),
        {r["message_id"] for r in records},
        messages=expected_messages,
        episodes=expected_episodes,
        rank=message_rank(records),
        conversation_of=conversation_of,
    )
    keys = {f"{r['unit']}#{i}" for r in records for i in range(len(r["events"]))}
    eventuality, excluded = base.check_eventuality(keys, json.loads((root / inputs["eventuality"]).read_text()))
    frames = frame_index(frame_dirs)
    pos = {
        rs: frames[rs]["pos"] if rs in frames else "verb"
        for rs in sorted({e["roleset"] for r in records for e in r["events"]})
    }
    return {
        "records": records,
        "episodes": episodes,
        "eventuality": eventuality,
        "excluded": excluded,
        "pos": pos,
        "conversation_of": conversation_of,
    }


def project_u2(data: dict, *, group: str, frame_dirs, mode: str = "b", **extra: object) -> dict:
    """The ``project_u2`` projection; ``extra`` passes the write-stage keywords (``ingest_pass``, ``referee``,
    ``saga_vectors``, ``referent_policy``) through.

    ``role_themes`` defaults to the VerbNet 3.4 rolelinks of ``frame_dirs[0]`` (holder and direction from frame
    data); pass ``role_themes=lambda _rs: {}`` to project without them (the C2 gate's arm A). ``None`` is refused
    by ``project`` with a roleset index (a ratchet)."""
    upstream_frames = frame_dirs[0]
    extra.setdefault("role_themes", lambda roleset: verbnet_role_themes(roleset, upstream_frames))
    return project(
        data["records"],
        eventuality=data["eventuality"],
        episodes=data["episodes"],
        group_id=group,
        created_at=base.CREATED_AT,
        existing_sagas=data.get("existing_sagas") or [],  # a forward append: its chat's last saga (write.chain_sagas)
        chain=data.get("chain"),  # a forward append's chain view (``write.project``); None = a full build
        run=data.get("run", 0),
        excluded=data["excluded"],
        predicate_pos=data["pos"],
        roleset_index=load_propbank_roleset_index(upstream_frames),
        verbnet=lambda roleset: verbnet_classes(roleset, upstream_frames),
        render_mode=mode,
        conversation_of=data.get("conversation_of"),
        **extra,  # type: ignore[arg-type]
    )
