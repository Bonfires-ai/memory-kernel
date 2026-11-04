"""referents: an append's decisions on specific mentions, before the projection writes rows.

Collect (one projection with a collecting referee), plan (``graph.referents.plan_session`` over the pinned chain
ground), ask Jev for the hard cases (``graph.referent_jev``, through the run's shard, stage cap $0.05), settle, and
write ``decisions.json`` once. A full build writes a marker and calls nothing. A failed or partial ask writes
nothing, and nothing creates the stage dir before that one write (the stage is not resumable: a started dir stops the
driver): a rerun replays the archived wires and asks the rest.

The rows the planner and the wires read (``_collect``): the chain's evidence rows (``bonfire.message_rows``: every
ground row's first and last mention, kept in the chain because retention prunes runs), the chat's pinned tail, the session's records and its own select rows. Every
candidate's first and last mention and every message "before" a hard case must be among them, else the stage is
refused naming the missing id: no wire is sent with missing evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memory_kernel.graph import referents as rf
from memory_kernel.graph.referent_jev import answer_yes, question_wire
from memory_kernel.pipeline.config import load_forward
from memory_kernel.pipeline.stages._common import cap_check
from memory_kernel.pipeline.stages._g1_cards import write_once
from memory_kernel.pipeline.stages.write import Refused, _out

if TYPE_CHECKING:
    from memory_kernel.pipeline.runner import Ctx

DECISIONS = "decisions.json"
STAGE_CAP_USD = 0.05
#: The messages before a hard case's message that its wire carries (``question_wire``'s ``new_message_context``).
BEFORE = 2


def run_dir(cfg: Any) -> Path:
    """The directory of an append run: its forward input is ``<run dir>/forward.json`` (``bonfire.FORWARD_INPUT``
    beside the run root ``<run dir>/<sample_id>``), and the bonfire's evidence rows are ``<run dir>/evidence.json``."""
    return Path(cfg.forward_input).parent


def _collect(ctx: Ctx) -> tuple[list[rf.Point], dict[str, dict], list[str]]:
    """The decision points (one projection with a collecting referee), the rows the planner reads, and the session's
    message ids in session order."""
    from memory_kernel.pipeline.bonfire import message_rows
    from memory_kernel.pipeline.stages import write

    points: list[rf.Point] = []
    data, _, _ = write.project(ctx, referee=lambda p: points.append(p) or None)
    pinned = load_forward(ctx.cfg) or {}
    rows = message_rows(run_dir(ctx.cfg))
    rows.update({r["message_id"]: r for r in (pinned.get("boundary") or {}).get("tail") or ()})
    rows.update({r["message_id"]: r for r in data["records"]})
    rows.update({r["message_id"]: r for r in ctx.sources()})  # the session's select rows: what the ground fold reads
    return points, rows, list(dict.fromkeys(r["message_id"] for r in data["records"]))


class Answers(list):
    """``_ask``'s p(yes) list; ``cap_record`` is what the stage records when its spend went past ``STAGE_CAP_USD``
    (``llm.stage_cap_record``), else empty."""

    cap_record: dict = {}


def _ask(ctx: Ctx, wires: list[dict]) -> list[float]:
    """p(yes) per wire, through the run's shard; raises on any failure (no partial answers are returned). The stage
    cap is recorded, never a stop; past it, a hold that would cross ``--cap-usd`` is refused (``llm.unit_guard``)."""
    from memory_kernel.construction.memory_jev_archive import read_jev_archive
    from memory_kernel.pipeline import llm

    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}

    def on_new_send(archive: dict) -> None:
        receipt = json.loads(archive["receipt"])["usage"]
        usage["requests"] += 1
        usage["input_tokens"] += receipt["input_tokens"]
        usage["output_tokens"] += receipt["output_tokens"]
        usage["usd"] += float(receipt["cost_usd"])

    shard = ctx.shard(on_new_send=on_new_send, reserve_guard=llm.unit_guard(ctx.cfg, usage, STAGE_CAP_USD))
    out = Answers()
    with ctx.cost("referents", run=ctx.path("referents").name, note="referent yes/no (shared shard)") as cost:
        try:
            ledger = None
            try:  # the ledger is opened inside the try that closes it: no failure leaves it open
                manifest, provenance, _ = shard.load()
                ledger = shard.open(manifest, provenance)
                for compiled in wires:
                    answers: dict[str, Any] = {}
                    for wire in compiled["batches"]:
                        answers.update(read_jev_archive(wire, ledger.evaluate(wire), ledger.policy)["answers"])
                    out.append(answer_yes(answers))
            finally:
                if ledger is not None:
                    ledger.close()
        finally:
            cost.add("jev", **usage)  # what was sent is recorded even when the ask failed
    if usage["usd"] > STAGE_CAP_USD:
        out.cap_record = llm.stage_cap_record(ctx.cfg, "referents", STAGE_CAP_USD, usage["usd"])
    return out


def _agents(cfg: Any, view: Mapping) -> list[str]:
    """The chain's agent handles (the view's union) and the run conversation's (the union it runs with, the one
    ``pipeline.commit`` folds the ground with); the planner compares them with or without "@"
    (``graph.referents.person_key``)."""
    handles = list(view.get("agent_handles") or [])
    conversation = getattr(cfg, "conversation", None)
    if conversation is not None and Path(conversation).is_file():
        handles += json.loads(Path(conversation).read_bytes())["conversation"].get("agent_handles") or []
    return list(dict.fromkeys(handles))


def _missing(hard: list[rf.HardCase], before: Mapping[int, list[str]], ground: Mapping, rows: Mapping) -> list[str]:
    need = [m for h in hard for m in (h.point.message_id, *before[h.index])]
    need += [m for h in hard for u in h.candidates for m in (ground[u].first_mid, ground[u].last_mid)]
    return sorted({m for m in need if m not in rows})


def referents(ctx: Ctx, live: bool) -> dict:  # noqa: ARG001 (one live call, as index)
    cfg = ctx.cfg
    target = _out(ctx, "referents", DECISIONS)
    pinned = load_forward(cfg)
    if pinned is None:
        write_once(target, {"mode": "full", "decisions": {}})
        return {"mode": "full"}
    view = pinned["chain"]
    if view.get("ground") is None:
        raise Refused(f"{cfg.forward_input}: the run's chain view carries no ground (run mk bonfire ground)")
    ground = {r["uuid"]: rf.Ground.from_row(r) for r in view["ground"]}
    stored_keys = {e["uuid"]: e["key"] for e in view["entities"]}
    names = {e["uuid"]: e["display"] for e in view["entities"]}
    chat = str(pinned["chat"])
    points, rows, session = _collect(ctx)
    provisional, hard, counts = rf.plan_session(
        points, chat=chat, rows=rows, ground=ground, stored_keys=stored_keys, agents=_agents(cfg, view)
    )
    # a session is one chat: the messages before one of its messages are the chat's tail and the session's earlier ones
    chat_order = [r["message_id"] for r in (pinned.get("boundary") or {}).get("tail") or ()] + session
    before = {}
    for h in hard:
        i = chat_order.index(h.point.message_id)
        before[h.index] = chat_order[max(0, i - BEFORE) : i]
    if missing := _missing(hard, before, ground, rows):
        raise Refused(f"referents: no message row for {missing}: a hard case's evidence is missing, nothing was asked")
    wires, keys = [], []
    for h in hard:
        for u in h.candidates:
            wires.append(question_wire(h, u, rows=rows, before=before[h.index], ground=ground, chat=chat, names=names))
            keys.append((h.index, u))
    if wires:
        cap_check(ctx, STAGE_CAP_USD)
    answers = _ask(ctx, wires) if wires else []
    p = dict(zip(keys, answers, strict=True))
    final = rf.settle(provisional, hard, p, session_keys=rf.session_keys(points, provisional))
    unknown = sorted({d["join"] for d in final.values() if "join" in d} - set(stored_keys))
    if unknown:  # forward.resolve raises on a join to an unknown uuid; refuse here, before anything is written
        raise Refused(f"referents: joins to uuids the chain view does not hold: {unknown}")
    body = {
        "mode": "append",
        "decisions": dict(sorted(final.items())),
        "hard": [
            {"point": h.point.point, "key": h.point.key, "candidates": [[u, p[(h.index, u)]] for u in h.candidates]}
            for h in hard
        ],
        "counts": counts,
    }
    write_once(target, body)
    return {"mode": "append", "counts": counts, "pairs": len(wires), **getattr(answers, "cap_record", {})}
