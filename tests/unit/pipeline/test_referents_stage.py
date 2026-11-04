"""The referents stage. Offline: projection and Jev are faked at the stage's seams. Invented names only."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest

from memory_kernel.graph.referents import Point
from memory_kernel.pipeline.stages import referents as stage


@pytest.fixture(autouse=True)
def no_programme_cap(monkeypatch):
    monkeypatch.setattr(stage, "cap_check", lambda _ctx, _usd: {})


def make_ctx(tmp_path, *, forward):
    """A stage context: ``path(key)`` is ``tmp_path / key``; a forward input (None: a full build) in ``tmp_path``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pinned = None
    if forward is not None:
        pinned = tmp_path / "forward.json"
        pinned.write_text(json.dumps(forward))
    cost = SimpleNamespace(add=lambda *_a, **_k: None, note="")
    return SimpleNamespace(
        cfg=SimpleNamespace(sample_id="conv-ember", forward_input=pinned, conversation=tmp_path / "no-conv.json"),
        counts=SimpleNamespace(messages=1),
        path=lambda key: tmp_path / key,
        cost=lambda *_a, **_k: contextlib.nullcontext(cost),
    )


def row(mid, speaker, at, text, addressees=()):
    return {"message_id": mid, "speaker": speaker, "timestamp": at, "text": text, "addressees": list(addressees)}


TAIL = row("m8", "Oren", "2026-08-20T09:00:00+00:00", "morning all")


def forward_with_ground() -> dict:
    """The stored "kiln" (e-kiln) was mentioned in ANOTHER chat (tg-a), by Ilse, long ago (stale): a hard case when
    Ilse speaks of "the kiln" in tg-b."""
    kiln = {"uuid": "e-kiln", "key": "kiln", "kind": "Entity", "display": "kiln", "surfaces": ["kiln"], "run": 0,
            "order": 0}  # fmt: skip
    ground = {"uuid": "e-kiln", "chats": ["tg-a"], "people": ["ilse"], "first_mid": "m1", "last_mid": "m1",
              "last_at": "2026-07-01T10:00:00+00:00", "mentions": 1}  # fmt: skip
    return {
        "chat": "tg-b",
        "boundary": {"chat": "tg-b", "tail": [TAIL]},
        "chain": {"entities": [kiln], "ground": [ground], "agent_handles": ["@helper"]},
    }


def points_for_kiln() -> list[Point]:
    return [Point("u#0|arg1", "m9", "kiln", "the kiln", "the kiln", "e-kiln")]


def session_rows() -> dict[str, dict]:
    return {
        "m1": row("m1", "Ilse", "2026-07-01T10:00:00+00:00", "a kiln arrived at the studio"),
        "m8": TAIL,
        "m9": row("m9", "Ilse", "2026-08-20T10:00:00+00:00", "the kiln cracked again"),
    }


def collected():
    return points_for_kiln(), session_rows(), ["m9"]


def forward_input_without_ground(tmp_path):
    """``bonfire.forward_input`` on a chain made before referent resolution (``ground`` is None)."""
    from memory_kernel.pipeline import bonfire

    state = SimpleNamespace(ground=None, chats={}, run=0, explicit_since=None, episodes=0, rolesets={}, messages=0)
    plan = SimpleNamespace(chat="tg-b", session_id="conv-ember-session_2", number=2, first_order=5)
    return bonfire.forward_input(state, plan, voyage_cache=tmp_path / "cache.json")


def test_a_full_build_writes_the_marker_and_calls_nothing(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, forward=None)
    monkeypatch.setattr(stage, "_collect", lambda _ctx: pytest.fail("a full build collects nothing"))
    monkeypatch.setattr(stage, "_ask", lambda _ctx, _wires: pytest.fail("a full build asks nothing"))
    stage.referents(ctx, True)
    assert json.loads((tmp_path / "referents" / "decisions.json").read_text()) == {"mode": "full", "decisions": {}}


def test_an_append_without_ground_is_refused(tmp_path):
    from memory_kernel.pipeline.bonfire import BonfireError

    with pytest.raises(BonfireError, match="mk bonfire ground"):
        forward_input_without_ground(tmp_path)


def test_a_view_without_ground_is_refused_by_the_stage(tmp_path, monkeypatch):
    from memory_kernel.pipeline.stages.write import Refused

    forward = forward_with_ground()
    forward["chain"]["ground"] = None
    ctx = make_ctx(tmp_path, forward=forward)
    monkeypatch.setattr(stage, "_collect", lambda _ctx: pytest.fail("refused before the projection"))
    with pytest.raises(Refused, match="no ground"):
        stage.referents(ctx, True)
    assert not (tmp_path / "referents").exists()


def test_hard_cases_go_to_jev_and_settle(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, forward=forward_with_ground())  # stored "kiln" in another chat, shared speaker, stale
    monkeypatch.setattr(stage, "_collect", lambda _ctx: collected())
    asked: list = []
    monkeypatch.setattr(stage, "_ask", lambda _ctx, wires: asked.extend(wires) or [0.9 for _ in wires])
    out = stage.referents(ctx, True)
    d = json.loads((tmp_path / "referents" / "decisions.json").read_text())
    assert d["decisions"] == {"u#0|arg1": {"join": "e-kiln"}} and out["counts"]["hard"] == 1
    assert d["mode"] == "append" and d["hard"] == [
        {"point": "u#0|arg1", "key": "kiln", "candidates": [["e-kiln", 0.9]]}
    ]
    # the wire's evidence: the tail message before m9 in its chat, and the candidate's first mention
    state = asked[0]["batches"][0]["state"]
    assert [m["text"] for m in state["new_message_context"]] == ["morning all"]
    assert state["earlier_first_mention"]["text"] == "a kiln arrived at the studio"
    assert state["earlier_messages_where"] == "another chat"


def test_a_no_answer_makes_a_new_referent(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, forward=forward_with_ground())
    monkeypatch.setattr(stage, "_collect", lambda _ctx: collected())
    monkeypatch.setattr(stage, "_ask", lambda _ctx, wires: [0.1 for _ in wires])
    stage.referents(ctx, True)
    d = json.loads((tmp_path / "referents" / "decisions.json").read_text())
    new = d["decisions"]["u#0|arg1"]["new"]
    assert new.startswith("kiln#") and new != "kiln"  # the stored entity holds "kiln": discriminated


def test_a_missing_answer_fails_the_stage(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, forward=forward_with_ground())
    monkeypatch.setattr(stage, "_collect", lambda _ctx: collected())

    def boom(_ctx, wires):
        raise RuntimeError("provider 529")

    monkeypatch.setattr(stage, "_ask", boom)
    with pytest.raises(RuntimeError):
        stage.referents(ctx, True)
    assert not (tmp_path / "referents" / "decisions.json").exists()
    assert not (tmp_path / "referents").exists()  # the stage is not resumable: a rerun must find no started dir


def test_missing_evidence_is_refused_before_any_wire(tmp_path, monkeypatch):
    from memory_kernel.pipeline.stages.write import Refused

    ctx = make_ctx(tmp_path, forward=forward_with_ground())
    rows = session_rows()
    del rows["m1"]  # the candidate's first mention (a pruned run's select rows, say)
    monkeypatch.setattr(stage, "_collect", lambda _ctx: (points_for_kiln(), rows, ["m9"]))
    monkeypatch.setattr(stage, "_ask", lambda _ctx, _wires: pytest.fail("no wire without its evidence"))
    with pytest.raises(Refused, match="m1"):
        stage.referents(ctx, True)
    assert not (tmp_path / "referents").exists()


def test_a_replay_decides_the_same(tmp_path, monkeypatch):
    """Two runs on the same inputs with the same answers write identical decisions."""
    monkeypatch.setattr(stage, "_collect", lambda _ctx: collected())
    monkeypatch.setattr(stage, "_ask", lambda _ctx, wires: [0.9 for _ in wires])
    for name in ("a", "b"):
        stage.referents(make_ctx(tmp_path / name, forward=forward_with_ground()), True)
    a = (tmp_path / "a" / "referents" / "decisions.json").read_bytes()
    assert a == (tmp_path / "b" / "referents" / "decisions.json").read_bytes()


# -- write.project's referee ------------------------------------------------------------------------------------------


def fake_projection(monkeypatch, tmp_path, *, forward):
    """``write.project`` with its loaders faked: returns what ``project_u2`` was called with."""
    from memory_kernel.pipeline.stages import write

    seen: dict = {}
    data = {"records": [], "episodes": {}}
    monkeypatch.setattr(write.u2, "load_inputs", lambda *_a, **_k: dict(data))
    monkeypatch.setattr(write.u2, "project_u2", lambda _data, **kw: seen.update(kw) or {})
    monkeypatch.setattr(write, "conversations_of", lambda _ctx: {})
    monkeypatch.setattr(write, "expected_episodes", lambda _ctx: 0)
    monkeypatch.setattr(write, "chain_sagas", lambda *_a: [])
    monkeypatch.setattr("memory_kernel.pipeline.chain.from_projection_view", lambda _v: SimpleNamespace(run=0))
    ctx = make_ctx(tmp_path, forward=forward)
    ctx.cfg.root = tmp_path
    ctx.cfg.group = "g-ember"
    ctx.cfg.frame_dirs = (tmp_path,)
    ctx.cfg.ingest_pass = "pass-1"
    return write, ctx, seen


def test_project_in_an_append_reads_the_decisions(monkeypatch, tmp_path):
    write, ctx, seen = fake_projection(monkeypatch, tmp_path, forward={"chain": {}, "chat": "tg-b"})
    with pytest.raises(write.Refused, match="decisions.json: missing"):
        write.project(ctx)
    (tmp_path / "referents").mkdir()
    (tmp_path / "referents" / "decisions.json").write_text(json.dumps({"decisions": {"u#0|arg1": {"join": "e-kiln"}}}))
    write.project(ctx)
    referee = seen["referee"]
    assert referee(points_for_kiln()[0]) == {"join": "e-kiln"}
    assert referee(Point("u#1|arg0", "m9", "glaze", "the glaze", "the glaze", "e-glaze")) is None
    write.project(ctx, referee=None)  # the backfill (Task 6): no referee, no decisions file read
    assert seen["referee"] is None


def test_project_in_a_full_build_has_no_referee(monkeypatch, tmp_path):
    write, ctx, seen = fake_projection(monkeypatch, tmp_path, forward=None)
    write.project(ctx)
    assert seen["referee"] is None


# -- bonfire.message_rows ---------------------------------------------------------------------------------------------


def test_the_stage_finds_evidence_in_its_run_directory(tmp_path, monkeypatch):
    """The kiln's first mention (m1) came from a run no directory remembers. The run pinned the bonfire's evidence
    (``evidence.json`` beside ``forward.json``), so the real ``_collect`` (projection faked) finds it and the case is
    asked."""
    from memory_kernel.pipeline import bonfire
    from memory_kernel.pipeline import chain as ch
    from memory_kernel.pipeline.stages import write

    run_dir = tmp_path / "runs" / "0002"  # the run in flight
    run_dir.mkdir(parents=True)
    (run_dir / bonfire.EVIDENCE_INPUT).write_text(json.dumps({"m1": ch.evidence_row(session_rows()["m1"])}))
    ctx = make_ctx(run_dir, forward=forward_with_ground())
    assert ctx.cfg.forward_input == run_dir / "forward.json"
    m9 = session_rows()["m9"]
    ctx.sources = lambda: [m9]

    def project(_ctx, *, referee):
        referee(points_for_kiln()[0])
        return {"records": [m9]}, {}, {}

    monkeypatch.setattr(write, "project", project)
    asked: list = []
    monkeypatch.setattr(stage, "_ask", lambda _ctx, wires: asked.extend(wires) or [0.9 for _ in wires])
    stage.referents(ctx, True)
    d = json.loads((run_dir / "referents" / "decisions.json").read_text())
    assert d["decisions"] == {"u#0|arg1": {"join": "e-kiln"}}
    assert asked[0]["batches"][0]["state"]["earlier_first_mention"]["text"] == "a kiln arrived at the studio"


# -- the real _ask: a real cost recorder, a fake shard ledger whose archives read_jev_archive accepts -------------------


class FakeLedger:
    """The ledger boundary (``Shard.open``'s ``RoundBudget``): ``evaluate`` returns a synthetic archive in the shape
    ``read_jev_archive`` accepts and reports it as a new send (the shard's ``on_new_send``); ``fail`` raises instead."""

    def __init__(self, on_new_send, *, p_yes: float, fail: bool = False) -> None:
        from memory_kernel.pipeline.ledger import MODEL

        self.on_new_send, self.p_yes, self.fail = on_new_send, p_yes, fail
        self.policy = {"evidence_mode": "synthetic", "model": MODEL}
        self.closed = 0

    def evaluate(self, wire):
        from memory_kernel.construction.memory_jev_archive import RECEIPT_SCHEMA
        from memory_kernel.construction.memory_semantic_jev import seal, sha
        from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256
        from memory_kernel.pipeline.ledger import MODEL

        if self.fail:
            raise RuntimeError("provider 529")
        p = {"yes": self.p_yes, "no": round(1 - self.p_yes, 6)}
        answers = {q: {"type": "choice", "choice": max(p, key=p.get), "probabilities": p} for q in wire["questions"]}
        raw = canonical_json({"model": MODEL, "usage": {"input_tokens": 40, "output_tokens": 5, "cost": 0.0002},
                              "answers": answers})  # fmt: skip
        receipt = seal(
            {
                "schema_version": RECEIPT_SCHEMA,
                "evidence_mode": "synthetic",
                "request_sha256": content_sha256(wire),
                "raw_response_sha256": sha(raw.encode()),
                "ledger_record_sha256": None,
                "usage": {"input_tokens": 40, "output_tokens": 5, "cost_usd": 0.0002},
            }
        )
        archive = {"wire": canonical_json(wire), "raw": raw, "receipt": canonical_json(receipt)}
        self.on_new_send(archive)
        return archive

    def close(self) -> None:
        self.closed += 1


def real_ask_ctx(tmp_path, *, p_yes: float, fail: bool = False):
    """A ``runner.Ctx`` (its real ``cost``: a ``provenance.CostRecorder`` on a tmp ledger) whose shard is faked."""
    from memory_kernel.pipeline import runner

    cfg = SimpleNamespace(sample_id="conv-ember", ledger=tmp_path / "cost-ledger.jsonl", conv_root=tmp_path / "conv")
    ctx = runner.Ctx(cfg=cfg, concurrency=1)  # type: ignore[arg-type]
    made: list[FakeLedger] = []

    def shard(on_new_send, reserve_guard=None):  # noqa: ARG001 (the unit guard arms past the stage cap only)
        def open_(_manifest, _provenance):
            made.append(FakeLedger(on_new_send, p_yes=p_yes, fail=fail))
            return made[-1]

        return SimpleNamespace(load=lambda: ({}, {}, None), open=open_)

    ctx.shard = shard  # type: ignore[method-assign]
    return ctx, made


def kiln_wire():
    from memory_kernel.graph.referent_jev import question_wire
    from memory_kernel.graph.referents import Ground, HardCase

    g = Ground.from_row(forward_with_ground()["chain"]["ground"][0])
    case = HardCase(0, points_for_kiln()[0], ("e-kiln",))
    return question_wire(case, "e-kiln", rows=session_rows(), before=["m8"], ground={"e-kiln": g}, chat="tg-b",
                         names={"e-kiln": "kiln"})  # fmt: skip


def test_the_real_ask_records_its_cost_and_closes_the_ledger(tmp_path):
    ctx, made = real_ask_ctx(tmp_path, p_yes=0.9)
    assert stage._ask(ctx, [kiln_wire()]) == [0.9]
    rows = [json.loads(line) for line in (tmp_path / "cost-ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["stage"] == "referents" and rows[0]["status"] == "ok"
    assert rows[0]["providers"]["jev"] == {"requests": 1, "input_tokens": 40, "output_tokens": 5, "usd": 0.0002}
    assert [m.closed for m in made] == [1]


def test_a_failed_ask_still_closes_the_ledger_and_records_the_error(tmp_path):
    ctx, made = real_ask_ctx(tmp_path, p_yes=0.9, fail=True)
    with pytest.raises(RuntimeError, match="provider 529"):
        stage._ask(ctx, [kiln_wire()])
    rows = [json.loads(line) for line in (tmp_path / "cost-ledger.jsonl").read_text().splitlines()]
    assert rows[0]["stage"] == "referents" and rows[0]["status"] == "error" and "jev" in rows[0]["providers"]
    assert [m.closed for m in made] == [1]


# -- replay determinism of the wire: the archive is keyed on the sealed batch's content_sha256 ------------------------


def batch_shas(compiled: dict) -> list[str]:
    from memory_kernel.construction.mined_affordance_contract import content_sha256

    return [content_sha256(wire) for wire in compiled["batches"]]


def test_the_same_hard_case_builds_the_same_archived_wire():
    first, again = kiln_wire(), kiln_wire()
    assert first["batches"] and batch_shas(first) == batch_shas(again)


def test_another_candidate_builds_another_wire():
    """The candidate's uuid never reaches the wire: its name, mentions and chat do, so the other candidate differs in
    those (here: another name, mentioned in this chat)."""
    from memory_kernel.graph.referent_jev import question_wire
    from memory_kernel.graph.referents import Ground, HardCase

    kiln = Ground.from_row(forward_with_ground()["chain"]["ground"][0])
    urn = Ground("e-urn", ("tg-b",), ("oren",), "m8", "m8", "2026-08-20T09:00:00+00:00", 1)
    case = HardCase(0, points_for_kiln()[0], ("e-kiln", "e-urn"))
    other = question_wire(case, "e-urn", rows=session_rows(), before=["m8"], ground={"e-kiln": kiln, "e-urn": urn},
                          chat="tg-b", names={"e-kiln": "kiln", "e-urn": "urn"})  # fmt: skip
    assert set(batch_shas(other)).isdisjoint(batch_shas(kiln_wire()))
