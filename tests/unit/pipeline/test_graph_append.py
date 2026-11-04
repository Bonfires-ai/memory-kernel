"""Kernel 2.0: the append (``pipeline.graph_append``) runs one session into a bonfire whose state lives in Neo4j.

The bonfire is a real one on the test Neo4j (``test_commit.seed``: a two-chat bootstrap, committed run 0, with a full
build config stored in its ``BonfireMeta``). The build and the commit are stood in (a fake ``run_build`` that writes
what a sealed run would and records the config it was given; a fake ``commit_append``), the lease and the spend are an
in-memory ``MemoryKernelOps``, so the order of the steps, what the run dir holds and what is left behind are what is
pinned. $0: no stage child, no Jev, no Voyage. Invented names only."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_kernel.ops.base import LeaseLost
from memory_kernel.ops.memory import MemoryKernelOps
from memory_kernel.pipeline import bonfire, graph_append, provenance
from memory_kernel.pipeline.bonfire import AppendBusy, BonfireError, PendingSession
from memory_kernel.pipeline.config import LAST_STAGE, BuildConfig
from memory_kernel.pipeline.embed import message_texts
from unit.pipeline.test_commit import neo, seed, vector

NEW_IDS = ("a1", "a2")
FORWARD_KEYS = {
    "schema",
    "chain_run",
    "session_id",
    "session_number",
    "chat",
    "first_order",
    "explicit_since",
    "boundary",
    "rolesets",
    "voyage_cache",
    "chain",
    "totals",
    "skipped_ids",
}


def session_of(ids: tuple[str, ...] = NEW_IDS, *, chat: str = "k2") -> dict:
    return {
        "conversation_id": chat,
        "messages": [
            {
                "message_id": mid,
                "speaker": "Dov" if i % 2 else "Ana",
                "text": f"line {mid} about the kiln",
                "timestamp": f"2026-06-06T12:{i:02d}:00+00:00",
            }
            for i, mid in enumerate(ids)
        ],
    }


def fake_ledger_summary(ledger: Path) -> dict:
    """The cost ledger stood in: the file holds the paid total as a number, or ``corrupt``."""
    text = ledger.read_text().strip()
    if text == "corrupt":
        raise ValueError("cost ledger chain broken at line 1")
    return {"paid_usd": float(text)}


class Bonfire:
    """A seeded 2.0 bonfire: its group, a ``kernel_ops`` record, the Neo4j settings and the scratch root."""

    def __init__(self, target: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, ceiling=None) -> None:
        self.target, self.tmp_path, self.group, self.mp = target, tmp_path, target.group, monkeypatch
        self.scratch = tmp_path / "scratch"
        self.chain, self.sources = neo(target, lambda d: seed(d, self.group))
        config = BuildConfig(
            sample_id=self.group,
            conversation=tmp_path / "seed.json",
            group=self.group,
            workdir=tmp_path / "boot",
            root=tmp_path,
            referent_policy="structure",
            boundary_policy="segmenter",
        )
        self.set_meta(build_config_json=json.dumps(config.to_json(), sort_keys=True))
        self.ops = MemoryKernelOps()
        self.ops.create(self.group, ceiling_usd=ceiling, state="ready")
        self.overrides = {
            "neo4j_uri": target.uri,
            "neo4j_user": target.user,
            "neo4j_password": target.password,
        }
        self.log: list[str] = []
        self.runs: list[SimpleNamespace] = []  # what each fake build saw
        self.commits: list[Any] = []
        self.build_status = "through"
        self.build_raises: BaseException | None = None
        self.commit_raises: BaseException | None = None
        self.ledger: str | None = "0.25"  # what the build's cost ledger holds (None: none written)
        self.on_build: Any = None
        monkeypatch.setattr(graph_append, "run_build", self._run_build)
        monkeypatch.setattr(graph_append, "commit_input", self._commit_input)
        monkeypatch.setattr(graph_append, "commit_append", self._commit_append)
        monkeypatch.setattr(graph_append.provenance, "ledger_summary", fake_ledger_summary)
        real_take, real_world = self.ops.take_lease, graph_append.read_world

        def take(*a: Any, **k: Any) -> Any:
            self.log.append("lease")
            return real_take(*a, **k)

        async def world(driver: Any, group: str) -> Any:
            self.log.append("chain")
            return await real_world(driver, group)

        self.ops.take_lease = take  # type: ignore[method-assign]
        monkeypatch.setattr(graph_append, "read_world", world)

    def chain_meta(self) -> dict:
        from memory_kernel.graph.state_read import read_meta

        return neo(self.target, lambda d: read_meta(d, self.group))

    def bonfire_id(self) -> str:
        return self.group

    def set_meta(self, **props: Any) -> None:
        async def go(driver: Any) -> None:
            async with driver.session() as s:
                await (await s.run("MATCH (m:BonfireMeta {group_id: $g}) SET m += $p", g=self.group, p=props)).consume()

        neo(self.target, go)

    def _run_build(self, cfg: BuildConfig, **_kw: Any) -> dict:
        self.log.append("build")
        forward = json.loads(Path(cfg.forward_input).read_bytes())
        cfg.conv_root.mkdir(parents=True, exist_ok=True)  # what the stages would have made
        if self.ledger is not None:
            cfg.ledger.write_text(self.ledger)
        self.runs.append(
            SimpleNamespace(
                cfg=cfg,
                forward=forward,
                run_dir=Path(cfg.forward_input).parent,
                files=sorted(p.name for p in Path(cfg.forward_input).parent.iterdir()),
                cache=json.loads(Path(forward["voyage_cache"]).read_bytes()),
                held=self.ops.get(self.group).held,
                index=json.loads(cfg.bonfire_index_record.read_bytes()),
            )
        )
        if self.on_build is not None:
            self.on_build(cfg)
        if self.build_raises is not None:
            raise self.build_raises
        return {"status": self.build_status, "stage": LAST_STAGE}

    def _commit_input(self, run_root: Path, **kw: Any) -> SimpleNamespace:
        self.log.append("commit_input")
        return SimpleNamespace(run_root=run_root, **kw)

    async def _commit_append(self, _driver: Any, ci: Any) -> dict:
        self.log.append("commit")
        self.commits.append(ci)
        if self.commit_raises is not None:
            raise self.commit_raises
        return {"created": {"nodes": 3, "edges": 2}, "keys": {}, "guard": {}, "already_committed": False}

    def append(self, session: dict | None = None, **kw: Any) -> dict:
        kw.setdefault("owner", "worker-1")
        kw.setdefault("allow_parser_change", True)
        return graph_append.append(
            self.group,
            session or session_of(),
            ops=self.ops,
            scratch_root=self.scratch,
            overrides=self.overrides,
            **kw,
        )

    def scratch_left(self) -> list[str]:
        return (
            sorted(str(p.relative_to(self.scratch)) for p in self.scratch.rglob("*")) if self.scratch.exists() else []
        )


@pytest.fixture
def bf(neo4j_target, tmp_path, monkeypatch) -> Bonfire:
    return Bonfire(neo4j_target, tmp_path, monkeypatch)


# -- the lease, the read, the plan ---------------------------------------------------------------------------------------


def test_append_takes_the_lease_before_reading_the_chain(bf):
    out = bf.append()
    assert out["status"] == "appended"
    assert bf.log == ["lease", "chain", "build", "commit_input", "commit"]
    lease = bf.ops.get(bf.group).lease
    assert lease is None  # a committed run clears it


def test_the_lease_names_the_next_run_and_the_session_key(bf):
    seen = []
    bf.on_build = lambda _cfg: seen.append(bf.ops.get(bf.group).lease)
    out = bf.append()
    (lease,) = seen
    assert (lease.run, lease.owner, lease.session_id) == (1, "worker-1", graph_append.session_key(session_of()))
    assert out["run"] == "0001" and out["sessions"] == 4 and out["new_messages"] == 2 and out["skipped_ids"] == []
    assert len(lease.session_id) == 16 and graph_append.session_key(session_of(("a1", "a3"))) != lease.session_id


def test_session_key_is_the_conversation_and_the_ids(bf):
    a, b = session_of(), session_of()
    b["messages"][0]["text"] = "other words"  # the text is not part of the key
    assert graph_append.session_key(a) == graph_append.session_key(b)
    assert graph_append.session_key(a) != graph_append.session_key(session_of(chat="k1"))
    assert graph_append.session_key(a) != graph_append.session_key(session_of(("a2", "a1")))


def test_no_such_bonfire_takes_no_lease(bf):
    with pytest.raises(BonfireError, match="no such bonfire"):
        graph_append.append(
            "mk20-test-nothere", session_of(), ops=bf.ops, scratch_root=bf.scratch, owner="w", overrides=bf.overrides
        )
    assert bf.log == [] and bf.ops.get(bf.group).lease is None


def test_a_live_lease_of_another_owner_is_busy_and_reads_nothing(bf):
    bf.ops.take_lease(bf.group, owner="other", session_id="s", run=1, committed_run=0, ttl_s=60)
    bf.log.clear()
    with pytest.raises(AppendBusy):
        bf.append()
    assert "chain" not in bf.log and bf.scratch_left() == []


def test_duplicate_session_releases_the_lease_and_spends_nothing(bf):
    built = session_of(("b1", "b2"), chat="k1")  # both stored by the bootstrap
    out = bf.append(built)
    assert (out["status"], out["run"], out["new_messages"], out["result"]) == ("duplicate", None, 0, None)
    assert out["sessions"] == 3  # the bonfire's session count
    rec = bf.ops.get(bf.group)
    assert rec.lease is None and rec.held is None and rec.spent_usd == 0 and bf.runs == []
    bf.append(session_of())  # nothing is pending: the next session runs
    assert len(bf.runs) == 1


def test_an_overlapping_session_is_refused_unless_skipping_and_releases_the_lease(bf):
    both = session_of(("b1", "a1"), chat="k1")
    with pytest.raises(BonfireError, match=bonfire.OVERLAP):
        bf.append(both)
    assert bf.ops.get(bf.group).lease is None and bf.ops.get(bf.group).held is None
    out = bf.append(both, on_built="skip")
    assert out["status"] == "appended" and out["skipped_ids"] == ["b1"] and out["new_messages"] == 1


def test_a_malformed_session_is_refused_before_the_lease(bf):
    with pytest.raises(BonfireError, match="has no messages"):
        bf.append({"conversation_id": "k2", "messages": []})
    assert bf.log == []


# -- what the run holds --------------------------------------------------------------------------------------------------


def test_forward_input_keys_are_unchanged(bf):
    bf.append()
    (run,) = bf.runs
    assert set(run.forward) == FORWARD_KEYS
    assert run.forward["chain_run"] == 0 and run.forward["session_number"] == 4 and run.forward["chat"] == "k2"
    assert run.forward["totals"] == {"messages": 7, "episodes": 4}
    assert run.forward["boundary"]["policy"] == "segmenter"
    assert [f for f in run.files if f not in {bf.group, "cost-ledger.jsonl"}] == sorted(
        [
            bonfire.INPUT_COPY,
            bonfire.RUN_CONVERSATION,
            bonfire.FORWARD_INPUT,
            bonfire.EVIDENCE_INPUT,
            graph_append.INDEX_RECORD,
            "embeddings",
        ]
    )


def test_settled_rolesets_carry_their_stored_answer(bf):
    bf.append()
    rolesets = bf.runs[0].forward["rolesets"]
    settled = {"label", "p_yes", "compiled_sha256"}
    assert settled <= set(rolesets["like.01"]) and set(rolesets["like.01"]) - settled == {"gloss", "examples"}
    assert rolesets["like.01"]["label"] == "occurrence" and rolesets["like.01"]["compiled_sha256"] == "1a" * 32
    assert set(rolesets["meet.01"]) == {"gloss", "examples"}  # stored without an answer: 1.8's keys exactly


def test_the_run_config_is_the_scratch_graph_mode_append(bf):
    bf.append()
    (run,) = bf.runs
    cfg = run.cfg
    root = bf.scratch / bf.group / "runs" / "0001"
    assert run.run_dir == root and cfg.workdir == root and cfg.conv_root == root / bf.group
    assert (cfg.shard_dir, cfg.capture_cache, cfg.write_mode, cfg.graph_commit) == (None, None, "append", True)
    assert cfg.append_base_usd == "0" and cfg.append_messages == 2 and cfg.through == LAST_STAGE
    assert cfg.bonfire_index_record == root / graph_append.INDEX_RECORD
    assert run.index == {"recipe_tag": bf.chain_meta()["recipe_tag"]}
    assert cfg.neo4j_uri == bf.target.uri and cfg.boundary_policy == "segmenter"
    assert cfg.spend_ceiling_usd is None  # no ceiling on the record


def test_the_ceiling_the_run_may_spend_is_the_holds_headroom_plus_the_estimate(bf):
    bf.ops.set_ceiling(bf.group, Decimal("5"))
    bf.append()
    cfg = bf.runs[0].cfg
    assert Decimal(cfg.spend_ceiling_usd) == Decimal(5)  # nothing spent yet: the whole ceiling


def test_voyage_cache_is_seeded_from_tail_embeddings(bf):
    bf.append()
    cache = bf.runs[0].cache
    tail = bf.chain.chats["k2"].tail  # the chat's last stored messages (b4 only)
    k1_tail = bf.chain.chats["k1"].tail
    expect = {text: vector(text) for text in message_texts([*k1_tail, *tail]).values()}
    assert expect.items() <= cache.items()  # the tails, and the chats' last episodes too


def test_voyage_cache_skips_a_vector_whose_text_changed(bf):
    async def go(driver: Any) -> None:
        async with driver.session() as s:
            await (
                await s.run(
                    "MATCH (e:Episodic {group_id: $g, message_id: 'b4'}) SET e.message_text_sha256 = 'stale'",
                    g=bf.group,
                )
            ).consume()

    neo(bf.target, go)
    bf.append()
    texts = message_texts(bf.chain.chats["k2"].tail).values()
    assert not set(texts) & set(bf.runs[0].cache)


def test_referents_read_evidence_from_the_run_dir(bf):
    seen: dict[str, Any] = {}
    bf.on_build = lambda cfg: seen.update(rows=bonfire.message_rows(Path(cfg.forward_input).parent))
    bf.append()
    assert seen["rows"] == bf.chain.evidence and seen["rows"]


def test_the_commit_gets_the_chain_the_run_started_from(bf):
    bf.append()
    (ci,) = bf.commits
    assert (ci.group, ci.run) == (bf.bonfire_id(), 1)
    assert ci.generation == bf.chain_meta()["generation"] == "gen-seed-1"
    assert ci.chain.run == 0 and ci.chain.sessions == 3 and ci.run_root.name == bf.group


# -- the scratch dir -----------------------------------------------------------------------------------------------------


def test_scratch_is_empty_after_an_append(bf):
    bf.append()
    assert bf.scratch_left() == []


def test_scratch_is_empty_after_a_stop(bf):
    bf.build_status = "stopped"
    bf.append()
    assert bf.scratch_left() == []


def test_scratch_is_empty_after_an_exception(bf):
    bf.build_raises = RuntimeError("the build died")
    with pytest.raises(RuntimeError, match="the build died"):
        bf.append()
    assert bf.scratch_left() == []
    bf.build_raises, bf.commit_raises = None, BonfireError("commit refused")
    with pytest.raises(BonfireError, match="commit refused"):
        bf.append()
    assert bf.scratch_left() == []


def test_run_dir_survives_until_the_commit_returns(bf):
    seen: dict[str, Any] = {}

    async def commit(_driver: Any, ci: Any) -> dict:
        seen["root_exists"] = ci.run_root.is_dir()
        return {"already_committed": False}

    bf.mp.setattr(graph_append, "commit_append", commit)
    bf.append()
    assert seen["root_exists"] is True and bf.scratch_left() == []


# -- the stop, the crash, the retry --------------------------------------------------------------------------------------


def test_stopped_run_settles_spend_and_keeps_the_session_pending(bf):
    bf.build_status = "stopped"
    out = bf.append()
    assert (out["status"], out["run"]) == ("stopped", "0001")
    assert bf.commits == []  # nothing was committed
    rec = bf.ops.get(bf.group)
    assert rec.spent_usd == Decimal("0.25") and rec.held is None  # settled
    assert rec.lease is not None and rec.lease.run == 1  # kept: the stopped session names the run
    with pytest.raises(PendingSession):  # any other session waits behind it
        bf.append(session_of(("c1", "c2")), owner="worker-2")
    again = bf.append()  # the same session, even the same owner, takes the run back
    assert again["status"] == "stopped"


def test_a_stopped_session_sent_again_takes_the_same_run_and_lands(bf):
    bf.build_status = "stopped"
    bf.append()
    bf.build_status = "through"
    out = bf.append(owner="worker-2")
    assert out["status"] == "appended" and out["run"] == "0001"
    assert [r.cfg.workdir.name for r in bf.runs] == ["0001", "0001"]
    rec = bf.ops.get(bf.group)
    assert rec.lease is None and rec.spent_usd == Decimal("0.50")


def test_retry_after_a_crash_takes_the_same_run_and_clears_leftover_scratch(bf):
    bf.build_raises = RuntimeError("worker killed")
    with pytest.raises(RuntimeError):
        bf.append()
    # a killed process leaves its run dir behind (no finally ran)
    leftover = bf.scratch / bf.group / "runs" / "0001"
    (leftover / bf.group / "05-write").mkdir(parents=True)
    (leftover / bf.group / "05-write" / "summary.json").write_text("{}")
    bf.build_raises = None
    bf.on_build = lambda cfg: bf.log.append(
        f"fresh:{not (Path(cfg.forward_input).parent / bf.group / '05-write').exists()}"
    )
    out = bf.append(owner="worker-2")
    assert out["run"] == "0001" and "fresh:True" in bf.log
    assert bf.scratch_left() == []


def test_a_crash_settles_the_hold_and_the_lease_names_the_session(bf):
    bf.build_raises = RuntimeError("worker killed")
    with pytest.raises(RuntimeError):
        bf.append()
    rec = bf.ops.get(bf.group)
    assert rec.held is None and rec.spent_usd == Decimal("0.25") and rec.lease.run == 1


def test_an_already_committed_run_is_a_success(bf):
    async def commit(_driver: Any, _ci: Any) -> dict:
        return {"created": {"nodes": 0, "edges": 0}, "already_committed": True}

    bf.mp.setattr(graph_append, "commit_append", commit)
    out = bf.append()
    assert out["status"] == "appended" and out["result"]["committed"]["already_committed"] is True
    assert bf.ops.get(bf.group).lease is None  # cleared: the run landed


def test_a_refused_commit_propagates_and_keeps_the_session_pending(bf):
    from memory_kernel.pipeline.commit import StaleCommit

    bf.commit_raises = StaleCommit("another commit landed first")
    with pytest.raises(StaleCommit):
        bf.append()
    rec = bf.ops.get(bf.group)
    assert rec.held is None and rec.spent_usd == Decimal("0.25") and rec.lease is not None


def test_a_lost_lease_commits_nothing(bf):
    class Lost:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def start(self) -> Lost:
            return self

        def check(self) -> None:
            raise LeaseLost("lost")

        def stop(self) -> None:
            pass

    bf.mp.setattr(graph_append, "LeaseKeeper", Lost)
    with pytest.raises(LeaseLost):
        bf.append()
    assert bf.commits == []


# -- spend ---------------------------------------------------------------------------------------------------------------


def test_the_hold_is_the_estimate_and_the_settle_is_the_paid_total(bf):
    bf.append()
    (run,) = bf.runs
    assert run.held == (1, bonfire.append_estimate(2, []))
    rec = bf.ops.get(bf.group)
    assert rec.held is None and rec.spent_usd == Decimal("0.25")
    (entry,) = bf.ops.history(bf.group)
    assert (entry["run"], entry["new_messages"], entry["spent_usd"]) == (1, 2, Decimal("0.25"))


def test_ceiling_refusal_comes_from_kernel_ops(bf):
    bf.ops.set_ceiling(bf.group, Decimal("0.01"))
    with pytest.raises(BonfireError, match=bonfire.CEILING):
        bf.append()
    rec = bf.ops.get(bf.group)
    assert rec.held is None and rec.spent_usd == 0 and rec.lease is None  # nothing ran, nothing is pending
    assert bf.runs == [] and bf.scratch_left() == []


def test_a_refused_parser_change_holds_nothing(bf):
    bf.set_meta(extraction_model_json=json.dumps({"name": "some_other_model", "version": "1"}))
    with pytest.raises(BonfireError, match="some_other_model"):
        bf.append(allow_parser_change=False)
    assert bf.ops.get(bf.group).held is None and bf.ops.get(bf.group).lease is None and bf.runs == []


@pytest.mark.parametrize(
    ("recorded", "env", "allow", "refused"),
    [
        ("en_core_web_sm", "", False, True),  # a bonfire parsed with sm, the default now trf
        (None, "", False, True),  # no record = built before 1.4 = sm
        ("en_core_web_sm", "en_core_web_sm", False, False),  # the env opt-out
        (None, "en_core_web_sm", False, False),
        ("en_core_web_sm", "", True, False),  # --allow-parser-change
        ("en_core_web_trf", "", False, False),  # same model (a version difference alone is a note)
    ],
)
def test_require_extraction_model_reads_the_meta(monkeypatch, recorded, env, allow, refused):
    meta = {"extraction_model_json": None if recorded is None else json.dumps({"name": recorded, "version": "0.0.1"})}
    monkeypatch.setenv("MK_EXTRACTION_SPACY_MODEL", env)
    if not refused:
        bonfire.require_extraction_model(meta, allow_parser_change=allow)
        return
    with pytest.raises(BonfireError, match="en_core_web_trf") as caught:
        bonfire.require_extraction_model(meta, allow_parser_change=allow)
    assert ("before 1.4" in str(caught.value)) == (recorded is None)


def test_the_neo4j_password_is_required(bf):
    with pytest.raises(BonfireError, match="neo4j_password"):
        graph_append.append(bf.group, session_of(), ops=bf.ops, scratch_root=bf.scratch, owner="w", overrides={})
    assert bf.log == []


def test_a_real_format_cost_ledger_settles_what_ledger_summary_reads(bf, monkeypatch):
    """The ledger the stages write (``provenance.CostRecorder``, chained rows): the settled spend is its ``paid_usd``."""
    monkeypatch.undo()  # the real ``provenance.ledger_summary``, not the numeric stand-in
    monkeypatch.setattr(graph_append, "run_build", bf._run_build)
    monkeypatch.setattr(graph_append, "commit_input", bf._commit_input)
    monkeypatch.setattr(graph_append, "commit_append", bf._commit_append)

    paid: list[float] = []

    def write_rows(cfg: BuildConfig) -> None:
        for stage, usd in (("taxonomy", 0.0123), ("jev1", 0.1)):
            with provenance.CostRecorder(stage, run="r", ledger=cfg.ledger) as cost:
                cost.add("openrouter", requests=2, input_tokens=10, output_tokens=5, usd=usd)
        paid.append(provenance.ledger_summary(cfg.ledger)["paid_usd"])  # the run dir is gone after the append

    bf.ledger, bf.on_build = None, write_rows
    bf.append()
    assert paid == [pytest.approx(0.1123)]
    assert bf.ops.get(bf.group).spent_usd == Decimal(str(round(paid[0], 9)))


# -- fix round: pending leases, crashed attempts' spend, the ledger, the id, the seed ------------------------------------


def test_a_refused_resend_of_a_pending_session_keeps_it_pending(bf):
    bf.build_status = "stopped"
    bf.append()  # Y stopped: its lease is the pending marker
    bf.ops.set_ceiling(bf.group, Decimal("0.01"))
    with pytest.raises(BonfireError, match=bonfire.CEILING):  # Y resent, refused at the hold
        bf.append()
    lease = bf.ops.get(bf.group).lease
    assert lease is not None and lease.run == 1  # still pending, not cleared
    with pytest.raises(PendingSession):  # so Z cannot land before Y
        bf.append(session_of(("c1", "c2")), owner="worker-2")


def test_a_refusal_of_a_fresh_session_still_clears_the_lease(bf):
    bf.ops.set_ceiling(bf.group, Decimal("0.01"))
    with pytest.raises(BonfireError, match=bonfire.CEILING):
        bf.append()
    assert bf.ops.get(bf.group).lease is None


def leave_crashed_attempt(bf, ledger: str, *, run: int = 1) -> Path:
    """What a killed worker leaves: a hold for ``run`` in kernel_ops and its scratch run dir with a cost ledger."""
    bf.ops.hold(bf.group, run=run, estimate_usd=Decimal("0.5"))
    left = bf.scratch / bf.group / "runs" / f"{run:04d}"
    (left / bf.group / "05-write").mkdir(parents=True)
    (left / "cost-ledger.jsonl").write_text(ledger)
    return left


def test_a_crash_after_the_commit_is_settled_before_the_dup_clears_the_dir(bf):
    left = leave_crashed_attempt(bf, "0.3")
    out = bf.append(session_of(("b1", "b2"), chat="k1"))  # its commit landed: the retry is a dup
    assert out["status"] == "duplicate" and not left.exists() and bf.scratch_left() == []
    rec = bf.ops.get(bf.group)
    assert rec.held is None and rec.spent_usd == Decimal("0.3")


def test_a_crash_before_the_commit_is_settled_and_the_retry_pays_its_own(bf):
    left = leave_crashed_attempt(bf, "0.3")
    out = bf.append()
    assert out["status"] == "appended" and out["run"] == "0001" and not left.exists()
    rec = bf.ops.get(bf.group)
    assert rec.held is None and rec.spent_usd == Decimal("0.55")


def test_a_leftover_for_another_run_is_deleted_without_a_settle(bf):
    left = leave_crashed_attempt(bf, "0.3", run=7)
    bf.ops.settle(bf.group, run=7, spent_usd=Decimal(0), new_messages=0, budget_usd=Decimal(0))
    bf.append()
    assert not left.exists() and bf.ops.get(bf.group).spent_usd == Decimal("0.25")


def test_a_corrupt_ledger_settles_the_holds_estimate(bf, caplog):
    bf.ledger = "corrupt"
    with caplog.at_level("ERROR"):
        bf.append()
    assert bf.ops.get(bf.group).spent_usd == bonfire.append_estimate(2, [])
    assert "does not read" in caplog.text


def test_a_missing_ledger_settles_zero(bf):
    bf.ledger = None
    bf.append()
    assert bf.ops.get(bf.group).spent_usd == 0


def test_a_leftover_with_a_corrupt_ledger_settles_its_estimate(bf):
    leave_crashed_attempt(bf, "corrupt")
    bf.append(session_of(("b1", "b2"), chat="k1"))
    assert bf.ops.get(bf.group).spent_usd == Decimal("0.5")


@pytest.mark.parametrize("bad", ["", "..", "a/b", "../x", ".", "a b"])
def test_a_bonfire_id_that_is_not_a_plain_name_is_refused_before_the_lease(bf, bad):
    with pytest.raises(BonfireError, match="plain bonfire id"):
        graph_append.append(bad, session_of(), ops=bf.ops, scratch_root=bf.scratch, owner="w", overrides=bf.overrides)
    assert bf.log == [] and bf.scratch_left() == []


def test_an_open_episode_longer_than_the_tail_embeds_nothing_live(bf, monkeypatch):
    from memory_kernel.graph import state_read

    monkeypatch.setattr(state_read, "TAIL", 1)  # the tail keeps one row; the last episode of k1 holds b5, b6, b7
    bf.append()
    cache = bf.runs[0].cache
    rows = [r for r in bf.sources if r["message_id"] in {"b4", "b5", "b6", "b7"}]
    assert {message_texts([r])[r["message_id"]] for r in rows} <= set(cache)


def test_the_seeds_settle_row_does_not_scale_the_first_appends_estimate(bf):
    bf.ops.set_ceiling(bf.group, Decimal(5))
    bf.ops.hold(bf.group, run=0, estimate_usd=Decimal(3))
    bf.ops.settle(bf.group, run=0, spent_usd=Decimal("1.2"), new_messages=2, budget_usd=Decimal(3))
    ids = tuple(f"n{i}" for i in range(14))
    bf.append(session_of(ids))
    assert bf.runs[0].held == (1, bonfire.append_budget(14, planned_jev_usd=Decimal(0)))  # the floor, not 14 x 1.50
    assert bf.ops.get(bf.group).lease is None


def test_a_prior_append_row_still_scales_the_estimate(bf):
    bf.ops.hold(bf.group, run=0, estimate_usd=Decimal(3))
    bf.ops.settle(bf.group, run=0, spent_usd=Decimal("1.2"), new_messages=2, budget_usd=Decimal(3))
    bf.ops.hold(bf.group, run=1, estimate_usd=Decimal(1))
    bf.ops.settle(bf.group, run=1, spent_usd=Decimal("0.1"), new_messages=2, budget_usd=Decimal(1))
    bf.set_meta(committed_run=1)
    ids = tuple(f"n{i}" for i in range(14))
    bf.append(session_of(ids))
    assert bf.runs[0].held == (2, Decimal("7.000000"))  # run 1's $0.50 per message x 14
