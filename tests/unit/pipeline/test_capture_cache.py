"""The per-job capture cache: a cumulative capture runs the engine only for jobs it has not seen.

Offline: a fake engine counts the job calls (``A-*`` phase-A requests, ``B-*`` phase-B redos), the phase-B redo is a
counting stand-in, and the sources are invented one-sentence messages with hand-made stored parses."""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

from memory_kernel.construction.fcg.native_warm_source_units import SYNTAX_SCHEMA
from memory_kernel.construction.memory_discourse_contract import discourse_source
from memory_kernel.construction.memory_semantic_jev import seal
from memory_kernel.construction.mined_affordance_contract import canonical_json
from memory_kernel.pipeline import engine as eng
from memory_kernel.pipeline.stages import _capture_cache as cc
from memory_kernel.pipeline.stages import _capture_shards as cs
from memory_kernel.pipeline.stages import _parse_capture as pc
from memory_kernel.pipeline.stages import parse

LIVE = "http://127.0.0.1:9801"
SCOPE = {"bonfire_id": "mk-demo", "conversation_id": "demo"}
#: one word per message; "Slowly" times out in phase A, so its job also gets a phase-B redo
WORDS = ["Hello", "Slowly", "Thanks", "Welcome"]


def _source(order: int) -> dict:
    return discourse_source(
        scope=SCOPE,
        message_id=f"m{order}",
        order=order,
        text=f"{WORDS[order]}!",
        speaker="Ana",
        session_id="demo-session_1",
        timestamp="2026-01-01T00:00:00+00:00",
    )


def _syntax(text: str, word: str) -> dict:
    tree = [
        {"end": 2, "node_id": 1, "node_type": "phrase", "parent_constituent": None, "phrase_types": ["NP"],
         "start": 0, "string": text},
        {"dependency_head": 2, "dependency_label": "ROOT", "end": 1, "lemma": word.lower(), "lex_class": "NN",
         "named_entity_type": "", "node_id": 2, "node_type": "leaf", "parent_constituent": 1, "phrase_types": None,
         "start": 0, "string": word},
        {"dependency_head": 2, "dependency_label": "punct", "end": 2, "lemma": "!", "lex_class": ".",
         "named_entity_type": "", "node_id": 3, "node_type": "leaf", "parent_constituent": 1, "phrase_types": None,
         "start": 1, "string": "!"},
    ]  # fmt: skip
    return seal(
        {
            "schema_version": SYNTAX_SCHEMA,
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "parser": [LIVE, "en_benepar"],
            "parser_calls": 1,
            "trees": [{"sent_id": 0, "tree": tree}],
        }
    )


class FakeEngine:
    """What ``run_shard`` touches of ``engine.Engine``; ``http`` records every job label it is sent."""

    process = None

    def __init__(self, calls: list):
        self.calls, self.parser = calls, None

    def launch(self):
        return {}

    def arm(self, parser):
        self.parser = parser
        return {}

    def set_parser(self, parser):
        self.parser = parser

    def readback(self):
        return [], {"parser": self.parser}

    def reset_ids(self):
        pass

    def idle_cpu(self, _s):
        return 0.0

    def teardown(self):
        return {"port_listening_after": False, "own_fcg_after": []}

    def close_attempt(self):
        return "teardown", self.teardown()

    def http(self, request, label, timeout=None, allowed=None):
        self.calls.append(label)
        slow = "Slowly" in request
        row = {"id": "p00", "seconds": 0.25, "status": "returned", "meaningKind": "TIME-OUT" if slow else "nil"}
        return {"status": "ok", "payload": {"passes": [row]}}


def _fake_redo(engine, job, analysis, pass_id):
    engine.http("redo", f"B-{job['origin']['source_unit_id'][-8:]}-{pass_id}")
    text = job["source"]["text"]
    return {
        "status": "ok",
        "seconds": 0.5,
        "segments": [{"text": text, "status": "ok", "seconds": 0.5, "jobs": [{"request_sha256": "c" * 64}]}],
        "frames": [],
    }


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A site with dummy engine files, the stored parses of every word, snapshot pins, and the fakes wired in."""
    calls: list[str] = []
    (tmp_path / "bin").write_bytes(b"engine")
    (tmp_path / "img").write_bytes(b"image")
    ecfg = eng.EngineConfig(
        port=9701,
        binary=tmp_path / "bin",
        image=tmp_path / "img",
        parser_url=LIVE,
        binary_sha256=eng.sha_file(tmp_path / "bin"),
        image_sha256=eng.sha_file(tmp_path / "img"),
    )
    site = pc.Site(root=tmp_path, engine=ecfg, parser_lock=tmp_path / "p.lock", parser=None)
    for order, word in enumerate(WORDS):
        path = tmp_path / "syntax" / f"{order:04d}" / "syntax-00.json"
        path.parent.mkdir(parents=True)
        path.write_text(canonical_json(_syntax(f"{word}!", word)))
    pins = parse.snapshot_pins_for(tmp_path / "pins")
    monkeypatch.setattr(pc, "Engine", lambda cfg, attempt_dir: FakeEngine(calls))
    monkeypatch.setattr(pc, "_process_gate", lambda port: None)
    monkeypatch.setattr(pc, "parser_probe", lambda engine, live: {"parser": [live, "en_benepar"]})
    monkeypatch.setattr(pc, "redo_pass", _fake_redo)
    monkeypatch.setattr(pc.time, "monotonic", lambda: 100.0)  # receipts' wall-clock seconds are constant
    monkeypatch.setattr(parse, "site_for", lambda ctx: site)
    monkeypatch.setattr(parse, "_compat_valid", lambda runs: True)
    monkeypatch.setattr(cs, "usable_cpus", lambda: 16)
    return types.SimpleNamespace(root=tmp_path, site=site, pins=pins, calls=calls)


def _capture(world, name: str, n: int, *, cache: Path | None, shards: int = 1) -> tuple[Path, dict]:
    """``parse.capture`` over the first ``n`` messages in a fresh conversation root ``name``."""
    root = world.root / name
    paths = {s: root / s for s in ("capture", "runs", "select")}
    paths["syntax"] = world.root / "syntax"
    paths["select"].mkdir(parents=True)
    sources = [_source(order) for order in range(n)]
    (paths["select"] / "sources.json").write_text(canonical_json(sources))
    cfg = types.SimpleNamespace(
        capture_shards=shards,
        snapshot_pins=world.pins,
        conv_root=root,
        root=world.root,
        capture_cache=cache,
        capture_slot=None,
        sample_id="demo",
    )
    ctx = types.SimpleNamespace(cfg=cfg, path=paths.__getitem__, counts=types.SimpleNamespace(messages=n))
    world.calls.clear()
    return paths["capture"], parse.capture(ctx)


def _merged(capture: Path) -> dict[str, bytes]:
    files = sorted(capture.glob("[0-9][0-9][0-9][0-9]/*.json"))
    return {str(p.relative_to(capture)): p.read_bytes() for p in files}


def _jobs(calls):
    return [c for c in calls if c[:2] in ("A-", "B-")]


# -- the key ----------------------------------------------------------------------------------------------------------


def _manifest() -> dict:
    return {
        "schema_version": pc.HYBRID_CAPTURE_SCHEMA,
        "sources_sha256": "1" * 64,
        "source_count": 3,
        "syntax_from": "syntax",
        "t_pass": 5,
        "deadline_cap": 30,
        "encoding": "B1",
        "binary_sha256": "2" * 64,
        "image_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "walker_sha256": "5" * 64,
        "code_pins": {"src/a.py": "6" * 64},
        "parsers": {"phase_a": eng.DEAD_PARSER, "phase_b": LIVE},
        "snapshot_pins": {"file": "snapshot-pins.json", "sha256": "7" * 64, "files": {"src/a.py": "6" * 64}},
        "syntax_manifest_sha256": "8" * 64,
        "content_sha256": "9" * 64,
    }


def _keys(manifest: dict) -> tuple[str, str]:
    pins = cc.pins_of(manifest)
    return (
        cc.key_a(pins, job_sha256="a" * 64, request_sha256="b" * 64, deadline_s=6.0),
        cc.key_b(pins, job_sha256="a" * 64, phase_a_sha256="c" * 64),
    )


PIN_CHANGES = {
    "t_pass": 4,
    "deadline_cap": 20,
    "encoding": "B2",
    "binary_sha256": "e" * 64,
    "image_sha256": "e" * 64,
    "config_sha256": "e" * 64,
    "walker_sha256": "e" * 64,
    "code_pins": {"src/a.py": "e" * 64},
    "schema_version": "other/v1",
}


@pytest.mark.parametrize("field", sorted(PIN_CHANGES))
def test_the_key_changes_when_a_manifest_pin_changes(field):
    changed = {**_manifest(), field: PIN_CHANGES[field]}
    a0, b0 = _keys(_manifest())
    a1, b1 = _keys(changed)
    assert a0 != a1 and b0 != b1


def test_the_key_changes_with_the_package_tree_the_parsers_and_the_id_reset(monkeypatch):
    a0, b0 = _keys(_manifest())
    tree = _manifest()
    tree["snapshot_pins"] = {**tree["snapshot_pins"], "files": {"src/a.py": "6" * 64, "src/b.py": "e" * 64}}
    assert _keys(tree)[0] != a0 and _keys(tree)[1] != b0
    dead = _manifest()
    dead["parsers"] = {**dead["parsers"], "phase_a": "http://127.0.0.1:1"}
    assert _keys(dead) == (_keys(dead)[0], b0) and _keys(dead)[0] != a0  # phase A keys the dead parser only
    live = _manifest()
    live["parsers"] = {**live["parsers"], "phase_b": "http://127.0.0.1:9802"}
    assert _keys(live) == (a0, _keys(live)[1]) and _keys(live)[1] != b0  # phase B keys the live parser
    monkeypatch.setattr(cc, "RESET_IDS_FORM", "(progn 0)")
    a1, b1 = _keys(_manifest())
    assert a1 != a0 and b1 != b0


def test_the_key_changes_with_the_job_content():
    pins = cc.pins_of(_manifest())
    base = cc.key_a(pins, job_sha256="a" * 64, request_sha256="b" * 64, deadline_s=6.0)
    assert cc.key_a(pins, job_sha256="d" * 64, request_sha256="b" * 64, deadline_s=6.0) != base
    assert cc.key_a(pins, job_sha256="a" * 64, request_sha256="d" * 64, deadline_s=6.0) != base
    assert cc.key_a(pins, job_sha256="a" * 64, request_sha256="b" * 64, deadline_s=11.0) != base
    b = cc.key_b(pins, job_sha256="a" * 64, phase_a_sha256="c" * 64)
    assert cc.key_b(pins, job_sha256="a" * 64, phase_a_sha256="d" * 64) != b  # another phase-A answer
    assert cc.key_b(pins, job_sha256="d" * 64, phase_a_sha256="c" * 64) != b


def test_the_key_ignores_what_seals_the_run_to_its_own_sources():
    base = _keys(_manifest())
    for field, value in {
        "sources_sha256": "e" * 64,
        "source_count": 4,
        "syntax_from": "elsewhere",
        "syntax_manifest_sha256": "e" * 64,
        "content_sha256": "e" * 64,
    }.items():
        assert _keys({**_manifest(), field: value}) == base, field
    moved = _manifest()
    moved["snapshot_pins"] = {**moved["snapshot_pins"], "sha256": "e" * 64}  # the pins file's basis names the head
    assert _keys(moved) == base


def test_no_snapshot_pins_means_no_cache():
    manifest = _manifest()
    del manifest["snapshot_pins"]
    assert cc.pins_of(manifest) is None


# -- the capture ------------------------------------------------------------------------------------------------------


def test_an_append_capture_runs_only_the_new_message(world):
    cache = world.root / "cache"
    first, result = _capture(world, "run-1", 3, cache=cache)
    assert sorted(_jobs(world.calls))[0].startswith("A-") and len(_jobs(world.calls)) == 4  # 3 A + 1 B (Slowly)
    assert result["capture_cache"]["hits"] == {"A": 0, "B": 0}
    assert result["capture_cache"]["stored"] == {"A": 3, "B": 1}

    second, result = _capture(world, "run-2", 4, cache=cache)
    assert [c[:2] for c in _jobs(world.calls)] == ["A-"]  # exactly one engine job: the new message
    assert result["capture_cache"]["hits"] == {"A": 3, "B": 1}
    assert result["capture_cache"]["stored"] == {"A": 1, "B": 0}
    rows = {p.parent.name: json.loads(p.read_bytes()) for p in sorted((second / "phase-a").glob("*/unit-00.json"))}
    assert [rows[o].get("cached") for o in sorted(rows)] == [True, True, True, None]
    receipts = sorted(p.name for p in (second / cc.RECEIPTS).glob("*.json"))
    assert receipts == ["A-0000-00.json", "A-0001-00.json", "A-0002-00.json", "B-0001-00.json"]
    assert all(json.loads((second / cc.RECEIPTS / r).read_bytes())["cached"] is True for r in receipts)

    cold, _ = _capture(world, "cold", 4, cache=None)
    assert len(_jobs(world.calls)) == 5
    assert _merged(second) == _merged(cold) and len(_merged(cold)) == 8
    assert (
        json.loads((second / "merge.json").read_bytes())["counts"]
        == json.loads((cold / "merge.json").read_bytes())["counts"]
    )


def test_the_cache_works_across_parallel_shards(world, monkeypatch):
    def in_process(out, site, k, log):
        return [pc.run(out, (i, k), site=site) for i in cs.pending_shards(out, k)]

    monkeypatch.setattr(cs, "run_parallel", in_process)
    monkeypatch.setattr(parse, "_managed_parser", lambda site, out: __import__("contextlib").nullcontext())
    cache = world.root / "cache"
    _capture(world, "run-1", 3, cache=cache, shards=2)
    second, result = _capture(world, "run-2", 4, cache=cache, shards=2)
    assert [c[:2] for c in _jobs(world.calls)] == ["A-"]
    assert result["capture_shards"] == 2 and result["capture_cache"]["hits"] == {"A": 3, "B": 1}
    cold, _ = _capture(world, "cold", 4, cache=None)
    assert _merged(second) == _merged(cold)


def test_the_cache_is_unused_when_unset(world):
    first, result = _capture(world, "run-1", 3, cache=None)
    assert "capture_cache" not in result
    _, _ = _capture(world, "run-2", 3, cache=None)
    assert len(_jobs(world.calls)) == 4  # everything again
    assert not (first / cc.RECEIPTS).exists()
    assert all("cached" not in json.loads(p.read_bytes()) for p in (first / "phase-a").glob("*/unit-*.json"))
    assert not list(world.root.rglob("capture-cache*"))


def test_a_pin_change_misses(world):
    cache = world.root / "cache"
    _capture(world, "run-1", 3, cache=cache)
    for entry in cache.rglob("*.json"):  # the same results under other pins: never reused
        doc = json.loads(entry.read_bytes())
        doc["key"] = "0" * 64
        entry.write_text(canonical_json(doc))
    _, result = _capture(world, "run-2", 3, cache=cache)
    assert len(_jobs(world.calls)) == 4 and result["capture_cache"]["hits"] == {"A": 0, "B": 0}


def test_an_unreadable_entry_is_a_miss(world):
    cache = world.root / "cache"
    _capture(world, "run-1", 3, cache=cache)
    for entry in cache.rglob("*.json"):
        entry.write_text("{ not json")
    _, result = _capture(world, "run-2", 3, cache=cache)
    assert len(_jobs(world.calls)) == 4 and result["capture_cache"]["hits"] == {"A": 0, "B": 0}


def test_a_transport_failure_is_never_stored(world, monkeypatch):
    cache = world.root / "cache"
    real = FakeEngine.http

    def flaky(self, request, label, timeout=None, allowed=None):
        if label.startswith("A-0000"):
            self.calls.append(label)
            raise ConnectionError("reset")
        return real(self, request, label, timeout, allowed)

    monkeypatch.setattr(FakeEngine, "http", flaky)
    _, result = _capture(world, "run-1", 3, cache=cache)
    assert result["capture_cache"]["stored"] == {"A": 2, "B": 1}
    monkeypatch.setattr(FakeEngine, "http", real)
    _, result = _capture(world, "run-2", 3, cache=cache)
    assert [c[:6] for c in _jobs(world.calls)] == ["A-0000"]


# -- fix round 1 ------------------------------------------------------------------------------------------------------


def test_a_redo_with_an_errored_segment_is_never_cached(world, monkeypatch):
    cache = world.root / "cache"

    def flaky_redo(engine, job, analysis, pass_id):
        redo = _fake_redo(engine, job, analysis, pass_id)
        redo["status"], redo["segments"][0]["status"], redo["segments"][0]["error"] = "partial", "error", "lost"
        return redo

    monkeypatch.setattr(pc, "redo_pass", flaky_redo)
    _, result = _capture(world, "run-1", 3, cache=cache)
    assert result["capture_cache"]["stored"] == {"A": 3, "B": 0}
    monkeypatch.setattr(pc, "redo_pass", _fake_redo)
    _, result = _capture(world, "run-2", 3, cache=cache)
    assert [c[:2] for c in _jobs(world.calls)] == ["B-"]  # the errored redo is redone; phase A all hit
    assert result["capture_cache"]["hits"] == {"A": 3, "B": 0}
    assert result["capture_cache"]["stored"] == {"A": 0, "B": 1}


def test_a_tampered_entry_is_not_reused_and_is_replaced(world):
    cache = world.root / "cache"
    _capture(world, "run-1", 3, cache=cache)
    entries = sorted((cache / "A").rglob("*.json"))
    tampered = next(p for p in entries if json.loads(p.read_bytes())["result"]["response"]["payload"]["passes"][0]
                    ["meaningKind"] == "nil")  # fmt: skip
    doc = json.loads(tampered.read_bytes())
    doc["result"]["response"]["payload"]["passes"][0]["meaningKind"] = "TIME-OUT"  # result_sha256 left as it was
    tampered.write_text(canonical_json(doc))
    _, result = _capture(world, "run-2", 3, cache=cache)
    assert [c[:2] for c in _jobs(world.calls)] == ["A-"] and result["capture_cache"]["hits"] == {"A": 2, "B": 1}
    assert result["capture_cache"]["stored"] == {"A": 1, "B": 0}
    fresh = json.loads(tampered.read_bytes())
    assert fresh["result"]["response"]["payload"]["passes"][0]["meaningKind"] == "nil"
    assert fresh["result_sha256"] == cc._result_sha(fresh["result"])
    assert len(list((cache / cc.QUARANTINE).glob("A-*.json"))) == 1
    _, result = _capture(world, "run-3", 3, cache=cache)
    assert _jobs(world.calls) == [] and result["capture_cache"]["hits"] == {"A": 3, "B": 1}


def test_a_cache_write_failure_never_aborts_the_capture(world, monkeypatch):
    def full(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cc, "_store", full)
    capture, result = _capture(world, "run-1", 3, cache=world.root / "cache")
    assert (capture / "merge.json").exists() and result["units"] == 3
    assert "No space left on device" in result["capture_cache"]["store_error"]
    assert "No space left on device" in (capture / "capture-cache.log").read_text()


@pytest.mark.parametrize(
    ("recorded", "env", "refused"),
    [
        ({"name": "en_core_web_sm"}, "", True),
        ({}, "", True),
        ({}, "en_core_web_sm", False),
        ({"name": "en_core_web_trf"}, "", False),
    ],
)
def test_a_capture_resume_refuses_another_extraction_model_than_the_recorded_one(monkeypatch, recorded, env, refused):
    monkeypatch.setenv("MK_EXTRACTION_SPACY_MODEL", env)
    manifest = {"extraction_model": recorded} if recorded else {}  # no record = a capture made before 1.4 (sm)
    if refused:
        with pytest.raises(ValueError, match="MK_EXTRACTION_SPACY_MODEL=en_core_web_sm"):
            pc.require_extraction_model(manifest, what="capture")
    else:
        pc.require_extraction_model(manifest, what="capture")
