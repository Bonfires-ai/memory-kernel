"""The values ``mk build`` falls back to when a flag is absent: the Purpose example, the recipe computed from the
code constants, the image pin, and the invariants without a reference; and the neutral ids: bonfire id, boundary
window prefix, no replay gate, any group name.

All in-process and offline; nothing here starts a stage child, reaches Neo4j or spends."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
#: sha256 of the Purpose sentence ``examples/purpose.txt`` ships (read raw: a trailing newline would change it).
PURPOSE_SHA256 = "6f732c6f2cdb0a8455505b2771b88e2ff6c7400009fd229f87d160c88474fe48"
SEVEN_CHECKS = {
    "derivation_acyclic",
    "observations_card_free",
    "plans_resolve",
    "codings_traceable",
    "codings_anchored",
    "observations_traceable",
    "folds_current",
}


def test_the_example_purpose_file_reads_back_to_the_pinned_sentence():
    from memory_kernel.cli import read_purpose

    purpose = read_purpose(REPO_ROOT / "examples/purpose.txt")
    assert hashlib.sha256(purpose.encode()).hexdigest() == PURPOSE_SHA256
    assert not purpose.endswith("\n")


def test_default_recipe_is_the_sealed_recipe_the_answerer_runs():
    from memory_kernel.pipeline.config import default_recipe
    from memory_kernel.search.kg_context import ContextConfig
    from memory_kernel.search.kg_recipe import SEALED_RECIPE
    from memory_kernel.serve.answer import RECIPE_TAG

    recipe = default_recipe()
    assert set(recipe) == {"kg_recipe_config", "context_config", "key_recipe_tag"}
    assert recipe["kg_recipe_config"] == asdict(SEALED_RECIPE)
    assert recipe["key_recipe_tag"] == RECIPE_TAG
    context = recipe["context_config"]
    assert context["timeline_shapes"] == ["how_long", "when"]  # a sorted list, JSON-able
    assert context["evidence_k"] == 40
    assert context == {**asdict(ContextConfig()), "timeline_shapes": sorted(ContextConfig().timeline_shapes)}


def test_the_budget_invariant_is_skipped_without_a_reference_and_the_seven_checks_still_run():
    from memory_kernel.pipeline.stages.write import invariants

    report = invariants({"nodes": [], "edges": []}, None)
    assert report["budget"] == {"skipped": "no invariants reference"}
    assert set(report["violations"]) == SEVEN_CHECKS
    assert report["gated_violations"] == {}


def _config(tmp_path: Path, **kw):
    from memory_kernel.pipeline.config import BuildConfig

    return BuildConfig(
        **{
            "sample_id": "conv-syn",
            "conversation": tmp_path / "conversation.json",
            "group": "g1-test-c2-syn",
            "purpose": "remember the conversation",
            "workdir": tmp_path / "workdir",
            "root": tmp_path,
            "vault": tmp_path,
            **kw,
        }
    )


def test_the_grammar_image_pin_is_the_flag_then_the_manifest_then_the_image_file(tmp_path):
    from memory_kernel.pipeline.config import grammar_image_sha

    image = tmp_path / "engine.fci"
    image.write_bytes(b"an engine image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"grammar_image_sha256": "from-the-manifest"}')
    from_file = hashlib.sha256(b"an engine image").hexdigest()

    assert grammar_image_sha(_config(tmp_path, engine_image=image)) == from_file
    assert grammar_image_sha(_config(tmp_path, engine_image=image, recipe_manifest=manifest)) == "from-the-manifest"
    flagged = _config(tmp_path, engine_image=image, recipe_manifest=manifest, grammar_image_sha256="from-the-flag")
    assert grammar_image_sha(flagged) == "from-the-flag"

    manifest.write_text("{}")  # a manifest that is given must carry the pin: no silent fall-through to the file
    with pytest.raises(ValueError, match="has no grammar_image_sha256"):
        grammar_image_sha(_config(tmp_path, engine_image=image, recipe_manifest=manifest))
    with pytest.raises(ValueError, match="cannot be read"):
        grammar_image_sha(_config(tmp_path, engine_image=tmp_path / "missing.fci"))


def test_a_build_config_without_the_id_flags_carries_the_neutral_ids(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.bonfire_id == "mk-conv-syn"
    assert cfg.boundary_window_prefix == "full-active"
    assert cfg.boundary_replay_archive is None  # the replay gate is opt-in
    assert cfg.boundary_replay_scope is None
    assert _config(tmp_path, bonfire_id="my-bonfire").bonfire_id == "my-bonfire"


def test_any_group_name_is_accepted_and_a_blank_one_is_not(tmp_path):
    assert _config(tmp_path, group="my-group").group == "my-group"
    with pytest.raises(ValueError, match="group is empty"):
        _config(tmp_path, group="  ")
    with pytest.raises(ValueError, match="boundary_window_prefix is empty"):
        _config(tmp_path, boundary_window_prefix="")


def test_a_build_config_that_still_carries_the_retired_gate_switch_loads(tmp_path):
    from memory_kernel.pipeline.config import BuildConfig

    doc = _config(tmp_path).to_json()
    doc["boundary_replay_gate"] = True  # written by an older mk
    doc["boundary_replay_archive"] = [str(tmp_path / "a.json"), str(tmp_path / "s.json")]  # the old derived pair
    assert BuildConfig.from_json(doc, secrets={}).boundary_replay_scope is None


def test_the_concurrent_group_prefix_defaults_to_none_and_rejects_a_blank(tmp_path):
    assert (
        _config(tmp_path).concurrent_group_prefix is None
    )  # the default: no prefix (other groups count only under strict_db_snapshot, where none is tolerated)
    assert _config(tmp_path, concurrent_group_prefix="g1-test-c2-").concurrent_group_prefix == "g1-test-c2-"
    for blank in (
        "",
        "  ",
    ):  # a blank prefix would tolerate every group: the strict default must be spelled by omission
        with pytest.raises(ValueError, match="concurrent_group_prefix is empty"):
            _config(tmp_path, concurrent_group_prefix=blank)


def test_the_prefix_survives_the_stage_childs_config_round_trip_and_an_older_config_loads_strict(tmp_path):
    import json

    from memory_kernel.pipeline.config import BuildConfig

    doc = json.loads(json.dumps(_config(tmp_path, concurrent_group_prefix="g1-test-c2-").to_json()))
    assert (
        BuildConfig.from_json(doc, secrets={}).concurrent_group_prefix == "g1-test-c2-"
    )  # the child rebuilds from JSON
    older = json.loads(json.dumps(_config(tmp_path).to_json()))
    del older["concurrent_group_prefix"]  # a build-config.json written before the key existed
    assert "concurrent_group_prefix" not in older
    assert BuildConfig.from_json(older, secrets={}).concurrent_group_prefix is None


def test_referent_policy_round_trip(tmp_path):
    import json

    from memory_kernel import cli
    from memory_kernel.pipeline.config import BuildConfig

    phrase = _config(tmp_path)
    assert phrase.referent_policy == "phrase"
    doc = json.loads(json.dumps(phrase.to_json()))
    assert "referent_policy" not in doc
    assert BuildConfig.from_json(doc, secrets={}).referent_policy == "phrase"  # an absent key reads as phrase
    struct = json.loads(json.dumps(_config(tmp_path, referent_policy="structure").to_json()))
    assert struct["referent_policy"] == "structure"
    assert BuildConfig.from_json(struct, secrets={}).referent_policy == "structure"
    with pytest.raises(ValueError, match="referent_policy"):
        _config(tmp_path, referent_policy="noun")
    args = cli.parser().parse_args(["build", "c.json", "--workdir", "w", "--group", "g", "--purpose", "p"])
    assert args.referents == "structure"


def test_the_replay_scope_file_is_checked(tmp_path):
    import json

    from memory_kernel.pipeline.config import load_replay_scope

    good = {
        "scope": {"bonfire_id": "b", "profile": "p", "conversation_id": "c"},
        "episode_prefix": "e-",
        "window_prefix": "w",
    }
    file = tmp_path / "scope.json"
    file.write_text(json.dumps(good))
    assert load_replay_scope(file) == good
    file.write_text(json.dumps({**good, "step_schema": "old-step/v1"}))
    assert load_replay_scope(file) == {**good, "step_schema": "old-step/v1"}
    for bad in (
        {**good, "extra": 1},
        {**good, "step_schema": ""},
        {**good, "step_schema": 1},
        {**good, "window_prefix": ""},
        {**good, "scope": {"bonfire_id": "b"}},
        {**good, "scope": {**good["scope"], "profile": 3}},
        [],
    ):
        file.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="boundary replay scope"):
            load_replay_scope(file)
    with pytest.raises(ValueError, match="No such file"):
        load_replay_scope(tmp_path / "missing.json")


def test_the_window_prefix_reaches_the_boundary_wire():
    """Two arrivals of one day: the second is a same-day arrival, so a wire is compiled and its window id is the prefix."""
    from memory_kernel.pipeline.stages import _boundary_walk as ub
    from memory_kernel.pipeline.stages.ingest import build

    item = {
        "sample_id": "conv-syn",
        "conversation": {
            "speaker_a": "Ann",
            "speaker_b": "Bob",
            "session_1_date_time": "2:41 pm on 9 March, 2024",
            "session_1": [
                {"speaker": "Ann", "dia_id": "D1:1", "text": "Hi Bob, I started a bakery."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Congrats Ann!"},
            ],
        },
    }
    sources = build(item, {"bonfire_id": "b", "conversation_id": "conv-syn"})[3]
    wires = {
        prefix: ub.boundary_wire(
            sources,
            ub.timestamp_rows(sources),
            index=1,
            history=[],
            previous_cursor=0,
            scope=ub.window_scope(sources),
            window_prefix=prefix,
        )
        for prefix in ("alpha", "beta")
    }
    assert "alpha-0-cursor-0" in str(wires["alpha"]["wire"])
    assert "alpha" not in str(wires["beta"]["wire"]) and "beta-0-cursor-0" in str(wires["beta"]["wire"])
    assert wires["alpha"]["request_sha256"] != wires["beta"]["request_sha256"]


def test_the_step_schema_is_sealed_in_the_step_body_and_never_reaches_the_wire():
    """A replay of an archived run passes that run's step id: the step digest changes, the wire and its sha do not."""
    from memory_kernel.pipeline.stages import _boundary_walk as ub
    from memory_kernel.pipeline.stages.ingest import build

    item = {
        "sample_id": "conv-syn",
        "conversation": {
            "speaker_a": "Ann",
            "speaker_b": "Bob",
            "session_1_date_time": "2:41 pm on 9 March, 2024",
            "session_1": [
                {"speaker": "Ann", "dia_id": "D1:1", "text": "Hi Bob, I started a bakery."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Congrats Ann!"},
            ],
        },
    }
    sources = build(item, {"bonfire_id": "b", "conversation_id": "conv-syn"})[3]
    common = dict(index=1, history=[], previous_cursor=0, scope=ub.window_scope(sources), window_prefix="w")
    default = ub.boundary_wire(sources, ub.timestamp_rows(sources), **common)
    legacy = ub.boundary_wire(sources, ub.timestamp_rows(sources), step_schema="old-step/v1", **common)
    assert default["schema_version"] == "mk-full-timed-boundary-step/v1"
    assert legacy["schema_version"] == "old-step/v1"
    assert legacy["wire"] == default["wire"] and legacy["request_sha256"] == default["request_sha256"]
    assert legacy["content_sha256"] != default["content_sha256"]


def test_preflight_replays_the_boundary_pair_and_refuses_a_pair_that_does_not_reproduce(tmp_path, monkeypatch):
    from memory_kernel.pipeline import runner
    from memory_kernel.pipeline.stages import _boundary_walk as ub

    archive, sources = tmp_path / "archive.json", tmp_path / "sources.json"
    sources.write_text("[]")
    seen = []

    def fake(results):
        def replay(archived, srcs, **scope):
            seen.append((archived, srcs, scope))
            if isinstance(results, Exception):
                raise results
            return results

        return replay

    scope = {"scope": {}, "episode_prefix": "e-", "window_prefix": "w", "step_schema": "old-step/v1"}
    monkeypatch.setattr(ub, "replay_equivalence", fake({"match": 3, "total": 3, "episodes_match": True}))
    assert runner.boundary_replay_problems((archive, sources), scope) == []
    assert seen[-1] == (archive, [], scope)
    monkeypatch.setattr(ub, "replay_equivalence", fake({"match": 2, "total": 3, "episodes_match": True}))
    (problem,) = runner.boundary_replay_problems((archive, sources), scope)
    assert "2/3" in problem and "step_schema" in problem
    monkeypatch.setattr(ub, "replay_equivalence", fake({"match": 0, "total": 0, "episodes_match": True}))
    assert len(runner.boundary_replay_problems((archive, sources), scope)) == 1
    monkeypatch.setattr(ub, "replay_equivalence", fake({"match": 3, "total": 3, "episodes_match": False}))
    assert len(runner.boundary_replay_problems((archive, sources), scope)) == 1
    monkeypatch.setattr(ub, "replay_equivalence", fake(KeyError("rows")))
    (problem,) = runner.boundary_replay_problems((archive, sources), scope)
    assert "does not replay" in problem


def test_default_config_has_only_1_5_keys(tmp_path):
    new_in_1_6 = {"strict_db_snapshot", "append_base_usd", "spend_ceiling_usd"}  # fields 1.5.2's BuildConfig lacked
    new_in_1_7 = {"genesis"}
    doc = _config(tmp_path, append_messages=3).to_json()  # append_messages is a 1.5.2 field
    assert not (new_in_1_6 | new_in_1_7) & set(doc)
    doc = _config(tmp_path, strict_db_snapshot=True, append_base_usd="1", spend_ceiling_usd="2").to_json()
    assert new_in_1_6 <= set(doc)  # a set value is still written
    assert new_in_1_7 <= set(_config(tmp_path, genesis=True).to_json())
