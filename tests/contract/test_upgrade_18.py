"""Kernel 1.8: the projection carries no codings, topic labels or cards, reads graph_a, and its Saga rows carry
``inventory_sha: None`` (a stored hex value on an old bonfire's continued Saga passes the forward gate). The 14-stage
set and its compatibility layer: an older ``build-config.json`` loads (``RETIRED_KEYS``, archived ``through`` names),
an older bonfire's frozen list is skipped, a full build's sanity is a receipt, the spend constants are not raised.

Fixtures (also used by later 1.8 tests): ``rows`` = ``g1_projection.project`` over the minimal records of
``tests/unit/graph/test_saga_chain.py`` (Sagas and Episodics, no events); ``cfg`` = a ``BuildConfig`` on ``tmp_path``;
``ctx_on_tmp_root`` = a ``Ctx`` over it (the path helpers read no file); ``config_17_doc`` = a 1.7 ``to_json()``
(``tests/fixtures/build-config-1.7.json``); ``tmp_bonfire_17`` = a bonfire whose ``bonfire.json`` lists 1.7's frozen
stages. Offline, invented names only."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from memory_kernel import cli
from memory_kernel.graph.g1_projection import project
from memory_kernel.pipeline import bonfire, config, embed, llm, runner
from memory_kernel.pipeline.config import BuildConfig
from memory_kernel.pipeline.runner import Ctx
from memory_kernel.pipeline.stages import events, index, receipt, write
from memory_kernel.pipeline.stages.write import Refused
from unit.graph.test_saga_chain import EPISODES, GROUP, records

CODING_NODES = {"Question", "Plan", "Purpose"}
CODING_EDGES = {"ANSWERS", "ANSWERS_VIA", "DERIVED_FROM"}


@pytest.fixture
def rows() -> dict:
    return project(
        records(),
        eventuality={},
        episodes=EPISODES,
        group_id=GROUP,
        created_at="2026-10-14T00:00:00Z",
        ingest_pass="mk-conv-test-001",
    )


@pytest.fixture
def cfg(tmp_path) -> BuildConfig:
    return BuildConfig(
        sample_id="conv-1", conversation=tmp_path / "c.json", group="g", purpose="p", workdir=tmp_path, root=tmp_path
    )


@pytest.fixture
def ctx_on_tmp_root(cfg) -> Ctx:
    return Ctx(cfg=cfg, concurrency=1)


def test_projection_emits_no_codings(rows):
    assert not {l for n in rows["nodes"] for l in n["labels_neo4j"]} & CODING_NODES  # noqa: E741
    assert not {e["type"] for e in rows["edges"]} & CODING_EDGES


def test_episodic_rows_carry_no_topic_keys(rows):
    ep = [n["props"] for n in rows["nodes"] if "Episodic" in n["labels_neo4j"]]
    assert ep and all("topic_labels" not in p and "topic_p" not in p for p in ep)


def test_saga_rows_carry_inventory_sha_none_and_no_labels_clause(rows):
    sagas = [n["props"] for n in rows["nodes"] if "Saga" in n["labels_neo4j"]]
    assert sagas and all(p["inventory_sha"] is None and "top labels" not in p["summary"] for p in sagas)


def test_continued_saga_with_stored_inventory_sha_passes_gate(rows):
    saga = next(n for n in rows["nodes"] if "Saga" in n["labels_neo4j"])
    have = {"labels_neo4j": saga["labels_neo4j"], "props": {**saga["props"], "inventory_sha": "ab" * 32}}
    assert "inventory_sha" not in write._frozen_changes(saga, have)


def test_projection_inputs_read_graph_a(ctx_on_tmp_root):
    inputs = write.projection_inputs(ctx_on_tmp_root)
    assert set(inputs) == {"graph", "eventuality", "episode_map"}
    assert inputs["graph"].endswith("06-graph-001/graph.json")


def test_events_and_ctx_read_graph_a(ctx_on_tmp_root):
    for inputs in (events.events_inputs(ctx_on_tmp_root), events.ctx_inputs(ctx_on_tmp_root)):
        assert inputs["graph"].endswith("06-graph-001/graph.json")
        assert inputs["inventory"].endswith("06-graph-001/empty-inventory.json")


# -- the 1.8 stage set and its compatibility layer (Task 4) ---------------------------------------------------------

ORDER_18 = ("select", "boundary", "capture", "graph_picks", "graph_a", "events", "ctx", "referents",
            "build_dry", "write", "rewrite", "index", "sanity", "seal")  # fmt: skip


@pytest.fixture
def config_17_doc() -> dict:
    """A 1.7 ``BuildConfig.to_json()`` (generated under the 1.7 tree, synthetic paths): ``taxonomy_gate``,
    ``lost_wires`` and ``remine_unit_price_budget`` set, ``through: "jev2"``."""
    return json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "build-config-1.7.json").read_text())


def test_order_is_the_18_stage_set():
    assert runner.ORDER == ORDER_18 and not set(config.ARCHIVED_STAGES) & set(runner.STAGE_DIRS)


def test_sealed_stages_and_caps():
    assert set(index.SEALED_STAGES) <= set(ORDER_18) | {"shard", "syntax", "runs"}
    assert set(llm.STAGE_CAPS) == {"events", "ctx"} and "sanity" not in runner.EMBED_STAGES


def test_spend_constants_not_raised():
    assert runner.INDEX_USD_PER_MESSAGE <= 0.0025 and runner.INDEX_USD["sanity"] == 0.0
    assert Decimal("0.10") >= bonfire.APPEND_FLOOR_USD


def test_pilot_stage_is_events(cfg):
    assert runner.stage_concurrency(cfg, "events", False) == min(cfg.jev_concurrency, 4)


def test_17_build_config_loads(config_17_doc):
    cfg = BuildConfig.from_json(config_17_doc, secrets={})
    assert not hasattr(cfg, "taxonomy_gate")


@pytest.mark.parametrize("old,new", sorted(config.ARCHIVED_STAGES.items()))
def test_stored_through_maps_to_kept_stage(config_17_doc, old, new):
    assert BuildConfig.from_json({**config_17_doc, "through": old}, secrets={}).through == new


def test_cli_refuses_archived_through(capsys):
    with pytest.raises(SystemExit):
        cli.main(["build", "x.json", "--workdir", "/tmp/w", "--group", "g", "--through", "jev2"])  # argparse choices


def test_purpose_is_optional(config_17_doc):
    cfg = BuildConfig.from_json({**config_17_doc, "purpose": None}, secrets={})
    assert cfg.purpose is None and cfg.to_json()["purpose"] is None


def test_full_build_sanity_is_a_receipt():
    body = receipt.build_full_receipt(index_result={"stale": 0, "missing": 0, "tag_equals_freeze": True,
                                                    "n_messages": 3, "recipe_tag": "t", "keys_stamped_with_tag": 3})  # fmt: skip
    assert body["schema"] == "mk-build-receipt/v1" and body["mode"] == "full"
    with pytest.raises(Refused):
        receipt.build_full_receipt(index_result={**body, "stale": 1, "tag_equals_freeze": True})


def test_cached_embedder_round_trip(tmp_path):
    from memory_kernel.graph.kg_writer import KEY_EMBED_DIM, KEY_EMBED_MODEL

    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"schema_version": embed.CACHE_SCHEMA, "model": KEY_EMBED_MODEL, "dim": KEY_EMBED_DIM}))
    calls: list[list[str]] = []

    def inner(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    emb = embed.CachedEmbedder(cache, "document", fallback=inner)
    first = emb(["a", "bb"])
    assert calls == [["a", "bb"]] and emb.misses == 2
    second = emb(["a", "bb"])
    assert calls == [["a", "bb"]] and emb.hits == 2 and second == first
