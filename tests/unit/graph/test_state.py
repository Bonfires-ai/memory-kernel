"""Kernel 2.0 state rows (``graph.state``): the properties a run's rows carry into Neo4j, the kind-count round trip and
the Roleset / Route / BonfireMeta rows. Pure: the vectors are a stand-in cache, nothing is embedded."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from memory_kernel.graph.g1_kinds import KindEvidence
from memory_kernel.graph.kg_contract import validate_rows
from memory_kernel.graph.state import kind_props, kind_row, meta_props, meta_row, roleset_rows, route_rows, stamp_state
from memory_kernel.pipeline.embed import message_texts
from memory_kernel.pipeline.stages import write
from unit.graph._rows import CHATS, GROUP, ROLESETS, projection, sources


def records() -> list[dict]:
    """The select rows, each of a session named ``<sample>-session_N`` (N = 1 + its order)."""
    return [{**s, "session_id": f"conv-session_{1 + s['order']}"} for s in sources()]


def vectors_for(rows: list[dict]) -> dict[str, list[float]]:
    return {t: [float(i), 1.0] for i, t in enumerate(message_texts(rows).values())}


def episodic(rows: dict, mid: str) -> dict:
    return next(
        n["props"] for n in rows["nodes"] if "Episodic" in n["labels_neo4j"] and n["props"]["message_id"] == mid
    )


def stamped(selected: list[dict], run: int = 1, vectors: dict | None = None) -> tuple[dict, dict, dict]:
    rows = projection()
    vectors = vectors if vectors is not None else vectors_for(records())
    counts = stamp_state(rows, run=run, records=selected, chat_of=CHATS, vectors=vectors)
    return rows, vectors, counts


def test_new_episodic_rows_carry_the_state_props():
    recs = records()
    rows, vectors, counts = stamped(recs)
    record = recs[1]  # D2:1, chat k2, session 2
    ep = episodic(rows, record["message_id"])
    assert json.loads(ep["source_row_json"]) == record
    assert ep["message_embedding"] == vectors[message_texts([record])[record["message_id"]]]
    assert (ep["order"], ep["chat"], ep["session"], ep["run"]) == (record["order"], "k2", 2, 1)
    assert ep["addressees"] == ["Ana"] and ep["text"] == record["text"]
    assert ep["message_embedding_model"] == "voyage-3/document" and len(ep["message_text_sha256"]) == 64
    assert counts["episodic"] == 4 and counts["edges"] == len(rows["edges"])


def test_message_embedding_is_of_the_boundary_text_not_content():
    recs = records()
    recs[0] = {**recs[0], "attachments": [{"caption": " with a photo"}]}
    vectors = {t: [float(i)] for i, t in enumerate(message_texts(recs).values())}
    rows, _, _ = stamped(recs, vectors=vectors)
    ep = episodic(rows, "D1:1")
    assert message_texts(recs)["D1:1"] == "Bo: Bo likes pears with a photo"
    assert ep["message_embedding"] == vectors["Bo: Bo likes pears with a photo"] and ep["text"] == "Bo likes pears"


def test_resent_stored_episodic_is_not_stamped():
    recs = records()
    rows = projection()
    resent = copy.deepcopy(next(n for n in rows["nodes"] if n["props"].get("message_id") == "D3:1"))
    stored = copy.deepcopy(resent)
    stamp_state(rows, run=1, records=recs[:3], chat_of=CHATS, vectors=vectors_for(recs))
    now = next(n for n in rows["nodes"] if n["props"].get("message_id") == "D3:1")
    assert now == resent and "run" not in now["props"] and write._frozen_changes(now, stored) == []


def test_entities_get_minted_order_mint_kind_and_kind_counts():
    rows = projection()
    minted = rows["forward"]["minted"]
    ev = KindEvidence(person=True, loc_roles=2)
    ev.head["Place"] += 3
    rows["forward"]["kinds"] = {minted[0].uuid: ev.to_row()}
    stamp_state(rows, run=0, records=records(), chat_of=CHATS, vectors=vectors_for(records()))
    by_uuid = {n["props"]["uuid"]: n["props"] for n in rows["nodes"]}
    for m in minted:
        p = by_uuid[m.uuid]
        assert (p["minted_order"], p["mint_kind"], p["ground_legacy"], p["run"]) == ([0, m.order], m.kind, True, 0)
    assert kind_row(by_uuid[minted[0].uuid]) == ev.to_row()
    assert not any(k.startswith("kind_") for k in by_uuid[minted[1].uuid])

    later = projection()
    later["forward"]["minted"] = [replace(m, run=3) for m in later["forward"]["minted"]]
    stamp_state(later, run=3, records=records(), chat_of=CHATS, vectors=vectors_for(records()))
    got = {n["props"]["uuid"]: n["props"] for n in later["nodes"]}[minted[0].uuid]
    assert (got["ground_legacy"], got["run"], got["minted_order"][0]) == (False, 3, 3)


def test_a_saga_of_a_stamped_first_episode_gets_chat_and_run():
    rows, _, counts = stamped(records())
    sagas = [n["props"] for n in rows["nodes"] if "Saga" in n["labels_neo4j"]]
    assert (
        counts["saga"] == len(sagas) and {s["chat"] for s in sagas} == {"k1", "k2"} and {s["run"] for s in sagas} == {1}
    )


def test_kind_props_round_trip():
    ev = KindEvidence(person=False, loc_roles=1, located=True)
    ev.supplied["Organisation"] += 2
    ev.head["Place"] += 4
    row = ev.to_row()
    assert kind_row(kind_props(row)) == row
    assert kind_row(kind_props(KindEvidence().to_row())) == KindEvidence().to_row()
    assert kind_props(row)["kind_head_Place"] == 4 and kind_props(row)["kind_supplied_Organisation"] == 2


def test_roleset_route_and_meta_rows_validate():
    rows = projection()
    entity = next(n for n in rows["nodes"] if "Entity" in n["labels_neo4j"] and "Event" not in n["labels_neo4j"])
    full = {**ROLESETS["like.01"], "gloss_source": "chain", "p_yes": {"state": 0.9}, "compiled_sha256": "ab" * 32}
    rolesets = roleset_rows(GROUP, {"like.01": full, "meet.01": ROLESETS["meet.01"]}, run=2)
    routes, edges = route_rows(GROUP, {"the pears": entity["props"]["uuid"]}, run=2)
    meta = meta_row(
        GROUP,
        meta_props(
            bonfire_id="conv",
            build_config={"ingest_pass": "pass-1"},
            recipe_tag="r",
            extraction_model={"name": "m", "version": "1"},
            agent_handles=["@bot"],
            agent_handle_runs=[0],
            chat_tagged=False,
            explicit_since=None,
            sessions=3,
            committed_run=2,
            generation="gen-1",
        ),
    )
    assert validate_rows([entity, *rolesets, *routes, meta], edges) == []
    assert rolesets[0]["props"]["first_run"] == 2 and edges[0]["props"]["run"] == 2


def test_roleset_rows_accept_settled_and_unsettled_eventuality_rows():
    full = {
        "gloss": "g",
        "gloss_source": "chain",
        "examples": ["e"],
        "label": "state",
        "p_yes": {"state": 0.9, "occurrence": 0.1},
        "compiled_sha256": "ab" * 32,
    }
    bare = {"gloss": "g", "examples": ["e"]}
    got = {n["props"]["roleset"]: n["props"] for n in roleset_rows(GROUP, {"a.01": full, "b.01": bare}, run=0)}
    assert json.loads(got["a.01"]["p_yes_json"]) == full["p_yes"] and got["a.01"]["label"] == "state"
    assert got["a.01"]["compiled_sha256"] == full["compiled_sha256"] and got["a.01"]["gloss_source"] == "chain"
    assert got["b.01"]["label"] is None and got["b.01"]["p_yes_json"] is None
    assert got["b.01"]["compiled_sha256"] is None and got["b.01"]["examples"] == ["e"]


def test_meta_query_cache_key_is_the_config_sha():
    import hashlib

    config = {"b": 1, "a": {"c": 2}}
    props = meta_props(
        bonfire_id="x", build_config=config, recipe_tag=None, extraction_model={}, agent_handles=[],
        agent_handle_runs=[], chat_tagged=True, explicit_since="t", sessions=0, committed_run=0, generation="g",
    )  # fmt: skip
    assert props["query_cache_key"] == hashlib.sha256(props["build_config_json"].encode()).hexdigest()[:12]
    assert json.loads(props["build_config_json"]) == config


@pytest.mark.parametrize("config", [{"voyage_api_key": "x"}, {"a": {"neo4j_password": "x"}}])
def test_meta_refuses_a_secret_field(config):
    with pytest.raises(ValueError, match="secret"):
        meta_props(
            bonfire_id="x", build_config=config, recipe_tag=None, extraction_model={}, agent_handles=[],
            agent_handle_runs=[], chat_tagged=False, explicit_since=None, sessions=0, committed_run=0, generation="g",
        )  # fmt: skip


def test_missing_vector_refuses():
    recs = records()
    vectors = vectors_for(recs)
    del vectors[message_texts(recs)["D3:1"]]
    with pytest.raises(ValueError, match="no message embedding for D3:1"):
        stamped(recs, vectors=vectors)


def test_meta_keeps_each_handles_join_run_parallel_to_the_handles():
    kw = dict(bonfire_id="x", build_config={}, recipe_tag=None, extraction_model={}, chat_tagged=False,
              explicit_since=None, sessions=4, committed_run=3, generation="g-1")  # fmt: skip
    props = meta_props(agent_handles=["@helper", "@scribe"], agent_handle_runs=[0, 3], **kw)
    assert props["agent_handles"] == ["@helper", "@scribe"] and props["agent_handle_runs"] == [0, 3]
    assert props["generation"] == "g-1"
    with pytest.raises(ValueError, match="agent_handle_runs has 1 runs for 2 agent handles"):
        meta_props(agent_handles=["@helper", "@scribe"], agent_handle_runs=[0], **kw)
