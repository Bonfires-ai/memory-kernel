import os

import pytest

from memory_kernel.graph.kg_contract import (
    BASE_ENTITY,
    BASE_EPISODIC,
    BASE_HAS_EPISODE,
    BASE_MENTIONS,
    BASE_NEXT_EPISODE,
    BASE_RELATES_TO,
    BASE_SAGA,
    EVOLVING,
    NODE_BASE,
    validate_edge,
    validate_node,
    validate_rows,
)
from memory_kernel.graph.kg_writer import DECISION_FIELDS, _split

T = "2026-01-01T00:00:00Z"


def entity(uuid="e1", **over):
    props = {
        "uuid": uuid,
        "name": "Ruth",
        "group_id": "g",
        "created_at": T,
        "name_embedding": [0.1],
        "summary": "",
        "roleset": "go.01",
    }
    props.update(over)
    return {"labels_neo4j": ["Entity", "Person"], "props": props}


def episodic(uuid="ep1", **over):
    props = {
        "uuid": uuid,
        "name": "ep",
        "group_id": "g",
        "created_at": T,
        "source": "message",
        "source_description": "d",
        "content": "hi",
        "valid_at": T,
        "entity_edges": [],
    }
    props.update(over)
    return {"labels_neo4j": ["Episodic"], "props": props}


def relates(**over):
    props = {
        "uuid": "r1",
        "group_id": "g",
        "created_at": T,
        "name": "WENT_TO",
        "fact": "Ruth went to a group",
        "fact_embedding": [0.1],
        "episodes": ["ep1"],
        "valid_at": None,
        "invalid_at": None,
        "expired_at": None,
        "reference_time": T,
        "source_uuid": "e1",
        "target_uuid": "e2",
        "roleset": "go.01",
    }
    props.update(over)
    return {"type": "RELATES_TO", "source": "e1", "target": "e2", "props": props}


def mentions(**over):
    props = {"uuid": "m1", "group_id": "g", "created_at": T}
    props.update(over)
    return {"type": "MENTIONS", "source": "ep1", "target": "e1", "props": props}


def test_base_tables_match_the_graphiti_fork_models(monkeypatch):
    # importing graphiti_core loads a dotenv into os.environ (VOYAGE_API_KEY and others); undo it when the test ends
    monkeypatch.setattr(os, "environ", os.environ.copy())
    nodes = pytest.importorskip("graphiti_core.nodes")
    edges = pytest.importorskip("graphiti_core.edges")
    drop = {"labels", "attributes"}
    assert set(nodes.EntityNode.model_fields) - drop == BASE_ENTITY
    assert set(nodes.EpisodicNode.model_fields) - drop == BASE_EPISODIC
    rel = set(edges.EntityEdge.model_fields) - {"source_node_uuid", "target_node_uuid", "attributes"}
    assert rel | {"source_uuid", "target_uuid"} == BASE_RELATES_TO
    assert set(edges.EpisodicEdge.model_fields) - {"source_node_uuid", "target_node_uuid"} == BASE_MENTIONS


def test_extended_entity_passes():
    assert validate_node(entity(extra_kernel_field=1)) == []
    assert validate_node(episodic()) == []


def test_missing_base_field_fails():
    row = entity()
    del row["props"]["group_id"]
    assert validate_node(row) == ["e1: missing base field group_id"]


def test_node_type_checks():
    assert validate_node(entity(name=3)) == ["e1: name must be str"]
    assert validate_node(entity(name_embedding="x")) == ["e1: name_embedding must be a list of numbers or None"]
    assert validate_node(episodic(entity_edges="x")) == ["ep1: entity_edges must be a list"]


def test_node_with_both_kind_labels_fails():
    row = entity()
    row["labels_neo4j"].append("Episodic")
    assert validate_node(row) != []
    assert validate_node({"labels_neo4j": ["Person"], "props": entity()["props"]}) != []


def test_new_relationship_type_must_carry_edge_base():
    edge = relates()
    edge["type"] = "ROLE"
    assert validate_edge(edge) == []
    del edge["props"]["source_uuid"]
    assert "missing base field source_uuid" in validate_edge(edge)[0]


def test_edge_value_checks():
    assert validate_edge(relates(fact="")) == ["r1: fact must be a non-empty str"]
    assert validate_edge(relates(episodes="x")) == ["r1: episodes must be a list"]
    assert validate_edge(relates(source_uuid="zz")) == ["r1: source_uuid does not match edge source"]
    assert validate_edge(relates(target_uuid="zz")) == ["r1: target_uuid does not match edge target"]


def test_mentions_validated():
    assert validate_edge(mentions()) == []
    row = mentions()
    del row["props"]["group_id"]
    assert validate_edge(row) == ["m1: missing base field group_id"]
    nodes = [episodic(), entity()]
    assert validate_rows(nodes, [mentions()]) == []
    bad = mentions()
    bad["source"], bad["target"] = "e1", "ep1"
    assert len(validate_rows(nodes, [bad])) == 2


def test_edge_endpoint_must_exist_and_be_entity():
    nodes = [entity("e1"), entity("e2")]
    assert validate_rows(nodes, [relates()]) == []
    assert validate_rows([entity("e1")], [relates()]) != []
    assert validate_rows([entity("e1")], [relates()], previous={"e2": entity("e2")}) == []
    assert validate_rows([entity("e1"), episodic("e2")], [relates()]) != []


def test_graphiti_summary_update_is_a_legal_continuation():
    prior = entity(summary="old")
    new = entity(summary="new", name_embedding=[0.9], name="Caro", roleset="other")
    assert validate_rows([new], [], previous={"e1": prior}) == []


def test_immutable_node_field_rewrite_fails():
    errors = validate_rows([entity(group_id="h")], [], previous={"e1": entity()})
    assert errors == ["e1: base field group_id rewritten on append"]


def test_edge_fact_rewrite_on_append_fails():
    nodes = [entity("e1"), entity("e2")]
    errors = validate_rows(nodes, [relates(fact="other")], previous={"r1": relates()})
    assert errors == ["r1: base field fact rewritten on append"]
    assert validate_rows(nodes, [relates(invalid_at=T, fact_embedding=[0.5])], previous={"r1": relates()}) == []


def test_episodes_may_only_grow():
    nodes = [entity("e1"), entity("e2")]
    assert validate_rows(nodes, [relates(episodes=["ep1", "ep2"])], previous={"r1": relates()}) == []
    errors = validate_rows(nodes, [relates(episodes=["ep2"])], previous={"r1": relates()})
    assert errors == ["r1: base field episodes rewritten on append"]


def test_neo4j_readback_without_null_fields_validates():
    row = entity()
    del row["props"]["name_embedding"]
    assert validate_node(row) == []
    edge = relates()
    for field in ("valid_at", "invalid_at", "expired_at", "reference_time", "fact_embedding"):
        del edge["props"][field]
    assert validate_edge(edge) == []
    nodes = [entity("e1"), entity("e2")]
    assert validate_rows(nodes, [edge], previous={"r1": edge}) == []
    assert validate_rows(nodes, [relates(invalid_at=T)], previous={"r1": edge}) == []


def saga(uuid="s1", **over):
    props = {
        "uuid": uuid,
        "name": "episode-0",
        "group_id": "g",
        "created_at": T,
        "summary": "span; speakers",
        "first_episode_uuid": "ep1",
        "last_episode_uuid": "ep2",
    }
    props.update(over)
    return {"labels_neo4j": ["Saga"], "props": props}


def structural(kind, source, target, uuid="h1"):
    return {"type": kind, "source": source, "target": target, "props": {"uuid": uuid, "group_id": "g", "created_at": T}}


def _save_query_keys(query: str) -> set[str]:
    """Property keys a fork save query stores: the MERGE key plus every SET target."""
    import re

    keys = set(re.findall(r"\{(\w+): \$\w+\}", query))
    body = re.search(r"SET n = \{(.*?)\}", query, re.S)
    if body:
        keys |= set(re.findall(r"(\w+): \$", body.group(1)))
    keys |= set(re.findall(r"\be\.(\w+) = \$", query))
    return keys


def test_saga_and_episode_edges_match_the_fork_models_and_save_queries():
    nodes = pytest.importorskip("graphiti_core.nodes")
    edges = pytest.importorskip("graphiti_core.edges")
    nq = pytest.importorskip("graphiti_core.models.nodes.node_db_queries")
    eq = pytest.importorskip("graphiti_core.models.edges.edge_db_queries")
    from graphiti_core.driver.driver import GraphProvider

    assert set(nodes.SagaNode.model_fields) - {"labels"} == BASE_SAGA
    assert _save_query_keys(nq.get_saga_node_save_query(GraphProvider.NEO4J)) == BASE_SAGA
    ends = {"source_node_uuid", "target_node_uuid"}
    assert set(edges.HasEpisodeEdge.model_fields) - ends == BASE_HAS_EPISODE
    assert set(edges.NextEpisodeEdge.model_fields) - ends == BASE_NEXT_EPISODE
    assert _save_query_keys(eq.HAS_EPISODE_EDGE_SAVE) == BASE_HAS_EPISODE
    assert _save_query_keys(eq.NEXT_EPISODE_EDGE_SAVE) == BASE_NEXT_EPISODE


def test_saga_validates_and_is_a_third_node_kind():
    assert validate_node(saga(extra=1)) == []
    row = saga()
    del row["props"]["summary"]
    assert validate_node(row) == ["s1: missing base field summary"]
    assert validate_node(saga(summary=3)) == ["s1: summary must be str"]
    both = saga()
    both["labels_neo4j"].append("Episodic")
    assert validate_node(both) != []


def test_saga_nullable_fields_may_be_absent_on_readback():
    row = saga()
    for field in ("first_episode_uuid", "last_episode_uuid"):
        del row["props"][field]
    assert validate_node(row) == []


def test_saga_summary_and_last_episode_evolve_first_episode_does_not():
    assert {"summary", "last_episode_uuid", "last_summarized_at"} <= EVOLVING
    assert validate_rows([saga(summary="new", last_episode_uuid="ep3")], [], previous={"s1": saga()}) == []
    errors = validate_rows([saga(first_episode_uuid="ep9")], [], previous={"s1": saga()})
    assert errors == ["s1: base field first_episode_uuid rewritten on append"]


def test_has_episode_and_next_episode_endpoints():
    nodes = [saga(), episodic("ep1"), episodic("ep2"), entity("e1")]
    good = [structural("HAS_EPISODE", "s1", "ep1"), structural("NEXT_EPISODE", "ep1", "ep2", "n1")]
    assert validate_rows(nodes, good) == []
    assert validate_rows(nodes, [structural("HAS_EPISODE", "ep1", "ep2")]) == ["h1: source ep1 must be :Saga"]
    assert validate_rows(nodes, [structural("NEXT_EPISODE", "s1", "e1")]) == [
        "h1: source s1 must be :Episodic",
        "h1: target e1 must be :Episodic",
    ]
    assert validate_rows(nodes, [structural("MENTIONS", "s1", "e1")]) == ["h1: source s1 must be :Episodic"]
    assert validate_rows(nodes, [relates(source_uuid="s1", target_uuid="e1") | {"source": "s1"}]) != []


def test_episode_edges_carry_only_their_base_and_reject_missing_fields():
    edge = structural("NEXT_EPISODE", "ep1", "ep2")
    assert validate_edge(edge) == []
    del edge["props"]["created_at"]
    assert validate_edge(edge) == ["h1: missing base field created_at"]
    assert validate_rows([episodic("ep1"), episodic("ep2")], [structural("HAS_EPISODE", "ep1", "ep2")], previous={})


def test_i5_read_mode_tolerates_graphiti_bulk_edges_without_endpoint_uuids():
    edge = relates()
    for k in ("source_uuid", "target_uuid", "reference_time"):
        del edge["props"][k]
    edge["props"]["attributes"] = "{}"
    assert validate_edge(edge, mode="read") == []
    assert any("source_uuid" in e for e in validate_edge(edge))  # write mode still requires them
    nodes = [entity("e1"), entity("e2")]
    assert validate_rows(nodes, [edge], mode="read") == []
    assert validate_rows(nodes, [edge]) != []


def test_i5_read_mode_still_checks_endpoints_when_present():
    edge = relates(source_uuid="zzz")
    assert validate_edge(edge, mode="read") == ["r1: source_uuid does not match edge source"]


def contract_node(kind, uuid, **over):
    props = {"uuid": uuid, "name": kind.lower(), "group_id": "g", "created_at": T} | over
    return {"labels_neo4j": [kind], "props": props}


def coding(etype, src, dst, uuid="c1", **over):
    base = {
        "ANSWERS": {"observation": "ev1", "role": "arg0", "plan": "pl1", "pass": "pass0"},
        "ANSWERS_VIA": {"role": "arg0", "wh": "who"},
    }.get(etype, {})
    props = {"uuid": uuid, "group_id": "g", "created_at": T} | base | over
    return {"type": etype, "source": src, "target": dst, "props": props}


@pytest.mark.parametrize("kind", ["Question", "Plan", "Purpose", "RevisionPass"])
def test_contract_node_kinds_validate_and_need_a_name(kind):
    assert validate_node(contract_node(kind, "x1", qid="q1")) == []
    row = contract_node(kind, "x1")
    del row["props"]["name"]
    assert validate_node(row) == ["x1: missing base field name"]


def test_coding_edges_accept_their_endpoint_kinds():
    nodes = [
        episodic("ep1"),
        entity("e1"),
        contract_node("Question", "q1"),
        contract_node("Plan", "pl1"),
        contract_node("Purpose", "pu1"),
        contract_node("RevisionPass", "rp1"),
    ]
    good = [
        coding("ANSWERS", "ep1", "q1", "a1", p=None),
        coding("ANSWERS_VIA", "pl1", "q1", "a2"),
        coding("DERIVED_FROM", "q1", "pu1", "a3"),
        coding("INVALIDATED_BY", "pl1", "rp1", "a4"),
        coding("INVALIDATED_BY", "q1", "rp1", "a5"),
    ]
    assert validate_rows(nodes, good) == []
    assert validate_rows(nodes, [coding("ANSWERS", "e1", "q1", "a1")]) == ["a1: source e1 must be :Episodic"]
    assert validate_rows(nodes, [coding("ANSWERS", "q1", "ep1", "a1")]) == [
        "a1: source q1 must be :Episodic",
        "a1: target ep1 must be :Question",
    ]
    assert validate_rows(nodes, [coding("INVALIDATED_BY", "pu1", "rp1", "a1")]) == [
        "a1: source pu1 must be :Plan or :Question"
    ]


def test_coding_edges_reject_missing_base_fields():
    edge = coding("ANSWERS", "ep1", "q1", "a1")
    del edge["props"]["plan"]
    assert validate_edge(edge) == ["a1: missing base field plan"]
    via = coding("ANSWERS_VIA", "pl1", "q1", "a2")
    del via["props"]["wh"]
    assert validate_edge(via) == ["a2: missing base field wh"]


def test_new_labels_validate_and_stay_off_entity():
    assert validate_node(contract_node("Roleset", "r1", roleset="go.01")) == []
    row = {"labels_neo4j": ["Entity", "Roleset"], "props": entity()["props"] | {"roleset": "go.01"}}
    assert "exactly one of" in validate_node(row)[0]


def test_routes_to_needs_route_and_entity_endpoints():
    route = contract_node("Route", "ro1", key="k1")
    ent = entity()
    edge = {
        "type": "ROUTES_TO",
        "source": "ro1",
        "target": "e1",
        "props": {"uuid": "rt1", "group_id": "g", "created_at": T},
    }
    assert validate_rows([route, ent], [edge]) == []
    bad = dict(edge, source="ep1")
    errors = validate_rows([episodic(), ent], [bad])
    assert any("must be :Route" in e for e in errors)


def test_message_embedding_must_be_numbers():
    assert validate_node(episodic(message_embedding=[0.1, 0.2])) == []
    assert validate_node(episodic(message_embedding=None)) == []
    assert any("message_embedding" in e for e in validate_node(episodic(message_embedding=["x"])))


def test_decision_fields_are_create_only():
    assert not DECISION_FIELDS & set(_split({f: 1 for f in DECISION_FIELDS} | {"uuid": "u"}, NODE_BASE["Entity"])[1])
