"""Contract invariants as row checks: one positive and one negative fixture per check."""

import copy

from memory_kernel.graph import g1_invariants as inv
from memory_kernel.graph.g1_folds import fold as fold_of


def _n(uuid, *labels, **props):
    return {"labels_neo4j": list(labels), "props": {"uuid": uuid, **props}}


def _e(uuid, etype, source, target, **props):
    return {"type": etype, "source": source, "target": target, "props": {"uuid": uuid, **props}}


def _good():
    return {
        "nodes": [
            _n("saga1", "Saga", ingest_pass=["p1"]),
            _n("saga2", "Saga", ingest_pass=["p1"]),
            _n("ep1", "Episodic"),
            _n("ep2", "Episodic"),
            _n("ev1", "Entity", "Event", plan="lang:occurrence@a"),
            _n("ent", "Entity"),
            _n("purpose", "Purpose", name="why"),
            _n("q1", "Question", name="Who?", **fold_of("accumulate", [{"uuid": "a1", "filler": "Dana"}])),
            _n("plan1", "Plan", name="g1r0:go.02@abcd1234"),
            _n("rev1", "RevisionPass", name="rev-001"),
        ],
        "edges": [
            _e("h1", "HAS_EPISODE", "saga1", "ep1"),
            _e("h2", "HAS_EPISODE", "saga2", "ep2"),
            _e("ns", "NEXT_SAGA", "saga1", "saga2"),
            _e("d1", "DERIVED_FROM", "q1", "purpose"),
            _e("r1", "RELATES_TO", "ev1", "ent", plan="lang:occurrence@a"),
            _e("av", "ANSWERS_VIA", "plan1", "q1", role="arg0", wh="who"),
            _e(
                "a1",
                "ANSWERS",
                "ep1",
                "q1",
                observation="ev1",
                observation_kind="event",
                role="arg0",
                filler="Dana",
                plan="g1r0:go.02@abcd1234",
                **{"pass": "p1"},
            ),
        ],
    }


def test_all_hold_on_good_rows():
    rows = _good()
    for check in (
        inv.derivation_acyclic,
        inv.observations_card_free,
        inv.plans_resolve,
        inv.codings_traceable,
        inv.codings_anchored,
        inv.observations_traceable,
        inv.folds_current,
    ):
        assert check(rows) == [], check.__name__
    assert (
        inv.budget(
            rows,
            baseline_mean=9.0,
            baseline_total_edges=99,
            baseline_node_uuids={"saga1", "saga2", "ep1", "ep2", "ev1", "ent"},
        )
        == []
    )


def test_derivation_cycle_caught():
    rows = _good()
    rows["edges"].append(_e("d2", "DERIVED_FROM", "purpose", "q1"))  # Question -> Purpose -> Question
    assert any("cycle" in v for v in inv.derivation_acyclic(rows))
    rows = _good()
    rows["edges"].append(_e("ns2", "NEXT_SAGA", "saga2", "saga1"))
    assert inv.derivation_acyclic(rows)
    rows = _good()
    rows["edges"].append(_e("inv", "INVALIDATED_BY", "plan1", "plan1"))  # self-loop
    assert inv.derivation_acyclic(rows)


def test_observation_with_card_plan_caught():
    rows = _good()
    rows["nodes"][4]["props"]["plan"] = "g1r0:go.02@x"
    assert len(inv.observations_card_free(rows)) == 1
    rows = _good()
    rows["edges"][4]["props"]["plan"] = "g1r0:go.02@x"
    assert len(inv.observations_card_free(rows)) == 1
    rows = _good()
    rows["nodes"][4]["props"]["card"] = "g1r0:go.02"
    assert len(inv.observations_card_free(rows)) == 1


def test_plan_must_resolve():
    rows = _good()
    rows["edges"][-1]["props"]["plan"] = "g1r0:gone.01@00000000"
    assert any("resolves to no Plan" in v for v in inv.plans_resolve(rows))
    rows = _good()
    rows["edges"][5]["source"] = "q1"  # ANSWERS_VIA from a non-Plan
    assert any("not a Plan" in v for v in inv.plans_resolve(rows))


def test_codings_traceable_negatives():
    rows = _good()
    del rows["edges"][-1]["props"]["pass"]
    assert any("no pass" in v for v in inv.codings_traceable(rows))
    rows = _good()
    rows["edges"][-1]["props"]["pass"] = "unknown"
    assert any("neither" in v for v in inv.codings_traceable(rows))
    rows = _good()
    del rows["edges"][-1]["props"]["plan"]
    assert any("no plan" in v for v in inv.codings_traceable(rows))
    rows = _good()
    rows["edges"] = [e for e in rows["edges"] if e["type"] != "DERIVED_FROM"]
    assert any("Purpose" in v for v in inv.codings_traceable(rows))


def test_revision_pass_name_is_a_valid_pass():
    rows = _good()
    rows["edges"][-1]["props"]["pass"] = "rev-001"
    assert inv.codings_traceable(rows) == []


def test_observations_traceable_negatives():
    rows = _good()
    rows["edges"].append(_e("h3", "HAS_EPISODE", "saga2", "ep1"))  # two incoming
    assert any("2 incoming" in v for v in inv.observations_traceable(rows))
    rows = _good()
    rows["edges"] = [e for e in rows["edges"] if e["props"]["uuid"] != "h2"]
    assert any("0 incoming" in v for v in inv.observations_traceable(rows))
    rows = _good()
    rows["nodes"][1]["props"]["ingest_pass"] = []
    assert any("empty ingest_pass" in v for v in inv.observations_traceable(rows))


def test_budget_negatives():
    rows = _good()
    base = {"saga1", "saga2", "ep1", "ep2", "ev1", "ent"}
    mean = inv.mean_episodic_degree(rows)
    assert mean == 1.5  # ep1: HAS_EPISODE + ANSWERS = 2; ep2: HAS_EPISODE = 1
    assert inv.budget(rows, baseline_mean=mean - 0.1, baseline_total_edges=99, baseline_node_uuids=base)
    assert inv.budget(
        rows, baseline_mean=mean, baseline_total_edges=inv.observation_edge_count(rows) - 1, baseline_node_uuids=base
    )
    bad = copy.deepcopy(rows)
    bad["nodes"].append(_n("hub", "Entity", "TaxonomyLabel"))
    assert any(
        "per-message node" in v
        for v in inv.budget(bad, baseline_mean=mean, baseline_total_edges=99, baseline_node_uuids=base)
    )


def test_budget_requires_baseline_nodes_and_reports_new_message_node():
    import pytest

    rows = _good()
    with pytest.raises(TypeError):
        inv.budget(rows, baseline_mean=9.0, baseline_total_edges=99)  # type: ignore[call-arg]
    bad = copy.deepcopy(rows)
    bad["nodes"].append(_n("msgnode", "Entity"))
    out = inv.budget(
        bad,
        baseline_mean=9.0,
        baseline_total_edges=99,
        baseline_node_uuids={"saga1", "saga2", "ep1", "ep2", "ev1", "ent"},
    )
    assert len(out) == 1 and "msgnode" in out[0]


def test_mixed_type_cycle_caught():
    rows = _good()
    rows["edges"].append(
        _e("inv2", "INVALIDATED_BY", "purpose", "q1")
    )  # q1 -DERIVED_FROM-> purpose -INVALIDATED_BY-> q1
    assert any("cycle" in v for v in inv.derivation_acyclic(rows))


def test_plan_without_name_does_not_resolve_missing_plan():
    rows = _good()
    rows["nodes"].append(_n("noname", "Plan"))
    del rows["edges"][-1]["props"]["plan"]
    assert any("resolves to no Plan" in v for v in inv.plans_resolve(rows))


def _summary_rows(line_id="L1", source="ep1", obs="ent"):
    rows = _good()
    ent = next(n for n in rows["nodes"] if n["props"]["uuid"] == "ent")
    ent["props"]["summary_line_ids"] = ["L1", "L2"]
    ent["props"]["summary_sources"] = ["ep1", "ep2"]
    a = rows["edges"][-1]["props"]
    a.update(observation=obs, observation_kind="summary_line", summary_line=line_id)
    rows["edges"][-1]["source"] = source
    return rows


def test_summary_line_anchor_passes_at_own_source_position():
    assert inv.codings_anchored(_summary_rows()) == []


def test_summary_line_at_another_sources_position_fails():
    assert any("is not the coding's Episodic" in v for v in inv.codings_anchored(_summary_rows(line_id="L2")))


def test_unknown_summary_line_fails():
    assert any("is not on entity" in v for v in inv.codings_anchored(_summary_rows(line_id="nope")))


def _appended_rows(line_id="A1", source="ep1"):
    """A forward append: the entity is stored, so its row carries only the lines the writer appends to its summary."""
    rows = _summary_rows(line_id=line_id, source=source)
    ent = next(n for n in rows["nodes"] if n["props"]["uuid"] == "ent")["props"]
    del ent["summary_line_ids"], ent["summary_sources"]
    ent["summary_append_line_ids"] = ["A1", "A2"]
    ent["summary_append_sources"] = ["ep1", "ep2"]
    return rows


def test_summary_line_appended_to_a_stored_entity_is_an_anchor():
    assert inv.codings_anchored(_appended_rows()) == []


def test_appended_summary_line_at_another_sources_position_fails():
    assert any("is not the coding's Episodic" in v for v in inv.codings_anchored(_appended_rows(line_id="A2")))


def test_summary_line_data_is_parallel_to_the_line_ids():
    """Per-line data (texts, identity ids) is all absent or full length: texts one per line id, re-joining to the
    summary, never one list without the other; identity ids a subset of the line ids, in line order. The appended
    lists of a stored row likewise."""
    rows, appended = _summary_rows(), _appended_rows()
    ent = next(n for n in rows["nodes"] if n["props"]["uuid"] == "ent")["props"]
    ent |= {"summary": "a\nb\nc", "summary_line_texts": ["a", "b\nc"], "summary_identity_line_ids": ["L2"]}
    app = next(n for n in appended["nodes"] if n["props"]["uuid"] == "ent")["props"]
    app |= {"summary_append": ["a", "b"], "summary_append_identity_line_ids": ["A1"]}
    assert inv.codings_anchored(rows) == [] and inv.codings_anchored(appended) == []
    for props, bad in (
        (ent, {"summary_line_texts": ["a"]}),
        (ent, {"summary_line_texts": None}),  # identity ids without texts
        (ent, {"summary_identity_line_ids": None}),  # texts without identity ids
        (ent, {"summary_identity_line_ids": ["L9"]}),
        (ent, {"summary_identity_line_ids": ["L2", "L1"]}),
        (app, {"summary_append_identity_line_ids": ["L1"]}),
        (app, {"summary_append": ["a"]}),
    ):
        kept = dict(props)
        props |= bad
        assert inv.codings_anchored(rows) + inv.codings_anchored(appended), bad
        props.clear()
        props |= kept


def test_event_anchor_must_be_an_event():
    rows = _good()
    rows["edges"][-1]["props"]["observation"] = "ent"
    assert any("is not an Event" in v for v in inv.codings_anchored(rows))


def test_fact_edge_anchor_needs_event_keys():
    rows = _good()
    a = rows["edges"][-1]["props"]
    a.update(observation="r1", observation_kind="fact_edge")
    assert any("not a fact edge" in v for v in inv.codings_anchored(rows))
    rows["edges"][4]["props"]["event_keys"] = ["k"]
    assert inv.codings_anchored(rows) == []


def test_unknown_anchor_kind_fails():
    rows = _good()
    rows["edges"][-1]["props"]["observation_kind"] = "vibes"
    assert any("observation_kind" in v for v in inv.codings_anchored(rows))


def test_answers_without_observation_kind_is_not_traceable():
    rows = _good()
    del rows["edges"][-1]["props"]["observation_kind"]
    assert any("no observation anchor" in v for v in inv.codings_traceable(rows))
    assert inv.codings_anchored(rows)
    rows = _good()
    del rows["edges"][-1]["props"]["observation"]
    assert any("no observation anchor" in v for v in inv.codings_traceable(rows))


def test_budget_counts_observation_edges_only_d2():
    rows = _good()
    base = {"saga1", "saga2", "ep1", "ep2", "ev1", "ent"}
    obs = inv.observation_edge_count(rows)
    assert obs == len(rows["edges"]) - 3  # DERIVED_FROM, ANSWERS_VIA, ANSWERS are coding edges
    mean = inv.mean_episodic_degree(rows)
    kw = {"baseline_mean": mean + 5, "baseline_node_uuids": base}
    # coding edges push the total over the baseline while observation edges stay at it
    assert inv.budget(rows, baseline_total_edges=obs, **kw) == []
    assert len(rows["edges"]) > obs
    # one extra observation edge is a violation
    rows["edges"].append(_e("r2", "RELATES_TO", "ev1", "ent", plan="lang:occurrence@a"))
    assert any("observation edges" in v for v in inv.budget(rows, baseline_total_edges=obs, **kw))


def test_a_cached_fold_that_disagrees_with_the_live_answers_is_caught():
    rows = _good()
    assert inv.folds_current(rows) == []
    rows["nodes"][7]["props"]["fold_count"] = 2
    assert len(inv.folds_current(rows)) == 1


def test_an_invalidated_answers_is_not_counted():
    rows = _good()
    rows["edges"][-1]["props"]["invalidated_by"] = "rev-001"
    assert len(inv.folds_current(rows)) == 1  # the cached fold still counts it
    rows["nodes"][7]["props"].update(fold_count=0, fold_answer=[], fold_sha=fold_of("accumulate", [])["fold_sha"])
    assert inv.folds_current(rows) == []
