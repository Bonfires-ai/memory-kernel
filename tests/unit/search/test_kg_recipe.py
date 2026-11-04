import asyncio

from memory_kernel.search.kg_recipe import (
    KgRecipeConfig,
    expanded_key,
    rank_recipe,
    search_messages_v2,
)


def _rows(order=1):
    msgs = [
        {
            "uuid": "m1",
            "message_id": "D1:1",
            "content": "I went to a mosaic class.",
            "told_at": "2024-02-12T10:00:00+00:00",
            "sender_name": "Ruth",
            "saga": "s1",
        },
        {
            "uuid": "m2",
            "message_id": "D1:2",
            "content": "I love knitting scarves.",
            "told_at": "2024-03-01T10:00:00+00:00",
            "sender_name": "Dana",
            "saga": "s1",
        },
    ]
    facts = [
        {
            "uuid": "f2",
            "fact": "Dana knit scarves (arg1: scarves)",
            "episodes": ["m2"],
            "source": "ev2",
            "target": "scarf",
            "emb": [0.0, 1.0],
            "asserted": True,
            "modality": "actual",
        },
        {
            "uuid": "f1",
            "fact": "Ruth go to mosaic class (arg4: mosaic class)",
            "episodes": ["m1"],
            "source": "ev1",
            "target": "per1",
            "emb": [1.0, 0.0],
            "asserted": True,
            "modality": "actual",
        },
        {
            "uuid": "f3",
            "fact": "Ruth go to mosaic class",
            "episodes": ["m1"],
            "source": "per1",
            "target": "grp",
            "emb": [1.0, 0.1],
            "asserted": True,
            "modality": "actual",
        },
    ]
    ents = {
        "per1": {"name": "Ruth", "kinds": ["Person"]},
        "grp": {"name": "mosaic class", "kinds": []},
        "scarf": {"name": "scarves", "kinds": []},
        "per2": {"name": "Dana", "kinds": ["Person"]},
        "ev1": {"name": "Ruth go to mosaic class", "kinds": ["Event"], "happened_at": "2024-02-11"},
        "ev2": {"name": "Dana knit scarves", "kinds": ["Event"], "happened_at": None},
        "lab": {"name": "groups", "kinds": ["TaxonomyLabel"]},
    }
    mentions = [
        ("m1", "per1", None),
        ("m1", "grp", None),
        ("m1", "ev1", None),
        ("m1", "lab", 0.8),
        ("m2", "per2", None),
        ("m2", "ev2", None),
        ("m2", "scarf", None),
    ]
    if order == -1:
        msgs, facts, mentions = msgs[::-1], facts[::-1], mentions[::-1]
    return msgs, facts, ents, mentions


def test_expansion_is_deterministic_dedupes_and_has_no_attribution():
    assert expanded_key("c", ["x (arg0: y)", "x"], [], []) == "c\nx"
    a = expanded_key("I went.", ["b (arg1: z)", "a"], ["ev"], ["Ruth"])
    b = expanded_key("I went.", ["a", "b"], ["ev"], ["Ruth"])
    assert a == b and "arg1" not in a


def _index():
    from memory_kernel.search.kg_index import KgIndex

    return KgIndex(
        keys={"m1": "k1", "m2": "k2"},
        vectors={"m1": [1.0, 0.0], "m2": [0.0, 1.0]},
        msg_facts={
            "m1": [{"uuid": "f1", "fact": "Ruth go", "emb": [1.0, 0.0], "asserted": True, "modality": "actual"}],
            "m2": [],
        },
        msg_events={"m1": [{"uuid": "ev1", "name": "Ruth go", "asserted": True, "modality": "actual"}], "m2": []},
        msg_entities={"m1": [], "m2": []},
        told_at={"m1": "2024-02-12", "m2": "2024-03-01"},
        messages={
            "m1": {"uuid": "m1", "message_id": "D1:1", "content": "a", "sender_name": "C", "saga": "s1"},
            "m2": {"uuid": "m2", "message_id": "D1:2", "content": "b", "sender_name": "M", "saga": "s1"},
        },
        dim=2,
    )


def test_rank_recipe_is_cosine_over_the_configured_key_with_a_list_limit():
    idx = _index()
    assert [m for m, _ in rank_recipe(idx, qvec=[0.0, 1.0], cfg=KgRecipeConfig())] == ["m2", "m1"]
    assert [m for m, _ in rank_recipe(idx, qvec=[1.0, 0.0], cfg=KgRecipeConfig(list_limit=1))] == ["m1"]


def test_search_messages_v2_returns_hits_with_support_and_sagas():
    res = asyncio.run(
        search_messages_v2(None, group_id="g", query="q", qvec=[1.0, 0.0], shape="what", label=None, index=_index())
    )
    assert [h.message for h in res.hits] == ["m1", "m2"]
    assert res.hits[0].support == ("f1", "ev1")
    assert res.sagas[0][0] == "s1" and res.messages["m1"]["message_id"] == "D1:1"


def test_recipe_defaults_are_the_measured_winner():
    import dataclasses

    from memory_kernel.search.kg_recipe import SEALED_RECIPE, KgRecipeConfig

    cfg = KgRecipeConfig()
    assert dataclasses.asdict(cfg) == {
        "key": "expanded",
        "list_limit": 50,
        "max_support": 3,
        "key_render": "legacy",
        "bm25_weight": 0.0,
    }
    assert cfg == SEALED_RECIPE


def test_no_sibling_recipe_config_name_clash():
    import memory_kernel.search.kg_recipe as kr

    assert not hasattr(kr, "RecipeConfig")


def test_linear_blend_matches_production_formula():
    import pytest

    from memory_kernel.search.kg_recipe import blend

    out = blend({"a": 0.9, "b": 0.5}, {"a": 1.0, "b": 3.0}, bm25_weight=0.3)
    assert out["a"] == pytest.approx(0.7 * 1.0 + 0.3 * 0.0)
    assert out["b"] == pytest.approx(0.7 * 0.0 + 0.3 * 1.0)


def test_bm25_weight_defaults_off_and_blend_reranks_by_keyword():
    assert KgRecipeConfig().bm25_weight == 0.0
    idx = _index()
    idx.keys = {"m1": "alpha beta", "m2": "gamma delta"}
    dense_only = rank_recipe(idx, qvec=[0.6, 0.55], cfg=KgRecipeConfig(), query="gamma")
    assert [m for m, _ in dense_only][0] == "m1"
    blended = rank_recipe(idx, qvec=[0.6, 0.55], cfg=KgRecipeConfig(bm25_weight=0.6), query="gamma")
    assert [m for m, _ in blended][0] == "m2"
