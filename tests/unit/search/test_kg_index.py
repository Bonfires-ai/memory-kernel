import asyncio

import pytest

from memory_kernel.search.kg_index import (
    IndexModelMismatch,
    KgIndex,
    build_index,
    build_keys,
    check_model,
    check_query_dim,
)
from memory_kernel.search.kg_recipe import KgRecipeConfig, search_messages_v2


def _row(**kw):
    base = {
        "uuid": "m1",
        "content": "Ruth: I went yesterday.",
        "told_at": "2024-02-12T09:20:00+00:00",
        "valid_at": None,
        "facts": [],
        "fact_invalid": [],
        "events": [],
        "entities": [],
    }
    return base | kw


def test_key_falls_back_to_valid_at_for_graphiti_episodes():
    keys = build_keys([_row(told_at=None, valid_at="2024-02-12T09:20:00+00:00")])
    assert "yesterday = 11 February 2024" in keys["m1"]  # the deixis seam renders the long date


def test_invalidated_facts_are_excluded_from_keys():
    keys = build_keys([_row(facts=["Ruth go to group"], fact_invalid=[True])])
    assert "go to group" not in keys["m1"]


def test_valid_facts_events_and_entities_are_in_the_key():
    keys = build_keys(
        [
            _row(
                facts=["Ruth go to group (arg1: group)"],
                fact_invalid=[False],
                events=["go ev"],
                entities=["Ruth"],
            )
        ]
    )
    assert "Ruth go to group" in keys["m1"] and "arg1" not in keys["m1"]
    assert "go ev" in keys["m1"]


def test_model_or_dim_mismatch_refuses():
    with pytest.raises(IndexModelMismatch):
        check_model(stored_model="text-embedding-3-small", stored_dim=1536, model="voyage-3", dim=1024)
    with pytest.raises(IndexModelMismatch):
        check_model(stored_model="voyage-3", stored_dim=512, model="voyage-3", dim=1024)
    check_model(stored_model="voyage-3", stored_dim=1024, model="voyage-3", dim=1024)


def test_wrong_dimension_query_raises_index_model_mismatch_not_value_error():
    with pytest.raises(IndexModelMismatch):
        check_query_dim([0.1, 0.2], 1024)
    check_query_dim([0.1] * 4, 4)


def _index(dim=2):
    return KgIndex(
        keys={"m1": "k1", "m2": "k2", "m3": "k3"},
        vectors={"m1": [1.0, 0.0][:dim], "m2": [0.0, 1.0][:dim], "m3": [0.7, 0.7][:dim]},
        msg_facts={
            "m1": [{"uuid": "f1", "fact": "Ruth go", "emb": [1.0, 0.0], "asserted": True, "modality": "actual"}],
            "m2": [],
            "m3": [],
        },
        msg_events={
            "m1": [{"uuid": "ev1", "name": "Ruth go", "asserted": True, "modality": "actual"}],
            "m2": [],
            "m3": [],
        },
        msg_entities={"m1": [{"uuid": "per1", "name": "Ruth"}], "m2": [], "m3": []},
        told_at={"m1": "2024-02-12"},
        messages={
            m: {
                "uuid": m,
                "message_id": f"D1:{i}",
                "content": m,
                "sender_name": "x",
                "told_at": "2024-02-12",
                "saga": "s",
            }
            for i, m in enumerate(("m1", "m2", "m3"))
        },
        model="voyage-3",
        dim=dim,
    )


def test_search_v2_returns_only_list_limit_hits_with_support():
    res = asyncio.run(
        search_messages_v2(
            None, group_id="g", query="q", qvec=[1.0, 0.0], config=KgRecipeConfig(list_limit=2), index=_index()
        )
    )
    assert [h.message for h in res.hits] == ["m1", "m3"]
    assert "f1" in res.hits[0].support and "ev1" in res.hits[0].support
    assert res.sagas[0][0] == "s"


def test_search_v2_wrong_dim_query_is_index_model_mismatch():
    with pytest.raises(IndexModelMismatch):
        asyncio.run(search_messages_v2(None, group_id="g", query="q", qvec=[1.0, 0.0, 0.0], index=_index()))


def test_build_index_facts_are_indexed_by_message_once_and_invalid_dropped():
    msgs = [
        {
            "uuid": "m1",
            "message_id": "D1:1",
            "content": "c",
            "sender_name": "x",
            "told_at": None,
            "valid_at": None,
            "saga": None,
        }
    ]
    facts = [
        {
            "uuid": "f1",
            "fact": "a b",
            "episodes": ["m1", "m1"],
            "emb": None,
            "asserted": True,
            "modality": None,
            "invalid": False,
        },
        {
            "uuid": "f2",
            "fact": "old",
            "episodes": ["m1"],
            "emb": None,
            "asserted": True,
            "modality": None,
            "invalid": True,
        },
    ]
    idx = build_index(msgs, facts, [], {}, model="voyage-3", dim=2)
    assert [f["uuid"] for f in idx.msg_facts["m1"]] == ["f1"]
    assert "old" not in idx.keys["m1"]


def test_search_raises_index_stale_unless_allowed_and_excludes_missing():
    from dataclasses import replace

    from memory_kernel.search.kg_index import IndexStale

    idx = replace(_index(), stale=frozenset({"m2"}), missing=frozenset({"m3"}))
    idx.vectors.pop("m3")
    with pytest.raises(IndexStale) as e:
        asyncio.run(search_messages_v2(None, group_id="g", query="q", qvec=[1.0, 0.0], index=idx))
    assert e.value.stale == frozenset({"m2"}) and e.value.missing == frozenset({"m3"})
    assert not isinstance(e.value, IndexModelMismatch)
    res = asyncio.run(search_messages_v2(None, group_id="g", query="q", qvec=[1.0, 0.0], index=idx, allow_stale=True))
    assert [h.message for h in res.hits] == ["m1", "m2"]


def test_fact_invalid_shorter_than_facts_is_refused():
    with pytest.raises(ValueError):
        build_keys([_row(facts=["a b", "c d"], fact_invalid=[False])])


# --- compact keys ----------------------------------------------------------------------------------------


def _event(**kw):
    base = {
        "uuid": "e1",
        "roleset": "go.02",
        "happened_at": "2024-02-11",
        "time_source": "deictic",
        "roles": [
            {"tag": "PPT", "role": "arg0", "target": "Ruth"},
            {"tag": "GOL", "role": "arg4", "target": "mosaic class"},
        ],
        "texts": {"argm_tmp_text": "yesterday"},
    }
    return base | kw


def _compact_row(content, events, states=()):
    return _row(content=content, compact_events=list(events), compact_states=list(states))


def test_compact_fragment_for_d1_3_is_exact():
    row = _compact_row("Ruth: I went to a mosaic class yesterday.", [_event()])
    key = build_keys([row], key_render="compact")["m1"]
    assert key.split("\n")[1] == "Ruth go mosaic class 2024-02-11"


def test_compact_keeps_fragment_contained_in_its_message():
    row = _compact_row("Ruth: I went to a mosaic class.", [_event(happened_at=None, time_source="told_default")])
    assert build_keys([row], key_render="compact")["m1"].split("\n")[1:] == ["Ruth go mosaic class"]


def test_compact_told_default_event_carries_no_date():
    row = _compact_row("Ruth: We saw a sunset.", [_event(time_source="told_default")])
    key = build_keys([row], key_render="compact")["m1"]
    assert "Ruth go mosaic class" in key and "2024-02-11" not in key


def test_compact_state_fragment_and_non_entity_filler_text():
    ev = _event(roles=[{"tag": "PAG", "role": "arg0", "target": "Dana"}], texts={"arg1_text": "a small brass lantern"})
    st = {
        "roleset": "happy.01",
        "source": "Dana",
        "target": "the news",
        "happened_at": "2024-03-01",
        "time_source": "told_default",
    }
    key = build_keys([_compact_row("Dana: hello.", [ev], [st])], key_render="compact")["m1"]
    assert "Dana go a small brass lantern" in key and "Dana happy the news" in key


def test_compact_recipe_tag_and_mismatch_refuses():
    from memory_kernel.search.kg_index import recipe_tag

    assert recipe_tag("compact") == "compact|deixis=on|v3" and recipe_tag() == "expanded|deixis=on|v1"
    assert KgRecipeConfig().key_render == "legacy"
    with pytest.raises(IndexModelMismatch):
        from memory_kernel.search.kg_index import check_recipe

        check_recipe(stored="expanded|deixis=on|v1", expected=recipe_tag("compact"), uuid="m1")


def test_legacy_key_layout_and_recipe_tag_enforced():
    from memory_kernel.search.kg_index import check_recipe, recipe_tag

    assert recipe_tag("legacy") == "legacy|deixis=on|v4"
    with pytest.raises(IndexModelMismatch):
        check_recipe(stored=recipe_tag("compact"), expected=recipe_tag("legacy"), uuid="m1")
    msgs = [{"uuid": "m1", "content": "Ruth: I went.", "told_at": "2024-02-12T09:20:00+00:00", "valid_at": None}]
    ments = [
        {
            "m": "m1",
            "uuid": "e1",
            "name": "Ruth went there.",
            "labels": ["Entity", "Event"],
            "key_text": "Ruth go there",
        },
        {"m": "m1", "uuid": "n1", "name": "Ruth", "labels": ["Entity", "Person"]},
    ]
    facts = [{"uuid": "f1", "fact": "Ruth went there.", "key_text": "Ruth go there", "episodes": ["m1"]}]
    idx = build_index(msgs, facts, ments, {}, model="m", dim=2, key_render="legacy")
    assert idx.keys["m1"].split("\n")[1:] == ["Ruth go there", "Ruth"]
    # A Graphiti-written fact has no key_text; the key falls back to the display fact text, never raises
    no_kt = build_index(msgs, [{**facts[0], "key_text": None}], ments, {}, model="m", dim=2, key_render="legacy")
    assert "Ruth went there." in no_kt.keys["m1"]


def test_legacy_env_key_renders_only_the_message_line_through_the_envelope():
    from memory_kernel.search.kg_index import recipe_tag

    assert recipe_tag("legacy-env") == "legacy-env|deixis=on|v5"
    msgs = [
        {
            "uuid": "m1",
            "content": "Ruth: I went to a mosaic class yesterday and my mom loved it.",
            "sender_name": "Ruth",
            "addressee_name": "Dana",
            "told_at": "2024-02-12T09:20:00+00:00",
            "valid_at": None,
        }
    ]
    ments = [
        {
            "m": "m1",
            "uuid": "e1",
            "name": "Ruth went.",
            "labels": ["Entity", "Event"],
            "key_text": "Ruth go there",
        },
        {"m": "m1", "uuid": "n1", "name": "Ruth", "labels": ["Entity", "Person"]},
    ]
    facts = [{"uuid": "f1", "fact": "Ruth went.", "key_text": "Ruth go there", "episodes": ["m1"]}]
    env = build_index(msgs, facts, ments, {}, model="m", dim=2, key_render="legacy-env").keys["m1"].split("\n")
    old = build_index(msgs, facts, ments, {}, model="m", dim=2, key_render="legacy").keys["m1"].split("\n")
    assert env[0] == "Ruth went to a mosaic class on 2024-02-11 and Ruth's mom loved it."
    assert old[0].startswith("Ruth: I went to a mosaic class yesterday")
    assert env[1:] == old[1:] == ["Ruth go there", "Ruth"]


def test_recipe_tag_follows_the_deixis_flag_and_refuses_an_unknown_render(monkeypatch):
    from memory_kernel.knowledge.temporal_deixis import TEMPORAL_DEIXIS_ENV
    from memory_kernel.search.kg_index import recipe_tag

    monkeypatch.setenv(TEMPORAL_DEIXIS_ENV, "0")
    assert recipe_tag("legacy") == "legacy|deixis=off|v4"
    monkeypatch.delenv(TEMPORAL_DEIXIS_ENV)
    assert recipe_tag("legacy") == "legacy|deixis=on|v4"
    with pytest.raises(ValueError):
        recipe_tag("nope")


@pytest.mark.parametrize(
    ("text", "addressee", "expected"),
    [
        ("I'm here and I've won", "Milo", "Ana is here and Ana has won"),
        ("I'd gone home and I'd love tea", None, "Ana had gone home and Ana would love tea"),
        ("you're kind and you'll see", "Milo", "Milo is kind and Milo will see"),
        ("you're kind and you'll see", None, "you're kind and you'll see"),
        (
            "we're here, we've won, we'd go",
            "Milo",
            "Ana and Milo are here, Ana and Milo have won, Ana and Milo would go",
        ),
        ("you both win, and it took us two hours", "Milo", "Ana and Milo win, and it took Ana and Milo two hours"),
        ("the US team, me-time and my dog", "Milo", "the US team, me-time and Ana's dog"),
        ("a friend of mine and yours", "Milo", "a friend of Ana's and Milo's"),
        ("Wed. I really want it", None, "Wed. Ana really wants it"),
    ],
)
def test_render_reads_pronouns_and_clitics_from_the_parse(text, addressee, expected):
    from memory_kernel.envelope.render import render

    assert render(text, speaker="Ana", addressee=addressee, ts="2024-02-12T09:20:00+00:00") == expected
