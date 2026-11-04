import asyncio
import functools

import pytest

from memory_kernel.grammar.dependencies import wordnet_ready
from memory_kernel.search import kg_reader
from memory_kernel.search.kg_reader import IDENTITY_ALIASES_CYPHER, _top_entities, aliased_uuids, build_answer_context
from memory_kernel.search.kg_recipe import KgRecipeConfig
from memory_kernel.search.kg_types import SearchResult


def _result():
    return SearchResult(
        hits=[],
        sagas=[],
        trace={"u-bo": {"score": 0.9}, "u-cy": {"score": 0.5}},
        items={"u-bo": {"name": "Bo"}, "u-cy": {"name": "Cy"}},
        lists={"bm25_nodes": ["u-bo"], "semantic_nodes": ["u-cy"]},
    )


def test_alias_named_entity_comes_first_even_outside_the_pool():
    assert _top_entities(_result(), "What fruit does Dee like?", 3, alias_named=["u-ana"]) == ["u-ana", "u-bo", "u-cy"]


def test_alias_named_entity_in_the_pool_is_not_repeated():
    assert _top_entities(_result(), "What does Dee like?", 3, alias_named=["u-cy"]) == ["u-cy", "u-bo"]


def test_without_alias_named_the_order_is_unchanged():
    assert _top_entities(_result(), "What fruit does Dee like?", 3) == ["u-bo", "u-cy"]
    assert _top_entities(_result(), "What fruit does Dee like?", 3, alias_named=()) == ["u-bo", "u-cy"]
    assert _top_entities(_result(), "Is Cy here?", 3, alias_named=()) == ["u-cy", "u-bo"]


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_aliased_uuids_match_whole_words_only():
    rows = [
        {"uuid": "u-ana", "name": "Ana", "identity_aliases": ["Dee", "Ana"]},
        {"uuid": "u-kim", "name": "Kim", "identity_aliases": ["Deep sea"]},
        {"uuid": "u-bo", "name": "Bo", "identity_aliases": None},
    ]
    assert aliased_uuids(rows, "What fruit does Dee like?") == ["u-ana"]
    assert aliased_uuids(rows, "What does dee like?") == ["u-ana"]
    assert aliased_uuids(rows, "Is Deep sea fishing fun?") == ["u-kim"]
    assert aliased_uuids(rows, "Did Deeds happen?") == []
    assert aliased_uuids([], "Dee") == []


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_a_common_noun_alias_matches_only_capitalised_and_not_sentence_initial():
    rows = [{"uuid": "u-ana", "name": "Ana", "identity_aliases": ["Egg"]}]
    assert aliased_uuids(rows, "Who cooked the egg for breakfast?") == []
    assert aliased_uuids(rows, "Egg is life.") == []
    assert aliased_uuids(rows, "What did I ask Egg about?") == ["u-ana"]


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_a_proper_only_alias_matches_lowercase():
    rows = [{"uuid": "u-ana", "name": "Ana", "identity_aliases": ["Dee", "Mary"]}]
    assert aliased_uuids(rows, "what fruit does dee like") == ["u-ana"]
    assert aliased_uuids(rows, "mary likes what?") == ["u-ana"]


def test_cypher_reads_only_group_entities_with_aliases():
    assert "identity_aliases IS NOT NULL" in IDENTITY_ALIASES_CYPHER and "$g" in IDENTITY_ALIASES_CYPHER


class _Driver:
    """Answers each Cypher by what it asks for; records the identity-alias round trips."""

    def __init__(self, alias_rows):
        self.alias_rows, self.alias_calls, self.sheet_uuids = alias_rows, 0, []

    def session(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, cypher, **params):
        rows = []
        if cypher == IDENTITY_ALIASES_CYPHER:
            self.alias_calls += 1
            rows = self.alias_rows
        elif "n.summary" in cypher:
            self.sheet_uuids = list(params["uuids"])
            rows = [{"uuid": u, "name": u, "summary": ""} for u in params["uuids"]]
        return _Rows(rows)


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        async def gen():
            for r in self.rows:
                yield r

        return gen()


async def _search(driver, **_):
    return _result()


def _ask(driver, query, monkeypatch):
    from memory_kernel.search.kg_context import ContextConfig

    # the token encoder is runtime data (`mk setup`); what is assembled is not under test here
    monkeypatch.setattr(kg_reader, "assemble_context", lambda *a, **k: {})

    return asyncio.run(
        build_answer_context(
            driver,
            query,
            [0.0],
            group_id="g",
            shape="fact",
            label=None,
            search_config=KgRecipeConfig(),
            context_config=ContextConfig(),
            search_fn=functools.partial(_search),
        )
    )


def test_build_answer_context_fetches_the_aliased_sheet_with_one_round_trip(monkeypatch):
    driver = _Driver([{"uuid": "u-ana", "name": "Ana", "identity_aliases": ["Dee"]}])
    _ask(driver, "What fruit does Dee like?", monkeypatch)
    assert driver.alias_calls == 1
    assert driver.sheet_uuids[0] == "u-ana"


def test_build_answer_context_without_aliases_keeps_the_pool_order(monkeypatch):
    driver = _Driver([])
    _ask(driver, "What fruit does Dee like?", monkeypatch)
    assert driver.alias_calls == 1
    assert driver.sheet_uuids == ["u-bo", "u-cy"]
