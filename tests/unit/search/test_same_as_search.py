"""Search follows ``SAME_AS``: a component is one person, named once, with its facts and timeline unioned."""

import asyncio

import pytest

from memory_kernel.search.kg_context import ContextConfig
from memory_kernel.search.kg_reader import (
    ENTITY_SHEET_CYPHER,
    IDENTITY_ALIASES_CYPHER,
    SAME_AS_COMPONENT_CYPHER,
    TIMELINE_CYPHER,
    build_answer_context,
    expand_same_as,
)
from memory_kernel.search.kg_recipe import KgRecipeConfig
from memory_kernel.search.kg_types import SearchResult


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        async def gen():
            for r in self.rows:
                yield r

        return gen()


class _Graph:
    """A fake reader: nodes ``{uuid: (name, summary, minted_order)}``, ``SAME_AS`` edges ``(from, to)`` and events
    ``{uuid: (name, happened_at, [entity uuids])}``. It answers the SAME_AS query by an undirected BFS bounded by the
    hop count written in the Cypher text itself."""

    def __init__(self, nodes, edges=(), events=None):
        self.nodes, self.edges, self.events = nodes, list(edges), events or {}
        self.calls: list[str] = []
        self.timeline_uuids: list[str] = []

    def session(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, cypher, **p):
        self.calls.append(cypher)
        if cypher == SAME_AS_COMPONENT_CYPHER:
            return _Rows(self._components(p["uuids"]))
        if cypher == IDENTITY_ALIASES_CYPHER:
            return _Rows([])
        if cypher == ENTITY_SHEET_CYPHER:
            return _Rows(
                [
                    {"uuid": u, "name": self.nodes[u][0], "summary": self.nodes[u][1]}
                    for u in p["uuids"]
                    if u in self.nodes
                ]
            )
        if cypher == TIMELINE_CYPHER:
            self.timeline_uuids = list(p["uuids"])
            out = []
            for eu, (name, when, ents) in self.events.items():
                hit = [u for u in ents if u in p["uuids"]]
                if hit:
                    out.append(
                        {
                            "uuid": eu,
                            "name": name,
                            "happened_at": when,
                            "told_at": when,
                            "time_source": "told",
                            "time_text": None,
                            "asserted": True,
                            "modality": None,
                            "attributes": None,
                            "entities": hit,
                        }
                    )
            return _Rows(out)
        return _Rows([])

    def _components(self, starts):
        hops = int(SAME_AS_COMPONENT_CYPHER.split("*1..")[1].split("]")[0])
        rows = []
        for s in starts:
            if s not in self.nodes:
                continue
            seen, frontier, paths = {s}, [(s, [])], []
            for _ in range(hops):
                nxt = []
                for node, path in frontier:
                    for a, b in self.edges:
                        if node in (a, b):
                            other = b if node == a else a
                            if other not in seen:
                                seen.add(other)
                                nxt.append((other, [*path, [a, b]]))
                                paths.append((other, [*path, [a, b]]))
                frontier = nxt
            rows += [
                {
                    "start": s,
                    "start_name": self.nodes[s][0],
                    "start_minted_order": self.nodes[s][2],
                    "uuid": t,
                    "name": self.nodes[t][0],
                    "minted_order": self.nodes[t][2],
                    "edges": e,
                }
                for t, e in paths
            ]
        return rows


def _result(*uuids):
    return SearchResult(
        hits=[],
        sagas=[],
        trace={u: {"score": 1.0 - i / 10} for i, u in enumerate(uuids)},
        items={u: {"name": u} for u in uuids},
        lists={"bm25_nodes": list(uuids), "semantic_nodes": []},
    )


def _ask(graph, query, cands, shape="fact"):
    async def search(driver, **_):
        return _result(*cands)

    return asyncio.run(
        build_answer_context(
            graph,
            query,
            [0.0],
            group_id="g",
            shape=shape,
            label=None,
            search_config=KgRecipeConfig(),
            context_config=ContextConfig(),
            search_fn=search,
        )
    )


def _tokens_ready():
    try:
        from memory_kernel.search.kg_context import count_tokens

        count_tokens("x")
        return True
    except Exception:
        return False


needs_tokens = pytest.mark.skipif(not _tokens_ready(), reason="token encoder not installed (mk setup)")


def test_the_cypher_is_undirected_bounded_and_group_scoped():
    c = SAME_AS_COMPONENT_CYPHER
    assert "[:SAME_AS*1..3]" in c and "->" not in c and "<-" not in c
    assert "$g" in c and "$uuids" in c and "group_id" in c


def test_expand_follows_either_direction_to_one_component():
    g = _Graph(
        {"dee": ("Dee", "", None), "ana": ("Ana", "", [2, 0]), "kit": ("Kit", "", [3, 1])},
        [("dee", "ana"), ("kit", "ana")],
    )
    comps = asyncio.run(expand_same_as(g, ["dee", "kit"], "g"))
    assert comps["dee"].id == comps["kit"].id == "ana"
    assert comps["dee"].name == "Ana" and comps["dee"].also == ("Dee", "Kit")
    assert set(comps["dee"].members) == {"dee", "ana", "kit"}


def test_a_four_hop_chain_is_cut_at_three_hops():
    names = ["n0", "n1", "n2", "n3", "n4"]
    g = _Graph({n: (n.upper(), "", None) for n in names}, [(a, b) for a, b in zip(names, names[1:], strict=False)])
    comp = asyncio.run(expand_same_as(g, ["n0"], "g"))["n0"]
    assert set(comp.members) == {"n0", "n1", "n2", "n3"}


def test_no_edges_means_no_components_and_one_query():
    g = _Graph({"a": ("A", "", None)})
    assert asyncio.run(expand_same_as(g, ["a"], "g")) == {}
    assert asyncio.run(expand_same_as(g, [], "g")) == {} and len(g.calls) == 1


def test_canonical_is_the_sink_else_earliest_minted_with_missing_as_run_zero():
    # b -> a and b -> c leave two sinks (a and c): the earliest minted of them is canonical
    g = _Graph({"a": ("Ana", "", [4, 0]), "b": ("Bea", "", None), "c": ("Cy", "", [1, 5])}, [("b", "a"), ("b", "c")])
    comp = asyncio.run(expand_same_as(g, ["b"], "g"))["b"]
    assert comp.id == "c"  # [1,5] is earlier than [4,0]
    g.nodes["c"] = ("Cy", "", None)  # a bootstrap entity counts as run 0: earliest of all
    assert asyncio.run(expand_same_as(g, ["b"], "g"))["b"].id == "c"
    g.nodes["a"] = ("Ana", "", None)  # both run 0: uuid breaks the tie
    assert asyncio.run(expand_same_as(g, ["b"], "g"))["b"].id == "a"


@needs_tokens
@pytest.mark.parametrize("shape", ["when", "how_long"])  # the shapes that get the timeline
def test_a_dee_question_reaches_both_nodes_and_names_the_person_once(shape):
    g = _Graph(
        {
            "dee": ("Dee", "[2026-01-01] Dee likes pears", None),
            "ana": ("Ana", "[2026-02-01] Ana likes oranges", [2, 0]),
        },
        [("dee", "ana")],
        {"ev1": ("went hiking", "2026-01-05", ["dee"]), "ev2": ("went sailing", "2026-02-05", ["ana"])},
    )
    ctx = _ask(g, "When did Dee go out?", ["dee"], shape=shape)
    text = ctx["text"]
    assert "Ana (also: Dee)" in text and text.count("Ana") >= 1
    assert "Dee likes pears" in text and "Ana likes oranges" in text
    assert sorted(g.timeline_uuids) == ["ana", "dee"]
    assert "went hiking" in text and "went sailing" in text
    assert text.count("Dee likes pears") == 1


@needs_tokens
def test_candidates_of_one_component_collapse_into_one_sheet():
    g = _Graph(
        {
            "dee": ("Dee", "Dee likes pears", None),
            "ana": ("Ana", "Ana likes oranges", None),
            "bo": ("Bo", "Bo swims", None),
        },
        [("dee", "ana")],
    )
    text = _ask(g, "Who likes fruit?", ["dee", "bo", "ana"])["text"]
    assert text.count("Ana (also: Dee)") == 1 and "\nDee\n" not in text and "Bo" in text


# Recorded from the pre-change reader (no SAME_AS edges anywhere): the output must not move by a byte.
_NO_EDGES_SNAPSHOT = "## Entity sheet\nBo\n[2026-01-02] Bo swims\n[2026-01-09] Bo bikes\nCy\nCy reads\n\n## Timeline\n[said 2026-01-03] swam"


@needs_tokens
def test_a_graph_without_same_as_renders_as_before():
    g = _Graph(
        {"bo": ("Bo", "[2026-01-02] Bo swims\n[2026-01-09] Bo bikes", None), "cy": ("Cy", "Cy reads", [1, 0])},
        [],
        {"ev1": ("swam", "2026-01-03", ["bo"])},
    )
    text = _ask(g, "When did Bo swim?", ["bo", "cy"], shape="when")["text"]
    assert text == _NO_EDGES_SNAPSHOT
