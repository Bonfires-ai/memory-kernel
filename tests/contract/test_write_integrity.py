"""Write integrity on a shared Neo4j: what a change of ANOTHER group does to a write.

By default nothing: a write checks only its own group (an append: the writer's foreign-uuid refusal and the per-uuid
readback of what it wrote), so another tenant writing the same database at the same time never fails it, and no
database-wide snapshot is taken. ``BuildConfig.strict_db_snapshot`` (the bench's strict mode) snapshots every group
before, during and after the write, and any change of another group fails it; ``--concurrent-group-prefix``
(``BuildConfig.concurrent_group_prefix``) then tolerates GROWTH of the other groups whose name starts with the prefix
and nothing else: not a shrink, and never the ungrouped bucket. All in-process and offline: the snapshots are plain
dicts, one driver-shaped case runs ``graphdb.snapshot`` against a stub driver, and the append cases run the write
stage against the append tests' fake Neo4j."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from unit.pipeline import test_write_append_mode as append_mode

PREFIX = "g1-test-c2-"
MINE = "mine"
db = append_mode.db  # the append tests' fake Neo4j behind the write stage, as a fixture here


def other_groups(before: dict, after: dict) -> dict:
    """``result["other_groups"]`` for two snapshots ``{"nodes": {group: n}, "edges": {group: n}}`` (this build's group
    ``MINE`` is always in both and never counted)."""
    from memory_kernel.pipeline import graphdb

    base = {"nodes": {MINE: 1}, "edges": {MINE: 1}}
    return graphdb.other_groups_unchanged(
        {k: {**base[k], **before.get(k, {})} for k in base}, {k: {**base[k], **after.get(k, {})} for k in base}, MINE
    )


def test_a_prefix_tolerates_growth_of_the_groups_it_names_and_nothing_else():
    from memory_kernel.pipeline.stages.write import foreign_changes

    before = {"nodes": {PREFIX + "a": 4, "live": 9}, "edges": {PREFIX + "a": 2}}
    after = {
        "nodes": {PREFIX + "a": 6, PREFIX + "new": 5, "live": 12},  # one grew, one is new, one is not a prefix match
        "edges": {PREFIX + "a": 3, PREFIX + "new": 7},
    }
    changed = foreign_changes(other_groups(before, after), prefix=PREFIX)
    assert set(changed) == {"nodes:live"}  # only the group the prefix does not name still fails
    assert changed["nodes:live"] == {"before": 9, "after": 12}
    # the prefix is a prefix: a group that merely contains it, or shares part of it, is not named
    after = {"nodes": {"x" + PREFIX + "a": 1, "g1-test-c3-a": 1}}
    assert set(foreign_changes(other_groups({}, after), prefix=PREFIX)) == {f"nodes:x{PREFIX}a", "nodes:g1-test-c3-a"}


def test_a_shrink_of_any_group_fails_even_under_a_matching_prefix():
    from memory_kernel.pipeline.stages.write import foreign_changes

    shrunk = other_groups(
        {"nodes": {PREFIX + "a": 5, PREFIX + "gone": 2}, "edges": {PREFIX + "a": 4}},
        {"nodes": {PREFIX + "a": 3}, "edges": {PREFIX + "a": 4}},  # a lost nodes, gone vanished
    )
    assert set(foreign_changes(shrunk, prefix=PREFIX)) == {f"nodes:{PREFIX}a", f"nodes:{PREFIX}gone"}
    assert foreign_changes(shrunk, prefix=PREFIX)[f"nodes:{PREFIX}gone"] == {"before": 2, "after": 0}


class _Rows:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __aiter__(self):
        async def gen():
            for row in self.rows:
                yield row

        return gen()


class _Session:
    def __init__(self, nodes: list[dict], edges: list[dict]):
        self.nodes, self.edges = nodes, edges

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query: str):
        return _Rows(self.nodes if query.startswith("MATCH (n)") else self.edges)


def test_the_ungrouped_bucket_is_never_tolerated_whatever_the_prefix():
    from memory_kernel.pipeline import graphdb
    from memory_kernel.pipeline.stages.write import UNGROUPED, foreign_changes

    # the key the guard excludes IS the key ``graphdb.snapshot`` gives nodes and edges that have no group_id
    session = _Session([{"g": None, "c": 3}, {"g": "a", "c": 1}], [{"g": None, "c": 2}])
    snapshot = asyncio.run(graphdb.snapshot(SimpleNamespace(session=lambda: session)))
    assert snapshot == {"nodes": {UNGROUPED: 3, "a": 1}, "edges": {UNGROUPED: 2}}

    grown = other_groups(
        {"nodes": {UNGROUPED: 3}, "edges": {UNGROUPED: 2}}, {"nodes": {UNGROUPED: 8}, "edges": {UNGROUPED: 9}}
    )
    assert set(grown["changed"]) == {f"nodes:{UNGROUPED}", f"edges:{UNGROUPED}"}
    for prefix in (None, PREFIX, "N", "No", UNGROUPED):  # the last three would match the text "None"
        assert set(foreign_changes(grown, prefix=prefix)) == set(grown["changed"]), prefix
