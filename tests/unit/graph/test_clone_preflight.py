"""A uuid is a global identity. A bonfire whose chain was minted under another group (a copy that kept its source's
uuids: a clone) is refused before an append does anything, and the writer's foreign-uuid refusal names
the group that holds each uuid.

Offline: the writer runs against the append-mode tests' fake Neo4j. Invented names only."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from memory_kernel.graph import forward, kg_writer
from memory_kernel.graph.g1_ids import content_uuid
from memory_kernel.pipeline import bonfire, chain
from memory_kernel.pipeline.chain import Minted
from unit.pipeline.test_write_append_mode import GROUP, Emb, Neo4j, bootstrap, edge, ent, ep, episodic


def minted(group: str, key: str) -> Minted:
    return Minted(content_uuid("ent", group, key), key, "entity", key, (key,), 0, 0)


def test_clone_is_refused_at_append_start():
    state = chain.empty()
    own = minted(GROUP, "kiln")
    forward.check_chain_group(replace(state, entities={own.uuid: own}), GROUP)  # the bonfire's own mint passes
    source = minted("g-source", "glaze")  # minted under the group the copy was made from
    with pytest.raises(forward.Refused, match=r"chain minted under another group \(a clone\): 1 of the chain's 2"):
        forward.check_chain_group(replace(state, entities={own.uuid: own, source.uuid: source}), GROUP)
    assert isinstance(forward.Refused("x"), bonfire.BonfireError)  # mk append's refusal (exit 2), never a traceback


def test_foreign_uuid_error_names_the_owning_group(monkeypatch):
    monkeypatch.setattr(kg_writer, "validate_rows", lambda *_a, **_k: [])
    db = Neo4j()
    db.foreign = {ent("kiln"): "g-source"}  # the kiln's uuid is held by the group this one was cloned from
    with pytest.raises(kg_writer.ForeignUuids) as held:
        asyncio.run(kg_writer.write_rows(db, bootstrap(), group_id=GROUP, embed=Emb()))
    assert held.value.rows == [(ent("kiln"), "g-source")] and isinstance(held.value, ValueError)
    assert "already held by group(s) ['g-source']" in str(held.value)

    db.foreign = {}
    db.seed({"nodes": [episodic("m1")], "edges": []})
    lone = {"nodes": [], "edges": [edge("MENTIONS", ep("m1"), ent("ana"))]}  # Ana is in neither the batch nor the group
    with pytest.raises(kg_writer.ForeignUuids) as missing:
        asyncio.run(kg_writer.write_rows(db, lone, group_id=GROUP, embed=Emb()))
    assert missing.value.rows == [(ent("ana"), None)]
    assert f"in neither the batch nor group {GROUP!r}" in str(missing.value)
    assert not [q for q in db.statements if "MERGE" in q]
