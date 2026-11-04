"""Index and sanity of a forward append (kernel 1.2): the session's keys only, gated on the group's total.

All offline. The stage's gate runs on a faked ``_persist`` result; ``_persist`` itself runs on a fake async driver that
answers the reader and ``persist_keys`` statements over an in-memory group (and logs them).
Invented names only.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from types import SimpleNamespace

import pytest

from memory_kernel.pipeline.stages import index as stage

TAG = "legacy|deixis=on|v4"
GROUP = "g-ember"


def persisted(*, keys: int, stamps: dict) -> dict:
    """A ``_persist`` result: ``keys`` keys in the index, the group's Episodic rows stamped as ``stamps``."""
    return {
        "group": GROUP,
        "recipe_tag": TAG,
        "tag_equals_freeze": True,
        "n_messages": keys,
        "keys_stamped_with_tag": stamps.get(TAG, 0),
        "stamps": stamps,
        "stale": 0,
        "missing": 0,
        "voyage_tokens": 0,
        "voyage_requests": 0,
        "key_sha256": {},
        "keys": {},
    }


def run_index(monkeypatch, tmp_path, result: dict, *, messages: int, mode: str, record=None, totals=None) -> dict:
    """The index stage of a run with ``messages`` (the session's), a forward input with the chain's ``totals`` (None:
    a full build) and a write summary naming the session's Episodic rows."""
    seen: dict = {}

    async def fake_persist(_cfg, scope=None):
        seen["scope"] = scope
        return dict(result)

    monkeypatch.setattr(stage, "_persist", fake_persist)
    monkeypatch.setattr(stage, "cap_check", lambda _ctx, _usd: {})
    cost = SimpleNamespace(add=lambda *_a, **_k: None, note="")
    pinned = None
    if totals is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        pinned = tmp_path / "forward.json"
        pinned.write_text(json.dumps({"totals": {"messages": totals, "episodes": 1}}))
        (tmp_path / "write").mkdir()
        forward = {"episodic": [f"ep-new-{i}" for i in range(messages)]}
        (tmp_path / "write" / "summary.json").write_text(json.dumps({"append": {"forward": forward}}))
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(sample_id="conv-ember", write_mode=mode, bonfire_index_record=record, forward_input=pinned),
        counts=SimpleNamespace(messages=messages),
        path=lambda key: tmp_path / key,
        cost=lambda *_a, **_k: contextlib.nullcontext(cost),
    )
    stage.index(ctx, True)
    out = json.loads((tmp_path / "index" / "index.json").read_bytes())
    return out | {"_scope": seen["scope"]}


def test_an_append_index_without_the_writes_forward_result_is_refused(monkeypatch, tmp_path):
    (tmp_path / "write").mkdir(parents=True)
    (tmp_path / "write" / "summary.json").write_text(json.dumps({"append": {}}))
    (tmp_path / "forward.json").write_text(json.dumps({"totals": {"messages": 3, "episodes": 1}}))
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(forward_input=tmp_path / "forward.json"),
        counts=SimpleNamespace(messages=2),
        path=lambda key: tmp_path / key,
    )
    with pytest.raises(stage.Refused, match="carries no forward result"):
        stage.forward_scope(ctx)


def test_fresh_index_output_is_unchanged(monkeypatch, tmp_path):
    out = run_index(monkeypatch, tmp_path, persisted(keys=6, stamps={TAG: 6}), messages=6, mode="fresh")
    assert out["passed"] and out["_scope"] is None and out["expected_messages"] == 6
    assert not {"write_mode", "group_episodic", "append_gate"} & set(out)


# -- _persist over a fake group: the session's keys only -----------------------------------------------------------------


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        async def gen():
            for r in self.rows:
                yield r

        return gen()


class Group:
    """Episodic rows (uuid -> props) and MENTIONS (message uuid -> entity names) of one group; ``statements`` logs
    what was asked. Answers the reader's and ``persist_keys``' statements."""

    def __init__(self, messages: dict, mentions: dict) -> None:
        self.messages, self.mentions, self.statements = messages, mentions, []

    def session(self, **_kw):
        group = self

        class S:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def run(self, query, **params):
                return _Result(group.run(query, params))

        return S()

    async def close(self):
        return None

    def run(self, q: str, p: dict) -> list[dict]:
        from memory_kernel.search import kg_index

        self.statements.append(q)
        ids = set(p.get("ids") or self.messages)
        if q == kg_index.SCOPED_MESSAGES_CYPHER:
            return [{"uuid": u, **m} for u, m in self.messages.items() if u in ids]
        if q == kg_index.SCOPED_FACTS_CYPHER:
            return []
        if q == kg_index.SCOPED_MENTIONS_CYPHER:
            return [
                {"m": u, "uuid": f"ent-{n}", "name": n, "labels": ["Entity"]}
                for u, names in self.mentions.items()
                if u in ids
                for n in names
            ]
        if q.startswith("UNWIND $rows AS r MATCH (m:Episodic"):
            for r in p["rows"]:
                self.messages[r["uuid"]] |= {
                    "search_key": r["key"],
                    "search_key_sha": r["sha"],
                    "search_key_embedding": r["emb"],
                    "search_key_model": p["tag"],
                    "search_key_recipe": p["recipe"],
                }
            return [{"n": len(p["rows"])}]
        if "SET m.search_key_recipe = $recipe" in q:
            return [{"n": 0}]
        if "count(*) AS n" in q:  # the stamp read: count-only, one row per (recipe, model)
            from collections import Counter

            got = Counter((m.get("search_key_recipe"), m.get("search_key_model")) for m in self.messages.values())
            return [{"recipe": r, "model": mo, "n": n} for (r, mo), n in got.items()]
        if q.startswith(("MATCH (m:Episodic {group_id: $g})", "MATCH (m:Episodic {group_id:$g})")):
            scoped = "$ids" in q  # else every message of the group (the stamp reads)
            return [
                {alias: (u if prop == "uuid" else m.get(prop)) for prop, alias in re.findall(r"m\.(\w+) AS (\w+)", q)}
                for u, m in self.messages.items()
                if not scoped or u in ids
            ]
        raise AssertionError(f"unexpected statement: {q}")


def test_persist_keys_only_the_sessions_messages_and_never_rekeys_an_old_one(monkeypatch):
    """The append touched the kiln (a new mention): the old message that names it keeps its key, unread; only the new
    message is keyed and embedded. The old key is what a fresh key of the old message gives (names are frozen)."""
    from memory_kernel.graph.kg_writer import KEY_EMBED_DIM, KEY_EMBED_MODEL
    from memory_kernel.pipeline import embed, graphdb
    from memory_kernel.search import kg_index

    monkeypatch.delenv("MK_TEMPORAL_DEIXIS", raising=False)
    tag = kg_index.model_tag(KEY_EMBED_MODEL, KEY_EMBED_DIM)
    old = {"message_id": "m1", "content": "Ana: the kiln is new", "told_at": "2026-10-01T09:00:00+00:00"}
    new = {"message_id": "m4", "content": "Ana: the kiln is hot", "told_at": "2026-10-02T09:00:00+00:00"}
    mentions = {"ep-old": ["kiln"], "ep-new": ["kiln"]}
    old_key = kg_index.keys_from_graph([{"uuid": "ep-old", **old}], [], [
        {"m": "ep-old", "uuid": "ent-kiln", "name": "kiln", "labels": ["Entity"]}
    ], key_render="legacy")["ep-old"]  # fmt: skip
    stored = {"search_key": old_key, "search_key_sha": kg_index.key_sha(old_key), "search_key_model": tag}
    group = Group({"ep-old": old | stored | {"search_key_recipe": TAG}, "ep-new": dict(new)}, mentions)
    embedded: list[str] = []

    class Emb:
        tokens = requests = 0

        def __call__(self, texts):
            embedded.extend(texts)
            return [[0.0] * KEY_EMBED_DIM for _ in texts]

    monkeypatch.setattr(graphdb, "driver", lambda _cfg: group)
    monkeypatch.setattr(embed, "embedder", lambda _cfg, _kind: Emb())
    monkeypatch.setattr(embed, "record", lambda _e: {})
    cfg = SimpleNamespace(group=GROUP, recipe_manifest=None)
    out = asyncio.run(stage._persist(cfg, ["ep-new"]))

    assert out["persist"] == {"keys": 1, "embedded": 1, "written": 1, "stamped": 0} and out["scope"] == 1
    assert len(embedded) == 1 and "the kiln is hot" in embedded[0]
    assert (out["n_messages"], out["keys_stamped_with_tag"], out["stale"], out["missing"]) == (2, 2, 0, 0)
    assert group.messages["ep-old"]["search_key"] == old_key  # untouched, never re-keyed
    assert set(out["keys"]) == {"ep-new"}
    # every read is the session's rows, or a count over the group (its stamps)
    assert not [q for q in group.statements if "$ids" not in q and "count(*)" not in q and "UNWIND" not in q]
    assert [q for q in group.statements if "m.content" in q] == [kg_index.SCOPED_MESSAGES_CYPHER]
    assert kg_index.MENTIONS_CYPHER not in group.statements and kg_index.FACTS_CYPHER not in group.statements

    with pytest.raises(stage.Refused, match="of the session's Episodic rows are not in the group"):
        asyncio.run(stage._persist(cfg, ["ep-gone"]))


# -- sanity ----------------------------------------------------------------------------------------------------------


def test_an_appends_sanity_is_the_receipt_and_embeds_nothing(monkeypatch, tmp_path):
    from memory_kernel.pipeline import embed
    from memory_kernel.pipeline.stages import receipt

    monkeypatch.setattr(embed, "embedder", lambda *_a, **_k: pytest.fail("an append's sanity builds no embedder"))
    pinned = tmp_path / "forward.json"
    pinned.write_text(json.dumps({"totals": {"messages": 4, "episodes": 1}}))
    fwd = {"run": 2, "minted": [], "routes": {}, "same_as": [], "conflicts": [], "touched": [], "episodic": ["ep-1"]}
    for key, name, body in (("write", "summary.json", {"append": {"passed": True, "forward": fwd}}),
                            ("index", "index.json", {"passed": True, "append_gate": {"passed": True}}),
                            ("referents", "decisions.json", {"mode": "append", "decisions": {}, "hard": [],
                                                             "counts": {}})):  # fmt: skip
        (tmp_path / key).mkdir()
        (tmp_path / key / name).write_text(json.dumps(body))
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(sample_id="conv-ember", group=GROUP, write_mode="append", forward_input=pinned),
        counts=SimpleNamespace(messages=1),
        path=lambda key: tmp_path / key,
        cost=lambda *_a, **_k: contextlib.nullcontext(SimpleNamespace(add=lambda *_a, **_k: None, note="")),
    )
    stage.sanity(ctx, True)
    out = json.loads((tmp_path / "sanity" / "sanity.json").read_bytes())
    assert out["schema"] == receipt.RECEIPT_SCHEMA and out["voyage_requests"] == 0


# -- the recipe tag: the bootstrap's index record -----------------------------------------------------------------------


def test_the_preflight_refuses_an_append_whose_recipe_tag_differs(tmp_path, monkeypatch):
    from memory_kernel.knowledge.temporal_deixis import TEMPORAL_DEIXIS_ENV
    from memory_kernel.pipeline import runner
    from memory_kernel.pipeline.config import BuildConfig, default_recipe

    monkeypatch.delenv(TEMPORAL_DEIXIS_ENV, raising=False)
    record = tmp_path / "11-index-001" / "index.json"

    def cfg():
        return BuildConfig(
            sample_id="conv-ember",
            conversation=tmp_path / "c.json",
            group=GROUP,
            purpose="remember the kiln",
            workdir=tmp_path / "w",
            root=tmp_path,
            write_mode="append",
            bonfire_index_record=record,
        )

    missing = runner._recipe_problems(cfg())
    assert len(missing) == 1 and "bonfire index record" in missing[0]

    record.parent.mkdir()
    record.write_text(json.dumps({"recipe_tag": default_recipe()["key_recipe_tag"]}))
    assert runner._recipe_problems(cfg()) == []  # the bootstrap's tag: accepted

    record.write_text(json.dumps({"recipe_tag": "legacy|deixis=off|v4"}))  # the bootstrap ran with the flag off
    (problem,) = runner._recipe_problems(cfg())
    assert "differs from the bonfire's 'legacy|deixis=off|v4'" in problem and TEMPORAL_DEIXIS_ENV in problem


def test_a_fresh_build_has_no_bonfire_recipe_check(tmp_path, monkeypatch):
    from memory_kernel.pipeline import runner
    from memory_kernel.pipeline.config import BuildConfig

    monkeypatch.delenv("MK_TEMPORAL_DEIXIS", raising=False)
    cfg = BuildConfig(
        sample_id="conv-ember",
        conversation=tmp_path / "c.json",
        group=GROUP,
        purpose="remember the kiln",
        workdir=tmp_path / "w",
        root=tmp_path,
    )
    assert cfg.bonfire_index_record is None and runner._recipe_problems(cfg) == []
