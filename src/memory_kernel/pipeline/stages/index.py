"""index, sanity and seal: persist the search-key index of the written group, the build's receipt (``sanity``, kernel
1.8: no search), and the seal before any unseal.

Ported from the reference harness:
  * ``index`` / ``sanity`` / ``seal``                                        <- the harness's build module
  * ``freeze_recipe`` / ``assert_recipe_tag`` / ``tree_digest`` / ``ledger_pin`` /
    ``build_seal`` / ``_persist`` / ``EXCLUDE_PARTS``                         <- the harness's index module
  * ``SEALED_STAGES`` (``_SEALED_STAGES``)                                   <- the harness's conversation module
  * ``SEAL_SCHEMA``                                                          <- the harness's freeze module

Rebindings replaced:
  * the index module's ``GROUP`` (rebound by the harness) -> ``cfg.group`` argument of ``_persist``;
  * ``stack_driver()`` -> ``graphdb.driver(cfg)`` (index: ``persist_keys`` SETs);
  * ``VoyageEmbedder(container_env(...))`` -> ``embed.embedder(cfg, "document")``;
  * ``_cap_check(index_projection_usd(spec))`` -> ``cap_check(ctx, write.index_projection_usd(ctx))``;
  * ``spec.messages`` -> ``ctx.counts.messages``;
  * the index module's ``FREEZE`` default (rebound by the harness) -> ``recipe_of(cfg)`` (``--recipe-manifest``'s
    ``final_recipe``, else ``config.default_recipe()``);
  * the index module's ``SEALED_DIRS`` (rebound to ``sealed_dirs(spec)``) -> ``SEALED_DIRS`` from ``SEALED_STAGES`` (a base
    conversation: the plain stage dir names);
  * ``ledger_pin`` / ``build_seal`` ``ledger=COST_LEDGER`` default -> explicit ``ledger=cfg.ledger``;
  * ``CostRecorder`` -> ``ctx.cost`` (ledger ``cfg.ledger``, tag ``conv``);
  * (kept, declared) ``kg_index.recipe_tag`` reads ``MK_TEMPORAL_DEIXIS``; mk never sets it and
    ``assert_recipe_tag`` refuses a mismatch before any write, as the harness did.
Append mode (``mk append``, a kernel 1.2 forward append of one session): ``index`` computes the keys of the session's
new Episodic rows only (``scope``: the write's ``forward.episodic``) and ``persist_keys`` embeds those whose text is
new. An older message's key cannot change: it is built from the message's own content, its fact edges' text, its
Events' names and the names of the entities it mentions, and a forward append never renames a stored entity
(``write.forward_gate`` rule 2) nor rewrites a stored fact's text; so the entities the append touched
(``forward.touched``) re-key no stored message, and the index reads nothing but the session's rows. Its gate holds the
group's keys stamped with the tag (every Episodic row's stamp, ``group_episodic``) to the chain's message total plus
the session's (``expected_messages``), and the recipe tag to the bonfire's (its bootstrap's index record; the
driver's preflight refuses a mismatch before any stage). Every build's ``sanity`` is a receipt (``stages.receipt``,
$0): a forward append's records its gates and decisions, a full build's the index's results (kernel 1.8, D1: the
taxonomy-question search left with the taxonomy layer).
Declared addition: ``index.json`` carries ``"embedder"`` (``embed.record``): kind, cache hits / misses and the
stand-in note, so a cached or stand-in index can never be mistaken for a live one.
Graph mode (kernel 2.0, ``write.graph_mode``): ``_dry_index`` replays the dry write's rows (``write.commit_rows``) and
runs ``_persist`` in ONE rolled-back transaction with a ``RecordingEmbed``, under the same gate; ``keys.json`` keeps
the keys and their vectors for ``pipeline.commit`` and ``index.json`` says ``"dry": true``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memory_kernel.pipeline import embed, graphdb
from memory_kernel.pipeline.config import default_recipe, load_forward
from memory_kernel.pipeline.runner import STAGE_DIRS
from memory_kernel.pipeline.stages._common import cap_check
from memory_kernel.pipeline.stages._g1_cards import write_once
from memory_kernel.pipeline.stages.write import Refused, _out, commit_rows, graph_mode, index_projection_usd

if TYPE_CHECKING:
    from memory_kernel.pipeline.config import BuildConfig
    from memory_kernel.pipeline.runner import Ctx

SEAL_SCHEMA = "mk-seal/v1"  # the sealed.json written by the seal step (the harness's SEAL_SCHEMA)
#: Stage dirs hashed by the seal: select through sanity (no cache, logs, seal or 12-*).
SEALED_STAGES: tuple[str, ...] = (
    "select",
    "boundary",
    "shard",
    "syntax",
    "capture",
    "runs",
    "graph_a",
    "events",
    "ctx",
    "referents",
    "build_dry",
    "write",
    "index",
    "sanity",
)
SEALED_DIRS: tuple[str, ...] = tuple(STAGE_DIRS[s] for s in SEALED_STAGES)
EXCLUDE_PARTS = frozenset({"budget", "__pycache__"})
STAGE_FILES = {"index": "index.json", "sanity": "sanity.json", "seal": "sealed.json"}
#: Graph mode: the dry index's ``{recipe_tag, key_render, keys, vectors}`` (keys: uuid -> key; vectors: key -> vector).
KEYS = "keys.json"


# -- pure helpers (the harness's index module) ---------------------------------------------------------------------


def _recipe_view(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kg_recipe_config": doc["kg_recipe_config"],
        "key_recipe_tag": doc["key_recipe_tag"],
        "context": doc["context_config"],
    }


def freeze_recipe(manifest: Path) -> dict[str, Any]:
    return _recipe_view(json.loads(Path(manifest).read_bytes())["final_recipe"])


def recipe_of(cfg: BuildConfig) -> dict[str, Any]:
    """``--recipe-manifest``'s ``final_recipe`` when given, else the recipe computed from the code constants."""
    if cfg.recipe_manifest is not None:
        return freeze_recipe(cfg.recipe_manifest)
    return _recipe_view(default_recipe())


def assert_recipe_tag(expected: str, key_render: str) -> str:
    from memory_kernel.search import kg_index

    got = kg_index.recipe_tag(key_render)
    if got != expected:
        raise ValueError(f"key recipe tag {got!r} differs from the freeze's {expected!r} (check the deixis env)")
    return got


def tree_digest(root: Path) -> dict:
    """{relative path: sha256} of every file under ``root`` (excluded parts skipped), plus a combined digest."""
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not (EXCLUDE_PARTS & set(path.relative_to(root).parts)):
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"files": len(files), "digest": combined, "manifest": files}


def ledger_pin(ledger: Path) -> dict:
    lines = [ln for ln in ledger.read_bytes().splitlines() if ln.strip()]
    return {"rows": len(lines), "last_row_sha256": hashlib.sha256(lines[-1]).hexdigest()}


def build_seal(
    proofs: Path,
    dirs: Sequence[str],
    eval_qa: Path,
    *,
    now: datetime | None = None,
    ledger: Path,
    genesis: bool = False,
) -> dict:
    """The sealed.json body. The eval-qa is only HASHED (bytes), never parsed. A genesis build (``genesis``) is marked
    ``"genesis": true``; any other body carries no such key (byte-identical to before)."""
    from memory_kernel.construction.memory_semantic_jev import seal

    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    present = {d: tree_digest(proofs / d) for d in dirs if (proofs / d).is_dir()}
    missing = [d for d in dirs if not (proofs / d).is_dir()]
    body = {
        "schema_version": SEAL_SCHEMA,
        "sealed_at": stamp,
        "stages": {d: {"files": v["files"], "digest": v["digest"]} for d, v in present.items()},
        "stages_missing": missing,
        "eval_qa_sha256": hashlib.sha256(eval_qa.read_bytes()).hexdigest(),
        "cost_ledger_pin": ledger_pin(ledger) if ledger.is_file() else None,
        "note": "stage digests are over file bytes; budget dirs excluded; the eval-qa is hashed, never parsed",
        **({"genesis": True} if genesis else {}),
    }
    return seal(body)


# -- Neo4j (the harness's index module) ------------------------------------------------------------------------------


async def scoped_keys(d, group: str, scope: list[str], render: str) -> dict[str, str]:
    """The keys of the ``scope`` messages as the graph holds them (``read_key_rows(uuids=)``; a compact render also
    reads the group's per-event rows, kept for the scope): what a forward append's index persists and the 2.0 commit
    checks again. ``Refused`` when a scope message is not in the group."""
    from memory_kernel.search import kg_index

    msgs, facts, mentions = await kg_index.read_key_rows(d, group, uuids=scope)
    if missing_rows := sorted(set(scope) - {m["uuid"] for m in msgs}):
        raise Refused(
            f"index: {len(missing_rows)} of the session's Episodic rows are not in the group: {missing_rows[:3]}"
        )
    compact = None
    if render == "compact":  # the per-event rows of the whole group, kept for the scope
        compact = {u: v for u, v in (await kg_index.read_compact_rows(d, group)).items() if u in set(scope)}
    return kg_index.keys_from_graph(msgs, facts, mentions, key_render=render, compact=compact)


async def _scoped_index(d, group: str, scope: list[str], render: str, emb) -> tuple[dict, dict, list, list]:
    """A forward append's keys: computed for the ``scope`` messages only (``scoped_keys``), persisted
    (``persist_keys`` embeds the new ones), then read back: ``(keys, persisted, stale, missing)``, the last two over
    the scope (a stored key that is not this run's key, or no key at all)."""
    from memory_kernel.graph.kg_writer import KEY_EMBED_DIM, KEY_EMBED_MODEL
    from memory_kernel.search import kg_index

    keys = await scoped_keys(d, group, scope, render)
    persisted = await kg_index.persist_keys(d, group, keys, emb, key_render=render)
    tag = kg_index.model_tag(KEY_EMBED_MODEL, KEY_EMBED_DIM)
    async with d.session() as s:
        stored = {
            r["uuid"]: r
            async for r in await s.run(
                "MATCH (m:Episodic {group_id:$g}) WHERE m.uuid IN $ids RETURN m.uuid AS uuid, m.search_key AS key, "
                "m.search_key_sha AS sha, m.search_key_model AS model",
                g=group,
                ids=sorted(scope),
            )
        }
    missing = sorted(u for u in scope if not (stored.get(u) or {}).get("key"))
    stale = sorted(
        u
        for u in scope
        if u not in missing and (stored[u]["sha"] != kg_index.key_sha(keys[u]) or stored[u]["model"] != tag)
    )
    return keys, persisted, stale, missing


async def _persist(cfg: BuildConfig, scope: list[str] | None = None, *, driver: Any = None, emb: Any = None) -> dict:
    """The index of the group: every key (a full build), or with ``scope`` only those messages' (a forward append,
    ``_scoped_index``). ``n_messages`` is the group's Episodic total either way. ``driver`` / ``emb`` (graph mode: a
    ``tx.TxScope`` and a ``RecordingEmbed``) replace ``graphdb.driver(cfg)`` (an injected driver is not closed) and
    ``embed.embedder(cfg, "document")``."""
    from memory_kernel.graph.kg_writer import KEY_EMBED_DIM, KEY_EMBED_MODEL
    from memory_kernel.search import kg_index

    group = cfg.group
    recipe = recipe_of(cfg)
    render = recipe["kg_recipe_config"]["key_render"]
    tag = assert_recipe_tag(recipe["key_recipe_tag"], render)
    d = driver if driver is not None else graphdb.driver(cfg)
    try:
        if emb is None:
            emb = embed.embedder(cfg, "document")
        if scope is not None:
            keys, persisted, stale, missing = await _scoped_index(d, group, scope, render, emb)
        else:
            msgs, facts, mentions = await kg_index.read_key_rows(d, group)
            keys = kg_index.keys_from_graph(msgs, facts, mentions, key_render=render)
            persisted = await kg_index.persist_keys(d, group, keys, emb, key_render=render)
            idx = await kg_index.load_index(d, group, model=KEY_EMBED_MODEL, dim=KEY_EMBED_DIM, key_render=render)
            if idx.stale or idx.missing:
                raise kg_index.IndexStale(idx.stale, idx.missing)
            keys, stale, missing = idx.keys, [], []
        async with d.session() as s:  # count-only: one row per (recipe, model), whatever the group's size
            rows = [
                r
                async for r in await s.run(
                    "MATCH (m:Episodic {group_id:$g}) RETURN m.search_key_recipe AS recipe, "
                    "m.search_key_model AS model, count(*) AS n",
                    g=group,
                )
            ]
    finally:
        if driver is None:
            await d.close()
    stamped: Counter[Any] = Counter()
    models: Counter[Any] = Counter()
    for r in rows:
        stamped[r["recipe"]] += int(r["n"])
        models[r["model"]] += int(r["n"])
    scoped = {} if scope is None else {"scope": len(scope), "n_keys": len(keys)}
    return {
        "group": group,
        "key_render": render,
        "recipe_tag": tag,
        "freeze_recipe_tag": recipe["key_recipe_tag"],
        "tag_equals_freeze": tag == recipe["key_recipe_tag"],
        "persist": persisted,
        "voyage_tokens": emb.tokens,
        "voyage_requests": emb.requests,
        "embedder": embed.record(emb),
        "n_messages": sum(stamped.values()) if scope is not None else len(keys),
        "keys_stamped_with_tag": stamped.get(tag, 0),
        "stamps": dict(stamped),
        "models": dict(models),
        "stale": len(stale),
        "missing": len(missing),
        "load_index": (
            "scoped: the session's keys read back"
            if scope is not None
            else "returned without IndexModelMismatch; stale/missing empty"
        ),
        **scoped,
        "key_sha256": {u: kg_index.key_sha(t) for u, t in sorted(keys.items())},
        "keys": keys,
    }


async def _dry_index(ctx: Ctx, scope: list[str]) -> dict:
    """Graph mode's index: in ONE transaction that is rolled back, the dry write's rows replayed with their recorded
    vectors (``write.commit_rows``, ``ReplayEmbed``), then ``_persist`` of the session's keys over that transaction
    with the live embedder recorded (``RecordingEmbed``). Writes ``KEYS`` (the keys and the vectors of their texts, for
    ``pipeline.commit``) and returns ``_persist``'s result with ``"dry": true``."""
    from memory_kernel.graph.kg_writer import write_rows
    from memory_kernel.graph.tx import transaction

    cfg = ctx.cfg
    rows, vectors = commit_rows(ctx)
    emb = embed.RecordingEmbed(embed.embedder(cfg, "document"))
    d = graphdb.driver(cfg)
    try:
        async with transaction(d, commit=False) as tx:
            await write_rows(tx, rows, group_id=cfg.group, embed=embed.ReplayEmbed(vectors), append=True)
            result = await _persist(cfg, scope, driver=tx, emb=emb)
    finally:
        await d.close()
    doc = {"recipe_tag": result["recipe_tag"], "key_render": result["key_render"], "keys": result["keys"]}
    write_once(ctx.path("index") / KEYS, {**doc, "vectors": emb.vectors})
    return {**result, "dry": True}


# -- the stages ---------------------------------------------------------------------------------------


def _voyage_cost(cost, result: Mapping) -> None:
    tokens = int(result["voyage_tokens"])
    cost.add("voyage", requests=int(result["voyage_requests"]), input_tokens=tokens, usd=embed.voyage_usd(tokens))
    cost.note += f"; voyage usd = tokens x ${embed.VOYAGE_USD_PER_M}/M (ASSUMED list rate)"


def forward_scope(ctx: Ctx) -> tuple[list[str] | None, int]:
    """``(scope, expected messages)``: a forward append's new Episodic uuids (the write's ``forward.episodic``) and
    the group's message total after it (the chain's plus the session's); a full build's ``(None, its messages)``."""
    forward = load_forward(ctx.cfg)
    if forward is None:
        return None, ctx.counts.messages
    summary = ctx.path("write") / "summary.json"
    try:
        scope = json.loads(summary.read_bytes())["append"]["forward"]["episodic"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refused(f"index: {summary} carries no forward result ({type(exc).__name__})") from None
    return list(scope), int(forward["totals"]["messages"]) + ctx.counts.messages


def index(ctx: Ctx, live: bool) -> dict:  # noqa: ARG001 (single live call, as the harness)
    """The harness index module's persist step (``_persist``) with the ``== spec.messages`` gate (a forward append:
    the session's keys, ``forward_scope``, and the group's total)."""
    cfg = ctx.cfg
    target = _out(ctx, "index", STAGE_FILES["index"])
    cap = cap_check(ctx, index_projection_usd(ctx))
    scope, expected = forward_scope(ctx)
    with ctx.cost("index", run=target.parent.name, note=f"C2 persist_keys {cfg.sample_id}") as cost:
        if graph_mode(cfg):
            if scope is None:
                raise Refused("index: graph mode is an append's, and the run has no forward input")
            result = asyncio.run(_dry_index(ctx, scope))
        else:
            result = asyncio.run(_persist(cfg))  # a full build (scope None); an append never gets here
        _voyage_cost(cost, result)
    result["programme_cap"] = cap
    result["expected_messages"] = expected
    result["passed"] = bool(
        result["tag_equals_freeze"]
        and result["keys_stamped_with_tag"] == result["n_messages"] == expected
        and not result["stale"]
        and not result["missing"]
    )
    if cfg.write_mode == "append":  # only an append-mode result carries these keys: a fresh index.json is unchanged
        result.update(append_index_gate(cfg, result, expected))
        result["passed"] = bool(result["passed"] and result["append_gate"]["passed"])
    write_once(target, result)
    return {k: v for k, v in result.items() if k not in ("key_sha256", "keys")}


def append_index_gate(cfg: BuildConfig, result: Mapping[str, Any], messages: int) -> dict[str, Any]:
    """The append-mode index gate: the keys stamped with the tag equal the group's Episodic total (every Episodic
    row's stamp, ``stamps``), which equals ``messages`` (the chain's total plus the session's: the write's rule 3), and
    the tag is the bonfire's (its bootstrap's index record) when ``cfg.bonfire_index_record`` is set."""
    group_episodic = sum(result["stamps"].values())
    bonfire_tag = None
    if cfg.bonfire_index_record is not None:
        try:
            bonfire_tag = json.loads(cfg.bonfire_index_record.read_bytes()).get("recipe_tag")
        except (OSError, ValueError, AttributeError) as exc:
            raise Refused(
                f"index: the bonfire's index record {cfg.bonfire_index_record} does not read ({type(exc).__name__}): "
                "the bootstrap's index stage must have run"
            ) from None
        if not bonfire_tag:
            raise Refused(f"index: the bonfire's index record {cfg.bonfire_index_record} has no recipe_tag")
    gate = {
        "group_episodic": group_episodic,
        "keys_equal_group": result["keys_stamped_with_tag"] == group_episodic == messages,
        "bonfire_recipe_tag": bonfire_tag,
        "tag_equals_bonfire": bonfire_tag is None or bonfire_tag == result["recipe_tag"],
    }
    gate["passed"] = gate["keys_equal_group"] and gate["tag_equals_bonfire"]
    return {"write_mode": "append", "group_episodic": group_episodic, "append_gate": gate}


def sanity(ctx: Ctx, live: bool) -> dict:  # noqa: ARG001
    """The build's receipt, $0 (``stages.receipt``). A forward append: the gates' results, the resolution and the
    referents decisions. A full build: the index's results (kernel 1.8, D1: no search, no Voyage)."""
    from memory_kernel.pipeline.stages import receipt

    if load_forward(ctx.cfg) is not None:
        return receipt.append_receipt(ctx)
    return receipt.full_receipt(ctx)


def seal(ctx: Ctx, live: bool) -> dict:  # noqa: ARG001
    """Sibling of the harness index module's seal step: the conversation root, ``SEALED_DIRS`` and its eval-qa; $0."""
    cfg = ctx.cfg
    target = _out(ctx, "seal", STAGE_FILES["seal"])
    eval_qa = ctx.path("select") / f"{cfg.stem}-eval-qa.json"
    note = f"C2 seal {cfg.sample_id} before unseal, $0"
    with ctx.cost("index", run=target.parent.name, note=note):
        body = build_seal(ctx.root, SEALED_DIRS, eval_qa, ledger=cfg.ledger, genesis=cfg.genesis)
        if body["stages_missing"]:
            raise Refused(f"sealed stage dirs missing: {body['stages_missing']}")
        write_once(target, body)
    return {
        "stages": {k: v["files"] for k, v in body["stages"].items()},
        "eval_qa_sha256": body["eval_qa_sha256"],
        "sealed_at": body["sealed_at"],
        "cost_ledger_pin": body["cost_ledger_pin"],
    }
