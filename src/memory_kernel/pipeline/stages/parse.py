"""capture: syntax + hybrid parse capture + compat runs on the build's private FCG engine.

Ported from the reference harness:
  * ``capture`` / ``conserve`` / ``_complete`` / ``_compat_valid`` / ``_require_complete``  <- the harness's capture module

The frozen bodies are in ``_parse_capture`` (syntax, prepare, run, gated merge, status), ``_parse_hybrid_pairs``
(``compat_run``), ``_fenced_http`` / ``_capture_record`` / ``_unicode_repair`` /
``_resident_stock_capture_runner`` (the transport and the request builders) and ``engine.Engine`` (the engine
lifecycle). Rebindings replaced:
  * the harness's lane (process-global rebinds of its spare-engine and parse-capture modules) -> one
    engine per build, an explicit ``Site`` (root, ``EngineConfig``, parser lock path) passed to every function;
  * ``ROOT`` / ``BINARY`` / ``IMAGE`` / ``LIVE_PARSER`` -> ``Site``;
  * the capture code pins name this package's files and are hashed in the IMPORTED
    package (``provenance.code_path``); ``cfg.root`` stays the base of the data paths (``syntax_from``);
  * the snapshot-pin list is computed from the package (``snapshot_pins_for``) unless ``--snapshot-pins`` names
    one.

A finished stage is detected through ``_parse_capture``'s own readers and skipped, so a capture that is already
complete and merged launches NO engine and writes NO file.

With ``cfg.capture_cache`` (``mk append``: the bonfire's ``capture-cache``), cached job results are written into the
capture before the shards run and the engine's new results are filed after (``_capture_cache``); unset, nothing changes.

With ``cfg.capture_slot`` (the warm capture slot's socket, ``memory_kernel.pipeline.slot``), a capture that needs an
engine attempt first takes the slot's lease and holds it through the syntax step, the run and the merge. A warm grant
runs one shard on the slot's engine and parser (``Site.parser`` None, ``grant=`` to the attempts); a cold grant (a
recorded shard count above 1, or a slot that vacated) runs today's path on the ports the slot freed; no slot answering
runs today's path without a lease and says so in the result. Unset, nothing changes.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from memory_kernel.pipeline import provenance, slot
from memory_kernel.pipeline.config import engine_paths, engine_pins, live_parser_url
from memory_kernel.pipeline.engine import (
    EngineConfig,
    ParserConfig,
    ParserService,
    parser_teardown_problems,
    require_local,
)
from memory_kernel.pipeline.stages import _capture_cache as cc
from memory_kernel.pipeline.stages import _capture_shards as cs
from memory_kernel.pipeline.stages import _parse_capture as pc
from memory_kernel.pipeline.stages._parse_hybrid_pairs import COMPAT_SCHEMA, compat_run

if TYPE_CHECKING:
    from memory_kernel.pipeline.runner import Ctx

#: The single shard "1 of 1": parse_capture shards are 0-based (in_shard: order % k == i), so (1, 1) would select nothing.
SHARD = (0, 1)


def snapshot_pins_for(out_dir: Path, *, code_root: Path | None = None) -> Path:
    """The capture snapshot-pin list computed from the package: ``tree_census`` of the
    imported package (``code_root`` in tests and the gate's negative controls), written content-addressed
    (``snapshot-pins-<sha12>.json``) under ``out_dir``. ``prepare`` seals it; ``check_pins`` then VOIDs the capture on
    any edit or delete in the package."""
    doc = {
        "schema": pc.SNAPSHOT_SCHEMA,
        "basis": f"mk tree census at capture prepare, kernel head {provenance.code_head(code_root)}",
        "files": provenance.tree_census(code_root),
    }
    return provenance.write_content_addressed(out_dir, "snapshot-pins", doc)


def site_for(ctx: Ctx) -> pc.Site:
    """The capture's site. Starts nothing: with no ``--parser-url`` it only DESCRIBES the managed parser
    (``Site.parser``), which ``_parse_capture`` starts inside the syntax and run attempts, never for a complete capture."""
    cfg = ctx.cfg
    parser_url = live_parser_url(cfg)
    require_local(parser_url, allow_remote=cfg.allow_remote_parser)
    binary, image = engine_paths(cfg)  # EngineNotSetUp (a ValueError) when neither a flag nor `mk setup` names them
    binary_sha256, image_sha256 = engine_pins(cfg)
    engine = EngineConfig(
        port=cfg.engine_port,
        binary=binary,
        image=image,
        parser_url=parser_url,
        binary_sha256=binary_sha256,
        image_sha256=image_sha256,
    )
    managed = None if cfg.parser_url is not None else ParserConfig(port=cfg.parser_port, data_dir=cfg.data_dir)
    return pc.Site(root=cfg.root, engine=engine, parser_lock=cfg.workdir / ".mk" / "parser.lock", parser=managed)


def _complete(capture: Path) -> dict | None:
    return pc.latest_reports(capture).get(pc.shard_tag(SHARD))


def _compat_valid(runs: Path) -> bool:
    manifest = runs / "manifest.json"
    if not manifest.exists():
        return False
    return json.loads(manifest.read_bytes()).get("schema_version") == COMPAT_SCHEMA


def conserve(ctx: Ctx, capture: Path, sources: list[dict]) -> dict:
    """Conservation gate: merged message count == the conversation's messages and every source has ``complete.json``."""
    state = pc.status(capture)
    if state["complete_messages"] != ctx.counts.messages:  # the message count (message sources)
        raise ValueError(f"{ctx.cfg.sample_id}: merged messages {state['complete_messages']} != {ctx.counts.messages}")
    missing = [s["order"] for s in sources if not (capture / f"{s['order']:04d}" / "complete.json").exists()]
    if missing:
        raise ValueError(f"{ctx.cfg.sample_id}: sources without complete.json: {missing[:10]}")
    return state


def _require_complete(stage: str, report: dict | None, directory: Path, prefix: str) -> None:
    status = (report or {}).get("status")
    if status != "complete":
        attempt = (report or {}).get("attempt")
        raise RuntimeError(f"{stage}: status {status!r}; report {directory / f'{prefix}-a{attempt}.json'}")


@contextlib.contextmanager
def _managed_parser(site: pc.Site, out: Path) -> Iterator[None]:
    """The ONE managed parser the parallel shards share (``site.parser`` None = an external parser: nothing started).
    Its teardown receipt is checked like ``_parse_capture._with_parser`` does: a dirty one is an error."""
    if site.parser is None:
        yield
        return
    service = ParserService(site.parser, out / "parser-shared")
    if service.url != site.live_parser:
        raise RuntimeError(f"the managed parser {service.url} is not the site's live parser {site.live_parser}")
    try:
        service.launch()
        yield
    finally:
        try:
            teardown = service.teardown()
        except BaseException as exc:  # noqa: BLE001 -- a failed teardown is reported as a dirty one
            teardown = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        problems = parser_teardown_problems(teardown)
        if problems:
            raise RuntimeError(f"shared parser teardown dirty: {problems}")


def _needs_attempt(syn: Path, out: Path, sources: list[dict]) -> bool:
    """Whether this call runs an engine attempt: syntax parts missing, or a run neither merged nor complete. Reads only:
    a capture that needs no attempt takes no slot lease, as it launches nothing."""
    if pc.syntax_missing(syn, sources):
        return True
    if (out / "merge.json").exists():
        return False
    recorded = cs.recorded_shard_count(out)
    return recorded is None or bool(cs.pending_shards(out, recorded))


def _lease_request(site: pc.Site, want: Literal["warm", "cold"], data_dir: Path | None) -> slot.LeaseRequest:
    """What the capture will seal, computed as ``pc.manifest_body`` computes it, plus what a cold attempt would launch
    and arm with (the engine port, the parser's data dir, ``slot.ARM_ENV``): the slot grants warm only on equal pins. A
    ``data_dir`` unset or missing is sent as "", which no slot boots from."""
    return slot.LeaseRequest(
        binary_sha256=pc.sha_file(site.engine.binary),
        image_sha256=pc.sha_file(site.engine.image),
        binary_path=str(site.engine.binary.resolve(strict=True)),
        image_path=str(site.engine.image.resolve(strict=True)),
        engine_port=site.engine.port,
        live_parser=site.live_parser,
        parser_data_dir=str(data_dir.resolve(strict=True)) if data_dir is not None and data_dir.exists() else "",
        arm_env=slot.arm_env(),
        code_pins={p: pc.sha(pc.code_path(p, code_root=site.code_root).read_bytes()) for p in site.pin_files},
        walker_sha256=pc.current_walker_sha256(),
        want=want,
    )


def _attempts(
    ctx: Ctx,
    site: pc.Site,
    sources_path: Path,
    sources: list[dict],
    *,
    grant: slot.Grant | None = None,
    ended: dict | None = None,
) -> tuple[dict | None, int, int | None, dict]:
    """syntax -> prepare -> run (one engine, or the shards) -> cache store -> gated merge; ``(report, k, recorded,
    cached)``. With a warm ``grant`` the attempts run on the slot's engine and parser, one shard (k = 1). ``ended``
    (under a lease) collects the first non-complete attempt status and the run attempt's job count."""
    cfg = ctx.cfg
    syn, out = ctx.path("syntax"), ctx.path("capture")
    leased = {} if grant is None else {"grant": grant}

    def seen(report: dict | None) -> dict | None:
        if ended is not None and ended["status"] is None and (report or {}).get("status") != "complete":
            ended["status"] = str((report or {}).get("status"))
        return report

    if pc.syntax_missing(syn, sources):
        _require_complete("syntax", seen(pc.syntax(syn, sources_path, site=site, **leased)), syn, "report-syntax")
        if pc.syntax_missing(syn, sources):
            raise RuntimeError(f"syntax: parts still missing after a complete report ({syn})")
    if not (out / "manifest.json").exists():
        pins = cfg.snapshot_pins or snapshot_pins_for(cfg.conv_root / "logs/mk")
        pc.prepare(out, sources_path, syn, live_parser=site.live_parser, site=site, snapshot_pins=pins)
    report = _complete(out)
    merged = (out / "merge.json").exists()
    recorded = cs.recorded_shard_count(out)  # a capture's shard count is fixed by its existing reports
    k = recorded or cs.effective_shards(cfg.capture_shards, cs.usable_cpus())
    if grant is not None:
        k = 1  # one engine in the slot: a warm capture runs one shard, whatever capture_shards says
        cs.file_logger(out / "slot.log")(f"warm lease: k = 1 on the slot's engine ({cfg.capture_shards} requested)")
    cache = cfg.capture_cache
    cached: dict = {}
    if k == 1:
        if not merged and not (report and report.get("status") == "complete"):
            cs.check_shard_count(out, 1)
            if cache is not None:
                cached |= cc.prefill(out, site, cache)
            report = seen(pc.run(out, SHARD, site=site, **leased))
            if ended is not None:
                ended["jobs"] += int((report or {}).get("jobs") or 0)
            _require_complete("run", report, out, f"report-{pc.shard_tag(SHARD)}")
    elif not merged:
        cs.check_shard_count(out, k)
        cs.shard_ports(site.engine.port, k)
        if cs.pending_shards(out, k):  # every shard complete = only the merge is left: start no parser
            if cache is not None:
                cached |= cc.prefill(out, site, cache)
            shared = pc.Site(root=site.root, engine=site.engine, parser_lock=site.parser_lock, parser=None)
            with cs.sigterm_as_exception(), _managed_parser(site, out):
                reports = cs.run_parallel(out, shared, k, log=cs.file_logger(out / "shards.log"))
            bad = [r for r in reports if r.get("status") != "complete"]
            if bad:
                seen(bad[0])
                raise RuntimeError(f"capture shards not complete: {bad}")
    if cache is not None and not merged:  # every shard is complete here: file what the engine produced
        try:
            cached |= cc.store(out, cache)
        except OSError as exc:  # disk full, permissions: the cache is an optimisation, the capture goes on
            cached["store_error"] = f"{type(exc).__name__}: {exc}"[:500]
            cs.file_logger(out / "capture-cache.log")(f"capture cache store failed: {cached['store_error']}")
    if not (out / "merge.json").exists():
        pc.gated_merge(out, site)
    return report, k, recorded, cached


def _on_lease(ctx: Ctx, site: pc.Site, sources_path: Path, sources: list[dict], held: slot.Lease):
    """The attempts under a held lease, released in a ``finally`` with how they ended (``complete`` only when every
    attempt of this call completed and nothing raised) and the run attempt's job count. Warm: the slot's engine and
    parser (``Site.parser`` None). Cold: today's path, on the ports the slot freed."""
    grant = held.grant
    ended: dict = {"status": None, "jobs": 0}
    status = "crashed: unknown"
    try:
        if grant.kind == "warm":
            slot_parser = grant.parser["url"]  # a warm grant always names the slot's parser
            if slot_parser != site.live_parser:  # the cold path refuses a mismatched managed parser too (_with_parser)
                ended["status"] = "stopped: parser url differs"
                raise RuntimeError(
                    f"the slot's parser {slot_parser} is not the capture's live parser {site.live_parser}"
                )
            done = _attempts(ctx, replace(site, parser=None), sources_path, sources, grant=grant, ended=ended)
        else:
            done = _attempts(ctx, site, sources_path, sources, ended=ended)
        status = ended["status"] or "complete"
    except BaseException as exc:
        status = ended["status"] or f"crashed: {type(exc).__name__}"
        raise
    finally:
        try:
            released = held.release(status, ended["jobs"])
        except Exception as exc:  # noqa: BLE001 -- never masks the attempts' own outcome; recorded in the result
            released = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    return done, {"kind": grant.kind, "served": grant.served, "release": released}


def capture(ctx: Ctx, live: bool = True) -> dict:  # noqa: ARG001 -- capture has no plan mode
    cfg = ctx.cfg
    out, runs = ctx.path("capture"), ctx.path("runs")
    sources_path = ctx.path("select") / "sources.json"
    sources = json.loads(sources_path.read_bytes())
    site = site_for(ctx)
    held: slot.Lease | None = None
    slot_doc: dict | None = None
    with contextlib.ExitStack() as stack:
        if cfg.capture_slot is not None and _needs_attempt(ctx.path("syntax"), out, sources):
            want: Literal["warm", "cold"] = "cold" if (cs.recorded_shard_count(out) or 0) > 1 else "warm"
            try:  # only the lease itself: a capture under a lease never falls back to cold
                held = stack.enter_context(slot.lease(cfg.capture_slot, _lease_request(site, want, cfg.data_dir)))
            except slot.SlotUnavailable as exc:
                slot_doc = {"unavailable": str(exc)}
        if held is None:
            report, k, recorded, cached = _attempts(ctx, site, sources_path, sources)
        else:
            (report, k, recorded, cached), slot_doc = _on_lease(ctx, site, sources_path, sources, held)
    state = conserve(ctx, out, sources)
    if not _compat_valid(runs):
        compat_run(out, runs, cfg.root)
    counts = json.loads((out / "merge.json").read_bytes())["counts"]
    cache = cfg.capture_cache
    return (
        {
            "capture": str(out),
            "runs": str(runs),
            "units": state["merged_units"],
            "statuses": {k: v for k, v in counts.items() if k != "units"},
            "engine_s": report.get("engine_s") if report else None,
            "capture_shards": k,
            "capture_shards_requested": cfg.capture_shards,
            "capture_shards_fixed_by_reports": recorded is not None,
        }
        | ({"capture_cache": {"dir": str(cache), **cached}} if cache is not None else {})
        | ({"slot": slot_doc} if slot_doc is not None else {})
    )
