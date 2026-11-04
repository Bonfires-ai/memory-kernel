"""``mk serve``: a warm per-bonfire context server over HTTP (FastAPI; the ``serve`` extra, imported lazily).

    POST /v1/bonfires/{bonfire_id}/context   {"query": str, "speaker"?: str, "answer_shape"?: str}  (speaker = the
                                             asker: "me"/"my"/"I" resolve to them; the body then also carries
                                             "resolved_query"; answer_shape = a TurnPlan's ``answer_shape``, the
                                             frame's shape instead of the wh-rule's)
        200 {engine: "kernel", context, message_ids, tokens, frame, run, degraded: false}
        200 {..., context: "", message_ids: [], tokens: 0, frame: null, degraded: true, reason}  any other error
        404 {engine, degraded: true, reason}   no such bonfire (its group has no BonfireMeta)
        503 {engine, degraded: true, reason}   Neo4j does not answer the bonfire lookup
        422 {engine, degraded: true, reason}   the body has no non-empty "query", or a non-string speaker or
                                               answer_shape
    POST /v1/bonfires/{bonfire_id}/entities/sheet   {"name": str} or {"uuid": str}  (exactly one; null counts as absent)
        200 {entities: [{uuid, name, kinds, summary, lines: [{text, message_ids, episode_ids}], aliases, identities,
            timeline: [{name, happened_at, told_at, time_source, asserted, modality}]}]}   at most 3; read-only, never paid
        404 {entities: [], ...}   no such bonfire, or nothing matches
        503 {entities: [], ...}   the bonfire's lookup or its server does not open
        422 {entities: [], ...}   the body is not exactly one non-empty string "name" or "uuid"
    GET /v1/bonfires/{bonfire_id}/conversations/{conversation_id}/watermark
        200 {engine, conversation_id, watermark: {message_id, order, timestamp} | null}   null = never seen; the
            highest ``order`` of the chat's Episodic nodes, ``message_id`` and ``timestamp`` from its stored select row
    GET /healthz                             200 {status: "ok", version, bonfires: {bonfire_id: state},
                                                 spend: {bonfire_id: spend}, parser_notes?}  ready
        503 {status: "warming", version, bonfires, spend, parser_notes?}   not ready yet (see Warm-up)
        state = the loaded run | "warming" | "failed: <reason>" | null (no warm-up: not open yet; always a 200)
        version = the installed memory-kernel's version ("unknown" when it has no metadata)
        spend = {charged_usd, ceiling_usd}: ``kernel_ops``'s held-or-spent total and the bonfire's spend ceiling
                (null = none), as decimal strings (``ops.spend``); null without ops, or when the bonfire has no
                ``kernel_ops`` record

The runtime grammar (``serve.runtime.Runtimes``: one ``Runtime`` per bonfire, its engine started on first use):

    POST /v1/bonfires/{bonfire_id}/plan      {"text", "chat_id", "speaker"?, "addressee"?, "participants"?,
                                              "message_id"?, "at"?,
                                              "committed"?: [{tool, args, value}],
                                              "context"?: [Frame]}  (``at`` defaults to the server's UTC time;
                                              ``committed`` = effects already done; ``context`` = at most 8 open
                                              frames of the turn before, which a follow-up composes with)
        200 TurnPlan.to_json()               ``licence`` null on an unread or repair plan; an unread plan with the
                                             diagnostic "runtime busy: teaching" while the bonfire teaches
    POST /v1/bonfires/{bonfire_id}/plan/{plan_id}/resume   {"results": [{"step_id", "tool", "args", "verdict",
                                              "value", "detail"?}, ...]}
        200 TurnPlan.to_json()   404 unknown or expired plan   409 a resume that cannot apply
        503 + Retry-After        the bonfire still teaches after the resume waited (30 s)
    POST /v1/bonfires/{bonfire_id}/teach     {"forms", "body", "defaults"?, "teacher", "chat_id", "replace"?}
        200 {draft_id, specs, problems, competes}   a refused teach is a 200 with problems (well under 2 s)
        409 a TeachError
        503 the engine cannot run it now (every engine in use, an engine that does not start, a failure under the
            teach)
    POST /v1/bonfires/{bonfire_id}/teach/{draft_id}/confirm  {"by"}
        200 {draft_id, specs}   403 not the draft's teacher   409 any other refusal
        503 as for teach                     (reads nothing)
    GET  /v1/bonfires/{bonfire_id}/constructions
        200 {engine, bonfire_id, constructions: [stored spec rows]}   503 the bonfire's runtime does not load
    POST /v1/bonfires/{bonfire_id}/plan/{plan_id}/retry
        200 TurnPlan.to_json()   404 unknown or expired plan   409 the plan has no failed step to retry
        503 + Retry-After        as for resume
    GET  /v1/bonfires/{bonfire_id}/catalogue       200 {sha256, tools: [the stored catalogue's tool rows]}   (read now; null and [] until a PUT)
    PUT  /v1/bonfires/{bonfire_id}/catalogue       the catalogue's JSON as the body (BonfireMeta.catalogue_json)
        200 {sha256, tools: [names]}   the catalogue is installed and the bonfire's runtime reloads
        400 the CatalogueError text (the previous catalogue stays)
    every runtime route: 404 unknown bonfire, 422 a malformed body (a wrong type, a string that is not valid UTF-8,
    a field past its cap: text 4000 characters, ids 256, participants 256, teach forms/body 32 entries of 500)
    A plan never answers 503: what cannot be read now is a 200 ``unread`` plan naming why.

**Discovery.** A bonfire is a group with a ``BonfireMeta`` node (``serve.registry.BonfireRegistry``); its id is the
group name. A group without one (still seeding, or a 1.8 group) is not a bonfire: 404. A question reads the bonfire's
``BonfireMeta`` fresh, so a bonfire is answerable the moment its meta lands; the listing (``/healthz``, the warm-up)
is cached for ``refresh_s``.

**Warm-up.** ``mk serve`` (``create_app(warm=True)``) is ready when it can answer. At start (``Pool.warm_all``, in the
background: ``/healthz`` answers meanwhile) it loads the query and turn models, the WordNet corpus and the token
encoder once (``warm_models``, in a worker thread), then opens every listed bonfire's server at its ``committed_run``
through the path a question takes (``Pool.acquire``), each bonfire in its own task. An open that cannot connect (Neo4j
not accepting connections yet, or its host name not resolving yet: right after a restart it may come up after the
kernel) is tried again after 5 s, doubling up to 30 s, and the bonfire stays ``warming``. A bonfire whose open is
refused (a refusal, a stale index, no password) is listed as ``failed: <reason>``, is not tried again by the warm-up
(a question still tries it) and does not hold the others. ``/healthz`` is a 200 once every listed bonfire is warm or
failed and at least one is warm, else a 503; a bonfire that appears later is warmed when ``/healthz`` first lists it,
and is listed ``warming`` meanwhile without making a server that has been ready a 503.
Without ``warm`` (tests, library use) nothing is opened before a question: ``/healthz`` is a 200 and lists a bonfire
not open yet as null.

**Warm swap.** One ``Server`` per bonfire stays open (its Neo4j driver and loaded index) between questions, with the
``committed_run`` and ``generation`` of the ``BonfireMeta`` it was opened at. On each question both are read fresh; a
CHANGE of either (a rebuild is a new generation and may land on run 0 again) is opened in a background task (off the
request path: questions keep the loaded server meanwhile), then swapped in, and the old server is closed once the questions it is answering finish. An append in progress is invisible: its
``committed_run`` moves in the append's one commit transaction. A run that fails to open is logged and not tried again
for ``open_backoff_s`` (60 s); the loaded one keeps answering, and only a bonfire with no server yet answers degraded.
The run label of a body is ``"bootstrap"`` for run 0, else ``f"{n:04d}"`` (``api.run_label``).

Nothing is written to disk: the question vectors are an in-process LRU (``serve.answer``), the answers are in memory.

**Secrets** come from ``mk serve``'s resolution (``cli.resolve_secrets``, as for ``mk ask``) through
``serve_cfg_overrides``. They are never logged: a reason is built from the error's type and message with every known
secret value replaced by ``<redacted>``, and a Neo4j error is reduced to its class and code.

No ``from __future__ import annotations`` here: FastAPI resolves the route annotations (``Request``) at definition
time, inside ``create_app`` where the lazy import binds them.
"""

import asyncio
import functools
import logging
import re
import socket
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from memory_kernel import api
from memory_kernel.serve.answer import ServeMiss, resolve_first_person
from memory_kernel.serve.runtime import (
    MAX_CATALOGUE,
    MAX_ID,
    EngineFactory,
    RuntimeFactory,
    Runtimes,
    RuntimeSettings,
    parse_confirm,
    parse_plan,
    parse_results,
    parse_teach,
)

ENGINE = "kernel"
#: seconds before a run that failed to open is tried again (``mk serve --open-backoff``)
OPEN_BACKOFF_S = 60.0
#: the reason of a cache-only context (``Pool.context(paid=False)``) that missed the caches
NO_CACHED_CONTEXT = "no cached context"
SECRET_FIELDS = ("neo4j_password", "openrouter_api_key", "voyage_api_key")
REDACTED = "<redacted>"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
#: seconds before the warm-up tries again an open that could not connect, doubled after each try up to the cap
WARM_RETRY_S = 5.0
WARM_RETRY_CAP_S = 30.0
#: ``/healthz``: a bonfire not open yet (and the 503's status); a bonfire whose warm-up stopped is ``FAILED + reason``
WARMING = "warming"
FAILED = "failed: "
#: the text the warm-up parses once on each loaded pipeline (a pipeline's first call is its slowest)
WARM_TEXT = "Kira Dace met Milo at Brask Labs."
#: seconds a bonfire's ``/healthz`` spend is kept before ``ops.spend`` is read again
SPEND_TTL_S = 300.0
log = logging.getLogger("memory_kernel.serve.http")


@functools.cache
def package_version() -> str:
    """The installed memory-kernel's version (its distribution metadata, read once), "unknown" without one."""
    from memory_kernel.setup_runtime import installed_version

    return installed_version("memory-kernel") or "unknown"


def valid_bonfire_id(bonfire_id: str) -> bool:
    """A plain name: what a bonfire id (a Neo4j group) may be in a URL."""
    return bool(_ID.fullmatch(bonfire_id)) and set(bonfire_id) != {"."}


def warm_models() -> None:
    """Load what a question reads with, once, before the bonfires open (``Pool.warm_all`` runs it in a worker thread):
    the WordNet corpus (its first read loads every index file: seconds), the query and turn spaCy pipelines, each
    parsing ``WARM_TEXT`` once, and last the token encoder the context budget counts with (``count_tokens``). The
    pipelines' own fallback applies (``g1_nlp.nlp``); a raise means a model or the encoder does not load."""
    from memory_kernel.grammar.dependencies import wordnet_ready
    from memory_kernel.graph.g1_nlp import nlp
    from memory_kernel.search.kg_context import count_tokens

    wordnet_ready()  # under the WordNet lookup lock (the probe takes it)
    pipes: dict[int, Any] = {id(p): p for p in (nlp("query"), nlp("turn"))}  # one entry when both read on one model
    for pipe in pipes.values():
        pipe(WARM_TEXT)
    count_tokens(WARM_TEXT)  # last: an encoder that will not load (``TokenizerUnavailable``) leaves the models warm


def transient(exc: BaseException) -> bool:
    """An open that failed because Neo4j does not accept connections (yet), or because its host name does not resolve
    (yet: at a container start the name may come up after the kernel; the driver raises a ``ValueError`` from the
    resolver's ``socket.gaierror`` for most resolver errors): the warm-up tries it again. Anything else (an
    ``AuthError`` or any other refusal, a ``ConfigurationError``, a stale index, no password) is permanent."""
    from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

    if isinstance(exc, socket.gaierror) or isinstance(exc.__cause__, socket.gaierror):  # or the driver's ValueError's
        return True
    return isinstance(exc, ConnectionError | TimeoutError | ServiceUnavailable | SessionExpired | TransientError)


def refusal(reason: str) -> dict[str, Any]:
    return {"engine": ENGINE, "degraded": True, "reason": reason}


def degraded(run: str | None, reason: str) -> dict[str, Any]:
    return {
        "engine": ENGINE,
        "context": "",
        "message_ids": [],
        "tokens": 0,
        "frame": None,
        "run": run,
        "degraded": True,
        "reason": reason,
    }


@dataclass
class Warm:
    """One bonfire's open server and the ``committed_run`` it answers from; ``inflight`` questions keep a retired one
    open."""

    run: int
    server: Any
    generation: str | None = None
    inflight: int = 0
    retired: bool = False
    closed: bool = False

    @property
    def label(self) -> str:
        return api.run_label(self.run)


def committed_run(meta: Mapping[str, Any]) -> int:
    return int(meta.get("committed_run") or 0)


def meta_generation(meta: Mapping[str, Any]) -> str | None:
    """The seal of the bonfire (``BonfireMeta.generation``): a reseed mints a new one and usually lands on run 0 again."""
    return meta.get("generation")


class Pool:
    """The warm servers of the bonfires ``registry`` lists (``serve.registry.BonfireRegistry``: ``ids()``, ``meta(id)``,
    ``watermark(id, chat)``). ``server_factory(cfg)`` makes an (unopened) async context manager with
    ``context(question)``; the default is ``serve.answer.Server``. ``ops`` (a ``memory_kernel.ops.KernelOps``) gives
    ``/healthz`` its spend; without one it is null."""

    def __init__(
        self,
        registry: Any,
        *,
        ops: Any = None,
        overrides: Mapping[str, Any] | None = None,
        server_factory: Callable[[Any], Any] | None = None,
        open_backoff_s: float = OPEN_BACKOFF_S,
        warm: bool = False,
        warm_retry_s: float = WARM_RETRY_S,
    ) -> None:
        self.registry = registry
        self.ops = ops
        self.overrides = dict(overrides or {})
        self.factory = server_factory
        self.open_backoff_s = open_backoff_s
        self._warm: dict[str, Warm] = {}
        self._opening: dict[str, asyncio.Lock] = {}
        self._swaps: dict[str, asyncio.Task[None]] = {}
        #: (bonfire id, committed run) -> (monotonic time, reason) of its last failed open: not retried within the backoff
        self._failed: dict[tuple[str, int], tuple[float, str]] = {}
        self._secrets: list[str] | None = None
        #: the warm-up (module docstring): on with ``warm``; ``warm_all``'s task; each bonfire's task; why one stopped
        self.warm = warm
        self.warm_retry_s = warm_retry_s
        self._warm_all: asyncio.Task[None] | None = None
        self._warming: dict[str, asyncio.Task[None]] = {}
        self._warm_failed: dict[str, str] = {}
        self._models_warm = False
        #: ``health`` has once been ready: a bonfire found later is listed warming without making the server a 503
        self._was_ready = False
        #: the runtime grammar's runtimes (``create_app`` sets them)
        self.runtimes: Runtimes | None = None
        #: bonfire id -> (monotonic time, spend) of its last ``ops.spend`` read
        self._spend: dict[str, tuple[float, dict[str, str | None] | None]] = {}

    # -- discovery -----------------------------------------------------------------------------------------------------

    async def meta(self, bonfire_id: str) -> dict[str, Any] | None:
        """The bonfire's ``BonfireMeta`` (read fresh), or None when it is not a bonfire: the group has none, or the id is
        not a plain name. A Neo4j error propagates."""
        if not valid_bonfire_id(bonfire_id):
            return None
        return await self.registry.meta(bonfire_id)

    async def health(self) -> tuple[bool, dict[str, str | None]]:
        """``(ready, {bonfire id: its loaded run | "warming" | "failed: <reason>" | None})``. With ``warm``: ready once
        every listed bonfire is warm or failed and at least one is warm; zero bonfires is ready once the models are
        warm. Once ready (``_was_ready``), a bonfire that is still warming (it appeared later) keeps the server ready
        even when none is warm yet (the first bonfire of an empty kernel); a server whose listed bonfires have all
        failed, with none warm or warming, is not ready. A listed bonfire that has no state yet is warmed now.
        Without ``warm`` nothing opens before a question: a bonfire not open yet is None and the server is ready.
        A listing that Neo4j cannot give is logged: the bonfires already open are listed, and a ``warm`` server is
        ready only if it has been ready before."""
        states: dict[str, str | None] = {}
        try:
            ids = await self.registry.ids()
        except Exception as exc:
            log.warning("healthz: the bonfires do not list: %s", self.reason(exc))
            return (not self.warm) or self._was_ready, {b: w.label for b, w in self._warm.items()}
        if self.warm and not ids:
            self._was_ready = self._was_ready or self._models_warm
            return self._models_warm, states
        warm = pending = False
        for b in ids:
            failed = self._warm_failed.get(b)
            if (w := self._warm.get(b)) is not None:
                states[b], warm = w.label, True
            elif not self.warm:
                states[b] = None
            elif failed is not None:
                states[b] = FAILED + failed
            else:  # no state yet
                states[b], pending = WARMING, True
                if self._models_warm:
                    self._start_warm(b)
        if not self.warm:
            return True, states
        self._was_ready = self._was_ready or (warm and not pending)
        return (warm or pending) and (not pending or self._was_ready), states

    async def spend(self, bonfire_ids: list[str]) -> dict[str, dict[str, str | None] | None]:
        """``{bonfire id: {"charged_usd", "ceiling_usd"}}`` from ``ops.spend`` (the held-or-spent total of the bonfire's
        appends and its spend ceiling, None = none, as decimal strings). None for a bonfire with no ``kernel_ops``
        record or whose read fails, and for every bonfire without ``ops``. The blocking read runs in a worker thread."""
        if self.ops is None:
            return dict.fromkeys(bonfire_ids)

        async def one(bonfire_id: str) -> dict[str, str | None] | None:
            cached = self._spend.get(bonfire_id)
            if cached is not None and time.monotonic() - cached[0] < SPEND_TTL_S:
                return cached[1]
            try:
                value = await asyncio.to_thread(self.ops.spend, bonfire_id)
            except Exception as exc:  # no record, or ops down: /healthz still answers
                log.warning("bonfire %s: spend does not read: %s", bonfire_id, self.reason(exc))
                value = None
            self._spend[bonfire_id] = (time.monotonic(), value)
            return value

        return dict(zip(bonfire_ids, await asyncio.gather(*(one(b) for b in bonfire_ids)), strict=True))

    # -- reasons -------------------------------------------------------------------------------------------------------

    def secrets(self) -> list[str]:
        if self._secrets is None:
            from memory_kernel.cli import resolve_secrets

            env = resolve_secrets(env_file=None, container=None).values
            values = [self.overrides.get(k) for k in SECRET_FIELDS] + [env.get(k) for k in SECRET_FIELDS]
            self._secrets = sorted({v for v in values if isinstance(v, str) and v}, key=len, reverse=True)
        return self._secrets

    def scrub(self, text: str) -> str:
        for value in self.secrets():
            text = text.replace(value, REDACTED)
        return text

    def reason(self, exc: BaseException) -> str:
        try:
            from neo4j.exceptions import DriverError, Neo4jError
        except ImportError:  # pragma: no cover - neo4j is a base dependency
            DriverError = Neo4jError = ()  # type: ignore[assignment,misc]
        if isinstance(exc, Neo4jError):
            return f"neo4j: refused ({type(exc).__name__} {getattr(exc, 'code', None)})"
        if isinstance(exc, DriverError):
            return f"neo4j: unreachable or failing ({type(exc).__name__})"
        return self.scrub(f"{type(exc).__name__}: {exc}")

    # -- the warm servers ----------------------------------------------------------------------------------------------

    async def _open(self, bonfire_id: str, meta: Mapping[str, Any]) -> Warm:
        cfg = await asyncio.to_thread(api.serve_config, bonfire_id, meta, overrides=self.overrides)
        if self.factory is not None:
            server = self.factory(cfg)
        else:
            from memory_kernel.serve import answer

            server = answer.Server(cfg)
        opened = await server.__aenter__()
        return Warm(
            run=committed_run(meta),
            generation=meta_generation(meta),
            server=opened if opened is not None else server,
        )

    async def _close(self, warm: Warm) -> None:
        if warm.closed:
            return
        warm.closed = True
        try:
            await warm.server.__aexit__(None, None, None)
        except Exception as exc:  # a failing close never fails a question
            log.warning("closing run %s: %s", warm.label, self.reason(exc))

    def _backing_off(self, bonfire_id: str, run: int) -> str | None:
        """The reason of ``run``'s last failed open while it is within the backoff, else None."""
        failed = self._failed.get((bonfire_id, run))
        if failed is not None and time.monotonic() - failed[0] < self.open_backoff_s:
            return failed[1]
        return None

    def _failed_open(self, bonfire_id: str, run: int, exc: BaseException) -> str:
        reason = self.reason(exc)
        self._failed[(bonfire_id, run)] = (time.monotonic(), reason)
        return reason

    async def acquire(self, bonfire_id: str, meta: Mapping[str, Any], *, backoff: bool = True) -> Warm:
        """The warm server to answer from. With nothing loaded, the bonfire is opened at its ``committed_run`` here
        (raises when it fails, or failed within the backoff). With a server loaded, a changed ``(committed_run, generation)`` is
        opened in the background (``_swap``) and the loaded server answers meanwhile; a run that failed to open is not
        retried within the backoff. ``backoff`` False (the warm-up, which keeps its own) opens even within the
        backoff."""
        run = committed_run(meta)
        current = self._warm.get(bonfire_id)
        if current is None:
            async with self._opening.setdefault(bonfire_id, asyncio.Lock()):
                current = self._warm.get(bonfire_id)
                if current is None:
                    if backoff and (reason := self._backing_off(bonfire_id, run)) is not None:
                        raise RuntimeError(
                            f"run {api.run_label(run)} did not open (retried after {self.open_backoff_s:g} s): {reason}"
                        )
                    try:
                        current = await self._open(bonfire_id, meta)
                    except Exception as exc:
                        self._failed_open(bonfire_id, run, exc)
                        raise
                    self._failed.pop((bonfire_id, run), None)
                    self._warm[bonfire_id] = current
        elif (run, meta_generation(meta)) != (
            current.run,
            current.generation,
        ):  # ANY change: a rebuild's run may repeat
            self._start_swap(bonfire_id, meta)
        current.inflight += 1
        return current

    def _start_swap(self, bonfire_id: str, meta: Mapping[str, Any]) -> None:
        swap = self._swaps.get(bonfire_id)
        if (swap is None or swap.done()) and self._backing_off(bonfire_id, committed_run(meta)) is None:
            self._swaps[bonfire_id] = asyncio.create_task(self._swap(bonfire_id, meta))

    async def _swap(self, bonfire_id: str, meta: Mapping[str, Any]) -> None:
        """Open the new run off the request path, swap it in, retire the old server (closed after its last question)."""
        run = committed_run(meta)
        try:
            new = await self._open(bonfire_id, meta)
        except Exception as exc:
            reason = self._failed_open(bonfire_id, run, exc)
            log.warning(
                "bonfire %s: run %s does not open (%s); answering from the loaded run",
                bonfire_id,
                api.run_label(run),
                reason,
            )
            return
        self._failed.pop((bonfire_id, run), None)
        old = self._warm.get(bonfire_id)
        if old is not None and (old.run, old.generation) == (new.run, new.generation):
            await self._close(new)
            return
        self._warm[bonfire_id] = new
        if old is not None:
            log.info("bonfire %s: swapped run %s -> %s", bonfire_id, old.label, new.label)
            old.retired = True
            if old.inflight == 0:
                await self._close(old)

    # -- the warm-up ---------------------------------------------------------------------------------------------------

    def start(self) -> None:
        """With ``warm``: start ``warm_all`` in the background (``mk serve``'s lifespan)."""
        if self.warm and self._warm_all is None:
            self._warm_all = asyncio.create_task(self.warm_all())

    async def warm_all(self) -> None:
        """Load the models once (``warm_models``, in a worker thread), then open every listed bonfire, each in its own
        task: one bonfire's retries never hold another."""
        try:
            await asyncio.to_thread(warm_models)
        except Exception as exc:  # each question then loads what it reads with; the bonfires still open
            log.warning("warm-up: the models did not load: %s", self.reason(exc))
        self._models_warm = True
        try:
            ids = await self.registry.ids()
        except Exception as exc:  # Neo4j is not up yet: ``health`` lists and warms them once it is
            log.warning("warm-up: the bonfires do not list yet: %s", self.reason(exc))
            ids = []
        for bonfire_id in ids:
            self._start_warm(bonfire_id)
        await asyncio.gather(*self._warming.values(), return_exceptions=True)
        await (
            self.health()
        )  # latches readiness even if nothing polled /healthz: a bonfire found later never makes it a 503

    def _start_warm(self, bonfire_id: str) -> None:
        task = self._warming.get(bonfire_id)
        if task is None or task.done():
            self._warming[bonfire_id] = asyncio.create_task(self._warm_one(bonfire_id))

    async def _warm_one(self, bonfire_id: str) -> None:
        """Open one bonfire through ``acquire``, the path a question takes. A lookup or an open that cannot connect
        (``transient``) is tried again after ``warm_retry_s``, doubled each time up to ``WARM_RETRY_CAP_S`` (the
        ``BonfireMeta`` is read again each time); any other failure is recorded and the warm-up stops."""
        delay = self.warm_retry_s
        while bonfire_id not in self._warm:  # a question may open it while this waits
            try:  # the warm-up keeps its own backoff, not a question's: a question's failed open never stops it
                meta = await self.meta(bonfire_id)
                if meta is None:  # no BonfireMeta (any more): ``health`` warms it again if it is listed
                    return
                warm = await self.acquire(bonfire_id, meta, backoff=False)
            except Exception as exc:
                if not transient(exc):
                    self._stop_warm(bonfire_id, self.reason(exc))
                    return
                log.warning(
                    "bonfire %s: cannot connect yet (%s); trying again in %g s",
                    bonfire_id,
                    self.reason(exc),
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, WARM_RETRY_CAP_S)
                continue
            await self.release(warm)
            log.info("bonfire %s: warm on run %s", bonfire_id, warm.label)

    def _stop_warm(self, bonfire_id: str, reason: str) -> None:
        self._warm_failed[bonfire_id] = reason
        log.warning("bonfire %s: not warmed: %s", bonfire_id, reason)

    async def wait_swaps(self) -> None:
        """Wait for the background swaps under way (tests, shutdown)."""
        await asyncio.gather(*self._swaps.values(), return_exceptions=True)

    async def release(self, warm: Warm) -> None:
        warm.inflight -= 1
        if warm.retired and warm.inflight == 0:
            await self._close(warm)

    async def context(
        self,
        bonfire_id: str,
        query: str,
        speaker: str | None = None,
        *,
        paid: bool = True,
        answer_shape: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """``(status, body)`` of one question (the module docstring's shapes). ``speaker`` (the asker) resolves the
        question's first-person words before retrieval (``answer.resolve_first_person``). ``paid`` False (the runtime
        grammar's recall) answers from the caches only, whatever ``allow_paid`` says: a cache miss is a degraded body
        with the reason ``NO_CACHED_CONTEXT``. ``answer_shape`` (a ``TurnPlan.answer_shape`` the caller has) is the
        frame's shape in place of the wh-rule's; without one, today's path."""
        try:
            meta = await self.meta(bonfire_id)
        except Exception as exc:  # Neo4j does not answer: never a traceback
            return 503, refusal(f"bonfire {bonfire_id!r} does not load: {self.reason(exc)}")
        if meta is None:
            return 404, refusal(f"unknown bonfire {bonfire_id!r}")
        run = api.run_label(committed_run(meta))
        try:
            warm = await self.acquire(bonfire_id, meta)
        except Exception as exc:
            reason = self.reason(exc)
            log.warning("bonfire %s: run %s does not open: %s", bonfire_id, run, reason)
            return 200, degraded(run, reason)
        try:
            resolved = resolve_first_person(query, speaker)
            kwargs: dict[str, Any] = {} if paid else {"paid": False}
            if answer_shape is not None:
                kwargs["shape"] = answer_shape
            ctx = await warm.server.context(resolved, **kwargs)
            body = {
                "engine": ENGINE,
                "context": ctx["text"],
                "message_ids": list(ctx["message_ids"]),
                "tokens": ctx["tokens"],
                "frame": ctx["frame"],
                "run": warm.label,
                "degraded": False,
                **({"resolved_query": resolved} if resolved != query else {}),
            }
        except Exception as exc:
            if not paid and isinstance(exc, ServeMiss):  # cache-only: a miss, never a send
                return 200, degraded(warm.label, NO_CACHED_CONTEXT)
            reason = self.reason(exc)
            log.warning("bonfire %s run %s: degraded answer: %s", bonfire_id, warm.label, reason)
            return 200, degraded(warm.label, reason)
        finally:
            await self.release(warm)
        return 200, body

    async def entity_sheet(self, bonfire_id: str, *, name: str | None, uuid: str | None) -> tuple[int, dict[str, Any]]:
        """``(status, body)`` of the entity-sheet route: ``{"entities": [...]}``, 404 (empty list) when the bonfire is
        unknown or nothing matches, 503 when it does not look up or open. Never paid."""
        empty = {"entities": []}
        try:
            meta = await self.meta(bonfire_id)
        except Exception as exc:
            return 503, {**refusal(f"bonfire {bonfire_id!r} does not load: {self.reason(exc)}"), **empty}
        if meta is None:
            return 404, {**refusal(f"unknown bonfire {bonfire_id!r}"), **empty}
        try:
            warm = await self.acquire(bonfire_id, meta)
        except Exception as exc:
            return 503, {**refusal(self.reason(exc)), **empty}
        try:
            entities = await warm.server.entity_sheet(name=name, uuid=uuid)
        except Exception as exc:
            return 503, {**refusal(self.reason(exc)), **empty}
        finally:
            await self.release(warm)
        return (200 if entities else 404), {"engine": ENGINE, "run": warm.label, "entities": entities}

    async def close(self) -> None:
        warmups = [t for t in (self._warm_all, *self._warming.values()) if t is not None]
        for task in [*warmups, *self._swaps.values()]:
            task.cancel()
        await asyncio.gather(*warmups, return_exceptions=True)  # before the servers close: no open lands after
        await self.wait_swaps()
        for warm in list(self._warm.values()):
            await self._close(warm)
        self._warm.clear()
        if self.runtimes is not None:
            await self.runtimes.close()


def create_app(
    registry: Any,
    *,
    ops: Any = None,
    serve_cfg_overrides: Mapping[str, Any] | None = None,
    server_factory: Callable[[Any], Any] | None = None,
    open_backoff_s: float = OPEN_BACKOFF_S,
    runtime_settings: RuntimeSettings | None = None,
    runtime_factory: RuntimeFactory | None = None,
    engine_factory: EngineFactory | None = None,
    writer_factory: Callable[[str], Any] | None = None,
    warm: bool = False,
    warm_retry_s: float = WARM_RETRY_S,
) -> Any:
    """The FastAPI app of ``mk serve`` over the bonfires ``registry`` lists (see the module docstring).
    ``runtime_settings`` default to the environment's (``RuntimeSettings.from_env``). ``warm`` (``mk serve`` sets it)
    opens the models and every bonfire at start, and ``/healthz`` is not ready until they are (Warm-up)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    pool = Pool(
        registry,
        ops=ops,
        overrides=serve_cfg_overrides,
        server_factory=server_factory,
        open_backoff_s=open_backoff_s,
        warm=warm,
        warm_retry_s=warm_retry_s,
    )
    runtimes = pool.runtimes = Runtimes(
        pool,
        runtime_settings,
        runtime_factory=runtime_factory,
        engine_factory=engine_factory,
        writer_factory=writer_factory,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        pool.start()  # the warm-up runs in the background: /healthz answers while it does
        yield
        await pool.close()

    app = FastAPI(title="memory-kernel", lifespan=lifespan)
    app.state.pool = pool

    def unknown(bonfire_id: str) -> JSONResponse:
        return JSONResponse(refusal(f"unknown bonfire {bonfire_id!r}"), status_code=404)

    async def found(bonfire_id: str) -> dict[str, Any] | JSONResponse:
        """The bonfire's ``BonfireMeta``, or the 404 (not a bonfire) / 503 (Neo4j does not answer) to return."""
        try:
            meta = await pool.meta(bonfire_id)
        except Exception as exc:
            return JSONResponse(refusal(f"bonfire {bonfire_id!r} does not load: {pool.reason(exc)}"), status_code=503)
        return unknown(bonfire_id) if meta is None else meta

    @app.post("/v1/bonfires/{bonfire_id}/context")
    async def bonfire_context(bonfire_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            body = None
        query = body.get("query") if isinstance(body, dict) else None
        if not isinstance(query, str) or not query.strip():
            return JSONResponse(refusal('the body must be {"query": <a non-empty string>}'), status_code=422)
        speaker = body.get("speaker") if isinstance(body, dict) else None
        if speaker is not None and not isinstance(speaker, str):
            return JSONResponse(refusal('"speaker" must be a string (the asker) when given'), status_code=422)
        shape = body.get("answer_shape") if isinstance(body, dict) else None
        if shape is not None and (not isinstance(shape, str) or not shape or len(shape) > MAX_ID):
            reason = '"answer_shape" must be a non-empty string (a TurnPlan\'s answer_shape) when given'
            return JSONResponse(refusal(reason), status_code=422)
        status, out = await pool.context(bonfire_id, query, speaker, answer_shape=shape)
        return JSONResponse(out, status_code=status)

    @app.post("/v1/bonfires/{bonfire_id}/entities/sheet")
    async def bonfire_entity_sheet(bonfire_id: str, request: Request) -> JSONResponse:
        """Who is X: ``{"name": str}`` or ``{"uuid": str}`` -> ``{"entities": [...]}`` (at most 3), each with its summary
        lines and their source message ids. 422 on a bad body, 404 with an empty list when nothing matches."""
        try:
            body = await request.json()
        except ValueError:
            body = None
        keys = {k: body[k] for k in ("name", "uuid") if isinstance(body, dict) and body.get(k) is not None}
        if len(keys) != 1 or not all(isinstance(v, str) and v.strip() and len(v) <= 512 for v in keys.values()):
            reason = 'the body must be {"name": <a non-empty string>} or {"uuid": <a non-empty string>}'
            return JSONResponse({**refusal(reason), "entities": []}, status_code=422)
        status, out = await pool.entity_sheet(
            bonfire_id, name=keys["name"].strip() if "name" in keys else None, uuid=keys.get("uuid")
        )
        return JSONResponse(out, status_code=status)

    @app.get("/v1/bonfires/{bonfire_id}/conversations/{conversation_id}/watermark")
    async def bonfire_watermark(bonfire_id: str, conversation_id: str) -> JSONResponse:
        """How far the bonfire has taken in one conversation (``registry.watermark``: the highest ``order`` of its
        Episodic nodes): the caller keeps everything of that conversation after it in its own history, since the graph
        does not hold it yet."""
        meta = await found(bonfire_id)
        if isinstance(meta, JSONResponse):
            return meta
        try:
            mark = await registry.watermark(bonfire_id, conversation_id)
        except Exception as exc:  # Neo4j does not answer
            return JSONResponse(refusal(pool.reason(exc)), status_code=503)
        return JSONResponse({"engine": ENGINE, "conversation_id": conversation_id, "watermark": mark})

    async def json_body(request: Request) -> Any:
        try:
            return await request.json()
        except ValueError:
            return None

    @app.post("/v1/bonfires/{bonfire_id}/plan")
    async def bonfire_plan(bonfire_id: str, request: Request) -> JSONResponse:
        """Read one turn with the bonfire's runtime grammar (``Runtime.plan``, with its ``committed`` effects and the
        ``context`` frames it may compose with): the ``TurnPlan``."""
        fire = await found(bonfire_id)
        if isinstance(fire, JSONResponse):
            return fire
        parsed = parse_plan(await json_body(request))
        if isinstance(parsed, str):
            return JSONResponse(refusal(parsed), status_code=422)
        turn, committed, context = parsed
        status, out = await runtimes.plan(bonfire_id, fire, turn, committed, context)
        return JSONResponse(out, status_code=status)

    @app.post("/v1/bonfires/{bonfire_id}/plan/{plan_id}/resume")
    async def bonfire_resume(bonfire_id: str, plan_id: str, request: Request) -> JSONResponse:
        """Continue a ``needs_tools`` plan with its tool results (``Runtime.resume``)."""
        if isinstance(missing := await found(bonfire_id), JSONResponse):
            return missing
        results = parse_results(await json_body(request))
        if isinstance(results, str):
            return JSONResponse(refusal(results), status_code=422)
        status, out, headers = await runtimes.resume(bonfire_id, plan_id, results)
        return JSONResponse(out, status_code=status, headers=headers)

    @app.post("/v1/bonfires/{bonfire_id}/plan/{plan_id}/retry")
    async def bonfire_retry(bonfire_id: str, plan_id: str) -> JSONResponse:
        """Issue again the step a verdict refused (``Runtime.retry``)."""
        if isinstance(missing := await found(bonfire_id), JSONResponse):
            return missing
        status, out, headers = await runtimes.retry(bonfire_id, plan_id)
        return JSONResponse(out, status_code=status, headers=headers)

    @app.get("/v1/bonfires/{bonfire_id}/catalogue")
    async def bonfire_catalogue(bonfire_id: str) -> JSONResponse:
        """The bonfire's tool catalogue as installed: ``{sha256, tools}``."""
        fire = await found(bonfire_id)
        if isinstance(fire, JSONResponse):
            return fire
        status, out = await runtimes.catalogue(bonfire_id)
        return JSONResponse(out, status_code=status)

    @app.put("/v1/bonfires/{bonfire_id}/catalogue")
    async def bonfire_put_catalogue(bonfire_id: str, request: Request) -> JSONResponse:
        """Validate and install a catalogue (the file's JSON as the body), then reload the bonfire's runtime."""
        fire = await found(bonfire_id)
        if isinstance(fire, JSONResponse):
            return fire
        raw = await request.body()
        if len(raw) > MAX_CATALOGUE:
            return JSONResponse(refusal(f"the catalogue is larger than {MAX_CATALOGUE} bytes"), status_code=400)
        status, out = await runtimes.set_catalogue(bonfire_id, fire, raw)
        return JSONResponse(out, status_code=status)

    @app.post("/v1/bonfires/{bonfire_id}/teach")
    async def bonfire_teach(bonfire_id: str, request: Request) -> JSONResponse:
        """Draft a construction from a definition (``Runtime.teach``): its body and its own renderings read (under 2 s)."""
        fire = await found(bonfire_id)
        if isinstance(fire, JSONResponse):
            return fire
        req = parse_teach(await json_body(request))
        if isinstance(req, str):
            return JSONResponse(refusal(req), status_code=422)
        status, out = await runtimes.teach(bonfire_id, fire, req)
        return JSONResponse(out, status_code=status)

    @app.post("/v1/bonfires/{bonfire_id}/teach/{draft_id}/confirm")
    async def bonfire_confirm(bonfire_id: str, draft_id: str, request: Request) -> JSONResponse:
        """The teacher makes a draft live bonfire-wide (``Runtime.confirm``); as long as a teach."""
        fire = await found(bonfire_id)
        if isinstance(fire, JSONResponse):
            return fire
        parsed = parse_confirm(await json_body(request))
        if isinstance(parsed, str):
            return JSONResponse(refusal(parsed), status_code=422)
        status, out = await runtimes.confirm(bonfire_id, fire, draft_id, parsed[0])
        return JSONResponse(out, status_code=status)

    @app.get("/v1/bonfires/{bonfire_id}/constructions")
    async def bonfire_constructions(bonfire_id: str) -> JSONResponse:
        """The bonfire's stored constructions (live and staged rows)."""
        fire = await found(bonfire_id)
        if isinstance(fire, JSONResponse):
            return fire
        status, out = await runtimes.constructions(bonfire_id, fire)
        return JSONResponse(out, status_code=status)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        from memory_kernel.graph.g1_nlp import parser_note

        ready, states = await pool.health()
        note = parser_note("query")  # the query purpose fell back to sm: a deployment sees it here, not in the logs
        spend = await pool.spend(list(states))
        body = {"status": "ok" if ready else WARMING, "version": package_version(), "bonfires": states, "spend": spend}
        return JSONResponse(
            body | ({"parser_notes": {"query": note}} if note else {}), status_code=200 if ready else 503
        )

    return app


__all__ = ["ENGINE", "Pool", "create_app"]
