"""``mk serve``'s runtime grammar: one ``Runtime`` per bonfire, its FCG engine started lazily and capped.

``Runtimes`` holds the runtimes of the bonfires under the served root (``http.Pool``) and answers the runtime routes
(``http.create_app``): plan, resume, teach, confirm and the constructions listing. Each method returns
``(status, body)`` (``resume`` also the response headers); the bodies are ``TurnPlan.to_json()``, a ``Draft``
(``draft_json``) or spec rows (``CxnSpec.to_json``).

**Engines.** A bonfire's runtime is made on its first runtime request; its engine starts on its first engine request
(plan, teach, confirm), on a port from ``RuntimeSettings.runtime_ports``. At most ``max_runtime_engines`` engines run
(the host runs two FCG engines in all): starting one more first stops, by pid, the engine of the least recently used
bonfire that is idle (no plan, teach or confirm in flight). Its runtime is kept (its plans, turn window and resumes do
not need the engine) and gets an engine again on its next plan. When every engine is in use, a plan is ``unread``
(diagnostic ``ENGINES_BUSY``) and a teach or confirm is a 503; so is a stop that fails (the engine stays counted:
the cap is never exceeded). A port something else listens on (``PortBusy``: a foreign process, or the engine of a
live owner) is skipped for the next free port of the range. Every engine stops when serving stops (``close``; a slot
that fails is logged and the others go on).

**Teaching.** A teach reads its body clauses and its forms' own renderings (well under 2 s; a confirm reads nothing)
and holds the bonfire's engine. While a bonfire teaches or confirms, its plans wait for it to end, at most
``teach_wait_ms`` (default 15000), then answer with an ``unread`` plan (``licence`` null, diagnostic
``BUSY_TEACHING``), so the agent runtime runs the turn as today. A resume waits for the teach to end, at most
``resume_wait_s``, then answers 503 with ``Retry-After``. Other bonfires are not affected.

**Catalogue.** A bonfire's ``BonfireMeta.catalogue_json`` (absent until the first PUT: no tools, no cards, as a
missing catalogue file was in 1.8) gives its runtime the tools and their generated cards (``Runtime(cards=...)``; never
stored). ``GET /catalogue`` reads it through the narrow writer, fresh; ``set_catalogue`` validates one, sets
``catalogue_json`` and reloads the bonfire's runtime; a refused one changes nothing. A PUT does not move
``committed_run``, so the server pool does not swap.

**Graph.** The default runtime reads the bonfire's Neo4j group through ``GraphReader`` (READ sessions only), at the
serve's Neo4j address, with the Neo4j password ``mk serve`` resolved; its ``recall`` (the fallback on an empty result)
is the bonfire's ``/context`` from the caches only (``recall_for``): a plan never makes a paid call. Its taught
phrases, their counters and the catalogue are written through the ``NarrowWriter`` (a sync driver, the process's only
write path); its rolesets are the group's ``Roleset`` nodes. A runtime reads its rolesets and stored rows once, when
it is made.

**Bodies** are refused (422) when a string is not valid UTF-8 or a field passes its cap (``MAX_TEXT``, ``MAX_ID``,
``MAX_PARTICIPANTS``, ``MAX_TEACH_ENTRIES``, ``MAX_TEACH_ENTRY``).

**Settings** come from the environment (``RuntimeSettings.from_env``): the variable of a field is ``MK_SERVE_`` + the
field name in capitals (``MK_SERVE_MAX_RUNTIME_ENGINES``, ``MK_SERVE_RUNTIME_PORTS``, ``MK_SERVE_RESUME_WAIT_S``, ``MK_SERVE_TEACH_WAIT_MS``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from memory_kernel.config import env_float, env_int
from memory_kernel.pipeline.engine import PORT_RANGE, PROTECTED_PORTS
from memory_kernel.runtime.api import VERDICTS, ResumeError, UnknownPlan, committed_entries, unread_plan
from memory_kernel.runtime.executor import Turn
from memory_kernel.runtime.frames import context_frames
from memory_kernel.runtime.teach import WARNINGS, Draft, NotTeacher, TeachError, TeachRequest

if TYPE_CHECKING:
    from memory_kernel.runtime.graph_store import NarrowWriter

#: the diagnostic of a plan answered while its bonfire teaches or confirms
BUSY_TEACHING = "runtime busy: teaching"
#: the diagnostic of a plan answered while the bonfire's tool catalogue is replaced
BUSY_RELOADING = "runtime busy: reloading the tool catalogue"
#: the diagnostic of a plan answered while every runtime engine is in use
ENGINES_BUSY = "runtime busy: every runtime engine is in use"
#: seconds a client waits before it retries a 503
RETRY_AFTER_S = 5
log = logging.getLogger("memory_kernel.serve.runtime")
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"


@dataclass(frozen=True)
class RuntimeSettings:
    """The runtime engines of ``mk serve``. ``runtime_ports``: ``"A-B"`` or one port, private 97xx ports."""

    ENV_PREFIX: ClassVar[str] = "MK_SERVE_"

    max_runtime_engines: int = 2
    runtime_ports: str = "9712-9719"  # 9711 is MK_RUNTIME_FCG_PORT's default (mk plan)
    resume_wait_s: float = 30.0
    teach_wait_ms: int = 15000  # how long a plan waits behind a teach before it answers unread (BUSY_TEACHING)
    ports: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", _ports(self.runtime_ports))
        if self.max_runtime_engines < 1:
            raise ValueError(f"max_runtime_engines must be at least 1, got {self.max_runtime_engines}")
        if len(self.ports) < self.max_runtime_engines:
            raise ValueError(
                f"runtime_ports {self.runtime_ports!r} holds {len(self.ports)} ports for "
                f"{self.max_runtime_engines} engines"
            )
        if self.resume_wait_s < 0:
            raise ValueError(f"resume_wait_s must not be negative, got {self.resume_wait_s}")
        if self.teach_wait_ms < 0:
            raise ValueError(f"teach_wait_ms must not be negative, got {self.teach_wait_ms}")

    @classmethod
    def env_names(cls) -> dict[str, str]:
        """Field -> its environment variable (the prefix + the field name in capitals)."""
        return {f.name: cls.ENV_PREFIX + f.name.upper() for f in fields(cls) if f.init}

    @classmethod
    def from_env(cls) -> RuntimeSettings:
        """The settings from the environment, read when called; unset or empty = the default, malformed = raises."""
        names, default = cls.env_names(), cls()
        ports = str(os.environ.get(names["runtime_ports"]) or "").strip() or default.runtime_ports
        return cls(
            max_runtime_engines=env_int(names["max_runtime_engines"], default.max_runtime_engines),
            runtime_ports=ports,
            resume_wait_s=env_float(names["resume_wait_s"], default.resume_wait_s),
            teach_wait_ms=env_int(names["teach_wait_ms"], default.teach_wait_ms),
        )


def _ports(text: str) -> tuple[int, ...]:
    low, _, high = text.strip().partition("-")
    try:
        first, last = int(low), int(high or low)
    except ValueError:
        raise ValueError(f"runtime_ports {text!r} is not A-B or one port") from None
    ports = tuple(range(first, last + 1))
    bad = [p for p in ports if p not in PORT_RANGE or p in PROTECTED_PORTS]
    if not ports or bad:
        raise ValueError(
            f"runtime_ports {text!r}: private ports {PORT_RANGE.start}-{PORT_RANGE.stop - 1} only, never "
            f"{PROTECTED_PORTS}"
        )
    return ports


# -- the graph --------------------------------------------------------------------------------------------------------


class GraphReader:
    """A runtime ``Reader`` over one Neo4j server: every session is a READ session (``answer.ReadOnlyDriver``); the
    driver opens on the first query and closes with ``close``. Without a password every query raises (the turn is a
    repair naming the cause)."""

    def __init__(self, uri: str, user: str, password: str | None) -> None:
        self.uri, self.user = uri, user
        self._password = password
        self._driver: Any = None

    async def rows(self, cypher: str, params: Mapping[str, Any]) -> list[dict]:
        if self._driver is None:
            from memory_kernel.serve.answer import ServeConfig, open_driver

            cfg = ServeConfig(group="", neo4j_uri=self.uri, neo4j_user=self.user, neo4j_password=self._password)
            self._driver = open_driver(cfg)
        async with self._driver.session() as session:
            result = await session.run(cypher, dict(params))
            return [dict(record) async for record in result]

    async def close(self) -> None:
        driver, self._driver = self._driver, None
        if driver is not None:
            await driver.close()


class EmptyReader:
    """A runtime ``Reader`` over an empty graph: every query returns no row (``mk plan`` without Neo4j)."""

    async def rows(self, cypher: str, params: Mapping[str, Any]) -> list[dict]:
        return []


def neo4j_target(boot: Mapping[str, Any], overrides: Mapping[str, Any]) -> tuple[str, str, str | None]:
    """``(uri, user, password)`` of a graph: ``overrides``, else the bootstrap config ``boot``'s address (empty for a
    bonfire that lives on the graph), else the defaults; the password from ``overrides`` or the process environment."""
    from memory_kernel.cli import resolve_secrets

    secrets = resolve_secrets(env_file=None, container=None).values
    uri = overrides.get("neo4j_uri") or boot.get("neo4j_uri") or DEFAULT_NEO4J_URI
    user = overrides.get("neo4j_user") or boot.get("neo4j_user") or DEFAULT_NEO4J_USER
    password = overrides.get("neo4j_password") or secrets.get("neo4j_password")
    return str(uri), str(user), password


# -- request bodies ---------------------------------------------------------------------------------------------------

TURN_FIELDS = frozenset({"text", "speaker", "addressee", "participants", "chat_id", "message_id", "at"})


def now_iso() -> str:
    """The server's current UTC time, ISO 8601 (a turn's ``at`` when the request leaves it out)."""
    return datetime.now(UTC).isoformat()


#: request caps (a longer field is a 422): the turn's text, any id (speaker, addressee, participant, chat, message,
#: teacher, confirmer, step), the participants, a teach's forms or body (entries and each entry's length)
MAX_TEXT = 4000
MAX_ID = 256
MAX_PARTICIPANTS = 256
MAX_TEACH_ENTRIES = 32
MAX_TEACH_ENTRY = 500
MAX_COMMITTED = 256
MAX_CATALOGUE = 1_000_000  # bytes


def unencodable(value: Any) -> bool:
    """Whether any string in ``value`` (a JSON body: keys too) does not encode as UTF-8 (a lone surrogate, which JSON
    can carry as an escape): such a string could never be read, answered or written back."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return True
        return False
    if isinstance(value, dict):
        return any(unencodable(k) or unencodable(v) for k, v in value.items())
    if isinstance(value, list):
        return any(unencodable(v) for v in value)
    return False


UNENCODABLE = "the body holds a string that is not valid UTF-8 (a lone surrogate)"


def _too_long(key: str, value: Any, cap: int) -> str | None:
    return f'"{key}" is longer than {cap} characters' if isinstance(value, str) and len(value) > cap else None


def parse_turn(body: Any) -> Turn | str:
    """The ``Turn`` a plan request names, or why the body is refused (422). ``text`` (at most ``MAX_TEXT``
    characters) and ``chat_id`` are required strings; ``speaker``, ``addressee`` and ``message_id`` strings or absent;
    ``participants`` a list of at most ``MAX_PARTICIPANTS`` strings; every id at most ``MAX_ID`` characters; ``at`` an
    ISO 8601 timestamp, the server's UTC time when absent. Any other field, or a string that is not valid UTF-8, is
    refused (the turn window is the runtime's, never the caller's)."""
    if not isinstance(body, dict):
        return "the body must be a JSON object of turn fields"
    if unencodable(body):
        return UNENCODABLE
    unknown = sorted(set(body) - TURN_FIELDS)
    if unknown:
        return f"unknown turn fields {unknown} (a turn is {sorted(TURN_FIELDS)})"
    text, chat_id = body.get("text"), body.get("chat_id")
    if not isinstance(text, str) or not text.strip():
        return '"text" must be a non-empty string'
    if not isinstance(chat_id, str) or not chat_id:
        return '"chat_id" must be a non-empty string'
    for key in ("speaker", "addressee", "message_id"):
        if body.get(key) is not None and not isinstance(body[key], str):
            return f'"{key}" must be a string when given'
    participants = body.get("participants", [])
    if not isinstance(participants, list) or not all(isinstance(p, str) for p in participants):
        return '"participants" must be a list of strings'
    if len(participants) > MAX_PARTICIPANTS:
        return f'"participants" holds more than {MAX_PARTICIPANTS} names'
    long = _too_long("text", text, MAX_TEXT) or next(
        (
            reason
            for key in ("chat_id", "speaker", "addressee", "message_id")
            if (reason := _too_long(key, body.get(key), MAX_ID))
        ),
        None,
    )
    if long is None and any(len(p) > MAX_ID for p in participants):
        long = f"a participant is longer than {MAX_ID} characters"
    if long is not None:
        return long
    at = body.get("at")
    if at is None:
        at = now_iso()
    elif not isinstance(at, str) or len(at) > MAX_ID or not _iso(at):
        return '"at" must be an ISO 8601 timestamp when given'
    return Turn(
        text=text,
        speaker=body.get("speaker"),
        addressee=body.get("addressee"),
        participants=tuple(participants),
        chat_id=chat_id,
        message_id=body.get("message_id"),
        at=at,
    )


def _iso(text: str) -> bool:
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def parse_plan(body: Any) -> tuple[Turn, list[dict[str, Any]], list[dict[str, Any]]] | str:
    """A plan request: a turn (``parse_turn``) with an optional ``committed`` list of ``{tool, args, value}`` (at most
    ``MAX_COMMITTED``): effects already done in the world; and an optional ``context`` list of at most
    ``frames.MAX_CONTEXT`` Frame objects (``frames.context_frames``; names and role names at most ``MAX_ID``
    characters, role texts at most ``MAX_TEXT``): the open frames of the turn before, which a follow-up composes
    with. Returns ``(turn, committed, context)``, or why the body is refused."""
    if unencodable(body):
        return UNENCODABLE
    if not isinstance(body, dict):
        return "the body must be a JSON object of turn fields"
    entries: tuple[dict[str, Any], ...] = ()
    if "committed" in body:
        committed = body["committed"]
        if isinstance(committed, list) and len(committed) > MAX_COMMITTED:
            return f'"committed" holds more than {MAX_COMMITTED} entries'
        try:
            entries = committed_entries(committed)
        except ValueError as exc:
            return f'"committed": {exc}'
    frames: tuple[dict[str, Any], ...] = ()
    if "context" in body:
        try:
            frames = context_frames(body["context"])
        except ValueError as exc:
            return f'"context" {exc}'
        if (reason := _long_frame(frames)) is not None:
            return reason
    turn = parse_turn({k: v for k, v in body.items() if k not in ("committed", "context")})
    return turn if isinstance(turn, str) else (turn, list(entries), list(frames))


def _long_frame(frames: Sequence[Mapping[str, Any]]) -> str | None:
    """Why a context frame passes a cap (its name, construction, ``composed_with`` or a role name past ``MAX_ID``, a
    role's text past ``MAX_TEXT``), or None."""
    for index, frame in enumerate(frames):
        ids = [frame["name"], frame.get("construction"), frame.get("composed_with"), *frame["roles"]]
        if any(isinstance(v, str) and len(v) > MAX_ID for v in ids):
            return f'"context" frame {index} holds a name longer than {MAX_ID} characters'
        if any(isinstance(v, str) and len(v) > MAX_TEXT for v in frame["roles"].values()):
            return f'"context" frame {index} holds a role text longer than {MAX_TEXT} characters'
    return None


def parse_results(body: Any) -> list[dict[str, Any]] | str:
    """A resume body ``{"results": [{"step_id", "tool", "args", "verdict", "value", "detail"?}, ...]}`` (non-empty; the
    verdict one of ``VERDICTS``; step ids at most ``MAX_ID`` characters; every string valid UTF-8), or why not."""
    if unencodable(body):
        return UNENCODABLE
    results = body.get("results") if isinstance(body, dict) else None
    shape = f"{{step_id: <str>, tool: <str>, args: <object>, verdict: one of {', '.join(VERDICTS)}, value, detail?}}"
    if not isinstance(results, list) or not results:
        return f'the body must be {{"results": [{shape}, ...]}} (non-empty)'
    for row in results:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("step_id"), str)
            or not isinstance(row.get("tool"), str)
            or not isinstance(row.get("args"), dict)
            or row.get("verdict") not in VERDICTS
            or not (row.get("detail") is None or isinstance(row.get("detail"), str))
        ):
            return f"a result is {shape}, got {row!r}"
        if len(row["step_id"]) > MAX_ID:
            return f"a step id is longer than {MAX_ID} characters"
    return results


def _body_entry(entry: Any) -> bool:
    """Whether a teach body entry is a clause (a string) or a structured step ``{"call": str, "args"?: {str: str}}``
    within the caps (a clause, a value: ``MAX_TEACH_ENTRY``; the tool, a parameter: ``MAX_ID``; ``MAX_TEACH_ENTRIES``
    parameters)."""
    if isinstance(entry, str):
        return len(entry) <= MAX_TEACH_ENTRY
    if not isinstance(entry, dict) or set(entry) - {"call", "args"}:
        return False
    call, args = entry.get("call"), entry.get("args", {})
    if not isinstance(call, str) or not call.strip() or len(call) > MAX_ID:
        return False
    return (
        isinstance(args, dict)
        and len(args) <= MAX_TEACH_ENTRIES
        and all(isinstance(v, str) and len(k) <= MAX_ID and len(v) <= MAX_TEACH_ENTRY for k, v in args.items())
    )


def parse_teach(body: Any) -> TeachRequest | str:
    """A ``TeachRequest`` from ``{"forms": [str], "body": [str | {"call": str, "args"?: {str: str}}], "defaults"?:
    {str: str}, "teacher": str, "chat_id": str, "replace"?: bool}``, or why the body is refused (422): at most
    ``MAX_TEACH_ENTRIES`` forms and body entries, a form, a clause or a step's value of at most ``MAX_TEACH_ENTRY``
    characters, ids (and a step's tool and parameter names) at most ``MAX_ID``, every string valid UTF-8. What a teach
    refuses by content (a form with no literal, a body clause that does not read, a step naming no tool) is a ``Draft``
    with problems, not a 422."""
    if not isinstance(body, dict):
        return "the body must be a JSON object"
    if unencodable(body):
        return UNENCODABLE
    known = {"forms", "body", "defaults", "teacher", "chat_id", "replace"}
    unknown = sorted(set(body) - known)
    if unknown:
        return f"unknown teach fields {unknown} (a teach is {sorted(known)})"
    forms, steps = body.get("forms"), body.get("body")
    if not isinstance(forms, list) or not forms or not all(isinstance(v, str) for v in forms):
        return '"forms" must be a non-empty list of strings'
    if len(forms) > MAX_TEACH_ENTRIES or any(len(v) > MAX_TEACH_ENTRY for v in forms):
        return f'"forms" holds at most {MAX_TEACH_ENTRIES} entries of at most {MAX_TEACH_ENTRY} characters'
    if not isinstance(steps, list) or not steps or len(steps) > MAX_TEACH_ENTRIES:
        return f'"body" must be a non-empty list of at most {MAX_TEACH_ENTRIES} entries'
    if not all(_body_entry(v) for v in steps):
        return (
            f'"body" holds clauses of at most {MAX_TEACH_ENTRY} characters or structured steps {{"call": <tool>, '
            f'"args": {{<param>: <value>}}}} (a tool and parameter names of at most {MAX_ID} characters, at most '
            f"{MAX_TEACH_ENTRIES} parameters, values strings of at most {MAX_TEACH_ENTRY} characters)"
        )
    defaults = body.get("defaults", {})
    if not isinstance(defaults, dict) or not all(isinstance(v, str) for v in defaults.values()):
        return '"defaults" must be an object of slot name -> text'
    if len(defaults) > MAX_TEACH_ENTRIES or any(
        len(k) > MAX_ID or len(v) > MAX_TEACH_ENTRY for k, v in defaults.items()
    ):
        return f'"defaults" holds at most {MAX_TEACH_ENTRIES} slots of at most {MAX_TEACH_ENTRY} characters'
    for key in ("teacher", "chat_id"):
        if not isinstance(body.get(key), str) or not body[key].strip():
            return f'"{key}" must be a non-empty string'
        if (reason := _too_long(key, body[key], MAX_ID)) is not None:
            return reason
    replace = body.get("replace", False)
    if not isinstance(replace, bool):
        return '"replace" must be true or false'
    return TeachRequest(
        forms=list(body["forms"]),
        body=list(steps),
        defaults=dict(defaults),
        teacher=body["teacher"],
        chat_id=body["chat_id"],
        replace=replace,
    )


def parse_confirm(body: Any) -> str | tuple[str]:
    """``(by,)`` from a confirm body ``{"by": <the teacher>}``, or why the body is refused (422)."""
    if unencodable(body):
        return UNENCODABLE
    by = body.get("by") if isinstance(body, dict) else None
    if not isinstance(by, str) or not by.strip():
        return 'the body must be {"by": <the teacher>}'
    if (reason := _too_long("by", by, MAX_ID)) is not None:
        return reason
    return (by,)


def draft_json(draft: Draft) -> dict[str, Any]:
    """A teach's answer: the draft id, the staged spec rows, the problems, and ``competes`` (form text -> the frames the
    live grammar reads that form's own renderings as, before the draft: ``Draft.competes``)."""
    return {
        "draft_id": draft.draft_id,
        "specs": [s.to_json() for s in draft.specs],
        "problems": list(draft.problems),
        "competes": {form: [dict(f) for f in frames] for form, frames in draft.competes.items()},
    }


def refusal(reason: str) -> dict[str, Any]:
    return {"engine": "kernel", "degraded": True, "reason": reason}


# -- the runtimes -----------------------------------------------------------------------------------------------------


class EnginesBusy(RuntimeError):
    """Every runtime engine is in use: none can be stopped for another bonfire now."""


@dataclass
class _Slot:
    """One bonfire's runtime and its engine's state (``port`` None while its engine is stopped)."""

    runtime: Any
    bonfire_id: str
    port: int | None = None
    used: float = 0.0
    active: int = 0


RuntimeFactory = Callable[[str, Mapping[str, Any], Any], Any]
EngineFactory = Callable[[int], Any]


class Runtimes:
    """The runtimes of ``pool``'s bonfires (see the module docstring). ``runtime_factory(bonfire_id, meta, engine)``
    makes a bonfire's runtime (default: a ``Runtime`` over its graph); ``engine_factory(port)`` an unstarted engine
    (default: a ``RuntimeEngine`` on that port)."""

    def __init__(
        self,
        pool: Any,
        settings: RuntimeSettings | None = None,
        *,
        runtime_factory: RuntimeFactory | None = None,
        engine_factory: EngineFactory | None = None,
        writer_factory: Callable[[str], NarrowWriter] | None = None,
    ) -> None:
        self.pool = pool
        self.settings = settings if settings is not None else RuntimeSettings.from_env()
        self.runtime_factory = runtime_factory or self._default_runtime
        self.engine_factory = engine_factory or _default_engine
        self.writer_factory = writer_factory
        self._driver: Any = None
        self._driver_lock = threading.Lock()
        self._slots: dict[str, _Slot] = {}
        self._making: dict[str, asyncio.Lock] = {}
        self._starting: asyncio.Lock | None = None
        self._teaching: dict[str, int] = {}
        #: catalogue reloads in flight per bonfire (a subset of ``_teaching``: the mark ends with the last one), and
        #: the bonfires reloaded since start
        self._reloading: Counter[str] = Counter()
        self._reloaded: set[str] = set()
        self._idle: dict[str, asyncio.Event] = {}

    # -- runtimes and engines ------------------------------------------------------------------------------------------

    def sync_driver(self) -> Any:
        """The sync Neo4j driver of the narrow writer and the rolesets read, opened on first use."""
        with self._driver_lock:
            if self._driver is None:
                from memory_kernel.runtime.graph_store import open_sync_driver

                uri, user, password = neo4j_target({}, self.pool.overrides)
                if password is None:
                    raise ValueError(
                        "no Neo4j password configured (NEO4J_PASSWORD, --env-file or --secrets-from-container)"
                    )
                self._driver = open_sync_driver(uri, user, password)
            return self._driver

    def writer(self, bonfire_id: str) -> NarrowWriter:
        """The bonfire's narrow writer (``writer_factory``'s, else over ``sync_driver``)."""
        if self.writer_factory is not None:
            return self.writer_factory(bonfire_id)
        from memory_kernel.runtime.graph_store import NarrowWriter

        return NarrowWriter(self.sync_driver(), bonfire_id)

    def _default_runtime(self, bonfire_id: str, meta: Mapping[str, Any], engine: Any) -> Any:
        from memory_kernel.runtime.api import Runtime, default_registry
        from memory_kernel.runtime.catalogue import parse_catalogue
        from memory_kernel.runtime.lexicon import Lexicon, QuestionMap, load_rolesets
        from memory_kernel.runtime.store import ConstructionStore

        group = str(meta.get("group_id") or bonfire_id)
        uri, user, password = neo4j_target({}, self.pool.overrides)
        raw = meta.get("catalogue_json")
        # absent: no catalogue (as a missing catalogue file was); a bad one: the runtime does not load
        catalogue = parse_catalogue(raw.encode("utf-8"), "catalogue") if raw else None
        writer = self.writer(bonfire_id)
        return Runtime(
            bonfire_id,
            engine,
            GraphReader(uri, user, password),
            default_registry(catalogue.tools if catalogue else ()),
            group_id=group,
            store=ConstructionStore(writer),
            lexicon=Lexicon(load_rolesets(self.sync_driver(), group), QuestionMap.load()),
            recall=self.recall_for(bonfire_id),
            cards=catalogue.cards if catalogue else (),
        )

    def recall_for(self, bonfire_id: str) -> Callable[[str], Awaitable[dict]]:
        """The runtime's ``recall`` (the fallback on an empty result): the bonfire's ``/context`` from the CACHES ONLY
        (``Pool.context(paid=False)``), so a plan never makes a paid call. A context that did not come back (a cache
        miss: "no cached context", or any other failure) raises ``LookupError`` with its reason: the plan then carries
        the diagnostic ``recall unavailable: <reason>`` and no ``recall`` result. The turn's ``speaker`` resolves the text's
        first-person words, as for ``/context``."""

        async def recall(text: str, speaker: str | None = None) -> dict:
            status, body = await self.pool.context(bonfire_id, text, speaker, paid=False)
            if status != 200 or (body.get("degraded") and not body.get("context")):
                raise LookupError(str(body.get("reason") or "no cached context"))
            return body

        return recall

    async def _slot(self, bonfire_id: str, meta: Mapping[str, Any]) -> _Slot:
        """The bonfire's slot, its runtime made on first use (off the event loop: it reads the store and the
        lexicon), with an unstarted engine on the first port of the range."""
        slot = self._slots.get(bonfire_id)
        if slot is not None:
            return slot
        async with self._making.setdefault(bonfire_id, asyncio.Lock()):
            slot = self._slots.get(bonfire_id)
            if slot is None:
                engine = self.engine_factory(self.settings.ports[0])
                runtime = await asyncio.to_thread(self.runtime_factory, bonfire_id, meta, engine)
                slot = self._slots[bonfire_id] = _Slot(runtime, bonfire_id)
        return slot

    async def _engaged(self, bonfire_id: str, meta: Mapping[str, Any]) -> _Slot:
        """The bonfire's slot with its engine running and one more user in flight (``_release`` after). Starting an
        engine past the cap first stops the least recently used idle one; ``EnginesBusy`` when none is idle, or when
        its stop fails (it stays counted: the cap is never exceeded). A port something else holds (``PortBusy``) is
        skipped for the next free one of the range."""
        slot = await self._slot(bonfire_id, meta)
        if slot.port is not None:  # running: no await between the check and the claim
            slot.active += 1
            slot.used = time.monotonic()
            return slot
        if self._starting is None:
            self._starting = asyncio.Lock()
        async with self._starting:
            if self._slots.get(bonfire_id) is not slot:  # a catalogue PUT popped it while this call waited
                raise EnginesBusy(BUSY_RELOADING)
            if slot.port is None:
                running = [s for s in self._slots.values() if s.port is not None]
                if len(running) >= self.settings.max_runtime_engines:
                    idle = [s for s in running if s.active == 0]
                    if not idle:
                        raise EnginesBusy(f"{ENGINES_BUSY} ({self.settings.max_runtime_engines})")
                    victim = min(idle, key=lambda s: s.used)
                    held, victim.port = victim.port, None  # claimed before the await: no request starts on it
                    try:
                        await asyncio.to_thread(victim.runtime.stop)
                    except Exception as exc:
                        victim.port = held  # it may still run: keep it counted
                        why = self.pool.reason(exc)
                        log.warning("bonfire %s: stopping its idle engine failed: %s", victim.bonfire_id, why)
                        raise EnginesBusy(
                            f"{ENGINES_BUSY}: stopping the idle engine of {victim.bonfire_id!r} failed: {why}"
                        ) from exc
                slot.active += (
                    1  # a cold start is a call in flight: a catalogue PUT waits for it instead of dropping the slot
                )
                try:
                    slot.port = await self._start(slot)
                    if self._slots.get(slot.bonfire_id) is not slot:  # replaced meanwhile: nothing may keep this engine
                        slot.port = None
                        await asyncio.to_thread(slot.runtime.stop)
                        raise EnginesBusy(f"{ENGINES_BUSY}: the bonfire's runtime was replaced while it started")
                except BaseException:
                    slot.active -= 1
                    raise
                slot.used = time.monotonic()
                return slot
        slot.active += 1
        slot.used = time.monotonic()
        return slot

    async def _start(self, slot: _Slot) -> int:
        """Start the slot's engine on the first free port of the range that nothing else holds; the port."""
        from memory_kernel.runtime.engine import PortBusy

        taken = {s.port for s in self._slots.values() if s.port is not None}
        busy: PortBusy | None = None
        for port in [p for p in self.settings.ports if p not in taken]:
            if slot.runtime.engine.port != port:
                slot.runtime.use_engine(self.engine_factory(port))
            try:
                await asyncio.to_thread(slot.runtime.start)
            except PortBusy as exc:  # a foreign listener (or a live owner's engine): try the next port
                await asyncio.to_thread(slot.runtime.stop)
                log.warning("bonfire %s: port %s is held by another process: trying the next", slot.bonfire_id, port)
                busy = exc
                continue
            except BaseException:
                await asyncio.to_thread(slot.runtime.stop)
                raise
            return port
        raise busy or EnginesBusy(f"{ENGINES_BUSY}: no free port in {self.settings.runtime_ports}")

    @staticmethod
    def _release(slot: _Slot) -> None:
        slot.active -= 1
        slot.used = time.monotonic()

    def engines(self) -> dict[str, int]:
        """Bonfire id -> the port of its running engine."""
        return {b: s.port for b, s in self._slots.items() if s.port is not None}

    # -- teaching marks ------------------------------------------------------------------------------------------------

    def _teach_begin(self, bonfire_id: str) -> None:
        self._teaching[bonfire_id] = self._teaching.get(bonfire_id, 0) + 1
        self._idle.setdefault(bonfire_id, asyncio.Event()).clear()

    def _teach_end(self, bonfire_id: str) -> None:
        self._teaching[bonfire_id] -= 1
        if not self._teaching[bonfire_id]:
            del self._teaching[bonfire_id]
            self._idle[bonfire_id].set()

    def _busy(self, bonfire_id: str) -> str:
        """The diagnostic of a plan refused behind a mark: a catalogue reload or a teach."""
        return BUSY_RELOADING if self._reloading[bonfire_id] else BUSY_TEACHING

    # -- the routes ----------------------------------------------------------------------------------------------------

    async def plan(
        self,
        bonfire_id: str,
        meta: Mapping[str, Any],
        turn: Turn,
        committed: Sequence[Mapping[str, Any]] = (),
        context: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[int, dict[str, Any]]:
        """200 with the ``TurnPlan``. A plan that cannot be read now (the bonfire teaches past ``teach_wait_ms``,
        every engine is in use, the engine or the runtime fails) is an ``unread`` plan naming why: the turn falls
        back to today's path. A runtime that does not start (``runtime.plan_unavailable``) or raises
        (``runtime.plan_failed``) is also logged, with the plan's reason (secrets redacted, no traceback); all engines
        in use is not.
        ``committed``: effects already done in the world; ``context``: the open frames of the turn before
        (``Runtime.plan``)."""
        if self._teaching.get(bonfire_id):
            with contextlib.suppress(TimeoutError):  # a short teach is waited out: the turn is read once it ends
                await asyncio.wait_for(self._idle[bonfire_id].wait(), self.settings.teach_wait_ms / 1000)
        if self._teaching.get(bonfire_id):
            return 200, unread_plan(bonfire_id, turn, [self._busy(bonfire_id)]).to_json()
        try:
            slot = await self._engaged(bonfire_id, meta)
        except EnginesBusy as exc:
            return 200, unread_plan(bonfire_id, turn, [str(exc)]).to_json()
        except Exception as exc:  # a store that does not load, an engine that does not start: never a 500
            reason = self.pool.reason(exc)
            log.error("runtime.plan_unavailable: bonfire %s: %s", bonfire_id, reason)
            return 200, unread_plan(bonfire_id, turn, [f"runtime unavailable: {reason}"]).to_json()
        if self._teaching.get(bonfire_id):  # a teach began while the engine started: never read under it
            self._release(slot)
            return 200, unread_plan(bonfire_id, turn, [self._busy(bonfire_id)]).to_json()
        try:
            given = {"committed": list(committed), "context": list(context)}
            plan = await slot.runtime.plan(turn, **{k: v for k, v in given.items() if v})
        except Exception as exc:
            reason = self.pool.reason(exc)
            log.error("runtime.plan_failed: bonfire %s: %s", bonfire_id, reason)
            return 200, unread_plan(bonfire_id, turn, [f"runtime error: {reason}"]).to_json()
        finally:
            self._release(slot)
        return 200, plan.to_json()

    async def resume(
        self, bonfire_id: str, plan_id: str, results: Sequence[Mapping[str, Any]]
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        """200 with the ``TurnPlan``; 404 unknown or expired plan; 409 a resume that cannot apply; 503 (with
        ``Retry-After``) while the bonfire still teaches after ``resume_wait_s``. Never starts an engine."""
        return await self._continue(bonfire_id, plan_id, "resume", list(results))

    async def retry(self, bonfire_id: str, plan_id: str) -> tuple[int, dict[str, Any], dict[str, str]]:
        """``Runtime.retry``: the step a verdict refused is issued again. Answers as ``resume`` does."""
        return await self._continue(bonfire_id, plan_id, "retry")

    async def _continue(
        self, bonfire_id: str, plan_id: str, call: str, *args: Any
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        if self._teaching.get(bonfire_id):
            try:
                await asyncio.wait_for(self._idle[bonfire_id].wait(), self.settings.resume_wait_s)
            except TimeoutError:
                retry = {"Retry-After": str(RETRY_AFTER_S)}
                return 503, refusal(f"bonfire {bonfire_id!r} is teaching: {call} again later"), retry
        slot = self._slots.get(bonfire_id)
        if slot is None:
            return (
                404,
                refusal(self._no_plan(plan_id, bonfire_id)),
                {},
            )
        slot.active += 1  # in flight: a catalogue reload waits for it
        try:
            plan = await getattr(slot.runtime, call)(plan_id, *args)
        except UnknownPlan as exc:
            return 404, refusal(str(exc)), {}
        except ResumeError as exc:
            return 409, refusal(str(exc)), {}
        finally:
            self._release(slot)
        return 200, plan.to_json(), {}

    def _no_plan(self, plan_id: str, bonfire_id: str) -> str:
        if bonfire_id in self._reloaded:
            return (
                f"no plan {plan_id!r}: the plan expired or the catalogue of {bonfire_id!r} was reloaded "
                "- re-plan the turn with committed"
            )
        return f"no plan {plan_id!r}: this server made no plan for {bonfire_id!r} since it started"

    async def teach(self, bonfire_id: str, meta: Mapping[str, Any], req: TeachRequest) -> tuple[int, dict[str, Any]]:
        """200 with the ``Draft`` (a refused teach is a draft with problems); 409 a ``TeachError``; 503 when the
        engine cannot run it now. Marks the bonfire as teaching before anything else (its plans go unread)."""
        return await self._teaching_call(bonfire_id, meta, "teach", req)

    async def confirm(
        self, bonfire_id: str, meta: Mapping[str, Any], draft_id: str, by: str
    ) -> tuple[int, dict[str, Any]]:
        """200 with the live specs; 403 a confirm by someone other than the teacher; 409 any other ``TeachError``;
        503 when the engine cannot run it now."""
        return await self._teaching_call(bonfire_id, meta, "confirm", draft_id, by)

    async def _teaching_call(self, bonfire_id: str, meta: Mapping[str, Any], call: str, *args: Any) -> tuple[int, dict]:
        self._teach_begin(bonfire_id)  # synchronous, before the first await: a plan arriving now sees it
        try:
            try:
                slot = await self._engaged(bonfire_id, meta)
            except Exception as exc:
                return 503, refusal(f"runtime unavailable: {self.pool.reason(exc)}")
            try:
                out = await getattr(slot.runtime, call)(*args)
            except NotTeacher as exc:
                return 403, refusal(str(exc))
            except TeachError as exc:
                return 409, refusal(str(exc))
            except Exception as exc:  # the engine failed under the teach, a store write failed: never a traceback
                return 503, refusal(f"runtime: {self.pool.reason(exc)}")
            finally:
                self._release(slot)
        finally:
            self._teach_end(bonfire_id)
        if call == "teach":
            return 200, draft_json(out)
        warnings = sorted({w for s in out for w in s.provenance.get(WARNINGS, [])})
        return 200, {
            "draft_id": args[0],
            "specs": [s.to_json() for s in out],
            **({"warnings": warnings} if warnings else {}),
        }

    async def catalogue(self, bonfire_id: str) -> tuple[int, dict[str, Any]]:
        """200 ``{sha256, tools}``: the bonfire's catalogue as stored, read now through the narrow writer (``sha256``
        null and no tools when none was ever put). 503 when the graph does not answer or the stored one does not load."""
        from memory_kernel.runtime.catalogue import CatalogueError, parse_catalogue

        try:
            stored = await asyncio.to_thread(self.writer(bonfire_id).catalogue)
        except Exception as exc:
            return 503, refusal(f"the catalogue could not be read: {self.pool.reason(exc)}")
        if not stored:
            return 200, {"sha256": None, "tools": []}
        try:
            raw = stored.encode("utf-8")
            cat = parse_catalogue(raw, "catalogue")  # the same bytes give the sha256 and the tools
            rows = json.loads(raw)["tools"]
        except (CatalogueError, ValueError, KeyError) as exc:
            return 503, refusal(f"the installed catalogue does not load: {exc}")
        return 200, {"sha256": cat.sha256, "tools": rows}

    async def set_catalogue(self, bonfire_id: str, meta: Mapping[str, Any], raw: bytes) -> tuple[int, dict[str, Any]]:
        """Validate ``raw`` and set it as the bonfire's ``catalogue_json``, then reload its runtime. 400 with the
        ``CatalogueError`` text (the previous catalogue stays). The bonfire is marked as teaching while its runtime
        reloads (its plans wait, as behind a teach); the reload waits for the plans in flight, stops the engine and
        drops the runtime: the next plan makes a new one (its plans and turn window are lost)."""
        from memory_kernel.runtime.catalogue import CatalogueError, parse_catalogue

        try:
            cat = parse_catalogue(raw, "catalogue")
        except CatalogueError as exc:
            return 400, refusal(str(exc))
        self._teach_begin(bonfire_id)
        self._reloading[bonfire_id] += 1
        try:
            slot = self._slots.get(bonfire_id)
            if slot is not None:  # new plans wait behind the mark; the ones in flight finish first
                deadline = time.monotonic() + self.settings.teach_wait_ms / 1000
                while slot.active:
                    if time.monotonic() >= deadline:
                        return 503, refusal("the bonfire's runtime is busy: the catalogue is unchanged, retry later")
                    await asyncio.sleep(0.01)
            try:
                await asyncio.to_thread(self.writer(bonfire_id).set_catalogue, raw.decode("utf-8"))
            except Exception as exc:
                return 503, refusal(f"the catalogue could not be installed: {self.pool.reason(exc)}")
            self._reloaded.add(bonfire_id)
            if slot is not None:
                self._slots.pop(bonfire_id, None)
                slot.port = None
                try:
                    await asyncio.to_thread(slot.runtime.stop)
                except Exception as exc:
                    log.warning("bonfire %s: stopping its runtime engine failed: %s", bonfire_id, self.pool.reason(exc))
        finally:
            self._reloading[bonfire_id] -= 1
            if not self._reloading[bonfire_id]:
                del self._reloading[bonfire_id]
            self._teach_end(bonfire_id)
        return 200, {"sha256": cat.sha256, "tools": json.loads(raw)["tools"]}  # exactly what a GET returns now

    async def constructions(self, bonfire_id: str, meta: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """200 with the bonfire's stored construction rows (live and staged; the base set is not stored)."""
        try:
            slot = await self._slot(bonfire_id, meta)
        except Exception as exc:
            return 503, refusal(f"runtime: {self.pool.reason(exc)}")
        rows = [spec.to_json() for spec in slot.runtime.store.all()]
        return 200, {"engine": "kernel", "bonfire_id": bonfire_id, "constructions": rows}

    async def close(self) -> None:
        """Stop every engine (by pid) and close the graph readers; a slot that fails is logged and the others go on."""
        for slot in list(self._slots.values()):
            slot.port = None
            try:
                await asyncio.to_thread(slot.runtime.stop)
            except Exception as exc:
                log.warning(
                    "bonfire %s: stopping its runtime engine failed: %s", slot.bonfire_id, self.pool.reason(exc)
                )
            reader = getattr(slot.runtime, "reader", None)
            if isinstance(reader, GraphReader):
                try:
                    await reader.close()
                except Exception as exc:
                    log.warning(
                        "bonfire %s: closing its graph reader failed: %s", slot.bonfire_id, self.pool.reason(exc)
                    )
        driver, self._driver = self._driver, None
        if driver is not None:
            try:
                await asyncio.to_thread(driver.close)
            except Exception as exc:
                log.warning("closing the narrow writer's driver failed: %s", self.pool.reason(exc))


def _default_engine(port: int) -> Any:
    from memory_kernel.runtime.engine import RuntimeEngine

    return RuntimeEngine(port)


__all__ = [
    "BUSY_RELOADING",
    "BUSY_TEACHING",
    "ENGINES_BUSY",
    "EmptyReader",
    "MAX_CATALOGUE",
    "GraphReader",
    "RuntimeSettings",
    "Runtimes",
    "draft_json",
    "neo4j_target",
    "parse_confirm",
    "parse_plan",
    "parse_results",
    "parse_teach",
    "parse_turn",
]
