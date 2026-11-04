"""``mk``: build a conversation's memory, ask it, inspect a build.

    mk build CONVERSATION_JSON --workdir DIR --group G [--sample-id ID] [--purpose-file F | --purpose TEXT] ...
    mk append BONFIRE_ID SESSION_JSON --ops-uri URI --scratch-dir DIR [--owner O] [--on-built skip] [secrets / flags]
    mk build ... --seed --ops-uri URI [--spend-ceiling-usd X]            (kernel 2.0: seed a bonfire into its group)
    mk bonfire empty GROUP [--neo4j-uri URI]                             (delete a group with no BonfireMeta, in batches)
    mk bonfire empty GROUP --rebuild --ops-uri URI                       (rebuild: delete a sealed group too; record -> seeding)
    mk bonfire release ID --ops-uri URI                                  (clear a lease that is not live)
    mk bonfire ceiling ID USD --ops-uri URI                              (set a bonfire's spend ceiling in kernel_ops)
    mk ask BONFIRE_ID "QUESTION" [--neo4j-uri U] [--neo4j-user U] [--json]
    mk serve [--neo4j-uri U] [--neo4j-user U] [--ops-uri URI] [--host H] [--port N] [--no-allow-paid]
                                                                      (warm per-bonfire context server over Neo4j; [serve] extra)
    mk plan BONFIRE_ID "TEXT" --speaker NAME [--chat ID]             (debug: the runtime grammar's TurnPlan as JSON)
    mk catalogue install BONFIRE_ID FILE                               (set a bonfire's tool catalogue)
    mk keywords                                                        (the runtime grammar's keyword glossary)
    mk inspect WORKDIR_CONV_ROOT                                       (read-only status)
    mk setup [--check [--verify]] [--offline --adopt-...] [--json]     (the pinned runtime data a build needs)
    mk slot serve --socket PATH --data-dir DIR [--engine-port N] [--parser-port N] [--rss-cap-gib G]
                                                                      (the warm capture slot: one engine + parser)
    mk slot status|stop --socket PATH

This is the only place that reads the SECRETS of ``mk build``, ``mk ask`` and ``mk serve``: the OpenRouter and Voyage keys and the
Neo4j password are resolved here once and passed down inside ``BuildConfig`` / ``ServeConfig`` (a stage child gets them
back through its own environment, see ``BuildConfig.from_json``). For every key the order is the process environment,
then the explicit ``--secrets-from-container`` opt-in, then ``--env-file`` (a stale ``.env`` never beats a running
container). The driver logs which source each key came from (never the value) in its ``driver-start`` event. Other
modules read non-secret settings from the environment themselves, for example ``MK_DATA_DIR`` (``setup_runtime``),
``TIKTOKEN_CACHE_DIR`` (``search.kg_context``), ``MK_FCG_PARSER_HOST`` (``construction.fcg.penelope_host``), the
temporal-deixis flag (``knowledge.temporal_deixis``).
Exit codes: 0 = complete / through, 2 = stopped / refused.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NamedTuple

REFUSED = 2


def _refuse(reason: str) -> int:
    print(json.dumps({"refused": reason}))
    return REFUSED


# -- secrets -----------------------------------------------------------------------------------------------------------

SECRET_NAMES: dict[str, tuple[str, ...]] = {
    "openrouter_api_key": ("OPENROUTER_API_KEY",),
    "voyage_api_key": ("VOYAGE_API_KEY",),
    "neo4j_password": ("NEO4J_PASSWORD",),
}


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def container_env(name: str) -> dict[str, str]:
    """Explicit opt-in: the environment of a running container (``docker exec NAME env``).

    A failed ``docker exec`` is a ``ValueError`` (a refusal) that names the container and the exit status ONLY. The
    command's stdout IS the container's environment, i.e. its secrets, and its stderr is unvetted: neither goes into
    the message. ``check=False`` and a status test, rather than ``CalledProcessError``, so that no exception carrying
    the output exists to chain from."""
    proc = subprocess.run(["docker", "exec", name, "env"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ValueError(f"cannot read the environment of container {name!r}: `docker exec` exited {proc.returncode}")
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


#: The source names ``resolve_secrets`` reports (and ``driver-start`` logs) per key. Precedence is this order.
SOURCE_ENV = "env"
SOURCE_CONTAINER = "container"
SOURCE_ENV_FILE = "env-file"


class ResolvedSecrets(NamedTuple):
    """``values``: field -> secret (None = unset). ``sources``: field -> SOURCE_* name (None = unset); never a value."""

    values: dict[str, str | None]
    sources: dict[str, str | None]


def resolve_secrets(
    *, env_file: Path | None, container: str | None, environ: Mapping[str, str] | None = None
) -> ResolvedSecrets:
    """Per field, the first source that sets one of the field's names wins; the source order is the process
    environment, then the container (only when ``container`` is given), then the env file. A live container beats a
    stale ``.env`` (a stale ``.env`` may carry an old Neo4j password)."""
    sources: list[tuple[str, Mapping[str, str]]] = [(SOURCE_ENV, os.environ if environ is None else environ)]
    if container is not None:
        sources.append((SOURCE_CONTAINER, container_env(container)))
    if env_file is not None:
        sources.append((SOURCE_ENV_FILE, read_env_file(env_file)))
    values: dict[str, str | None] = {}
    origin: dict[str, str | None] = {}
    for field_name, names in SECRET_NAMES.items():
        values[field_name], origin[field_name] = next(
            ((src[n], label) for label, src in sources for n in names if src.get(n)), (None, None)
        )
    return ResolvedSecrets(values, origin)


# -- build -------------------------------------------------------------------------------------------------------------


def read_purpose(path: Path) -> str:
    """``research_question`` of a JSON object that has it, else the file's raw text."""
    text = path.read_text()
    try:
        doc = json.loads(text)
    except ValueError:
        return text
    if isinstance(doc, dict) and doc.get("research_question"):
        return str(doc["research_question"])
    return text


def build_config(args: argparse.Namespace, *, sources_out: dict[str, str | None] | None = None):
    """The ``BuildConfig`` of parsed ``mk build`` args. ``sources_out`` (when given) receives, per secret field, the
    NAME of the source it came from (never the value); ``cmd_build`` hands it to the driver for ``driver-start``."""
    from memory_kernel import setup_runtime
    from memory_kernel.pipeline.config import BuildConfig, lexical_abspath, load_conversation

    conversation = lexical_abspath(args.conversation)
    root = lexical_abspath(args.root) if args.root else lexical_abspath(".")
    vault = lexical_abspath(args.vault) if args.vault else None
    data_dir = setup_runtime.resolve_data_dir(args.data_dir, os.environ)
    item, _ = load_conversation(conversation, args.sample_id)  # ConversationError (a ValueError) = refusal
    if bool(args.boundary_replay_archive) != bool(args.boundary_replay_scope):
        raise ValueError(
            "--boundary-replay-archive and --boundary-replay-scope go together (the replay gate is opt-in)"
        )
    sample_id = args.sample_id or item["sample_id"]
    purpose = None  # optional (kernel 1.8): a record in build-config.json, nothing reads it
    if args.purpose is not None:
        purpose = args.purpose
    elif args.purpose_file:
        purpose = read_purpose(lexical_abspath(args.purpose_file, base=root))
    secrets = resolve_secrets(
        env_file=lexical_abspath(args.env_file) if args.env_file else None, container=args.secrets_from_container
    )
    if sources_out is not None:
        sources_out.update(secrets.sources)

    def opt(p: str | None) -> Path | None:
        return lexical_abspath(p, base=root) if p else None

    return BuildConfig(
        sample_id=sample_id,
        conversation=conversation,
        group=args.group,
        purpose=purpose,
        ingest_pass=args.ingest_pass,
        bonfire_id=args.bonfire_id,
        workdir=lexical_abspath(args.workdir),
        root=root,
        vault=vault,
        data_dir=data_dir,
        frame_dirs=tuple(lexical_abspath(p, base=root) for p in args.frame_dir or ()),
        harness_root=lexical_abspath(args.harness_root) if args.harness_root else None,
        snapshot_pins=opt(args.snapshot_pins),
        freeze_manifest=opt(args.freeze_manifest),
        invariants_reference=opt(args.invariants_reference),
        recipe_manifest=opt(args.recipe_manifest),
        grammar_image_sha256=args.grammar_image_sha256,
        boundary_replay_archive=(
            (opt(args.boundary_replay_archive[0]), opt(args.boundary_replay_archive[1]))
            if args.boundary_replay_archive
            else None
        ),
        boundary_window_prefix=args.boundary_window_prefix,
        boundary_policy=args.boundary,
        referent_policy=args.referents,
        boundary_replay_scope=opt(args.boundary_replay_scope),
        embedding_cache=opt(args.embedding_cache),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=secrets.values["neo4j_password"],
        concurrent_group_prefix=args.concurrent_group_prefix,
        cap_usd=args.cap_usd,
        jev_concurrency=args.jev_concurrency,
        openrouter_api_key=secrets.values["openrouter_api_key"],
        voyage_api_key=secrets.values["voyage_api_key"],
        engine_port=args.engine_port,
        capture_shards=args.capture_shards,
        engine_binary=opt(args.engine_binary),
        engine_image=opt(args.engine_image),
        allow_unpinned_engine=args.allow_unpinned_engine,
        parser_url=args.parser_url,
        parser_port=args.parser_port,
        allow_remote_parser=args.allow_remote_parser,
        through=args.through,
        max_shift_share=args.max_shift_share,
        strict_gates=args.strict_gates,
        genesis=args.genesis,
    )


#: Printed at the end of ``mk build --help``. Keep it free of option-looking words: the gate reads the option list.
BUILD_EPILOG = (
    "Captures: a COMPLETE capture made elsewhere is read as is; an INCOMPLETE one cannot be resumed by mk (VOID: its "
    "manifest pins the code, the parser address and the snapshot of the run that made it, and mk checks them on "
    "every resume); start a new capture."
)


OPS_URI_ENV = "MK_OPS_URI"


def _amount(flag: str, text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise ValueError(f"{flag} {text!r} is not an amount") from None
    if not value.is_finite() or value < 0:
        raise ValueError(f"{flag} {text!r} is not an amount")
    return value


def open_ops(uri: str | None) -> Any:
    """A ``MongoKernelOps`` on ``uri`` (else ``MK_OPS_URI``). The URI may carry credentials: it is never printed."""
    from memory_kernel.ops.mongo import MongoKernelOps
    from memory_kernel.pipeline.bonfire import BonfireError

    uri = uri or os.environ.get(OPS_URI_ENV)
    if not uri:
        raise BonfireError(f"no kernel_ops address: pass --ops-uri or set {OPS_URI_ENV}")
    return MongoKernelOps.from_uri(uri)


def seed_bonfire(cfg: Any, args: argparse.Namespace, secret_sources: Mapping[str, str | None]) -> dict:
    """``mk build --seed``: ``graph_seed.seed`` of ``cfg`` on the bonfire's ``kernel_ops`` record."""
    from memory_kernel.pipeline import graph_seed

    ceiling = None if args.spend_ceiling_usd is None else _amount("--spend-ceiling-usd", args.spend_ceiling_usd)
    ops = open_ops(args.ops_uri)
    try:
        return graph_seed.seed(cfg, ops=ops, ceiling_usd=ceiling, secret_sources=secret_sources)
    finally:
        close = getattr(ops, "close", None)
        if callable(close):
            close()


def cmd_build(args: argparse.Namespace) -> int:
    from memory_kernel.pipeline.engine import RemoteParserRefused
    from memory_kernel.pipeline.runner import Stop, run_build

    secret_sources: dict[str, str | None] = {}
    try:
        cfg = build_config(args, sources_out=secret_sources)
        # runner.preflight refuses before any stage (and before any mkdir); driver-start logs the secret SOURCES
        out = seed_bonfire(cfg, args, secret_sources) if args.seed else run_build(cfg, secret_sources=secret_sources)
    except (ValueError, OSError, RemoteParserRefused, Stop) as exc:  # bad input -> exit 2; exit 1 = a genuine bug
        return _refuse(f"{type(exc).__name__}: {exc}" if isinstance(exc, OSError) else str(exc))
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0 if out.get("status") != "stopped" else REFUSED


# -- append ------------------------------------------------------------------------------------------------------------


def append_overrides(args: argparse.Namespace, *, sources_out: dict[str, str | None] | None = None) -> dict:
    """The ``BuildConfig`` fields ``mk append`` sets over the bootstrap's config: the secrets (resolved here, as for
    ``mk build``) and every runtime flag that was given. A flag left out keeps the bootstrap's value; the install paths
    (``bonfire.HOST_PATH_FIELDS``: data dir, engine, embedding cache, frame dirs, harness root, ...) keep it only where
    the path exists on this host, else take this host's default (``bonfire.host_paths``)."""
    from memory_kernel import setup_runtime
    from memory_kernel.pipeline.config import lexical_abspath

    secrets = resolve_secrets(
        env_file=lexical_abspath(args.env_file) if args.env_file else None, container=args.secrets_from_container
    )
    if sources_out is not None:
        sources_out.update(secrets.sources)
    out: dict = dict(secrets.values)
    if args.data_dir is not None:
        out["data_dir"] = setup_runtime.resolve_data_dir(args.data_dir, os.environ)
    paths = {
        "engine_binary": args.engine_binary,
        "engine_image": args.engine_image,
        "embedding_cache": args.embedding_cache,
    }
    out |= {k: lexical_abspath(v) for k, v in paths.items() if v is not None}
    values = {
        "parser_url": args.parser_url,
        "parser_port": args.parser_port,
        "engine_port": args.engine_port,
        "capture_shards": args.capture_shards,
        "jev_concurrency": args.jev_concurrency,
        "cap_usd": args.cap_usd,
        "neo4j_uri": args.neo4j_uri,
        "neo4j_user": args.neo4j_user,
    }
    out |= {k: v for k, v in values.items() if v is not None}
    out |= {k: True for k in ("allow_remote_parser", "allow_unpinned_engine") if getattr(args, k)}
    # the bootstrap's reproduction inputs are cleared for an append (bonfire.CLEARED_ON_APPEND) unless given here
    reproduction = {
        "invariants_reference": args.invariants_reference,
        "freeze_manifest": args.freeze_manifest,
        "boundary_replay_scope": args.boundary_replay_scope,
    }
    out |= {k: lexical_abspath(v) for k, v in reproduction.items() if v is not None}
    if args.boundary_replay_archive:
        out["boundary_replay_archive"] = tuple(lexical_abspath(v) for v in args.boundary_replay_archive)
    return out


def cmd_append(args: argparse.Namespace) -> int:
    from memory_kernel.pipeline import graph_append
    from memory_kernel.pipeline.bonfire import read_session
    from memory_kernel.pipeline.config import lexical_abspath
    from memory_kernel.pipeline.engine import RemoteParserRefused
    from memory_kernel.pipeline.receipts import receipts_from_env
    from memory_kernel.pipeline.runner import Stop

    secret_sources: dict[str, str | None] = {}
    ops = None
    try:
        receipts = receipts_from_env(os.environ)
        overrides = append_overrides(args, sources_out=secret_sources)
        session, _ = read_session(lexical_abspath(args.session))
        ops = open_ops(args.ops_uri)
        out = graph_append.append(
            args.bonfire_id,
            session,
            ops=ops,
            scratch_root=lexical_abspath(args.scratch_dir),
            owner=args.owner or f"mk-append-{uuid.uuid4().hex[:12]}",
            on_built=args.on_built,
            overrides=overrides,
            secret_sources=secret_sources,
            receipts=receipts,
            allow_parser_change=args.allow_parser_change,
        )
    except (ValueError, OSError, RemoteParserRefused, Stop) as exc:  # BonfireError is a ValueError
        return _refuse(f"{type(exc).__name__}: {exc}" if isinstance(exc, OSError) else str(exc))
    finally:
        close = getattr(ops, "close", None)
        if callable(close):
            close()
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0 if out.get("status") != "stopped" else REFUSED


def _close_ops(ops: Any) -> None:
    close = getattr(ops, "close", None)
    if callable(close):
        close()


def cmd_bonfire_empty(args: argparse.Namespace) -> int:
    import asyncio

    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import DriverError, Neo4jError

    from memory_kernel.pipeline import graph_seed

    try:
        password = resolve_secrets(env_file=None, container=None).values["neo4j_password"]
        if password is None:
            return _refuse("no Neo4j password configured (NEO4J_PASSWORD)")

        async def run() -> int:
            driver = AsyncGraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, password))
            try:
                return await graph_seed.empty_group(driver, args.group, rebuild=args.rebuild)
            finally:
                await driver.close()

        try:
            if args.rebuild:  # the lease is held for the whole rebuild: refused (AppendBusy) while a live one is held
                ops = open_ops(args.ops_uri)
                try:
                    with graph_seed.rebuild_lease(ops, args.group):
                        removed = asyncio.run(run())
                finally:
                    _close_ops(ops)
            else:
                removed = asyncio.run(run())
        except (Neo4jError, DriverError) as exc:
            return _refuse(neo4j_refusal(exc, uri=args.neo4j_uri, user=args.neo4j_user))
    except (ValueError, OSError) as exc:  # BonfireError is a ValueError
        return _refuse(f"{type(exc).__name__}: {exc}" if isinstance(exc, OSError) else str(exc))
    print(json.dumps({"group": args.group, "removed_nodes": removed, "rebuild": args.rebuild}, sort_keys=True))
    return 0


def cmd_bonfire_release(args: argparse.Namespace) -> int:
    """Clear a lease that is not live (expired, or released uncommitted); a live lease is refused."""
    try:
        ops = open_ops(args.ops_uri)
        try:
            ops.clear_stale_lease(args.id)
        finally:
            _close_ops(ops)
    except (ValueError, OSError) as exc:  # BonfireError (AppendBusy too) is a ValueError
        return _refuse(f"{type(exc).__name__}: {exc}" if isinstance(exc, OSError) else str(exc))
    print(json.dumps({"id": args.id, "lease": None}, sort_keys=True))
    return 0


def cmd_bonfire_ceiling(args: argparse.Namespace) -> int:
    try:
        ceiling = _amount("USD", args.usd)
        ops = open_ops(args.ops_uri)
        try:
            ops.set_ceiling(args.id, ceiling)
        finally:
            close = getattr(ops, "close", None)
            if callable(close):
                close()
    except (ValueError, OSError) as exc:  # BonfireError is a ValueError
        return _refuse(f"{type(exc).__name__}: {exc}" if isinstance(exc, OSError) else str(exc))
    print(json.dumps({"id": args.id, "ceiling_usd": str(ceiling)}, sort_keys=True))
    return 0


# -- ask / inspect -----------------------------------------------------------------------------------------------------


def neo4j_refusal(exc: BaseException, *, uri: str, user: str) -> str:
    """One refusal line for a Neo4j failure out of ``mk ask``: the error class, the server and the user, never the
    driver's own message (it can echo connection details)."""
    from neo4j.exceptions import AuthError, Neo4jError

    where = f"{uri} as {user!r}"
    if isinstance(exc, AuthError):
        return f"neo4j: authentication failed at {where} (check NEO4J_PASSWORD)"
    if isinstance(exc, Neo4jError):
        return f"neo4j: refused at {where} ({type(exc).__name__} {getattr(exc, 'code', None)})"
    return f"neo4j: unreachable or failing at {where} ({type(exc).__name__})"


def cmd_ask(args: argparse.Namespace) -> int:
    import asyncio

    from neo4j.exceptions import DriverError, Neo4jError

    from memory_kernel import api
    from memory_kernel.pipeline.config import lexical_abspath
    from memory_kernel.serve.answer import ServeMiss, Server

    try:
        secrets = resolve_secrets(
            env_file=lexical_abspath(args.env_file) if args.env_file else None, container=args.secrets_from_container
        )
        overrides = {k: v for k, v in secrets.values.items() if v is not None} | {
            "neo4j_uri": args.neo4j_uri,
            "neo4j_user": args.neo4j_user,
            "allow_paid": args.allow_paid,
        }

        async def run() -> dict:
            cfg, _meta = await api.bonfire_config(args.bonfire_id, overrides)
            async with Server(cfg) as srv:
                return await srv.ask(args.question)

        out = asyncio.run(run())
    except (ServeMiss, ValueError) as exc:  # BonfireError (not a bonfire) is a ValueError
        return _refuse(str(exc))
    except (Neo4jError, DriverError) as exc:  # a server that is down, a wrong password: the first thing a user meets
        return _refuse(neo4j_refusal(exc, uri=args.neo4j_uri, user=args.neo4j_user))
    except OSError as exc:  # an unreadable env file, no docker, a refused socket
        return _refuse(f"{type(exc).__name__}: {exc}")
    if args.json:
        print(json.dumps(out, indent=1, sort_keys=True, default=str))
    else:
        print(out["answer"])
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from memory_kernel.pipeline.config import lexical_abspath

    try:
        import uvicorn

        from memory_kernel.serve.answer import open_driver
        from memory_kernel.serve.http import create_app
        from memory_kernel.serve.registry import BonfireRegistry
        from memory_kernel.serve.runtime import RuntimeSettings
    except ImportError as exc:
        return _refuse(f"mk serve needs the serve extra (pip install 'memory-kernel[serve]'): {exc.name} is missing")
    try:
        runtime_settings = RuntimeSettings.from_env()  # MK_SERVE_*: refused at start, never mid-serve
    except ValueError as exc:
        return _refuse(f"runtime engine settings: {exc}")
    try:
        secrets = resolve_secrets(
            env_file=lexical_abspath(args.env_file) if args.env_file else None, container=args.secrets_from_container
        )
    except (ValueError, OSError) as exc:
        return _refuse(f"{type(exc).__name__}: {exc}" if isinstance(exc, OSError) else str(exc))
    overrides: dict[str, object] = {k: v for k, v in secrets.values.items() if v is not None}
    overrides["allow_paid"] = args.allow_paid
    overrides["neo4j_uri"] = args.neo4j_uri
    overrides["neo4j_user"] = args.neo4j_user
    ops = None
    ops_uri = args.ops_uri or os.environ.get(OPS_URI_ENV)  # the env carries a URI with credentials: never argv
    if ops_uri:
        from memory_kernel.ops.mongo import MongoKernelOps

        try:
            ops = MongoKernelOps.from_uri(ops_uri)
        except ValueError as exc:  # BonfireError: the ops extra is missing
            return _refuse(str(exc))
    from memory_kernel.serve.answer import ServeConfig

    try:
        driver = open_driver(
            ServeConfig(
                group="",
                neo4j_uri=args.neo4j_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=secrets.values["neo4j_password"],
            )
        )
    except ValueError as exc:  # no Neo4j password
        return _refuse(str(exc))
    app = create_app(
        BonfireRegistry(driver),
        ops=ops,
        serve_cfg_overrides=overrides,
        open_backoff_s=args.open_backoff,
        runtime_settings=runtime_settings,
        warm=True,
    )
    start = {
        "serving": args.neo4j_uri,
        "host": args.host,
        "port": args.port,
        "allow_paid": args.allow_paid,
        "ops": bool(ops),
    }
    print(json.dumps(start | {"secret_sources": secrets.sources}), flush=True)  # source names, never a value
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _bonfire_driver(args: argparse.Namespace, secrets: Any) -> tuple[Any, str, str]:
    """``(sync driver, uri, user)`` for a bonfire's graph from the command line's address and the resolved password;
    ``ValueError`` when no password resolves."""
    from memory_kernel.runtime.graph_store import open_sync_driver
    from memory_kernel.serve.runtime import neo4j_target

    overrides = {
        "neo4j_password": secrets.values["neo4j_password"],
        "neo4j_uri": args.neo4j_uri,
        "neo4j_user": args.neo4j_user,
    }
    uri, user, password = neo4j_target({}, {k: v for k, v in overrides.items() if v})
    if not password:
        raise ValueError("no Neo4j password configured (NEO4J_PASSWORD, --env-file or --secrets-from-container)")
    return open_sync_driver(uri, user, password), uri, user


def cmd_plan(args: argparse.Namespace) -> int:
    """Read one turn with a bonfire's runtime grammar and print the ``TurnPlan`` JSON (a debugging aid). It runs the
    same ``Runtime`` as ``mk serve``, on its own engine (``MK_RUNTIME_FCG_PORT``, default 9711), stopped before exit.
    The bonfire is a group of the Neo4j that ``--neo4j-uri`` names, which holds its phrases, rolesets and catalogue
    (a Neo4j password must resolve: ``NEO4J_PASSWORD``, ``--env-file``, ``--secrets-from-container``); ``--empty-graph``
    reads an empty graph for the turn's names instead. It never writes the bonfire: the construction store is opened
    read-only and no standing is recorded. Refused (exit 2) before any engine starts: an argument that is not valid
    UTF-8, a group with no ``BonfireMeta``, no pyfcg (the ``build`` extra)."""
    import asyncio
    from datetime import UTC, datetime

    from memory_kernel.runtime.api import Runtime, default_registry
    from memory_kernel.runtime.catalogue import CatalogueError, parse_catalogue
    from memory_kernel.runtime.engine import RuntimeEngine
    from memory_kernel.runtime.executor import Turn
    from memory_kernel.runtime.graph_store import NarrowWriter
    from memory_kernel.runtime.lexicon import Lexicon, QuestionMap, load_rolesets
    from memory_kernel.runtime.store import ConstructionStore, StoreError
    from memory_kernel.serve.runtime import EmptyReader, GraphReader, unencodable

    if unencodable([a for a in vars(args).values() if isinstance(a, str)] + list(args.participant)):
        return _refuse("an argument is not valid UTF-8")
    try:
        import pyfcg  # noqa: F401  (the runtime engine registers its grammar through pyfcg)
    except ImportError as exc:
        return _refuse(f"mk plan needs pyfcg (pip install 'memory-kernel[build]'): {exc}")
    from memory_kernel.pipeline.config import lexical_abspath

    group = args.bonfire_id
    driver = None
    try:
        secrets = resolve_secrets(
            env_file=lexical_abspath(args.env_file) if args.env_file else None, container=args.secrets_from_container
        )
        driver, uri, user = _bonfire_driver(args, secrets)
        with driver.session() as session:
            if not list(session.run("MATCH (m:BonfireMeta {group_id: $g}) RETURN m.group_id AS g", g=group)):
                return _refuse(f"{group!r} is not a bonfire of {uri} (its group has no BonfireMeta)")
        writer = NarrowWriter(driver, group)
        reader: object
        if args.empty_graph:
            reader = EmptyReader()
            print("mk plan: reader: empty (--empty-graph): every name names nobody", file=sys.stderr)
        else:
            reader = GraphReader(uri, user, secrets.values["neo4j_password"])
            print(f"mk plan: reader: neo4j {uri} as {user!r}, group {group!r}", file=sys.stderr)
        engine = RuntimeEngine(
            data_dir=lexical_abspath(args.data_dir) if args.data_dir else None,
            out=lexical_abspath(args.engine_dir) if args.engine_dir else None,
        )
        store = ConstructionStore(writer, read_only=True)
        raw = writer.catalogue()
        catalogue = parse_catalogue(raw.encode("utf-8"), "catalogue") if raw else None
        lexicon = Lexicon(load_rolesets(driver, group), QuestionMap.load())
    except (ValueError, OSError, StoreError, CatalogueError) as exc:
        if driver is not None:
            driver.close()
        return _refuse(f"{type(exc).__name__}: {exc}" if isinstance(exc, OSError) else str(exc))
    except Exception as exc:  # a Neo4j that does not answer
        if driver is not None:
            driver.close()
        return _refuse(neo4j_refusal(exc, uri=args.neo4j_uri or "the default", user=args.neo4j_user or "neo4j"))
    participants = tuple(args.participant) or tuple(p for p in (args.speaker, args.addressee) if p)
    turn = Turn(
        text=args.text,
        speaker=args.speaker,
        addressee=args.addressee,
        participants=participants,
        chat_id=args.chat,
        message_id=None,
        at=args.at or datetime.now(UTC).isoformat(),
    )

    async def run():
        runtime = Runtime(
            group,
            engine,
            reader,  # type: ignore[arg-type]
            default_registry(catalogue.tools if catalogue else ()),
            group_id=group,
            record_standing=False,
            store=store,
            lexicon=lexicon,
            cards=catalogue.cards if catalogue else (),
        )
        try:
            return await runtime.plan(turn)
        finally:
            if isinstance(reader, GraphReader):
                await reader.close()

    try:
        plan = asyncio.run(run())
    finally:
        engine.stop()
        driver.close()
    sys.stdout.write(json.dumps(plan.to_json(), indent=1, ensure_ascii=False) + "\n")
    return 0


def cmd_catalogue(args: argparse.Namespace) -> int:
    """``mk catalogue install BONFIRE_ID FILE``: validate a tool catalogue file and set it as the bonfire's
    ``catalogue_json`` (a refused file changes nothing). Prints ``{sha256, tools, cards}``. A running ``mk serve`` reads
    it on its next runtime load: use ``PUT /v1/bonfires/{id}/catalogue`` to reload it live."""
    from memory_kernel.pipeline.config import lexical_abspath
    from memory_kernel.runtime.catalogue import CatalogueError, parse_catalogue
    from memory_kernel.runtime.graph_store import NarrowWriter

    source = lexical_abspath(args.file)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        return _refuse(f"{source}: cannot be read ({exc})")
    try:
        cat = parse_catalogue(raw, str(source))
        text = raw.decode("utf-8")
    except CatalogueError as exc:
        return _refuse(str(exc))
    driver = None
    try:
        secrets = resolve_secrets(
            env_file=lexical_abspath(args.env_file) if args.env_file else None, container=args.secrets_from_container
        )
        driver, _uri, _user = _bonfire_driver(args, secrets)
        NarrowWriter(driver, args.bonfire_id).set_catalogue(text)
    except LookupError as exc:
        return _refuse(f"{args.bonfire_id!r} is not a bonfire (its group has no BonfireMeta): {exc}")
    except ValueError as exc:
        return _refuse(str(exc))
    except Exception as exc:  # a Neo4j that does not answer
        return _refuse(neo4j_refusal(exc, uri=args.neo4j_uri or "the default", user=args.neo4j_user or "neo4j"))
    finally:
        if driver is not None:
            driver.close()
    print(json.dumps({"sha256": cat.sha256, "tools": [t.name for t in cat.tools], "cards": len(cat.cards)}))
    return 0


def cmd_keywords(args: argparse.Namespace) -> int:
    """Print the keyword glossary (``docs/keywords.md`` is this output)."""
    from memory_kernel.runtime.keywords import glossary

    sys.stdout.write(glossary())
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from memory_kernel.pipeline.config import BuildConfig, lexical_abspath
    from memory_kernel.pipeline.runner import status, status_at

    conv_root = lexical_abspath(args.conv_root)
    if not conv_root.is_dir():
        return _refuse(f"{conv_root} is not a directory")
    config = conv_root / "logs/mk/build-config.json"
    try:
        if config.is_file():
            doc = status(BuildConfig.from_json(json.loads(config.read_bytes()), secrets={}))
        else:
            doc = status_at(conv_root, conv_root.parent / "cost-ledger.jsonl")
    except (ValueError, OSError, TypeError) as exc:
        return _refuse(f"{config}: {type(exc).__name__}: {exc}")
    print(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return 0


# -- setup -------------------------------------------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> int:
    from memory_kernel import setup_runtime

    try:
        request = setup_runtime.Request(
            data=setup_runtime.resolve_data_dir(args.data_dir, os.environ),
            components=tuple(args.component or ()),
            check=args.check,
            verify=args.verify,
            offline=args.offline,
            allow_unpinned_engine=args.allow_unpinned_engine,
            adopt={
                name: Path(os.path.abspath(os.path.expanduser(value)))
                for name, dest in setup_runtime.ADOPT_FLAGS.items()
                if (value := getattr(args, dest)) is not None
            },
        )
        return setup_runtime.run(request, as_json=args.json, out=sys.stdout, err=sys.stderr)
    except setup_runtime.SetupError as exc:
        return _refuse(str(exc))


# -- slot --------------------------------------------------------------------------------------------------------------


def cmd_slot(args: argparse.Namespace) -> int:
    from memory_kernel.pipeline import slot

    socket = Path(os.path.abspath(os.path.expanduser(args.socket)))
    if args.slot_cmd == "serve":
        if args.rss_cap_gib <= 0:
            return _refuse(f"--rss-cap-gib {args.rss_cap_gib}: must be positive")
        return slot.main_serve(
            socket,
            Path(os.path.abspath(os.path.expanduser(args.data_dir))),
            engine_port=args.engine_port,
            parser_port=args.parser_port,
            rss_cap_gib=args.rss_cap_gib,
        )
    try:
        reply = slot.control(socket, args.slot_cmd)
    except slot.SlotUnavailable as exc:
        return _refuse(str(exc))
    print(json.dumps(reply, indent=1, sort_keys=True, default=str))
    return 0


# -- the hidden stage child --------------------------------------------------------------------------------------------


def cmd_stage(args: argparse.Namespace) -> int:
    import memory_kernel
    from memory_kernel.pipeline.runner import Stop, run_stage_child

    if args.expect_package is not None and str(Path(memory_kernel.__file__)) != args.expect_package:
        print(
            f"mk _stage: imported {memory_kernel.__file__}, the driver runs {args.expect_package}: refusing",
            file=sys.stderr,
        )
        return REFUSED
    try:
        return run_stage_child(
            Path(args.config),
            args.stage,
            live=args.live,
            result=Path(args.result) if args.result else None,
            lock_fd=args.lock_fd,
            concurrency=args.concurrency,
        )
    except Stop as stop:
        return _refuse(str(stop))


def cmd_capture_shard(args: argparse.Namespace) -> int:
    from memory_kernel.pipeline.engine import EngineConfig
    from memory_kernel.pipeline.stages import _parse_capture as pc

    d = json.loads(args.site_json)
    ecfg = EngineConfig(
        port=args.port,
        binary=Path(d["binary"]),
        image=Path(d["image"]),
        parser_url=d["parser_url"],
        binary_sha256=d["binary_sha256"],
        image_sha256=d["image_sha256"],
    )
    site = pc.Site(root=Path(d["root"]), engine=ecfg, parser_lock=Path(d["parser_lock"]), parser=None)
    from memory_kernel.pipeline.stages import _capture_shards as cs

    out = Path(args.out)
    report = pc.run(out, (args.i, args.k), site=site)
    result = cs.result_path(out, args.i, args.k)
    tmp = result.with_name(result.name + ".tmp")
    tmp.write_text(json.dumps({"shard": [args.i, args.k], "status": report.get("status")}))
    tmp.replace(result)  # the parent reads this file, never the child's stdout
    return 0


# -- parser ------------------------------------------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    from memory_kernel.pipeline.bonfire import ON_BUILT
    from memory_kernel.pipeline.config import BOUNDARY_POLICIES, DEFAULT_THROUGH, REFERENT_POLICIES
    from memory_kernel.pipeline.engine import DEFAULT_PARSER_PORT, PARSER_PORT_RANGE
    from memory_kernel.pipeline.runner import ORDER

    p = argparse.ArgumentParser(prog="mk", description="Build a conversation's memory, ask it, inspect a build.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="{build,append,bonfire,ask,serve,inspect,setup,slot}")

    b = sub.add_parser(
        "build",
        help="build a conversation's memory through --through (default build_dry)",
        epilog=BUILD_EPILOG,
    )
    b.add_argument(
        "conversation",
        help="sessions JSON {sample_id, conversation: {session_N, ...}, qa?}, or a list of such samples with "
        "--sample-id. Sessions are numbered 1..N",
    )
    b.add_argument("--workdir", required=True, help="programme dir; the conversation root is WORKDIR/SAMPLE_ID")
    b.add_argument("--group", required=True, help="graph group to build into (the first write needs it empty)")
    b.add_argument("--sample-id", default=None, help="default: the JSON's sample_id (required for a list input)")
    purpose = b.add_mutually_exclusive_group()
    purpose.add_argument(
        "--purpose-file",
        default=None,
        help="JSON with research_question, or plain text: the Purpose, recorded in build-config.json (optional)",
    )
    purpose.add_argument(
        "--purpose", default=None, help="the Purpose text, recorded in build-config.json (optional; default none)"
    )
    b.add_argument(
        "--through",
        choices=ORDER,
        default=DEFAULT_THROUGH,
        help=f"last stage to run (default {DEFAULT_THROUGH}: no Neo4j, but the paid stages before it run); write..seal write the Neo4j group",
    )
    b.add_argument(
        "--ingest-pass",
        default=None,
        help="ingest-pass id recorded on every Saga (default mk-<sample_id>-001)",
    )
    b.add_argument(
        "--bonfire-id",
        default=None,
        help="bonfire id of the sources' scope and of the boundary shard's round (default mk-<sample_id>)",
    )
    b.add_argument(
        "--boundary-window-prefix",
        default="full-active",
        help="head of every boundary window id; it is part of the wire sent to Jev",
    )
    b.add_argument(
        "--boundary",
        choices=BOUNDARY_POLICIES,
        default="walk",
        help="how episodes are cut: walk = the Jev topic walk (default); segmenter = embedding cuts per session, no "
        "Jev (needs VOYAGE_API_KEY). An append inherits the bonfire's value",
    )
    b.add_argument(
        "--referents",
        choices=REFERENT_POLICIES,
        default="structure",
        help="how entities are keyed: structure = referring cores, kinds from evidence, events named by frame (default "
        "for new builds); phrase = the whole filler (every bonfire built before this). An append inherits the "
        "bonfire's value",
    )
    b.add_argument("--cap-usd", type=float, default=20.0)
    b.add_argument("--jev-concurrency", type=int, default=8, help="Jev requests in flight (1..8; 1 = sequential)")
    b.add_argument(
        "--root",
        default=None,
        help="base of relative DATA paths (inputs, proofs, frame dirs); never code (default: cwd)",
    )
    b.add_argument(
        "--harness-root",
        default=None,
        help="root of the harness files a harness-made ledger shard pins (a bench checkout); needed only to open such a "
        "shard (default: none, mk shards only)",
    )
    b.add_argument(
        "--vault",
        default=None,
        help="a directory the frame dirs may sit under: a frame dir inside it is labelled with its path relative to it in "
        "the sealed output, any other with a path-free label (default: none)",
    )
    b.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="where `mk setup` placed the engine, frames, models and corpora (default: MK_DATA_DIR, else the user data dir)",
    )
    b.add_argument(
        "--frame-dir",
        action="append",
        default=None,
        metavar="DIR",
        help="a PropBank frames dir; repeat for several, in order (the first is the upstream source of VerbNet classes and "
        "role themes, the first dir to define a roleset wins). The first dir must lie at least four levels below the "
        "filesystem root (a label is derived from its 4th parent): a shallower one is refused at preflight when "
        "events, ctx, build_dry, write or rewrite is pending. Default: the "
        "dir `mk setup` placed in the data dir, then the packaged annex",
    )
    b.add_argument(
        "--snapshot-pins",
        default=None,
        help="capture snapshot-pin list (default: computed from the tree at capture prepare, logs/mk/snapshot-pins-*)",
    )
    b.add_argument("--freeze-manifest", default=None, help="check frozen code against this manifest (default: off)")
    b.add_argument(
        "--invariants-reference",
        default=None,
        help="dir with the rows.jsonl of a reference build, for the budget invariant (default: none, the budget is "
        "skipped and recorded as such; the other invariants still run)",
    )
    b.add_argument(
        "--recipe-manifest",
        default=None,
        help="grammar_image_sha256 + final_recipe source (default: the recipe computed from the code constants)",
    )
    b.add_argument(
        "--grammar-image-sha256",
        default=None,
        help="G0 pin of a new shard (default: the recipe manifest's, else the engine image file's sha256)",
    )
    b.add_argument(
        "--boundary-replay-archive",
        nargs=2,
        default=None,
        metavar=("ARCHIVE", "SOURCES"),
        help="opt-in replay gate: an earlier boundary run's applied-boundaries archive and the sources it was made from; "
        "every same-day wire is recompiled and compared to the archived rows at $0 (needs --boundary-replay-scope)",
    )
    b.add_argument(
        "--boundary-replay-scope",
        default=None,
        metavar="FILE",
        help="JSON {scope, episode_prefix, window_prefix[, step_schema]}: the constants the replay pair was made with "
        "(needed with --boundary-replay-archive; replayed at $0 in preflight while the boundary stage is pending)",
    )
    b.add_argument(
        "--embedding-cache",
        default=None,
        help="mk-embedding-cache/v1 JSON (text sha -> vector) in front of the Voyage embedder of write, rewrite and "
        "index; a miss goes to Voyage when a key is set, and without one it stops the stage with EmbeddingMiss. The "
        "events stage does not read it (it needs a Voyage key or its own cache file under logs/)",
    )
    b.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    b.add_argument("--neo4j-user", default="neo4j")
    b.add_argument(
        "--concurrent-group-prefix",
        default=None,
        metavar="PREFIX",
        help="write integrity under strict_db_snapshot (by default a write checks only its own group): tolerate "
        "GROWTH of other Neo4j groups whose name starts with PREFIX (another build writing at the same time); default "
        "none = any change of another group fails a strict write. A shrink and the ungrouped nodes and edges are never "
        "tolerated",
    )
    b.add_argument("--engine-port", type=int, default=9701)
    b.add_argument(
        "--capture-shards",
        type=int,
        default=1,
        metavar="N",
        help="capture on N engines in parallel (1..8; ports --engine-port .. +N-1, one shared parser; default 1)",
    )
    b.add_argument(
        "--max-shift-share",
        type=float,
        default=0.10,
        metavar="X",
        help="boundary stop rule: the largest share of same-day arrivals that may start a new topic (0 < X <= 1; "
        "default 0.10). A miss is recorded, and stops the build only with --strict-gates",
    )
    b.add_argument(
        "--strict-gates",
        action="store_true",
        help="stop the build on a failed quality gate that is otherwise recorded as a metric (the boundary stop rules)",
    )
    b.add_argument(
        "--genesis",
        action="store_true",
        help="a bonfire's bootstrap from a seed of a few Setup sentences: the picks check accepts the zero such a seed "
        "gives (no picks). Recorded in build-config.json, the driver-start event and sealed.json; a root resumes only "
        "with the flag it was started with",
    )
    b.add_argument(
        "--seed",
        action="store_true",
        help="kernel 2.0: seed a bonfire into --group (which is also the sample id): the full build through seal, "
        "then its BonfireMeta last, on the kernel_ops record --ops-uri holds. Repeating a finished seed does nothing",
    )
    b.add_argument(
        "--ops-uri",
        default=None,
        help=f"--seed: the kernel_ops MongoDB address (default: ${OPS_URI_ENV}; set that variable for a URI with "
        "credentials, so they stay out of the process list)",
    )
    b.add_argument(
        "--spend-ceiling-usd", default=None, metavar="X", help="--seed: the bonfire's spend ceiling (default: none)"
    )
    b.add_argument(
        "--engine-binary", default=None, help="FCG Go binary (default: the one `mk setup` placed in the data dir)"
    )
    b.add_argument(
        "--engine-image", default=None, help="engine grammar image (default: the one `mk setup` placed in the data dir)"
    )
    b.add_argument(
        "--allow-unpinned-engine",
        action="store_true",
        help="accept an engine binary or grammar image whose sha256 is not the pinned one; the build records the actual "
        "shas and `pinned: false` in build-config.json and the driver-start event",
    )
    b.add_argument(
        "--parser-url",
        default=None,
        help="an external syntactic-parser service for the capture stage (default: none = mk starts its own parser on "
        "127.0.0.1 for the syntax and run attempts only; it needs the [parser] extra and `mk setup --component "
        "parser-model`)",
    )
    b.add_argument(
        "--parser-port",
        type=int,
        default=DEFAULT_PARSER_PORT,
        metavar="N",
        help=f"port of the managed parser, {PARSER_PORT_RANGE.start}-{PARSER_PORT_RANGE.stop - 1} (default "
        f"{DEFAULT_PARSER_PORT}); refused when busy. Its URL is sealed into every capture, so a capture can only be "
        "resumed on the port it started with",
    )
    b.add_argument(
        "--allow-remote-parser",
        action="store_true",
        help="accept a parser URL that is not on this machine (message text then leaves the machine)",
    )
    b.add_argument(
        "--env-file",
        default=None,
        help="KEY=VALUE file read for API keys and the Neo4j password; it ranks LAST (environment, then "
        "--secrets-from-container, then this file)",
    )
    b.add_argument(
        "--secrets-from-container",
        default=None,
        metavar="NAME",
        help="opt-in: docker exec NAME env; it beats --env-file and loses to the process environment",
    )
    b.set_defaults(func=cmd_build)

    ap = sub.add_parser(
        "append",
        help="append ONE new session to a bonfire (kernel 2.0: its state is in Neo4j)",
        description="Append one closed session (a session file: {conversation_id, messages, [agent_handles]}) to the "
        "bonfire BONFIRE_ID, whose state lives in its Neo4j group: the run is built in a scratch directory and "
        "committed in one Neo4j transaction, and only the session's messages cost. The bonfire's lease and spend are "
        "kept in kernel_ops. Refused before anything runs when the bonfire does not exist, its lease is held, the "
        "session's message ids are all built (duplicate) or partly built (overlap, unless --on-built skip), nothing is "
        "left after cleanup, the chat is not the bonfire's, or the run's estimate would take the bonfire past its "
        "spend ceiling.",
    )
    ap.add_argument("bonfire_id", help="the bonfire (also its Neo4j group)")
    ap.add_argument(
        "session",
        metavar="SESSION_JSON",
        help="the session file: {conversation_id, messages: [{message_id, speaker, text, timestamp}], agent_handles?}",
    )
    ap.add_argument(
        "--ops-uri",
        default=None,
        help=f"the kernel_ops MongoDB address (default: ${OPS_URI_ENV}; set that variable for a URI with credentials, "
        "so they stay out of the process list)",
    )
    ap.add_argument(
        "--scratch-dir",
        required=True,
        metavar="DIR",
        help="where the run is built (<DIR>/<bonfire>/runs/NNNN); the run's directory is deleted when it ends",
    )
    ap.add_argument(
        "--owner", default=None, help="this writer's name in the bonfire's lease (default: unique per call)"
    )
    ap.add_argument(
        "--on-built",
        choices=ON_BUILT,
        default="refuse",
        help="a session whose message ids are only partly built already: refuse it (default), or skip the built ids "
        "and append the rest",
    )
    ap.add_argument("--data-dir", default=None, metavar="DIR", help="default: the bootstrap's")
    ap.add_argument("--engine-binary", default=None, help="default: the bootstrap's")
    ap.add_argument("--engine-image", default=None, help="default: the bootstrap's")
    ap.add_argument("--engine-port", type=int, default=None, help="default: the bootstrap's")
    ap.add_argument("--capture-shards", type=int, default=None, metavar="N", help="default: the bootstrap's")
    ap.add_argument("--allow-unpinned-engine", action="store_true", help="as for build")
    ap.add_argument(
        "--allow-parser-change",
        action="store_true",
        help="append although the extraction model (MK_EXTRACTION_SPACY_MODEL) differs from the one the bonfire was "
        "built with (a bonfire built before 1.4 was parsed with en_core_web_sm)",
    )
    ap.add_argument("--parser-url", default=None, help="default: the bootstrap's")
    ap.add_argument("--parser-port", type=int, default=None, metavar="N", help="default: the bootstrap's")
    ap.add_argument("--allow-remote-parser", action="store_true", help="as for build")
    ap.add_argument("--jev-concurrency", type=int, default=None, help="default: the bootstrap's")
    ap.add_argument("--cap-usd", type=float, default=None, help="default: the bootstrap's")
    ap.add_argument("--embedding-cache", default=None, help="default: the bootstrap's")
    ap.add_argument("--neo4j-uri", default=None, help="default: the bootstrap's")
    ap.add_argument("--neo4j-user", default=None, help="default: the bootstrap's")
    ap.add_argument("--env-file", default=None, help="KEY=VALUE file for API keys and the Neo4j password; ranks last")
    ap.add_argument(
        "--secrets-from-container",
        default=None,
        metavar="NAME",
        help="opt-in: docker exec NAME env; beats --env-file, loses to the process environment",
    )
    ap.add_argument("--invariants-reference", default=None, help="default: none (the bootstrap's is not inherited)")
    ap.add_argument("--freeze-manifest", default=None, help="default: none (the bootstrap's is not inherited)")
    ap.add_argument(
        "--boundary-replay-archive",
        nargs=2,
        default=None,
        metavar=("ARCHIVE", "SOURCES"),
        help="default: none (the bootstrap's is not inherited); needs --boundary-replay-scope",
    )
    ap.add_argument("--boundary-replay-scope", default=None, metavar="FILE", help="default: none")
    ap.set_defaults(func=cmd_append)

    bf = sub.add_parser("bonfire", help="manage a bonfire's Neo4j group and spend ceiling (kernel 2.0)")
    bf_sub = bf.add_subparsers(dest="bonfire_cmd", required=True, metavar="{empty,ceiling,release}")
    be = bf_sub.add_parser(
        "empty",
        help="kernel 2.0: delete every node and edge of GROUP, in batches (refused when it has a BonfireMeta)",
        description="Empty a group a failed seed left behind. A group with a BonfireMeta is a bonfire and is never "
        "emptied. The password comes from NEO4J_PASSWORD.",
    )
    be.add_argument("group")
    be.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    be.add_argument("--neo4j-user", default="neo4j")
    be.add_argument(
        "--rebuild",
        action="store_true",
        help="the 2.0 recovery path: also delete a SEALED group (its BonfireMeta too) and reset the kernel_ops record "
        "to seeding (spent and ceiling kept); refused under a live lease. Needs --ops-uri",
    )
    be.add_argument(
        "--ops-uri",
        default=None,
        help=f"the kernel_ops MongoDB address, for --rebuild (default: ${OPS_URI_ENV})",
    )
    be.set_defaults(func=cmd_bonfire_empty)

    br = bf_sub.add_parser(
        "release",
        help="kernel 2.0: clear bonfire ID's lease when it is not live (expired, or released uncommitted)",
        description="After a stuck session is abandoned: its stopped lease keeps naming it, so later sessions see "
        "PendingSession. A live lease is refused.",
    )
    br.add_argument("id")
    br.add_argument(
        "--ops-uri",
        default=None,
        help=f"the kernel_ops MongoDB address (default: ${OPS_URI_ENV}; set that variable for a URI with credentials)",
    )
    br.set_defaults(func=cmd_bonfire_release)

    bc = bf_sub.add_parser(
        "ceiling",
        help="kernel 2.0: set bonfire ID's spend ceiling (USD) on its kernel_ops record",
    )
    bc.add_argument("id")
    bc.add_argument("usd")
    bc.add_argument(
        "--ops-uri",
        default=None,
        help=f"the kernel_ops MongoDB address (default: ${OPS_URI_ENV}; set that variable for a URI with credentials)",
    )
    bc.set_defaults(func=cmd_bonfire_ceiling)

    a = sub.add_parser("ask", help="answer a question from a bonfire's graph (cache-only unless --allow-paid)")
    a.add_argument("bonfire_id", help="the bonfire's id: its Neo4j group, which holds a BonfireMeta node")
    a.add_argument("question")
    a.add_argument("--json", action="store_true")
    a.add_argument("--allow-paid", action="store_true", help="send on a cache miss (default: refuse)")
    a.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    a.add_argument("--neo4j-user", default="neo4j")
    a.add_argument(
        "--env-file", default=None, help="KEY=VALUE file for API keys and the Neo4j password; ranks last (see build)"
    )
    a.add_argument(
        "--secrets-from-container",
        default=None,
        metavar="NAME",
        help="opt-in: docker exec NAME env; beats --env-file, loses to the process environment",
    )
    a.set_defaults(func=cmd_ask)

    sv = sub.add_parser(
        "serve",
        help="serve the kernel's context for every bonfire in a Neo4j over HTTP (needs the serve extra)",
        description='A warm context server per bonfire: POST /v1/bonfires/{id}/context {"query": ...} answers from '
        "the bonfire's graph (a bonfire is a group with a BonfireMeta node; a newer committed_run is swapped in in the "
        "background), with the query frame read from the wh rules (no Jev call). Nothing is written to disk. The models "
        "and every bonfire open at start; GET /healthz lists the bonfires and is a 503 until they are open (or failed, "
        "with one open). Errors answer with degraded: true, never a traceback.",
    )
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8740)
    sv.add_argument(
        "--allow-paid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="send on a cache miss: the query embedding (Voyage) "
        "(default: on; --no-allow-paid answers from the caches only)",
    )
    sv.add_argument(
        "--open-backoff",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="a run that fails to open is not tried again for this long (default 60)",
    )
    sv.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    sv.add_argument("--neo4j-user", default="neo4j")
    sv.add_argument(
        "--ops-uri",
        default=None,
        metavar="URI",
        help="the MongoDB holding kernel_ops (the ops extra): /healthz reads each bonfire's spend there. "
        f"Default: ${OPS_URI_ENV}; without either /healthz spend is null. A command line is visible to other users: "
        "give a URI with credentials through that variable",
    )
    sv.add_argument(
        "--env-file", default=None, help="KEY=VALUE file for API keys and the Neo4j password; ranks last (see build)"
    )
    sv.add_argument(
        "--secrets-from-container",
        default=None,
        metavar="NAME",
        help="opt-in: docker exec NAME env; beats --env-file, loses to the process environment",
    )
    sv.set_defaults(func=cmd_serve)

    pl = sub.add_parser(
        "plan",
        help="read one turn with a bonfire's runtime grammar and print the TurnPlan JSON (debugging)",
        description="Runs the bonfire's runtime grammar (the base set and its stored constructions) on one turn, on "
        "its own FCG engine (MK_RUNTIME_FCG_PORT, default 9711), and prints the TurnPlan. The graph is the bonfire's "
        "Neo4j group when a Neo4j password resolves, else an empty graph; stderr says which. Tools never run.",
    )
    pl.add_argument("bonfire_id", help="the bonfire (its Neo4j group)")
    pl.add_argument("text")
    pl.add_argument("--speaker", default=None, help='who says the turn ("me" resolves to them)')
    pl.add_argument("--addressee", default=None)
    pl.add_argument(
        "--participant", action="append", default=[], help="repeatable (default: the speaker and the addressee)"
    )
    pl.add_argument("--chat", default="mk-plan", help="the chat id (default: mk-plan)")
    pl.add_argument("--at", default=None, help="the turn's ISO 8601 time (default: now, UTC)")
    pl.add_argument("--empty-graph", action="store_true", help="read an empty graph even when Neo4j is configured")
    pl.add_argument("--neo4j-uri", default=None, help="default: bolt://localhost:7687")
    pl.add_argument("--neo4j-user", default=None, help="default: neo4j")
    pl.add_argument("--data-dir", default=None, help="where `mk setup` placed the engine (default: MK_DATA_DIR)")
    pl.add_argument(
        "--engine-dir", default=None, help="the engine's log and state file (default: <data dir>/runtime-fcg/<port>)"
    )
    pl.add_argument("--env-file", default=None, help="KEY=VALUE file for the Neo4j password; ranks last")
    pl.add_argument(
        "--secrets-from-container",
        default=None,
        metavar="NAME",
        help="opt-in: docker exec NAME env; beats --env-file, loses to the process environment",
    )
    pl.set_defaults(func=cmd_plan)

    ct = sub.add_parser(
        "catalogue",
        help="a bonfire's tool catalogue (BonfireMeta.catalogue_json)",
        description="`mk catalogue install BONFIRE_ID FILE` validates FILE (kernel-catalogue/v1: typed tools, "
        "x-side-effect, x-args, x-forms) and sets it as the bonfire's catalogue. `mk plan` and `mk serve` read it.",
    )
    ct_sub = ct.add_subparsers(dest="catalogue_cmd", required=True)
    ci = ct_sub.add_parser("install", help="validate FILE and set it as the bonfire's catalogue")
    ci.add_argument("bonfire_id", help="the bonfire (its Neo4j group)")
    ci.add_argument("file")
    ci.add_argument("--neo4j-uri", default=None, help="default: bolt://localhost:7687")
    ci.add_argument("--neo4j-user", default=None, help="default: neo4j")
    ci.add_argument("--env-file", default=None, help="KEY=VALUE file for the Neo4j password; ranks last")
    ci.add_argument(
        "--secrets-from-container",
        default=None,
        metavar="NAME",
        help="opt-in: docker exec NAME env; beats --env-file, loses to the process environment",
    )
    ci.set_defaults(func=cmd_catalogue)

    kw = sub.add_parser("keywords", help="print the runtime grammar's keyword glossary (Markdown)")
    kw.set_defaults(func=cmd_keywords)

    i = sub.add_parser("inspect", help="per-stage status of a conversation root (read-only)")
    i.add_argument("conv_root")
    i.set_defaults(func=cmd_inspect)

    from memory_kernel import setup_runtime

    t = sub.add_parser(
        "setup",
        help="fetch (or adopt), verify and check the pinned runtime data (engine, frames, models, corpora)",
        description="Fetch the pinned runtime data into the data dir, verifying every sha256 before it is placed; "
        "or check it (--check never touches the network and never writes). Linux x86_64 only.",
        epilog="Components: " + ", ".join(setup_runtime.COMPONENTS) + ". Exit 0 = every selected component verifies, "
        "1 = something is missing or differs, 2 = refused.",
    )
    t.add_argument(
        "--component",
        action="append",
        choices=setup_runtime.COMPONENTS,
        metavar="C",
        help="limit to this component (repeatable; default: all, or those with an --adopt-* option)",
    )
    t.add_argument("--check", action="store_true", help="report presence and receipts only: no network, no writes")
    t.add_argument("--verify", action="store_true", help="with --check: re-hash every placed file")
    t.add_argument(
        "--offline",
        action="store_true",
        help="no network: components without an --adopt-* option are only checked",
    )
    t.add_argument("--adopt-fcg-go", default=None, metavar="PATH", help="the FCG Go binary, its directory, or its zip")
    t.add_argument("--adopt-fcg-image", default=None, metavar="PATH", help="the grammar image file")
    t.add_argument(
        "--adopt-frames", default=None, metavar="DIR", help="a propbank-frames checkout (frames/ and LICENSE)"
    )
    t.add_argument("--adopt-wordnet", default=None, metavar="PATH", help="wordnet.zip, or a wordnet corpus directory")
    t.add_argument("--adopt-tiktoken", default=None, metavar="FILE", help="the cl100k_base BPE file")
    t.add_argument(
        "--adopt-spacy-model",
        default=None,
        metavar="DIR",
        help="an installed en_core_web_sm package dir (its .dist-info sits next to it); copied into the running "
        "interpreter's site-packages",
    )
    t.add_argument(
        "--adopt-parser-model",
        default=None,
        metavar="PATH",
        help="the benepar_en3 parser model: its zip, or the unpacked benepar_en3 directory",
    )
    t.add_argument(
        "--allow-unpinned-engine",
        action="store_true",
        help="accept an FCG Go binary or grammar image whose sha256 differs from the pin; the receipt records "
        "pinned: false and the actual shas (nothing else has an override)",
    )
    t.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="where the data and the receipt (setup.json) live (default: MK_DATA_DIR, else the user data dir)",
    )
    t.add_argument("--json", action="store_true", help="print the result as one JSON document")
    t.set_defaults(func=cmd_setup)

    sl = sub.add_parser(
        "slot",
        help="the warm capture slot: one FCG engine and parser kept up for the capture stage to lease",
        description="One long-lived process keeps an armed FCG engine and the managed parser up; a capture whose config "
        "names the slot's socket leases them instead of booting its own. Opt-in: nothing starts a slot by default.",
    )
    sl_sub = sl.add_subparsers(dest="slot_cmd", required=True, metavar="{serve,status,stop}")
    sl_serve = sl_sub.add_parser("serve", help="boot the engine and parser, then serve leases until stopped")
    sl_status = sl_sub.add_parser("status", help="print the slot's status (identities, served counts, RSS) as JSON")
    sl_stop = sl_sub.add_parser("stop", help="tear down after the current lease and exit")
    for command in (sl_serve, sl_status, sl_stop):
        command.add_argument("--socket", required=True, metavar="PATH", help="the slot's unix socket")
        command.set_defaults(func=cmd_slot)
    sl_serve.add_argument("--data-dir", required=True, metavar="DIR", help="the `mk setup` data dir (engine, parser)")
    sl_serve.add_argument("--engine-port", type=int, default=9701, help="the engine's private port (default 9701)")
    sl_serve.add_argument(
        "--parser-port",
        type=int,
        default=DEFAULT_PARSER_PORT,
        help=f"the managed parser's port (default {DEFAULT_PARSER_PORT}; sealed into every capture)",
    )
    sl_serve.add_argument(
        "--rss-cap-gib",
        type=float,
        default=6.0,
        help="recycle the engine when its RSS after a lease is above this (default 6)",
    )

    s = sub.add_parser("_stage")  # hidden: the driver's per-stage child
    s.add_argument("stage", choices=ORDER)
    s.add_argument("--config", required=True)
    s.add_argument("--result", default=None)
    s.add_argument("--live", action="store_true")
    s.add_argument("--lock-fd", type=int, default=None)
    s.add_argument("--concurrency", type=int, required=True)
    s.add_argument("--expect-package", default=None)
    s.set_defaults(func=cmd_stage)

    c = sub.add_parser("_capture-shard")  # hidden: one parallel-capture shard child
    c.add_argument("out")
    c.add_argument("i", type=int)
    c.add_argument("k", type=int)
    c.add_argument("port", type=int)
    c.add_argument("site_json")
    c.set_defaults(func=cmd_capture_shard)
    return p


def wire_runtime_data(data_dir: str | os.PathLike[str] | None = None) -> Path | None:
    """Point NLTK at the WordNet corpus ``mk setup`` placed (``<data>/nltk_data``), when that directory exists.

    The directory goes first on ``nltk.data.path`` and into ``NLTK_DATA``, which every stage child inherits (nltk reads it
    when it is imported, so it ranks ahead of ``~/nltk_data``). Only ``is_dir`` is asked of the data dir: nothing is ever
    created. Returns the directory it wired, else None."""
    from memory_kernel import setup_runtime

    corpus = setup_runtime.resolve_data_dir(data_dir, os.environ) / "nltk_data"
    if not corpus.is_dir():
        return None
    rest = [p for p in os.environ.get("NLTK_DATA", "").split(os.pathsep) if p and p != str(corpus)]
    os.environ["NLTK_DATA"] = os.pathsep.join([str(corpus), *rest])
    nltk_data = sys.modules.get("nltk.data")  # importing nltk costs ~1 s: only touch it when it is already loaded
    if nltk_data is not None and str(corpus) not in nltk_data.path:
        nltk_data.path.insert(0, str(corpus))
    return corpus


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.cmd in {"build", "append", "ask", "_stage"}:  # the commands that read the WordNet corpus
        wire_runtime_data(getattr(args, "data_dir", None))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
