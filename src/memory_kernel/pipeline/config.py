"""The explicit build configuration.

Every value the C2 build reads is a field here, passed down explicitly: no module global or function default below
the CLI carries programme state. Secrets (API keys, the Neo4j password) are fields too, but they are never printed
and never written to disk: ``to_json`` replaces them with a marker, and a stage child gets them back through its
environment (``MK_OPENROUTER_API_KEY`` / ``MK_VOYAGE_API_KEY``, see ``from_json``). Some modules read non-secret
settings from the environment themselves; ``cli.py``'s docstring names examples.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from memory_kernel.pipeline.engine import DEFAULT_PARSER_PORT, check_parser_port, managed_parser_url
from memory_kernel.pipeline.frames import default_frame_dirs
from memory_kernel.pipeline.provenance import sha_file

MAX_CAPTURE_SHARDS = 8
MAX_CONCURRENCY = 8  # = the Jev provider's httpx Limits(max_connections=8)
#: The last stage mk runs: write / rewrite / index write Neo4j, seal hashes the conversation root. The
#: evaluation stages belong to the benchmark harness.
LAST_STAGE = "seal"
#: ``--through`` when none is given: writing Neo4j is always an explicit ``--through write|rewrite|index|sanity|seal``.
DEFAULT_THROUGH = "build_dry"
SECRET_FIELDS: tuple[str, ...] = ("openrouter_api_key", "voyage_api_key", "neo4j_password")
SECRET_ENV: dict[str, str] = {
    "openrouter_api_key": "MK_OPENROUTER_API_KEY",
    "voyage_api_key": "MK_VOYAGE_API_KEY",
    "neo4j_password": "MK_NEO4J_PASSWORD",
}
#: Keys of an older ``build-config.json`` that ``from_json`` ignores (``mk inspect`` on an older root): ``shard_code_pins``
#: was removed (every new shard pins ``ledger.MK_CODE``); so were ``jev_model`` / ``chat_model`` / ``gap_model``
#: (nothing read them: the models are constants next to the senders that use them) and ``boundary_replay_gate``
#: (the replay gate is opt-in: it runs only when ``boundary_replay_archive`` is given), and ``prune_orphans`` (kernel 1.2:
#: a forward append leaves no orphans, so ``mk append --prune-orphans`` is gone), and (kernel 1.8, the taxonomy and question
#: layer archived) ``taxonomy_gate``, ``lost_wires`` and ``remine_unit_price_budget``.
RETIRED_KEYS = frozenset(
    {
        "shard_code_pins",
        "jev_model",
        "chat_model",
        "gap_model",
        "boundary_replay_gate",
        "prune_orphans",
        "taxonomy_gate",
        "lost_wires",
        "remine_unit_price_budget",
    }
)
#: The stages kernel 1.8 archived, each to the nearest kept stage before it in 1.7's
#: order: a stored ``through`` that names one (an older ``build-config.json``) is read as that stage, never refused.
#: ``--through`` on the command line offers only ``runner.ORDER``.
ARCHIVED_STAGES: dict[str, str] = {
    "taxonomy": "select",
    "jev1_pilot": "graph_a",
    "jev1": "graph_a",
    "taxonomy_events": "graph_a",
    "taxq": "graph_a",
    "labelq": "graph_a",
    "jev2": "graph_a",
    "remine": "graph_a",
    "graph_b": "graph_a",
}
#: Keys ``write_config`` records beside the fields (the engine identity of an ``--allow-unpinned-engine`` build) and
#: ``from_json`` does not read back: a record, not a setting.
RECORD_KEYS = frozenset({"engine_record"})
_PATH_FIELDS = frozenset(
    {
        "conversation",
        "workdir",
        "root",
        "vault",
        "harness_root",
        "data_dir",
        "snapshot_pins",
        "freeze_manifest",
        "invariants_reference",
        "recipe_manifest",
        "embedding_cache",
        "engine_binary",
        "engine_image",
        "boundary_replay_scope",
        "shard_dir",
        "capture_cache",
        "capture_slot",
        "bonfire_index_record",
        "forward_input",
    }
)
#: How the write stage meets the group: ``fresh`` = the first write into an empty group; ``append`` = ``mk append``
#: merging the whole conversation so far into the bonfire's existing group.
WRITE_MODES: frozenset[str] = frozenset({"fresh", "append"})
#: how the boundary stage cuts episodes: ``walk`` = the Jev topic walk (the default), ``segmenter`` = embedding cuts per session
BOUNDARY_POLICIES: tuple[str, ...] = ("walk", "segmenter")
#: how entities are keyed: ``phrase`` = the whole filler (every bonfire built before the field), ``structure`` = referring
#: cores, kinds from evidence, events named by frame
REFERENT_POLICIES: tuple[str, ...] = ("phrase", "structure")


def stage_order() -> tuple[str, ...]:
    from memory_kernel.pipeline.runner import ORDER

    return ORDER


@dataclass(frozen=True, kw_only=True)
class BuildConfig:
    # identity
    sample_id: str
    conversation: Path
    group: str
    #: the Purpose text: a record in ``build-config.json`` (nothing reads it); None = a build without one
    purpose: str | None = None
    ingest_pass: str | None = None
    stem: str | None = None
    bonfire_id: str | None = None
    episode_prefix: str | None = None
    # places
    workdir: Path
    #: base of every relative DATA path (inputs, proofs, frame dirs); never the code
    root: Path
    #: only the base ``gloss_source`` labels are made relative to (frame dirs under it); None = path-free labels
    vault: Path | None = None
    #: where ``mk setup`` placed the engine, frames, models and corpora (``--data-dir`` / ``MK_DATA_DIR`` / the user data
    #: dir, resolved by the CLI); None = nothing is looked up there
    data_dir: Path | None = None
    #: root of the harness files a harness-made ledger shard pins (``--harness-root``); None = only ``mk`` shards open
    harness_root: Path | None = None
    #: PropBank frame dirs in order (the first is the upstream source of VerbNet classes and role themes); () = the
    #: setup-managed upstream dir under ``data_dir`` and the packaged annex (``frames.default_frame_dirs``)
    frame_dirs: tuple[Path, ...] = ()
    #: None = computed from the tree at capture prepare (``parse.snapshot_pins_for``); a path = an explicit list
    snapshot_pins: Path | None = None
    freeze_manifest: Path | None = None
    #: a reference build's ``rows.jsonl`` dir for the budget invariant; None = the budget is skipped and recorded as such
    invariants_reference: Path | None = None
    #: head of every boundary window id (``<prefix>-<active_start>-cursor-<cursor>``): part of the wire Jev is sent
    boundary_window_prefix: str = "full-active"
    #: ``walk`` (omitted from the JSON, so a walk config is byte-identical to before) or ``segmenter``
    boundary_policy: Literal["walk", "segmenter"] = "walk"
    #: ``phrase`` (omitted from the JSON, so a phrase config is byte-identical to before) or ``structure``
    referent_policy: Literal["phrase", "structure"] = "phrase"
    #: (archive, sources) of an earlier boundary run to check this build's boundary wires against; None = no replay gate
    boundary_replay_archive: tuple[Path, Path] | None = None
    #: JSON ``{scope, episode_prefix, window_prefix}``: the constants the replay pair was made with (``load_replay_scope``)
    boundary_replay_scope: Path | None = None
    grammar_image_sha256: str | None = None
    #: read-only source of ``grammar_image_sha256`` (boundary) and ``final_recipe`` (index); None = the recipe
    #: computed from the code constants (``default_recipe``) and the engine image file's sha
    recipe_manifest: Path | None = None
    embedding_cache: Path | None = None
    # neo4j (write / rewrite / index)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    #: the bench's strict mode: the write snapshots every Neo4j group and fails on any change of another group
    strict_db_snapshot: bool = False
    #: under ``strict_db_snapshot`` (``write.foreign_changes``): None = ANY change of another group fails the write. A
    #: prefix tolerates GROWTH of the other groups whose name starts with it (a concurrent build's own first write); a
    #: shrink and the ungrouped bucket are never tolerated
    concurrent_group_prefix: str | None = None
    # money / concurrency
    cap_usd: float = 20.0
    jev_concurrency: int = 8
    # credentials (never printed, never written to disk)
    openrouter_api_key: str | None = field(default=None, repr=False)
    voyage_api_key: str | None = field(default=None, repr=False)
    neo4j_password: str | None = field(default=None, repr=False)
    # engine
    engine_port: int = 9701
    #: capture parallelism: k engine shards on ports engine_port..engine_port+k-1 sharing one managed parser
    #: (1 = the single-engine capture; clamped further to the usable CPUs at run time)
    capture_shards: int = 1
    #: None = the ``mk setup`` receipt's (``engine_paths``); an explicit path wins over the receipt
    engine_binary: Path | None = None
    engine_image: Path | None = None
    #: accept an engine binary or grammar image whose sha256 is not the one ``runtime-pins.json`` pins
    allow_unpinned_engine: bool = False
    #: an external syntactic-parser service; None = the managed in-package parser on ``127.0.0.1:<parser_port>``, started
    #: only for the capture's syntax and run attempts (``engine.ParserService``)
    parser_url: str | None = None
    #: the managed parser's port (9800-9899). Its URL is sealed into every capture, so a new port means a new capture
    parser_port: int = DEFAULT_PARSER_PORT
    allow_remote_parser: bool = False
    # run control
    through: str | None = None
    #: the boundary stage's topic-shift stop rule: the largest share of same-day arrivals answered ``shift_topic``
    max_shift_share: float = 0.10
    #: True = a failed advisory gate (the boundary stop rules; in a forward append also ``ctx-prior-fallback``) stops
    #: the build; False = it is recorded. An append inherits the bootstrap's value: a bonfire's bootstrap is built
    #: without it
    strict_gates: bool = False
    #: ``mk build --genesis``: a bonfire's bootstrap from a seed of a few Setup sentences. The picks check then accepts
    #: the zero such a seed gives (``genesis_zero_ok``); every other check is unchanged. Recorded in
    #: ``build-config.json`` (omitted while False, so a real build's config is byte-identical to before), in the
    #: ``driver-start`` recipe and in ``sealed.json``; an append is never one (``bonfire.run_config``)
    genesis: bool = False
    # bonfires (``mk append``; ``pipeline/bonfire.py``)
    #: the round-ledger shard the stages open instead of ``<conv_root>/03-boundary-001-ledger`` (its ``-pairs`` sibling
    #: follows it): a bonfire's ONE persistent shard, so archived answers replay under the same pins on every run.
    #: None = the shard inside the conversation root
    shard_dir: Path | None = None
    #: a bonfire's per-message capture cache dir (``mk append`` passes ``<bonfire>/capture-cache``); None = no cache
    capture_cache: Path | None = None
    #: the warm capture slot's unix socket (``mk slot serve``; a long-lived worker that runs the slot passes it): the
    #: capture stage leases the slot's engine + parser instead of booting its own. None = no slot, every capture boots
    #: cold
    capture_slot: Path | None = None
    #: ``fresh`` = the first write into an empty group; ``append`` = merge into the bonfire's existing group
    write_mode: Literal["fresh", "append"] = "fresh"
    #: kernel 2.0: an append writes nothing before its commit. Its write, rewrite and index stages run as rolled-back
    #: transactions that record their vectors, and ``pipeline.commit`` replays them in ONE transaction. Every 2.0
    #: append sets it (``bonfire.run_config``) and the stages act on ``write_mode`` alone. Omitted from the JSON while
    #: False
    graph_commit: bool = False
    #: messages this run adds to a bonfire (``mk append``): with ``append_base_usd`` they size the run's budget
    #: (``bonfire.append_budget``); the run's hard limit is ``bonfire.append_limit`` (this base + the budget, never past
    #: ``spend_ceiling_usd``), and ``kernel_ops`` holds the estimate before the run. None = not an append
    append_messages: int | None = None
    #: the bonfire shard's held-or-spent total when ``mk append`` planned this run, a decimal string: the run's stages
    #: open the shard with the limit this plus the run's budget (``bonfire.append_limit``) instead of its sealed caps.
    #: None = not an append (the shard's sealed caps hold). Omitted from the JSON while None, so a config that sets
    #: neither this nor ``spend_ceiling_usd`` carries only keys 1.5.x knows
    append_base_usd: str | None = None
    #: the bonfire's spend ceiling (``bonfire.json`` ``spend_ceiling_usd``, set by ``mk bonfire init``), a decimal
    #: string: an append's shard never goes past it. None = no ceiling (omitted from the JSON)
    spend_ceiling_usd: str | None = None
    #: the bonfire bootstrap's index record (``<bootstrap>/11-index-001/index.json``; ``mk append`` sets it): its
    #: ``recipe_tag`` is the key recipe the bonfire's group is indexed under, so the preflight refuses an append whose
    #: recipe tag (the recipe's key render and ``MK_TEMPORAL_DEIXIS``) differs. None = no such check
    bonfire_index_record: Path | None = None
    #: a forward append's run input (``bonfire.FORWARD_INPUT``, written by ``mk append`` from the chain state): the
    #: session's first order and number, its chat's boundary walk state and tail, the history's addressing flag, the
    #: frozen roleset wires and the bonfire's Voyage cache (``load_forward``). None = a full build
    forward_input: Path | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.jev_concurrency <= MAX_CONCURRENCY:
            raise ValueError(f"jev_concurrency={self.jev_concurrency}: must be 1..{MAX_CONCURRENCY}")
        if not 1 <= self.capture_shards <= MAX_CAPTURE_SHARDS:
            raise ValueError(f"capture_shards={self.capture_shards}: must be 1..{MAX_CAPTURE_SHARDS}")
        if not self.group.strip():
            raise ValueError("group is empty: name the graph group to build into")
        if not self.boundary_window_prefix.strip():
            raise ValueError("boundary_window_prefix is empty")
        if self.concurrent_group_prefix is not None and not self.concurrent_group_prefix.strip():
            raise ValueError(
                "concurrent_group_prefix is empty: it would tolerate every group; omit it for the strict default"
            )
        check_parser_port(self.parser_port)  # ValueError: a refusal
        if self.parser_url is not None and not self.parser_url.strip():
            raise ValueError("parser_url is empty: omit it for the managed parser")
        order = stage_order()
        if self.through is not None and self.through not in order:
            raise ValueError(f"unknown through stage {self.through!r}: mk runs {order[0]}..{LAST_STAGE}")
        if self.cap_usd < 0:
            raise ValueError("cap_usd must be non-negative")
        if not 0 < self.max_shift_share <= 1:
            raise ValueError(f"max_shift_share={self.max_shift_share}: must be in (0, 1]")
        if self.boundary_policy not in BOUNDARY_POLICIES:
            raise ValueError(f"boundary_policy {self.boundary_policy!r}: must be one of {list(BOUNDARY_POLICIES)}")
        if self.referent_policy not in REFERENT_POLICIES:
            raise ValueError(f"referent_policy {self.referent_policy!r}: must be one of {list(REFERENT_POLICIES)}")
        if self.write_mode not in WRITE_MODES:
            raise ValueError(f"write_mode {self.write_mode!r}: must be one of {sorted(WRITE_MODES)}")
        if self.genesis and (self.write_mode == "append" or self.forward_input is not None):
            raise ValueError(
                "genesis is a bootstrap build: an append (write_mode append, a forward input) is never one"
            )
        if self.graph_commit and self.strict_db_snapshot:
            from memory_kernel.errors import BonfireError

            raise BonfireError(
                "graph_commit with strict_db_snapshot: a graph-mode append writes nothing before its commit, so its "
                "stages take no database snapshot to compare (turn strict_db_snapshot off)"
            )
        if self.append_messages is not None and self.append_messages < 0:
            raise ValueError(f"append_messages={self.append_messages}: must be non-negative")
        for name in ("append_base_usd", "spend_ceiling_usd"):
            value = getattr(self, name)
            if value is not None and not _usd(value):
                raise ValueError(f"{name}={value!r}: must be a finite non-negative decimal string")
        for name in (
            "workdir",
            "root",
            "vault",
            "harness_root",
            "data_dir",
            "shard_dir",
            "capture_cache",
            "capture_slot",
            "bonfire_index_record",
            "forward_input",
        ):
            value = getattr(self, name)
            if value is not None and not value.is_absolute():
                raise ValueError(f"{name} must be absolute (lexical, never resolved): {value}")
        stem = self.stem or "conv" + self.sample_id.removeprefix("conv-")
        derived = {
            "ingest_pass": self.ingest_pass or f"mk-{self.sample_id}-001",
            "stem": stem,
            "bonfire_id": self.bonfire_id or f"mk-{self.sample_id}",
            "episode_prefix": self.episode_prefix or f"{stem}-utc-day-episode-",
            "frame_dirs": tuple(Path(p) for p in self.frame_dirs)
            or (default_frame_dirs(self.data_dir) if self.data_dir is not None else ()),
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)

    @property
    def conv_root(self) -> Path:
        return self.workdir / self.sample_id

    @property
    def ledger(self) -> Path:
        return self.workdir / "cost-ledger.jsonl"

    @property
    def config_path(self) -> Path:
        """Where the parent writes this config (no secrets) for every stage child."""
        return self.conv_root / "logs/mk/build-config.json"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "boundary_policy" and value == "walk":
                continue
            if f.name == "referent_policy" and value == "phrase":
                continue
            if f.name in ("strict_db_snapshot", "genesis", "graph_commit") and not value:
                continue
            if f.name in ("append_base_usd", "spend_ceiling_usd") and value is None:
                continue
            if f.name in SECRET_FIELDS:
                out[f.name] = None if value is None else "<set>"
            elif isinstance(value, Path):
                out[f.name] = str(value)
            elif isinstance(value, tuple):
                out[f.name] = [str(v) if isinstance(v, Path) else v for v in value]
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any], *, secrets: Mapping[str, str | None]) -> BuildConfig:
        """Rebuild from ``to_json``; secrets come from ``secrets`` (a stage child passes its environment). An older
        config loads: its ``RETIRED_KEYS`` are dropped and a ``through`` naming an archived stage is read as
        ``ARCHIVED_STAGES``'s kept stage."""
        known = {f.name for f in fields(cls)}
        doc = {k: v for k, v in doc.items() if k not in RETIRED_KEYS and k not in RECORD_KEYS}
        unknown = set(doc) - known
        if unknown:
            raise ValueError(f"unknown build-config keys: {sorted(unknown)}")
        kw: dict[str, Any] = {}
        for name, value in doc.items():
            if name in SECRET_FIELDS:
                continue
            if value is not None and name in _PATH_FIELDS:
                value = Path(value)
            elif name == "frame_dirs":
                value = tuple(Path(v) for v in value)
            elif name == "boundary_replay_archive" and value is not None:
                value = (Path(value[0]), Path(value[1]))
            elif name == "through":
                value = ARCHIVED_STAGES.get(value, value)
            kw[name] = value
        for name in SECRET_FIELDS:
            kw[name] = secrets.get(SECRET_ENV[name]) or None
        return cls(**kw)

    def secret_env(self) -> dict[str, str]:
        """The child-only env carrying the secrets (``from_json`` reads it back)."""
        return {SECRET_ENV[n]: v for n in SECRET_FIELDS if (v := getattr(self, n)) is not None}


def _usd(value: object) -> bool:
    """``value`` is a decimal string of a finite non-negative amount."""
    if not isinstance(value, str):
        return False
    try:
        amount = Decimal(value)
    except ArithmeticError:
        return False
    return amount.is_finite() and amount >= 0


def default_recipe() -> dict[str, Any]:
    """The index recipe computed from the code constants, used when ``--recipe-manifest`` is absent.

    The same three keys a recipe manifest's ``final_recipe`` supplies to ``index``: the sealed retrieval
    recipe, the answer-context config (``timeline_shapes`` as a sorted list) and the key recipe tag."""
    from dataclasses import asdict

    from memory_kernel.search.kg_context import ContextConfig
    from memory_kernel.search.kg_recipe import SEALED_RECIPE
    from memory_kernel.serve.answer import RECIPE_TAG

    context = asdict(ContextConfig(evidence_k=40))
    context["timeline_shapes"] = sorted(context["timeline_shapes"])
    return {"kg_recipe_config": asdict(SEALED_RECIPE), "context_config": context, "key_recipe_tag": RECIPE_TAG}


def _read_json_file(path: Path, what: str) -> Any:
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{what} {path}: {type(exc).__name__}: {exc}") from None


#: The three string keys of a replay scope (``windows.construction_window._SCOPE_FIELDS``).
REPLAY_SCOPE_KEYS = ("bonfire_id", "profile", "conversation_id")


def load_replay_scope(path: Path) -> dict[str, Any]:
    """The ``--boundary-replay-scope`` file: ``{"scope": {bonfire_id, profile, conversation_id}, "episode_prefix": str,
    "window_prefix": str}`` plus an optional ``"step_schema": str``, the constants an archived boundary run was made with
    (``step_schema``: the id that run sealed its boundary steps with, when it is not this package's). ValueError (a
    refusal) otherwise."""
    what = "boundary replay scope"
    doc = _read_json_file(path, what)
    required = {"scope", "episode_prefix", "window_prefix"}
    if not isinstance(doc, dict) or not required <= set(doc) <= required | {"step_schema"}:
        raise ValueError(
            f"{what} {path}: expected an object with exactly scope, episode_prefix and window_prefix (and optionally "
            "step_schema)"
        )
    scope = doc["scope"]
    if not isinstance(scope, dict) or set(scope) != set(REPLAY_SCOPE_KEYS):
        raise ValueError(f"{what} {path}: scope must be an object with exactly {', '.join(REPLAY_SCOPE_KEYS)}")
    for key, value in (
        *scope.items(),
        ("episode_prefix", doc["episode_prefix"]),
        ("window_prefix", doc["window_prefix"]),
        *((("step_schema", doc["step_schema"]),) if "step_schema" in doc else ()),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{what} {path}: {key} must be a non-empty string")
    return doc


def grammar_image_sha(cfg: BuildConfig) -> str:
    """The G0 image pin of a new shard: ``cfg.grammar_image_sha256``, else the recipe manifest's, else the sha256 of the
    engine image file (computed here, so only when a boundary stage creates a shard). ValueError when none is
    available; a manifest that is given but unreadable, or has no pin, is an error, never a fall-through."""
    if cfg.grammar_image_sha256:
        return cfg.grammar_image_sha256
    if cfg.recipe_manifest is not None:
        try:
            doc = json.loads(cfg.recipe_manifest.read_bytes())
        except (OSError, ValueError) as exc:
            raise ValueError(f"recipe manifest {cfg.recipe_manifest}: {type(exc).__name__}: {exc}") from None
        sha = doc.get("grammar_image_sha256") if isinstance(doc, dict) else None
        if not sha:
            raise ValueError(f"recipe manifest {cfg.recipe_manifest} has no grammar_image_sha256")
        return str(sha)
    image = engine_image(cfg)  # EngineNotSetUp is a ValueError: a refusal
    try:
        return sha_file(image)
    except OSError as exc:
        raise ValueError(
            f"no --grammar-image-sha256 and no --recipe-manifest, and the engine image {image} cannot be read "
            f"({type(exc).__name__})"
        ) from None


def live_parser_url(cfg: BuildConfig) -> str:
    """The parser the capture's engine calls: ``--parser-url``, else the managed parser's loopback URL. The capture
    seals this string."""
    return cfg.parser_url or managed_parser_url(cfg.parser_port)


class EngineNotSetUp(ValueError):
    """No FCG engine binary or grammar image: no flag names one and the ``mk setup`` receipt has none (a refusal)."""


def _managed_engine(cfg: BuildConfig, component: str) -> Path | None:
    """The path ``mk setup`` recorded for ``component`` under ``cfg.data_dir`` when it is still on disk, else None."""
    if cfg.data_dir is None:
        return None
    from memory_kernel import setup_runtime

    try:
        return setup_runtime.managed_path(cfg.data_dir, component)
    except setup_runtime.SetupError as exc:
        raise EngineNotSetUp(str(exc)) from None


def engine_binary(cfg: BuildConfig) -> Path:
    """The FCG Go binary: ``--engine-binary``, else the one ``mk setup`` placed, else a refusal."""
    path = cfg.engine_binary or _managed_engine(cfg, "fcg-go")
    if path is None:
        raise EngineNotSetUp(
            "no FCG Go engine binary: pass --engine-binary, or run `mk setup --component fcg-go` "
            f"(looked under {cfg.data_dir or 'no data dir'})"
        )
    return path


def engine_image(cfg: BuildConfig) -> Path:
    """The grammar image: ``--engine-image``, else the one ``mk setup`` placed, else a refusal."""
    path = cfg.engine_image or _managed_engine(cfg, "fcg-image")
    if path is None:
        raise EngineNotSetUp(
            "no engine grammar image: pass --engine-image, or run `mk setup --component fcg-image` "
            f"(looked under {cfg.data_dir or 'no data dir'})"
        )
    return path


def engine_paths(cfg: BuildConfig) -> tuple[Path, Path]:
    """(binary, image) of the FCG engine, in this order: the config's flags, the ``mk setup`` receipt, else
    ``EngineNotSetUp`` (one source for preflight and ``parse.site_for``)."""
    return engine_binary(cfg), engine_image(cfg)


def engine_pins(cfg: BuildConfig) -> tuple[str | None, str | None]:
    """(binary sha256, image sha256) the engine files must have: the full pins of ``runtime-pins.json``. ``(None,
    None)`` with ``allow_unpinned_engine``: the shas are then recorded (``engine_record``), not enforced."""
    if cfg.allow_unpinned_engine:
        return None, None
    from memory_kernel import setup_runtime

    components = setup_runtime.load_pins()["components"]
    return components["fcg-go"]["binary_sha256"], components["fcg-image"]["sha256"]


def engine_record(cfg: BuildConfig) -> dict[str, Any] | None:
    """What an ``--allow-unpinned-engine`` build writes into ``build-config.json`` and the ``driver-start`` event:
    ``{pinned, binary_sha256, image_sha256}`` of the engine files as they are (a sha is None when the file cannot be
    resolved or read). ``pinned`` is true only when both equal the pins. None when the flag is off: a pinned build
    records nothing new."""
    if not cfg.allow_unpinned_engine:
        return None
    from memory_kernel import setup_runtime

    components = setup_runtime.load_pins()["components"]
    pinned = (components["fcg-go"]["binary_sha256"], components["fcg-image"]["sha256"])
    found: list[str | None] = []
    for resolve in (engine_binary, engine_image):
        try:
            found.append(sha_file(resolve(cfg)))
        except (EngineNotSetUp, OSError):
            found.append(None)
    return {"pinned": tuple(found) == pinned, "binary_sha256": found[0], "image_sha256": found[1]}


# -- the conversation input --------------------------------------------------------------------------------------------


class ConversationError(ValueError):
    """The conversation input does not load (a refusal, never a traceback)."""


_SESSION = re.compile(r"session_(\d+)")
_SESSION_CONVERSATION = re.compile(r"session_(\d+)_conversation")


def session_numbers(conversation: Mapping[str, Any]) -> list[int]:
    return sorted(
        int(m.group(1)) for k, v in conversation.items() if (m := _SESSION.fullmatch(k)) and isinstance(v, list)
    )


def load_forward(cfg: BuildConfig) -> dict[str, Any] | None:
    """The forward append's run input (``cfg.forward_input``), or None for a full build."""
    path = getattr(cfg, "forward_input", None)  # a stage context's stand-in config may not carry the field
    if path is None:
        return None
    return json.loads(Path(path).read_bytes())


def genesis_zero_ok(cfg: BuildConfig) -> bool:
    """True in a genesis build (``BuildConfig.genesis``, ``mk build --genesis``), whose seed of a few Setup sentences
    gives zeros a full build refuses. The predicate of the check that then accepts its zero: the picks
    (``graph._no_picks_needed``: the recorded plan counted no message). False for every other build, and for a stage
    context's stand-in config without the field."""
    return getattr(cfg, "genesis", False) is True


def load_conversation(path: Path, sample_id: str | None) -> tuple[dict, bytes]:
    """``(item, raw)``: one sample object, or the item ``sample_id`` of a list of sample objects (sessions JSON).

    ``raw`` is the file read (its sha is ``selection.json["dataset_sha256"]``). Every way the input can be wrong raises
    ``ConversationError`` with a one-line reason."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ConversationError(f"conversation {path}: {type(exc).__name__}: {exc.strerror or exc}") from None
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise ConversationError(f"conversation {path}: not JSON ({exc})") from None
    if isinstance(doc, list):
        if not sample_id:
            raise ConversationError(f"conversation {path} is a list of samples: pass --sample-id")
        found = [x for x in doc if isinstance(x, dict) and x.get("sample_id") == sample_id]
        if not found:
            raise ConversationError(f"conversation {path}: sample {sample_id!r} not found")
        item = found[0]
    elif isinstance(doc, dict):
        item = doc
    else:
        raise ConversationError(f"conversation {path}: expected a sample object or a list, got {type(doc).__name__}")
    for key in ("sample_id", "conversation"):
        if key not in item:
            raise ConversationError(f"conversation {path}: sample has no {key!r}")
    if sample_id and item["sample_id"] != sample_id:
        raise ConversationError(f"conversation {path}: sample_id {item['sample_id']!r} != {sample_id!r}")
    conv = item["conversation"]
    if not isinstance(conv, dict):
        raise ConversationError(f"conversation {path}: 'conversation' is not an object")
    numbers = session_numbers(conv)
    if not numbers:
        raise ConversationError(f"conversation {path}: no session_N list")
    from memory_kernel.pipeline.stages._conversation import first_session

    try:
        first = first_session(conv)  # a forward append's one-session input numbers its session as the bonfire does
    except ValueError as exc:
        raise ConversationError(f"conversation {path}: {exc}") from None
    if numbers != list(range(first, first + len(numbers))):
        span = "1..N" if first == 1 else f"{first}..N"
        raise ConversationError(f"conversation {path}: sessions are not numbered {span}: {numbers}")
    for key, value in conv.items():  # the optional conversation (chat) of a session, for the boundary walk
        if (m := _SESSION_CONVERSATION.fullmatch(key)) is None:
            continue
        if int(m.group(1)) not in numbers:
            raise ConversationError(f"conversation {path}: {key} names no session_{m.group(1)}")
        if not isinstance(value, str) or not value.strip():
            raise ConversationError(f"conversation {path}: {key} must be a non-empty string")
    untagged = [f"session_{n}" for n in numbers if f"session_{n}_conversation" not in conv]
    if untagged and len(untagged) < len(numbers):  # all or none: a partly tagged input mixes chats by accident
        raise ConversationError(
            f"conversation {path}: some sessions name a chat (session_N_conversation) and {untagged} do not: tag "
            "every session or none"
        )
    handles = conv.get("agent_handles")
    if handles is not None and (
        not isinstance(handles, list) or not all(isinstance(h, str) and h.strip("@ ") for h in handles)
    ):
        raise ConversationError(f"conversation {path}: agent_handles must be a list of non-empty strings")
    if not isinstance(item.get("qa") or [], list):
        raise ConversationError(f"conversation {path}: 'qa' is not a list")
    return item, raw


def lexical_abspath(path: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    """Absolute, normalised, NEVER resolved: ``data`` is a symlink and harness path strings are ``relative_to(ROOT)``."""
    p = Path(path)
    if not p.is_absolute():
        p = (base if base is not None else Path.cwd()) / p
    return Path(os.path.normpath(p))
