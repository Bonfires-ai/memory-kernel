"""The build reads its runtime data where ``mk setup`` put it: the engine and its full sha256 pins, the frame dirs
and their path-free labels, the Voyage sources, the tiktoken file, the WordNet corpus, and the packaged annex.

All in-process and offline. Nothing here starts an engine, loads a model, reaches Neo4j or spends; the heavy libraries are
replaced by recorders."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNEX = REPO_ROOT / "src/memory_kernel/data/propbank-frames-annex"
LICENSE_SHA256 = "1fdd47d0526982b4e71694ca4faf1de06ef8957faec92e3d023a26d187ac9cd3"
ADAPTED = ("buddy", "father", "friend", "kid", "mentor", "mother", "neighbor", "partner")


def config(tmp_path: Path, **kw):
    from memory_kernel.pipeline.config import BuildConfig

    return BuildConfig(
        **{
            "sample_id": "conv-syn",
            "conversation": tmp_path / "conversation.json",
            "group": "g1-test-c2-syn",
            "purpose": "remember the conversation",
            "workdir": tmp_path / "workdir",
            "root": tmp_path,
            **kw,
        }
    )


def place(data: Path, component: str, rel: str, content: bytes = b"x") -> Path:
    """What ``mk setup`` leaves behind for ``component``: the file and its receipt entry."""
    from memory_kernel import setup_runtime

    target = data / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    receipt = setup_runtime.read_receipt(data)
    receipt["components"][component] = {"path": rel, "pinned": True}
    setup_runtime._write_receipt(data, receipt)
    return target


# -- the engine ---------------------------------------------------------------------------------------------------------


def test_engine_paths_are_the_flags_then_the_setup_receipt_then_a_refusal(tmp_path):
    from memory_kernel.pipeline.config import EngineNotSetUp, engine_paths

    data = tmp_path / "data"
    with pytest.raises(EngineNotSetUp, match=r"mk setup --component fcg-go"):  # no flag, no data dir at all
        engine_paths(config(tmp_path))
    with pytest.raises(EngineNotSetUp, match=r"mk setup --component fcg-go"):  # a data dir with no receipt
        engine_paths(config(tmp_path, data_dir=data))

    binary = place(data, "fcg-go", "fcg-go/abc/FCG Go")
    with pytest.raises(EngineNotSetUp, match=r"mk setup --component fcg-image"):  # the binary alone is not an engine
        engine_paths(config(tmp_path, data_dir=data))
    image = place(data, "fcg-image", "fcg-go/resources/pb.fci")
    assert engine_paths(config(tmp_path, data_dir=data)) == (binary, image)

    flag_binary, flag_image = tmp_path / "mine/engine", tmp_path / "mine/image.fci"
    got = engine_paths(config(tmp_path, data_dir=data, engine_binary=flag_binary, engine_image=flag_image))
    assert got == (flag_binary, flag_image)  # a flag wins over the receipt, and is not required to exist here
    assert engine_paths(config(tmp_path, data_dir=data, engine_binary=flag_binary)) == (flag_binary, image)

    binary.unlink()  # the receipt names a file that is gone: the same refusal, not a path to nowhere
    with pytest.raises(EngineNotSetUp, match=r"mk setup --component fcg-go"):
        engine_paths(config(tmp_path, data_dir=data))


def test_the_engine_constants_of_the_old_defaults_are_gone():
    from memory_kernel.pipeline import config as cfg_module

    assert not hasattr(cfg_module, "ENGINE_BINARY_HOME_REL")
    assert not hasattr(cfg_module, "ENGINE_IMAGE_REL")


def test_engine_pins_are_the_full_runtime_pins_and_the_override_drops_them(tmp_path):
    from memory_kernel import setup_runtime
    from memory_kernel.pipeline.config import engine_pins
    from memory_kernel.pipeline.engine import EngineConfig

    pins = setup_runtime.load_pins()["components"]
    binary_sha, image_sha = engine_pins(config(tmp_path))
    assert (binary_sha, image_sha) == (pins["fcg-go"]["binary_sha256"], pins["fcg-image"]["sha256"])
    assert len(binary_sha) == len(image_sha) == 64  # full digests, not the old 8-character prefixes
    assert engine_pins(config(tmp_path, allow_unpinned_engine=True)) == (None, None)
    assert {"binary_sha256", "image_sha256"} <= set(EngineConfig.__dataclass_fields__)
    assert not {"binary_sha_prefix", "image_sha_prefix"} & set(EngineConfig.__dataclass_fields__)


def test_pin_problems_compare_the_full_sha_and_skip_what_is_not_pinned():
    from memory_kernel.pipeline.engine import pin_problems

    pins = ("a" * 64, "b" * 64)
    assert pin_problems(pins, binary_sha256="a" * 64, image_sha256="b" * 64) == []
    prefix_only = "a" * 8 + "c" * 56  # the old 8-character check would have passed this
    wrong = pin_problems(pins, binary_sha256=prefix_only, image_sha256="b" * 64)
    assert len(wrong) == 1 and prefix_only in wrong[0] and "a" * 64 in wrong[0]
    assert len(pin_problems(pins, binary_sha256="c" * 64, image_sha256="d" * 64)) == 2
    assert pin_problems((None, None), binary_sha256="c" * 64, image_sha256="d" * 64) == []
    assert len(pin_problems((None, "b" * 64), binary_sha256="c" * 64, image_sha256="d" * 64)) == 1


def test_the_capture_preflight_refuses_an_unpinned_engine_unless_it_is_allowed(tmp_path):
    from memory_kernel.pipeline.runner import engine_problems

    binary, image = tmp_path / "engine", tmp_path / "image.fci"
    binary.write_bytes(b"a binary that is not the pinned one")
    image.write_bytes(b"an image that is not the pinned one")

    refused = engine_problems(config(tmp_path, engine_binary=binary, engine_image=image))
    assert len(refused) == 2 and all("!= pinned" in p and "--allow-unpinned-engine" in p for p in refused)
    assert hashlib.sha256(binary.read_bytes()).hexdigest() in refused[0]  # the ACTUAL sha is named
    assert engine_problems(config(tmp_path, engine_binary=binary, engine_image=image, allow_unpinned_engine=True)) == []

    missing = engine_problems(config(tmp_path, engine_binary=tmp_path / "nope", engine_image=image))
    assert missing == [f"capture: engine binary {tmp_path / 'nope'} is not a file"]
    assert "mk setup --component fcg-go" in engine_problems(config(tmp_path))[0]  # nothing set up


def test_an_unpinned_engine_build_records_its_shas_and_a_pinned_one_records_nothing(tmp_path):
    from memory_kernel.pipeline.config import BuildConfig, engine_record
    from memory_kernel.pipeline.runner import write_config

    binary, image = tmp_path / "engine", tmp_path / "image.fci"
    binary.write_bytes(b"binary")
    image.write_bytes(b"image")
    plain = config(tmp_path, engine_binary=binary, engine_image=image)
    assert engine_record(plain) is None

    allowed = config(tmp_path, engine_binary=binary, engine_image=image, allow_unpinned_engine=True)
    record = engine_record(allowed)
    assert record == {
        "pinned": False,
        "binary_sha256": hashlib.sha256(b"binary").hexdigest(),
        "image_sha256": hashlib.sha256(b"image").hexdigest(),
    }
    gone = engine_record(config(tmp_path, allow_unpinned_engine=True))  # nothing resolves: no claim of being pinned
    assert gone == {"pinned": False, "binary_sha256": None, "image_sha256": None}

    written = json.loads(write_config(allowed, engine=record).read_bytes())
    assert written["engine_record"] == record and written["allow_unpinned_engine"] is True
    assert "engine_record" not in json.loads(write_config(plain).read_bytes())
    again = BuildConfig.from_json(written, secrets={})  # a record is not a setting: the stage children still load it
    assert again.allow_unpinned_engine is True and again.engine_binary == binary


# -- the config ---------------------------------------------------------------------------------------------------------


def test_an_older_build_config_without_the_new_keys_still_loads(tmp_path):
    from memory_kernel.pipeline.config import BuildConfig

    cfg = config(tmp_path, vault=tmp_path / "vault", data_dir=tmp_path / "data", allow_unpinned_engine=True)
    doc = cfg.to_json()
    assert doc["data_dir"] == str(tmp_path / "data") and doc["allow_unpinned_engine"] is True
    assert BuildConfig.from_json(doc, secrets={}).data_dir == tmp_path / "data"  # a Path again, not a str

    older = {k: v for k, v in doc.items() if k not in {"data_dir", "allow_unpinned_engine"}}
    loaded = BuildConfig.from_json(older, secrets={})
    assert loaded.data_dir is None and loaded.allow_unpinned_engine is False
    assert loaded.frame_dirs == cfg.frame_dirs  # the stored frame dirs are explicit; nothing is re-derived


def test_mk_build_takes_a_data_dir_frame_dirs_and_the_engine_override(tmp_path, monkeypatch):
    from memory_kernel import cli
    from memory_kernel.pipeline import frames

    monkeypatch.delenv("MK_DATA_DIR", raising=False)
    args = build_argv(tmp_path)
    plain = cli.build_config(cli.parser().parse_args(args))
    assert plain.allow_unpinned_engine is False
    assert plain.vault is None  # --vault defaults to none: it is only the base of gloss_source labels
    assert plain.data_dir is not None and plain.data_dir.is_absolute()  # the user data dir, resolved by the CLI
    assert plain.frame_dirs == frames.default_frame_dirs(plain.data_dir)

    monkeypatch.setenv("MK_DATA_DIR", str(tmp_path / "from-env"))
    from_env = cli.build_config(cli.parser().parse_args(args))
    assert from_env.data_dir == tmp_path / "from-env"
    assert from_env.frame_dirs == frames.default_frame_dirs(tmp_path / "from-env")
    chosen = cli.parser().parse_args(
        [
            *args,
            "--data-dir",
            str(tmp_path / "chosen"),
            "--allow-unpinned-engine",
            "--frame-dir",
            "b",
            "--frame-dir",
            "a",
            "--vault",
            str(tmp_path / "v"),
        ]
    )
    cfg = cli.build_config(chosen)
    assert cfg.data_dir == tmp_path / "chosen"  # the flag beats MK_DATA_DIR
    assert cfg.allow_unpinned_engine is True
    assert cfg.vault == tmp_path / "v"
    assert cfg.frame_dirs == (
        tmp_path / "b",
        tmp_path / "a",
    )  # relative to --root, in the order given; no defaults added


def test_the_frame_dir_defaults_of_the_old_vault_layout_are_gone(tmp_path):
    from memory_kernel.pipeline import config as cfg_module
    from memory_kernel.pipeline import frames

    assert not hasattr(cfg_module, "FRAME_DIRS_REL")
    assert config(tmp_path, vault=tmp_path / "vault").frame_dirs == ()  # a vault no longer names frame dirs
    managed = config(tmp_path, data_dir=tmp_path / "data")
    assert managed.frame_dirs == frames.default_frame_dirs(tmp_path / "data")
    explicit = config(tmp_path, data_dir=tmp_path / "data", frame_dirs=(tmp_path / "x",))
    assert explicit.frame_dirs == (tmp_path / "x",)


def build_argv(tmp_path: Path) -> list[str]:
    conversation = tmp_path / "conversation.json"
    conversation.write_text(json.dumps({"sample_id": "conv-syn", "conversation": {"session_1": []}}))
    return [
        "build",
        str(conversation),
        "--workdir",
        str(tmp_path / "workdir"),
        "--group",
        "g",
        "--purpose",
        "remember",
        "--root",
        str(tmp_path),
    ]


# -- frames -------------------------------------------------------------------------------------------------------------


def test_the_packaged_annex_is_24_rolesets_with_their_licence_and_notice():
    from memory_kernel.pipeline import frames

    assert frames.annex_dir() == ANNEX / "frames"
    files = sorted(p.name for p in frames.annex_dir().glob("*.xml"))
    assert len(files) == 24
    assert hashlib.sha256((ANNEX / "LICENSE").read_bytes()).hexdigest() == LICENSE_SHA256  # the upstream licence text
    notice = (ANNEX / "NOTICE").read_text()
    for lemma in ADAPTED:
        assert f"{lemma}.xml" in notice and f"{lemma}.02" in notice  # the 8 adaptations are named
    for name in files:
        assert name.removesuffix(".xml") in notice
    assert "c66e0ccf28b53f00051b187db83e937b5bee2e32" in notice and "CC BY-SA 4.0" in notice
    assert sorted(p.name for p in ANNEX.iterdir()) == ["LICENSE", "NOTICE", "frames"]


def test_the_package_data_globs_cover_the_annex_files():
    import tomllib

    globs = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"][
        "memory_kernel"
    ]
    package = REPO_ROOT / "src/memory_kernel"
    covered = {p.relative_to(package).as_posix() for pattern in globs for p in package.glob(pattern)}
    for path in ANNEX.rglob("*"):
        if path.is_file():
            assert path.relative_to(package).as_posix() in covered, path


def test_default_frame_dirs_are_the_setup_upstream_then_the_packaged_annex(tmp_path):
    from memory_kernel.pipeline import frames

    data = tmp_path / "data"
    upstream, annex = frames.default_frame_dirs(data)
    assert upstream == data / "propbank-frames/c66e0ccf28b5/frames"
    assert annex == ANNEX / "frames"


def test_frame_labels_keep_vault_relative_paths_and_hold_no_host_path_otherwise(tmp_path, monkeypatch):
    from memory_kernel.pipeline import frames
    from memory_kernel.pipeline.frames import frame_label

    vault, data = tmp_path / "vault", tmp_path / "data"
    under = vault / "data/grammar-lab/propbank-frames/frames"
    assert frame_label(under, vault=vault, data_dir=data) == "data/grammar-lab/propbank-frames/frames"  # today's bytes

    upstream, annex = frames.default_frame_dirs(data)
    assert frame_label(upstream, vault=None, data_dir=data) == "propbank-frames@c66e0ccf28b5"
    assert frame_label(annex, vault=None, data_dir=data) == "memory-kernel:propbank-frames-annex"
    other = tmp_path / "elsewhere/frames"
    anything = frame_label(other, vault=None, data_dir=data)
    assert anything == "frames:" + hashlib.sha256(str(other).encode()).hexdigest()[:12]
    assert frame_label(other, vault=vault, data_dir=data) == anything  # a dir outside the vault is labelled the same

    # the vault branch wins for a managed dir that happens to sit under the vault (the replay's bytes stay)
    inside = vault / "setup/propbank-frames/c66e0ccf28b5/frames"
    assert frame_label(inside, vault=vault, data_dir=vault / "setup") == "setup/propbank-frames/c66e0ccf28b5/frames"

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for label in (
        frame_label(upstream, vault=None, data_dir=data),
        anything,
        frame_label(annex, vault=None, data_dir=data),
    ):
        assert str(tmp_path) not in label and "/home" not in label


def test_the_events_stage_seals_path_free_labels_for_setup_managed_frames(tmp_path):
    """The gloss of a roleset and the label of the dir it came from, through the real ``_frame_index``."""
    from memory_kernel.pipeline import frames
    from memory_kernel.pipeline.stages import _events_core as ev

    data = tmp_path / "data"
    upstream, annex = frames.default_frame_dirs(data)
    upstream.mkdir(parents=True)
    (upstream / "run.xml").write_text(
        '<frameset><predicate lemma="run"><roleset id="run.01" name="operate"><roles>'
        '<role n="0" descr="runner"/><role n="1" descr="thing run"/></roles></roleset></predicate></frameset>'
    )
    index = ev._frame_index([upstream, annex], lambda d: frames.frame_label(d, vault=tmp_path / "vault", data_dir=data))
    assert index["run.01"] == ("operate (roles: arg0 = runner; arg1 = thing run)", "propbank-frames@c66e0ccf28b5")
    assert index["mom.01"][1] == "memory-kernel:propbank-frames-annex"
    assert len(index) == 25  # 1 upstream + 24 annex
    assert str(tmp_path) not in json.dumps(index)


def test_the_preflight_names_a_missing_setup_frame_dir_and_wants_a_dir_at_all(tmp_path):
    from memory_kernel.pipeline import frames
    from memory_kernel.pipeline.runner import frame_dir_problems

    data = tmp_path / "data"
    upstream, annex = frames.default_frame_dirs(data)
    cfg = config(tmp_path, data_dir=data, frame_dirs=(upstream, annex))
    refused = frame_dir_problems(cfg, ["events"])
    assert refused == [
        f"frame dir {upstream} is not a directory (needed by ['events']): run `mk setup --component propbank-frames`"
    ]
    upstream.mkdir(parents=True)
    assert frame_dir_problems(cfg, ["events"]) == []

    mine = tmp_path / "mine"
    both = config(tmp_path, data_dir=data, frame_dirs=(upstream, mine))  # EVERY dir must exist, not only the first
    assert frame_dir_problems(both, ["ctx"]) == [f"frame dir {mine} is not a directory (needed by ['ctx'])"]
    bare = frame_dir_problems(config(tmp_path), ["ctx"])
    assert len(bare) == 1 and "no frame dirs" in bare[0] and "--frame-dir" in bare[0]


def test_a_shallow_first_frame_dir_is_refused_at_preflight_before_the_sealed_code_would_fail_on_it(tmp_path, capsys):
    """The eventuality code takes ``Path(dirs[0]).parents[3]``: a first dir with fewer than 4 parents is a bare IndexError
    after the paid stages. The preflight refuses it with the reason and the fix; only the FIRST dir is constrained."""
    from memory_kernel import cli
    from memory_kernel.pipeline.runner import frame_dir_problems
    from memory_kernel.pipeline.stages._eventuality_ctx import frame_index

    for shallow in (Path("/pb/frames"), Path("/a/b/frames"), Path("frames")):
        with pytest.raises(IndexError):  # what the build would raise after the spend
            frame_index([shallow])
        refused = frame_dir_problems(config(tmp_path, frame_dirs=(shallow,)), ["events"])
        assert any(
            f"has only {len(shallow.parents)} parent directories" in p and "four levels below the filesystem root" in p
            for p in refused
        ), refused

    deep = tmp_path / "a" / "b" / "frames"  # tmp_path is itself deep: every parent exists
    deep.mkdir(parents=True)
    assert frame_dir_problems(config(tmp_path, frame_dirs=(deep,)), ["events"]) == []
    assert frame_index([deep]) == {}  # and the sealed code takes it
    exactly_four = Path("/srv/mk/propbank/frames")  # four parents: just enough, whether or not it exists
    assert len(exactly_four.parents) == 4
    assert not any(
        "parent directories" in p for p in frame_dir_problems(config(tmp_path, frame_dirs=(exactly_four,)), ["ctx"])
    )
    assert frame_index([exactly_four]) == {}
    shallow_second = frame_dir_problems(config(tmp_path, frame_dirs=(deep, Path("/pb/frames"))), ["ctx"])
    assert not any("parent directories" in p for p in shallow_second)  # only frame_dirs[0] feeds the label base

    with pytest.raises(SystemExit):
        cli.main(["build", "--help"])
    assert "four levels below the filesystem root" in " ".join(
        capsys.readouterr().out.split()
    )  # `--frame-dir` names it


# -- the models and corpora ---------------------------------------------------------------------------------------------


def test_a_pending_embedding_stage_with_no_cache_and_no_voyage_key_is_refused_at_preflight(tmp_path):
    """``embed.embedder`` raises only when its stage runs, after the paid stages: the preflight asks for a source."""
    import ast

    from memory_kernel.pipeline.runner import EMBED_STAGES, preflight

    def refusals(**kw) -> list[str]:
        return [p for p in preflight(config(tmp_path, **kw)) if p.startswith("embedding:")]

    refused = refusals(through="sanity")
    assert len(refused) == 1
    assert "VOYAGE_API_KEY" in refused[0] and "--embedding-cache" in refused[0]
    assert "['write', 'rewrite', 'index']" in refused[0]
    assert (
        refusals(through="write") != [] and refusals(through="build_dry") == []
    )  # embed.embedder serves nothing before write (the events stage's own cache has its own "events:" refusal)
    assert refusals(through="sanity", voyage_api_key="a-key") == []  # a key embeds on a cache miss
    cache = tmp_path / "embeddings.json"
    cache.write_text("{}")
    assert refusals(through="sanity", embedding_cache=cache) == []  # a cache is a source (a miss raises, before Neo4j)
    assert (
        refusals(through="sanity", embedding_cache=tmp_path / "no-such.json") == []
    )  # its own refusal says "not a file"

    # the stage set is the callers of embed.embedder: a new caller must be added to EMBED_STAGES
    src = REPO_ROOT / "src/memory_kernel"
    callers = {
        str(path.relative_to(src))
        for path in src.rglob("*.py")
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "embedder"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "embed"
            for node in ast.walk(ast.parse(path.read_text()))
        )
    }
    assert callers == {"pipeline/stages/write.py", "pipeline/stages/index.py"}  # write + rewrite; index
    assert sorted(EMBED_STAGES) == ["index", "rewrite", "write"]


def test_a_pending_events_stage_with_no_voyage_key_and_no_message_cache_is_refused_at_preflight(tmp_path, monkeypatch):
    """The events stage embeds the conversation's messages (to segment them) through its OWN cache file and the sealed
    embedder, not through ``embed.embedder``, and without a key an uncached message raises only when the stage runs,
    after the paid stages. The preflight asks for a key or that file. Presence only: a present cache that lacks a
    message keeps the runtime error (the offline-replay design)."""
    import ast

    from memory_kernel.pipeline import runner
    from memory_kernel.pipeline.stages import events

    def refusals(**kw) -> list[str]:
        return [p for p in runner.preflight(config(tmp_path, **kw)) if p.startswith("events:")]

    cache = runner.events_voyage_cache(config(tmp_path))
    refused = refusals(through="events")
    assert len(refused) == 1
    assert "VOYAGE_API_KEY" in refused[0] and str(cache) in refused[0]
    assert "pass --embedding-cache" not in refused[0]  # that flag serves write, rewrite and index only
    assert refusals() == refused  # the default --through build_dry runs events
    assert refusals(through="graph_a") == []  # no stage before events needs this cache
    assert refusals(through="events", voyage_api_key="a-key") == []  # a key embeds a cache miss
    embeddings = tmp_path / "embeddings.json"
    embeddings.write_text("{}")
    assert refusals(through="events", embedding_cache=embeddings) == refused  # --embedding-cache does not serve events
    cache.parent.mkdir(parents=True)
    cache.write_text("{}")
    assert refusals(through="events") == []  # the stage's own cache is there: presence only
    cache.unlink()
    done = runner.marker(config(tmp_path), "events")
    done.parent.mkdir(parents=True)
    done.touch()
    assert refusals(through="build_dry") == []  # a finished events stage embeds nothing more

    # the preflight looks where the stage reads: the path the stage hands to the sealed embedder
    handed: dict = {}
    monkeypatch.setattr(events.ev, "run_events", lambda **kw: handed.update(kw) or {})
    events._run_events(runner.Ctx(cfg=config(tmp_path), concurrency=1), None, tmp_path / "out", live=False, plan=True)
    assert handed["voyage_cache"] == cache

    # every place a Voyage embedder is built: embed.embedder (write, rewrite, index: EMBED_STAGES), the shared
    # text cache embed.TextVoyageCache (events, this refusal) and mk ask (--allow-paid). A new one needs its own preflight decision.
    src = REPO_ROOT / "src/memory_kernel"

    def builds_voyage_embedder(node: ast.AST) -> bool:
        """``VoyageEmbedder(...)`` and the attribute form ``kg_writer.VoyageEmbedder(...)``."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Name) and func.id == "VoyageEmbedder") or (
            isinstance(func, ast.Attribute) and func.attr == "VoyageEmbedder"
        )

    builders = {
        str(path.relative_to(src))
        for path in src.rglob("*.py")
        if any(builds_voyage_embedder(node) for node in ast.walk(ast.parse(path.read_text())))
    }
    assert builders == {"pipeline/embed.py", "serve/answer.py"}


def test_a_missing_wordnet_corpus_is_refused_at_preflight_for_any_stage_after_select(tmp_path, monkeypatch):
    """The envelope renderer raises without the corpus, after the paid stages. nltk's own lookup answers (the mock
    stands in for it: no HOME or NLTK_DATA is repointed); presence only, so nothing is loaded."""
    import nltk.data

    from memory_kernel.pipeline.runner import preflight

    looked_up: list[str] = []

    def not_found(resource, *args, **kwargs):
        looked_up.append(resource)
        raise LookupError(resource)

    monkeypatch.setattr(nltk.data, "find", not_found)

    def refusals(through: str) -> list[str]:
        return [p for p in preflight(config(tmp_path, through=through)) if p.startswith("wordnet:")]

    assert refusals("select") == [] and looked_up == []  # select renders nothing: no lookup at all
    for through in ("boundary", "build_dry"):
        refused = refusals(through)
        assert len(refused) == 1 and "run `mk setup --component wordnet`" in refused[0], through
    # both forms nltk.corpus.wordnet tries: the zip archive, then the extracted directory
    assert looked_up == ["corpora/wordnet.zip/wordnet/", "corpora/wordnet"] * 2

    def only(form: str):
        def find(resource, *args, **kwargs):
            if resource != form:
                raise LookupError(resource)
            return tmp_path / "found"

        return find

    for form in ("corpora/wordnet", "corpora/wordnet.zip/wordnet/"):  # a host with only one of the two still loads
        monkeypatch.setattr(nltk.data, "find", only(form))
        assert refusals("build_dry") == [], form


def test_the_token_encoder_resolves_the_env_dir_then_the_data_dir_and_restores_the_env(tmp_path, monkeypatch):
    import tiktoken

    from memory_kernel import setup_runtime
    from memory_kernel.search import kg_context

    url = setup_runtime.load_pins()["components"]["tiktoken"]["url"]
    key = hashlib.sha1(url.encode()).hexdigest()
    env_dir, data = tmp_path / "env-cache", tmp_path / "data"
    seen: list[str | None] = []

    class Encoder:
        def encode(self, text, disallowed_special=()):
            return text.split()

    def get_encoding(name):
        assert name == "cl100k_base"
        seen.append(os.environ.get("TIKTOKEN_CACHE_DIR"))
        return Encoder()

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)
    monkeypatch.setenv("MK_DATA_DIR", str(data))
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(env_dir))

    def fresh() -> int:
        monkeypatch.setattr(kg_context, "_ENC", None)
        return kg_context.count_tokens("a b c")

    assert fresh() == 3 and seen == [str(env_dir)]  # nothing holds the file: tiktoken's own resolution, env untouched
    (data / "tiktoken").mkdir(parents=True)
    (data / "tiktoken" / key).write_bytes(b"")
    assert fresh() == 3 and seen[-1] == str(data / "tiktoken")  # the setup dir holds it
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(env_dir)  # restored
    env_dir.mkdir()
    (env_dir / key).write_bytes(b"")
    assert fresh() == 3 and seen[-1] == str(env_dir)  # TIKTOKEN_CACHE_DIR ranks first

    monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    (env_dir / key).unlink()
    assert fresh() == 3 and seen[-1] == str(data / "tiktoken")
    assert "TIKTOKEN_CACHE_DIR" not in os.environ  # an unset variable is unset again
    assert kg_context.count_tokens("") == 0  # an empty text needs no encoder


def test_a_token_encoder_that_cannot_load_is_a_refusal_not_a_word_count(tmp_path, monkeypatch):
    import tiktoken

    from memory_kernel.search import kg_context

    def broken(_name):
        raise ConnectionError("offline")

    monkeypatch.setattr(tiktoken, "get_encoding", broken)
    monkeypatch.setattr(kg_context, "_ENC", None)
    monkeypatch.setenv("MK_DATA_DIR", str(tmp_path / "empty"))
    with pytest.raises(kg_context.TokenizerUnavailable, match=r"mk setup --component tiktoken"):
        kg_context.count_tokens("three little words")
    assert issubclass(kg_context.TokenizerUnavailable, ValueError)  # `mk ask` turns a ValueError into a refusal
    assert not hasattr(kg_context, "TOKENIZER")


def test_importing_the_context_module_never_loads_the_encoder():
    import subprocess

    code = (
        "import sys; import memory_kernel.search.kg_context as k; "
        "assert k._ENC is None; assert 'tiktoken' not in sys.modules, 'imported tiktoken eagerly'"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def test_wordnet_wiring_prepends_the_setup_corpus_and_never_creates_the_data_dir(tmp_path, monkeypatch):
    from memory_kernel import cli

    monkeypatch.setenv("NLTK_DATA", os.pathsep.join(["/somewhere/else"]))
    absent = tmp_path / "absent"
    assert cli.wire_runtime_data(absent) is None
    assert not absent.exists()  # only is_dir was asked
    assert os.environ["NLTK_DATA"] == "/somewhere/else"

    data = tmp_path / "data"
    (data / "nltk_data").mkdir(parents=True)
    assert cli.wire_runtime_data(data) == data / "nltk_data"
    assert os.environ["NLTK_DATA"].split(os.pathsep) == [str(data / "nltk_data"), "/somewhere/else"]
    cli.wire_runtime_data(data)  # idempotent
    assert os.environ["NLTK_DATA"].split(os.pathsep).count(str(data / "nltk_data")) == 1

    loaded = types.SimpleNamespace(path=["/srv/x/nltk_data"])  # nltk already imported: its search path is edited too
    monkeypatch.setitem(sys.modules, "nltk.data", loaded)
    cli.wire_runtime_data(data)
    assert loaded.path == [str(data / "nltk_data"), "/srv/x/nltk_data"]
    cli.wire_runtime_data(data)
    assert loaded.path.count(str(data / "nltk_data")) == 1


def test_a_stage_child_gets_the_drivers_data_dir(tmp_path):
    from memory_kernel.pipeline.runner import child_env

    assert child_env(config(tmp_path), {"A": "1"}).get("MK_DATA_DIR") is None
    assert child_env(config(tmp_path, data_dir=tmp_path / "d"), {"MK_DATA_DIR": "/stale"})["MK_DATA_DIR"] == str(
        tmp_path / "d"
    )
