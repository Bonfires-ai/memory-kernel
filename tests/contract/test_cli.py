"""The ``mk`` command line: what ``mk --help`` promises and what ``mk build`` does at $0, offline.

Every case runs ``python -m memory_kernel.cli`` (the ``mk`` entry point) in a child under the blocked env, with the
workdir in ``tmp_path``. Nothing here reaches Neo4j or a paid API: ``select`` is the first, free stage, and a
refused build stops in preflight before any stage.
"""

from __future__ import annotations

import fcntl
import importlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ID = "conv-syn"

#: Two sessions, three messages. ``select`` must report exactly these counts.
CONVERSATION = {
    "sample_id": SAMPLE_ID,
    "conversation": {
        "speaker_a": "Ann",
        "speaker_b": "Bob",
        "session_1_date_time": "2:41 pm on 9 March, 2024",
        "session_1": [
            {"speaker": "Ann", "dia_id": "D1:1", "text": "Hi Bob, I started a bakery."},
            {"speaker": "Bob", "dia_id": "D1:2", "text": "Congrats Ann!"},
        ],
        "session_2_date_time": "10:05 am on 10 March, 2024",
        "session_2": [{"speaker": "Bob", "dia_id": "D2:1", "text": "How is the bakery going?"}],
    },
    "qa": [],
}


def mk(child_env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "memory_kernel.cli", *args],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=300,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def build_args(tmp_path: Path, *, through: str, vault: Path) -> list[str]:
    conversation = tmp_path / "conversation.json"
    conversation.write_text(json.dumps(CONVERSATION))
    return [
        "build",
        str(conversation),
        "--workdir",
        str(tmp_path / "workdir"),
        "--group",
        "g1-test-c2-syn",
        "--purpose",
        "remember the conversation",
        "--through",
        through,
        "--root",
        str(REPO_ROOT),
        "--vault",
        str(vault),
    ]


def test_the_entry_point_in_pyproject_resolves_to_main():
    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == {"mk": "memory_kernel.cli:main"}
    assert callable(importlib.import_module("memory_kernel.cli").main)


def test_mk_help_lists_build_append_bonfire_ask_inspect_setup(child_env):
    done = mk(child_env, "--help")
    assert done.returncode == 0, done.stderr
    assert re.search(r"\{build,append,bonfire,ask,serve,inspect,setup,slot\}", done.stdout)
    assert "shard-code-pins" not in done.stdout


def test_mk_setup_check_json_on_an_empty_data_dir_finds_every_component_missing(tmp_path, monkeypatch, capsys):
    """``mk setup --check --json``: rc 1, stdout is one JSON document, every component ``missing``, a ``run mk setup``
    line per component on stderr, no network, and the data dir is not even created. The spaCy model belongs to the
    running interpreter (not to the data dir), so the first call injects "not installed"; the second shows the real
    interpreter's answer for it and still finds the data-dir components missing."""
    import socket
    import urllib.request

    from memory_kernel import cli, setup_runtime

    def no_network(*_args, **_kwargs):
        raise AssertionError("mk setup --check touched the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    data = tmp_path / "data"
    monkeypatch.setenv("MK_DATA_DIR", str(data))

    with monkeypatch.context() as injected:
        injected.setattr(setup_runtime, "installed_version", lambda _dist: None)
        assert cli.main(["setup", "--check", "--json"]) == 1
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    default = list(setup_runtime.COMPONENTS)
    assert [c["name"] for c in doc["components"]] == default
    assert {c["status"] for c in doc["components"]} == {"missing"}
    assert doc["ok"] is False
    assert captured.err.count("run `mk setup") == len(default)
    assert not data.exists()

    assert cli.main(["setup", "--check", "--json"]) == 1
    status = {c["name"]: c["status"] for c in json.loads(capsys.readouterr().out)["components"]}
    assert status.pop("spacy-model") in {"ok", "missing"}
    assert status.pop("spacy-model-trf") in {"ok", "missing"}  # also the interpreter's, like the small model
    assert set(status.values()) == {"missing"}
    assert not data.exists()


def test_mk_setup_skips_trf_without_the_parser_extra(tmp_path, monkeypatch, capsys):
    """Base and ``[serve]`` installs cannot load trf: ``mk setup`` skips it with a note naming the extra and the sm
    opt-out, and ``--check`` reports it OK-with-note (rc 0 for that component), never a download or a failure."""
    from memory_kernel import cli, setup_runtime

    monkeypatch.setenv("MK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(setup_runtime, "parser_extra_installed", lambda: False)
    monkeypatch.setattr(setup_runtime.SpacyModel, "install", lambda *_a, **_k: pytest.fail("trf was installed"))
    for argv in (["setup", "--check"], ["setup"]):
        assert cli.main([*argv, "--component", "spacy-model-trf", "--json"]) == 0
        (row,) = json.loads(capsys.readouterr().out)["components"]
        assert row["status"] == "ok" and row["error"] is None
        assert "[parser]" in row["detail"] and "MK_EXTRACTION_SPACY_MODEL=en_core_web_sm" in row["detail"]


def test_mk_build_help_stops_at_seal_and_has_no_shard_code_pins(child_env):
    done = mk(child_env, "build", "--help")
    assert done.returncode == 0, done.stderr
    found = re.search(r"--through \{([^}]*)\}", done.stdout)
    assert found, done.stdout
    stages = found.group(1).split(",")
    assert stages[0] == "select"
    assert stages[-1] == "seal"
    assert "predictions" not in stages  # the eval stages belong to the bench, not to mk
    assert "--shard-code-pins" not in done.stdout


def test_mk_build_through_select_writes_the_sources_at_zero_dollars(child_env, tmp_path):
    done = mk(child_env, *build_args(tmp_path, through="select", vault=tmp_path / "vault"))
    assert done.returncode == 0, done.stdout + done.stderr
    assert '"status": "through"' in done.stdout

    conv_root = tmp_path / "workdir" / SAMPLE_ID
    sources = json.loads((conv_root / "01-select-001" / "sources.json").read_bytes())
    assert len(sources) == 3
    assert {s["scope"]["bonfire_id"] for s in sources} == {"mk-conv-syn"}  # the neutral default
    selection = json.loads((conv_root / "01-select-001" / "selection.json").read_bytes())
    assert selection["counts"]["messages"] == 3
    assert selection["counts"]["sessions"] == 2
    assert (tmp_path / "workdir" / "cost-ledger.jsonl").read_bytes() == b""  # nothing was sent, nothing was billed

    inspected = mk(child_env, "inspect", str(conv_root))
    assert inspected.returncode == 0, inspected.stdout + inspected.stderr


def test_mk_build_takes_a_bonfire_id_and_any_group_name(child_env, tmp_path):
    (tmp_path / "b").mkdir()
    args = build_args(tmp_path / "b", through="select", vault=tmp_path / "vault")
    args[args.index("--group") + 1] = "my-group"
    done = mk(child_env, *args, "--bonfire-id", "my-bonfire")
    assert done.returncode == 0, done.stdout + done.stderr
    sources = json.loads((tmp_path / "b/workdir" / SAMPLE_ID / "01-select-001" / "sources.json").read_bytes())
    assert {s["scope"]["bonfire_id"] for s in sources} == {"my-bonfire"}


def test_mk_build_takes_a_concurrent_group_prefix_and_refuses_a_blank_one(child_env, tmp_path):
    from memory_kernel import cli

    helped = mk(child_env, "build", "--help")
    assert helped.returncode == 0, helped.stderr
    assert re.search(
        r"^  --concurrent-group-prefix PREFIX$", helped.stdout, re.MULTILINE
    )  # how the gate detects a flag

    args = build_args(tmp_path, through="select", vault=tmp_path / "vault")
    assert cli.build_config(cli.parser().parse_args(args)).concurrent_group_prefix is None  # strict by default
    prefixed = cli.parser().parse_args([*args, "--concurrent-group-prefix", "g1-test-c2-"])
    assert cli.build_config(prefixed).concurrent_group_prefix == "g1-test-c2-"

    done = mk(child_env, *args, "--concurrent-group-prefix", "")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "concurrent_group_prefix is empty" in json.loads(done.stdout)["refused"]
    assert not (tmp_path / "workdir").exists()


def test_mk_build_refuses_a_replay_archive_without_its_scope_file(child_env, tmp_path):
    args = build_args(tmp_path, through="select", vault=tmp_path / "vault")
    done = mk(child_env, *args, "--boundary-replay-archive", str(tmp_path / "a.json"), str(tmp_path / "s.json"))
    assert done.returncode == 2, done.stdout + done.stderr
    assert "--boundary-replay-scope" in json.loads(done.stdout)["refused"]
    assert not (tmp_path / "workdir").exists()


def test_mk_build_preflight_refusal_names_the_missing_frame_dir(child_env, tmp_path):
    """With no ``--frame-dir`` the frames are the ones ``mk setup`` placed in the data dir. An empty data dir is a
    refusal that says so (and that no engine is set up either), before any stage and before any mkdir."""
    data = tmp_path / "empty-data"
    env = {**child_env, "MK_DATA_DIR": str(data)}
    done = mk(env, *build_args(tmp_path, through="build_dry", vault=tmp_path / "no-such-vault"))
    assert done.returncode == 2, done.stdout + done.stderr
    refused = json.loads(done.stdout)["refused"]
    assert "refused before any stage" in refused
    upstream = data / "propbank-frames/c66e0ccf28b5/frames"
    assert f"frame dir {upstream} is not a directory" in refused
    assert "run `mk setup --component propbank-frames`" in refused
    assert "no FCG Go engine binary" in refused and "mk setup --component fcg-go" in refused
    assert not (tmp_path / "workdir").exists()  # refused before any mkdir

    named = mk(env, *build_args(tmp_path, through="build_dry", vault=tmp_path), "--frame-dir", str(tmp_path / "nope"))
    assert f"frame dir {tmp_path / 'nope'} is not a directory" in json.loads(named.stdout)["refused"]
    assert str(upstream) not in json.loads(named.stdout)["refused"]  # --frame-dir replaces the setup-managed pair


def test_secrets_resolve_env_then_container_then_env_file(tmp_path, monkeypatch):
    """The historical trap: a stale ``.env`` ``NEO4J_PASSWORD`` beat the live container's password, which the container
    exported under a ``GM_``-prefixed name. ``mk`` no longer reads any ``GM_`` name: the stub below returns the value
    under ``NEO4J_PASSWORD``, the name ``mk`` reads, and a container that exports only its ``GM_`` names resolves
    nothing (a launcher exports ``NEO4J_PASSWORD`` itself)."""
    from memory_kernel import cli

    env_file = tmp_path / "stack.env"
    env_file.write_text("NEO4J_PASSWORD=stale-file\nVOYAGE_API_KEY=file-voyage\nOPENROUTER_API_KEY=file-openrouter\n")
    asked: list[str] = []

    def stub_container(name: str) -> dict[str, str]:
        asked.append(name)
        return {"NEO4J_PASSWORD": "live-container", "VOYAGE_API_KEY": "container-voyage"}

    monkeypatch.setattr(cli, "container_env", stub_container)

    # all three sources set: env > container > env-file, per field, and the reported source names say so
    got = cli.resolve_secrets(env_file=env_file, container="stack", environ={"OPENROUTER_API_KEY": "live-env"})
    assert asked == ["stack"]
    assert got.values == {
        "openrouter_api_key": "live-env",
        "voyage_api_key": "container-voyage",
        "neo4j_password": "live-container",
    }
    assert got.sources == {"openrouter_api_key": "env", "voyage_api_key": "container", "neo4j_password": "container"}

    # a set process variable beats the container too (and the env-file)
    got = cli.resolve_secrets(env_file=env_file, container="stack", environ={"NEO4J_PASSWORD": "live-env"})
    assert got.values["neo4j_password"] == "live-env"
    assert got.sources["neo4j_password"] == "env"
    assert got.sources["openrouter_api_key"] == "env-file"  # nothing else sets it, and the container has no such name

    # no container opt-in: the env-file is the last resort; a field nobody sets is None
    got = cli.resolve_secrets(env_file=env_file, container=None, environ={})
    assert got.values["neo4j_password"] == "stale-file"
    assert got.sources == {"openrouter_api_key": "env-file", "voyage_api_key": "env-file", "neo4j_password": "env-file"}
    assert asked == ["stack", "stack"]  # the third call did not touch the container
    # the retired GM_* aliases are not read from any source
    got = cli.resolve_secrets(
        env_file=None, container=None, environ={"GM_VOYAGE_API_KEY": "x", "GM_DEFAULT_GRAPHITI_PASSWORD": "y"}
    )
    assert got.values == {"openrouter_api_key": None, "voyage_api_key": None, "neo4j_password": None}
    got = cli.resolve_secrets(env_file=None, container=None, environ={})
    assert got.values == {"openrouter_api_key": None, "voyage_api_key": None, "neo4j_password": None}
    assert got.sources == {"openrouter_api_key": None, "voyage_api_key": None, "neo4j_password": None}


def test_driver_start_logs_the_secret_sources_and_never_a_value(child_env, tmp_path):
    env_file = tmp_path / "stack.env"
    env_file.write_text("VOYAGE_API_KEY=value-from-the-file\nNEO4J_PASSWORD=value-from-the-file\n")
    child = {**child_env, "NEO4J_PASSWORD": "value-from-the-env"}
    done = mk(child, *build_args(tmp_path, through="select", vault=tmp_path / "vault"), "--env-file", str(env_file))
    assert done.returncode == 0, done.stdout + done.stderr

    events_file = tmp_path / "workdir" / SAMPLE_ID / "driver-events.jsonl"
    starts = [e for e in map(json.loads, events_file.read_text().splitlines()) if e["event"] == "driver-start"]
    assert starts[-1]["secret_sources"] == {
        "openrouter_api_key": "env",  # the blocked env sets it
        "voyage_api_key": "env-file",
        "neo4j_password": "env",  # the process variable beats the file's
    }
    on_disk = events_file.read_text() + (tmp_path / "workdir" / SAMPLE_ID / "logs/mk/build-config.json").read_text()
    assert "value-from-the" not in on_disk


def test_a_container_that_cannot_be_read_is_a_refusal_that_never_carries_its_environment(tmp_path, monkeypatch, capsys):
    """``docker exec NAME env`` prints the container's environment, i.e. its secrets. When it fails, `mk build` and
    `mk ask` refuse (exit 2) naming the container and the status; neither stream, the message, nor any chained
    exception holds a byte of the command's output."""
    from memory_kernel import cli

    sentinel = "-".join(("planted", "environment", "0f3c9a"))

    def failing_exec(argv, **kwargs):
        assert argv[:3] == ["docker", "exec", "stack"]
        return subprocess.CompletedProcess(argv, 125, stdout=f"NEO4J_PASSWORD={sentinel}\n", stderr=f"warn {sentinel}")

    monkeypatch.setattr(cli.subprocess, "run", failing_exec)
    with pytest.raises(ValueError) as caught:
        cli.container_env("stack")
    exc = caught.value
    assert str(exc) == "cannot read the environment of container 'stack': `docker exec` exited 125"
    assert (
        exc.__cause__ is None and exc.__context__ is None
    )  # no CalledProcessError (it keeps the output) to chain from
    assert sentinel not in " ".join(map(repr, (exc, exc.args, str(exc), exc.__cause__, exc.__context__)))

    ask = cli.parser().parse_args(["ask", "ember", "who?", "--secrets-from-container", "stack"])
    assert cli.cmd_ask(ask) == 2
    out = capsys.readouterr()
    assert "container 'stack'" in json.loads(out.out)["refused"]
    assert sentinel not in out.out + out.err

    conversation = tmp_path / "conversation.json"
    conversation.write_text(json.dumps(CONVERSATION))
    build = cli.parser().parse_args(
        ["build", str(conversation), "--workdir", str(tmp_path / "work"), "--group", "g1-test-c2-syn", "--purpose", "p"]
        + ["--secrets-from-container", "stack", "--through", "select"]
    )
    assert cli.cmd_build(build) == 2
    out = capsys.readouterr()
    assert "container 'stack'" in json.loads(out.out)["refused"]
    assert sentinel not in out.out + out.err
    assert not (tmp_path / "work").exists()  # refused while resolving the config, before any mkdir


def test_mk_ask_against_a_neo4j_that_is_down_or_refuses_exits_2_not_with_a_traceback(monkeypatch, capsys):
    """The first thing a user meets without a running Neo4j: the driver's error out of the bonfire lookup becomes the
    documented refusal."""
    from neo4j.exceptions import AuthError, ClientError, ServiceUnavailable

    from memory_kernel import api, cli

    monkeypatch.setenv("NEO4J_PASSWORD", "not-a-real-password")
    args = cli.parser().parse_args(["ask", "ember", "who?", "--neo4j-uri", "bolt://127.0.0.1:7699"])
    wanted = {
        ServiceUnavailable("Couldn't connect to 127.0.0.1:7699"): "unreachable or failing at bolt://127.0.0.1:7699",
        AuthError("The client is unauthorized"): "authentication failed at bolt://127.0.0.1:7699 as 'neo4j'",
        ClientError("denied"): "refused at bolt://127.0.0.1:7699 as 'neo4j' (ClientError",
        ConnectionRefusedError("[Errno 111] refused"): "ConnectionRefusedError",
    }
    for raised, text in wanted.items():

        async def failing(bonfire_id, overrides=None, _raised=raised):
            raise _raised

        monkeypatch.setattr(api, "bonfire_config", failing)
        assert cli.cmd_ask(args) == 2, raised
        refused = json.loads(capsys.readouterr().out)["refused"]
        assert text in refused, (raised, refused)


def test_mk_ask_refuses_a_group_that_is_not_a_bonfire(monkeypatch, capsys):
    from memory_kernel import api, cli
    from memory_kernel.graph.state_read import META
    from unit.serve.test_registry import FakeDriver

    monkeypatch.setenv("NEO4J_PASSWORD", "not-a-real-password")
    monkeypatch.setattr(api.answer, "open_driver", lambda cfg: FakeDriver({META: []}))
    assert cli.main(["ask", "g18", "who?"]) == cli.REFUSED
    assert "not a bonfire" in json.loads(capsys.readouterr().out)["refused"]


def test_mk_build_refuses_a_bad_conversation_input_before_any_stage(child_env, tmp_path):
    """Every way the input can be wrong is exit 2 with ``{"refused": ...}`` and nothing created, including a
    session date that used to fall back to the wall clock."""
    dated = json.loads(json.dumps(CONVERSATION))
    dated["conversation"]["session_2_date_time"] = "sometime in the spring"
    cases = {
        "missing-file": (None, "FileNotFoundError"),
        "not-json": ("not json", "not JSON"),
        "no-conversation-key": (json.dumps({"sample_id": SAMPLE_ID}), "sample has no 'conversation'"),
        "bad-session-date": (
            json.dumps(dated),
            "session_2_date_time: unparseable session date 'sometime in the spring'",
        ),
    }
    for name, (text, expect) in cases.items():
        case = tmp_path / name
        case.mkdir()
        args = build_args(case, through="select", vault=case / "vault")
        if text is None:
            args[1] = str(case / "no-such-file.json")
        else:
            (case / "conversation.json").write_text(text)
        done = mk(child_env, *args)
        assert done.returncode == 2, (name, done.stdout, done.stderr)
        assert expect in json.loads(done.stdout)["refused"], (name, done.stdout)
        assert "Traceback" not in done.stderr, (name, done.stderr)
        assert not (case / "workdir").exists(), name  # refused before any mkdir


def test_inspect_and_lock_held_never_create_the_lock(child_env, tmp_path):
    """``mk inspect`` is read-only. An absent lock file means free (and stays absent); a lock held by another
    open file description is seen as held."""
    from memory_kernel.pipeline.runner import _Probe, lock_held

    root = tmp_path / "workdir" / SAMPLE_ID
    root.mkdir(parents=True)
    done = mk(child_env, "inspect", str(root))
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout)["lock_held"] is False
    assert sorted(p.name for p in root.iterdir()) == [], "mk inspect must not create driver.lock or any other file"

    probe = _Probe(root)
    assert lock_held(probe) is False  # type: ignore[arg-type]
    assert not (root / "driver.lock").exists()
    (root / "driver.lock").write_text("")
    assert lock_held(probe) is False  # type: ignore[arg-type]  # an existing, unlocked file: free
    with (root / "driver.lock").open("rb") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert lock_held(probe) is True  # type: ignore[arg-type]
    assert lock_held(probe) is False  # type: ignore[arg-type]


def test_an_unpinned_engine_build_logs_its_shas_in_driver_start_and_build_config(child_env, tmp_path):
    """``--allow-unpinned-engine`` records ``{pinned, binary_sha256, image_sha256}`` in the ``driver-start`` event
    and in ``build-config.json``; a build without the flag records nothing of the kind. Runs through ``select`` only, so no
    engine starts: the files are plain bytes."""
    import hashlib

    binary, image = tmp_path / "engine", tmp_path / "image.fci"
    binary.write_bytes(b"a binary")
    image.write_bytes(b"an image")
    conv_root = tmp_path / "workdir" / SAMPLE_ID
    args = build_args(tmp_path, through="select", vault=tmp_path / "vault")

    plain = mk(child_env, *args, "--engine-binary", str(binary), "--engine-image", str(image))
    assert plain.returncode == 0, plain.stdout + plain.stderr
    starts = [
        e
        for e in map(json.loads, (conv_root / "driver-events.jsonl").read_text().splitlines())
        if e["event"] == "driver-start"
    ]
    assert "engine" not in starts[-1]
    assert "engine_record" not in json.loads((conv_root / "logs/mk/build-config.json").read_text())

    flagged = mk(
        child_env, *args, "--engine-binary", str(binary), "--engine-image", str(image), "--allow-unpinned-engine"
    )
    assert flagged.returncode == 0, flagged.stdout + flagged.stderr
    record = {
        "pinned": False,
        "binary_sha256": hashlib.sha256(b"a binary").hexdigest(),
        "image_sha256": hashlib.sha256(b"an image").hexdigest(),
    }
    starts = [
        e
        for e in map(json.loads, (conv_root / "driver-events.jsonl").read_text().splitlines())
        if e["event"] == "driver-start"
    ]
    assert starts[-1]["engine"] == record
    config = json.loads((conv_root / "logs/mk/build-config.json").read_text())
    assert config["engine_record"] == record and config["allow_unpinned_engine"] is True
