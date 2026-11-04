"""``mk append`` (kernel 2.0, the graph append): the CLI maps its flags onto ``graph_append.append`` and closes the
``kernel_ops`` handle. Offline: the append and the ops opener are stood in; nothing reaches Neo4j or a paid API. One
case runs the real ``mk`` entry point in a child for ``--help``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_kernel.pipeline import bonfire

REPO_ROOT = Path(__file__).resolve().parents[2]


def turn(session: int, i: int, at: str, text: str, who: str = "Ivo") -> dict:
    return {"speaker": who, "dia_id": f"D{session}:{i}", "text": text, "timestamp": f"2026-06-03T{at}:00+00:00"}


SESSIONS: dict[int, tuple[str, str, list[dict]]] = {  # n -> (chat, date, turns); chats k1 and k2 overlap in time
    1: (
        "k1",
        "9:00 am on 3 June, 2026",
        [
            turn(1, 1, "09:00", "The kiln is warm again."),
            turn(1, 2, "09:10", "Then the bowls go in tonight.", "Rua"),
            turn(1, 3, "09:40", "I glazed six of them blue."),
        ],
    ),
    2: (
        "k2",
        "9:20 am on 3 June, 2026",
        [
            turn(2, 1, "09:20", "Is the market stall booked for Sunday?", "Sol"),
            turn(2, 2, "09:30", "Yes, the corner by the bakery.", "Tam"),
        ],
    ),
    3: (
        "k1",
        "11:00 am on 3 June, 2026",
        [
            turn(3, 1, "11:00", "Two bowls cracked in the cooling.", "Rua"),
            turn(3, 2, "11:05", "We keep them for the garden."),
        ],
    ),
    4: (
        "k2",
        "1:00 pm on 3 June, 2026",
        [
            turn(4, 1, "13:00", "Bring the price cards on Sunday.", "Sol"),
            turn(4, 2, "13:10", "And the blue bowls, they sell first.", "Tam"),
        ],
    ),
}


def write(path: Path, doc: dict) -> Path:
    path.write_text(json.dumps(doc))
    return path


def session_doc(n: int, *, edit=None) -> dict:
    """Session ``n`` of ``SESSIONS`` as a session file (``SessionInput``'s shape)."""
    chat, _, turns = SESSIONS[n]
    doc = {
        "conversation_id": chat,
        "messages": [
            {"message_id": t["dia_id"], "speaker": t["speaker"], "text": t["text"], "timestamp": t["timestamp"]}
            for t in turns
        ],
    }
    if edit is not None:
        edit(doc)
    return doc


def session_file(tmp_path: Path, n: int, *, edit=None, name: str = "next.json") -> Path:
    return write(tmp_path / name, session_doc(n, edit=edit))


# -- mk append ---------------------------------------------------------------------------------------------------------


def mk(child_env, *args: str) -> subprocess.CompletedProcess[str]:
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


def test_mk_append_help_takes_a_session_file(child_env):
    done = mk(child_env, "append", "--help")
    assert done.returncode == 0 and "SESSION_JSON" in done.stdout and "CONVERSATION_JSON" not in done.stdout


# -- mk append (kernel 2.0: the graph append) --------------------------------------------------------------------------


def run_cli(monkeypatch, tmp_path: Path, argv: list[str], result):
    """``cli.main(["append", ...])`` with the append and the ``kernel_ops`` opener stood in: returns the exit code, what
    the append was called with, and whether the ops were closed."""
    from memory_kernel import cli
    from memory_kernel.pipeline import graph_append

    called: dict = {}
    ops = SimpleNamespace(close=lambda: called.__setitem__("closed", True))

    def fake_append(*args, **kwargs):
        called["args"], called["kwargs"] = args, kwargs
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(graph_append, "append", fake_append)
    monkeypatch.setattr(cli, "open_ops", lambda uri: called.__setitem__("uri", uri) or ops)
    called["ops"] = ops
    code = cli.main(["append", "bf-kiln", str(session_file(tmp_path, 4)), *argv])
    return code, called


def test_append_cli_maps_flags(tmp_path, monkeypatch, capsys):
    scratch = tmp_path / "scratch"
    code, called = run_cli(
        monkeypatch,
        tmp_path,
        [
            "--ops-uri",
            "mongodb://ops.invalid:27017",
            "--scratch-dir",
            str(scratch),
            "--owner",
            "worker-7",
            "--on-built",
            "skip",
            "--neo4j-uri",
            "bolt://graph.invalid:7687",
            "--neo4j-user",
            "kernel",
            "--allow-parser-change",
        ],
        {"status": "appended", "run": "0004"},
    )
    assert code == 0 and json.loads(capsys.readouterr().out)["run"] == "0004"
    bonfire_id, session = called["args"]
    assert bonfire_id == "bf-kiln" and session["conversation_id"] == "k2" and len(session["messages"]) == 2
    kw = called["kwargs"]
    assert kw["ops"] is called["ops"] and called["uri"] == "mongodb://ops.invalid:27017" and called.get("closed")
    assert (kw["scratch_root"], kw["owner"], kw["on_built"], kw["allow_parser_change"]) == (
        scratch,
        "worker-7",
        "skip",
        True,
    )
    assert kw["overrides"]["neo4j_uri"] == "bolt://graph.invalid:7687" and kw["overrides"]["neo4j_user"] == "kernel"


def test_append_cli_defaults_and_a_stopped_run_exits_2(tmp_path, monkeypatch, capsys):
    code, called = run_cli(
        monkeypatch, tmp_path, ["--scratch-dir", str(tmp_path / "s")], {"status": "stopped", "run": "0004"}
    )
    kw = called["kwargs"]
    assert code == 2 and kw["on_built"] == "refuse" and kw["allow_parser_change"] is False
    assert kw["owner"].startswith("mk-append-") and "neo4j_uri" not in kw["overrides"]


def test_append_cli_reads_the_receipts_env(tmp_path, monkeypatch, capsys):
    argv = ["--scratch-dir", str(tmp_path / "s")]
    _, off = run_cli(monkeypatch, tmp_path, argv, {"status": "appended", "run": "0004"})
    assert off["kwargs"]["receipts"] is None
    monkeypatch.setenv("MK_RECEIPTS", "on")
    monkeypatch.setenv("MK_RECEIPTS_DIR", str(tmp_path / "rc"))
    _, on = run_cli(monkeypatch, tmp_path, argv, {"status": "appended", "run": "0004"})
    assert on["kwargs"]["receipts"].root == tmp_path / "rc"
    monkeypatch.delenv("MK_RECEIPTS_DIR")
    assert run_cli(monkeypatch, tmp_path, argv, {"status": "appended"})[0] != 0


def test_append_cli_refuses_in_json_and_closes_the_ops(tmp_path, monkeypatch, capsys):
    code, called = run_cli(
        monkeypatch,
        tmp_path,
        ["--scratch-dir", str(tmp_path / "s")],
        bonfire.BonfireError("busy: bonfire is leased by someone"),
    )
    assert code == 2 and called.get("closed")
    assert json.loads(capsys.readouterr().out)["refused"] == "busy: bonfire is leased by someone"


def test_append_cli_needs_a_scratch_dir(tmp_path, capsys):
    from memory_kernel import cli

    with pytest.raises(SystemExit):
        cli.main(["append", "bf-kiln", str(session_file(tmp_path, 4))])
    assert "--scratch-dir" in capsys.readouterr().err
