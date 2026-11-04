"""The runtime FCG engine: its own process, its own grammar, deterministic readings (needs the pinned FCG Go)."""

from __future__ import annotations

import json
import os
import signal

import pytest

PORT = 9728  # 9711 is the runtime default, 9712-9719 mk serve's, 9720-9727 the other engine test modules'

#: one REPL call: the grammar settings the last comprehension ran with (max-search-depth is the positive control)
SETTINGS_READBACK = (
    "(let ((inv (fcg::construction-inventory (third fcg-go::*runtime-last*))))"
    " (list (fcg::get-configuration inv :max-search-depth)"
    ' (format nil "~s" (fcg::get-configuration inv :shuffle-cxns-before-application))))'
)


def _seq(text: str, left: str, right: str) -> list[str]:
    return ["sequence", json.dumps(text, ensure_ascii=False), left, right]


def _question(pyfcg, name: str, left: str, right: str, meaning: list) -> object:
    """A question construction with one np slot between two literal anchors (as ``fcgkit.template_cxn``)."""
    form = [_seq(left, "?a0-l", "?a0-r"), _seq(right, "?a1-l", "?a1-r")]
    return pyfcg.Construction(
        f"{name}-cxn",
        conditional_pole=[[f"?{name}-unit", {"#meaning": meaning}, {"#form": form}]],
        contributing_pole=[
            [f"?{name}-unit", {"subunits": ["?x-unit"], "boundaries": ["?a0-l", "?a1-r"]}],
            ["?x-unit", {"referent": "?x", "slot": "np", "boundaries": ["?a0-r", "?a1-l"]}],
        ],
        attributes={"score": 0.5, "label": "cxn"},
    )


def _fillers(pyfcg) -> list:
    """The bare and definite np fillers. Each consumes its slot's text (a ``.{1,200}`` sequence), so a clean reading
    leaves nothing in the root and passes FCG's goal tests."""

    def filler(name, form, meaning, score):
        comp = {"slot": "np", "boundaries": ["?l", "?r"], "footprints": ["not", "filled"], "#form": form}
        return pyfcg.Construction(
            f"{name}-np-cxn",
            conditional_pole=[["?u", {"referent": "?x", "#meaning": meaning}, comp]],
            contributing_pole=[["?u", {"footprints": ["filled"]}]],
            attributes={"score": score, "label": name},
        )

    rest = _seq(".{1,200}", "?m", "?r")
    bare = filler("bare", [_seq(".{1,200}", "?l", "?r")], [["name-span", "?x", "?l", "?r"]], 0.3)
    the = filler("the", [_seq("the ", "?l", "?m"), rest], [["definite", "?x"], ["name-span", "?x", "?m", "?r"]], 0.8)
    return [bare, the]


def _where_work(pyfcg, *, sentinels: bool = False) -> list:
    """The spike-1 where-work set, authored against raw text ("where do kira dace work ?"); with ``sentinels`` it
    spans the normaliser's whole text ("⟦ where do kira dace ⟪work⟫ ? ⟧"), as a compiled root does."""
    meaning = [
        ["find-entity", "?e", "?x"],
        ["identity-lines", "?l", "?e"],
        ["slot-of-kind", "?o", "?l", "Organisation"],
    ]
    meaning.append(["answer", "?o"])
    left, right = ("⟦ where do ", " ⟪work⟫ [?] ⟧") if sentinels else ("where do ", " work [?]")
    return [_question(pyfcg, "where-work", left, right, meaning), *_fillers(pyfcg)]


def _data_dir():
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.fixture
def engine(tmp_path):
    """One engine per test, started and stopped by pid: never two of ours at once."""
    from memory_kernel.runtime.engine import RuntimeEngine

    eng = RuntimeEngine(PORT, _data_dir(), out=tmp_path)
    eng.start()
    try:
        yield eng
    finally:
        eng.stop()


@pytest.mark.needs_fcg
def test_engine_is_deterministic(engine):
    import pyfcg
    import pyfcg.fcg_go_bridge as bridge

    from memory_kernel.runtime.normalise import normalise

    # another engine's address in pyfcg's process globals (as the extraction capture leaves them) must not matter
    saved = (bridge.server_address, bridge.server_port, bridge.fcg_go_initialized)
    bridge.server_address, bridge.server_port, bridge.fcg_go_initialized = "127.0.0.1", 9, True
    try:
        engine.load(_where_work(pyfcg, sentinels=True))
        engine.load(_where_work(pyfcg, sentinels=True))  # clear and re-add: still one grammar, same config
        norm = normalise("where does Kira Dace work?")
        readings = [engine.comprehend(norm.text) for _ in range(5)]
        assert all(r is not None and r.meaning == readings[0].meaning for r in readings)
        first = readings[0]
        assert first.complete and first.boundaries == (0, len(norm.text))
        [(left, right)] = [p[2:] for p in first.meaning if p[0] == "name-span"]
        assert (left, right) == (11, 20) and norm.spans.cut(left, right) == "Kira Dace"
        assert first.applied == ("where-work-cxn", "bare-np-cxn")  # application order
        assert engine._lisp(SETTINGS_READBACK) == [25, "NIL"]  # shuffle off on the inventory that just comprehended
        # "the team", not "the Kira": spaCy tags "does" in "Where does the Kira work?" a VERB and "work" a noun
        competing = [engine.comprehend(normalise("Where does the team work?").text) for _ in range(5)]
        assert all(r is not None and r.meaning == competing[0].meaning for r in competing)
        # both fillers read "the team" completely; with this grammar the bare filler's analysis is found first
        assert competing[0].complete and competing[0].applied == ("where-work-cxn", "bare-np-cxn")
        assert [p[2:] for p in competing[0].meaning if p[0] == "name-span"] == [[11, 19]]
        assert engine.comprehend(normalise("how are you?").text) is None
        assert (bridge.server_address, bridge.server_port, bridge.fcg_go_initialized) == ("127.0.0.1", 9, True)
    finally:
        bridge.server_address, bridge.server_port, bridge.fcg_go_initialized = saved


@pytest.mark.needs_fcg
def test_complete_comes_from_the_engine(engine):
    """A where-work reading with its slot bound is still incomplete when FCG's goal tests fail (text left in root).
    Raw text without the normaliser's sentinels: the root here is anchored on its literals only."""
    import pyfcg

    engine.load(_where_work(pyfcg))
    clean = engine.comprehend("where do kira dace work ?")
    assert clean is not None and clean.complete
    for text in ("what where do kira dace work ?", "where do kira dace work ? and more"):
        reading = engine.comprehend(text)
        assert reading is not None and reading.applied == ("where-work-cxn", "bare-np-cxn")
        assert any(p[0] == "name-span" for p in reading.meaning)
        assert reading.complete is False, text
        assert reading.boundaries is not None and reading.boundaries != (0, len(text))


@pytest.mark.needs_fcg
def test_slot_offsets_cut_back_to_the_surface(engine):
    """Review Focus 1 end to end: a form authored against the normalised text, offsets bound by FCG, cut by the
    normaliser's token map."""
    import pyfcg

    from memory_kernel.runtime.normalise import normalise

    meaning = [["find-entity", "?e", "?x"], ["answer", "?e"]]
    say = _question(pyfcg, "where-say", "⟦ where do ", " ⟪say⟫ it [?] ⟧", meaning)
    engine.load([say, *_fillers(pyfcg)])
    norm = normalise("Where does Jean-Paul Vorn's report say it?")
    reading = engine.comprehend(norm.text)
    assert reading is not None and reading.complete and reading.applied == ("where-say-cxn", "bare-np-cxn")
    [(left, right)] = [p[2:] for p in reading.meaning if p[0] == "name-span"]
    assert norm.spans.cut(left, right) == "Jean-Paul Vorn's report"


@pytest.mark.needs_fcg
def test_engine_death_is_an_error_then_a_restart(engine):
    import pyfcg

    from memory_kernel.pipeline.engine import proc_state
    from memory_kernel.runtime.engine import RuntimeEngineError

    engine.load(_where_work(pyfcg))
    text = "where do kira dace work ?"
    before = engine.comprehend(text)
    dead = engine.pid
    os.kill(dead, signal.SIGKILL)  # our own engine, by pid
    # /proc shows Z while the engine's threads still exit, before waitpid can reap it: wait for the exit itself
    assert engine._engine is not None and engine._engine.process.wait(10) == -9
    assert proc_state(dead) is None
    with pytest.raises(RuntimeEngineError, match="engine died rc=-9"):
        engine.comprehend(text)
    assert engine.pid is None
    after = engine.comprehend(text)  # restarted, the last complete set reloaded
    assert engine.pid not in (None, dead) and after is not None and after.meaning == before.meaning


@pytest.mark.needs_fcg
def test_timeout_marks_dirty_then_reloads(engine):
    import pyfcg

    from memory_kernel.runtime.engine import RuntimeEngineError

    engine.load(_where_work(pyfcg))
    text = "where do kira dace work ?"
    before = engine.comprehend(text)
    timed_out = engine.pid
    engine.request_timeout = 1e-6
    with pytest.raises(RuntimeEngineError):
        engine.comprehend(text)
    assert engine.pid is None  # dirty: torn down, never answered from an engine in an unknown state
    engine.request_timeout = 5.0
    after = engine.comprehend(text)
    assert engine.pid not in (None, timed_out) and after is not None and after.meaning == before.meaning


@pytest.mark.needs_fcg
def test_failed_load_refuses_comprehend(engine):
    import pyfcg

    from memory_kernel.runtime.engine import RuntimeEngineError

    refused = pyfcg.Construction("refused-cxn", conditional_pole="not a pole")  # the engine answers HTTP 500
    with pytest.raises(RuntimeEngineError, match="fcg-add-cxn"):
        engine.load([*_where_work(pyfcg), refused])
    assert engine.pid is None
    with pytest.raises(RuntimeEngineError, match="refused until a load completes"):
        engine.comprehend("where do kira dace work ?")
    engine.load(_where_work(pyfcg))
    good = engine.comprehend("where do kira dace work ?")
    assert good is not None and good.complete
    # a later failed load: the next call restarts on the last COMPLETE set, never the half-loaded one
    with pytest.raises(RuntimeEngineError):
        engine.load([refused])
    again = engine.comprehend("where do kira dace work ?")
    assert again is not None and again.meaning == good.meaning


@pytest.mark.needs_fcg
def test_stop_kills_the_engine_by_pid(engine, tmp_path):
    from memory_kernel.pipeline.engine import own_fcg_processes, port_listening, proc_state

    pid = engine.pid
    assert json.loads((tmp_path / "engine.json").read_text())["pid"] == pid
    engine.stop()
    assert proc_state(pid) is None and not port_listening(PORT) and own_fcg_processes(port=PORT) == []
    assert not (tmp_path / "engine.json").exists()
    engine.stop()  # idempotent


@pytest.mark.needs_fcg
def test_start_reclaims_the_orphan_its_state_file_names(tmp_path):
    from memory_kernel.runtime.engine import RuntimeEngine

    data_dir = _data_dir()
    first, second = RuntimeEngine(PORT, data_dir, out=tmp_path), RuntimeEngine(PORT, data_dir, out=tmp_path)
    first.start()
    try:
        orphan = first.pid
        assert orphan is not None
        state = json.loads((tmp_path / "engine.json").read_text())
        assert state["owner"]["pid"] == os.getpid()  # the process that started it
        state["owner"] = {**state["owner"], "start_ticks": "0"}  # its owner is now a dead process
        (tmp_path / "engine.json").write_text(json.dumps(state))
        second.start()  # first's engine stands in for an orphan of a dead owner: proven, killed, port reused
        assert first._engine is not None and first._engine.process.wait(5) is not None
        assert second.pid not in (None, orphan)
        assert json.loads((tmp_path / "engine.json").read_text())["pid"] == second.pid
        first.stop()  # the instance that did not write the current file never unlinks it
        assert json.loads((tmp_path / "engine.json").read_text())["pid"] == second.pid
        second.stop()
        assert not (tmp_path / "engine.json").exists()
    finally:
        first.stop()
        second.stop()


@pytest.mark.needs_fcg
def test_a_second_instance_never_kills_a_live_owners_engine(tmp_path):
    """A second instance on the same port and state dir (a second mk serve, an mk plan) finds an engine whose owner is
    alive: it leaves it running and its own start is ``PortBusy`` (mk serve then tries its next port)."""
    from memory_kernel.runtime.engine import PortBusy, RuntimeEngine

    data_dir = _data_dir()
    first, second = RuntimeEngine(PORT, data_dir, out=tmp_path), RuntimeEngine(PORT, data_dir, out=tmp_path)
    first.start()
    try:
        alive = first.pid
        with pytest.raises(PortBusy):
            second.start()
        assert first.pid == alive and first._engine.process.poll() is None
        assert json.loads((tmp_path / "engine.json").read_text())["pid"] == alive
    finally:
        second.stop()
        first.stop()


@pytest.mark.needs_fcg
def test_text_grammar_boots_without_the_kernel_ext_pin(tmp_path, monkeypatch):
    """kernel-ext.lisp serves the unit grammar only: an fcg-go pin it was not tested against boots the text grammar
    and reads with it; ``load(tiers=)`` alone refuses that pin, and the text engine keeps running."""
    import pyfcg

    from memory_kernel.runtime import engine as engine_mod
    from memory_kernel.runtime.engine import RuntimeEngine, RuntimeEngineError

    monkeypatch.setattr(engine_mod, "KERNEL_EXT_IMAGE", "f" * 12)
    eng = RuntimeEngine(PORT, _data_dir(), out=tmp_path)
    try:
        eng.start()
        eng.load(_where_work(pyfcg))
        reading = eng.comprehend("where do kira dace work ?")
        assert reading is not None and reading.complete
        pid = eng.pid
        with pytest.raises(RuntimeEngineError, match="pinned to FCG Go ffffffffffff"):
            eng.load([], tiers=("lex",))
        assert eng.pid == pid and eng.comprehend("where do kira dace work ?").meaning == reading.meaning
    finally:
        eng.stop()
