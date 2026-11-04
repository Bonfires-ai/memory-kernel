"""The answer shape comes from the runtime grammar's reading when the caller has one (a ``TurnPlan.answer_shape``):
serve's answer directive and its context frame then parse nothing. Without one, the shape is read from the question's
parse (``runtime.shape.read_shape``): no wh-word list is consulted anywhere. Also the evaluation script (``scripts/runtime_eval.py``) over the invented smoke set, on a fake graph."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from memory_kernel.graph.g1_nlp import turn_doc
from memory_kernel.search.recipe_affordances import answer_directive
from memory_kernel.serve import answer
from memory_kernel.serve.answer import ServeConfig, Server

REPO = Path(__file__).resolve().parents[3]
SMOKE = REPO / "tests" / "fixtures" / "runtime_eval" / "questions.jsonl"
PORT = 9719  # 9712-9717 are the other runtime test modules'; 9718 is the evaluation's own


def _no_nlp(_text: str) -> Any:
    raise AssertionError("the question was parsed for its wh-word although the reading gave the answer shape")


def _server(tmp_path: Path, **kwargs: Any) -> Server:
    cfg = ServeConfig(group="g", **kwargs)
    return Server(cfg)


def test_answer_shape_comes_from_the_reading(tmp_path):
    # the answer directive: the reading's shape, no wh-slot parse
    prompt = answer.answer_prompt("when did Kira Dace publish the report?", "ctx", nlp=_no_nlp, answer_shape="when")
    assert answer_directive("when") in prompt
    polar = answer.answer_prompt("did Milo join Brask Labs?", "ctx", nlp=_no_nlp, answer_shape="polar")
    assert "ANSWER DIRECTIVE" not in polar  # a polar reading asks for no open-slot directive
    # the context frame: the reading's shape, no wh-rule and no parse
    srv = _server(tmp_path)
    srv._nlp = _no_nlp
    assert srv.query_frame("where does Kira Dace work?", shape="who") == {"shape": "who"}


def test_without_a_reading_the_shape_is_read_from_the_parse(tmp_path):
    calls: list[str] = []

    def counting(text: str) -> Any:
        calls.append(text)
        return turn_doc(text)

    def broken(_text: str) -> Any:
        raise OSError("no parser")

    srv = _server(tmp_path)
    for text, shape in [
        ("when did Ana leave Zorvex?", "when"),
        ("Whose report did Milo publish?", "who"),
        ("How many reports did Milo publish?", "how_many"),
        ("Did Ana say who left Zorvex?", "polar"),
        ("Zorvex", "polar"),
    ]:
        assert srv.query_frame(text)["shape"] == shape, text
    # the directive follows: an open question gets one (a degree too), a yes/no question with an embedded wh gets none
    for text, directive in [
        ("How old is Milo?", True),
        ("Whose report did Milo publish?", True),
        ("Would Ana enjoy Zorvex?", True),
        ("Did Ana say who left Zorvex?", False),
    ]:
        assert ("ANSWER DIRECTIVE" in answer.answer_prompt(text, "ctx", nlp=turn_doc)) is directive, text
    # one parse per question text, however many readers
    answer.question_shape("Who left Zorvex?", counting)
    answer.directive_for("Who left Zorvex?", counting)
    assert calls == ["Who left Zorvex?"]
    # a failing parser never raises: polar
    assert answer.question_shape("Who left Brask Labs?", broken) == "polar"
    srv._nlp = broken
    assert srv.query_frame("when did Milo leave Brask Labs?") == {"shape": "polar"}


# -- /context: an optional answer_shape in the body -------------------------------------------------------------------


def test_context_route_takes_the_answer_shape(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from memory_kernel.serve import http
    from unit.serve.fakes import FakeRegistry

    seen: list[str | None] = []

    class Shaped:
        def __init__(self, cfg: ServeConfig) -> None:
            self.cfg = cfg

        async def __aenter__(self) -> Shaped:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def context(self, question: str, *, shape: str | None = None) -> dict[str, Any]:
            seen.append(shape)
            frame = {"shape": shape or "what"}
            return {"frame": frame, "text": question, "message_ids": [], "tokens": 1, "retrieved": []}

    app = http.create_app(FakeRegistry().add("ember"), serve_cfg_overrides={"allow_paid": False}, server_factory=Shaped)
    with TestClient(app) as client:
        url = "/v1/bonfires/ember/context"
        assert client.post(url, json={"query": "when did Milo leave?", "answer_shape": "when"}).json()["frame"] == {
            "shape": "when"
        }
        assert client.post(url, json={"query": "what did Milo say?"}).status_code == 200
        bad = client.post(url, json={"query": "what did Milo say?", "answer_shape": 3})
        assert bad.status_code == 422 and "answer_shape" in bad.json()["reason"]
    assert seen == ["when", None]


# -- the evaluation script --------------------------------------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location("_runtime_eval", REPO / "scripts" / "runtime_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SmokeGraph:
    """Kira Dace works at Brask Labs (work.01); every other name names nobody."""

    KIRA = {"uuid": "k1", "name": "Kira Dace", "labels": ["Entity", "Person"]}
    BRASK = {"uuid": "b1", "name": "Brask Labs", "labels": ["Entity", "Organisation"]}

    async def rows(self, cypher: str, params: dict[str, Any]) -> list[dict]:
        if "n.name = $name" in cypher:
            return [dict(e) for e in (self.KIRA, self.BRASK) if e["name"] == params["name"]]
        if "$groups" in cypher and "work.01" in params["frames"]:
            if any("k1" in g["uuids"] for g in params["groups"]):
                return [{"uuid": "e1", "name": "e1"}]
            return []
        if "$pairs" in cypher and "e1" in params["uuids"]:
            return [{**self.BRASK, "event": "e1"}] if any(r == "arg2" for _f, r in params["pairs"]) else []
        return []


class _SmokeDriver:
    """A sync Neo4j driver stand-in for ``run_eval``: the bonfire's ``Roleset`` rows, and no phrases or catalogue
    (a read-only run over a bonfire with nothing taught)."""

    def __init__(self, rolesets: list[str]) -> None:
        self.rolesets = rolesets

    def session(self) -> _SmokeDriver:
        return self

    def __enter__(self) -> _SmokeDriver:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "Roleset" in cypher:
            return [{"roleset": r} for r in self.rolesets]
        return []


@pytest.mark.needs_fcg
def test_the_evaluation_runs_over_the_smoke_set(tmp_path):
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import RuntimeEngine, runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    script = _script()
    engine = RuntimeEngine(PORT, data_dir, out=tmp_path / "engine")
    questions = script.load_questions(SMOKE)
    try:
        rows = asyncio.run(
            script.run_eval(questions, "smoke", engine, SmokeGraph(), driver=_SmokeDriver(["work.01", "publish.01"]))
        )
    finally:
        engine.stop()
    by_id = {r["id"]: r for r in rows}
    assert by_id["k1"]["status"] == "answered" and by_id["k1"]["answer"] == ["Brask Labs"]
    assert by_id["k1"]["answer_shape"] == "where" and by_id["k1"]["comprehension_ms"] is not None
    assert by_id["k2"]["status"] == "repair" and by_id["k2"]["classes"] == ["names nobody"]
    assert all(r["status"] in {"answered", "repair", "unread"} for r in rows) and len(rows) == len(questions)
    summary = script.summarise(rows, {"k1": {"correct": True, "reason": "the graph's employer"}})
    assert "answered precision" in summary and "p50" in summary and "unread" in summary
