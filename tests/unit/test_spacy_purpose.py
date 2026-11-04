"""The spaCy model is chosen per purpose, en_core_web_trf is a pinned component, and a missing trf never fails a turn."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from memory_kernel import setup_runtime
from memory_kernel.grammar.dependencies import spacy_model
from memory_kernel.graph import g1_nlp
from memory_kernel.runtime.api import _plan_trace

ROOT = Path(__file__).resolve().parents[2]


def test_spacy_model_setting_per_purpose(monkeypatch):
    monkeypatch.delenv("MK_TURN_SPACY_MODEL", raising=False)
    monkeypatch.delenv("MK_EXTRACTION_SPACY_MODEL", raising=False)
    assert spacy_model("turn") == "en_core_web_trf"  # one parser for everything said in a turn (R9)
    assert spacy_model("extraction") == "en_core_web_trf"  # R9 for extraction too; _sm is the opt-out
    assert spacy_model("query") == spacy_model("extraction")  # serve reads with the graph's parser

    monkeypatch.setenv("MK_TURN_SPACY_MODEL", "en_core_web_md")
    monkeypatch.setenv("MK_EXTRACTION_SPACY_MODEL", "en_core_web_trf")
    assert (spacy_model("turn"), spacy_model("extraction")) == ("en_core_web_md", "en_core_web_trf")

    monkeypatch.setenv("MK_TURN_SPACY_MODEL", "  ")  # a blank setting is the default, not a model named " "
    assert spacy_model("turn") == "en_core_web_trf"


def test_trf_component_is_not_ok_when_the_model_cannot_load():
    # never OK when it cannot load: the manifest verifies but the load fails (the parser extra is not installed)
    import spacy

    def cannot_load(_name):
        raise ValueError("[E002] Can't find factory for 'curated_transformer'")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(setup_runtime.SpacyModel, "check", lambda self, ctx, entry: setup_runtime.Row(self.name, "ok"))
        patch.setattr(spacy, "load", cannot_load)
        row = setup_runtime.components()["spacy-model-trf"].check(setup_runtime.Context(ROOT), None)
    assert row.status != setup_runtime.OK
    assert "memory-kernel[parser]" in row.detail and "ValueError" in row.detail


@pytest.mark.parametrize("failure", [OSError, ImportError, ValueError, RuntimeError])
def test_missing_trf_falls_back_with_a_note(monkeypatch, caplog, failure):
    monkeypatch.delenv("MK_TURN_SPACY_MODEL", raising=False)
    monkeypatch.setenv(
        "MK_EXTRACTION_SPACY_MODEL", "en_core_web_sm"
    )  # the opt-out: this test is about the turn fallback
    loaded: list[str] = []

    def load(name: str):
        loaded.append(name)
        if name == "en_core_web_trf":
            raise failure("[E002] cannot load en_core_web_trf")
        return f"pipeline:{name}"

    monkeypatch.setattr(g1_nlp, "_load", load)
    g1_nlp.reset_pipelines()
    try:
        assert _plan_trace(1.0).keys() == {"steps", "queries", "comprehension_ms", "execution_ms"}
        with caplog.at_level(logging.WARNING):
            assert g1_nlp.nlp("turn") == "pipeline:en_core_web_sm"
        assert "falls back" in caplog.text and f"{failure.__name__}:" in caplog.text
        assert g1_nlp.nlp("turn") == "pipeline:en_core_web_sm" and loaded == ["en_core_web_trf", "en_core_web_sm"]
        assert g1_nlp.parser_note("turn") == "sm-fallback"
        assert g1_nlp.parser_note("extraction") is None
        assert _plan_trace(1.0).get("parser") == "sm-fallback"
        assert g1_nlp.nlp() == "pipeline:en_core_web_sm"  # extraction never asked for trf, never falls back
    finally:
        g1_nlp.reset_pipelines()
    assert "parser" not in _plan_trace(1.0)

    monkeypatch.setenv("MK_EXTRACTION_SPACY_MODEL", "en_core_web_trf")  # extraction fails loudly, with no note
    with pytest.raises(failure):
        g1_nlp.nlp()
    assert g1_nlp.parser_note("extraction") is None
    with caplog.at_level(logging.WARNING):  # serve's reader never raises over a model: sm, with a note and a warning
        assert g1_nlp.nlp("query") == "pipeline:en_core_web_sm"
    assert g1_nlp.parser_note("query") == "sm-fallback" and g1_nlp.parser_note("extraction") is None
    g1_nlp.reset_pipelines()


def test_serve_index_keys_read_on_the_query_parser_and_the_gpu_is_asked_once_on_success(monkeypatch):
    """``kg_index.load_index`` (serve) renders legacy-env keys with ``purpose="query"``; the build keeps ``extraction``.
    A ``prefer_gpu`` that raises is not remembered as asked."""
    from memory_kernel.search.kg_index import build_keys

    asked: list[str] = []
    real = g1_nlp.nlp
    monkeypatch.setattr(g1_nlp, "nlp", lambda purpose="extraction": asked.append(purpose) or real("query"))
    row = {
        "uuid": "u1",
        "content": "I like tea",
        "sender_name": "Ana",
        "told_at": "2024-02-12T09:20:00+00:00",
        "facts": [],
        "fact_invalid": [],
    }
    build_keys([row], key_render="legacy-env")
    build_keys([row], key_render="legacy-env", purpose="query")
    assert asked and set(asked[: len(asked) // 2]) == {"extraction"} and set(asked[len(asked) // 2 :]) == {"query"}

    calls: list[int] = []

    class Spacy:
        @staticmethod
        def prefer_gpu():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("no driver")
            return False

    monkeypatch.setenv(g1_nlp.GPU_ENV, "1")
    monkeypatch.setattr(g1_nlp, "_gpu_asked", False)
    with pytest.raises(RuntimeError):
        g1_nlp._ask_for_gpu(Spacy)
    g1_nlp._ask_for_gpu(Spacy)
    g1_nlp._ask_for_gpu(Spacy)
    assert len(calls) == 2


def test_one_pipeline_per_model(monkeypatch):
    import threading
    import time

    loads: list[str] = []

    def slow_load(name: str):
        loads.append(name)
        time.sleep(0.05)  # long enough for every thread to be inside the first call
        return object()

    monkeypatch.setattr(g1_nlp, "_load", slow_load)
    monkeypatch.setenv("MK_TURN_SPACY_MODEL", "en_core_web_sm")
    monkeypatch.setenv("MK_EXTRACTION_SPACY_MODEL", "en_core_web_sm")
    g1_nlp.reset_pipelines()
    try:
        got: list[object] = []
        threads = [threading.Thread(target=lambda: got.append(g1_nlp.pipeline("en_core_web_sm"))) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        # three purposes and the grammar accessor share the one copy; two threads never each load one
        copies = {id(x) for x in (*got, g1_nlp.nlp("turn"), g1_nlp.nlp("extraction"), g1_nlp.nlp("query"))}
        assert len(copies) == 1 and loads == ["en_core_web_sm"]
        assert g1_nlp.nlp("turn") is g1_nlp.nlp("extraction")
    finally:
        g1_nlp.reset_pipelines()


def test_parses_never_overlap_whatever_the_threads(monkeypatch):
    """A transformer pipeline run from two threads at once can parse differently from a serial run (borderline
    attributes), and a key rebuilt at open must equal the stored one. Every pipeline the loader hands out holds one
    process-wide lock for each ``nlp(text)`` and each ``nlp.pipe(...)``: no two parses are ever inside it at once."""
    import threading
    import time

    import spacy

    inside: list[int] = []
    peak: list[int] = [0]
    count = threading.Lock()

    class Pipeline:  # a stand-in that records how many parses run at once
        def _parse(self, text):
            with count:
                inside.append(1)
                peak[0] = max(peak[0], len(inside))
            time.sleep(0.02)
            with count:
                inside.pop()
            return text

        def __call__(self, text):
            return self._parse(text)

        def pipe(self, texts):
            return (self._parse(t) for t in texts)

        def make_doc(self, text):
            return text

    monkeypatch.setattr(spacy, "load", lambda name, disable=(): Pipeline())
    parser = g1_nlp._load("en_core_web_stand_in")
    threads = [threading.Thread(target=parser, args=("Kira Dace flies.",)) for _ in range(4)]
    threads += [threading.Thread(target=lambda: list(parser.pipe(["Milo sails.", "Ana rows."]))) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 1 and list(parser.pipe(["Ana rows."])) == ["Ana rows."]


@pytest.mark.parametrize("text", [" ", "\n", "\xa0", "", " \n \xa0"])
def test_whitespace_only_text_is_tokenized_not_parsed(text, monkeypatch):
    """en_core_web_trf raises ValueError ("Shape mismatch for blis.gemm") on whitespace-only text; every entry point
    returns the tokenized Doc instead. Pinned to the small model: the guard sits in the shared loader, whatever the
    model, and the test must not depend on which one a host has."""
    from memory_kernel.envelope.render import render

    monkeypatch.setenv("MK_TURN_SPACY_MODEL", "en_core_web_sm")
    monkeypatch.setenv("MK_EXTRACTION_SPACY_MODEL", "en_core_web_sm")
    docs = [g1_nlp.nlp(p)(text) for p in ("turn", "extraction", "query")]
    docs += [g1_nlp.turn_doc(text), g1_nlp.query_doc(text)]
    assert all(d.text == text and not any(t.dep_ for t in d) for d in docs)
    assert render(text, speaker="Ana", addressee=None, ts="2024-02-12T09:20:00+00:00") == text


@pytest.mark.skipif(
    setup_runtime.installed_version("en_core_web_trf") is None,
    reason="the turn parser (en_core_web_trf) is not installed",
)
def test_bare_command_is_read_behind_a_subject_frame(monkeypatch):
    monkeypatch.delenv("MK_TURN_SPACY_MODEL", raising=False)
    g1_nlp.reset_pipelines()
    try:
        for text in ("hum a tune", "brew", "serenade Ana"):
            doc = g1_nlp.turn_doc(text)
            assert (
                doc.text == text and doc[0].text == text.split()[0]
            )  # the frame word is gone, offsets are the input's
            assert doc[0].pos_ == "VERB" and doc[0].tag_ == "VB", text
            assert [t.idx for t in doc] == [t.idx for t in g1_nlp.nlp("turn").make_doc(text)], text
        # surrounding whitespace is read past, and the Doc is over the text as written: " hum a tune" is "hum a tune"
        for text in (" hum a tune", "  brew\n", "\xa0serenade Ana \n"):
            doc = g1_nlp.turn_doc(text)
            [verb] = [t for t in doc if not t.is_space][:1]
            assert doc.text == text and verb.pos_ == "VERB" and verb.tag_ == "VB", text
            assert [t.idx for t in doc] == [t.idx for t in g1_nlp.nlp("turn").make_doc(text)], text
            assert len(list(doc.sents)) >= 1 and all(0 <= t.head.i < len(doc) for t in doc)
        question = g1_nlp.turn_doc("who hosted the talk?")  # a question has a subject: it is left as parsed
        assert question.text == "who hosted the talk?" and question[0].text == "who"
        assert [t.text for t in question] == [t.text for t in g1_nlp.nlp("turn")("who hosted the talk?")]
    finally:
        g1_nlp.reset_pipelines()
