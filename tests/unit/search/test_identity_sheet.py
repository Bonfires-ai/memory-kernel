"""The entity sheet keeps an entity's identity lines (copular nominal statements about it) before its recent lines."""

from dataclasses import replace

import pytest

from memory_kernel.search import summary_lines
from memory_kernel.search.kg_context import ContextConfig, _sheet, identity_line
from memory_kernel.search.kg_reader import SameAsComponent, _person_sheet
from memory_kernel.search.summary_lines import SheetLine


def _tokens_ready():
    try:
        from memory_kernel.search.kg_context import count_tokens

        count_tokens("x")
        return True
    except Exception:
        return False


needs_tokens = pytest.mark.skipif(not _tokens_ready(), reason="token encoder not installed (mk setup)")


@pytest.fixture(autouse=True)
def _extraction_model_is_missing(monkeypatch):
    """serve reads on the query policy: a configured model that will not load (a bare install: no trf, no [parser])
    falls back to en_core_web_sm with a note instead of an empty context. Forced here, so these never skip."""
    from memory_kernel.graph import g1_nlp

    monkeypatch.setenv("MK_EXTRACTION_SPACY_MODEL", "en_core_web_missing")
    g1_nlp.reset_pipelines()
    yield
    g1_nlp.reset_pipelines()


def test_identity_line_is_a_copular_nominal_statement_about_the_entity():
    assert identity_line("[Kira Dace, 2016-12-27] Kira Dace is a security researcher at Zorvex.", "Kira Dace")
    assert not identity_line("[Kira Dace, 2016-12-27] Kira Dace thinks this is it.", "Kira Dace")
    assert not identity_line("[Kira Dace, 2016-12-27] We see that the pointer can be overwritten.", "Kira Dace")
    assert identity_line("[Kira Dace, 2016-12-27] I am a researcher at Zorvex.", "Kira Dace")
    assert not identity_line("[Bo Lin, 2016-12-27] I am a researcher at Zorvex.", "Kira Dace")
    assert not identity_line("[Bo Lin, 2016-12-27] Kira Dace's brother is a pilot.", "Kira Dace")
    assert identity_line("[Bo Lin, 2016-12-27] Jean-Paul Vorn is a captain.", "Jean-Paul Vorn")


def _entity(lines, identity=()):
    recs = [SheetLine(f"l{i}", ln, ln in identity) for i, ln in enumerate(lines)]
    return [{"name": "Kira Dace", "summary": "\n".join(lines), "lines": recs}]


@pytest.fixture
def no_parse(monkeypatch):
    """Serving parses only the question: a stored line is never parsed."""
    from memory_kernel.graph import g1_nlp

    def boom(*_a, **_k):
        raise AssertionError("a stored line was parsed at question time")

    monkeypatch.setattr(g1_nlp, "query_doc", boom)


@needs_tokens
def test_a_long_talking_entity_keeps_its_identity_line_first(no_parse):
    lines = ["[Kira Dace, 2016-12-27] Kira Dace is a security researcher at Zorvex."]
    lines += [f"[Kira Dace, 2016-12-27] The exploit chain uses step {i}." for i in range(2, 21)]
    out = _sheet(_entity(lines, identity=lines[:1]), ContextConfig(), 10_000, set())
    assert out[0] == "Kira Dace"
    assert out[1] == lines[0]  # the identity line comes from the stored record, not a parse
    assert len(out) == 1 + 12
    multi = [SheetLine("m", "[Kira Dace, 2016-12-28] Kira Dace is a pilot.\nShe flies.", True)]
    assert _sheet([{"name": "Kira Dace", "lines": multi}], ContextConfig(), 10_000, set())[1] == (
        "[Kira Dace, 2016-12-28] Kira Dace is a pilot. She flies."  # embedded newlines render as spaces
    )


@needs_tokens
def test_an_entity_without_an_identity_line_renders_as_before(no_parse):
    lines = [f"[Kira Dace, 2016-12-{d:02d}] Kira Dace ran test {d}." for d in range(1, 21)]
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    now = _sheet(_entity(lines), ContextConfig(), 10_000, seen_a)
    before = _sheet(_entity(lines), replace(ContextConfig(), identity_lines_cap=0), 10_000, seen_b)
    assert now == before == ["Kira Dace", *lines[-12:]]
    assert seen_a == seen_b


def test_legacy_rows_read_recency_only_and_same_as_merges_without_a_parse(no_parse, caplog, monkeypatch):
    monkeypatch.setattr(summary_lines, "_NOTED", set())
    # runs offline (CI): no token encoder needed for the budget
    monkeypatch.setattr("memory_kernel.search.kg_context.count_tokens", lambda text: len(text.split()))
    legacy = {"name": "Kira Dace", "group_id": "g9", "lines": None, "summary": "[Ana, 2016-12-01] Kira Dace is old."}
    with caplog.at_level("WARNING"):
        out = _sheet([legacy, legacy], ContextConfig(), 10_000, set())
    assert out == ["Kira Dace", "[Ana, 2016-12-01] Kira Dace is old."]  # recency only, no parse
    assert sum("identity lines unavailable" in r.message for r in caplog.records) == 1  # one note per group
    rec = {
        "summary": "x",
        "group_id": "g8",
        "lines": [SheetLine("a", "Ana is a pilot.", True), SheetLine("b", "Ana flew.", False)],
    }
    old = {"summary": "Ana flew.\nDee is a cook.", "group_id": "g8", "lines": None}
    comp = SameAsComponent(id="c", name="Ana", members=("m1", "m2"), also=("Dee",))
    caplog.clear()
    with caplog.at_level("WARNING"):
        sheet = _person_sheet("c", {"m1": rec, "m2": old}, {"c": comp})
        _person_sheet("c", {"m1": rec, "m2": old}, {"c": comp})
    assert sum("identity lines unavailable" in r.message for r in caplog.records) == 1  # mixed: one note
    assert [(ln.text, ln.identity) for ln in sheet["lines"]] == [
        ("Ana is a pilot.", True),
        ("Ana flew.", False),
        ("Dee is a cook.", False),
    ]


def test_query_time_tense_reads_on_the_fallback_too():
    """The tense reading under ``content_with_relative_time`` (the bare-weekday direction) is serve's reader as well: a
    missing model reads it on sm instead of degrading to no tense."""
    from memory_kernel.grammar.semantic_operators import get_semantic_operator_service
    from memory_kernel.pipeline import inprocess

    inprocess.clear_build_caches()  # the memoized load and parses of an earlier test
    try:
        assert get_semantic_operator_service().profile("We will meet on Friday at the lab.").tense_aspects == (
            "future",
        )
    finally:
        inprocess.clear_build_caches()
