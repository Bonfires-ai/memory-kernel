"""Construction specs: JSON rows that compile to the runtime grammar's FCG constructions."""

import json
import os

import pytest

from memory_kernel.runtime.constructions import (
    CONSUME,
    CompileError,
    CxnSpec,
    Standing,
    compile_spec,
    precedence,
)

PORT = 9713  # 9711 is the runtime default, 9728 the engine tests'

WHERE_WORK_MEANING = [
    ["find-entity", "?e", "?x"],
    ["identity-lines", "?l", "?e"],
    ["slot-of-kind", "?o", "?l", "Organisation"],
    ["answer", "?o"],
]
X = {"slot": "x", "fills": "np"}


def _where_work(**over) -> CxnSpec:
    """The spike-1 where-work question, written naturally (the compiler normalises it)."""
    row = {
        "id": "where-work",
        "kind": "base",
        "unit": "root",
        "form": ["Where does", X, "work?"],
        "meaning": WHERE_WORK_MEANING,
        "answer_shape": "where",
        "status": "live",
    }
    row.update(over)
    return CxnSpec.from_json(row)


def _filler(id_: str, form: list, meaning: list) -> CxnSpec:
    return CxnSpec.from_json(
        {"id": id_, "kind": "base", "unit": "filler", "fills": "np", "form": form, "meaning": meaning, "status": "live"}
    )


def _bare() -> CxnSpec:
    return _filler("bare-np", [CONSUME], [["name-span", "?x", "?l", "?r"]])


def _the() -> CxnSpec:
    return _filler("the-np", ["the", CONSUME], [["definite", "?x"], ["name-span", "?x", "?l", "?r"]])


def _seq(text: str, left: str, right: str) -> list[str]:
    return ["sequence", json.dumps(text, ensure_ascii=False), left, right]


def test_natural_form_compiles_to_the_normalised_shape():
    """Literals are written naturally and normalised at compile; a root spans sentinel to sentinel."""
    root = compile_spec(_where_work())
    assert root.id == "where-work" and root.unit == "root" and root.cxn.name == "where-work"
    assert root.cxn.conditional_pole == [
        [
            "?where-work-unit",
            {"#meaning": WHERE_WORK_MEANING},
            {"#form": [_seq("⟦ where do ", "?a0-l", "?a0-r"), _seq(" ⟪work⟫ [?] ⟧", "?a1-l", "?a1-r")]},
        ]
    ]
    assert root.cxn.contributing_pole == [
        ["?where-work-unit", {"subunits": ["?slot-x"], "boundaries": ["?a0-l", "?a1-r"]}],
        ["?slot-x", {"referent": "?x", "slot": "np", "boundaries": ["?a0-r", "?a1-l"]}],
    ]
    # the sequence pattern keeps non-ASCII characters as written (an escaped \\u27e6 never matches)
    assert "\\u" not in root.cxn.conditional_pole[0][2]["#form"][0][1]
    # a verb in a literal compiles to its marked lemma, in context, as the utterance will be ("works?" alone stays
    # "works")
    inflected = compile_spec(_where_work(form=["where does", X, "works?"]))
    assert inflected.cxn.conditional_pole[0][2]["#form"][1] == _seq(" ⟪work⟫ [?] ⟧", "?a1-l", "?a1-r")
    # a slot at an edge anchors on the sentinel
    brew = compile_spec(
        CxnSpec(id="brew", kind="taught", unit="root", form=["brew", X], meaning=[["brew", "?b", "?x"]])
    )
    assert brew.cxn.conditional_pole[0][2]["#form"] == [
        _seq("⟦ ⟪brew⟫ ", "?a0-l", "?a0-r"),
        _seq(" ⟧", "?a1-l", "?a1-r"),
    ]
    # a part is not anchored on the sentinels, and never has a slot at its edge (ruling, Task 6b); its edges are token
    # edges (1.5.1 round 2): a space on each side, so it never matches inside a word
    part = compile_spec(CxnSpec(id="x", kind="hand", form=["at", X, "work"], meaning=[["frame", "?x"]]))
    assert part.cxn.conditional_pole[0][2]["#form"] == [
        _seq(" at ", "?a0-l", "?a0-r"),
        _seq(" work ", "?a1-l", "?a1-r"),
    ]
    assert [u[0] for u in part.cxn.contributing_pole] == ["?x-unit", "?slot-x"]  # id "x" and slot "x" stay apart
    with pytest.raises(CompileError, match="edge"):
        compile_spec(CxnSpec(id="x", kind="hand", form=[X, "work"], meaning=[["frame", "?x"]]))

    # a filler mixing literals and the consuming pattern is ONE sequence over the slot (chained parts never apply)
    the = compile_spec(_the())
    assert the.cxn.conditional_pole == [
        [
            "?x-unit",
            {"referent": "?x", "#meaning": [["definite", "?x"], ["name-span", "?x", "?l", "?r"]]},
            {
                "slot": "np",
                "boundaries": ["?l", "?r"],
                "footprints": ["not", "filled"],
                "#form": [_seq("the .{1,200}", "?l", "?r")],
            },
        ]
    ]
    assert the.cxn.contributing_pole == [["?x-unit", {"footprints": ["filled"]}]]
    bare = compile_spec(_bare())
    assert bare.cxn.conditional_pole[0][2]["#form"] == [_seq(".{1,200}", "?l", "?r")]
    assert bare.cxn.attributes["label"] == "filler" and root.cxn.attributes["label"] == "root"


def test_compile_refuses_unsupported_regex_and_category():
    for bad in ("work+?", "(where) does", "where does)", "where\\ does", 'say "hi"', "where ⟦ does", "where ⟪does"):
        with pytest.raises(CompileError, match="unsupported"):
            compile_spec(_where_work(form=[bad, X, "work?"]))
    # the consuming pattern belongs to fillers only, and only as its own part
    with pytest.raises(CompileError, match="filler"):
        compile_spec(_where_work(form=["where does", X, CONSUME, "work?"]))
    with pytest.raises(CompileError, match="filler"):
        compile_spec(_where_work(form=["where does .{1,50}", X, "work?"]))
    with pytest.raises(CompileError, match="category"):
        compile_spec(_where_work(form=["where does", {"slot": "x", "fills": "np", "category": "np"}, "work?"]))
    with pytest.raises(CompileError, match="adjacent"):
        compile_spec(_where_work(form=["where does", X, {"slot": "y", "fills": "np"}, "work?"]))
    with pytest.raises(CompileError, match="fills"):
        compile_spec(CxnSpec.from_json({**_bare().to_json(), "fills": None}))
    with pytest.raises(CompileError, match="meaning"):
        compile_spec(_where_work(meaning=[["answer", "?o"], ["answer", "?l"]]))
    with pytest.raises(CompileError, match=r"\?y"):  # a slot whose referent the meaning never names
        compile_spec(_where_work(form=["where does", X, "work for", {"slot": "y", "fills": "np"}, "?"]))
    with pytest.raises(CompileError, match=r"\?b1"):  # a filler binds only ?l and ?r
        compile_spec(_filler("the-np", ["the", CONSUME], [["name-span", "?x", "?b1", "?r"]]))
    with pytest.raises(CompileError, match="answer"):
        compile_spec(_filler("bare-np", [CONSUME], [["name-span", "?x", "?l", "?r"], ["answer", "?x"]]))


def test_spec_rows_are_type_checked_never_coerced():
    row = _where_work().to_json()
    for field, value in (
        ("form", "where does"),
        ("form", None),
        ("meaning", "answer"),
        ("meaning", ["answer ?o"]),
        ("meaning", [["answer", 3.5]]),
        ("provenance", "teacher"),
        ("provenance", [["teacher", "Ana"]]),
        ("standing", 3),
    ):
        with pytest.raises(CompileError):
            CxnSpec.from_json({**row, field: value})
    with pytest.raises(CompileError, match="kind"):
        CxnSpec(id="x", kind="guessed", form=["hello"], meaning=[["greet", "?g"]])  # type: ignore[arg-type]


def test_precedence_orders_base_first_and_taught_last():
    def spec(id_, kind, ok=0, failed=0, confirmed=0):
        return CxnSpec(
            id=id_,
            kind=kind,
            form=["hello"],
            meaning=[["greet", "?g"]],
            standing=Standing(confirmed=confirmed, uses_ok=ok, uses_failed=failed),
        )

    specs = [
        spec("t", "taught"),
        spec("h", "hand"),
        spec("m-weak", "mined", ok=1, failed=3),
        spec("m-strong", "mined", ok=5, failed=1),
        spec("m-tied-b", "mined", ok=2, confirmed=1),
        spec("m-tied-a", "mined", ok=2, confirmed=1),
        spec("m-confirmed", "mined", ok=3, failed=1, confirmed=2),
        spec("b", "base", ok=99),
    ]
    order = [s.id for s in sorted(specs, key=precedence)]
    # ascending load order: the engine's recency tie-break lands on the highest precedence
    assert order == ["b", "m-weak", "m-tied-a", "m-tied-b", "m-confirmed", "m-strong", "h", "t"]
    assert precedence(spec("x", "taught", ok=1, failed=2, confirmed=3)) == (3, 0, -1, 3, "x")
    # specificity before standing: a well-used general root never overtakes a more specific one of its kind
    x, r = {"slot": "x", "fills": "np"}, {"slot": "r", "fills": "np"}
    general = CxnSpec(
        id="brew-x",
        kind="taught",
        unit="root",
        form=["brew", x],
        meaning=[["greet", "?x"]],
        standing=Standing(confirmed=1, uses_ok=10),
    )
    specific = CxnSpec(
        id="brew-x-for-r",
        kind="taught",
        unit="root",
        form=["brew", x, "for", r],
        meaning=[["greet", "?x", "?r"]],
        standing=Standing(confirmed=1),
    )
    assert [s.id for s in sorted([specific, general], key=precedence)] == ["brew-x", "brew-x-for-r"]
    # among fillers, the more specific (more literal characters) loads later and wins, whatever the id
    a_np = _filler("a-np", ["a", CONSUME], [["name-span", "?x", "?l", "?r"]])
    assert [s.id for s in sorted([_the(), a_np, _bare()], key=precedence)] == ["bare-np", "a-np", "the-np"]


def test_spec_json_round_trip():
    spec = _where_work(standing={"confirmed": 1, "uses_ok": 4, "uses_failed": 1}, provenance={"teacher": "Ana"})
    row = spec.to_json()
    assert json.loads(json.dumps(row)) == row
    assert CxnSpec.from_json(row) == spec
    assert row["standing"] == {"confirmed": 1, "uses_ok": 4, "uses_failed": 1, "uses_world": 0}
    with pytest.raises(CompileError, match="kind"):
        CxnSpec.from_json({**row, "kind": "guessed"})


def _data_dir():
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.mark.needs_fcg
def test_compiled_where_work_reads(tmp_path):
    from memory_kernel.runtime.engine import RuntimeEngine
    from memory_kernel.runtime.normalise import normalise
    from memory_kernel.runtime.program import Program, Var

    roots = [
        _where_work(),
        _where_work(id="where-work-plain", form=["where does", X, "work"]),  # no final punctuation
        CxnSpec(
            id="brew", kind="taught", unit="root", form=["brew", X], meaning=[["brew", "?b", "?x"], ["answer", "?b"]]
        ),
        _where_work(id="job-of", form=["what is", X, "’s job?"]),  # a non-ASCII literal (curly apostrophe)
    ]
    specs = sorted([_the(), *roots, _bare()], key=precedence)
    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path)
    engine.start()
    try:
        engine.load([compile_spec(s).cxn for s in specs])

        def read(text: str):
            norm = normalise(text)
            reading = engine.comprehend(norm.text)
            assert reading is not None and reading.complete is True, text
            [(left, right)] = [p[2:] for p in reading.meaning if p[0] == "name-span"]
            return reading, norm.spans.cut(left, right)

        reading, name = read("where does Kira Dace work?")
        assert reading.applied == ("where-work", "bare-np") and name == "Kira Dace"
        program = Program.from_meaning(reading.meaning)  # int offsets are constants
        ops = {op.name: op for op in program.ops}
        assert set(ops) == {"find-entity", "identity-lines", "slot-of-kind", "name-span"}
        assert isinstance(program.answer, Var) and program.answer == ops["slot-of-kind"].args[0]

        reading, name = read("where does Kira Dace work")  # no final punctuation: the right sentinel anchors
        assert reading.applied == ("where-work-plain", "bare-np") and name == "Kira Dace"
        reading, name = read("brew a tea")  # a slot at the right edge, anchored on the sentinel
        assert reading.applied == ("brew", "bare-np") and name == "tea"  # the cut drops the determiner (Task 6b)
        reading, name = read("what is Kira Dace’s job?")
        assert reading.applied == ("job-of", "bare-np") and name == "Kira Dace"
        reading, name = read("where does the team work?")  # the more specific filler loads later and wins
        assert reading.applied == ("where-work", "the-np") and name == "team"  # determiner dropped at cut
        assert ["definite", "?x"] not in reading.meaning and any(p[0] == "definite" for p in reading.meaning)

        partial = engine.comprehend(normalise("what where does Kira Dace work?").text)
        assert partial is None or partial.complete is False
    finally:
        engine.stop()
