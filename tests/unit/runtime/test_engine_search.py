"""The unit grammar's pinned search and the kernel Lisp extensions, against the pinned FCG Go.

The constructions are hand-written JSON lifted from the CLEVR search spike's grammar (the unit compiler is a later
task); each one's triggers are read off its own conditional pole, as the compiler derives them. The units are built
directly in the shape ``comprehend_units`` documents.
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
import subprocess
import time

import pytest

PORT = 9720  # one engine at a time; 9711-9719 are the runtime default's, mk serve's and the other runtime tests'
PINNED_IMAGE = "2cc1f08127cf"
TIERS = ("lex", "alex", "nom", "np", "vp", "cl", "q", "co")
S = json.dumps


# -- constructions (spike grammar, JSON route) ------------------------------------------------------------------------


def _not(*xs: str) -> list:
    return ["not", *xs]


def _u(name: str, merge: dict | None = None, match: dict | None = None) -> list:
    """A conditional unit: (name, formulation lock = merged in comprehension, comprehension lock = matched)."""
    return [name, merge or {}, match or {}]


def _lit(unit: str, word: str) -> list:
    return ["string", unit, S(word)]


def _triggers(cond: list) -> list[str]:
    """One trigger read off the conditional pole: a literal it consumes, else a lexical class it needs."""
    lits, cats = [], []
    for unit in cond:
        for pole in unit[1:]:
            for key, value in pole.items():
                if key == "#form":
                    for p in value:
                        if p[0] == "string" and p[2].startswith('"'):
                            lits.append(json.loads(p[2]))
                        if p[0] in ("cat", "acat") and p[2].startswith('"'):
                            cats.append(f"{p[0]}:{json.loads(p[2])}")
                if key == "lexcls" and value.startswith('"'):
                    cats.append("cat:" + json.loads(value))
    return [S(lits[0])] if lits else [S(cats[0])] if cats else []


def _cxn(name: str, label: str, cond: list, contrib: list) -> dict:
    return {
        "name": f"{name}-cxn",
        "conditional-pole": cond,
        "contributing-pole": contrib,
        "attributes": {"label": label, "score": 0.5, "triggers": _triggers(cond)},
    }


LEMMA = {"noun": "nlemma", "verb": "vlemma", "adj": "lemma", "prep": "lemma"}


def _grammar() -> list[dict]:
    """Lexical classes, nominals, noun phrases, a verb phrase, the inverted clause, where-questions and the
    utterance: enough for "where does Kira Dace work?" and for a long nominal run to branch."""
    out = []

    def base(cat):
        return [["string", "?w", "?s"], ["cat", "?w", S(cat)], ["span", "?w", "?l", "?r"]]

    for cat in ("noun", "adj", "prep", "det"):
        form = base(cat) + ([[LEMMA[cat], "?w", "?lem"]] if cat in LEMMA else [])
        lemma = "?lem" if cat in LEMMA else S("-")
        out.append(
            _cxn(
                f"lex-{cat}",
                "lex",
                [_u("?w", {}, {"#form": form})],
                [["?w", {"lexcls": S(cat), "lemma": lemma, "bounds": ["?l", "?r"]}]],
            )
        )
    out.append(
        _cxn(
            "lex-verb",
            "lex",
            [
                _u(
                    "?w",
                    {"#meaning": [["frames-of", "?fs", "?lem"]]},
                    {"#form": base("verb") + [["vlemma", "?w", "?lem"]]},
                )
            ],
            [["?w", {"lexcls": S("verb"), "fs": "?fs", "bounds": ["?l", "?r"]}]],
        )
    )
    out.append(
        _cxn(
            "lex-name",
            "lex",
            [
                _u(
                    "?w",
                    {"#meaning": [["name-span", "?x", "?l", "?r"], ["find-entity", "?e", "?x"]]},
                    {"#form": base("name")},
                )
            ],
            [["?w", {"lexcls": S("name"), "ref": "?e", "bounds": ["?l", "?r"]}]],
        )
    )
    out.append(
        _cxn(
            "base-nominal",
            "nom",
            [
                _u("?n", {"#meaning": [["filter", "?t", "?s", "?lem"]]}),
                _u(
                    "?w",
                    {"lemma": "?lem", "bounds": ["?l", "?r"]},
                    {"lexcls": S("noun"), "footprints": _not("nominal")},
                ),
            ],
            [
                [
                    "?n",
                    {
                        "phrase": S("nominal"),
                        "ref": "?t",
                        "src": "?s",
                        "lm": "?w",
                        "rm": "?w",
                        "bounds": ["?l", "?r"],
                        "subunits": ["?w"],
                    },
                ],
                ["?w", {"footprints": ["nominal"]}],
            ],
        )
    )
    for cls in ("adj", "noun"):
        out.append(
            _cxn(
                f"mod-{cls}-nominal",
                "nom",
                [
                    _u("?sup", {"#meaning": [["filter", "?t", "?mid", "?lem"]]}),
                    _u(
                        "?a",
                        {"lemma": "?lem", "bounds": ["?al", "?ar"]},
                        {"lexcls": S(cls), "footprints": _not("nominal")},
                    ),
                    _u(
                        "?n",
                        {"ref": "?mid", "src": "?s", "bounds": ["?nl", "?nr"], "rm": "?nrm"},
                        {
                            "phrase": S("nominal"),
                            "lm": "?nlm",
                            "footprints": _not("nominal"),
                            "#form": [["meets", "?a", "?nlm"]],
                        },
                    ),
                ],
                [
                    [
                        "?sup",
                        {
                            "phrase": S("nominal"),
                            "ref": "?t",
                            "src": "?s",
                            "lm": "?a",
                            "rm": "?nrm",
                            "bounds": ["?al", "?nr"],
                            "subunits": ["?a", "?n"],
                        },
                    ],
                    ["?a", {"footprints": ["nominal"]}],
                    ["?n", {"footprints": ["nominal"]}],
                ],
            )
        )

    def np_out(ref, lm, rm, left, right, sub):
        return {"phrase": S("np"), "ref": ref, "lm": lm, "rm": rm, "bounds": [left, right], "subunits": sub}

    out.append(
        _cxn(
            "det-np",
            "np",
            [
                _u("?np", {"#meaning": [["name-span", "?x", "?l", "?r"], ["find-entity", "?cand", "?x"]]}),
                _u("?d", {}, {"lexcls": S("det"), "footprints": _not("np")}),
                _u(
                    "?n",
                    {"ref": "?t", "src": "?cand", "bounds": ["?l", "?r"], "rm": "?nrm"},
                    {
                        "phrase": S("nominal"),
                        "lm": "?nlm",
                        "footprints": _not("nominal", "np"),
                        "#form": [["meets", "?d", "?nlm"]],
                    },
                ),
            ],
            [
                ["?np", np_out("?t", "?d", "?nrm", "?l", "?r", ["?d", "?n"])],
                ["?d", {"footprints": ["np"]}],
                ["?n", {"footprints": ["np"]}],
            ],
        )
    )
    out.append(
        _cxn(
            "name-np",
            "np",
            [
                _u("?np"),
                _u(
                    "?w",
                    {"ref": "?e", "bounds": ["?l", "?r"]},
                    {"lexcls": S("name"), "footprints": _not("np", "nominal")},
                ),
            ],
            [["?np", np_out("?e", "?w", "?w", "?l", "?r", ["?w"])], ["?w", {"footprints": ["np"]}]],
        )
    )
    out.append(
        _cxn(
            "base-vp",
            "vp",
            [
                _u("?vp", {"#meaning": [["events-of", "?e0", "?fs"]]}),
                _u("?v", {"fs": "?fs"}, {"lexcls": S("verb"), "footprints": _not("vp")}),
            ],
            [
                ["?vp", {"phrase": S("vp"), "fs": "?fs", "ev": "?e0", "lm": "?v", "rm": "?v", "subunits": ["?v"]}],
                ["?v", {"footprints": ["vp"]}],
            ],
        )
    )
    out.append(
        _cxn(
            "cl-inv",
            "cl",
            [
                _u(
                    "?cl",
                    {"#meaning": [["relate", "?e2", "?e1", "?s", "arg0"]]},
                    {"#form": [_lit("?aux", "do"), ["meets", "?aux", "?nlm"], ["meets", "?nrm", "?vlm"]]},
                ),
                _u(
                    "?vp",
                    {"fs": "?fs", "ev": "?e1", "rm": "?vrm"},
                    {"phrase": S("vp"), "lm": "?vlm", "footprints": _not("vp")},
                ),
                _u("?np", {"ref": "?s", "rm": "?nrm"}, {"phrase": S("np"), "lm": "?nlm", "footprints": _not("arg")}),
            ],
            [
                [
                    "?cl",
                    {
                        "phrase": S("clause"),
                        "fs": "?fs",
                        "ev": "?e2",
                        "lm": "?aux",
                        "rm": "?vrm",
                        "subunits": ["?vp", "?np"],
                        "subj": "?s",
                        "voice": S("active"),
                    },
                ],
                ["?vp", {"footprints": ["vp"]}],
                ["?np", {"footprints": ["arg"]}],
            ],
        )
    )
    where = [
        ["wh-role", "?wr", "?fs", "where"],
        ["role-filler", "?byev", "?ev", "?wr", "?s"],
        ["role-kinds", "?k", "?wr"],
        ["state-lines", "?l", "?s", "?fs"],
        ["slot-of-kind", "?bys", "?l", "?k"],
        ["either", "?ans", "?byev", "?bys"],
        ["answer", "?ans"],
    ]
    out.append(
        _cxn(
            "q-where",
            "q",
            [
                _u("?q", {"#meaning": where}, {"#form": [_lit("?wh", "where"), ["meets", "?wh", "?clm"]]}),
                _u(
                    "?cl",
                    {"ev": "?ev", "fs": "?fs", "lm": "?clm", "rm": "?crm", "subj": "?s"},
                    {"phrase": S("clause"), "footprints": _not("q")},
                ),
            ],
            [
                ["?q", {"phrase": S("question"), "topic": "?s", "lm": "?wh", "rm": "?crm", "subunits": ["?cl"]}],
                ["?cl", {"footprints": ["q"]}],
            ],
        )
    )
    out.append(
        _cxn(
            "utt-question",
            "co",
            [
                _u("?utt", {}, {"#form": [_lit("?qm", "?"), ["meets", "?qrm", "?qm"]]}),
                _u("?q", {"rm": "?qrm"}, {"phrase": S("question"), "footprints": _not("co", "utt")}),
            ],
            [["?utt", {"phrase": S("utt"), "subunits": ["?q"]}], ["?q", {"footprints": ["utt"]}]],
        )
    )
    return out


def _lexicon(n: int) -> list[dict]:
    """``n`` per-lemma lexical constructions (the lexicon the kernel never generates), keyed by lemmas no turn has."""
    return [
        {
            "name": f"zq{i:05d}-lexv-cxn",
            "conditional-pole": [
                [
                    "?w",
                    {"#meaning": [["frames-of", "?fs", f"zq{i:05d}"]]},
                    {"#form": [["string", "?w", S(f"zq{i:05d}")], ["cat", "?w", S("verb")]]},
                ]
            ],
            "contributing-pole": [["?w", {"lexcls": S("verb"), "fs": "?fs"}]],
            "attributes": {"label": "lex", "score": 0.5, "triggers": [S(f"lem:zq{i:05d}")]},
        }
        for i in range(n)
    ]


# -- units ------------------------------------------------------------------------------------------------------------


def _units(*words: tuple) -> list[dict]:
    """``(string, cat, *lemmas)`` per word, offsets as if the words were joined by single spaces."""
    units, at = [], 0
    for string, cat, *lemmas in words:
        units.append({"string": string, "cat": cat, "lemmas": tuple(lemmas), "start": at, "end": at + len(string)})
        at += len(string) + 1
    return units


WHERE_WORK = _units(("where", "lit"), ("do", "lit"), ("kira dace", "name"), ("work", "verb", "work"), ("?", "lit"))
#: Review Focus 2: a long turn (20 units) that never reads. The nominal run has many bracketings, and the "where" of
#: its second half has no clause to take, so every one of them fails
LONG = _units(
    ("where", "lit"),
    ("do", "lit"),
    ("the", "det"),
    *[(w, "adj", w) for w in ("old", "red", "big", "new", "bold", "dark")],
    *[(w, "noun", w) for w in ("spyware", "company", "data", "report", "team", "lab", "chain", "office")],
    ("work", "verb", "work"),
    ("where", "lit"),
    ("?", "lit"),
)


# -- engine -----------------------------------------------------------------------------------------------------------


def _tctl() -> float | None:
    if shutil.which("sensors") is None:
        return None
    found = re.search(r"Tctl:\s*\+?([0-9.]+)", subprocess.run(["sensors"], capture_output=True, text=True).stdout)
    return float(found.group(1)) if found else None


def _thermal_gate(high: float = 88.0, low: float = 80.0) -> None:
    """Before an engine batch: above ``high`` C, wait until the CPU is below ``low`` C."""
    t = _tctl()
    if t is not None and t > high:
        while t is not None and t > low:
            time.sleep(20)
            t = _tctl()


def _data_dir():
    import os

    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.fixture
def make_engine(tmp_path):
    """Start an engine with the given settings; never two of ours at once, every one stopped by pid."""
    from memory_kernel.runtime.engine import RuntimeEngine

    data_dir = _data_dir()
    started: list[RuntimeEngine] = []

    def make(**settings) -> RuntimeEngine:
        for engine in started:
            engine.stop()
        _thermal_gate()
        engine = RuntimeEngine(PORT, data_dir, out=tmp_path / "engine", **settings)
        started.append(engine)
        engine.start()
        return engine

    try:
        yield make
    finally:
        for engine in started:
            engine.stop()


def _cpu_after(engine, window: float = 0.5) -> float:
    """The engine's own CPU seconds (/proc/<pid>/stat utime + stime, all threads) over ``window`` after a reply."""
    from memory_kernel.pipeline.engine import proc_cputime

    before = proc_cputime(engine.pid)
    time.sleep(window)
    return proc_cputime(engine.pid) - before


def _lisp(engine, code: str):
    return engine._post("/fcg-read-eval-print", {"lisp-code": code, "package": "fcg"})


def _shape(meaning: list) -> list[str]:
    """A meaning network up to variable renaming (Weisfeiler-Lehman colour refinement over its variables)."""
    import hashlib

    def var(x):
        return isinstance(x, str) and x.startswith("?")

    names = {x for p in meaning for x in p[1:] if var(x)}
    colour = dict.fromkeys(names, "")
    for _ in range(6):
        colour = {
            v: hashlib.sha1(
                json.dumps(
                    sorted(
                        json.dumps([p[0], i, [colour[x] if var(x) else x for x in p[1:]]])
                        for p in meaning
                        for i, x in enumerate(p[1:])
                        if x == v
                    )
                ).encode()
            ).hexdigest()
            for v in names
        }
    return sorted(json.dumps([p[0], *(colour[x] if var(x) else x for x in p[1:])]) for p in meaning)


# -- tests ------------------------------------------------------------------------------------------------------------


@pytest.mark.needs_fcg
def test_extensions_load_against_the_pinned_image(make_engine):
    from memory_kernel.pipeline.engine import sha_file
    from memory_kernel.runtime.engine import KERNEL_EXT_IMAGE, runtime_binary

    assert KERNEL_EXT_IMAGE == PINNED_IMAGE
    assert sha_file(runtime_binary(_data_dir()))[:12] == PINNED_IMAGE  # the image under test is the pinned one
    engine = make_engine()
    engine.load(_grammar(), tiers=TIERS)
    # read back from the inventory (a hashed set's init forces its own supplier and hash mode)
    keys = {
        "construction-inventory-processor-mode": ":HEURISTIC-SEARCH",
        "search-algorithm": ":DEPTH-FIRST",
        "node-expansion-mode": ":DEFAULT",
        "cxn-supplier-mode": ":HASHED-ORDERED-BY-LABEL",
        "hash-mode": ":MK-TRIGGER",
        "parse-order": "(FCG-GO::LEX FCG-GO::ALEX FCG-GO::NOM FCG-GO::NP FCG-GO::VP FCG-GO::CL FCG-GO::Q FCG-GO::CO)",
        "cxn-sets-with-sequential-application": "(FCG-GO::LEX)",  # the route interns labels there; compared by name
        "node-tests": "(:MK-BUDGET :MK-CANONICAL)",
        "max-nr-of-nodes": "150",
        "parse-goal-tests": "(:NO-APPLICABLE-CXNS :MK-NO-STRINGS-IN-ROOT :CONNECTED-SEMANTIC-NETWORK :CONNECTED-STRUCTURE)",
        "de-render-mode": ":DE-RENDER-RAW",
        "render-mode": ":RENDER-STRING-MEETS-PRECEDES",
        "form-predicates": "(MEETS)",
        "shuffle-cxns-before-application": "NIL",
    }
    readback = " ".join(f'(format nil "~s" (fcg::get-configuration inv :{k}))' for k in keys)
    got = _lisp(engine, f"(let ((inv fcg-go::mk-runtime-units)) (list {readback}))")
    assert dict(zip(keys, got, strict=True)) == keys
    reading = engine.comprehend_units(WHERE_WORK)
    assert reading.complete and reading.refused is None and reading.budget is None
    assert reading.applied == (  # a tier's constructions are offered in the turn's trigger order
        "lex-verb-cxn",
        "lex-name-cxn",
        "name-np-cxn",
        "base-vp-cxn",
        "cl-inv-cxn",
        "q-where-cxn",
        "utt-question-cxn",
    )
    [(left, right)] = [p[2:] for p in reading.meaning if p[0] == "name-span"]
    assert (left, right) == (9, 18)  # the name unit's offsets, through the raw form's span predicate


@pytest.mark.needs_fcg
def test_hash_has_no_nil_bucket(make_engine):
    engine = make_engine()
    cxns = _grammar()
    assert all(c["attributes"]["triggers"] for c in cxns)
    engine.load(cxns, tiers=TIERS)
    nil_bucket, keyed = _lisp(
        engine,
        "(let ((inv fcg-go::mk-runtime-units))"
        " (list (length (gethash nil (fcg::constructions-hash-table inv)))"
        " (loop for c in (fcg::constructions-list inv)"
        "  collect (list (string-downcase (symbol-name (fcg::name c))) (fcg::hash c :mk-trigger)))))",
    )
    assert nil_bucket == 0
    assert sorted(keyed) == sorted([c["name"], [json.loads(t) for t in c["attributes"]["triggers"]]] for c in cxns)
    # a construction with no trigger (or none of the tiers) is refused before anything reaches the engine
    bare = {**cxns[0], "name": "bare-cxn", "attributes": {"label": "lex", "score": 0.5}}
    with pytest.raises(ValueError, match="bare-cxn.*no triggers"):
        engine.load([*cxns, bare], tiers=TIERS)
    stray = {**cxns[0], "name": "stray-cxn", "attributes": {**cxns[0]["attributes"], "label": "taught"}}
    with pytest.raises(ValueError, match="stray-cxn.*taught"):
        engine.load([*cxns, stray], tiers=TIERS)
    assert engine.comprehend_units(WHERE_WORK).complete  # the loaded grammar is untouched
    # and the engine's hash mode refuses one too, should a construction reach it without triggers
    with pytest.raises(Exception, match="NIL hash key"):
        engine._add_unit({**bare, "attributes": {"label": "lex", "score": 0.5}})


@pytest.mark.needs_fcg
def test_no_strings_in_root_goal(make_engine):
    engine = make_engine()
    engine.load(_grammar(), tiers=TIERS)
    clean = engine.comprehend_units(WHERE_WORK)
    assert clean.complete and clean.left == ()
    extra = _units(*[(u["string"], u["cat"], *u["lemmas"]) for u in WHERE_WORK], ("?", "lit"))
    reading = engine.comprehend_units(extra)
    assert reading.left == ("?",) and reading.complete is False and reading.refused is None
    # positive control: FCG Go's own goal tests accept that reading; only :mk-no-strings-in-root refuses it
    stock_goals = [":no-applicable-cxns", ":connected-semantic-network", ":connected-structure"]
    engine._post(
        "/fcg-set-grammar-configuration",
        {"key": "parse-goal-tests", "value": stock_goals, "grammar": "mk-runtime-units"},
    )
    stock = engine.comprehend_units(extra)
    assert stock.complete is True and stock.left == ("?",)


@pytest.mark.needs_fcg
def test_static_refusal_names_the_word(make_engine, monkeypatch):
    """Review Focus 1: a word no construction can consume is refused at 0 nodes, named, before any search."""
    engine = make_engine()
    engine.load(_grammar(), tiers=TIERS)
    publicly = _units(
        ("where", "lit"),
        ("do", "lit"),
        ("kira dace", "name"),
        ("work", "verb", "work"),
        ("publicly", "lit"),
        ("?", "lit"),
    )
    emoji = _units(
        ("where", "lit"),
        ("do", "lit"),
        ("kira dace", "name"),
        ("work", "verb", "work"),
        ("\U0001f600", "lit"),
        ("?", "lit"),
    )
    url = _units(
        ("where", "lit"), ("do", "lit"), ("https://zorvex.example", "lit"), ("work", "verb", "work"), ("?", "lit")
    )
    pron = _units(("where", "lit"), ("do", "lit"), ("she", "pron"), ("work", "verb", "work"), ("?", "lit"))
    assert engine.comprehend_units(WHERE_WORK).complete  # positive control: without the word it reads

    def no_engine(*args, **kwargs):
        raise AssertionError("a static refusal never reaches the engine")

    monkeypatch.setattr(engine, "_request", no_engine)
    for units, word in (
        (publicly, '"publicly"'),
        (emoji, '"\U0001f600"'),
        (url, '"https://zorvex.example"'),
        (pron, '"she"'),
    ):
        reading = engine.comprehend_units(units)
        assert reading.refused is not None and word in reading.refused, reading.refused
        assert (reading.nodes, reading.complete, reading.meaning, reading.applied) == (0, False, [], ())
    both = engine.comprehend_units(_units(("to", "lit"), *[(u["string"], u["cat"], *u["lemmas"]) for u in publicly]))
    assert both.refused == 'no construction can consume "to", "publicly"'  # every such word, in turn order


@pytest.mark.needs_fcg
def test_node_cap_refuses_before_the_deadline(make_engine):
    engine = make_engine(node_cap=150, deadline_ms=10_000)
    engine.load(_grammar(), tiers=TIERS)
    reading = engine.comprehend_units(LONG)
    assert reading.budget == "nodes" and reading.complete is False and reading.refused is None
    assert 150 <= reading.nodes <= 152  # the cap, plus at most one expansion
    assert reading.ms < 3_000  # far inside the deadline: the cap refuses, not the clock
    assert _cpu_after(engine) == 0.0


@pytest.mark.needs_fcg
def test_failing_parse_stops_inside_the_engine(make_engine):
    """Review Focus 2: with a cap it cannot reach, the deadline ends the search inside the engine, and nothing keeps
    searching after the reply (the stock node cap only stops enqueueing, and a client timeout cancels nothing)."""
    engine = make_engine(node_cap=1_000_000, deadline_ms=800)
    engine.load(_grammar(), tiers=TIERS)
    assert len(LONG) == 20
    started = time.perf_counter()
    reading = engine.comprehend_units(LONG)
    wall = (time.perf_counter() - started) * 1000
    assert reading.budget == "time" and reading.complete is False
    assert reading.nodes > 150  # positive control: it searched past the default cap before the clock stopped it
    assert wall < 800 + 700  # the deadline, one node expansion and the round trip
    assert _cpu_after(engine) == 0.0


@pytest.mark.needs_fcg
def test_canonical_order_prunes_and_loses_nothing(make_engine):
    """:mk-canonical keeps one order of a tier's independent applications: the same reading in fewer nodes, and an
    exhausted search (no budget limit tripped) explores fewer nodes. Positive control: the same turns without it."""
    engine = make_engine(node_cap=5000, deadline_ms=5000)
    engine.load(_grammar(), tiers=TIERS)
    run = [("where", "lit"), ("do", "lit"), ("the", "det"), *[(w, "noun", w) for w in ("data", "report", "team")]]
    complete = _units(*run, ("work", "verb", "work"), ("?", "lit"))
    failing = _units(*run, ("work", "verb", "work"), ("where", "lit"), ("?", "lit"))

    def search(node_tests):
        engine._post(
            "/fcg-set-grammar-configuration", {"key": "node-tests", "value": node_tests, "grammar": "mk-runtime-units"}
        )
        return engine.comprehend_units(complete), engine.comprehend_units(failing)

    plain, plain_failing = search([":mk-budget"])
    canonical, canonical_failing = search([":mk-budget", ":mk-canonical"])
    assert plain.complete and canonical.complete and canonical.applied == plain.applied
    assert _shape(canonical.meaning) == _shape(plain.meaning)
    assert canonical.nodes < plain.nodes
    assert (canonical_failing.budget, plain_failing.budget) == (None, None)  # both searches ran to exhaustion
    assert not canonical_failing.complete and not plain_failing.complete
    assert canonical_failing.nodes < plain_failing.nodes


@pytest.mark.needs_fcg
def test_lexicon_scale_flat(make_engine):
    """+5,000 lexical-class constructions: the same readings, the same node counts, and no slower (hashed supply)."""
    engine = make_engine()

    def measure():
        clean = engine.comprehend_units(WHERE_WORK)
        capped = [engine.comprehend_units(LONG) for _ in range(5)]
        return clean, capped[0].nodes, statistics.median(r.ms for r in capped)

    engine.load(_grammar(), tiers=TIERS)
    clean0, nodes0, ms0 = measure()
    engine.load([*_grammar(), *_lexicon(5000)], tiers=TIERS)
    clean5, nodes5, ms5 = measure()
    assert clean0.complete and (clean5.meaning, clean5.applied, clean5.nodes) == (
        clean0.meaning,
        clean0.applied,
        clean0.nodes,
    )
    assert nodes5 == nodes0 and ms0 >= 30  # the timed row is a real search (the node cap), not a 2 ms round trip
    assert ms5 <= 1.5 * ms0, (ms0, ms5)


@pytest.mark.needs_fcg
def test_client_timeout_kills_and_restarts(make_engine, monkeypatch):
    from memory_kernel.pipeline.engine import proc_state
    from memory_kernel.runtime.engine import Runaway

    engine = make_engine()
    engine.load(_grammar(), tiers=TIERS)
    before = engine.comprehend_units(WHERE_WORK)
    runaway = engine.pid
    monkeypatch.setattr(engine, "_client_timeout", lambda: 0.01)
    with pytest.raises(Runaway, match=f"pid {runaway} killed and restarted"):
        engine.comprehend_units(LONG)
    assert proc_state(runaway) is None  # killed by its pid
    assert engine.pid not in (None, runaway)  # restarted at once, the unit grammar reloaded
    monkeypatch.undo()
    after = engine.comprehend_units(WHERE_WORK)
    assert after.complete and after.meaning == before.meaning


# -- the text grammar: the leftmost open slot first ------------------------------------------------------------------

SLOT_ORDER_SENTENCE = "it would not work because nobody pays if fees rise but usage holds"


def _clause_slot(name: str) -> dict:
    return {"slot": name, "fills": "clause"}


def _slot_order_rows() -> list:
    """One root over a clause, three two-slot clause fillers and one operator (the argument grammar's shapes), all live:
    the live rows decide whether a grammar composes (``store.load_store``)."""
    from memory_kernel.runtime.constructions import CxnSpec

    span = [["clause-span", "?x", "?l", "?r"]]
    say = [["procedure", "?p", "say"], ["seq", "?p", "1", "?st"], ["statement", "?st", "?s"]]
    rows = [
        CxnSpec(id="stmt", kind="hand", unit="root", form=[_clause_slot("s")], meaning=say, score=0.2, status="live")
    ]
    for word in ("because", "if", "but"):
        form = [_clause_slot("h"), word, _clause_slot("d")]
        meaning = [*span, ["relation", "?rel", "?x", "?h", "?d"]]
        rows.append(
            CxnSpec(
                id=f"cx.{word}",
                kind="hand",
                unit="filler",
                fills="clause",
                form=form,
                meaning=meaning,
                score=0.7,
                status="live",
            )
        )
    form = [_clause_slot("a"), "not", _clause_slot("b")]
    meaning = [*span, ["mark", "?m", "?x", "?a", "?b"]]
    rows.append(
        CxnSpec(
            id="op.not",
            kind="mined",
            unit="filler",
            fills="clause",
            form=form,
            meaning=meaning,
            score=0.65,
            status="live",
        )
    )
    return rows


def _turn(text: str):
    from memory_kernel.runtime.executor import Turn

    return Turn(text, "Ana", "Milo", ("Ana", "Milo"), "c1", None, at="2026-11-19T10:00:00+00:00")


def _composed_runtime(engine):
    """A runtime on ``engine`` over the composing rows (``_slot_order_rows``, live); its first plan loads the grammar."""
    from memory_kernel.runtime.api import Runtime, default_registry
    from memory_kernel.runtime.lexicon import Lexicon, QuestionMap
    from memory_kernel.runtime.program import ToolSpec
    from unit.runtime.memory_writer import memory_store

    class NoGraph:
        async def rows(self, cypher, params):
            return []

    tools = default_registry(
        [
            ToolSpec.simple("statement", ("text",), False),
            ToolSpec.simple("relation", ("whole", "head", "text"), False),
            ToolSpec.simple("mark", ("whole", "scope", "rest"), False),
        ]
    )
    return Runtime(
        "bonfire",
        engine,
        NoGraph(),
        tools,
        lexicon=Lexicon([], QuestionMap.load()),
        store=memory_store(*_slot_order_rows()),
        record_standing=False,
    )


def _read_composed(engine):
    """Load the rows through a runtime (one plan loads the grammar), then read the sentence with the engine."""
    import asyncio

    from memory_kernel.runtime.normalise import normalise

    asyncio.run(_composed_runtime(engine).plan(_turn("ok"), window=5))
    return engine.comprehend(normalise(SLOT_ORDER_SENTENCE).text)


@pytest.mark.needs_fcg
def test_a_deeply_composed_sentence_reads_inside_the_default_budget(make_engine):
    """With the kernel's default settings, unchanged (the text grammar's search is bound by pyfcg's node cap and the
    request's time limit; a text ``Reading`` has no ``budget``), nested two-slot clause fillers read whole, each filler
    applied: the search fills the leftmost open slot first, never every order of the same slots. Without the
    slot-order node test this sentence ran past the request's 5 s and the engine was torn down."""
    reading = _read_composed(make_engine())
    assert reading is not None and reading.complete and reading.left == ()
    from memory_kernel.runtime.constructions import spec_id

    assert {"cx.because", "cx.if", "cx.but", "op.not"} <= {spec_id(a) for a in reading.applied}


def _node_tests_sent(engine, load) -> list:
    """Run ``load()`` and return the ``node-tests`` the engine was given last."""
    sent: list = []
    post = engine._post

    def spy(endpoint, data, *args, **kwargs):
        if endpoint == "/fcg-set-grammar-configuration" and data.get("key") == "node-tests":
            sent.append(data["value"])
        return post(endpoint, data, *args, **kwargs)

    engine._post = spy
    try:
        load()
    finally:
        del engine._post
    return sent[-1]


def test_a_grammar_composes_only_with_a_filler_that_holds_a_clause_slot():
    from memory_kernel.runtime.base import base_specs
    from memory_kernel.runtime.constructions import CxnSpec
    from memory_kernel.runtime.store import composes

    root = CxnSpec(id="stmt", kind="hand", unit="root", form=[_clause_slot("s")], meaning=[["x", "?s"]])
    np_filler = CxnSpec(
        id="np.of",
        kind="hand",
        unit="filler",
        fills="np",
        form=[{"slot": "h", "fills": "np"}, "of", {"slot": "d"}],
        meaning=[["np-span", "?x", "?l", "?r"]],
    )
    assert not composes(base_specs())
    assert not composes([*base_specs(), root])
    assert not composes([np_filler])
    assert composes([root, *_slot_order_rows()])
    assert composes([_slot_order_rows()[1]])


@pytest.mark.needs_fcg
def test_the_slot_order_test_is_on_only_for_a_store_that_composes(make_engine):
    from memory_kernel.runtime.engine import SLOT_ORDER_TEST
    from memory_kernel.runtime.store import load_store
    from unit.runtime.memory_writer import memory_store

    engine = make_engine()
    root = _slot_order_rows()[0]
    plain = memory_store(root)
    assert SLOT_ORDER_TEST not in _node_tests_sent(engine, lambda: load_store(plain, engine))
    composing = memory_store(*_slot_order_rows())
    assert SLOT_ORDER_TEST in _node_tests_sent(engine, lambda: load_store(composing, engine))
    assert SLOT_ORDER_TEST not in _node_tests_sent(engine, lambda: load_store(plain, engine))  # back off on reload


#: 201 normalised characters of no connective: the clause slot's generated filler takes at most 200
UNREADABLE_CLAUSE = " and ".join(["the team is small"] * 9) + " and 421"


@pytest.mark.needs_fcg
def test_a_partial_reading_stops_at_a_slot_nothing_fills(make_engine):
    """In a composing grammar a partial reading of a turn it does not read may bind less. "<201 characters> if fees rise
    but usage holds" is not read: ``cx.but`` holds the turn, its head slot is 201+ characters (past the clause filler's
    200) and cannot be filled whole, its dependent slot reads. With the slot-order node test the partial reading is
    ``stmt, cx.but, clause`` (``cx.if``, inside the unfillable head, is not applied);
    without it, ``stmt, cx.but, clause, cx.if``. Neither is complete; a complete reading is never lost to this.

    The pins are of the rows staged, loaded below the generated clause filler, as this test was written on, at their
    own scores (``store.load_store`` now compiles a staged row at ``DRAFT_SCORE``: the test puts their own back). A
    staged row no longer turns the node test on (``store.load_store``: the live rows decide), so the test turns it on
    here as it turns it off below. Loaded live (after the generated filler) this sentence reads ``stmt, cx.but, cx.if``
    with the node test and without it."""
    from dataclasses import replace
    from typing import Any

    from pyfcg import grammar as pyfcg_grammar

    from memory_kernel.runtime.constructions import spec_id
    from memory_kernel.runtime.engine import FRAME_PARTS_TEST, GRAMMAR_ID, SLOT_ORDER_TEST
    from memory_kernel.runtime.normalise import normalise
    from memory_kernel.runtime.store import compiled, load_store
    from unit.runtime.memory_writer import memory_store

    class Recorder:
        def __init__(self) -> None:
            self.cxns: list[Any] = []

        def load(self, cxns: Any, **_kw: Any) -> None:
            self.cxns = list(cxns)

    engine = make_engine()
    staged: Any = Recorder()  # what load_store loads for the rows staged, in its order
    rows = [replace(r, status="staged") for r in _slot_order_rows()]
    load_store(memory_store(*rows), staged)
    own = {c.name: c for r in rows for c in compiled(r)}  # each staged row at its own score, as the pins were taken
    engine.load([own.get(c.name, c) for c in staged.cxns], slot_order=True)
    text = normalise(f"{UNREADABLE_CLAUSE} if fees rise but usage holds").text
    on = engine.comprehend(text)
    plain = [t for t in pyfcg_grammar.DEFAULT_GRAMMAR_CONFIGURATION["node-tests"] if t != SLOT_ORDER_TEST]
    engine._post(
        "/fcg-set-grammar-configuration",
        {"key": "node-tests", "value": [FRAME_PARTS_TEST, *plain], "grammar": GRAMMAR_ID},
    )
    try:
        off = engine.comprehend(text)
    finally:
        engine._configure()
    assert on is not None and off is not None and not on.complete and not off.complete
    assert [spec_id(a) for a in on.applied] == ["stmt", "cx.but", "clause"]
    assert [spec_id(a) for a in off.applied] == ["stmt", "cx.but", "clause", "cx.if"]


def test_two_adjacent_slots_are_refused_so_every_open_slot_has_bounds():
    """The slot-order test ignores a slot whose boundaries are not integers; the compiler cannot build one: two slots
    with no literal between them are refused, so each slot sits between literal anchors or the span's own edges."""
    from memory_kernel.runtime.constructions import CompileError, CxnSpec, compile_spec

    adjacent = CxnSpec(
        id="cx.adjacent",
        kind="hand",
        unit="filler",
        fills="clause",
        form=[_clause_slot("h"), _clause_slot("d")],
        meaning=[["clause-span", "?x", "?l", "?r"], ["relation", "?rel", "?x", "?h", "?d"]],
    )
    with pytest.raises(CompileError, match="need a literal anchor between them"):
        compile_spec(adjacent)


# -- the text grammar: the search is offered only the constructions whose literals are in the text ------------------


def _literals(cxn) -> list[str]:
    """A compiled construction's literals, as text (its ``literals`` attribute holds them quoted)."""
    return [json.loads(q) for q in cxn.attributes.get("literals") or ()]


def _supply(engine, mode: str) -> None:
    from memory_kernel.runtime.engine import GRAMMAR_ID

    engine._post("/fcg-set-grammar-configuration", {"key": "cxn-supplier-mode", "value": mode, "grammar": GRAMMAR_ID})


def _mode(engine) -> str:
    return _lisp(engine, '(format nil "~s" (fcg::get-configuration fcg-go::mk-runtime-grammar :cxn-supplier-mode))')


def _renamed(meaning: list) -> list:
    """A meaning with its variables renamed in order of appearance (``?v0``, ``?v1``...): its predicates and their order
    as given, only the numbers FCG gave its variables gone."""
    names: dict[str, str] = {}
    return [
        [names.setdefault(a, f"?v{len(names)}") if isinstance(a, str) and a.startswith("?") else a for a in p]
        for p in meaning
    ]


def _copy(cxn, name: str, **attributes):
    """``cxn`` under another name, its attributes replaced by ``attributes`` (the same poles: it matches as it does)."""
    import pyfcg

    return pyfcg.Construction(
        name,
        conditional_pole=cxn.conditional_pole,
        contributing_pole=cxn.contributing_pole,
        attributes=attributes,
    )


def _patterns(cxn) -> list[str]:
    """The sequence patterns of a construction's conditional pole, as sent (quoted strings)."""
    return [
        s[1]
        for unit in cxn.conditional_pole
        for features in unit[1:]
        if isinstance(features, dict)
        for s in features.get("#form", [])
        if s[0] == "sequence"
    ]


def _verbatim(cxn) -> bool:
    """Every literal of ``cxn`` occurs, escaped as the compiler escapes it, in one of its own sequence patterns: the
    fact the supply rests on (a pattern can only match where its literal is in the text)."""
    from memory_kernel.runtime.constructions import _escape

    return all(any(_escape(lit) in p for p in _patterns(cxn)) for lit in _literals(cxn))


def test_a_compiled_construction_carries_its_literals():
    """Each variant's ``literals`` attribute is the normalised text of each of its literal parts, in form order, quoted
    as the unit grammar's triggers are, from the same map its sequence patterns are built from."""
    from memory_kernel.runtime.constructions import CONSUME, CxnSpec, compile_spec

    relation = [["clause-span", "?x", "?l", "?r"], ["relation", "?rel", "?x", "?h", "?d"]]

    def compiled_literals(spec) -> list[list[str]]:
        cxns = compile_spec(spec).cxns
        assert all(_verbatim(cxn) for cxn in cxns), [(c.name, _patterns(c)) for c in cxns]
        return [cxn.attributes["literals"] for cxn in cxns]

    because = CxnSpec(
        id="cx.because",
        kind="hand",
        unit="filler",
        fills="clause",
        form=[_clause_slot("h"), "because", _clause_slot("d")],
        meaning=relation,
    )
    assert compiled_literals(because) == [['"because"']]
    root = CxnSpec(id="stmt", kind="hand", unit="root", form=[_clause_slot("s")], meaning=[["statement", "?st", "?s"]])
    assert compiled_literals(root) == [[]]
    two = CxnSpec(
        id="cx.as-long-as",
        kind="hand",
        unit="filler",
        fills="clause",
        form=["As long as", _clause_slot("d"), ",", _clause_slot("h")],
        meaning=relation,
    )
    assert compiled_literals(two) == [['"as long as"', '","']]  # normalised (lowercased), in form order
    the = CxnSpec(
        id="np.the",
        kind="hand",
        unit="filler",
        fills="np",
        form=["The", CONSUME],
        meaning=[["definite", "?x"], ["name-span", "?x", "?l", "?r"]],
    )
    consumed = CxnSpec(
        id="np.any", kind="hand", unit="filler", fills="np", form=[CONSUME], meaning=[["name-span", "?x", "?l", "?r"]]
    )
    assert compiled_literals(the) == [['"the"']] and compiled_literals(consumed) == [[]]
    release = CxnSpec(
        id="t.release",
        kind="mined",
        unit="filler",
        fills="np",
        form=["release"],
        meaning=[["name-span", "?x", "?l", "?r"]],
    )
    assert compiled_literals(release) == [['"⟪release⟫"'], ['"release"']]  # each alternate its own variant's


def test_every_literal_the_kernel_loads_is_in_its_own_patterns():
    """Over everything ``load_store`` compiles (the base rows, the office catalogue's cards and their frame parts, the
    composing rows and the generated clause filler): every literal occurs verbatim in its construction's own sequence
    patterns, so a compiler change that stopped writing one as it is fails here, before the supply could withhold a
    construction that matches."""
    from pathlib import Path

    from memory_kernel.runtime.catalogue import parse_catalogue
    from memory_kernel.runtime.store import load_store
    from unit.runtime.memory_writer import memory_store

    class Recorder:
        def __init__(self) -> None:
            self.cxns: list = []

        def load(self, cxns, **_kw) -> None:
            self.cxns = list(cxns)

    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "office-catalogue.json"
    office = parse_catalogue(fixture.read_bytes(), str(fixture))
    loaded: dict[str, object] = {}
    for rows, cards in (((), ()), (_slot_order_rows(), ()), ((), office.cards), (_slot_order_rows(), office.cards)):
        store = memory_store(*rows)
        store.attach_cards(cards)
        engine = Recorder()
        load_store(store, engine)  # type: ignore[arg-type]
        loaded.update((c.name, c) for c in engine.cxns)
    assert {"clause", "np-bare", "verb", "cx.because", "stmt"} <= set(loaded)  # generated, base and composing rows
    assert any(n.startswith("frame.") for n in loaded) and any("#v" in n for n in loaded)  # frame parts, alternates
    with_literals = [c for c in loaded.values() if _literals(c)]
    assert len(with_literals) > 40
    assert [n for n, c in loaded.items() if not _verbatim(c)] == []


@pytest.mark.needs_fcg
def test_only_constructions_whose_literals_are_in_the_text_are_offered(make_engine):
    """For one text the search is offered, in ``:all-cxns``' order, only the constructions every one of whose literals
    occurs in it: ``offered`` names them, and no search applies any other (the first solution, the competing analyses
    and every analysis, their continued searches included)."""
    from memory_kernel.runtime.normalise import normalise
    from memory_kernel.runtime.store import load_store
    from unit.runtime.memory_writer import memory_store

    engine = make_engine()
    load_store(memory_store(*[r for r in _slot_order_rows() if r.id != "cx.but"]), engine)  # root, because, if, not
    text = normalise("it works because people pay").text
    offered = engine.offered(text)
    assert {"stmt", "cx.because", "clause"} <= set(offered)
    assert "cx.if" not in offered and "op.not" not in offered
    everything = engine._offer(text, ":all-cxns")  # what the stock supply offers at the same node
    literals = {c.name: _literals(c) for c in engine._complete}
    assert set(everything) == set(literals)
    assert offered == tuple(n for n in everything if all(lit in text for lit in literals[n]))  # its order, filtered
    assert len(offered) < len(everything)

    # the search is offered the same: a copy of "because" whose literals say "if" matches this text as "because" does,
    # and is never applied; offered everything, the search applies it (positive control)
    because = next(c for c in engine._complete if c.name == "cx.because")
    engine.add(_copy(because, "cx.because-if", **{**because.attributes, "literals": ['"if"']}))
    assert "cx.because-if" not in engine.offered(text)

    def applied() -> set[str]:
        first = engine.comprehend(text)
        others = engine.comprehend_all(text, 3)
        every, _ = engine.comprehend_every(text, cap=32, node_cap=500, deadline_ms=5000)
        assert first is not None and first.complete
        return {a for r in [first, *others, *every] for a in r.applied}

    assert "cx.because" in applied() and "cx.because-if" not in applied()
    _supply(engine, ":all-cxns")
    try:
        assert "cx.because-if" in applied()
    finally:
        engine._configure()


@pytest.mark.needs_fcg
def test_a_construction_without_a_literal_is_always_offered(make_engine):
    """A construction with no literal (a root that is one slot, a filler that only consumes or takes a token, and one
    added without a ``literals`` attribute at all) is offered for every text, one with none of the grammar's words
    included."""
    from memory_kernel.runtime.normalise import normalise
    from memory_kernel.runtime.store import load_store
    from unit.runtime.memory_writer import memory_store

    engine = make_engine()
    load_store(memory_store(*_slot_order_rows()), engine)
    because = next(c for c in engine._complete if c.name == "cx.because")
    engine.add(_copy(because, "cx.because-bare", score=0.7, label="filler"))  # built by hand: no literals attribute
    bare = {c.name for c in engine._complete if not _literals(c)}
    assert {"stmt", "clause", "verb", "np-bare", "cx.because-bare"} <= bare
    for sentence in ("zzz qqq", "it works because people pay", "nobody pays if fees rise but usage holds"):
        assert bare <= set(engine.offered(normalise(sentence).text)), sentence
    nothing = engine.offered(normalise("zzz qqq").text)
    assert set(nothing) == bare  # none of the grammar's literals is in it: what has none, and nothing else
    assert not {"cx.because", "cx.if", "cx.but", "op.not"} & set(nothing)


#: the deep sentence of the slot-order test and three more, each read completely by the composing rows
SUPPLY_SENTENCES = (
    SLOT_ORDER_SENTENCE,
    "it works because people pay",
    "nobody pays if fees rise",
    "it would not work but usage holds",
)


@pytest.mark.needs_fcg
def test_readings_are_the_same_with_and_without_the_supply(make_engine):
    """The supply only leaves out constructions that cannot apply: every reading (its meaning, predicate order
    included, applied, status, what is left, slots and span) is the one ``:all-cxns`` gives, the first solution's and
    the competing analyses' alike. Only the numbers in the meaning's variable names differ (``?h-2`` for ``?h-4``): FCG
    numbers a construction's variables each time it tries one, and the supply tries fewer. Slot unit names do not
    change."""
    from memory_kernel.runtime.engine import SUPPLY
    from memory_kernel.runtime.normalise import normalise

    engine = make_engine()
    _read_composed(engine)
    assert _mode(engine) == SUPPLY.upper()
    texts = [normalise(s).text for s in SUPPLY_SENTENCES]

    def read(text: str) -> list:
        first = engine.comprehend(text)
        assert first is not None
        return [
            (_renamed(r.meaning), r.applied, r.complete, r.left, r.left_spans, r.slots, r.boundaries)
            for r in [first, *engine.comprehend_all(text, 3)]
        ]

    supplied = [read(t) for t in texts]
    _supply(engine, ":all-cxns")
    try:
        everything = [read(t) for t in texts]
    finally:
        engine._configure()
    assert _mode(engine) == SUPPLY.upper()
    assert supplied == everything
    assert all(readings[0][2] for readings in supplied)  # each first reading is complete: real readings compared
    assert all(len(engine.offered(t)) < len(engine._complete) for t in texts)  # and the supply left some out of each


@pytest.mark.needs_fcg
def test_the_supply_survives_a_clear_and_a_restart(make_engine):
    """The supply is the text grammar's again after every clear (a load) and on a restarted engine."""
    import os
    import signal

    from memory_kernel.runtime.engine import SUPPLY, RuntimeEngineError
    from memory_kernel.runtime.normalise import normalise
    from memory_kernel.runtime.store import load_store
    from unit.runtime.memory_writer import memory_store

    engine = make_engine()
    store = memory_store(*_slot_order_rows())
    text = normalise("it works because people pay").text

    def filters() -> bool:
        offered = engine.offered(text)
        return "cx.because" in offered and not {"cx.if", "cx.but", "op.not"} & set(offered)

    load_store(store, engine)
    first = engine.comprehend(text)
    load_store(store, engine)  # the grammar cleared and every construction added again
    assert _mode(engine) == SUPPLY.upper() and filters()
    pid = engine.pid
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while engine._engine.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    with pytest.raises(RuntimeEngineError, match="died"):
        engine.comprehend(text)
    again = engine.comprehend(text)  # restarted, the last complete grammar reloaded
    assert engine.pid not in (None, pid)
    assert _mode(engine) == SUPPLY.upper() and filters()
    assert first is not None and again is not None
    assert (again.meaning, again.applied, again.complete) == (first.meaning, first.applied, first.complete)


@pytest.mark.needs_fcg
def test_alternatives_use_the_supply(make_engine):
    """``Runtime.alternatives`` (``comprehend_every``: a search continued after its first solution) finds what it finds
    when every construction is offered: the same analyses, in the same order, with the same paths and plans (their
    programs up to the numbers in variable names, as in ``test_readings_are_the_same_with_and_without_the_supply``)."""
    import asyncio

    from memory_kernel.runtime.engine import SUPPLY

    engine = make_engine()
    runtime = _composed_runtime(engine)
    runtime.start()
    turn = _turn(SLOT_ORDER_SENTENCE)

    def found():
        alternatives = asyncio.run(runtime.alternatives(turn))
        return alternatives.cut_short, [
            (a.path, _renamed(a.plan.program), a.plan.status, a.plan.diagnostics, a.plan.reading["applied"])
            for a in alternatives.found
        ]

    assert _mode(engine) == SUPPLY.upper()
    supplied = found()
    _supply(engine, ":all-cxns")
    try:
        everything = found()
        assert _mode(engine) == ":ALL-CXNS"  # the unfiltered leg ran unfiltered
    finally:
        engine._configure()
    assert supplied == everything
    assert len(supplied[1]) > 1  # several analyses: the continued search ran under each supply


@pytest.mark.needs_fcg
def test_a_formulation_is_offered_every_construction(make_engine):
    """``cxn-supplier-mode`` is one key for both directions, and a formulation's processor has no text to read (its
    root holds the meaning): the supply then offers what ``:all-cxns`` offers, so a construction with literals renders
    as it does under ``:all-cxns``."""
    from dataclasses import replace

    from memory_kernel.runtime.constructions import CxnSpec
    from memory_kernel.runtime.engine import GRAMMAR_ID, SUPPLY
    from memory_kernel.runtime.normalise import normalise
    from memory_kernel.runtime.store import load_store
    from unit.runtime.memory_writer import memory_store

    over = CxnSpec(id="root.over", kind="hand", unit="root", form=["The meeting is over"], meaning=[["over", "?o"]])
    engine = make_engine()
    load_store(memory_store(*_slot_order_rows(), replace(over, status="live")), engine)
    text = normalise("the meeting is over").text
    reading = engine.comprehend(text)
    assert reading is not None and reading.complete and reading.applied == ("root.over",)
    assert _literals(next(c for c in engine._complete if c.name == "root.over"))  # it carries literals
    meaning = [[a[1:] if isinstance(a, str) and a.startswith("?") else a for a in p] for p in reading.meaning]

    def formulate():
        return engine._post("/fcg-formulate", {"meaning": meaning, "grammar": GRAMMAR_ID, "package": None})

    assert _mode(engine) == SUPPLY.upper()
    supplied = formulate()
    _supply(engine, ":all-cxns")
    try:
        everything = formulate()
        assert _mode(engine) == ":ALL-CXNS"
    finally:
        engine._configure()
    assert everything == [text]  # positive control: under :all-cxns the construction renders its own text back
    assert supplied == everything
