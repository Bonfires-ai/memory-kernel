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
