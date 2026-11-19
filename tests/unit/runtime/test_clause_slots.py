"""Clause slots (K2 / K8): a slot written ``{Y:clause}`` in a taught form keeps its whole clause as written.

A clause slot is filled only by the clause filler, which ``store.load_store`` generates when a loaded root (live or a
draft) has a clause slot: never shipped in the base data, never stored. Its op, ``clause-span``, gives the slot's text
as written: no determiner dropped, no quantifier read. A grammar with no clause slot loads exactly what it loaded
before; an np slot (``{Y}``, or ``{Y:np}``) reads as it always did.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import memory_kernel.runtime.store as store_module
from memory_kernel.runtime.api import Runtime, default_registry
from memory_kernel.runtime.catalogue import CatalogueError, parse_catalogue
from memory_kernel.runtime.constructions import CONSUME, CxnSpec, precedence, spec_id
from memory_kernel.runtime.executor import Turn
from memory_kernel.runtime.lexicon import Lexicon, QuestionMap
from memory_kernel.runtime.normalise import normalise
from memory_kernel.runtime.program import ToolSpec
from memory_kernel.runtime.store import compiled, load_store, loaded_frame_parts
from memory_kernel.runtime.teach import TeachRequest, _parse_form, teach
from unit.runtime.memory_writer import memory_store

PORT = 9726  # 9713-9725 and 9727-9728 are the other engine test modules'
NOW = "2026-11-19T10:00:00+00:00"

X_NP = {"slot": "x", "fills": "np"}
Y_NP = {"slot": "y", "fills": "np"}
Y_CLAUSE = {"slot": "y", "fills": "clause"}
GROUND = [["procedure", "?p", "ground"], ["seq", "?p", "1", "?s1"], ["ground", "?s1", "?x", "?y"]]
HOLD = [["procedure", "?p", "hold"], ["seq", "?p", "1", "?s1"], ["hold", "?s1", "?x"]]
TOOLS = default_registry(
    [ToolSpec.simple("ground", ("about", "reason"), False), ToolSpec.simple("hold", ("about",), False)]
)
CLAUSE_FORM = "{X} is too slow because {Y:clause}"
NP_FORM = "{X} is too slow because {Y}"
BODY: list[str | dict[str, Any]] = [{"call": "ground", "args": {"about": "{X}", "reason": "{Y}"}}]
OPENING = "onboarding is too slow because "
#: the diagnosis's probe table (findings 09, section 6): each reason as written
REASONS = (
    "my team left",
    "Ana's team left",
    "it is broken",
    "no one owns it",
    "every team is busy",
    "the team is small",
)
#: reasons whose NORMALISED text (what the filler's ``.{1,200}`` consumes) is 200 and 201 characters long
REASON_200 = " and ".join(["the team is small"] * 9) + " and 42"
REASON_201 = REASON_200 + "1"
#: 224 characters as written, 200 normalised ("were" is "be", a run of spaces one space)
LONG_200 = " and ".join(["the  teams  were  small"] * 8) + " 4444"
#: 189 characters as written, 201 normalised ("owns" is "⟪own⟫")
SHORT_201 = " and ".join(["Ana owns it"] * 12) + " 4"


def _row(cxn_id: str, form: list, meaning: list = GROUND, status: str = "live") -> CxnSpec:
    return CxnSpec(
        id=cxn_id,
        kind="taught",
        unit="root",
        form=form,
        meaning=meaning,
        status=status,
        provenance={"teacher": "Ana", "chat_id": "c1"},
    )


NP_ROW = _row("slow-because.t0000000201.0", [X_NP, "is too slow because", Y_NP])
CLAUSE_ROW = _row("slow-because.t0000000202.0", [X_NP, "is too slow because", Y_CLAUSE])
SLOW = _row("slow.t0000000203.0", [X_NP, "is too slow"], HOLD)
SEND = _row("send.t0000000204.0", ["send", X_NP, "to", Y_NP])
NP_DRAFT = _row("drags.t0000000205.0", [X_NP, "drags"], HOLD, status="staged")
CLAUSE_DRAFT = replace(CLAUSE_ROW, status="staged")
#: a store's own clause filler, as the demo ships one today: it names its span with the NP cut
HAND_CLAUSE = CxnSpec(
    id="hand.clause",
    kind="hand",
    unit="filler",
    fills="clause",
    status="live",
    form=[CONSUME],
    meaning=[["name-span", "?x", "?l", "?r"]],
)


# -- the teach syntax -------------------------------------------------------------------------------------------------


def test_a_form_writes_a_slot_kind_after_a_colon():
    """``{Y:clause}`` is a clause slot named Y (the body still writes ``{Y}``); ``{Y:np}`` is ``{Y}``, an np slot."""
    assert _parse_form(CLAUSE_FORM) == ([X_NP, "is too slow because", Y_CLAUSE], ["X", "Y"], [])
    assert _parse_form("{X} is too slow because {Y:np}") == _parse_form(NP_FORM)
    assert _parse_form(NP_FORM) == ([X_NP, "is too slow because", Y_NP], ["X", "Y"], [])


@pytest.mark.parametrize("kind", ["bogus", "", "verb", "Clause"])
def test_an_unknown_slot_kind_is_a_teach_problem(kind: str):
    """Any kind but np and clause after the colon is refused with a reason, never read as an np slot; nothing is
    staged (teaching refuses the form before it reads anything)."""
    form = f"{{X}} is too slow because {{Y:{kind}}}"
    problem = (
        f"form {form!r}: slot {{Y:{kind}}} has an unknown kind {kind!r}: a slot is {{Name}}, {{Name:np}} or "
        "{Name:clause}"
    )
    assert _parse_form(form)[2] == [problem]
    store = memory_store()
    draft = teach(store, None, TeachRequest([form], BODY, {}, "Ana", "c1"), registry=TOOLS)  # type: ignore[arg-type]
    assert draft.draft_id == "" and draft.specs == [] and problem in draft.problems
    assert store.all() == []


def _catalogue(form: str) -> bytes:
    tool = {
        "name": "ground",
        "input_schema": {"type": "object", "properties": {"about": {"type": "string"}, "reason": {"type": "string"}}},
        "x-args": {"about": {"role": "theme"}, "reason": {"role": "cause"}},
        "x-side-effect": False,
        "x-forms": [form],
    }
    return json.dumps({"schema_version": "kernel-catalogue/v1", "tools": [tool]}).encode()


def test_a_catalogue_form_may_write_a_clause_slot():
    """A catalogue's forms are parsed as a taught form is (``teach._parse_form``): a card may hold a clause slot, and
    an unknown kind is refused."""
    catalogue = parse_catalogue(_catalogue("{about} is too slow because {reason:clause}"), "test")
    (card,) = catalogue.cards
    assert card.form == [
        {"slot": "about", "fills": "np"},
        "is too slow because",
        {"slot": "reason", "fills": "clause"},
    ]
    with pytest.raises(CatalogueError, match="unknown kind 'bogus'"):
        parse_catalogue(_catalogue("{about} is too slow because {reason:bogus}"), "test")


def test_the_twins_load_at_one_precedence():
    """A slot's kind is not a literal: the clause twin and the np twin have one specificity, so the order between
    roots is what it was."""
    assert precedence(CLAUSE_ROW)[:2] == precedence(NP_ROW)[:2]


# -- the generated clause filler --------------------------------------------------------------------------------------


class _LoadRecorder:
    """An engine that only records what ``load_store`` loads."""

    def __init__(self) -> None:
        self.cxns: list[Any] = []

    def load(self, cxns: Any) -> None:
        self.cxns = list(cxns)


def _serialised(cxns: list[Any]) -> list[str]:
    return [
        json.dumps(
            [c.name, c.conditional_pole, c.contributing_pole, c.attributes, c.feature_types],
            sort_keys=True,
            ensure_ascii=False,
        )
        for c in cxns
    ]


def _as_before(store: Any) -> tuple[list[Any], int]:
    """What ``load_store`` loaded before clause slots: the drafts, the frame parts of the live order, the live order."""
    order = store.load_order()
    specs = [*sorted(store.staged(), key=precedence), *loaded_frame_parts(order), *order]
    return [cxn for spec in specs for cxn in compiled(spec)], len(specs)


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "office-catalogue.json"
OFFICE = parse_catalogue(FIXTURE.read_bytes(), str(FIXTURE))
#: a taught np-only grammar: three live roots, a draft and a store's own clause filler
NP_TAUGHT = (NP_ROW, SLOW, SEND, NP_DRAFT, HAND_CLAUSE)
#: what the kernel before clause slots (3352780d, its own code: ``git archive``) loads for each grammar of
#: ``_no_clause_store``: (specs loaded, constructions loaded, sha256 of their ``_serialised`` JSON). Re-pin only for a
#: change to the base set or the compiler that is meant to change what every grammar loads.
AT_3352780D = {
    "base": (50, 52, "58830e644dee8cdbbffe3ce09cbef0d7ac6031575790d7f41bee6893e575786c"),
    "np-taught": (56, 58, "da12dca86a1d4c51b256c8101d183baa4e3e4d0a3a3e011df87e5aa735a7b0b4"),
    "office": (77, 81, "c29f825cad244ab6f449fa172e6331eee4e564f5d49ba9591683089ec78ab539"),
    "np-taught+office": (83, 87, "96ae687dd12cbff79907f529e3ffdbbb6d32b5db2258d7723b637c02730730f6"),
}


def _no_clause_store(name: str) -> Any:
    store = memory_store(*(NP_TAUGHT if name.startswith("np-taught") else ()))
    if name.endswith("office"):
        store.attach_cards(OFFICE.cards)
    return store


def _loaded_sha256(store: Any) -> tuple[int, int, str]:
    engine = _LoadRecorder()
    loaded = load_store(store, engine)  # type: ignore[arg-type]
    text = json.dumps([json.loads(s) for s in _serialised(engine.cxns)], sort_keys=True, ensure_ascii=False)
    return loaded, len(engine.cxns), hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("name", sorted(AT_3352780D))
def test_a_grammar_with_no_clause_slot_loads_as_before(name: str):
    """No clause slot, no generated filler: the base grammar, a taught np-only grammar (a draft, and a store's own
    clause filler, included), the office catalogue's cards, and both together load exactly what the kernel before
    clause slots loaded: the same constructions in the same order, by the hash of its own load, and by the old
    composition (drafts, frame parts, live order)."""
    store = _no_clause_store(name)
    assert _loaded_sha256(store) == AT_3352780D[name]
    engine = _LoadRecorder()
    loaded = load_store(store, engine)  # type: ignore[arg-type]
    before, count = _as_before(store)
    assert loaded == count
    assert _serialised(engine.cxns) == _serialised(before)


def test_a_row_with_the_generated_fillers_id_is_warned_about(caplog):
    """A stored row with the generated filler's id (``clause``) loads after it and replaces it in the engine: the
    store still opens and loads as it would, and the load says so once, naming the id (never the row's form). No
    warning without a clause slot (nothing is generated) or without the clash."""
    clash = CxnSpec(id="clause", kind="mined", unit="root", form=[X_NP, "is fine"], meaning=HOLD, status="live")
    for rows, warned in (((CLAUSE_ROW, clash), True), ((clash,), False), ((CLAUSE_ROW,), False)):
        store = memory_store(*rows)
        engine = _LoadRecorder()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="memory_kernel.runtime.store"):
            loaded = load_store(store, engine)  # type: ignore[arg-type]
        messages = [r.getMessage() for r in caplog.records if r.name == "memory_kernel.runtime.store"]
        before, count = _as_before(store)
        names = [c.name for c in engine.cxns]
        if not warned:
            assert messages == [], (rows, messages)
            continue
        assert messages == [
            "a loaded construction has the id 'clause', the generated clause filler's: it replaces that filler in "
            "the engine, so a clause slot reads only through a clause filler of the store's own"
        ]
        assert "is fine" not in messages[0]
        assert loaded == count + 1 and names.count("clause") == 2  # loaded as before: nothing refused or dropped


@pytest.mark.parametrize("rows", [(CLAUSE_ROW,), (NP_ROW, CLAUSE_DRAFT)], ids=["live", "a-draft-only"])
def test_a_clause_slot_loads_the_generated_clause_filler(rows: tuple[CxnSpec, ...]):
    """A root with a clause slot, live or a draft, loads the ONE clause filler, beside the frame parts (after the
    drafts and the parts, below the live order); it is never stored and no store lookup knows it."""
    filler = store_module.CLAUSE_FILLER
    assert (filler.id, filler.unit, filler.fills, filler.form, filler.meaning, filler.status) == (
        "clause",
        "filler",
        "clause",
        [CONSUME],
        [["clause-span", "?x", "?l", "?r"]],
        "live",
    )
    assert store_module.generated_fillers([NP_ROW, SLOW, NP_DRAFT, HAND_CLAUSE]) == []
    assert store_module.generated_fillers([NP_ROW, *rows]) == [filler]
    store = memory_store(*rows)
    engine = _LoadRecorder()
    loaded = load_store(store, engine)  # type: ignore[arg-type]
    before, count = _as_before(store)
    names = [c.name for c in engine.cxns]
    assert loaded == count + 1 and names.count("clause") == 1
    assert _serialised([c for c in engine.cxns if c.name != "clause"]) == _serialised(before)
    order = store.load_order()
    below = [*sorted(store.staged(), key=precedence), *loaded_frame_parts(order)]
    assert names.index("clause") == sum(len(compiled(s)) for s in below)
    assert "clause" not in {s.id for s in store.known_specs()}


def test_a_store_clause_filler_loads_after_the_generated_one():
    """A store's own clause filler is a live row, so it loads after the generated one (and wins the engine's
    recency tie-break: the next test reads through it)."""
    store = memory_store(CLAUSE_ROW, HAND_CLAUSE)
    engine = _LoadRecorder()
    load_store(store, engine)  # type: ignore[arg-type]
    names = [c.name for c in engine.cxns]
    assert names.index("clause") < names.index("hand.clause")


# -- reading: the engine ----------------------------------------------------------------------------------------------


def _data_dir():
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """One engine for the module, stopped by pid; each runtime loads its own bonfire's grammar into it."""
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("runtime-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


class _NoGraph:
    async def rows(self, cypher: str, params: Any) -> list[dict]:
        return []


def _runtime(engine, *rows: CxnSpec) -> Runtime:
    lexicon = Lexicon([], QuestionMap.load())
    store = memory_store(*rows)
    return Runtime("bonfire", engine, _NoGraph(), TOOLS, lexicon=lexicon, store=store, record_standing=False)


def _teach(runtime: Runtime, forms: list[str], *, replace: bool = False, **kwargs: Any):
    body, defaults = kwargs.pop("body", BODY), kwargs.pop("defaults", {})
    return asyncio.run(runtime.teach(TeachRequest(forms, body, defaults, "Ana", "c1", replace=replace)))


def _taught(runtime: Runtime, *forms: str, **kwargs: Any) -> list[CxnSpec]:
    draft = _teach(runtime, list(forms), **kwargs)
    assert not draft.problems, draft.problems
    return asyncio.run(runtime.confirm(draft.draft_id, "Ana"))


def _plan(runtime: Runtime, text: str, **kwargs: Any):
    turn = Turn(text, "Ana", "Milo", ("Ana", "Milo"), "c1", None, at=NOW)
    return asyncio.run(runtime.plan(turn, window=5, **kwargs))


def _args(plan) -> list[dict]:
    return [step["args"] for step in plan.tool_steps]


def _applied(plan) -> set[str]:
    return {spec_id(name) for name in plan.reading["applied"]}


@pytest.mark.needs_fcg
def test_a_clause_slot_keeps_each_reason_as_written(engine):
    """Every row of the probe table reads with Y exactly as written, in the tool's argument and in the frame: only the
    clause filler fills the clause slot (np-my, np-possessive, np-it never compete for it), no determiner is dropped,
    and a quantified clause is not refused (the plan's quantifier check reads name-span only)."""
    runtime = _runtime(engine)
    (row,) = _taught(runtime, CLAUSE_FORM)
    assert row.form == [X_NP, "is too slow because", Y_CLAUSE]
    for reason in REASONS:
        plan = _plan(runtime, OPENING + reason)
        assert plan.status == "needs_tools" and plan.diagnostics == [], (reason, plan.status, plan.diagnostics)
        assert _args(plan) == [{"about": "onboarding", "reason": reason}], (reason, plan.tool_steps)
        assert plan.frames[0]["roles"] == {"x": "onboarding", "y": reason}, (reason, plan.frames)
        assert _applied(plan) == {row.id, "np-bare", "clause"}, (reason, plan.reading["applied"])


@pytest.mark.needs_fcg
def test_a_clause_slot_never_takes_the_final_mark(engine):
    """KK3 holds for a clause slot: a final "." or "!" is the sentence's, so Y is the clause without it; "...because"
    and "...because." have no reason and are unread; a "?" the speaker wrote is never stripped, and no slot takes it,
    so the turn is unread."""
    runtime = _runtime(engine)
    _taught(runtime, CLAUSE_FORM)
    for reason in REASONS:
        for mark in (".", "!"):
            text = OPENING + reason + mark
            plan = _plan(runtime, text)
            assert plan.status == "needs_tools" and not plan.reading["retried"], (text, plan.status, plan.diagnostics)
            assert _args(plan) == [{"about": "onboarding", "reason": reason}], (text, plan.tool_steps)
            assert plan.frames[0]["roles"]["y"] == reason and plan.turn["text"] == text, (text, plan.frames)
    for text in (OPENING.strip(), OPENING.strip() + ".", OPENING.strip() + "!"):
        plan = _plan(runtime, text)
        assert plan.status == "unread" and plan.tool_steps == [] and not plan.reading["retried"], (text, plan.status)
        roles = [value for frame in plan.frames for value in frame["roles"].values()]
        assert not any(value in ("?", ".", "!") for value in roles), (text, plan.frames)
    asked = _plan(runtime, OPENING + "no one owns it?")
    assert asked.status == "unread" and asked.tool_steps == [] and asked.program == [], (asked.status, asked.reading)
    # a closing quote is no final mark (KK3): a mark inside the quotes is the clause's, one after them the sentence's
    for text, reason in (
        (OPENING + '"it is broken".', '"it is broken"'),
        (OPENING + '"it is broken."', '"it is broken."'),
    ):
        plan = _plan(runtime, text)
        assert plan.status == "needs_tools" and _args(plan) == [{"about": "onboarding", "reason": reason}], (
            text,
            plan.status,
            plan.tool_steps,
        )


@pytest.mark.needs_fcg
def test_the_np_twin_drops_the_determiner_and_refuses_a_quantifier_as_before(engine):
    """``{X} is too slow because {Y}`` is an np slot, unchanged: the NP cut drops a leading determiner, a quantified
    reason is a repair at plan time, and no clause filler is loaded for it."""
    runtime = _runtime(engine)
    _taught(runtime, NP_FORM)
    team = _plan(runtime, OPENING + "the team is small")
    assert team.status == "needs_tools", (team.status, team.diagnostics)
    assert _args(team) == [{"about": "onboarding", "reason": "team is small"}], team.tool_steps
    assert team.frames[0]["roles"] == {"x": "onboarding", "y": "team is small"}, team.frames
    assert "clause" not in _applied(team)
    # a nested np filler ("my {X}") names its inner span, not the slot's: the role keeps the NP cut of the slot
    mine = _plan(runtime, OPENING + "my team left")
    assert mine.frames[0]["roles"] == {"x": "onboarding", "y": "my team left"}, mine.frames
    for reason, norm in (("no one owns it", "no one ⟪own⟫ it"), ("every team is busy", "every team be busy")):
        plan = _plan(runtime, OPENING + reason)
        assert (plan.status, list(plan.diagnostics)) == (
            "repair",
            [f"{norm!r} is quantified: no keyword applies the quantifier yet"],
        ), reason


@pytest.mark.needs_fcg
def test_the_clause_filler_takes_at_most_200_characters(engine):
    """The cap is 200 NORMALISED characters (the filler's ``.{1,200}`` reads the normalised turn: a verb as its marked
    lemma, an auxiliary as its lemma, tokens joined by one space), not 200 as written: a clause of 200 reads whole,
    even when it is 224 characters as written (the tool gets those 224); one of 201 leaves the slot unfilled, even
    when it is 189 as written, and the turn is a repair. The unfilled slot's frame role is still the clause as written
    (no filler's op sits on it: its kind decides)."""
    runtime = _runtime(engine)
    (row,) = _taught(runtime, CLAUSE_FORM)
    edges = len("⟦ onboarding be too slow because ") + len(" ⟧")
    cases = ((REASON_200, 200, 200), (REASON_201, 201, 201), (LONG_200, 224, 200), (SHORT_201, 189, 201))
    for reason, written, normalised in cases:
        assert (len(reason), len(normalise(OPENING + reason).text) - edges) == (written, normalised), reason
    for reason in (REASON_200, LONG_200):
        whole = _plan(runtime, OPENING + reason)
        assert whole.status == "needs_tools" and _args(whole) == [{"about": "onboarding", "reason": reason}], reason
    for reason in (REASON_201, SHORT_201):
        long = _plan(runtime, OPENING + reason)
        assert (long.status, long.tool_steps) == ("repair", []), (long.status, long.diagnostics)
        assert list(long.diagnostics) == [f"no construction read the whole turn ({row.id} read part of it)"]
        assert long.frames[0]["roles"] == {"x": "onboarding", "y": reason} and not long.frames[0]["complete"]


@pytest.mark.needs_fcg
def test_a_clause_twin_competes_with_its_np_twin_at_teach_time(engine):
    """A slot's kind is not part of a form's skeleton: the clause twin of a live np form (the same literals, the same
    body) competes with it and is refused unless taught with replace, which swaps it in; and back again."""
    runtime = _runtime(engine)
    (np_row,) = _taught(runtime, NP_FORM)
    refused = _teach(runtime, [CLAUSE_FORM])
    assert refused.problems == [
        f"form {CLAUSE_FORM!r} competes with the live construction {np_row.id}: teach it with replace"
    ]
    (clause_row,) = _taught(runtime, CLAUSE_FORM, replace=True)
    assert [r["id"] for r in clause_row.provenance["replaced"]] == [np_row.id]
    assert [s.id for s in runtime.store.live() if s.unit == "root"] == [clause_row.id]
    plan = _plan(runtime, OPENING + "the team is small")
    assert _args(plan) == [{"about": "onboarding", "reason": "the team is small"}], plan.tool_steps
    back = _teach(runtime, [NP_FORM])
    assert back.problems == [
        f"form {NP_FORM!r} competes with the live construction {clause_row.id}: teach it with replace"
    ]


@pytest.mark.needs_fcg
def test_np_twins_are_still_refused_as_duplicates(engine):
    """Two np twins with one body are one construction: teaching the live np form again, as ``{Y}`` or ``{Y:np}``,
    with replace or without, is refused naming the live row, as before."""
    runtime = _runtime(engine)
    (row,) = _taught(runtime, NP_FORM)
    for form in (NP_FORM, "{X} is too slow because {Y:np}"):
        for replacing in (False, True):
            draft = _teach(runtime, [form], replace=replacing)
            assert draft.problems == [f"form {form!r} is already the live construction {row.id}"], (form, replacing)


@pytest.mark.needs_fcg
def test_a_store_clause_filler_wins_over_the_generated_one(engine):
    """A store that ships its own clause filler (the demo's ``hand.clause``, which names its span with the NP cut)
    keeps reading through it: it loads after the generated filler and wins the engine's recency tie-break. The frame
    cuts the role by the op that filler applied on the slot, so it carries the tool's text."""
    runtime = _runtime(engine, CLAUSE_ROW, HAND_CLAUSE)
    plan = _plan(runtime, OPENING + "the team is small")
    assert plan.status == "needs_tools", (plan.status, plan.diagnostics)
    assert "hand.clause" in _applied(plan) and "clause" not in _applied(plan), plan.reading["applied"]
    assert _args(plan) == [{"about": "onboarding", "reason": "team is small"}], plan.tool_steps
    assert plan.frames[0]["roles"] == {"x": "onboarding", "y": "team is small"}, plan.frames


@pytest.mark.needs_fcg
def test_a_body_clause_reads_through_a_clause_slot(engine):
    """A body clause read through a live root with a clause slot: a slot written in it ("{R}") becomes that slot's
    variable, any other text (a default) a constant kept as written ("the team is small")."""
    runtime = _runtime(engine)
    _taught(runtime, CLAUSE_FORM)
    _taught(
        runtime,
        "{X} drags because {R:clause}",
        "{X} is slow",
        body=["{X} is too slow because {R}"],
        defaults={"R": "the team is small"},
    )
    said = _plan(runtime, "onboarding drags because no one owns it")
    assert _args(said) == [{"about": "onboarding", "reason": "no one owns it"}], (said.status, said.diagnostics)
    default = _plan(runtime, "onboarding is slow")
    assert _args(default) == [{"about": "onboarding", "reason": "the team is small"}], (
        default.status,
        default.diagnostics,
    )


@pytest.mark.needs_fcg
def test_a_catalogue_card_with_a_clause_slot_reads_end_to_end(engine):
    """A card (kind ``hand``) with a clause slot: the generated filler fills it, the tool gets the clause as written,
    and the card's tool frame (keyed by the tool's parameters) carries the same text; KK3 holds."""
    catalogue = parse_catalogue(_catalogue("{about} is too slow because {reason:clause}"), "test")
    runtime = Runtime(
        "bonfire",
        engine,
        _NoGraph(),
        default_registry(catalogue.tools),
        lexicon=Lexicon([], QuestionMap.load()),
        store=memory_store(),
        cards=catalogue.cards,
        record_standing=False,
    )
    plan = _plan(runtime, OPENING + "no one owns it")
    assert plan.status == "needs_tools", (plan.status, plan.diagnostics)
    assert _applied(plan) == {"card.ground.0", "np-bare", "clause"}, plan.reading["applied"]
    assert _args(plan) == [{"about": "onboarding", "reason": "no one owns it"}], plan.tool_steps
    frame = plan.frames[0]
    assert (frame["name"], frame["kind"], frame["roles"]) == (
        "ground",
        "tool",
        {"about": "onboarding", "reason": "no one owns it"},
    ), plan.frames
    marked = _plan(runtime, OPENING + "the team is small.")
    assert _args(marked) == [{"about": "onboarding", "reason": "the team is small"}], marked.tool_steps
    asked = _plan(runtime, OPENING + "no one owns it?")
    assert asked.status == "unread" and asked.tool_steps == [], asked.status


@pytest.mark.needs_fcg
def test_a_follow_up_composes_into_an_open_clause_slot(engine):
    """A follow-up that reads as a bare fragment, against the open frame of a clause-slot phrase whose Y is open, is
    read again inside the phrase's form ("onboarding is too slow because <the fragment>"): Y is the fragment as
    written, with its final mark dropped, and the frame is composed with the phrase."""
    runtime = _runtime(engine)
    (row,) = _taught(runtime, CLAUSE_FORM)
    context = [{"name": row.id, "kind": "phrase", "roles": {"x": "onboarding", "y": None}}]
    for text, reason in (("no one owns it", "no one owns it"), ("the team is small.", "the team is small")):
        plan = _plan(runtime, text, context=context)
        assert plan.status == "needs_tools", (text, plan.status, plan.diagnostics)
        assert _args(plan) == [{"about": "onboarding", "reason": reason}], (text, plan.tool_steps)
        frame = plan.frames[0]
        assert (frame["composed_with"], frame["roles"]) == (row.id, {"x": "onboarding", "y": reason}), plan.frames
        assert plan.turn["text"] == text
