"""The construction store and teaching: a taught verb is a draft its teacher can use in the chat where it was taught,
live bonfire-wide once the teacher confirms; a form competing with a live construction is refused unless replaced; a
taught construction composes another one in a single comprehension. 1.5.1: teaching is admission: a teach reads its
body and its own renderings only, says what the live grammar reads them as (``competes``), and refuses a duplicate."""

import asyncio
import functools
import json
import os
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from memory_kernel.runtime.api import default_registry
from memory_kernel.runtime.base import _subject_shaped
from memory_kernel.runtime.catalogue import parse_catalogue
from memory_kernel.runtime.constructions import (
    CxnSpec,
    Standing,
    normalised_rendering,
    precedence,
    renderings,
    spec_id,
)
from memory_kernel.runtime.executor import VALUE_SPECS, Execution, Turn, execute
from memory_kernel.runtime.frames import _trigger
from memory_kernel.runtime.lexicon import LEXICON_SPECS
from memory_kernel.runtime.normalise import normalise
from memory_kernel.runtime.ops_memory import MEMORY_SPECS
from memory_kernel.runtime.program import Program, Registry, ToolParam, ToolSpec
from memory_kernel.runtime.store import (
    ConstructionStore,
    StoreError,
    _compiled,
    load_store,
)
from memory_kernel.runtime.teach import (
    TeachError,
    TeachRequest,
    _orphan_warnings,
    _skeleton_of,
    admissible,
    confirm,
    expand,
    resolver,
    teach,
)
from unit.runtime.memory_writer import MemoryWriter

PORT = 9722  # 9720-9727 are the engine tests' (9722 is no other module's engine); 9712-9719 are mk serve's
X = {"slot": "x", "fills": "np"}
Y = {"slot": "y", "fills": "np"}
#: the keyword set plus the catalogue's tools ("cards from another set"); give has a side effect
REGISTRY = Registry({**VALUE_SPECS, **MEMORY_SPECS, **LEXICON_SPECS}).with_tools(
    [
        ToolSpec.simple("take", ("item",), False),
        ToolSpec.simple("make", ("item",), False),
        ToolSpec.simple("give", ("item", "person"), True),
        ToolSpec.simple("put", ("item", "place"), False),
    ]
)


def _card(tool: str, form: list, args: list[str], id: str | None = None) -> CxnSpec:
    """The construction a catalogue tool arrives with: its clause reads as a one-step procedure."""
    meaning = [["procedure", "?p", tool], ["seq", "?p", "1", "?s1"], [tool, "?s1", *args]]
    return CxnSpec(id=id or f"card-{tool}", kind="hand", unit="root", form=form, meaning=meaning, status="live")


CARDS = [
    _card("take", ["take", X], ["?x"]),
    _card("make", ["make", X], ["?x"]),
    _card("give", ["give", X, "to", Y], ["?x", "?y"]),
    _card("put", ["put", X, "on", Y], ["?x", "?y"]),
    _card("zap", ["zap", X], ["?x"]),  # a card whose tool the catalogue does not register
]

BREW = TeachRequest(
    forms=["brew {X} for {R}", "brew {X}"],
    body=["take water", "take herb", "make {X}", "give {X} to {R}"],
    defaults={"R": "me"},
    teacher="Kira Dace",
    chat_id="c-workshop",
)

#: 1.5: the catalogue says what take and make return (the item), and make takes several materials
REGISTRY_15 = Registry({**VALUE_SPECS, **MEMORY_SPECS, **LEXICON_SPECS}).with_tools(
    [
        replace(ToolSpec.simple("take", ("item",), False), returns="item"),
        ToolSpec("make", (ToolParam("item"), ToolParam("material", many=True)), False, returns="item"),
        ToolSpec.simple("give", ("item", "person"), True),
        ToolSpec.simple("put", ("item", "place"), False),
    ]
)
#: the 1.4 make card reads only "make X"
CARDS_15 = [*CARDS, _card("make", ["make", X, "from", Y], ["?x", "?y"], id="card-make-from")]
#: "them" is the two takes' results, and X after make is what make returned
BREW_15 = TeachRequest(
    forms=["brew {X} for {R}", "brew {X}"],
    body=["take water", "take herb", "make {X} from them", "give {X} to {R}"],
    defaults={"R": "me"},
    teacher="Kira Dace",
    chat_id="c-workshop",
)


# -- the store (no engine) -------------------------------------------------------------------------------------------


def _row(cxn_id: str, kind: str, status: str = "live", **extra: Any) -> CxnSpec:
    return CxnSpec(
        id=cxn_id,
        kind=kind,  # type: ignore[arg-type]
        unit="root",
        form=["zork", X, "?"],
        meaning=[["find-entity", "?e", "?x"], ["answer", "?e"]],
        status=status,  # type: ignore[arg-type]
        **extra,
    )


# -- teaching (engine) -----------------------------------------------------------------------------------------------


def _data_dir():
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """One engine for the module, stopped by pid; each test loads its own store into it."""
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("runtime-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


def _snapshot(store: ConstructionStore) -> str:
    """What the graph holds of this store's phrases (the rows and their counters)."""
    return json.dumps(store.writer.nodes, sort_keys=True)


def _store(tmp_path, engine, cards: list[CxnSpec] = CARDS) -> ConstructionStore:
    store = ConstructionStore(MemoryWriter())
    for card in cards:
        store.put(card)
    load_store(store, engine)
    return store


def _read(engine, text: str):
    norm = normalise(text)
    return norm, engine.comprehend(norm.text)


def _turn(text: str, speaker: str = "Kira Dace", participants: tuple[str, ...] = ()) -> Turn:
    return Turn(
        text=text, speaker=speaker, addressee=None, participants=participants, chat_id="c-workshop", message_id=None
    )


class NoMemory:
    async def run(self, op: str, args: Mapping[str, Any]) -> Any:
        raise KeyError(op)


def _drive(
    reading, norm, turn: Turn, store=None, registry: Registry = REGISTRY
) -> tuple[Execution, list[tuple[str, str, dict]]]:
    """Run the reading's program with Task 2's executor, resuming the n-th suspended tool step with ``done-<n>``; with
    a ``store``, its references (``invoke``) are expanded first, as the runtime does."""
    meaning = reading.meaning
    if store is not None:
        meaning, _, problems = expand(meaning, resolver(store, turn.chat_id, turn.speaker or ""))
        assert problems == [], problems
    program = Program.from_meaning(meaning)
    resumed: dict[str, Any] = {}
    seen: list[tuple[str, str, dict]] = []
    ex = asyncio.run(execute(program, registry, NoMemory(), turn=turn, spans=norm.spans))
    while ex.pending:
        (step,) = ex.pending
        seen.append((step.step_id, step.tool, step.args))
        resumed[step.step_id] = f"done-{len(seen)}"
        ex = asyncio.run(execute(program, registry, NoMemory(), turn=turn, spans=norm.spans, resumed=resumed))
    return ex, seen


def _ids(reading) -> set[str]:
    """The spec ids a reading applied (an alternate "<id>#v<n>" counts as its spec)."""
    return {spec_id(name) for name in reading.applied}


def _form(draft, text: str) -> CxnSpec:
    return next(s for s in draft.specs if s.provenance["form"] == text)


@pytest.mark.needs_fcg
def test_teach_brew_then_use(engine, tmp_path):
    store = _store(tmp_path, engine)
    _, before = _read(engine, "brew a tea for Ana")
    assert before is None or not before.complete

    draft = teach(store, engine, BREW, registry=REGISTRY)
    assert draft.problems == [] and draft.draft_id
    assert len(draft.specs) == 2
    assert all((s.kind, s.unit, s.status) == ("taught", "root", "staged") for s in draft.specs)
    assert [s.id for s in store.staged()] == sorted(s.id for s in draft.specs)
    assert {s.id for s in store.live()} == {c.id for c in CARDS}  # nothing live changed

    norm, reading = _read(engine, "brew a tea for Ana")
    long_form = _form(draft, "brew {X} for {R}")
    assert reading is not None and reading.complete and long_form.id in _ids(reading)
    assert admissible(reading, store, "c-workshop", "Kira Dace")  # the teacher, in the chat where it was taught
    assert not admissible(reading, store, "c-workshop", "Milo")  # another user
    assert not admissible(reading, store, "c-elsewhere", "Kira Dace")  # another chat
    ex, seen = _drive(reading, norm, _turn("brew a tea for Ana"))
    assert ex.done and ex.problems == []
    assert [(tool, args) for _, tool, args in seen] == [
        ("take", {"item": "water"}),
        ("take", {"item": "herb"}),
        ("make", {"item": "tea"}),
        ("give", {"item": "tea", "person": "Ana"}),
    ]

    norm, reading = _read(engine, "brew tea")  # the short form: R defaults to "me", read by the grammar
    assert reading is not None and reading.complete and _form(draft, "brew {X}").id in _ids(reading)
    ex, seen = _drive(reading, norm, _turn("brew tea"))
    assert ex.done and seen[-1][1:] == ("give", {"item": "tea", "person": "Kira Dace"})

    with pytest.raises(TeachError, match="teacher"):
        confirm(store, engine, draft.draft_id, by="Milo")
    assert [s.id for s in store.staged()] == sorted(s.id for s in draft.specs)
    live = confirm(store, engine, draft.draft_id, by="Kira Dace")
    assert {s.id for s in live} == {s.id for s in draft.specs}
    assert all(s.status == "live" and s.standing.confirmed == 1 for s in live)
    assert store.staged() == []

    norm, reading = _read(engine, "brew a tea for Ana")
    assert reading is not None and reading.complete and long_form.id in _ids(reading)
    assert admissible(reading, store, "c-elsewhere", "Milo")  # live bonfire-wide
    with pytest.raises(TeachError):
        confirm(store, engine, draft.draft_id, by="Kira Dace")  # no longer a draft

    restarted = ConstructionStore(store.writer)  # survives a restart: the live rows are on the graph
    assert restarted.all() == store.all()
    load_store(restarted, engine)
    _, again = _read(engine, "brew a tea for Ana")
    assert again is not None and again.complete and again.applied == reading.applied


@pytest.mark.needs_fcg
def test_competing_form_refused_unless_replace(engine, tmp_path):
    """Review Focus 3: "brew {X}" taught twice. The second is refused unless replace=True and nothing changes in the
    live grammar (store bytes, engine process, the reading); a replace records who replaced what."""
    store = _store(tmp_path, engine)
    first_req = TeachRequest(
        forms=["brew {X}"],
        body=["take herb", "make {X}", "give {X} to me"],
        defaults={},
        teacher="Kira Dace",
        chat_id="c1",
    )
    first = teach(store, engine, first_req, registry=REGISTRY)
    assert first.problems == []
    (old,) = confirm(store, engine, first.draft_id, by="Kira Dace")
    stored, pid = _snapshot(store), engine.pid
    _, reading = _read(engine, "brew tea")
    assert reading is not None and old.id in _ids(reading)

    second_req = TeachRequest(
        forms=["brew {Y}"], body=["take water", "make {Y}", "give {Y} to me"], defaults={}, teacher="Milo", chat_id="c2"
    )
    refused = teach(store, engine, second_req, registry=REGISTRY)
    assert (refused.draft_id, refused.specs) == ("", [])
    assert any(old.id in p for p in refused.problems), refused.problems
    assert _snapshot(store) == stored and engine.pid == pid
    _, same = _read(engine, "brew tea")
    assert same is not None and same.applied == reading.applied

    base_req = TeachRequest(
        forms=["who is {X}?"], body=["take {X}"], defaults={}, teacher="Milo", chat_id="c2", replace=True
    )
    on_base = teach(store, engine, base_req, registry=REGISTRY)  # base changes only with a kernel release
    assert on_base.specs == [] and any("who-be-x" in p for p in on_base.problems), on_base.problems
    assert _snapshot(store) == stored

    draft = teach(store, engine, replace(second_req, replace=True), registry=REGISTRY)
    assert draft.problems == [] and draft.specs[0].provenance["replaces"] == [old.id]
    _, pending = _read(engine, "brew tea")  # a draft loads below everything live: no one's reading changes yet
    assert pending is not None and old.id in _ids(pending)
    with pytest.raises(TeachError, match="teacher"):
        confirm(store, engine, draft.draft_id, by="Kira Dace")
    (new,) = confirm(store, engine, draft.draft_id, by="Milo")
    assert store.get(old.id) is None
    assert new.provenance["replaced"] == [
        {"id": old.id, "kind": "taught", "teacher": "Kira Dace", "by": "Milo", "successor": new.id}
    ]
    norm, now = _read(engine, "brew tea")
    assert now is not None and now.complete and new.id in _ids(now) and old.id not in _ids(now)
    ex, seen = _drive(now, norm, _turn("brew tea", speaker="Milo"))
    assert ex.done and [(tool, args) for _, tool, args in seen][:2] == [
        ("take", {"item": "water"}),
        ("make", {"item": "tea"}),
    ]


@pytest.mark.needs_fcg
def test_teach_refusals_stage_nothing(engine, tmp_path):
    """A body clause the live grammar does not read, a tool outside the catalogue (the keyword set is closed), a slot
    the body never reads as its own span, a missing default, and a reference to an earlier step's result that names
    no one result (two candidates for "it", "them" into a parameter that takes one, only an invoke before it, a step
    that runs once per member), and an "everyone" that would fill both a parameter that takes several values and one
    that takes one: each refused with a reason, nothing staged."""
    store = _store(tmp_path, engine, CARDS_15)
    treat = TeachRequest(["treat {R}"], ["make tea from {R}", "give cake to {R}"], {}, "Kira Dace", "c-workshop")
    for req in (BREW_15, treat):  # "brew {X}" and "treat everyone" are invoke clauses below
        draft = teach(store, engine, req, registry=REGISTRY_15)
        assert draft.problems == [], draft.problems
        confirm(store, engine, draft.draft_id, by="Kira Dace")
    stored = _snapshot(store)
    cases = {
        ("does not read",): TeachRequest(["sip {X}"], ["slurp {X}"], {}, "Ana", "c3"),
        ("unknown op 'zap'",): TeachRequest(["sip {X}"], ["zap {X}"], {}, "Ana", "c3"),
        ("slot {I}",): TeachRequest(["sip {I}"], ["take {I}"], {}, "Ana", "c3"),
        ("no default",): TeachRequest(["sip {X} for {R}", "sip {X}"], ["give {X} to {R}"], {}, "Ana", "c3"),
        ("quantified",): TeachRequest(["stock up"], ["take every herb"], {}, "Ana", "c3"),  # no keyword applies "every"
        ("no literal",): TeachRequest(["{X}"], ["take {X}"], {}, "Ana", "c3"),
        ("letter or digit",): TeachRequest(["{X}!"], ["take {X}"], {}, "Ana", "c3"),
        ("letter or digit ",): TeachRequest(["{X}."], ["take {X}"], {}, "Ana", "c3"),
        (
            "could be",
            "'take water'",
            "'take herb'",
            "say which (use a slot of the form, or 'them' for all)",
        ): TeachRequest(["sip {X}"], ["take water", "take herb", "make {X} from it"], {}, "Ana", "c3"),
        ("several values",): TeachRequest(["serve {R}"], ["take water", "give them to {R}"], {}, "Ana", "c3"),
        ("no earlier step",): TeachRequest(["sip {X} for {R}"], ["brew {X}", "give it to {R}"], {}, "Ana", "c3"),
        ("'give it to R'", "runs once per member", "several results"): TeachRequest(
            ["serve {R}"], ["give tea to everyone", "give it to {R}"], {}, "Ana", "c3"
        ),
        ("'make X from them'", "runs once per member", "several results"): TeachRequest(
            ["sip {X}"], ["give tea to everyone", "make {X} from them"], {}, "Ana", "c3"
        ),
        ("'everyone' fills both", "('material')", "('person')", "two clauses"): TeachRequest(
            ["open the party"], ["treat everyone"], {}, "Ana", "c3"
        ),
    }
    missed = []
    for needles, req in cases.items():
        draft = teach(store, engine, req, registry=REGISTRY_15)
        if draft.specs or not any(all(n.strip() in p for n in needles) for p in draft.problems):
            missed.append((needles, draft.problems))
    assert missed == []
    assert _snapshot(store) == stored and store.staged() == []


@pytest.mark.needs_fcg
def test_composition_host_a_party(engine, tmp_path):
    """A taught "host a party" composes the taught brew: ONE comprehension reads a program that brews for each
    participant (take, take, make, give, member-major) and then puts the cake down once."""
    store = _store(tmp_path, engine)
    brew = teach(store, engine, BREW, registry=REGISTRY)
    confirm(store, engine, brew.draft_id, by="Kira Dace")
    host_req = TeachRequest(
        forms=["host a party"],
        body=["brew tea for everyone", "put the cake on the bench"],
        defaults={},
        teacher="Kira Dace",
        chat_id="c-workshop",
    )
    host = teach(store, engine, host_req, registry=REGISTRY)
    assert host.problems == []
    (party,) = confirm(store, engine, host.draft_id, by="Kira Dace")

    norm, reading = _read(engine, "host a party")
    assert reading is not None and reading.complete and reading.applied == (party.id,)  # one construction, one read
    guests = ("Ana", "Milo", "Kira Dace")
    ex, seen = _drive(reading, norm, _turn("host a party", participants=guests), store)
    assert ex.done and ex.problems == []
    expected = [
        step
        for guest in guests
        for step in (
            ("take", {"item": "water"}),
            ("take", {"item": "herb"}),
            ("make", {"item": "tea"}),
            ("give", {"item": "tea", "person": guest}),
        )
    ] + [("put", {"item": "cake", "place": "bench"})]
    assert [(tool, args) for _, tool, args in seen] == expected
    # Task 2 step ids: "<op index>:<occurrence>"; each brew step is one op run once per guest, then put once
    ops = [step_id.split(":") for step_id, _, _ in seen]
    for k in range(4):
        assert [ops[g * 4 + k] for g in range(3)] == [[ops[k][0], str(g)] for g in range(3)]
    assert ops[-1][1] == "0" and len({op for op, _ in ops}) == 5


@pytest.mark.needs_fcg
def test_composed_brew_feeds_each_result_forward(engine, tmp_path):
    """1.5: "them" in a taught body is the results of the earlier steps that fit, gathered into a parameter that takes
    several values (a list of one when one fits; "it" into such a parameter is a list of one too), and X after the
    step that returns it (make) is make's result: give gets the tea make returned, not the text "tea". The stored
    meaning holds no reference or plural marker. A step that runs once per member rebinds nothing: its slot keeps its
    text."""
    store = _store(tmp_path, engine, CARDS_15)
    draft = teach(store, engine, BREW_15, registry=REGISTRY_15)
    assert draft.problems == [], draft.problems
    stored_ops = [p[0] for spec in draft.specs for p in spec.meaning]
    assert "window" not in stored_ops and "plural" not in stored_ops and stored_ops.count("gather") == 2
    confirm(store, engine, draft.draft_id, by="Kira Dace")
    norm, reading = _read(engine, "brew tea for Ana")
    assert reading is not None and reading.complete and _form(draft, "brew {X} for {R}").id in _ids(reading)
    ex, seen = _drive(reading, norm, _turn("brew tea for Ana"), registry=REGISTRY_15)
    assert ex.done and ex.problems == []
    assert [(tool, args) for _, tool, args in seen] == [
        ("take", {"item": "water"}),
        ("take", {"item": "herb"}),
        ("make", {"item": "tea", "material": ["done-1", "done-2"]}),
        ("give", {"item": "done-3", "person": "Ana"}),
    ]

    for verb, reference in (("steep", "them"), ("infuse", "it")):  # a many parameter always gets a list
        req = TeachRequest(
            [f"{verb} {{X}}"], ["take water", f"make {{X}} from {reference}"], {}, "Kira Dace", "c-workshop"
        )
        one = teach(store, engine, req, registry=REGISTRY_15)
        assert one.problems == [], one.problems
        norm, reading = _read(engine, f"{verb} tea")
        assert reading is not None and reading.complete and one.specs[0].id in _ids(reading)
        ex, seen = _drive(reading, norm, _turn(f"{verb} tea"), registry=REGISTRY_15)
        assert ex.done and [(tool, args) for _, tool, args in seen] == [
            ("take", {"item": "water"}),
            ("make", {"item": "tea", "material": ["done-1"]}),
        ], (verb, seen)

    # a step that runs once per member has several results: the slot it returns keeps its text afterwards
    store.put(_card("pour", ["pour", X, "for", Y], ["?x", "?y"]))
    load_store(store, engine)
    pour = ToolSpec("pour", (ToolParam("item"), ToolParam("person")), False, returns="item")
    registry = REGISTRY_15.with_tools([pour])
    req = TeachRequest(["decant {X}"], ["pour {X} for everyone", "put {X} on the bench"], {}, "Kira Dace", "c-workshop")
    decant = teach(store, engine, req, registry=registry)
    assert decant.problems == [], decant.problems
    norm, reading = _read(engine, "decant tea")
    assert reading is not None and reading.complete and decant.specs[0].id in _ids(reading)
    ex, seen = _drive(reading, norm, _turn("decant tea", participants=("Ana", "Milo")), registry=registry)
    assert ex.done and [(tool, args) for _, tool, args in seen] == [
        ("pour", {"item": "tea", "person": "Ana"}),
        ("pour", {"item": "tea", "person": "Milo"}),
        ("put", {"item": "tea", "place": "bench"}),
    ]


@pytest.mark.needs_fcg
def test_singular_reference_binds_its_one_producer(engine, tmp_path):
    """1.5: "it" in a taught body is the result of the one earlier step that fits: give gets what take returned."""
    store = _store(tmp_path, engine, CARDS_15)
    req = TeachRequest(["fetch {X} for {R}"], ["take {X}", "give it to {R}"], {}, "Kira Dace", "c-workshop")
    draft = teach(store, engine, req, registry=REGISTRY_15)
    assert draft.problems == [], draft.problems
    norm, reading = _read(engine, "fetch tea for Ana")
    assert reading is not None and reading.complete and draft.specs[0].id in _ids(reading)
    ex, seen = _drive(reading, norm, _turn("fetch tea for Ana"), registry=REGISTRY_15)
    assert ex.done and [(tool, args) for _, tool, args in seen] == [
        ("take", {"item": "tea"}),
        ("give", {"item": "done-1", "person": "Ana"}),
    ]


@pytest.mark.needs_fcg
def test_composed_party_keeps_each_member_apart(engine, tmp_path):
    """Review Focus 1: a taught "throw a party" invokes the composed brew for everyone. Per participant, make gathers
    only that participant's two takes, and that participant's give gets that participant's make result. A taught body
    whose "everyone" fills an invoked phrase's slot that only a parameter taking several values reads keeps it whole."""
    store = _store(tmp_path, engine, CARDS_15)
    brew = teach(store, engine, BREW_15, registry=REGISTRY_15)
    assert brew.problems == [], brew.problems
    confirm(store, engine, brew.draft_id, by="Kira Dace")
    party_req = TeachRequest(
        forms=["throw a party"],
        body=["brew tea for everyone", "put the cake on the bench"],
        defaults={},
        teacher="Kira Dace",
        chat_id="c-workshop",
    )
    party = teach(store, engine, party_req, registry=REGISTRY_15)
    assert party.problems == [], party.problems
    confirm(store, engine, party.draft_id, by="Kira Dace")

    norm, reading = _read(engine, "throw a party")
    assert reading is not None and reading.complete
    guests = ("Ana", "Milo", "Kira Dace")
    ex, seen = _drive(reading, norm, _turn("throw a party", participants=guests), store, registry=REGISTRY_15)
    assert ex.done and ex.problems == []
    # _drive resumes the n-th step with "done-<n>": member g's takes are 4g+1 and 4g+2, its make 4g+3
    expected = [
        step
        for g, guest in enumerate(guests)
        for step in (
            ("take", {"item": "water"}),
            ("take", {"item": "herb"}),
            ("make", {"item": "tea", "material": [f"done-{4 * g + 1}", f"done-{4 * g + 2}"]}),
            ("give", {"item": f"done-{4 * g + 3}", "person": guest}),
        )
    ] + [("put", {"item": "cake", "place": "bench"})]
    assert [(tool, args) for _, tool, args in seen] == expected

    # an "everyone" that fills an invoked phrase's slot is decided by what that slot feeds, as in a live turn:
    # "stock {R}" makes tea from R (a parameter that takes several values), so "stock everyone" makes once, from all
    for req in (
        TeachRequest(["stock {R}"], ["make tea from {R}"], {}, "Kira Dace", "c-workshop"),
        TeachRequest(["fill the shelf"], ["stock everyone"], {}, "Kira Dace", "c-workshop"),
    ):
        draft = teach(store, engine, req, registry=REGISTRY_15)
        assert draft.problems == [], draft.problems
    norm, reading = _read(engine, "fill the shelf")
    assert reading is not None and reading.complete
    ex, seen = _drive(reading, norm, _turn("fill the shelf", participants=guests), store, registry=REGISTRY_15)
    assert ex.done and [(tool, args["item"], list(args["material"])) for _, tool, args in seen] == [
        ("make", "tea", list(guests))
    ], seen


@pytest.mark.needs_fcg
def test_standing_never_beats_specificity(engine, tmp_path):
    """A confirmed "brew {X}" with 10 good uses still loads below a confirmed "brew {X} for {R}": "brew a tea for ana"
    reads the for-R form (X=tea, R=ana), never X="a tea for ana" (the spike's shadowing, revived by standing)."""
    store = _store(tmp_path, engine)
    draft = teach(store, engine, BREW, registry=REGISTRY)
    confirm(store, engine, draft.draft_id, by="Kira Dace")
    short, long_form = _form(draft, "brew {X}"), _form(draft, "brew {X} for {R}")
    for _ in range(10):
        store.record_use(short.id, "ok")
    assert store.get(short.id).standing.uses_ok == 10 and store.get(long_form.id).standing.uses_ok == 0
    load_store(store, engine)
    norm, reading = _read(engine, "brew a tea for ana")
    assert reading is not None and reading.complete
    assert long_form.id in _ids(reading) and short.id not in _ids(reading)
    ex, seen = _drive(reading, norm, _turn("brew a tea for ana"))
    assert ex.done and seen[2][1:] == ("make", {"item": "tea"})
    assert seen[3][1:] == ("give", {"item": "tea", "person": "ana"})


@pytest.mark.needs_fcg
def test_second_draft_of_one_skeleton_is_refused_at_confirm(engine, tmp_path):
    """Two teachers stage a draft with the same skeleton (competition at teach time is against live roots only). The
    first confirm goes live; the second teacher's confirm is refused, naming the live id, and changes nothing."""
    store = _store(tmp_path, engine)
    kira = TeachRequest(["brew {X}"], ["take herb", "make {X}", "give {X} to me"], {}, "Kira Dace", "c1")
    milo = TeachRequest(["brew {Y}"], ["take water", "make {Y}", "give {Y} to me"], {}, "Milo", "c2")
    first, second = (teach(store, engine, req, registry=REGISTRY) for req in (kira, milo))
    assert first.problems == [] and second.problems == []
    (live,) = confirm(store, engine, first.draft_id, by="Kira Dace")
    stored = _snapshot(store)
    with pytest.raises(TeachError, match=live.id):
        confirm(store, engine, second.draft_id, by="Milo")
    assert _snapshot(store) == stored
    assert [s.id for s in store.staged()] == [second.specs[0].id] and store.get(live.id).status == "live"
    _, reading = _read(engine, "brew tea")
    assert reading is not None and live.id in _ids(reading)


@pytest.mark.needs_fcg
def test_forms_read_in_context(engine, tmp_path):
    """I-5: a form's literals are normalised in context, so a taught "host {X}" / "bake {X}" read every use: "host Zed"
    and "host the Zed" both plan the taught step. (The turn parser plus the subject frame reads "host" a verb in both;
    the compiler's alternates still serve a tagger that flips a word, and the sm fallback.)"""
    store = _store(tmp_path, engine)
    specs = {}
    for form in ("host {X}", "bake {X}"):
        draft = teach(store, engine, TeachRequest([form], ["make {X}"], {}, "Ana", "c3"), registry=REGISTRY)
        assert draft.problems == [], draft.problems
        (specs[form],) = confirm(store, engine, draft.draft_id, by="Ana")
    for text, form, item in (
        ("host the cake", "host {X}", "cake"),
        ("host Zed", "host {X}", "Zed"),
        ("host the Zed", "host {X}", "Zed"),
        ("host Milo", "host {X}", "Milo"),
        ("bake a pie", "bake {X}", "pie"),
        ("bake Zorvex", "bake {X}", "Zorvex"),
    ):
        norm, reading = _read(engine, text)
        assert reading is not None and reading.complete and specs[form].id in _ids(reading), (text, reading)
        assert admissible(reading, store, "c-any", "Milo")
        ex, seen = _drive(reading, norm, _turn(text))
        assert ex.done and [(t, a) for _, t, a in seen] == [("make", {"item": item})]
    used = store.record_uses({name: "ok" for name in reading.applied})
    assert used[specs["bake {X}"].id].standing.uses_ok == 1


@pytest.mark.needs_fcg
def test_draft_does_not_leak_through_composition(engine, tmp_path):
    """I-1: a body that reads through the teacher's own staged brew records it; the host cannot be confirmed (and so
    go live for everyone with brew baked in) until brew is. M-4: someone else cannot read through the draft at all."""
    store = _store(tmp_path, engine)
    brew = teach(store, engine, BREW, registry=REGISTRY)
    host_req = TeachRequest(["host a party"], ["brew tea for everyone"], {}, "Kira Dace", "c-workshop")
    host = teach(store, engine, host_req, registry=REGISTRY)
    long_form = _form(brew, "brew {X} for {R}")  # "brew tea for everyone" reads through the for-R form only
    assert host.problems == [] and host.specs[0].provenance["reads_staged"] == [long_form.id]
    elsewhere = teach(store, engine, TeachRequest(["sip {X}"], ["brew {X}"], {}, "Milo", "c2"), registry=REGISTRY)
    assert elsewhere.specs == [] and any("a draft not usable here" in p for p in elsewhere.problems)
    with pytest.raises(TeachError) as refused:
        confirm(store, engine, host.draft_id, by="Kira Dace")
    assert long_form.id in str(refused.value)
    assert {s.id for s in store.staged()} == {s.id for s in [*brew.specs, *host.specs]}
    confirm(store, engine, brew.draft_id, by="Kira Dace")
    (party,) = confirm(store, engine, host.draft_id, by="Kira Dace")
    assert party.status == "live"


@pytest.mark.needs_fcg
def test_body_clause_through_a_guarded_reading_is_refused(engine, tmp_path):
    """I-2: a body clause whose reading the do-support guard refuses (a subject-shaped card reading the auxiliary as
    its verb) is refused with the guard's reason."""
    v = {"slot": "v", "fills": "verb"}
    card = CxnSpec(
        id="card-perform",
        kind="hand",
        unit="root",
        form=["please", v, Y],
        meaning=[["procedure", "?p", "perform"], ["seq", "?p", "1", "?s1"], ["perform", "?s1", "?v", "?y"]],
        status="live",
    )
    store = _store(tmp_path, engine)
    store.put(card)
    load_store(store, engine)
    registry = REGISTRY.with_tools([ToolSpec.simple("perform", ("verb", "item"), False)])
    draft = teach(store, engine, TeachRequest(["chore {X}"], ["please do {X}"], {}, "Ana", "c3"), registry=registry)
    assert draft.specs == [] and any("auxiliary" in p for p in draft.problems), draft.problems


# -- 1.5.1: teaching is admission (the office's catalogue) ------------------------------------------------------------

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "office-catalogue.json"
OFFICE = parse_catalogue(FIXTURE.read_bytes(), str(FIXTURE))
OFFICE_REGISTRY = default_registry(OFFICE.tools)
#: the teach that froze the room on 10-23 (3,902 parses in the sweep)
LOOK_OFFICE = TeachRequest(
    forms=["look at the office"],
    body=["look at the inbox", "look at To do", "look at Doing", "look at Review", "look at the archive"],
    defaults={},
    teacher="Ana",
    chat_id="c1",
)
FILE = TeachRequest(["file {X}"], ["make a ticket from {X}"], {}, "Ana", "c1")
TRIAGE = TeachRequest(["triage {X} for {R}"], ["take {X}", "give {X} to {R}"], {}, "Ana", "c1")
STANDUP = TeachRequest(["standup"], ["look at Doing", "look at Review"], {}, "Ana", "c1")


def _office(tmp_path, engine) -> ConstructionStore:
    store = ConstructionStore(MemoryWriter())
    store.attach_cards(OFFICE.cards)
    load_store(store, engine)
    return store


class _Reads:
    """Every read the engine is asked for (``comprehend`` and ``comprehend_all``), by text. Past ``limit`` reads or
    ``budget_s`` seconds a read fails at once: a sweep reading thousands of texts would otherwise run for minutes."""

    def __init__(self, monkeypatch, engine, *, limit: int | None = None, budget_s: float | None = None) -> None:
        self.texts: list[str] = []
        self.limit, self.budget_s, self.started = limit, budget_s, time.perf_counter()
        for method in ("comprehend", "comprehend_all"):
            monkeypatch.setattr(engine, method, functools.partial(self._read, getattr(engine, method)))

    def restart(self, budget_s: float | None) -> None:
        self.texts.clear()
        self.budget_s, self.started = budget_s, time.perf_counter()

    def _read(self, real, text: str, *args: Any, **kwargs: Any):
        self.texts.append(text)
        if self.limit is not None and len(self.texts) > self.limit:
            raise AssertionError(f"more than {self.limit} reads: {self.texts[:12]}")
        if self.budget_s is not None and time.perf_counter() - self.started > self.budget_s:
            raise AssertionError(f"past {self.budget_s} s after {len(self.texts)} reads")
        return real(text, *args, **kwargs)


@pytest.mark.needs_fcg
def test_teach_reads_only_its_own_renderings(engine, tmp_path, monkeypatch):
    """A teach reads each body clause once and each of its own renderings twice (before the draft loads, for
    ``competes``; with the draft on top, the self-read): never another construction's renderings."""
    store = _office(tmp_path, engine)
    reads = _Reads(monkeypatch, engine, limit=len(LOOK_OFFICE.body) + 2)  # one rendering: "look at the office"
    draft = teach(store, engine, LOOK_OFFICE, registry=OFFICE_REGISTRY)
    assert draft.problems == [], draft.problems
    own = [normalise(r).text for spec in draft.specs for r in renderings(spec)]
    body = [normalise(clause).text for clause in LOOK_OFFICE.body]
    assert Counter(reads.texts) == Counter([*body, *own, *own]), reads.texts


@pytest.mark.needs_fcg
def test_teach_is_fast(engine, tmp_path, monkeypatch):
    """Spec section 5, on a runtime that has just loaded its grammar (the parser loaded; ``load_store``, at start and on
    every catalogue reload, compiled the live grammar, parsing each live form once) and never taught: the first teach
    of each form shape takes under 2 s for "look at the office", under 1 s for "file {X}", "triage {X} for {R}" and
    "standup" (whose shape is the office's)."""
    for cache in (_compiled, normalised_rendering, _subject_shaped, _trigger, _skeleton_of):  # every form cache emptied
        cache.cache_clear()
    # compiles (and so parses) the live grammar again, as a runtime does when it loads
    store = _office(tmp_path, engine)
    reads = _Reads(monkeypatch, engine)
    slow: list[tuple[list[str], list[str], float]] = []
    for req, budget in ((LOOK_OFFICE, 2.0), (FILE, 1.0), (TRIAGE, 1.0), (STANDUP, 1.0)):
        _skeleton_of.cache_clear()  # each the first teach of its shape in the process
        reads.restart(5 * budget)  # a runaway (a sweep of thousands of reads) stops here, not minutes later
        started = time.perf_counter()
        draft = teach(store, engine, req, registry=OFFICE_REGISTRY)
        took = time.perf_counter() - started
        if draft.problems or took >= budget:
            slow.append((req.forms, draft.problems, round(took, 2)))
    assert slow == [], slow


@pytest.mark.needs_fcg
def test_competes_names_the_card_it_overlaps(engine, tmp_path):
    """The draft says what the live grammar reads each form's own renderings as before it loads: the look card, with
    its place bound; a form nothing reads competes with nothing."""
    store = _office(tmp_path, engine)
    draft = teach(store, engine, LOOK_OFFICE, registry=OFFICE_REGISTRY)
    assert draft.problems == [] and set(draft.competes) == {"look at the office"}, draft.problems
    look = [f for f in draft.competes["look at the office"] if f["construction"] == "card.look.0"]
    assert [(f["name"], f["kind"], f["roles"], f["complete"]) for f in look] == [
        ("look", "tool", {"place": "office"}, True)
    ], draft.competes
    filed = teach(store, engine, FILE, registry=OFFICE_REGISTRY)
    assert filed.problems == [] and filed.competes == {"file {X}": []}, filed.competes


@pytest.mark.needs_fcg
def test_duplicate_is_refused_with_its_id(engine, tmp_path):
    """A form whose construction is already live (the same form and meaning up to slot names) is refused naming it, and
    nothing is staged; the same form with another body still competes."""
    store = _office(tmp_path, engine)
    draft = teach(store, engine, FILE, registry=OFFICE_REGISTRY)
    assert draft.problems == [], draft.problems
    (live,) = confirm(store, engine, draft.draft_id, by="Ana")
    stored = _snapshot(store)
    again = teach(store, engine, FILE, registry=OFFICE_REGISTRY)
    assert (again.draft_id, again.specs) == ("", [])
    assert again.problems == [f"form 'file {{X}}' is already the live construction {live.id}"]
    renamed = TeachRequest(["file {Y}"], ["make a ticket from {Y}"], {}, "Milo", "c2")
    assert teach(store, engine, renamed, registry=OFFICE_REGISTRY).problems == [
        f"form 'file {{Y}}' is already the live construction {live.id}"
    ]
    other = teach(store, engine, TeachRequest(["file {X}"], ["take {X}"], {}, "Milo", "c2"), registry=OFFICE_REGISTRY)
    assert other.specs == [] and any(f"competes with the live construction {live.id}" in p for p in other.problems)
    assert _snapshot(store) == stored and store.staged() == []


@pytest.mark.needs_fcg
def test_broad_phrase_is_no_longer_refused(engine, tmp_path):
    """ "who {X}?" over the base who-be-x is staged: competition is settled at read time (frames), not at teach."""
    store = _office(tmp_path, engine)
    draft = teach(store, engine, TeachRequest(["who {X}?"], ["take {X}"], {}, "Milo", "c2"), registry=OFFICE_REGISTRY)
    assert draft.problems == [] and [s.status for s in draft.specs] == ["staged"], draft.problems
    assert "shadows" not in draft.specs[0].provenance


@pytest.mark.needs_fcg
def test_confirm_runs_no_probe(engine, tmp_path, monkeypatch):
    """A confirm reads nothing: it checks the teacher, the drafts it waits for and the rivals, and goes live."""
    store = _office(tmp_path, engine)
    draft = teach(store, engine, FILE, registry=OFFICE_REGISTRY)
    assert draft.problems == [], draft.problems
    reads = _Reads(monkeypatch, engine, limit=0)
    (live,) = confirm(store, engine, draft.draft_id, by="Ana")
    assert reads.texts == [] and live.status == "live"


@pytest.mark.parametrize("n", [1, 50])
def test_room_refusal_does_not_lower_standing(tmp_path, n):
    """A repair the room caused (a precondition, a declined step, a tool error) is a ``world`` use: it counts
    ``uses_world`` only, so the construction's precedence and its failed uses stay as they were."""
    store = ConstructionStore(MemoryWriter())
    store.put(_row("m-job", "mined"))
    before = store.get("m-job")
    for _ in range(n):  # a flood of refusals never demotes either
        store.record_uses({"m-job": "world"})
    after = store.get("m-job")
    assert after.standing == replace(before.standing, uses_world=n)
    assert precedence(after) == precedence(before)
    assert ConstructionStore(store.writer).get("m-job").standing.uses_world == n  # written through


def test_a_removal_with_no_successor_names_the_rows_that_invoke_it(tmp_path):
    """A confirm that removes a row by generalisation (``successor`` None) orphans the rows invoking it, directly or
    through an id it once replaced; a removal with a successor orphans nobody."""
    store = ConstructionStore(MemoryWriter())
    old = _row("hum.old", "taught", provenance={"replaced": [{"id": "hum.older", "successor": "hum.old"}]})
    by_id = replace(_row("serenade.a", "taught"), meaning=[["invoke", "?s", "hum.old", "?x"]])
    by_older = replace(_row("serenade.b", "taught"), meaning=[["invoke", "?s", "hum.older", "?x"]])
    store.update(puts=[old, by_id, by_older, _row("plain", "taught")], replacing_taught=True)
    gone = [{"id": "hum.old", "successor": None}, {"id": "plain", "successor": "plain.new"}]
    assert _orphan_warnings(store, gone, set()) == [
        "hum.old was removed with no successor: serenade.a, serenade.b invoke it and will repair"
    ]
    assert _orphan_warnings(store, [{"id": "hum.old", "successor": "hum.new"}], set()) == []


def test_misread_lowers_standing(tmp_path):
    """A ``failed`` use (a misreading, an empty result, an unresolved name) counts against the construction."""
    store = ConstructionStore(MemoryWriter())
    store.put(_row("m-job", "mined"))
    before = precedence(store.get("m-job"))
    store.record_uses({"m-job": "failed"})
    after = store.get("m-job")
    assert after.standing.uses_failed == 1 and after.standing.uses_world == 0
    assert precedence(after) < before  # uses_ok - uses_failed fell


def test_record_uses_refuses_an_unknown_outcome_and_writes_nothing(tmp_path):
    """An outcome is one of ok / failed / world: a bool or any other string is a StoreError before any write."""
    store = ConstructionStore(MemoryWriter())
    store.put(_row("m-job", "mined"))
    before = _snapshot(store)
    with pytest.raises(StoreError, match="outcome"):
        store.record_uses({"m-job": True})  # type: ignore[dict-item]
    with pytest.raises(StoreError, match="outcome"):
        store.record_uses({"m-job": "ok", "m-job#v1": "bogus"})
    assert _snapshot(store) == before and store.get("m-job").standing == Standing()


# -- 1.5.2: compose any tool (structured steps, runners, untyped results fill string parameters) ----------------------

#: the agent's catalogue source as the Hermes plugin generates it: a read-only web search, and think
AGENT_ROWS = [
    {
        "name": "web_search",
        "description": "Search the web",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "x-args": {"query": {"role": "theme"}},
        "x-side-effect": False,
        "x-consent": "speaker_turn",
        "x-runner": "agent",
    },
    {
        "name": "think",
        "description": "the agent does this step itself with its own context",
        "input_schema": {
            "type": "object",
            "properties": {"instruction": {"type": "string"}, "input": {"type": "string"}},
            "required": ["instruction"],
        },
        "x-args": {"instruction": {"role": "theme"}, "input": {"role": "source"}},
        "x-side-effect": False,
        "x-consent": "speaker_turn",
        "x-runner": "agent",
    },
]
_OFFICE_ROWS = json.loads(FIXTURE.read_bytes())
COMPOSED = parse_catalogue(json.dumps({**_OFFICE_ROWS, "tools": [*_OFFICE_ROWS["tools"], *AGENT_ROWS]}).encode(), "c")
COMPOSED_REGISTRY = default_registry(COMPOSED.tools)


def _call(tool: str, **args: str) -> dict[str, Any]:
    return {"call": tool, "args": args}


SITREP = TeachRequest(
    ["sitrep {X}"],
    [
        _call("web_search", query="{X}"),
        _call("think", instruction="summarise it in three lines", input="it"),
        _call("write", body="it"),
    ],
    {},
    "Kira Dace",
    "c-workshop",
)


def _composed(tmp_path, engine) -> ConstructionStore:
    store = ConstructionStore(MemoryWriter())
    store.attach_cards(COMPOSED.cards)
    load_store(store, engine)
    return store


def _tool_ops(spec: CxnSpec) -> list[list[Any]]:
    return [p for p in spec.meaning if (s := COMPOSED_REGISTRY.specs.get(p[0])) is not None and s.kind == "tool"]


def _drive_phrase(engine, store, text: str) -> tuple[Execution, list[tuple[str, str, dict]]]:
    norm, reading = _read(engine, text)
    assert reading is not None and reading.complete, reading
    return _drive(reading, norm, _turn(text), store=store, registry=COMPOSED_REGISTRY)


@pytest.mark.needs_fcg
def test_a_structured_step_compiles_like_its_clause(engine, tmp_path):
    """C3: a body step written as a call compiles to the ops the grammar reads its clause as (the same canonical
    meaning, up to variable names), and its form carries the same slot notes."""
    from memory_kernel.runtime.teach import _canonical

    store = _composed(tmp_path, engine)
    for clause, call in (
        ("look at {X}", _call("look", place="{X}")),
        ("make a ticket from {X}", _call("make", title="{X}")),
    ):
        read = teach(
            store, engine, TeachRequest(["glance at {X}"], [clause], {}, "Ana", "c1"), registry=COMPOSED_REGISTRY
        )
        called = teach(
            store, engine, TeachRequest(["glance at {X}"], [call], {}, "Milo", "c2"), registry=COMPOSED_REGISTRY
        )
        assert read.problems == [] and called.problems == [], (read.problems, called.problems)
        (a,), (b,) = read.specs, called.specs
        assert Counter(_canonical(a)) == Counter(_canonical(b)), (a.meaning, b.meaning)
        assert a.form == b.form


@pytest.mark.needs_fcg
def test_structured_steps_compose_any_tool(engine, tmp_path):
    """Any tool of the catalogue composes, a tool with no forms too: web search, think, write; "it" is the step
    before's result; the request is kept as it was sent; the phrase runs its three steps in order."""
    store = _composed(tmp_path, engine)
    draft = teach(store, engine, SITREP, registry=COMPOSED_REGISTRY)
    assert draft.problems == [] and [s.status for s in draft.specs] == ["staged"], draft.problems
    (spec,) = draft.specs
    search, think, write = _tool_ops(spec)
    assert [search[0], think[0], write[0]] == ["web_search", "think", "write"]
    assert search[2] == "?x" and think[3] == search[1] and write[2] == think[1]
    consts = {p[1]: p[2] for p in spec.meaning if p[0] == "const"}
    assert consts[think[2]] == "summarise it in three lines"
    assert spec.provenance["request"]["body"] == SITREP.body
    ex, seen = _drive_phrase(engine, store, "sitrep news")
    assert ex.done and ex.problems == [], ex.problems
    assert [(tool, args) for _, tool, args in seen] == [
        ("web_search", {"query": "news"}),
        ("think", {"instruction": "summarise it in three lines", "input": "done-1"}),
        ("write", {"body": "done-2"}),
    ]


@pytest.mark.needs_fcg
def test_an_untyped_result_fills_a_string_parameter(engine, tmp_path):
    """C13: a result with no declared kind fills a kinded parameter (make's title is an Artifact), in a structured
    step and in a read clause alike; a producer of the parameter's kind still wins over an untyped one; live, the
    plan issues the step with that result and refuses nothing."""
    store = _composed(tmp_path, engine)
    body = [
        _call("web_search", query="{X}"),
        _call("think", instruction="pick the most urgent", input="it"),
        _call("make", title="it"),
    ]
    draft = teach(
        store, engine, TeachRequest(["urgent {X}"], body, {}, "Kira Dace", "c-workshop"), registry=COMPOSED_REGISTRY
    )
    assert draft.problems == [], draft.problems
    ex, seen = _drive_phrase(engine, store, "urgent news")
    assert ex.done and ex.problems == [], ex.problems
    assert seen[-1][1:] == ("make", {"title": "done-2"})
    # a read clause: "it" into make's title is the untyped search result (1.5.1 refused it)
    clause = TeachRequest(
        ["file news on {X}"], [_call("web_search", query="{X}"), "make a ticket from it"], {}, "Ana", "c1"
    )
    assert teach(store, engine, clause, registry=COMPOSED_REGISTRY).problems == []
    # typed first: take returns the ticket, look's result is untyped, so "it" into give's ticket is take's
    typed = TeachRequest(["hand {X} to {R}"], ["take {X}", "look at Doing", "give it to {R}"], {}, "Ana", "c1")
    staged = teach(store, engine, typed, registry=COMPOSED_REGISTRY)
    assert staged.problems == [], staged.problems
    take, _look, give = _tool_ops(staged.specs[0])
    assert give[2] == take[1]


@pytest.mark.needs_fcg
def test_it_in_a_sentence_after_untyped_results_is_the_nearest(engine, tmp_path):
    """C13 in a sentence: "it" after two untyped results (an agent tool's, think's) is the nearest of them, as a
    structured step's "it" is the step before, into a kind-less parameter (write's body) and a kinded one (make's
    title) alike; with a result of a tool that declares one among them, "it" still has to say which."""
    store = _composed(tmp_path, engine)
    lead = [_call("web_search", query="{X}"), _call("think", instruction="pick the most urgent", input="it")]
    for form, sentence, tool, param in (
        ("digest {X}", "write it", "write", "body"),
        ("hot {X}", "make a ticket from it", "make", "title"),
    ):
        draft = teach(
            store,
            engine,
            TeachRequest([form], [*lead, sentence], {}, "Kira Dace", "c-workshop"),
            registry=COMPOSED_REGISTRY,
        )
        assert draft.problems == [], (sentence, draft.problems)
        _search, think, last = _tool_ops(draft.specs[0])
        assert last[0] == tool and last[2] == think[1], (sentence, last, think)
        ex, seen = _drive_phrase(engine, store, form.replace("{X}", "news"))
        assert ex.done and ex.problems == [], (sentence, ex.problems)
        assert seen[-1][1:] == (tool, {param: "done-2"})
    mixed = TeachRequest(
        ["recap {X}"], ["take {X}", _call("think", instruction="say why", input="it"), "write it"], {}, "Ana", "c1"
    )
    refused = teach(store, engine, mixed, registry=COMPOSED_REGISTRY)
    assert refused.specs == [] and any("say which" in p and "structured step" in p for p in refused.problems), (
        refused.problems
    )


@pytest.mark.needs_fcg
def test_a_structured_step_is_checked(engine, tmp_path):
    """Each structured step is checked against the catalogue: an unknown tool or keyword, a parameter the tool does not
    have, one left out (a step gives every parameter the tool declares, think's input too), a slot no form has, "it"
    with no step before, "them" into one value, a value mixing a slot and text, a malformed call. A slot the form
    leaves out is its default's text."""
    store = _composed(tmp_path, engine)
    cases = [
        ([_call("web_serch", query="{X}")], "'web_serch'"),
        ([_call("recall", text="{X}")], "'recall'"),
        ([_call("web_search", query="{X}", limit="5")], "'limit'"),
        ([_call("web_search")], "'query'"),
        ([_call("think", instruction="{X}")], "'input'"),
        ([_call("look", place="{Y}")], "{Y}"),
        ([_call("write", body="it")], "'it'"),
        ([_call("web_search", query="{X}"), _call("write", body="them")], "'them'"),
        ([_call("web_search", query="news about {X}")], "news about {X}"),
        ([{"call": "look", "args": {"place": 3}}], "args"),
        ([{"tool": "look"}], "call"),
    ]
    for body, named in cases:
        draft = teach(store, engine, TeachRequest(["peek {X}"], body, {}, "Ana", "c1"), registry=COMPOSED_REGISTRY)
        assert draft.specs == [] and any(named in p for p in draft.problems), (body, draft.problems)
    assert store.staged() == []
    muse = TeachRequest(
        ["muse on {X} with {Y}", "muse on {X}"],
        [_call("think", instruction="{X}", input="{Y}")],
        {"Y": "the board"},
        "Kira Dace",
        "c-workshop",
    )
    assert teach(store, engine, muse, registry=COMPOSED_REGISTRY).problems == []
    ex, seen = _drive_phrase(engine, store, "muse on lunch")
    assert [(tool, args) for _, tool, args in seen] == [("think", {"instruction": "lunch", "input": "the board"})]


@pytest.mark.needs_fcg
def test_an_agent_step_after_a_board_step_is_refused(engine, tmp_path):
    """C12: a step that runs in the agent after an office step that waits for the board's approval (archive) could
    never run: refused, through an invoked phrase too. An office step after it, or an agent step before it, is fine."""
    store = _composed(tmp_path, engine)
    think = _call("think", instruction="say why it closed", input="it")
    draft = teach(
        store, engine, TeachRequest(["shut {X}"], ["archive {X}", think], {}, "Ana", "c1"), registry=COMPOSED_REGISTRY
    )
    assert draft.specs == [] and any("'think'" in p and "'archive'" in p for p in draft.problems), draft.problems
    ok = [
        TeachRequest(["shut {X}"], ["archive {X}", "write closed"], {}, "Ana", "c1"),
        TeachRequest(
            ["vet {X}"], [_call("think", instruction="is it done?", input="{X}"), "archive {X}"], {}, "Ana", "c1"
        ),
    ]
    for req in ok:
        assert teach(store, engine, req, registry=COMPOSED_REGISTRY).problems == [], req.forms
    closed = teach(
        store, engine, TeachRequest(["close {X}"], ["archive {X}"], {}, "Ana", "c1"), registry=COMPOSED_REGISTRY
    )
    confirm(store, engine, closed.draft_id, by="Ana")
    nested = TeachRequest(
        ["end {X}"], ["close {X}", _call("think", instruction="say why", input="{X}")], {}, "Ana", "c1"
    )
    refused = teach(store, engine, nested, registry=COMPOSED_REGISTRY)
    assert refused.specs == [] and any("'think'" in p and "'archive'" in p for p in refused.problems), refused.problems


class _NoGraph:
    async def rows(self, cypher: str, params: Any) -> list[dict]:
        return []


@pytest.mark.needs_fcg
def test_a_phrase_whose_tool_is_gone_is_refused(engine, tmp_path):
    """Review Focus 5: a confirmed phrase calls web_search; the catalogue installed next has no agent source: the plan
    of "sitrep news" says which tool is gone, never a traceback, and nothing is issued."""
    from memory_kernel.runtime.api import Runtime
    from memory_kernel.runtime.lexicon import Lexicon, QuestionMap

    writer = MemoryWriter()  # the one bonfire both catalogues are installed on

    def runtime(catalogue) -> Runtime:
        registry = default_registry(catalogue.tools)
        lexicon = Lexicon([], QuestionMap.load())
        return Runtime(
            "fire",
            engine,
            _NoGraph(),
            registry,
            lexicon=lexicon,
            store=ConstructionStore(writer),
            cards=catalogue.cards,
            record_standing=False,
        )

    first = runtime(COMPOSED)
    draft = asyncio.run(first.teach(SITREP))
    assert draft.problems == [], draft.problems
    asyncio.run(first.confirm(draft.draft_id, "Kira Dace"))
    office = runtime(OFFICE)
    turn = Turn("sitrep news", "Kira Dace", None, (), "c-workshop", None, "2026-10-23T12:00:00+00:00")
    plan = asyncio.run(office.plan(turn))
    assert plan.tool_steps == [] and plan.status != "done", plan
    assert any("web_search" in d for d in plan.diagnostics), plan.diagnostics
