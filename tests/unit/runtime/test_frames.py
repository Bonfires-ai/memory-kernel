"""Frames (1.5.1): every plan says what read which words of the turn. A frame part per leading verb of every live card
and taught phrase reads that verb anywhere ("make onboard josh" is a partial make frame), a complete reading reports
its root bound, up to three competing analyses surface, the editor and the replies are base rows, and no catch-all
ever completes a message."""

import asyncio
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from memory_kernel.runtime.api import Runtime, TurnPlan, default_registry
from memory_kernel.runtime.base import base_specs
from memory_kernel.runtime.catalogue import parse_catalogue
from memory_kernel.runtime.constructions import CxnSpec
from memory_kernel.runtime.executor import Turn
from memory_kernel.runtime.frames import frame_parts, trigger
from memory_kernel.runtime.lexicon import Lexicon, QuestionMap
from unit.runtime.memory_writer import memory_store

PORT = 9724  # 9720-9723, 9725-9727 are the other engine test modules'
FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "office-catalogue.json"
OFFICE = parse_catalogue(FIXTURE.read_bytes(), str(FIXTURE))
X = {"slot": "x", "fills": "np"}
R = {"slot": "r", "fills": "np"}

#: a live taught phrase whose form starts with a verb
TRIAGE = CxnSpec(
    id="triage.t0000000001.0",
    kind="taught",
    unit="root",
    form=["triage", X, "for", R],
    meaning=[
        ["procedure", "?t-proc", "triage"],
        ["seq", "?t-proc", "1", "?t-s1"],
        ["take", "?t-s1", "?x"],
        ["seq", "?t-proc", "2", "?t-s2"],
        ["give", "?t-s2", "?x", "?r"],
    ],
    status="live",
    provenance={"teacher": "Ana", "chat_id": "c1", "draft_id": "t0000000001", "form": "triage {X} for {R}"},
)
#: a live taught phrase as broad as the base memory question it competes with ("who is Zed?")
WHO = CxnSpec(
    id="who.t0000000002.0",
    kind="taught",
    unit="root",
    form=["who", X, "?"],
    meaning=[["find-entity", "?xe", "?x"], ["identity-lines", "?ans", "?xe"], ["answer", "?ans"]],
    status="live",
    provenance={"teacher": "Ana", "chat_id": "c1", "draft_id": "t0000000002", "form": "who {X}?"},
)
Y = {"slot": "y", "fills": "np"}


def _taught(draft: str, form: list, meaning: list, written: str) -> CxnSpec:
    word = form[0].split()[0]
    provenance = {"teacher": "Ana", "chat_id": "c1", "draft_id": draft, "form": written}
    return CxnSpec(
        id=f"{word}.{draft}.0",
        kind="taught",
        unit="root",
        form=form,
        meaning=meaning,
        status="live",
        provenance=provenance,
    )


#: live taught phrases that start with a card's verb: the verb affords both (spec section 7)
TAKE_FOR = _taught(
    "t0000000006",
    ["take", X, "for", Y],
    [
        ["procedure", "?t-proc", "take_for"],
        ["seq", "?t-proc", "1", "?t-s1"],
        ["take", "?t-s1", "?x"],
        ["seq", "?t-proc", "2", "?t-s2"],
        ["give", "?t-s2", "?x", "?y"],
    ],
    "take {X} for {Y}",
)
GIVE_NOW = _taught(
    "t0000000007",
    ["give", X, "to", Y, "now"],
    [["procedure", "?t-proc", "give_now"], ["seq", "?t-proc", "1", "?t-s1"], ["give", "?t-s1", "?x", "?y"]],
    "give {X} to {Y} now",
)
LOOK_OFFICE = _taught(
    "t0000000008",
    ["look at the office"],
    [
        ["procedure", "?t-proc", "standup"],
        ["seq", "?t-proc", "1", "?t-s1"],
        ["const", "?place", "office"],
        ["look", "?t-s1", "?place"],
    ],
    "look at the office",
)
OFFICE_VERBS = ("look", "take", "put", "give", "comment", "make", "archive", "write", "read", "search")


def _frame(name: str, kind: str, construction: str, roles: dict, complete: bool, analysis: int = 0) -> dict:
    return {
        "name": name,
        "kind": kind,
        "construction": construction,
        "roles": roles,
        "complete": complete,
        "analysis": analysis,
        "composed_with": None,
    }


# -- frame parts (no engine) ------------------------------------------------------------------------------------------

#: the small model reads a leading "archive" as a noun, so the card gets no frame part. The release check runs these on trf.
needs_trf = pytest.mark.skipif(
    os.environ.get("MK_EXTRACTION_SPACY_MODEL") == "en_core_web_sm", reason="the small model tags 'archive' as a noun"
)


@needs_trf
def test_frame_parts_one_per_leading_verb():
    parts = frame_parts(OFFICE.cards)
    assert {(p.id, p.unit, p.kind) for p in parts} >= {
        ("frame.make.make", "part", "hand"),
        ("frame.take.take", "part", "hand"),
    }
    assert all(p.form == [p.id.split(".")[2]] and p.meaning[0][0] == "frame" for p in parts)
    # one per (tool, anchor word): look's two forms share one part
    assert sorted(p.id for p in parts) == sorted(f"frame.{verb}.{verb}" for verb in OFFICE_VERBS)
    assert all(p.status == "live" and p.meaning == [["frame", "?frame", p.id.split(".")[1]]] for p in parts)


def test_frame_parts_for_a_taught_phrase():
    [part] = frame_parts([TRIAGE])
    assert part.id == f"frame.{TRIAGE.id}.triage"
    assert (part.kind, part.unit, part.form) == ("taught", "part", ["triage"])
    assert part.meaning == [["frame", "?frame", TRIAGE.id]]
    assert frame_parts([replace(TRIAGE, status="staged")]) == []  # a draft affords nothing until it is confirmed


#: a live taught phrase whose verb is its first word, one that opens on a determiner before its noun, and one with no
#: verb and no noun
SEND = _taught(
    "t0000000010",
    ["send", X, "to", Y],
    [["procedure", "?t-proc", "send"], ["seq", "?t-proc", "1", "?t-s1"], ["give", "?t-s1", "?x", "?y"]],
    "send {X} to {Y}",
)
REPORT = _taught(
    "t0000000011",
    ["the", X, "report"],
    [["procedure", "?t-proc", "report"], ["seq", "?t-proc", "1", "?t-s1"], ["take", "?t-s1", "?x"]],
    "the {X} report",
)
PLEASE = CxnSpec(
    id="please.t0000000012.0",
    kind="taught",
    unit="root",
    form=[X, "please"],
    meaning=[["procedure", "?t-proc", "please"], ["seq", "?t-proc", "1", "?t-s1"], ["take", "?t-s1", "?x"]],
    status="live",
    provenance={"teacher": "Ana", "chat_id": "c1", "draft_id": "t0000000012", "form": "{X} please"},
)


def test_part_anchors_on_the_verb():
    [part] = frame_parts([SEND])
    assert (part.id, part.form) == (f"frame.{SEND.id}.send", ["send"])
    # the verb, not the word the form opens on: "please" never affords the phrase
    polite = replace(SEND, id="please-send.t0000000013.0", form=["please send", X, "to", Y])
    [part] = frame_parts([polite])
    assert (part.id, part.form) == (f"frame.{polite.id}.send", ["send"])


def test_part_falls_back_to_the_noun():
    """A form with no verb is afforded by its first noun, never by a determiner that opens it."""
    [part] = frame_parts([REPORT])
    assert (part.id, part.form) == (f"frame.{REPORT.id}.report", ["report"])


def _patterns(part: CxnSpec) -> list[str]:
    """The sequence pattern of each construction a one-literal part compiles to (its alternates)."""
    from memory_kernel.runtime.store import compiled

    return sorted(json.loads(c.conditional_pole[0][2]["#form"][0][1]) for c in compiled(part))


@needs_trf
def test_a_frame_part_compiles_to_whole_tokens():
    """Round 2: a frame part matches whole tokens only, never inside another word. A word the parser marks as a verb on
    its own reads as that marked token (the marks bound it); a word it leaves plain on its own ("archive" alone is a
    noun) reads as the plain token, a space on each side, and as the marked verb spelled as written ("please archive
    it"); "means" alone is the marked "mean", and the marked "means" as written ("file X means ...")."""
    parts = {p.id: p for p in frame_parts(OFFICE.cards)}
    assert _patterns(parts["frame.take.take"]) == ["⟪take⟫"]
    assert _patterns(parts["frame.archive.archive"]) == [" archive ", "⟪archive⟫"]
    editor = {s.id: s for s in base_specs() if s.id.startswith("frame.editor.")}
    assert _patterns(editor["frame.editor.teach"]) == ["⟪teach⟫"]
    assert _patterns(editor["frame.editor.mean"]) == sorted(["⟪mean⟫", "⟪means⟫"])
    assert _patterns(editor["frame.editor.phrase"]) == [" phrase ", "⟪phrase⟫"]
    # the shared-word rule reads the same alternates the engine matches: a card and a phrase on "take" share it
    assert trigger(parts["frame.take.take"]) == {"⟪take⟫"} and trigger(parts["frame.archive.archive"]) == {
        "archive",
        "⟪archive⟫",
    }


# -- frames on every plan (the engine) --------------------------------------------------------------------------------


class EmptyGraph:
    async def rows(self, cypher: str, params: Any) -> list[dict]:
        return []


def _data_dir():
    from memory_kernel import setup_runtime
    from memory_kernel.runtime.engine import runtime_binary

    data_dir = setup_runtime.resolve_data_dir(None, os.environ)
    if runtime_binary(data_dir) is None:
        pytest.skip(f"no pinned FCG Go under {data_dir} (`mk setup --component fcg-go`, or set MK_DATA_DIR)")
    return data_dir


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from memory_kernel.runtime.engine import RuntimeEngine

    engine = RuntimeEngine(PORT, _data_dir(), out=tmp_path_factory.mktemp("runtime-fcg"))
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


def _runtime(engine, tmp_path: Path, *rows: CxnSpec) -> Runtime:
    return Runtime(
        "bonfire",
        engine,
        EmptyGraph(),
        default_registry(OFFICE.tools),
        store=memory_store(*rows),
        lexicon=Lexicon([], QuestionMap.load()),
        cards=OFFICE.cards,
        record_standing=False,
    )


def _plan(runtime: Runtime, text: str, context: list[dict] | None = None) -> TurnPlan:
    turn = Turn(text=text, speaker="Ana", addressee="hearth", participants=("Ana",), chat_id="c1", message_id=None)
    if context is None:
        return asyncio.run(runtime.plan(turn))
    return asyncio.run(runtime.plan(turn, context=context))


@pytest.mark.needs_fcg
def test_partial_message_reads_a_frame(engine, tmp_path):
    runtime = _runtime(engine, tmp_path)
    plan = _plan(runtime, "make onboard josh")
    assert plan.status == "unread" and plan.program == [] and plan.licence is None
    assert plan.frames == [_frame("make", "tool", "frame.make.make", {"title": None}, False)]
    assert plan.left == ["onboard josh"]
    assert plan.to_json()["frames"] == plan.frames and plan.to_json()["left"] == plan.left
    other = _plan(runtime, "make a ticket try the new tools once")
    assert [f["name"] for f in other.frames] == ["make"] and other.left == ["a ticket try the new tools once"]


@pytest.mark.needs_fcg
def test_a_verb_inside_a_bound_role_affords_its_frame(engine, tmp_path):
    """Live 10-23: search's query slot swallowed "… and write note about it", so the complete card reading hid the
    write. A bound role holding another marked verb is read on its own; its frames follow the card's."""
    runtime = _runtime(engine, tmp_path)
    plan = _plan(runtime, "search about the monkey situation and write note about it")
    assert plan.frames[0]["name"] == "search" and plan.frames[0]["complete"], plan.frames
    assert "write" in [f["name"] for f in plan.frames[1:]], plan.frames
    assert [f["name"] for f in _plan(runtime, "look at the inbox").frames] == ["look"]


@pytest.mark.needs_fcg
def test_two_frames_in_one_message(engine, tmp_path):
    plan = _plan(_runtime(engine, tmp_path), "please take JUJ-3 and put it on Doing")
    assert {f["name"] for f in plan.frames} == {"take", "put"}
    by_name = {f["name"]: f for f in plan.frames}
    assert by_name["take"]["roles"] == {"ticket": None} and by_name["put"]["roles"] == {"ticket": None, "place": None}
    assert not any(f["complete"] for f in plan.frames) and plan.status == "unread"


@pytest.mark.needs_fcg
def test_cards_still_read_their_sentences(engine, tmp_path):
    """Review Focus 2: a frame part never steals a card's sentence, even when the slot text holds another tool's
    verb; the complete reading is the card's, bound, and no frame op reaches the program."""
    runtime = _runtime(engine, tmp_path)
    cases = {
        "make a ticket from try the office": ("card.make.0", {"title": "try the office"}),
        "take JUJ-3": ("card.take.0", {"ticket": "JUJ-3"}),
        "look at the inbox": ("card.look.0", {"place": "inbox"}),
        "give JUJ-3 to Milo": ("card.give.0", {"ticket": "JUJ-3", "person": "Milo"}),
        "make a ticket from take out the trash": ("card.make.0", {"title": "take out the trash"}),
        "write look at the logs": ("card.write.0", {"body": "look at the logs"}),
    }
    for text, (card, roles) in cases.items():
        plan = _plan(runtime, text)
        assert plan.reading["complete"] and plan.reading["root"] == card, (text, plan.reading)
        assert plan.status != "unread" and not any(p[0] == "frame" for p in plan.program), (text, plan.program)
        assert plan.frames[0] == _frame(card.split(".")[1], "tool", card, roles, True), (text, plan.frames)
        assert plan.left == [], (text, plan.left)


@pytest.mark.needs_fcg
def test_competing_analyses_surface(engine, tmp_path):
    runtime = _runtime(engine, tmp_path, WHO)
    plan = _plan(runtime, "who is Zed?")
    found = {(f["kind"], f["construction"]) for f in plan.frames}
    assert {("phrase", WHO.id), ("question", "who-be-x")} <= found, plan.frames
    assert len({f["analysis"] for f in plan.frames}) <= 3
    assert plan.reading["root"] == WHO.id  # the taught phrase loads last and is the primary reading
    phrase = next(f for f in plan.frames if f["kind"] == "phrase")
    question = next(f for f in plan.frames if f["kind"] == "question")
    # each analysis binds its own roles: the broad phrase reads "is Zed" as its X, the memory question "Zed"
    assert phrase == _frame(WHO.id, "phrase", WHO.id, {"x": "is Zed"}, True, 0)
    assert question["name"] == "who-be-x" and question["roles"] == {"x": "Zed"} and question["analysis"] > 0
    # one analysis per construction that read the message: "the office" through np-the and np-bare is one look
    look = _plan(runtime, "look at the office")
    assert look.frames == [_frame("look", "tool", "card.look.0", {"place": "office"}, True)], look.frames


@pytest.mark.needs_fcg
def test_editor_frame(engine, tmp_path):
    runtime = _runtime(engine, tmp_path)
    for text in (
        "teach a phrase: file X means make a ticket from X",
        "when I say standup I mean look at the board",
        # no "teach": "means" alone affords the editor, whether the parser lemmatises it "mean" or keeps "means"
        "file X means make a ticket from X",
        "standup means look at the board",
    ):
        plan = _plan(runtime, text)
        editor = [f for f in plan.frames if f["kind"] == "editor"]
        assert editor and editor[0]["name"] == "teach_phrase" and editor[0]["roles"] == {}, (text, plan.frames)
        assert plan.status == "unread", text


@pytest.mark.needs_fcg
def test_make_a_phrase_affords_the_editor(engine, tmp_path):
    """C7 (b): asking for a phrase affords the editor (``frame.editor.phrase``), so the agent drafts one from its own
    context; nothing runs."""
    runtime = _runtime(engine, tmp_path)
    for text in ("make a phrase for that", "can you turn this into a phrase?"):
        plan = _plan(runtime, text)
        editor = [f for f in plan.frames if f["kind"] == "editor"]
        assert editor and editor[0]["name"] == "teach_phrase" and editor[0]["roles"] == {}, (text, plan.frames)
        assert editor[0]["construction"] == "frame.editor.phrase" and plan.status == "unread", (text, plan.frames)


@pytest.mark.needs_fcg
def test_reply_roots(engine, tmp_path):
    runtime = _runtime(engine, tmp_path)
    for word in ("yes", "no"):
        plan = _plan(runtime, word)
        assert plan.status == "unread" and plan.program == [] and plan.licence is None, (word, plan.diagnostics)
        assert plan.reading["root"] == f"reply.{word}" and plan.reading["complete"]
        assert plan.frames == [_frame(word, "reply", f"reply.{word}", {}, True)]
    assert runtime.window("c1") == []  # a reply draws nothing into the turn window
    # a final mark is still a reply: "yes." and "Yes!" confirm as a bare yes does (data rows, one per mark)
    for text, word, row in (
        ("yes.", "yes", "reply.yes.period"),
        ("Yes!", "yes", "reply.yes.exclaim"),
        ("no.", "no", "reply.no.period"),
        ("No!", "no", "reply.no.exclaim"),
    ):
        plan = _plan(runtime, text)
        assert plan.status == "unread" and plan.program == [] and plan.reading["root"] == row, (text, plan.reading)
        assert plan.frames == [_frame(word, "reply", row, {}, True)], (text, plan.frames)
    plan = _plan(runtime, "yes please take it")
    assert not any(f["kind"] == "reply" for f in plan.frames) and {f["name"] for f in plan.frames} == {"take"}


@pytest.mark.needs_fcg
def test_a_reply_under_any_run_of_marks(engine, tmp_path):
    """K3: the base data has one reply row per single mark; "yes!!" and "Yes..." read through none of them, so they
    are read once more without their final marks, and come back as the bare reply."""
    runtime = _runtime(engine, tmp_path)
    for text, word in (("yes!!", "yes"), ("Yes...", "yes"), ("no!!", "no"), ("No...", "no")):
        plan = _plan(runtime, text)
        assert plan.status == "unread" and plan.program == [] and plan.licence is None, (text, plan.diagnostics)
        assert plan.reading["root"] == f"reply.{word}" and not plan.reading["retried"], (text, plan.reading)
        assert plan.frames == [_frame(word, "reply", f"reply.{word}", {}, True)], (text, plan.frames)
        assert plan.turn["text"] == text


@pytest.mark.needs_fcg
def test_no_catch_all(engine, tmp_path):
    plan = _plan(_runtime(engine, tmp_path), "lovely weather")
    assert plan.status == "unread" and plan.frames == [] and plan.diagnostics == ["no construction read the turn"]
    assert plan.left == ["lovely weather"]


@pytest.mark.needs_fcg
def test_a_frame_part_never_reads_inside_a_word(engine, tmp_path):
    """Round 2: a frame part reads whole tokens only. "archives" affords nothing (it holds "archive" inside it);
    "please archive it" reads the verb once, and ``left`` holds every other word once."""
    runtime = _runtime(engine, tmp_path)
    plan = _plan(runtime, "the archives are full")
    assert plan.frames == [] and plan.status == "unread", plan.frames
    assert plan.left == ["the archives are full"], plan.left
    archive = _plan(runtime, "please archive it")
    assert [f["name"] for f in archive.frames] == ["archive"], archive.frames
    assert archive.left == ["please", "it"], archive.left
    both = _plan(runtime, "can you take it and archive it")
    assert {f["name"] for f in both.frames} == {"take", "archive"}, both.frames
    assert both.left == ["can you", "it and", "it"], both.left


@pytest.mark.needs_fcg
def test_phrase_with_no_verb_or_noun_has_no_part(engine, tmp_path):
    """A phrase whose form holds no verb and no noun ("{X} please") gets no frame part: it is reached by its full form
    only, and "please" alone affords nothing."""
    assert frame_parts([PLEASE]) == []
    runtime = _runtime(engine, tmp_path, PLEASE)
    plan = _plan(runtime, "JUJ-3 please")
    assert plan.reading["root"] == PLEASE.id and plan.reading["complete"], plan.reading
    assert plan.frames == [_frame(PLEASE.id, "phrase", PLEASE.id, {"x": "JUJ-3"}, True)], plan.frames
    assert _plan(runtime, "please").frames == []


# -- a stated affordance (2.1, spec part 3) ---------------------------------------------------------------------------


def test_a_stated_affordance_is_one_part_per_word():
    """``affordance`` states the words that raise a root's frame: one part per word, with the meaning and the name a
    derived part has, and none of the derived one; ``[]`` is no part at all; None (unstated) is derived as before."""
    parts = frame_parts([replace(TRIAGE, affordance=["fees", "Costs"])])
    assert [(p.id, p.kind, p.unit, p.form) for p in parts] == [
        (f"frame.{TRIAGE.id}.fees", "taught", "part", ["fees"]),
        (f"frame.{TRIAGE.id}.costs", "taught", "part", ["costs"]),
    ]
    assert all(p.meaning == [["frame", "?frame", TRIAGE.id]] and p.status == "live" for p in parts)
    assert frame_parts([replace(TRIAGE, affordance=[])]) == []
    assert [p.id for p in frame_parts([TRIAGE])] == [f"frame.{TRIAGE.id}.triage"]
    # a card's stated words name its tool's frame, as its derived part does
    look = next(c for c in OFFICE.cards if c.id == "card.look.0")
    assert [(p.id, p.meaning) for p in frame_parts([replace(look, affordance=["inbox"])])] == [
        ("frame.look.inbox", [["frame", "?frame", "look"]])
    ]
    # a draft affords nothing until it is confirmed, whatever it states
    assert frame_parts([replace(TRIAGE, affordance=["fees"], status="staged")]) == []


def test_an_unstated_affordance_is_stored_as_before():
    """A row with no stated affordance has no ``affordance`` key: it is stored byte for byte as before. A stated one
    (``[]`` included) is kept on the row and read back."""
    assert "affordance" not in TRIAGE.to_json() and TRIAGE.affordance is None
    assert CxnSpec.from_json(TRIAGE.to_json()) == TRIAGE
    for words in ([], ["fees", "costs"]):
        row = replace(TRIAGE, affordance=words).to_json()
        assert row["affordance"] == words and CxnSpec.from_json(row).affordance == words


@pytest.mark.parametrize(
    "spec",
    [
        {"unit": "filler", "fills": "np", "form": ["that crew"], "meaning": [["name-span", "?x", "?l", "?r"]]},
        {"unit": "part", "form": ["crew"], "meaning": [["frame", "?frame", "crew"]]},
        {"unit": "root", "form": ["crew"], "meaning": [["const", "?x", "y"]], "affordance": "crew"},
        {"unit": "root", "form": ["crew"], "meaning": [["const", "?x", "y"]], "affordance": ["crew", 3]},
    ],
)
def test_only_a_root_states_an_affordance_as_a_list_of_words(spec):
    from memory_kernel.runtime.constructions import CompileError

    row = {"id": "crew.t1", "kind": "taught", "status": "live", "affordance": ["crew"], **spec}
    refused = "only a root states an affordance|affordance must be a list of words"
    with pytest.raises(CompileError, match=refused):
        CxnSpec.from_json(row)
    with pytest.raises(CompileError, match=refused):  # the same check on a row built in code
        CxnSpec(**_fields(row), affordance=row["affordance"])  # pyright: ignore[reportArgumentType]


def _fields(row: dict[str, Any]) -> dict[str, Any]:
    """A row's JSON as ``CxnSpec`` keyword arguments (no affordance)."""
    spec = CxnSpec.from_json({k: v for k, v in row.items() if k != "affordance"})
    return {f: getattr(spec, f) for f in ("id", "kind", "form", "meaning", "status", "unit", "fills")}


@pytest.mark.parametrize(
    ("words", "problem"),
    [
        (["fees", "late costs"], "affordance word 'late costs' holds a space: each word is an entry of its own"),
        ([""], "affordance holds an empty word"),
        (["fees", "Fees"], "affordance words 'fees' and 'Fees' are one word"),
        (["don't"], 'affordance word "don\'t" is not one word: the normaliser reads it as 2 tokens'),
        (["--"], "affordance word '--' holds no letter or digit from a to z or 0 to 9 (a frame part is named by it)"),
        ("fees", "affordance is a list of words, or null to derive it from the form; got 'fees'"),
    ],
)
def test_an_affordance_word_is_one_token_the_compiler_takes(words, problem):
    from memory_kernel.runtime.frames import affordance_problems

    found = affordance_problems(words)
    assert len(found) == 1 and found[0].startswith(problem), found
    assert affordance_problems(["fees", "costs"]) == [] and affordance_problems([]) == []


@pytest.mark.needs_fcg
def test_a_stated_affordance_raises_the_frame_on_its_words(engine, tmp_path):
    """The stated words raise the phrase's frame wherever they occur ("what about costs", "fees again"), its roles
    open, and its verb no longer does; ``[]`` raises nothing on the verb; an unstated affordance raises the verb's frame
    as before. The phrase still reads its own sentences completely."""
    roles = {"x": None, "r": None}
    stated = _runtime(engine, tmp_path, replace(TRIAGE, affordance=["fees", "costs"]))
    for text, word in (("what about costs", "costs"), ("fees again", "fees")):
        plan = _plan(stated, text)
        assert plan.frames == [_frame(TRIAGE.id, "phrase", f"frame.{TRIAGE.id}.{word}", roles, False)], plan.frames
        assert plan.status == "unread"
    assert all(f["name"] != TRIAGE.id for f in _plan(stated, "triage it").frames)
    whole = _plan(stated, "triage JUJ-3 for Milo")
    assert whole.reading["root"] == TRIAGE.id and whole.reading["complete"], whole.reading
    none = _runtime(engine, tmp_path, replace(TRIAGE, affordance=[]))
    assert all(f["name"] != TRIAGE.id for f in _plan(none, "triage it").frames)
    derived = _plan(_runtime(engine, tmp_path, TRIAGE), "triage it").frames
    assert _frame(TRIAGE.id, "phrase", f"frame.{TRIAGE.id}.triage", roles, False) in derived, derived


#: every DERIVED frame part (the base rows' parts, the office cards', TRIAGE's) as a12d26f4's own code compiled them:
#: (how many, sha256 of the sorted (part id, its constructions' name, poles and attributes as JSON) pairs)
DERIVED_ON_A12D26F4 = (14, "d90004e6b15419520042836b34f133c421cd4706d17c883c0758bbff83f341d5")


@needs_trf
def test_a_derived_part_compiles_exactly_as_before():
    """Only a STATED word's part reads its word as written too: every part the kernel derives (the base set's, a card's,
    an unstated phrase's) compiles to the constructions it compiled to before stated affordances existed."""
    import hashlib

    from memory_kernel.runtime.store import compiled

    parts = [s for s in base_specs() if s.unit == "part"] + frame_parts([*OFFICE.cards, TRIAGE])
    assert not any(p.as_written for p in parts)
    rows = sorted(
        (
            p.id,
            json.dumps(
                [[c.name, c.conditional_pole, c.contributing_pole, c.attributes] for c in compiled(p)],
                sort_keys=True,
                ensure_ascii=False,
            ),
        )
        for p in parts
    )
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
    assert (len(rows), digest) == DERIVED_ON_A12D26F4


@needs_trf
def test_a_stated_word_read_as_a_verb_alone_also_reads_as_written():
    """A stated word the parser marks as a verb when it stands alone ("budget") reads as the marked verb and as the plain
    word as written ("the budget"), as a one-word filler does; a stated word read plain alone ("fees") reads as it did,
    plain and as the marked verb. Both uses share the word with any part that reads either."""
    [budget] = frame_parts([replace(TRIAGE, affordance=["budget"])])
    assert budget.as_written and _patterns(budget) == [" budget ", "⟪budget⟫"]
    assert trigger(budget) == {"budget", "⟪budget⟫"}
    [fees] = frame_parts([replace(TRIAGE, affordance=["fees"])])
    assert _patterns(fees) == [" fees ", "⟪fees⟫"]
    # the field is a part's alone, and absent from the JSON of every part that does not set it
    assert "as_written" not in frame_parts([TRIAGE])[0].to_json() and budget.to_json()["as_written"] is True
    from memory_kernel.runtime.constructions import CompileError

    with pytest.raises(CompileError, match="as_written"):
        replace(TRIAGE, as_written=True)


@pytest.mark.needs_fcg
def test_a_stated_verb_raises_the_frame_on_both_of_its_uses(engine, tmp_path):
    """``affordance=["budget"]``: the phrase's frame on the noun ("what about the budget") and on the verb ("please
    budget it")."""
    runtime = _runtime(engine, tmp_path, replace(TRIAGE, affordance=["budget"]))
    part = f"frame.{TRIAGE.id}.budget"
    for text in ("what about the budget", "please budget it"):
        frames = _plan(runtime, text).frames
        assert _frame(TRIAGE.id, "phrase", part, {"x": None, "r": None}, False) in frames, (text, frames)


#: ten clauses, each with an office verb
CLAUSES = (
    "take out the trash",
    "put it on Doing",
    "give it to Milo",
    "look at the inbox",
    "write the notes",
    "search the logs",
    "comment on it",
    "read the notepad",
    "make coffee",
    "archive the old one",
)


def _best_ms(read: Any, times: int = 3) -> tuple[float, Any]:
    """The fastest of ``times`` wall times of ``read()`` (ms) and its last result: a timing under machine load."""
    best, out = None, None
    for _ in range(times):
        started = time.perf_counter()
        out = read()
        ms = (time.perf_counter() - started) * 1000
        best = ms if best is None else min(best, ms)
    return float(best or 0.0), out


@pytest.mark.needs_fcg
def test_competing_analyses_stay_within_the_budget(engine, tmp_path, monkeypatch):
    """a read with competing analyses costs at most ~50 ms more than a single one, also when the card's slot
    text holds verbs: each is a frame part, and no complete analysis holds a part, so the n-best search never walks
    their branches (eight verbs in the title used to cost a second). The read is ``Runtime._read``, as a plan reads:
    the primary reading, then ``comprehend_all`` for the competing analyses, whose whole time the plan pays."""
    from memory_kernel.runtime.normalise import normalise

    runtime = _runtime(engine, tmp_path, WHO, TAKE_FOR)
    _plan(runtime, "look at the inbox")  # started and loaded
    text = "make a ticket from " + " and ".join(CLAUSES[:8])
    both, read = _best_ms(lambda: runtime._read(text))
    assert read.reading.complete and read.alternatives is None and len(read.readings) == 1, read
    calls: list[str] = []
    monkeypatch.setattr(engine, "comprehend_all", lambda text, n=3: calls.append(text) or [])
    single, alone = _best_ms(lambda: runtime._read(text))
    assert alone.reading.complete and calls, alone  # the same read, the n-best search left out
    assert both - single <= 50, (single, both)
    monkeypatch.undo()
    # the competition still surfaces: the broad phrase and the base question are both found
    who = normalise("who is Zed?").text
    assert len(engine.comprehend_all(who, 3)) == 2


@pytest.mark.needs_fcg
def test_a_long_partial_message_reads_in_bounded_time(engine, tmp_path):
    """a long message no construction completes reads every verb's frame without walking every order and
    subset of its frame parts (ten clauses used to cost 2.5 s a plan)."""
    runtime = _runtime(engine, tmp_path, TAKE_FOR)
    _plan(runtime, "look at the inbox")  # started and loaded
    ms, plan = _best_ms(lambda: _plan(runtime, "please " + " and ".join(CLAUSES) + " and thanks"), 2)
    assert plan.status == "unread" and {f["name"] for f in plan.frames} >= set(OFFICE_VERBS), plan.frames
    # the read itself (it was 1.1 s), and the whole plan: two parses and two reads, the retry mark's included
    assert plan.reading["ms"] < 300, plan.reading
    assert ms < 1000, ms


@pytest.mark.needs_fcg
def test_a_shared_verb_affords_every_frame(engine, tmp_path):
    """Spec section 7: a word that leads a card and a live taught phrase affords both. Only one frame part can consume
    the word, so the frame of every part sharing its trigger is reported."""
    runtime = _runtime(engine, tmp_path, TAKE_FOR, LOOK_OFFICE)
    for text, card, phrase in (
        ("please take the new one", "take", TAKE_FOR.id),
        ("can you look at JUJ-3 please", "look", LOOK_OFFICE.id),
    ):
        plan = _plan(runtime, text)
        assert plan.status == "unread", (text, plan.reading)
        found = {(f["name"], f["kind"]) for f in plan.frames}
        assert {(card, "tool"), (phrase, "phrase")} <= found, (text, plan.frames)
        assert not any(f["complete"] for f in plan.frames), (text, plan.frames)
    take = _plan(runtime, "please take the new one")
    by_name = {f["name"]: f for f in take.frames}
    assert by_name["take"]["roles"] == {"ticket": None} and by_name[TAKE_FOR.id]["roles"] == {"x": None, "y": None}


@pytest.mark.needs_fcg
def test_competition_survives_determiners(engine, tmp_path):
    """C4: the n-best counts distinct roots, not filler paths. A broad taught phrase with "the ..." in both slots has
    four filler paths (np-the, np-bare per slot); they must not crowd the competing card out of the analyses."""
    runtime = _runtime(engine, tmp_path, TAKE_FOR, GIVE_NOW)
    for text, phrase, card in (
        ("take the report for the team", TAKE_FOR.id, "card.take.0"),
        ("give the ticket to the boss now", GIVE_NOW.id, "card.give.0"),
    ):
        plan = _plan(runtime, text)
        assert plan.reading["root"] == phrase and plan.reading["complete"], (text, plan.reading)
        found = {(f["kind"], f["construction"]) for f in plan.frames}
        assert {("phrase", phrase), ("tool", card)} <= found, (text, plan.frames)
        assert len({f["analysis"] for f in plan.frames}) == len(plan.frames), plan.frames  # one frame per root


@pytest.mark.needs_fcg
def test_alternatives_failing_keeps_the_primary(engine, tmp_path, monkeypatch):
    """The n-best read never costs a complete reading its plan: when it fails, the primary reading alone stands."""
    from memory_kernel.runtime.engine import RuntimeEngineError

    def refuse(text: str, n: int = 3):
        raise RuntimeEngineError("runtime engine comprehend-all: refused")

    runtime = _runtime(engine, tmp_path)
    _plan(runtime, "look at the inbox")  # started and loaded before the n-best read is broken
    monkeypatch.setattr(engine, "comprehend_all", refuse)
    plan = _plan(runtime, "take JUJ-3")
    assert plan.status != "unread" and plan.reading["root"] == "card.take.0", (plan.status, plan.diagnostics)
    assert any(op[0] == "take" for op in plan.program), plan.program
    assert plan.frames == [_frame("take", "tool", "card.take.0", {"ticket": "JUJ-3"}, True)]
    assert "refused" in plan.reading["alternatives"]


# -- follow-ups compose with the open frame they answer (C11) ---------------------------------------------------------


def _open(name: str, kind: str, roles: dict, construction: str | None = None) -> dict:
    """A frame of the turn before, as kernel-turns sends it back in the plan's context."""
    complete = all(value is not None for value in roles.values())
    return _frame(name, kind, construction or f"frame.{name}.{name.split('.')[0]}", roles, complete)


def _composed(name: str, kind: str, construction: str, roles: dict, with_: str) -> dict:
    return _frame(name, kind, construction, roles, True) | {"composed_with": with_}


def test_render_fills_each_slot_and_needs_every_one():
    from memory_kernel.runtime.frames import render

    cards = {c.id: c for c in OFFICE.cards}
    assert render(cards["card.put.0"], {"ticket": "JUJ-3", "place": "Doing"}) == "put JUJ-3 on Doing"
    assert render(cards["card.make.0"], {"title": "onboard josh"}) == "make a ticket from onboard josh"
    assert render(cards["card.read.0"], {}) == "read the notepad"
    assert render(WHO, {"x": "Zed"}) == "who Zed ?"
    assert render(cards["card.put.0"], {"ticket": "JUJ-3"}) is None  # a slot with no value: nothing to read
    assert render(cards["card.put.0"], {"ticket": "JUJ-3", "place": "  "}) is None


@pytest.mark.needs_fcg
def test_bare_answer_fills_the_open_role(engine, tmp_path):
    runtime = _runtime(engine, tmp_path)
    plan = _plan(runtime, "JUJ-3", [_open("take", "tool", {"ticket": None})])
    assert plan.frames[0] == _composed("take", "tool", "card.take.0", {"ticket": "JUJ-3"}, "take"), plan.frames
    assert plan.turn["text"] == "JUJ-3" and plan.reading["composed"] == {"with": "take", "text": "take JUJ-3"}
    # the composed reading runs as the card's would: its spans are the rendered text's, the args the fragment
    assert plan.status == "needs_tools" and [(s["tool"], s["args"]) for s in plan.tool_steps] == [
        ("take", {"ticket": "JUJ-3"})
    ]
    assert plan.left == []
    # only a frame of the open frame's name is composed with it: any other analysis of the rendering is not
    assert all((f["composed_with"] == "take") == (f["name"] == "take") for f in plan.frames), plan.frames


@pytest.mark.needs_fcg
def test_a_composed_plan_reports_every_read_it_cost(engine, tmp_path, monkeypatch):
    """reading["ms"] (and the trace's comprehension_ms) of a composed plan add up the turn's own read and every
    composition read, not only the rendering that composed."""
    runtime = _runtime(engine, tmp_path)
    read, spent = runtime._read, []

    def counted(text: str, **kw: Any):
        out = read(text, **kw)
        spent.append(out.ms)
        return out

    monkeypatch.setattr(runtime, "_read", counted)
    plan = _plan(runtime, "JUJ-3", [_open("take", "tool", {"ticket": None})])
    assert plan.reading["composed"]["with"] == "take" and len(spent) == 2, (plan.reading, spent)
    assert plan.reading["ms"] == pytest.approx(sum(spent)) and plan.trace["comprehension_ms"] == plan.reading["ms"]


def test_composed_with_names_only_the_open_frames_own():
    from memory_kernel.runtime.api import _composed_frames

    take = _frame("take", "tool", "card.take.0", {"ticket": "JUJ-3"}, True)
    other = _frame("who", "question", "who-be-x", {"x": "JUJ-3"}, True, 1)
    assert _composed_frames([take, other], "take") == [take | {"composed_with": "take"}, other]


@pytest.mark.needs_fcg
def test_answer_needs_the_other_roles(engine, tmp_path):
    runtime = _runtime(engine, tmp_path)
    plan = _plan(runtime, "Doing", [_open("put", "tool", {"ticket": "JUJ-3", "place": None})])
    assert plan.frames[0] == _composed("put", "tool", "card.put.0", {"ticket": "JUJ-3", "place": "Doing"}, "put")
    assert [s["args"] for s in plan.tool_steps] == [{"ticket": "JUJ-3", "place": "Doing"}]
    alone = _plan(runtime, "Doing", [_open("put", "tool", {"ticket": None, "place": None})])
    assert alone.status == "unread" and alone.frames == [] and "composed" not in alone.reading, alone.frames
    # the answer's final mark is the turn's, not the slot's: it never lands mid-sentence in the rendering
    marked = _plan(runtime, "JUJ-3.", [_open("give", "tool", {"ticket": None, "person": "Milo"})])
    assert marked.reading["composed"] == {"with": "give", "text": "give JUJ-3 to Milo"}, marked.reading
    assert marked.frames[0]["roles"] == {"ticket": "JUJ-3", "person": "Milo"}, marked.frames
    assert [s["args"] for s in marked.tool_steps] == [{"ticket": "JUJ-3", "person": "Milo"}]


def test_a_fragment_drops_only_the_marks_that_end_its_sentence():
    """A closing quote or bracket (morph PunctType Quot, Brck) is half of a pair and stays with its opener; a turn of
    nothing but marks is no fragment at all."""
    from memory_kernel.runtime.api import _without_final_mark
    from memory_kernel.runtime.normalise import normalise

    for text, fragment in (
        ("JUJ-3.", "JUJ-3"),
        ("JUJ-3?!", "JUJ-3"),
        ('"JUJ-3"', '"JUJ-3"'),
        ('"JUJ-3".', '"JUJ-3"'),
        ("(JUJ-3)?", "(JUJ-3)"),
        ("“onboard josh”", "“onboard josh”"),
        ("?", ""),
        ("...", ""),
        ('""', ""),
    ):
        assert _without_final_mark(normalise(text)) == fragment, text


@pytest.mark.needs_fcg
def test_a_quoted_answer_keeps_its_quotes(engine, tmp_path):
    runtime = _runtime(engine, tmp_path)
    for text, ticket in (('"JUJ-3"', '"JUJ-3"'), ('"JUJ-3".', '"JUJ-3"'), ("(JUJ-3)", "(JUJ-3)")):
        plan = _plan(runtime, text, [_open("take", "tool", {"ticket": None})])
        assert plan.reading["composed"] == {"with": "take", "text": f"take {ticket}"}, (text, plan.reading)
        assert plan.frames[0]["roles"] == {"ticket": ticket}, (text, plan.frames)
    give = _plan(runtime, '"JUJ-3"', [_open("give", "tool", {"ticket": None, "person": "Milo"})])
    assert give.reading["composed"] == {"with": "give", "text": 'give "JUJ-3" to Milo'}, give.reading
    assert give.frames[0]["roles"] == {"ticket": '"JUJ-3"', "person": "Milo"}, give.frames


@pytest.mark.needs_fcg
def test_marks_alone_never_compose(engine, tmp_path):
    """A turn of nothing but marks stays an unread turn: "?" in answer to "which ticket?" is no ticket."""
    runtime = _runtime(engine, tmp_path)
    for text, frame in (
        ("?", _open("give", "tool", {"ticket": None, "person": "Milo"})),
        (".", _open("take", "tool", {"ticket": None})),
        ("...", _open("take", "tool", {"ticket": None})),
    ):
        plan = _plan(runtime, text, [frame])
        assert plan.status == "unread" and "composed" not in plan.reading, (text, plan.reading)
        assert plan.frames == [], (text, plan.frames)


@pytest.mark.needs_fcg
def test_a_turn_refused_for_its_mark_never_composes(engine, tmp_path):
    """K3: "take JUJ-3?" reads through the take card with the ticket "JUJ-3?": a slot holding the turn's final mark,
    so that reading is refused (a "?" is never stripped, and the turn is unread). It was still read by a root of its
    own, so it is never put into an open frame's slot (C11): never give(ticket="take JUJ-3")."""
    runtime = _runtime(engine, tmp_path)
    alone = _plan(runtime, "take JUJ-3?")
    assert alone.status == "unread" and alone.tool_steps == [] and alone.frames == [], (alone.status, alone.frames)
    plan = _plan(runtime, "take JUJ-3?", [_open("give", "tool", {"ticket": None, "person": "Milo"})])
    assert plan.status == "unread" and "composed" not in plan.reading, plan.reading
    assert plan.tool_steps == [] and plan.frames == [], (plan.tool_steps, plan.frames)


#: the small model does not split a full-width or emoji question mark from the word before it, so the turn has no
#: final mark to find. The release check runs this on trf.
@needs_trf
def test_a_question_mark_in_any_script_is_a_question():
    """A question mark is any punctuation or symbol character whose Unicode name says QUESTION ... MARK or INTERROBANG:
    the full-width one, the emoji, the inverted, Arabic and small ones too. A math relation with a question mark above
    it, a semicolon, an ellipsis or "!" is not one."""
    from memory_kernel.runtime.api import _asks, _final_marks, _question_mark
    from memory_kernel.runtime.normalise import normalise

    for ch in "?¿;؟‽⁇⁈⁉❓❔⸘⸮﹖？":
        assert _question_mark(ch), f"U+{ord(ch):04X}"
    for ch in ".!;:…¡⩻≟\U000e003f":
        assert not _question_mark(ch), f"U+{ord(ch):04X}"
    for text, asks in (
        ("take JUJ-3?", True),
        ("take JUJ-3？", True),  # FULLWIDTH QUESTION MARK
        ("take JUJ-3❓", True),  # BLACK QUESTION MARK ORNAMENT, the emoji
        ("take JUJ-3?!", True),
        ("take JUJ-3!?", True),
        ("take JUJ-3.", False),
        ("take JUJ-3!!", False),
        ("take JUJ-3...", False),
    ):
        norm = normalise(text)
        marks = _final_marks(norm)
        assert marks and _asks(norm, marks) is asks, (text, marks)


@pytest.mark.needs_fcg
def test_a_question_mark_in_any_script_is_never_stripped(engine, tmp_path):
    """A turn whose final marks hold a question mark, the full-width one, the emoji or one in a mixed run ("?!", "!?"),
    is never read without them: the take card's slot would hold the mark, so that reading is refused, and the turn is
    unread. A question never runs a command."""
    runtime = _runtime(engine, tmp_path)
    for text in ("take JUJ-3？", "take JUJ-3❓", "take JUJ-3?!", "take JUJ-3!?"):
        plan = _plan(runtime, text)
        assert plan.status == "unread" and plan.tool_steps == [] and plan.program == [], (text, plan.tool_steps)
        assert plan.frames == [] and not plan.reading["retried"], (text, plan.frames)


@pytest.mark.needs_fcg
def test_a_mark_inside_closing_quotes_is_the_quoted_texts(engine, tmp_path):
    """A closing quote is no final mark: a mark inside it belongs to the quoted text, and the slot keeps it (a quoted
    question reads, too); a mark after it is the sentence's, and the slot never takes it."""
    runtime = _runtime(engine, tmp_path)
    for text, roles in (
        ('take "JUJ-3."', {"ticket": '"JUJ-3."'}),
        ('take "JUJ-3?"', {"ticket": '"JUJ-3?"'}),
        ('take "JUJ-3".', {"ticket": '"JUJ-3"'}),
        ('write "did you see the release?"', {"body": '"did you see the release?"'}),
    ):
        plan = _plan(runtime, text)
        assert plan.status == "needs_tools", (text, plan.status, plan.diagnostics)
        assert [s["args"] for s in plan.tool_steps] == [roles] and plan.frames[0]["roles"] == roles, (text, plan.frames)


@pytest.mark.needs_fcg
def test_a_bare_verb_is_its_open_frame_never_a_question(engine, tmp_path):
    """K3b in its commonest form: "take" alone drew no root, so it was read again as "take ?", and the take card's slot
    took the "?" (take(ticket="?")). A retry whose "?" only feeds a slot is dropped: the turn is unread with the verb's
    frame part open, runs nothing, and (a frame of its own) is never composed into an open frame."""
    runtime = _runtime(engine, tmp_path)
    for text, role in (("take", "ticket"), ("write", "body")):
        plan = _plan(runtime, text)
        assert plan.status == "unread" and plan.tool_steps == [] and not plan.reading["retried"], (text, plan.reading)
        assert plan.frames == [_frame(text, "tool", f"frame.{text}.{text}", {role: None}, False)], (text, plan.frames)
    given = _plan(runtime, "take", [_open("give", "tool", {"ticket": None, "person": "Milo"})])
    assert given.status == "unread" and "composed" not in given.reading and given.tool_steps == [], given.reading


@pytest.mark.needs_fcg
def test_a_complete_message_ignores_context(engine, tmp_path):
    runtime = _runtime(engine, tmp_path)
    plan = _plan(runtime, "look at the inbox", [_open("take", "tool", {"ticket": None})])
    assert plan.reading["root"] == "card.look.0" and "composed" not in plan.reading
    assert plan.frames == [_frame("look", "tool", "card.look.0", {"place": "inbox"}, True)], plan.frames


@pytest.mark.needs_fcg
def test_committed_frames_are_not_context(engine, tmp_path):
    """Review Focus 3 at the kernel's edge: kernel-turns sends only the uncommitted frames, so after "take JUJ-3" ran,
    "JUJ-4" comes with an empty context and reads on its own: nothing."""
    runtime = _runtime(engine, tmp_path)
    plan = _plan(runtime, "JUJ-4", [])
    assert plan.status == "unread" and plan.frames == [] and plan.left == ["JUJ-4"]


def _kernel_turns_context(plan: TurnPlan) -> list[dict]:
    """What kernel-turns sends back as the next turn's context when nothing ran for ``plan``'s frames (its
    ``affordances.open_frames``: the tool and phrase frames, the first of each name and roles, at most 8)."""
    out, seen = [], set()
    for frame in plan.frames:
        ident = (frame["name"], tuple(sorted(frame["roles"].items())))
        if frame["kind"] in ("tool", "phrase") and ident not in seen:
            seen.add(ident)
            out.append(dict(frame))
    return out[:8]


@pytest.mark.needs_fcg
def test_the_answer_to_which_ticket_composes_with_the_frame_asked_about(engine, tmp_path):
    """Spec §5 item 5, the real two turns: "take the new one" reads COMPLETELY through the card (a ticket can be named
    by its title), the agent asks "which ticket?" instead of calling, nothing runs, so kernel-turns sends the bound
    take frame back as context; "JUJ-3" then composes with it as take(ticket=JUJ-3)."""
    runtime = _runtime(engine, tmp_path)
    first = _plan(runtime, "take the new one")
    assert first.frames[0] == _frame("take", "tool", "card.take.0", {"ticket": "new one"}, True), first.frames
    answer = _plan(runtime, "JUJ-3", _kernel_turns_context(first))
    assert answer.frames[0] == _composed("take", "tool", "card.take.0", {"ticket": "JUJ-3"}, "take"), answer.frames
    assert answer.reading["composed"] == {"with": "take", "text": "take JUJ-3"}
    # an open role is still the answer's first place: the bound frame is read again only after every open one
    put = _open("put", "tool", {"ticket": "JUJ-3", "place": None})
    both = _plan(runtime, "Doing", [*_kernel_turns_context(first), put])
    assert both.reading["composed"] == {"with": "put", "text": "put JUJ-3 on Doing"}, both.reading
    # which of two bound roles an answer replaces is the agent's to say, not the kernel's: no composition
    bound = _plan(
        runtime, "Doing", [_frame("put", "tool", "card.put.0", {"ticket": "JUJ-3", "place": "new lane"}, True)]
    )
    assert bound.status == "unread" and bound.frames == [] and "composed" not in bound.reading, bound.reading


@pytest.mark.needs_fcg
def test_a_phrase_frame_composes_and_a_message_with_a_verb_reads_on_its_own(engine, tmp_path):
    runtime = _runtime(engine, tmp_path, TRIAGE)
    plan = _plan(runtime, "JUJ-3", [_open(TRIAGE.id, "phrase", {"x": None, "r": "Milo"})])
    assert plan.frames[0] == _composed(TRIAGE.id, "phrase", TRIAGE.id, {"x": "JUJ-3", "r": "Milo"}, TRIAGE.id)
    assert plan.reading["composed"]["text"] == "triage JUJ-3 for Milo"
    # a partial message with a frame of its own is not a bare fragment: it never goes into the open frame's slot
    # (C11), it reads on its own as a partial make frame
    own = _plan(runtime, "make onboard josh", [_open("take", "tool", {"ticket": None})])
    assert "composed" not in own.reading and own.status == "unread", own.reading
    assert own.frames == [_frame("make", "tool", "frame.make.make", {"title": None}, False)], own.frames
    assert own.left == ["onboard josh"]
    answer = _plan(runtime, "yes please take it", [_open("give", "tool", {"ticket": None, "person": "Milo"})])
    assert "composed" not in answer.reading and answer.status == "unread", answer.reading
    assert all(f["composed_with"] is None for f in answer.frames) and answer.frames, answer.frames
