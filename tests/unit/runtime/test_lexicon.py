"""The question map (generated from PropBank function tags) and the lexicon ops that read it."""

import asyncio
import importlib.util
import json
import os
from pathlib import Path

import pytest

from memory_kernel.runtime.lexicon import (
    LEXICON_SPECS,
    CompositeMemory,
    Lexicon,
    LexiconOps,
    QuestionMap,
)

REPO = Path(__file__).parents[3]


def _script():
    path = REPO / "scripts" / "build_question_map.py"
    spec = importlib.util.spec_from_file_location("_build_question_map", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _role(n: str, f: str | None, descr: str, theme: str | None = None) -> str:
    tag = f' f="{f}"' if f is not None else ""
    link = (
        f'<rolelinks><rolelink class="x-1" resource="VerbNet" version="verbnet3.4">{theme}</rolelink></rolelinks>'
        if theme
        else ""
    )
    return f'<role descr="{descr}"{tag} n="{n}">{link}</role>'


def _frames(tmp_path: Path) -> Path:
    """Two invented rolesets in two frame files."""
    frames = tmp_path / "frames"
    frames.mkdir()
    zorvexate = "".join(
        (
            _role("0", "PAG", "zorvexer", "agent"),
            _role("1", "LOC", "place zorvexed"),
            _role("2", "PRD", "manner of zorvexing"),  # a tag outside the table, no VerbNet theme: unsettled
            _role("3", "", "untagged core role"),
            _role("M", "TMP", "time of zorvexing"),
            _role("M", "", "untagged modifier"),
        )
    )
    brask = "".join(
        (
            _role("0", "PPT", "thing brasked"),
            _role("1", "GOL", "brasked to"),
            _role("2", "PRD", "brasked into", "destination"),  # an unsettled tag settled by its VerbNet theme
            _role("3", "MNR", "means"),
            _role("4", "CAU", "cause"),
            _role("5", "EXT", "amount"),
        )
    )
    for lemma, roles in (("zorvexate", zorvexate), ("brask", brask)):
        (frames / f"{lemma}.xml").write_text(
            f'<?xml version="1.0"?><frameset><predicate lemma="{lemma}">'
            f'<roleset id="{lemma}.01" name="{lemma}"><roles>{roles}</roles></roleset>'
            "</predicate></frameset>",
            encoding="utf-8",
        )
    return frames


def test_question_map_from_tags(tmp_path):
    built = _script().build_question_map([_frames(tmp_path)])
    assert list(built) == sorted(built)
    assert built["*"] == [["when", "argm-tmp"], ["where", "argm-loc", ["Place"]]]  # its store kind (Task 11a)
    assert built["zorvexate.01"] == [
        ["who_or_what", "arg0"],
        ["where", "arg1"],
        ["unknown", "arg2"],
        ["who_or_what", "arg3"],
        ["unknown", "argm"],
        ["when", "argm-tmp"],
    ]
    assert built["brask.01"] == [
        ["who_or_what", "arg0"],
        ["where", "arg1"],
        ["where", "arg2", ["Place"]],  # a destination: its store kind (Task 11a)
        ["how", "arg3"],
        ["why", "arg4"],
        ["how_much", "arg5"],
    ]
    out = tmp_path / "question-map.json"
    _script().write_question_map(built, out)
    assert QuestionMap.load(out).roles("brask.01", "where") == ["arg1", "arg2", "argm-loc"]
    assert json.loads(out.read_text(encoding="utf-8")) == built


def test_a_recipient_goal_answers_who_or_what():
    """A GOL role with VerbNet theme recipient is who_or_what; every other GOL/DIR/SRC role stays where."""
    role_kind = _script().role_kind
    assert role_kind("arg2", "GOL", ("recipient",)) == "who_or_what"  # give.01 / tell.01 arg2
    assert role_kind("arg2", "GOL", ("beneficiary",)) == "where"  # work.01 arg2, the employer
    assert role_kind("arg2", "DIR", ("source",)) == "where"  # buy.01 arg2, the seller
    assert role_kind("arg2", "GOL", ("destination",)) == "where" and role_kind("arg2", "GOL", ()) == "where"
    shipped = QuestionMap.load()
    assert ("who_or_what", "arg2") in shipped.pairs("give.01") and ("who_or_what", "arg2") in shipped.pairs("tell.01")
    assert ("where", "arg2") in shipped.pairs("work.01") and ("where", "arg2") in shipped.pairs("buy.01")


def test_verbnet_theme_spellings_are_one_theme():
    """VerbNet writes ``co-patient`` and ``co_patient``: both settle an unlisted tag; a theme no rolelink carries is not
    in the table."""
    script = _script()
    assert script.role_kind("arg2", "PRD", ("co-patient",)) == script.role_kind("arg2", "PRD", ("co_patient",))
    assert script.role_kind("arg2", "PRD", ("co-patient",)) == "who_or_what"
    assert "time" not in script.THEME_KIND
    clasp = {role: wh for wh, role in QuestionMap.load().pairs("clasp.01")}
    assert clasp["arg2"] == "who_or_what"  # its arg2 is linked to VerbNet "co-patient"


def test_roleless_rolesets_are_never_offered():
    """A roleset the map holds with no role (a light-verb entry, say) answers no question: frames_of skips it, from
    the bonfire's rolesets and from the map's (a structural test, not an id pattern)."""
    qm = QuestionMap({"*": [], "brask.01": [["who_or_what", "arg0"]], "brask.02": [], "quill.01": []})
    assert qm.rolesets_of("brask") == ["brask.01"] and qm.roleless("brask.02") and not qm.roleless("unheard.01")
    lex = Lexicon(["brask.01", "brask.02", "zorvex.01"], qm)
    assert lex.frames_of("brask") == ["brask.01"]
    assert lex.frames_of("zorvex") == ["zorvex.01"]  # not in the map at all: kept, the bonfire has it
    assert lex.frames_of("quill") == []
    assert all(QuestionMap.load().pairs(r) for r in QuestionMap.load().rolesets_of("do"))


def test_question_map_queries_alias_who_and_what():
    qm = QuestionMap({"*": [["when", "argm-tmp"], ["where", "argm-loc"]], "meet.01": [["who_or_what", "arg1"]]})
    assert qm.roles("meet.01", "who") == ["arg1"] and qm.roles("meet.01", "what") == ["arg1"]
    assert qm.roles("meet.01", "who_or_what") == ["arg1"]
    assert qm.roles("meet.01", "when") == ["argm-tmp"]
    assert qm.roles("meet.01", "why") == [] and qm.roles("unheard.01", "where") == ["argm-loc"]
    assert qm.roles("meet.01", "unknown") == []  # an unsettled role never answers a question


def test_shipped_map_covers_the_frames():
    """Every roleset of the kernel's frame dirs (the setup-managed upstream frames + the packaged annex) has an entry."""
    from memory_kernel import setup_runtime
    from memory_kernel.knowledge.propbank_framesets import (
        load_propbank_roleset_index,
        load_propbank_rolesets,
        verbnet_role_themes,
    )
    from memory_kernel.pipeline.frames import annex_dir, upstream_dir

    shipped = QuestionMap.load()
    annex = {r.roleset_id for r in load_propbank_rolesets(annex_dir())}
    assert annex and annex <= set(shipped.rolesets())
    script = _script()
    upstream = upstream_dir(setup_runtime.resolve_data_dir(None, os.environ))
    if not upstream.is_dir():
        pytest.skip(f"no upstream PropBank frames under {upstream} (`mk setup --component propbank-frames`)")
    rolesets = load_propbank_roleset_index(upstream).values()
    # with the upstream frames present the map holds exactly their rolesets and the annex's: nothing else
    assert {r.roleset_id for r in rolesets} | annex == set(shipped.rolesets())
    excepted_tags = {tag for tag, _ in script.TAG_THEME_KIND}
    # a listed tag maps as the table says, or as a listed (tag, theme) exception: the theme fallback never overrides it
    for roleset in rolesets:
        pairs = dict((role, wh) for wh, role in shipped.pairs(roleset.roleset_id))
        for role in roleset.roles:
            tag = role.function_tag.upper()
            if tag in script.TAG_KIND and not role.role.startswith("argm"):
                themes = verbnet_role_themes(roleset.roleset_id, upstream) if tag in excepted_tags else {}
                exceptions = [
                    script.TAG_THEME_KIND[(tag, t)]
                    for t in themes.get(role.role, ())
                    if (tag, t) in script.TAG_THEME_KIND
                ]
                assert pairs[role.role] == (exceptions or [script.TAG_KIND[tag]])[0], (roleset.roleset_id, role)
    give = {role: wh for wh, role in shipped.pairs("give.01")}
    assert give["arg0"] == "who_or_what"  # give.01's PAG


def test_verbs_are_marked_and_cut_back():
    from memory_kernel.runtime.normalise import normalise

    norm = normalise("Kira Dace published the report")
    assert norm.text == "⟦ kira dace ⟪publish⟫ the report ⟧"
    left = norm.text.index("⟪publish⟫")
    assert norm.spans.cut(left, left + len("⟪publish⟫")) == "published"
    assert norm.spans.cut(left + 1, left + 3) == "published"  # a cut inside a marked token returns it as written


def test_frames_of_and_wh_role():
    qm = QuestionMap(
        {
            "*": [["when", "argm-tmp"], ["where", "argm-loc"]],
            "brew.01": [["who_or_what", "arg0"], ["who_or_what", "arg1"]],
            "brew.02": [["who_or_what", "arg0"], ["where", "arg2"]],
            "zorvex.01": [["who_or_what", "arg0"]],
        }
    )
    lex = Lexicon(["work.01", "work.02", "workout.01", "meet.03"], qm)
    assert lex.frames_of("work") == ["work.01", "work.02"]  # the bonfire's rolesets first
    assert lex.frames_of("brew") == ["brew.01", "brew.02"]  # none in the bonfire: the question map's
    assert lex.frames_of("quill") == []
    assert lex.wh_roles(["brew.01", "brew.02"], "where") == [
        ("brew.01", "argm-loc"),
        ("brew.02", "arg2"),
        ("brew.02", "argm-loc"),
    ]
    assert lex.wh_roles(["brew.02"], "who") == [("brew.02", "arg0")]

    ops = LexiconOps(lex)
    assert asyncio.run(ops.run("frames-of", {"lemma": "brew"})) == ["brew.01", "brew.02"]
    assert asyncio.run(ops.run("wh-role", {"frames": ["brew.02"], "wh": "when"})) == [("brew.02", "argm-tmp")]
    with pytest.raises(KeyError):
        asyncio.run(ops.run("find-entity", {"name": "Kira Dace"}))


class _Recorder:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[str] = []

    async def run(self, op, args):
        self.calls.append(op)
        return self.label


def test_composite_routes_by_spec():
    from memory_kernel.runtime.ops_memory import MEMORY_SPECS

    assert not set(LEXICON_SPECS) & set(MEMORY_SPECS)
    store, lexicon, shadow = _Recorder("store"), _Recorder("lexicon"), _Recorder("shadow")
    memory = CompositeMemory([(MEMORY_SPECS, store), (LEXICON_SPECS, lexicon), (LEXICON_SPECS, shadow)])
    assert asyncio.run(memory.run("find-entity", {"name": "Kira Dace"})) == "store"
    assert asyncio.run(memory.run("wh-role", {"frames": [], "wh": "where"})) == "lexicon"  # the first part wins
    assert shadow.calls == [] and store.calls == ["find-entity"]
    assert set(memory.specs) == set(MEMORY_SPECS) | set(LEXICON_SPECS)
    with pytest.raises(KeyError):
        asyncio.run(memory.run("brew", {}))


def test_composite_refuses_conflicting_specs():
    """Two parts may share an op name only with the same spec; a name defined differently is refused."""
    from memory_kernel.runtime.program import OpSpec

    same = _Recorder("same")
    CompositeMemory([(LEXICON_SPECS, same), (dict(LEXICON_SPECS), same)])  # an identical duplicate is fine
    clash = {"frames-of": OpSpec("frames-of", ("lemma", "sense"), ("frames",), "memory")}
    with pytest.raises(ValueError, match="frames-of"):
        CompositeMemory([(LEXICON_SPECS, same), (clash, same)])
