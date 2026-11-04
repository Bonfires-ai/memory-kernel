"""The per-bonfire catalogue file: typed tool parameters, refusals, and cards generated from its forms."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_kernel.runtime.catalogue import CatalogueError, load_catalogue
from memory_kernel.runtime.program import ToolParam


def _tool(**over):
    tool = {
        "name": "make_sound",
        "description": "Play a pattern on a source",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "source": {"type": "string"}},
        },
        "x-side-effect": True,
        "x-args": {
            "pattern": {"role": "theme", "constraint": "sound"},
            "source": {"role": "instrument", "kind": "Artifact", "many": True},
        },
        "x-forms": ["play {pattern} on {source}"],
    }
    tool.update(over)
    return tool


def _write(tmp_path: Path, *tools: dict) -> Path:
    path = tmp_path / "catalogue.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "kernel-catalogue/v1", "tools": list(tools)}))
    return path


def test_catalogue_loads_typed_tools(tmp_path):
    other = _tool(name="wave", **{"x-forms": []})
    other["x-consent"] = "speaker_turn"
    other["x-side-effect"] = False
    other["input_schema"] = {"type": "object", "properties": {"who": {"type": "string"}}}
    other["x-args"] = {"who": {"role": "agent", "kind": "Person"}}
    cat = load_catalogue(_write(tmp_path, _tool(), other))
    sound, wave = cat.tools
    assert sound.params == (
        ToolParam("pattern", role="theme", constraint="sound"),
        ToolParam("source", role="instrument", kind="Artifact", many=True),
    )
    assert sound.side_effect is True and sound.consent == "confirm"
    assert sound.forms == ("play {pattern} on {source}",)
    assert wave.params == (ToolParam("who", role="agent", kind="Person"),)
    assert wave.consent == "speaker_turn" and wave.side_effect is False
    assert len(cat.sha256) == 64


def test_returns_names_the_parameter_its_result_stands_for(tmp_path):
    from memory_kernel.runtime.api import default_registry

    make = _tool(name="make", **{"x-forms": [], "x-returns": {"of": "item"}})
    make["input_schema"]["properties"] = {"item": {}, "material": {}}
    make["x-args"] = {"item": {"role": "product"}, "material": {"role": "input", "many": True}}
    give = _tool(name="give", **{"x-forms": []})
    cat = load_catalogue(_write(tmp_path, make, give))
    made, given = cat.tools
    assert made.returns == "item" and given.returns is None
    registry = default_registry(cat.tools)
    assert registry.specs["make"].returns == "item" and registry.specs["give"].returns is None


def test_x_runner_is_validated(tmp_path):
    """C14: a tool runs in the office unless its row says ``"x-runner": "agent"``; the registry keeps the runner; any
    other value is refused naming the tool."""
    from memory_kernel.runtime.api import default_registry

    ask = _tool(name="ask", **{"x-forms": [], "x-runner": "agent", "x-side-effect": False})
    sound, asked = load_catalogue(_write(tmp_path, _tool(), ask)).tools
    assert (sound.runner, asked.runner) == ("office", "agent")
    assert default_registry([sound, asked]).specs["ask"].runner == "agent"
    for bad in ("elsewhere", None, 1):
        path = tmp_path / str(bad)
        with pytest.raises(CatalogueError, match=r"'make_sound'.*x-runner"):
            load_catalogue(_write(path, _tool(**{"x-runner": bad})))


def test_any_registry_failure_is_a_catalogue_error_naming_the_tool(tmp_path, monkeypatch):
    def boom(_tools):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr("memory_kernel.runtime.api.default_registry", boom)
    with pytest.raises(CatalogueError, match=r"'make_sound'.*registry exploded"):
        load_catalogue(_write(tmp_path, _tool()))


def test_missing_side_effect_is_refused(tmp_path):
    tool = _tool()
    del tool["x-side-effect"]
    with pytest.raises(CatalogueError, match=r"make_sound.*x-side-effect"):
        load_catalogue(_write(tmp_path, tool))


def test_unannotated_argument_is_refused(tmp_path):
    tool = _tool()
    del tool["x-args"]["source"]
    with pytest.raises(CatalogueError, match=r"make_sound.*source"):
        load_catalogue(_write(tmp_path, tool))


def test_forms_generate_cards(tmp_path):
    cat = load_catalogue(_write(tmp_path, _tool()))
    (card,) = cat.cards
    assert (card.id, card.kind, card.status) == ("card.make_sound.0", "hand", "live")
    assert [p["slot"] for p in card.form if isinstance(p, dict)] == ["pattern", "source"]
    assert card.form[0] == "play" and card.form[2] == "on"
    assert ["make_sound", "?card-step1", "?pattern", "?source"] in card.meaning
    assert [p[0] for p in card.meaning] == ["procedure", "seq", "make_sound"]
    assert card.meaning[:2] == [["procedure", "?card-proc", "make_sound"], ["seq", "?card-proc", "1", "?card-step1"]]


def test_form_slot_that_is_not_a_param_is_refused(tmp_path):
    with pytest.raises(CatalogueError, match=r"make_sound.*tempo"):
        load_catalogue(_write(tmp_path, _tool(**{"x-forms": ["play {pattern} at {tempo}"]})))


def _raw(tmp_path: Path, data) -> Path:
    path = tmp_path / "catalogue.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def _bad_tool(**over):
    return {"schema_version": "kernel-catalogue/v1", "tools": [_tool(**over)]}


def _without_role():
    tool = _tool()
    del tool["x-args"]["source"]["role"]
    return {"schema_version": "kernel-catalogue/v1", "tools": [tool]}


def _collision():
    tool = _tool(**{"x-forms": []})
    tool["input_schema"]["properties"] = {"Item": {}, "item": {}}
    tool["x-args"] = {"Item": {"role": "a"}, "item": {"role": "b"}}
    return {"schema_version": "kernel-catalogue/v1", "tools": [tool]}


@pytest.mark.parametrize(
    ("data", "needle"),
    [
        (_bad_tool(**{"x-consent": "always"}), "make_sound"),
        ({"schema_version": "v0", "tools": []}, "catalogue.json"),
        ({"schema_version": "kernel-catalogue/v1", "tools": [_tool(), _tool()]}, "make_sound"),
        (_bad_tool(**{"x-args": {"pattern": {"role": "t"}, "source": {"role": "s"}, "ghost": {"role": "g"}}}), "ghost"),
        (_bad_tool(**{"x-forms": ["play {pattern}"]}), "make_sound"),
        (_without_role(), "make_sound"),
        (_collision(), "Item"),
        (_bad_tool(name="play music"), "play music"),
        (_bad_tool(name="zorvex:tool"), "zorvex:tool"),
        ({"schema_version": "kernel-catalogue/v1", "tools": [_tool(name="Zap"), _tool(name="zap")]}, "zap"),
        (_bad_tool(name="recall"), "recall"),
        (_bad_tool(name="answer"), "answer"),
        (None, "catalogue.json"),
        ("{not json", "catalogue.json"),
        (b'{"a": "\xc3("}', "catalogue.json"),
        (_bad_tool(**{"x-returns": "pattern"}), "x-returns"),
        (_bad_tool(**{"x-returns": {"of": "colour"}}), "x-returns"),
        (_bad_tool(**{"x-returns": {"of": "source"}}), "many"),
        (_bad_tool(**{"x-returns": {"of": "pattern", "kind": "x"}}), "x-returns"),
    ],
    ids=[
        "consent",
        "schema",
        "duplicate",
        "ghost-arg",
        "unbound",
        "no-role",
        "case-collision",
        "id-charset-space",
        "id-charset-colon",
        "case-duplicate-tool",
        "keyword-collision",
        "reserved-answer",
        "missing",
        "json",
        "utf8",
        "returns-not-object",
        "returns-no-such-param",
        "returns-many-param",
        "returns-unknown-key",
    ],
)
def test_malformed_catalogue_is_refused(tmp_path, data, needle):
    path = tmp_path / "catalogue.json"
    if data is not None:
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            _raw(tmp_path, data)
    with pytest.raises(CatalogueError, match=needle):
        load_catalogue(path)
