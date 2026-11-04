import pytest

from memory_kernel.runtime.api import _licence, _repair_licence
from memory_kernel.runtime.executor import VALUE_SPECS
from memory_kernel.runtime.policy import DEFAULT_POLICY
from memory_kernel.runtime.program import Program, Registry, ToolSpec


@pytest.mark.parametrize(
    ("side_effect", "consent", "actor", "speaker", "expected"),
    [
        (False, "confirm", "Kira", "Kira", False),  # a pure tool never confirms
        (True, "speaker_turn", "Kira", "Kira", False),  # the speaker's own turn counts as consent
        (True, "confirm", "Kira", "Kira", True),  # a confirm tool still confirms
        (True, "speaker_turn", "Kira", None, True),  # no speaker, no consent
        (True, "speaker_turn", None, "Kira", True),  # no actor, no consent
        (True, "speaker_turn", "Milo", "Kira", True),  # another person's turn is not the speaker's
        (None, "speaker_turn", "Kira", "Kira", True),  # an unknown effect always confirms (R6)
    ],
)
def test_policy_decides_confirmation(side_effect, consent, actor, speaker, expected):
    tool = ToolSpec.simple("give", ("item",), side_effect, consent=consent)
    assert DEFAULT_POLICY.needs_confirmation(tool, actor=actor, speaker=speaker) is expected


def test_licence_rows_use_the_policy():
    registry = Registry(dict(VALUE_SPECS)).with_tools(
        [
            ToolSpec.simple("take", ("item",), False),
            ToolSpec.simple("give", ("item", "person"), True, consent="speaker_turn"),
        ]
    )
    program = Program.from_meaning(
        [
            ["take", "?s", "?x"],
            ["const", "?x", "tea"],
            ["give", "?s2", "?x2", "?r"],
            ["const", "?x2", "tea"],
            ["deixis", "?r", "speaker"],
        ]
    )
    rows = {row["tool"]: row["needs_confirmation"] for row in _licence(program, registry, DEFAULT_POLICY, "Kira")}
    assert rows == {"take": False, "give": False}
    # a tool the catalogue dropped since a row copied it inline: still listed, its effect unknown (R6: the game confirms)
    gone = Program.from_meaning([["take", "?s", "?x"], ["const", "?x", "tea"], ["vanish", "?s2", "?x"]])
    listed = {row["tool"]: row for row in _licence(gone, registry, DEFAULT_POLICY, "Kira")}
    assert listed["vanish"] == {
        "tool": "vanish",
        "side_effect": None,
        "needs_confirmation": True,
        "reads": [],
        "writes": [],
    }
    assert [r["tool"] for r in _repair_licence(gone, registry, DEFAULT_POLICY, "Kira") or []] == ["take", "vanish"]
