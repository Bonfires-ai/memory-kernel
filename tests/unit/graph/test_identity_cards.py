import json
from pathlib import Path

import pytest

from memory_kernel.graph.identity_cards import load_identity_cards


def test_shipped_cards_load():
    cards = load_identity_cards()
    kinds = {c.kind for c in cards.constructions}
    assert kinds == {"equative", "naming", "same_person"}
    assert "my name" in cards.self_holders
    assert "not" in cards.negation_cues and cards.strip_prefixes == {"to", "as"}
    assert {"if", "wish", "were", "as if"} <= set(cards.irrealis_cues)
    assert {"monday", "december", "today", "new year"} <= cards.not_names
    assert any(c.rolesets >= {"call.01", "name.01"} for c in cards.constructions)


def test_bad_kind_is_refused(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "identity-constructions/v1",
                "constructions": [
                    {"id": "x", "kind": "guess", "rolesets": ["be.01"], "holder_role": "arg1", "name_role": "arg2"}
                ],
                "self_holders": [],
                "same_person_markers": [],
                "max_name_tokens": 3,
            }
        )
    )
    with pytest.raises(ValueError, match="kind"):
        load_identity_cards(p)
