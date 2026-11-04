"""Identity constructions: which captured frames say that two person names are one person.
Vocabulary lives in the card file, not in code."""

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal, cast

SCHEMA = "identity-constructions/v1"
KINDS = ("equative", "naming", "same_person")
DEFAULT_PATH = cast(Path, resources.files("memory_kernel") / "data" / "identity-constructions.json")


@dataclass(frozen=True)
class IdentityConstruction:
    id: str
    kind: Literal["equative", "naming", "same_person"]
    rolesets: frozenset[str]
    holder_role: str | None
    name_role: str | None
    names_role: str | None


@dataclass(frozen=True)
class IdentityCards:
    constructions: tuple[IdentityConstruction, ...]
    self_holders: frozenset[str]
    same_person_markers: tuple[str, ...]
    max_name_tokens: int
    not_names: frozenset[str]
    negation_cues: tuple[str, ...]
    irrealis_cues: tuple[str, ...]
    strip_prefixes: frozenset[str]


def load_identity_cards(path: Path | None = None) -> IdentityCards:
    doc = json.loads(Path(path or DEFAULT_PATH).read_text())
    if doc.get("schema_version") != SCHEMA:
        raise ValueError(f"identity cards: schema_version must be {SCHEMA}")
    constructions = []
    for c in doc["constructions"]:
        if c.get("kind") not in KINDS:
            raise ValueError(f"identity card {c.get('id')!r}: unknown kind {c.get('kind')!r}")
        pair = c.get("holder_role") and c.get("name_role")
        if c["kind"] == "same_person" and not c.get("names_role") or c["kind"] != "same_person" and not pair:
            raise ValueError(f"identity card {c['id']!r}: roles missing for kind {c['kind']}")
        constructions.append(
            IdentityConstruction(
                id=c["id"],
                kind=c["kind"],
                rolesets=frozenset(c["rolesets"]),
                holder_role=c.get("holder_role"),
                name_role=c.get("name_role"),
                names_role=c.get("names_role"),
            )
        )
    return IdentityCards(
        constructions=tuple(constructions),
        self_holders=frozenset(s.lower() for s in doc["self_holders"]),
        same_person_markers=tuple(m.lower() for m in doc["same_person_markers"]),
        max_name_tokens=int(doc["max_name_tokens"]),
        not_names=frozenset(w.lower() for w in doc["not_names"]),
        negation_cues=tuple(w.lower() for w in doc["negation_cues"]),
        irrealis_cues=tuple(w.lower() for w in doc["irrealis_cues"]),
        strip_prefixes=frozenset(w.lower() for w in doc["strip_prefixes"]),
    )
