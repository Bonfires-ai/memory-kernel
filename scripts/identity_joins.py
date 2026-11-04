"""Print the identity claims, joins and conflicts of a built conversation's ``graph.json`` (read-only; writes nothing).

Usage: python scripts/identity_joins.py <graph.json>
Three sections of JSON lines: ``claim`` (holder, name, kind, message id), ``join`` (alias -> canonical) and ``conflict``.
Speakers are the ``speaker`` fields of the records."""

import json
import sys
from pathlib import Path

from memory_kernel.graph.identity import identity_claims, resolve_joins
from memory_kernel.graph.identity_cards import load_identity_cards


def report(records: list[dict]) -> list[dict]:
    speakers = sorted({r["speaker"] for r in records})
    claims = identity_claims(records, load_identity_cards(), speakers)
    joins = resolve_joins(claims, speakers)
    return [
        *(
            {"section": "claim", "holder": c.holder, "name": c.name, "kind": c.kind, "message_id": c.message_id}
            for c in claims
        ),
        *(
            {"section": "join", "alias": a, "canonical": joins.display[root]}
            for root, surfaces in sorted(joins.aliases.items())
            for a in surfaces
        ),
        *({"section": "conflict", "detail": d} for d in joins.conflicts),
    ]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    for line in report(json.loads(Path(argv[1]).read_text())["records"]):
        print(json.dumps(line, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
