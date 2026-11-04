"""G1 card inventory helpers: the write-once artifact writer and the card census.

Ported from a reference-harness module (unreached top-level definitions pruned).

A G1 card is a construction card (``construction.types.CardRecord``) that REFERENCES a PropBank roleset
(``discriminant``) and never re-authors its form: G0 (stock PropBank) is untouched (PropBank is never
extended: G1 is a new layer; PropBank role conventions kept). What G1 adds is the
meaning pole the questionnaire needs: which question shape the frame answers, through which roles.

Kernel 1.8 archived the remine that attested cards (``merge_taxonomy``, ``fold_alignments`` and the gap loop): a build's
inventory is graph_a's empty one (``graph.empty_inventory``), whose
``census`` is still sealed. A card counts toward closure only with attestations from >= 2 distinct pairs
(``census``); singletons are listed apart.
"""

from collections import defaultdict

from memory_kernel.construction.mined_affordance_contract import canonical_json
from memory_kernel.pipeline.ledger import write_new

GRAMMAR_ID = "mk-g1-grammar"


def write_once(path, value):
    """Write a sealed artifact once; an identical rewrite is a no-op, a changed one is refused."""
    data = canonical_json(value).encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"cached artifact changed: {path}; write to a new directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, value)


def census(cards):
    counted = [c for c in cards if c["support"] >= 2]
    shapes, by_label, by_shape = defaultdict(int), defaultdict(int), defaultdict(int)
    for card in counted:
        for _, shape, _ in card["meaning"]:
            shapes[shape] += 1
        for label, shape, _ in card.get("answers_by_label", []):
            by_label[label] += 1
            by_shape[shape] += 1
    return {
        "cards": len(cards),
        "counted": len(counted),
        "singletons": len(cards) - len(counted),
        "form_cards": sum(1 for c in cards if c.get("form")),
        "attestations": sum(len(c["provenance"]["attestations"]) for c in cards),
        "counted_meaning_rows_by_shape": dict(sorted(shapes.items())),
        "cards_with_answers_by_label": sum(1 for c in cards if c.get("answers_by_label")),
        "answers_by_label_rows": sum(len(c.get("answers_by_label", [])) for c in cards),
        "counted_answers_by_label_by_label": dict(sorted(by_label.items())),
        "counted_answers_by_label_by_shape": dict(sorted(by_shape.items())),
    }
