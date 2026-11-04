"""Generate the runtime grammar's question map from the PropBank frame files (no LLM, no word list).

Usage (from the repository root, with ``PYTHONPATH=src`` so the script imports this checkout's ``memory_kernel``):
    PYTHONPATH=src python scripts/build_question_map.py [--data-dir DIR] [--frame-dir DIR ...] [--out PATH]
    PYTHONPATH=src python scripts/build_question_map.py --oracle <taxq questions.json> [...] [--map PATH]

The map says which role of each roleset answers which kind of question: ``{roleset: [[wh, role], ...]}`` for every
roleset of the kernel's frame dirs (``memory_kernel.pipeline.frames``: the setup-managed upstream frames at the pinned
commit, then the packaged annex; ``--frame-dir`` replaces both; first dir wins on a repeated roleset id, and inside
one dir the kernel's roleset index decides, last file wins: overhang.01 is in hang.xml and overhang.xml), plus the
shared ``"*"`` entry: the time and place modifiers every frame admits. A role's kind comes from its function tag and
VerbNet theme together where ``TAG_THEME_KIND`` lists the pair, else from its function tag (``TAG_KIND``); a numbered
role (``arg0``..``arg5``) with no tag at all is a participant (``who_or_what``); any other role falls back to its
VerbNet theme (``THEME_KIND``; VerbNet writes some themes with a hyphen, ``co-patient``, others with an underscore:
both are read as ``co_patient``), else it is ``unknown``. A modifier role defined in a frame
(``n="M"``) is written ``argm-<tag>`` as the graph names it. Output is deterministic: sorted keys, each roleset's pairs
sorted by role then kind, one roleset per line.

A role whose VerbNet themes have store kinds (``THEME_LABELS``: a closed VerbNet thematic inventory -> the store's
``KIND_LABELS``) carries them as a third element, ``[wh, role, [label, ...]]``: the kinds of entity that fill it, which
the runtime's state path reads (``role-kinds``). A numbered role's kinds come from its own themes. A modifier's come
from its shared twin (the loader keys every modifier of a roleset as one ``argm``, so a modifier has no theme of its
own): the shared ``argm-loc`` is VerbNet's ``location`` (``SHARED_MODIFIERS``), and so is a frame's own ``argm-loc``.

The shipped ``question-map.json`` is generated from the default dirs: the upstream propbank-frames at the pinned
commit c66e0ccf28b5 (``runtime-pins.json``; placed by ``mk setup --component propbank-frames`` under
``<data>/propbank-frames/c66e0ccf28b5/frames``) plus the packaged annex (``memory_kernel/data/propbank-frames-annex/
frames``): ``MK_DATA_DIR=<data> PYTHONPATH=src python scripts/build_question_map.py``, or ``--frame-dir`` naming both dirs in that
order.

``--oracle`` prints agreement statistics between the map and the ``meaning_rows`` of bootstrap taxq question files
(``{roleset: [["answers", wh, role], ...]}``): a cross-check only; nothing is copied from them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from memory_kernel.graph.g1_names import ORGANISATION, PLACE
from memory_kernel.knowledge.propbank_framesets import load_propbank_roleset_index, verbnet_role_themes
from memory_kernel.runtime.lexicon import SHARED, UNKNOWN, QuestionMap

#: PropBank function tag -> question kind (the runtime-grammar plan's table; a ruling changes this one place)
TAG_KIND: Mapping[str, str] = {
    "LOC": "where",
    "DIR": "where",
    "GOL": "where",
    "SRC": "where",
    "TMP": "when",
    "MNR": "how",
    "CAU": "why",
    "PRP": "why",
    "EXT": "how_much",
    "PAG": "who_or_what",
    "PPT": "who_or_what",
    "COM": "who_or_what",
    "VSP": "who_or_what",
}
#: (tag, VerbNet theme) -> question kind, ahead of ``TAG_KIND``: the exceptions to a tag's kind (ruling, Task 6a fix
#: round 1: a goal that is a recipient answers who/what, not where -- give.01 and tell.01 arg2)
TAG_THEME_KIND: Mapping[tuple[str, str], str] = {
    ("GOL", "recipient"): "who_or_what",
}
#: VerbNet thematic role -> question kind, read only for a role whose tag ``TAG_KIND`` does not settle (``PRD``,
#: ``ADV``, ``SE1``, ...); the labels are VerbNet's role inventory, not English vocabulary
THEME_KIND: Mapping[str, str] = {
    "location": "where",
    "destination": "where",
    "source": "where",
    "initial_location": "where",
    "trajectory": "where",
    "manner": "how",
    "instrument": "how",
    "extent": "how_much",
    "agent": "who_or_what",
    "co_agent": "who_or_what",
    "patient": "who_or_what",
    "co_patient": "who_or_what",
    "theme": "who_or_what",
    "co_theme": "who_or_what",
    "experiencer": "who_or_what",
    "recipient": "who_or_what",
    "beneficiary": "who_or_what",
    "stimulus": "who_or_what",
    "causer": "who_or_what",
    "product": "who_or_what",
    "result": "who_or_what",
    "material": "who_or_what",
    "topic": "who_or_what",
}
#: VerbNet thematic role -> the store kinds (``g1_names.KIND_LABELS``) of the entities that fill it: the state path's
#: expected kinds (Task 11a). Spatial roles are places; a beneficiary that stays a ``where`` role is the employer
#: (work.01 arg2): an organisation. Person is in no row: a state line always names its own subject, so a Person kind
#: would answer the question's subject. A theme with no row gives no kind (the state path then answers nothing).
THEME_LABELS: Mapping[str, tuple[str, ...]] = {
    "location": (PLACE,),
    "destination": (PLACE,),
    "source": (PLACE,),
    "initial_location": (PLACE,),
    "trajectory": (PLACE,),
    "beneficiary": (ORGANISATION,),
}
#: the modifiers every frame admits, emitted once under ``"*"``: (wh, role, VerbNet theme or None). argm-loc is the
#: location modifier (PropBank's LOC: where the event takes place), VerbNet's ``location``; argm-tmp has no store kind.
SHARED_MODIFIERS = (("when", "argm-tmp", None), ("where", "argm-loc", "location"))
#: the numbered (core) roles an untagged role is a participant of
CORE_ROLES = frozenset(f"arg{n}" for n in range(6))
#: where the map ships, relative to the repository root
SHIPPED = Path("src/memory_kernel/data/runtime-grammar/question-map.json")


def role_kind(role: str, tag: str, themes: Sequence[str]) -> str:
    """The question kind of one role (see the module docstring)."""
    tag = tag.strip().upper()
    themes = [_theme(t) for t in themes]
    for theme in themes:
        if (tag, theme) in TAG_THEME_KIND:
            return TAG_THEME_KIND[(tag, theme)]
    if tag in TAG_KIND:
        return TAG_KIND[tag]
    if not tag and role in CORE_ROLES:
        return "who_or_what"
    for theme in themes:
        if theme in THEME_KIND:
            return THEME_KIND[theme]
    return UNKNOWN


def role_labels(themes: Sequence[str]) -> tuple[str, ...]:
    """The store kinds of a role with these VerbNet themes: the union of their ``THEME_LABELS`` rows, sorted."""
    return tuple(sorted({label for t in themes for label in THEME_LABELS.get(_theme(t), ())}))


def _entry(wh: str, role: str, labels: Sequence[str]) -> list:
    """``[wh, role]``, or ``[wh, role, [label, ...]]`` when the role has store kinds."""
    return [wh, role, list(labels)] if labels else [wh, role]


def _theme(theme: str) -> str:
    """A VerbNet theme as ``THEME_KIND`` keys it: ``co-patient`` and ``co_patient`` are one theme."""
    return theme.strip().lower().replace("-", "_")


def _role_id(role: str, tag: str) -> str:
    """``argm`` with a tag is the graph's ``argm-<tag>``; numbered roles stay as the loader writes them."""
    tag = tag.strip().lower()
    return f"argm-{tag}" if role == "argm" and tag else role


def build_question_map(frame_dirs: Sequence[Path]) -> dict[str, list[list]]:
    """``{roleset: [[wh, role(, labels)], ...]}`` for every roleset of ``frame_dirs`` (first dir wins), plus ``"*"``."""
    shared_theme = {role: theme for _, role, theme in SHARED_MODIFIERS if theme}
    out: dict[str, list[list]] = {
        SHARED: [_entry(wh, role, role_labels([theme] if theme else [])) for wh, role, theme in SHARED_MODIFIERS]
    }
    for directory in frame_dirs:
        for roleset in load_propbank_roleset_index(directory).values():
            if roleset.roleset_id in out:
                continue
            themes = verbnet_role_themes(roleset.roleset_id, directory)
            entries = set()
            for r in roleset.roles:
                role = _role_id(r.role, r.function_tag)
                own = themes.get(r.role, ())
                # the loader keys every modifier of a roleset as one "argm": a modifier's kinds are its shared twin's
                kinds_from = own if r.role in CORE_ROLES else ([shared_theme[role]] if role in shared_theme else [])
                entries.add((role, role_kind(r.role, r.function_tag, own), role_labels(kinds_from)))
            out[roleset.roleset_id] = [_entry(wh, role, labels) for role, wh, labels in sorted(entries)]
    return dict(sorted(out.items()))


def write_question_map(entries: Mapping[str, list[list]], path: Path) -> None:
    """One roleset per line, sorted keys, ASCII: small, stable diffs."""
    lines = [f"{json.dumps(key)}: {json.dumps(entries[key], separators=(',', ':'))}" for key in sorted(entries)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")


def default_frame_dirs(data_dir: str | None) -> list[Path]:
    from memory_kernel import setup_runtime
    from memory_kernel.pipeline import frames

    return list(frames.default_frame_dirs(setup_runtime.resolve_data_dir(data_dir, os.environ)))


def oracle_stats(question_map: QuestionMap, meaning_rows: Mapping[str, Iterable[Sequence[str]]]) -> dict:
    """How the map agrees with one taxq file's ``meaning_rows`` (rows ``["answers", wh, role]``)."""
    counts: Counter[str] = Counter()
    by_wh: dict[str, Counter[str]] = {}
    known = set(question_map.rolesets())
    for roleset, rows in sorted(meaning_rows.items()):
        counts["rolesets"] += 1
        counts["rolesets_in_map"] += roleset in known
        for row in rows:
            _, wh, role = row
            counts["rows"] += 1
            own = {r: label for label, r in question_map.pairs(roleset)}
            if roleset not in known:
                verdict = "roleset_missing"
            elif role in question_map.roles(roleset, wh):
                verdict = "agree"
            elif own.get(role) == UNKNOWN:
                verdict = "role_unknown"
            elif role in own:
                verdict = f"map_says_{own[role]}"
            else:
                verdict = "role_absent"
            counts[verdict] += 1
            by_wh.setdefault(wh, Counter())[verdict] += 1
    return {"totals": dict(counts), "by_wh": {wh: dict(c) for wh, c in sorted(by_wh.items())}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="the `mk setup` data dir (else MK_DATA_DIR, else the default)")
    parser.add_argument("--frame-dir", action="append", type=Path, help="a frames dir (repeatable; replaces both)")
    parser.add_argument("--out", type=Path, default=None, help=f"where to write the map (default {SHIPPED})")
    parser.add_argument("--oracle", nargs="+", type=Path, help="taxq questions.json files to cross-check against")
    parser.add_argument("--map", type=Path, default=None, help="the map --oracle reads (default: the shipped one)")
    args = parser.parse_args(argv)

    if args.oracle:
        question_map = QuestionMap.load(args.map)
        for path in args.oracle:
            rows = json.loads(path.read_text(encoding="utf-8"))["meaning_rows"]
            print(json.dumps({"oracle": path.name, **oracle_stats(question_map, rows)}, sort_keys=True))
        return 0

    frame_dirs = args.frame_dir or default_frame_dirs(args.data_dir)
    missing = [d for d in frame_dirs if not Path(d).is_dir()]
    if missing:
        print(f"frame dir(s) missing: {missing} (run `mk setup --component propbank-frames`)", file=sys.stderr)
        return 2
    entries = build_question_map(frame_dirs)
    out = args.out or Path(__file__).resolve().parents[1] / SHIPPED
    write_question_map(entries, out)
    kinds = Counter(pair[0] for key, pairs in entries.items() if key != SHARED for pair in pairs)
    labels = Counter("|".join(p[2]) for key, pairs in entries.items() if key != SHARED for p in pairs if len(p) > 2)
    summary = {"rolesets": len(entries) - 1, "roles_by_kind": dict(sorted(kinds.items()))}
    print(json.dumps({**summary, "roles_by_labels": dict(sorted(labels.items())), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
