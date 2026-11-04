"""Measure the parse-read answer shape (``runtime.shape.read_shape``) over question sets: shape, role and kind coverage,
the mined patterns per bonfire. (Its agreement with the retired wh word lists was measured at 1.4 Task 11, on the code
before Task 12 deleted them; the lists are gone, so this script no longer compares.)

Usage (from the repository root, ``PYTHONPATH=src``, the project venv's python):
    python scripts/shape_agreement.py --questions SOURCE=FILE [SOURCE=FILE ...] --taxq KEY=FILE [KEY=FILE ...] --out DIR

``--questions``: JSON lines, one question per row (``q``; an optional ``id`` and ``conv``). ``SOURCE`` names the
question set in the report. ``--taxq``: a bonfire's taxonomy-question output (``{"valid": [{"question", "answer_role",
...}]}``). A question row's patterns are mined from ``--taxq`` entry ``KEY`` = the row's ``conv``, else the row's
``SOURCE``: one bonfire, one pattern set, never pooled across ``KEY``s.

Per question the row holds ``read_shape`` (shape, answer role, expected kind, plural, polar) and the wh token's
dependency path; ``contaminated`` marks a question that appears verbatim (case-insensitive, whitespace-collapsed) among
its own bonfire's taxonomy questions.

Output (``--out``): ``rows.jsonl`` and ``summary.md`` (role and kind coverage, patterns per bonfire). Parsing is the
kernel's turn parser (spaCy only: no engine, no paid call)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _pairs(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        key, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=FILE, got {item!r}")
        out[key] = path
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", nargs="+", required=True, metavar="SOURCE=FILE")
    ap.add_argument("--taxq", nargs="+", required=True, metavar="KEY=FILE")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from memory_kernel.graph.g1_nlp import turn_doc
    from memory_kernel.runtime.shape import mine_shape_patterns, read_shape, wh_path

    started = time.time()
    taxq_files = _pairs(args.taxq)
    questions: list[tuple[str, dict]] = []
    for source, path in _pairs(args.questions).items():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                questions.append((source, json.loads(line)))
    needed = {row.get("conv") or source for source, row in questions}
    missing = needed - set(taxq_files)
    if missing:
        raise SystemExit(f"no --taxq entry for {sorted(missing)}")

    patterns: dict[str, list[dict]] = {}
    taxq_rows: dict[str, list[dict]] = {}
    for key in sorted(needed):
        taxq_rows[key] = json.loads(Path(taxq_files[key]).read_text(encoding="utf-8"))["valid"]
        patterns[key] = mine_shape_patterns(taxq_rows[key], turn_doc)
        print(
            f"mined {key}: {len(taxq_rows[key])} rows -> {len(patterns[key])} patterns ({time.time() - started:.0f}s)",
            file=sys.stderr,
            flush=True,
        )
    own = {key: {_norm(r["question"]) for r in rows} for key, rows in taxq_rows.items()}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as sink:
        for n, (source, row) in enumerate(questions):
            key = row.get("conv") or source
            doc = turn_doc(row["q"])
            got = read_shape(doc, patterns[key])
            where = wh_path(doc)
            rec = {
                "source": source,
                "bonfire": key,
                "id": row.get("id"),
                "q": row["q"],
                "shape": got.shape,
                "answer_role": got.answer_role,
                "expected_kind": got.expected_kind,
                "plural": got.plural,
                "polar": got.polar,
                "wh_path": where,
                "contaminated": _norm(row["q"]) in own[key],
            }
            rows.append(rec)
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                print(f"{n}/{len(questions)} ({time.time() - started:.0f}s)", file=sys.stderr, flush=True)

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} = {100 * a / b:.1f}%" if b else "0/0"

    sources = sorted({r["source"] for r in rows})
    lines = [
        "# Shape agreement",
        "",
        f"{len(rows)} questions, {time.time() - started:.0f}s.",
        "",
        "| source | n | contaminated |",
        "|---|---|---|",
    ]
    for s in sources:
        rs = [r for r in rows if r["source"] == s]
        lines.append(f"| {s} | {len(rs)} | {sum(r['contaminated'] for r in rs)} |")
    for s in sources:
        rs = [r for r in rows if r["source"] == s]
        lines += [
            "",
            f"## Role and kind coverage: {s}",
            "",
            f"- answer role found: {pct(sum(r['answer_role'] is not None for r in rs), len(rs))}",
            f"- expected kind found: {pct(sum(r['expected_kind'] is not None for r in rs), len(rs))}",
            f"- read_shape found no shape: {pct(sum(r['shape'] is None for r in rs), len(rs))}",
            f"- no wh word found: {pct(sum(r['wh_path'] is None for r in rs), len(rs))}",
        ]
        open_rs = [r for r in rs if r["wh_path"] is not None]
        lines.append(
            f"- of the {len(open_rs)} questions with a wh word, answer role found: "
            f"{pct(sum(r['answer_role'] is not None for r in open_rs), len(open_rs))}"
        )
        roles = Counter(r["answer_role"] for r in rs)
        lines.append(f"- answer roles: {dict(roles.most_common())}")
    lines += [
        "",
        "## Patterns per bonfire",
        "",
        "| bonfire | taxq rows | patterns | rows in patterns | top-9 patterns cover | questions |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(patterns):
        ps = patterns[key]
        total = len(taxq_rows[key])
        covered = sum(p["rows"] for p in ps)
        top9 = sum(p["rows"] for p in ps[:9])
        nq = sum(1 for r in rows if r["bonfire"] == key)
        lines.append(f"| {key} | {total} | {len(ps)} | {pct(covered, total)} | {pct(top9, total)} | {nq} |")
    contaminated = [r for r in rows if r["contaminated"]]
    lines += ["", f"## Contamination (question verbatim among its own bonfire's taxq rows): {len(contaminated)}", ""]
    for r in contaminated:
        lines.append(f"- {r['source']} {r['id']}: {r['q']}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"done {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
