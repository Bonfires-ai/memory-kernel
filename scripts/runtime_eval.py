"""Evaluate the runtime grammar over a question set: ``Runtime.plan`` per question, read-only, no paid call.

Usage (from the repository root, with ``PYTHONPATH=src`` so the script imports this checkout's ``memory_kernel``):
    PYTHONPATH=src python scripts/runtime_eval.py --bonfire BONFIRE_ID --questions FILE [--out DIR]
        [--neo4j-uri URI] [--neo4j-user USER] [--secrets-from-container NAME | --env-file PATH | --empty-graph]
        [--port 9718] [--engine-dir DIR] [--data-dir DIR] [--judgements FILE]

``--questions`` is JSON lines: ``{"id", "set", "q", "gold"}`` (``gold`` may be null: a question with no answer in the
graph). Each question is planned as its own chat (the turn window never carries from one question to the next), with
no speaker and a fixed ``at``. The runtime is the one ``mk serve`` runs, over the bonfire's Neo4j group (a Neo4j password must resolve): its
construction store opened read-only, no standing recorded (``record_standing=False``), no recall (a repair carries no ``/context``
result), every graph session a READ session (``GraphReader``). The engine is the script's own, on ``--port``, stopped
before exit.

Output (``--out``, default the questions file's directory): ``results.jsonl`` (one row per question: status, root,
answer_shape, answer, gold, diagnostics and their classes, program, applied constructions, comprehension_ms,
execution_ms, wall_ms, retried, and the hand judgement when ``--judgements`` has one) and ``summary.txt``:

* per set, the share answered / repair / unread, and the share the base set gives an answer shape (read = not unread);
* the repairs by diagnostic class (``CLASSES``: ``?noun``, ``?other``, ``?owner``, ``?anchor``, quantified, guarded
  auxiliary, names nobody, no frame, incomplete reading, empty answer, other);
* the answered precision against the hand judgements (``--judgements``: JSON lines ``{"id", "correct": true | false |
  "partial", "reason"}``; judged by a person, never by string matching);
* p50 / p95 of the wall time, the comprehension and the execution over the real questions (a warm-up turn first is
  left out);
* the do-support misses (an unread turn whose main verb the parser tagged as a noun after a name: the root's subject
  is a noun that ends the turn and has a proper-noun compound, "When did Kira Dace publish?") and the phrasal-verb
  turns (a ``prt`` dependent) and their status.

The classes are read from the runtime's own diagnostics; the do-support and phrasal tests look at POS and dependency
labels only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from memory_kernel.runtime.api import Runtime
from memory_kernel.runtime.executor import Turn
from memory_kernel.runtime.graph_store import NarrowWriter, open_sync_driver
from memory_kernel.runtime.lexicon import Lexicon, QuestionMap, load_rolesets
from memory_kernel.runtime.normalise import VERB_OPEN
from memory_kernel.runtime.store import ConstructionStore

#: a fixed time for every turn (``within`` reads it; the evaluation must not depend on the day it runs)
AT = "2026-10-18T12:00:00+00:00"
DEFAULT_PORT = 9718
#: the warm-up turn (an invented name; left out of every number)
WARM_UP = "where does Kira Dace work?"
#: (class, a fragment of the runtime's diagnostic): the first class whose fragment a diagnostic holds names it
CLASSES: tuple[tuple[str, str], ...] = (
    ("?noun", "?noun ("),
    ("?other", "?other ("),
    ("?owner", "?owner ("),
    ("?anchor", "?anchor ("),
    ("quantified", "is quantified"),
    ("guarded auxiliary", "do-support"),
    ("names nobody", "names nobody"),
    ("no frame", "names no frame"),
    ("incomplete reading", "read the whole turn"),
    ("empty answer", "found nothing for the turn"),
)
#: diagnostics that ride along with a repair and name no reason for it
NOT_A_REASON = ("recall unavailable",)
STATUSES = ("answered", "repair", "unread")
PARTICLE_DEP = "prt"
LIST_CAP = 12


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: a question id is repeated")
    return rows


def load_judgements(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def classify(diagnostics: Sequence[str]) -> list[str]:
    """The classes of a repair's diagnostics, in order, once each (``other`` for one no class names)."""
    out: list[str] = []
    for text in diagnostics:
        if any(text.startswith(skip) for skip in NOT_A_REASON):
            continue
        name = next((cls for cls, fragment in CLASSES if fragment in text), "other")
        if name not in out:
            out.append(name)
    return out


def _turn(question: Mapping[str, Any]) -> Turn:
    return Turn(
        text=question["q"],
        speaker=None,
        addressee=None,
        participants=(),
        chat_id=f"eval-{question['id']}",
        message_id=None,
        at=AT,
    )


def _compact(value: Any) -> Any:
    """An answer as names where it holds entities (at most ``LIST_CAP``), else as it is."""
    if isinstance(value, Mapping) and "name" in value:
        return value.get("name")
    if isinstance(value, Mapping):
        return {k: _compact(v) for k, v in value.items()}
    if isinstance(value, list):
        items = [_compact(v) for v in value[:LIST_CAP]]
        return items + [f"... {len(value) - LIST_CAP} more"] if len(value) > LIST_CAP else items
    return value


def _step(step: Mapping[str, Any]) -> dict[str, Any]:
    """One executor step, compact: op, args, how many values it gave (or the value), ok, ms."""
    out = step.get("out")
    return {
        "op": step.get("op"),
        "args": _compact(step.get("args")),
        "out": len(out) if isinstance(out, list) else _compact(out),
        "ok": step.get("ok"),
        "ms": round(float(step.get("ms") or 0.0), 1),
    }


def _verb_as_noun(doc: Any) -> bool:
    """The do-support miss's parse: the root's subject is a noun that ends the turn (only punctuation after it) and
    has a proper-noun compound ("When did Kira Dace publish?": "publish" NOUN nsubj, "Dace" PROPN compound)."""
    for token in doc:
        if token.dep_ != "nsubj" or token.head.dep_ != "ROOT" or token.pos_ != "NOUN":
            continue
        rest = [t for t in doc[token.i + 1 :] if not t.is_space]
        if all(t.pos_ == "PUNCT" for t in rest) and any(
            c.dep_ == "compound" and c.pos_ == "PROPN" for c in token.children
        ):
            return True
    return False


def _structure(text: str) -> dict[str, bool]:
    """POS and dependency facts of the question: a marked verb, the do-support miss's parse, a verb particle."""
    from memory_kernel.graph.g1_nlp import nlp
    from memory_kernel.runtime.normalise import normalise

    doc = nlp()(text)
    return {
        "marked_verb": VERB_OPEN in normalise(text).text,
        "verb_as_noun": _verb_as_noun(doc),
        "particle": any(t.dep_ == PARTICLE_DEP for t in doc),
    }


async def run_eval(
    questions: Sequence[Mapping[str, Any]],
    bonfire: str,
    engine: Any,
    reader: Any,
    *,
    driver: Any,
    warm_up: bool = True,
) -> list[dict[str, Any]]:
    """One row per question (see the module docstring). The runtime reads bonfire ``bonfire`` read-only; ``driver``
    is a sync Neo4j driver."""
    store = ConstructionStore(NarrowWriter(driver, bonfire), read_only=True)
    lexicon = Lexicon(load_rolesets(driver, bonfire), QuestionMap.load())
    runtime = Runtime(bonfire, engine, reader, record_standing=False, store=store, lexicon=lexicon)
    if warm_up:
        await runtime.plan(_turn({"id": "warm-up", "q": WARM_UP}))
    rows: list[dict[str, Any]] = []
    for question in questions:
        started = time.perf_counter()
        plan = await runtime.plan(_turn(question))
        wall = (time.perf_counter() - started) * 1000
        structure = _structure(question["q"])
        rows.append(
            {
                "id": question["id"],
                "set": question.get("set"),
                "q": question["q"],
                "gold": question.get("gold"),
                "status": plan.status,
                "root": plan.reading.get("root"),
                "answer_shape": plan.answer_shape,
                "answer": _compact(plan.answer),
                "diagnostics": plan.diagnostics,
                "classes": classify(plan.diagnostics) if plan.status == "repair" else [],
                "program": plan.program,
                "steps": [_step(step) for step in plan.trace.get("steps", [])],
                "applied": plan.reading.get("applied"),
                "complete": plan.reading.get("complete"),
                "retried": plan.reading.get("retried"),
                "normalised": plan.reading.get("normalised"),
                "comprehension_ms": plan.trace.get("comprehension_ms"),
                "execution_ms": plan.trace.get("execution_ms"),
                "wall_ms": round(wall, 1),
                "do_support_miss": plan.status == "unread" and structure["verb_as_noun"],
                "phrasal": structure["particle"],
                "marked_verb": structure["marked_verb"],
            }
        )
    return rows


def judged(rows: Iterable[dict[str, Any]], judgements: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``rows`` with the hand judgement of each answered row (``correct``, ``correct_reason``) when there is one."""
    out = []
    for row in rows:
        verdict = judgements.get(row["id"]) if row["status"] == "answered" else None
        extra = {"correct": verdict.get("correct"), "correct_reason": verdict.get("reason")} if verdict else {}
        out.append({**row, **extra})
    return out


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile; None over no values."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    return data[max(0, math.ceil(p / 100 * len(data)) - 1)]


def _share(n: int, total: int) -> str:
    return f"{n}/{total} ({100 * n / total:.0f}%)" if total else "0/0"


def summarise(rows: Sequence[dict[str, Any]], judgements: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """The summary text (see the module docstring)."""
    rows = judged(rows, judgements or {})
    lines: list[str] = ["runtime grammar evaluation", ""]
    sets = sorted({str(r.get("set")) for r in rows})
    lines.append("status per set (answered / repair / unread; read = an answer shape)")
    for name in [*sets, "all"]:
        part = [r for r in rows if name == "all" or str(r.get("set")) == name]
        counts = Counter(r["status"] for r in part)
        cells = "  ".join(f"{s} {_share(counts[s], len(part))}" for s in STATUSES)
        read = len(part) - counts["unread"]
        lines.append(f"  {name:<12} n={len(part):<3} {cells}  read {_share(read, len(part))}")
    lines += ["", "repairs by diagnostic class (a row counts once, under its first class; all classes in brackets)"]
    repairs = [r for r in rows if r["status"] == "repair"]
    first = Counter(r["classes"][0] if r["classes"] else "other" for r in repairs)
    every = Counter(c for r in repairs for c in r["classes"])
    for cls in [*(c for c, _ in CLASSES), "other"]:
        if first[cls] or every[cls]:
            lines.append(f"  {cls:<20} {first[cls]}  [{every[cls]}]")
    if not repairs:
        lines.append("  (none)")
    answered = [r for r in rows if r["status"] == "answered"]
    verdicts = Counter(str(r.get("correct")) for r in answered)
    unjudged = sum(1 for r in answered if r.get("correct") is None)
    lines += ["", "answered precision (hand judgements)"]
    if answered:
        strict = verdicts["True"] / len(answered)
        lenient = (verdicts["True"] + verdicts["partial"]) / len(answered)
        lines.append(
            f"  answered {len(answered)}: correct {verdicts['True']}, partial {verdicts['partial']}, "
            f"wrong {verdicts['False']}, unjudged {unjudged}; precision {strict:.2f} (partial as correct {lenient:.2f})"
        )
    else:
        lines.append("  no answered row")
    lines += ["", f"latency over the {len(rows)} real questions (ms, nearest rank; the warm-up turn left out)"]
    for key in ("wall_ms", "comprehension_ms", "execution_ms"):
        values = [r[key] for r in rows if r.get(key) is not None]
        p50, p95 = percentile(values, 50), percentile(values, 95)
        if p50 is None:
            lines.append(f"  {key:<17} n=0")
        else:
            lines.append(f"  {key:<17} n={len(values):<3} p50 {p50:.1f}  p95 {p95:.1f}  max {max(values):.1f}")
    lines += ["", "do-support misses (unread; the main verb parsed as a noun after a name)"]
    lines += [f"  {r['id']}: {r['q']}" for r in rows if r["do_support_miss"]] or ["  (none)"]
    lines += ["", "phrasal-verb turns (a verb particle) and their status"]
    lines += [f"  {r['id']}: {r['status']}: {r['q']}" for r in rows if r["phrasal"]] or ["  (none)"]
    lines += ["", "unread turns (no answer shape)"]
    lines += [f"  {r['id']}: {r['q']}" for r in rows if r["status"] == "unread"] or ["  (none)"]
    return "\n".join(lines) + "\n"


def _reader(args: argparse.Namespace) -> tuple[Any, str]:
    from memory_kernel.cli import resolve_secrets
    from memory_kernel.serve.runtime import EmptyReader, GraphReader

    if args.empty_graph:
        return EmptyReader(), "empty graph (--empty-graph)"
    secrets = resolve_secrets(env_file=args.env_file, container=args.secrets_from_container)
    password = secrets.values["neo4j_password"]
    if not password:
        raise SystemExit("no Neo4j password (NEO4J_PASSWORD, --env-file or --secrets-from-container); --empty-graph")
    source = secrets.sources["neo4j_password"]
    return GraphReader(args.neo4j_uri, args.neo4j_user, password), f"neo4j {args.neo4j_uri} (password from {source})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--bonfire", required=True, help="the bonfire id (its Neo4j group; read-only)")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--secrets-from-container", default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--empty-graph", action="store_true")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--engine-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--judgements", type=Path, default=None)
    args = parser.parse_args(argv)

    from memory_kernel.runtime.engine import RuntimeEngine

    out = args.out or args.questions.parent
    group = args.bonfire
    questions = load_questions(args.questions)
    reader, described = _reader(args)
    from memory_kernel.cli import resolve_secrets

    password = resolve_secrets(env_file=args.env_file, container=args.secrets_from_container).values["neo4j_password"]
    if not password:
        raise SystemExit("no Neo4j password (NEO4J_PASSWORD, --env-file or --secrets-from-container)")
    driver = open_sync_driver(args.neo4j_uri, args.neo4j_user, password)
    print(f"runtime_eval: {len(questions)} questions, group {group!r}, reader {described}", file=sys.stderr)
    engine = RuntimeEngine(args.port, args.data_dir, out=args.engine_dir or out / "engine")

    async def run() -> list[dict[str, Any]]:
        try:
            return await run_eval(questions, args.bonfire, engine, reader, driver=driver)
        finally:
            close = getattr(reader, "close", None)
            if close is not None:
                await close()

    try:
        rows = asyncio.run(run())
    finally:
        engine.stop()
        driver.close()
    judgements = load_judgements(args.judgements)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in judged(rows, judgements):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarise(rows, judgements)
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    sys.stdout.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
