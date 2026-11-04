"""Derive the closed-class anchor forms for one spaCy model. A REVIEW TOOL: its output is evidence for a human review of
the single set in ``data/closed-class-anchors.json`` (the anchors are a property of English, so one set serves every
parser); the kernel never loads a per-model file.

Usage (from the repository root, ``PYTHONPATH=src``, the project venv's python):
    python scripts/derive_closed_class_anchors.py --model en_core_web_sm [--corpus FILE ...] [--min-attested 3]
        [--overrides FILE] [--out FILE] [--report FILE]

Two sources, unioned; both are spaCy's own judgement, neither is hand-listed.

Source A, the lexicon (corpus-independent). Every alphabetic word in ``spacy.lang.en.stop_words.STOP_WORDS`` union the
keys of ``spacy.lang.en.tokenizer_exceptions.TOKENIZER_EXCEPTIONS`` is tagged in isolation. A candidate is closed iff
EVERY token spaCy splits it into carries a POS in ``CLOSED_CLASS_POS`` (the all-pieces rule admits "cannot" =
"can" + "not"). Open-class stopwords ("call", "name", "side", "give", ...) fall out as NOUN or VERB.

Source B, attestation (a corpus; it decides the ambiguous forms). ``--corpus`` files are plain text (one text per
non-empty line) or JSONL (``.jsonl``/``.ndjson``; each row an object with a ``text`` field, or a bare string). Every
lowercased alphabetic token FORM has its POS distribution recorded; a form is closed iff its MODAL POS is in
``CLOSED_CLASS_POS`` over at least ``--min-attested`` tokens.

``--overrides``: a JSON ``{"closed": [forms], "open": [forms]}`` applied last (``open`` wins over a source; ``closed``
adds). Forms are lowercased. The output records them.

``--out``: the data file (default: stdout). ``--report``: every form's evidence (the isolated Source A tags and the
Source B distributions) for diffing two models.

Output: ``{"schema", "spacy", "model", "model_version", "date", "closed_class_pos", "min_attested", "recipe", "sources",
"overrides", "forms"}``; ``forms`` is the sorted union."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: The universal POS tags that are closed classes. Chosen so Source A on ``en_core_web_sm`` reproduces the shipped
#: lexicon's 163-of-468 split (see the closed-class-anchors data file's provenance).
CLOSED_CLASS_POS = ("ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ")

SCHEMA = "mk-closed-class-anchors/v1"

RECIPE = (
    "Source A: alphabetic entries of spacy.lang.en STOP_WORDS union TOKENIZER_EXCEPTIONS keys, each tagged in isolation; "
    "closed iff every token's POS is in closed_class_pos. "
    "Source B: corpus attestation; closed iff the modal POS of a lowercased token form is in closed_class_pos over at "
    "least min_attested tokens. forms = A union B, then overrides (open wins)."
)


def lexicon_candidates() -> list[str]:
    from spacy.lang.en.stop_words import STOP_WORDS
    from spacy.lang.en.tokenizer_exceptions import TOKENIZER_EXCEPTIONS

    return sorted({w.lower() for w in set(STOP_WORDS) | set(TOKENIZER_EXCEPTIONS) if w.isalpha()})


def source_a(nlp, pos_set) -> tuple[set[str], list[str], dict[str, list[str]]]:
    candidates = lexicon_candidates()
    tags: dict[str, list[str]] = {}
    for word, doc in zip(candidates, nlp.pipe(candidates, batch_size=64), strict=True):
        tags[word] = [t.pos_ for t in doc]
    closed = {w for w, tg in tags.items() if tg and all(p in pos_set for p in tg)}
    return closed, candidates, tags


def read_corpus(paths: list[str]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        p = Path(path)
        jsonl = p.suffix in {".jsonl", ".ndjson"}
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if jsonl:
                row = json.loads(line)
                line = row if isinstance(row, str) else str(row.get("text") or "")
            if line.strip():
                texts.append(line.strip())
    return texts


def source_b(nlp, texts: list[str], pos_set, min_attested: int) -> tuple[set[str], dict[str, Counter], int]:
    dist: dict[str, Counter] = defaultdict(Counter)
    tokens = 0
    for doc in nlp.pipe(texts, batch_size=32):
        for tok in doc:
            form = tok.text.lower()
            if form.isalpha():
                dist[form][tok.pos_] += 1
                tokens += 1
    closed = set()
    for form, counts in dist.items():
        if sum(counts.values()) >= min_attested and counts.most_common(1)[0][0] in pos_set:
            closed.add(form)
    return closed, dist, tokens


def derive(nlp, model: str, texts: list[str], min_attested: int, overrides: dict, pos=CLOSED_CLASS_POS):
    import spacy

    pos_set = set(pos)
    a, candidates, a_tags = source_a(nlp, pos_set)
    b, dist, tokens = source_b(nlp, texts, pos_set, min_attested) if texts else (set(), {}, 0)
    force_closed = {str(f).lower() for f in overrides.get("closed", [])}
    force_open = {str(f).lower() for f in overrides.get("open", [])}
    forms = ((a | b) | force_closed) - force_open
    out = {
        "schema": SCHEMA,
        "spacy": spacy.__version__,
        "model": model,
        "model_version": str(nlp.meta.get("version", "")),
        "date": datetime.date.today().isoformat(),
        "closed_class_pos": sorted(pos_set),
        "min_attested": min_attested,
        "recipe": RECIPE,
        "sources": {
            "a_candidates": len(candidates),
            "a_admitted": len(a),
            "b_texts": len(texts),
            "b_tokens": tokens,
            "b_forms": len(dist),
            "b_admitted": len(b),
        },
        "overrides": {"closed": sorted(force_closed), "open": sorted(force_open)},
        "forms": sorted(forms),
    }
    report = {
        "a_tags": a_tags,
        "a_admitted": sorted(a),
        "b_admitted": sorted(b),
        "b_distributions": {f: dict(c) for f, c in dist.items()},
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", nargs="*", default=[], metavar="FILE")
    ap.add_argument("--min-attested", type=int, default=3)
    ap.add_argument("--overrides", help='JSON {"closed": [...], "open": [...]}')
    ap.add_argument("--out")
    ap.add_argument("--report")
    args = ap.parse_args(argv)

    import spacy

    nlp = spacy.load(args.model)
    overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8")) if args.overrides else {}
    out, report = derive(nlp, args.model, read_corpus(args.corpus), args.min_attested, overrides)
    text = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
