# Derived from an internal parser service, file facade_transform.py, source sha256
# 80898a2a3b5ea4aa55f850e1f34bf105f376ad7fa6502424f5da4550e8446521.
# Copied from the origin file; edits: this header, the package-relative imports and a prose-only scrub of docstrings and comments.
"""Doc -> FCG-Go parser wire-shape payloads.

Response keys here are NOT arbitrary — they are derived from how
``penelope-interface.lisp`` (the VUB client FCG-Go embeds) destructures the
JSON it gets back, via cl-json's identifier convention: a single dash in a
lisp keyword decodes an underscore in the JSON key, a DOUBLE dash decodes a
literal hyphen. So ``(assoc :noun--chunks ...)`` reads JSON key
``"noun-chunks"``, ``(assoc :tagged--words ...)`` reads ``"tagged-words"``,
``(assoc :texts--named--entities ...)`` reads ``"texts-named-entities"``,
and so on. Get any of these wrong and the lisp client's ``assoc`` silently
returns nil instead of erroring.
"""

from __future__ import annotations

from typing import Any

from .constituency import syntactic_trees
from .registry import get_benepar_pipe


def _dep_tree(tokens: Any) -> list[dict[str, Any]]:
    return [{"text": t.text, "tag": t.tag_, "dep": t.dep_, "head": t.head.text} for t in tokens]


def tokens_payload(doc: Any) -> dict[str, Any]:
    return {"tokens": [t.text for t in doc]}


def lemmas_payload(doc: Any) -> dict[str, Any]:
    return {"lemmas": [t.lemma_ for t in doc]}


def noun_chunks_payload(doc: Any) -> dict[str, Any]:
    return {"noun-chunks": [[t.text for t in chunk] for chunk in doc.noun_chunks]}


def pos_payload(doc: Any) -> dict[str, Any]:
    return {"tagged-words": [{"text": t.text, "tag": t.tag_} for t in doc]}


def named_entities_payload(doc: Any) -> dict[str, Any]:
    return {"named-entities": [{"text": e.text, "ent": e.label_} for e in doc.ents]}


def dependencies_payload(doc: Any) -> dict[str, Any]:
    return {"dependencies": [{"tree": _dep_tree(doc)}]}


def syntactic_parser_payload(doc: Any) -> dict[str, Any]:
    """Constituency+dependency structure — full parity with the hosted service's real
    wire (see ``memory_kernel.parser_service.constituency`` for the capture-derived shape:
    snake_case keys, unary-chain-collapsed ``phrase_types``, doc-global
    token offsets, per-sentence ``node_id`` restart). Benepar is attached to
    this doc lazily, on first use, via ``get_benepar_pipe()``."""

    get_benepar_pipe()(doc)
    return {"trees": syntactic_trees(doc)}


def sentences_payload(doc: Any) -> dict[str, Any]:
    return {"sentences": [s.text for s in doc.sents]}


def texts_tokens_payload(docs: list[Any]) -> dict[str, Any]:
    return {"texts-tokens": [[t.text for t in doc] for doc in docs]}


def texts_lemmas_payload(docs: list[Any]) -> dict[str, Any]:
    return {"texts-lemmas": [[t.lemma_ for t in doc] for doc in docs]}


def texts_noun_chunks_payload(docs: list[Any]) -> dict[str, Any]:
    return {
        "texts-noun-chunks": [
            [[t.text for t in chunk] for chunk in doc.noun_chunks] for doc in docs
        ]
    }


def texts_pos_tags_payload(docs: list[Any]) -> dict[str, Any]:
    return {"texts-pos-tags": [[{"text": t.text, "tag": t.tag_} for t in doc] for doc in docs]}


def texts_named_entities_payload(docs: list[Any]) -> dict[str, Any]:
    return {
        "texts-named-entities": [
            [{"text": e.text, "ent": e.label_} for e in doc.ents] for doc in docs
        ]
    }


def texts_dependencies_payload(docs: list[Any]) -> dict[str, Any]:
    # Faithfully mirrors get-penelope-texts-dependency-analysis's
    # destructuring: (mapcar (lambda (document) (mapcar ... (first document)))
    # texts--dependencies) — each per-text "document" is a singleton list
    # wrapping the list of per-sentence {"tree": ...} dicts.
    return {
        "texts-dependencies": [[[{"tree": _dep_tree(sent)} for sent in doc.sents]] for doc in docs]
    }


def texts_sentences_payload(docs: list[Any]) -> dict[str, Any]:
    return {"texts-sentences": [[s.text for s in doc.sents] for doc in docs]}


__all__ = [
    "dependencies_payload",
    "lemmas_payload",
    "named_entities_payload",
    "noun_chunks_payload",
    "pos_payload",
    "sentences_payload",
    "syntactic_parser_payload",
    "texts_dependencies_payload",
    "texts_lemmas_payload",
    "texts_named_entities_payload",
    "texts_noun_chunks_payload",
    "texts_pos_tags_payload",
    "texts_sentences_payload",
    "texts_tokens_payload",
    "tokens_payload",
]
