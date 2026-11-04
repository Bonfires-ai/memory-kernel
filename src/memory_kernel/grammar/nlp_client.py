"""``doc_to_parsed`` — the one seam where a spaCy ``Doc`` becomes a ``ParsedDoc``.

The row shapes live in :mod:`memory_kernel.grammar.rows`; this module builds them
from the local spaCy pipeline's ``Doc``. Results are positional and never
deduplicated: repeated texts get repeated rows, so a downstream fold sees the
same input multiplicity it was given.
"""

from __future__ import annotations

import logging
from typing import Any

from memory_kernel.grammar.rows import (
    MORPH_FEATURES,
    EntRow,
    NounChunkRow,
    ParsedDoc,
    SentRow,
    TokenRow,
)

logger = logging.getLogger(__name__)


def doc_to_parsed(doc: Any, text: str, stamp: str = "") -> ParsedDoc:
    """Build a :class:`ParsedDoc` from a live spaCy ``Doc``.

    Public: callers that already hold a ``Doc`` from an existing parse
    site (one already-paid spaCy call — front_pass's shared cache, the live
    lane's per-turn parse, ...) convert it directly here. ``stamp`` is
    optional — most callers don't need model provenance, only the row shape.
    """

    sent_idx_by_token: dict[int, int] = {}
    sent_rows: list[SentRow] = []
    for sent_idx, sent in enumerate(getattr(doc, "sents", []) or []):
        sent_rows.append(
            SentRow(
                start_token=int(sent.start),
                end_token=int(sent.end),
                char_start=int(sent.start_char),
                char_end=int(sent.end_char),
                text=str(sent.text),
            )
        )
        for tok in sent:
            sent_idx_by_token[int(tok.i)] = sent_idx

    token_rows: list[TokenRow] = []
    for tok in doc:
        i = int(tok.i)
        morph_obj = getattr(tok, "morph", None)
        morph: dict[str, tuple[str, ...]] = {}
        for feat in MORPH_FEATURES:
            values: list[str] = []
            if morph_obj is not None:
                try:
                    values = list(morph_obj.get(feat))
                except Exception as exc:  # noqa: BLE001 — per-token spaCy morphology accessor, degrade to empty
                    logger.debug("nlp_client: morph feature %s unavailable for token %s: %s", feat, i, exc)
                    values = []
            morph[feat] = tuple(values)
        token_rows.append(
            TokenRow(
                i=i,
                text=str(tok.text),
                lemma=str(tok.lemma_),
                pos=str(tok.pos_),
                tag=str(tok.tag_),
                dep=str(tok.dep_),
                head_i=int(tok.head.i),
                char_start=int(tok.idx),
                char_end=int(tok.idx) + len(tok.text),
                sent_idx=sent_idx_by_token.get(i, 0),
                whitespace=str(tok.whitespace_),
                is_alpha=bool(tok.is_alpha),
                is_punct=bool(tok.is_punct),
                is_space=bool(tok.is_space),
                is_stop=bool(tok.is_stop),
                morph=morph,
                ent_type=str(tok.ent_type_),
            )
        )

    ent_rows = [
        EntRow(
            text=str(ent.text),
            label=str(ent.label_),
            char_start=int(ent.start_char),
            char_end=int(ent.end_char),
            start_token=int(ent.start),
            end_token=int(ent.end),
        )
        for ent in getattr(doc, "ents", []) or []
    ]

    noun_chunk_rows: list[NounChunkRow] = []
    try:
        for chunk in getattr(doc, "noun_chunks", []) or []:
            noun_chunk_rows.append(
                NounChunkRow(
                    text=str(chunk.text),
                    start_token=int(chunk.start),
                    end_token=int(chunk.end),
                    lemma=str(getattr(chunk, "lemma_", "") or ""),
                )
            )
    except Exception as exc:  # noqa: BLE001 — noun_chunks requires a parser; degrade to empty like wordnet_semantics
        logger.debug("nlp_client: noun_chunks unavailable: %s", exc)

    return ParsedDoc(
        text=text,
        tokens=tuple(token_rows),
        sents=tuple(sent_rows),
        ents=tuple(ent_rows),
        noun_chunks=tuple(noun_chunk_rows),
        stamp=stamp,
    )


__all__ = ["doc_to_parsed"]
