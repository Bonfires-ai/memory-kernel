# Derived from an internal parser service, file constituency.py, source sha256
# da3c214f1f4de3244e374685f83e2ead6a08498cf125407223cf67687e526484.
# Copied from the origin file; edits: this header, the package-relative imports and a prose-only scrub of docstrings and comments.
"""Benepar constituency + spaCy dependency -> FCG-Go ``syntactic-parser`` wire.

This is the constituency half of the payload. The shape here is NOT guessed —
it was captured directly from the hosted parser service's ``/spacy-api/
syntactic-parser`` endpoint (30 fixtures drawn from development-conversation statements) and
cross-checked against ``pb-de-render.lisp`` (``fcg-propbank``'s
``create-initial-transient-structure-based-on-benepar-analysis`` and its
``node-*`` accessors), which is the actual consumer.

Two things the accessor names alone would NOT have told us — confirmed only
by capturing the live wire:

1. **Keys are snake_case** (``node_id``, ``node_type``, ``phrase_types``,
   ``parent_constituent``, ``lex_class``, ``dependency_label``,
   ``dependency_head``, ``named_entity_type``), not camelCase. The
   ``pb-de-render.lisp`` ``de-render`` method rebinds cl-json's decoder to
   ``camel-case-to-lisp`` before calling ``get-penelope-syntactic-analysis``
   (line 18-20), which reads as "expect camelCase" — but cl-json's
   camel-case-to-lisp reader treats an already-lower-cased, underscore-
   separated key exactly the same as camelCase (both collapse to the same
   upcased-with-single-dash keyword), so the accessors' double-dash keyword
   spelling (``:node--type`` etc.) is consistent with EITHER convention.
   Only the live capture disambiguates, and it says snake_case.
2. **``named_entity_type`` exists on every leaf** (spaCy's ``ent_type_``,
   ``""`` when the token isn't part of a named entity) — not documented
   anywhere upstream of the fixtures. It is not read by the accessors above
   (fcg-propbank doesn't consume it today) but it is part of the real wire,
   so it is part of parity.

Unary-chain collapsing (the "phrase_types is a list" design)
--------------------------------------------------------------
The wire's tree is NOT benepar's raw constituency tree. Benepar (like any
PTB-style parser) produces genuine unary productions -- e.g. a one-word
subject "April" parses as ``(NP (NNP April))``: a real NP node with exactly
one child, a preterminal NNP, with exactly one child, the word. The
wire collapses every such single-child run into ONE node:

- Climb from a node upward^Wdownward while the current node has exactly one
  child and that child is not the sentence-final branching point.
- If the run bottoms out at a preterminal (a node whose only child is the
  word itself), emit a ``"leaf"``: ``lex_class`` = the preterminal's POS
  tag, ``phrase_types`` = the labels collected on the way down (outermost
  first), EXCLUDING the preterminal's own tag.
- If the run bottoms out at a branching node (>1 children) or the word IS
  directly branching's child, emit a ``"phrase"``: ``phrase_types`` = the
  labels collected on the way down INCLUDING the branching node's own
  label, children = the branching node's actual children (each collapsed
  the same way).

Confirmed against fixtures: "is" (VBZ, direct child of a branching VP) gets
``phrase_types: []``; "April" (NNP under a singleton NP under branching S)
gets ``phrase_types: ["NP"]``; deeper stacks do occur (``["SBAR", "S"]``,
``["S", "VP"]``, even doubled ``["NP", "NP"]``) and the list is always
outermost-label-first.

``node_id`` and offsets
------------------------
- ``node_id`` is assigned by preorder DFS over the COLLAPSED tree (visit a
  node, assign the next id, then recurse into children left-to-right) and
  **restarts at 1 for every sentence** -- confirmed against a genuine
  two-sentence capture. It is used purely relationally by the consumer
  (parent/child and dependency-head pointers), never compared by absolute
  value.
- ``start``/``end`` are 0-based, half-open TOKEN offsets, and unlike
  ``node_id`` they do NOT reset per sentence -- they are document-global
  token indices, confirmed by the same two-sentence capture (sentence 2's
  nodes start at 5, not 0).
- The root's ``parent_constituent`` is ``null``. A leaf's
  ``dependency_head`` is the node id of its syntactic head's leaf node --
  for the ROOT token this is ITS OWN node id, never null/absent (spaCy's
  own convention: ``token.head is token`` for the sentence root).

One confirmed, deliberate divergence from the live server (see
the origin service's parity-test docstring and benepar
report for the full justification): a genuine multi-sentence
capture showed dependency/lemma/tag/NER fields present ONLY on the first
sentence's leaves -- every later sentence's leaves silently drop them. This
looks like a bug in the upstream service, not a documented contract; the
Lisp caller (``run-penelope-syntactic-parser``) is only ever invoked with a
single already-split sentence, so the bug is never exercised in practice.
We emit full leaf data for every sentence in the input rather than
reproducing the omission.
"""

from __future__ import annotations

from typing import Any

import nltk


def _collapse(
    tree: nltk.Tree,
    tokens: list[Any],
    doc: Any,
    tok_idx: list[int],
    nodes: list[dict[str, Any]],
    node_id: list[int],
    parent_id: int | None,
) -> dict[str, Any]:
    """Collapse one unary run starting at ``tree`` into one node dict.

    ``tok_idx`` is a one-element mutable counter into ``tokens`` (the
    sentence's tokens in order) -- leaves are consumed left-to-right as the
    (collapsed) tree is walked, which is guaranteed to match benepar's own
    leaf order since it parses over the existing spaCy tokenization.

    Offsets use ``token.i`` directly -- spaCy's token index is already
    document-global, which is exactly the "start/end don't reset per
    sentence" behavior confirmed against a live two-sentence capture (see
    module docstring). Only ``node_id`` resets per sentence (handled by the
    caller re-seeding ``node_id`` to 1 for each sentence).
    """

    labels: list[str] = []
    cur = tree
    while len(cur) == 1 and not isinstance(cur[0], str):
        labels.append(cur.label())
        cur = cur[0]

    my_id = node_id[0]
    node_id[0] += 1

    if len(cur) == 1 and isinstance(cur[0], str):
        # Preterminal: cur.label() is the POS tag (lex_class); cur[0] is the
        # word. Consume the aligned spaCy token for the rest of the leaf data.
        token = tokens[tok_idx[0]]
        tok_idx[0] += 1
        node = {
            "node_id": my_id,
            "node_type": "leaf",
            "phrase_types": list(labels),
            "start": token.i,
            "end": token.i + 1,
            "string": token.text,
            "parent_constituent": parent_id,
            "_token": token,  # resolved to dependency_head + stripped below
        }
        nodes.append(node)
        return node

    labels.append(cur.label())
    node = {
        "node_id": my_id,
        "node_type": "phrase",
        "phrase_types": labels,
        "start": None,
        "end": None,
        "string": None,
        "parent_constituent": parent_id,
    }
    nodes.append(node)
    children = [_collapse(child, tokens, doc, tok_idx, nodes, node_id, my_id) for child in cur]
    node["start"] = children[0]["start"]
    node["end"] = children[-1]["end"]
    # NOT a naive space-join of children's own strings (that loses original
    # spacing -- no space before punctuation, contractions like "y'all"
    # get split apart) -- confirmed against fixtures: slice the ORIGINAL
    # doc text for this node's token span, exactly like a leaf's "string"
    # is the token's own (correctly spelled) text.
    node["string"] = doc[node["start"] : node["end"]].text
    return node


def sentence_tree(sent: Any) -> list[dict[str, Any]]:
    """Collapsed constituency+dependency node list for one spaCy sentence Span.

    ``node_id`` restarts at 1; ``start``/``end`` use spaCy's document-global
    ``token.i`` and are NOT sentence-relative (see module docstring).
    """

    parse_string = sent._.parse_string
    tree = nltk.Tree.fromstring(parse_string)
    tokens = list(sent)

    nodes: list[dict[str, Any]] = []
    tok_idx = [0]
    node_id = [1]
    _collapse(tree, tokens, sent.doc, tok_idx, nodes, node_id, None)

    # Resolve dependency_head (needs every leaf's node_id known first) and
    # attach the remaining per-token leaf fields, then drop the scratch token ref.
    token_to_node_id = {n["_token"].i: n["node_id"] for n in nodes if n["node_type"] == "leaf"}
    for n in nodes:
        if n["node_type"] != "leaf":
            continue
        token = n.pop("_token")
        n["lemma"] = token.lemma_
        n["lex_class"] = token.tag_
        n["dependency_label"] = token.dep_
        n["dependency_head"] = token_to_node_id[token.head.i]
        n["named_entity_type"] = token.ent_type_

    return nodes


def syntactic_trees(doc: Any) -> list[dict[str, Any]]:
    """``{"trees": [...]}``-ready list: one ``{sent_id, tree}`` entry per sentence."""

    return [{"sent_id": i, "tree": sentence_tree(sent)} for i, sent in enumerate(doc.sents)]


__all__ = ["sentence_tree", "syntactic_trees"]
