"""THE CLOSED-CLASS ANCHOR FENCE — a phrasal card may not be anchored on a function word.

The rule is to cull the function-word cards before they are promoted. The staging grammar carried phrasal
``np.*`` cards anchored on closed-class tokens — ``np.the`` (attested in **2930 of 4593** statements of the production corpus,
63.8% of the corpus), ``np.he``, ``np.my``, ``np.our``, ``np.do``, ``np.cannot``. Their
referent value is ~zero (a determiner names nothing) and their comprehension cost is
~maximal: the anchor token occurs in nearly every sentence, so the card widens EVERY
token's candidate bucket on EVERY statement. A measurement of a 22-word statement already
exhausting the 5 s comprehend budget at 264 cards; the function-word cards are the cheapest
cards to delete and the most expensive to keep.

A cull without a gate regrows: the live tick RE-MINTED ``np.the`` on the night of the cull.
This module is the rule; the minting service's ``FUNCTION_WORD_ANCHOR_REASON``
is its refusal cause at ``validate_mint_cards``.

WHAT IS REFUSED
---------------
A **single-anchor phrasal card** — ``layer == "phrasal"`` with exactly one ``terms`` entry,
which is the shape every ``np.<lemma>`` card has — whose anchor form is closed-class.
Nothing else. Clause cards are untouched (their single term is the head verb, and a verb
lemma that happens to be closed-class, ``do``/``have``, is a legitimate predicate — this
gate has no opinion about it). Content-word phrasal cards are untouched: ``np.orange``,
``np.judge``, ``np.consultant``, ``np.name``, ``np.system``, ``np.side``, ``np.call``,
``np.help`` all ride, and the last four matter because a *stopword* list would have eaten
them (see the generation note).

WHY A FROZEN CONSTANT AND NOT spaCy AT THE GATE
-----------------------------------------------
The rule is stated over POS — an anchor whose part of speech is in
the file's ``closed_class_pos`` — but the gate does not, and must not, run a tagger. Three
reasons, in order of weight:

1. **POS is a token-level, context-dependent property; a card's anchor is a bare form with
   no context.** Asking spaCy in isolation gives the WRONG answer often enough to matter:
   ``nlp("the")[0].pos_`` is ``PRON`` (it is a ``DET``), ``nlp("one")[0].pos_`` is ``NUM``,
   ``nlp("do")[0].pos_`` is ``VERB`` (it is ``AUX`` 88% of the time in this corpus). The
   type-level POS of a form is only recoverable from a distribution over attested tokens —
   which is a one-time offline computation, not a per-mint one. A gate that tagged the bare
   lemma would be *principled in form and wrong in fact*.
2. **The minting service's ``validate_mint_cards`` → ``_first_mint_failure`` is PURE**, and
   is re-run offline by the replay/assembly drivers.
   Loading a model inside it would make the same batch judgeable differently after a spaCy
   model bump — silently, with no version in any artifact.
3. **Cost.** The gate runs to a fixpoint (O(n) rounds x n cards) once per topic per tick,
   on the very path whose latency this cull exists to protect.

Tradeoff accepted: the anchor set is a SNAPSHOT (``data/closed-class-anchors.json``) and widens only
when someone regenerates it. English closed classes are closed, so the set is near-static by construction;
``scripts/derive_closed_class_anchors.py`` regenerates it for any model.

GENERATION — how the set was derived
------------------------------------
``scripts/derive_closed_class_anchors.py --model <spacy model> --corpus FILE...`` implements the recipe (a review tool);
its ``CLOSED_CLASS_POS`` is ``ADP AUX CCONJ DET PART PRON SCONJ``. Two sources, UNIONED:

* **Source A, the lexicon.** Every alphabetic word of ``spacy.lang.en`` ``STOP_WORDS`` union ``TOKENIZER_EXCEPTIONS`` keys
  is tagged in isolation; closed iff EVERY token it splits into has a closed-class POS (``cannot`` = ``can``/``not``).
  Open-class stopwords (``call``, ``name``, ``side``, ``give``) drop out as NOUN/VERB.
* **Source B, attestation.** A form is closed iff its MODAL POS over at least 3 attested tokens is closed-class.

``data/closed-class-anchors.json`` is the set derived 2026-08-05 (spaCy 3.8.14, ``en_core_web_sm`` 3.8.0, a 4593-statement
corpus that is not available): 187 forms, with provenance. The anchors are a property of English, not of the tagger, so
one set serves every parser; a derivation under another model is evidence for a human review of this set.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

ANCHORS_FILE = ("data", "closed-class-anchors.json")


@lru_cache(maxsize=1)
def closed_class_anchor_forms() -> frozenset[str]:
    """The lowercased closed-class word FORMS (not lemmas: a card's ``terms`` carry surface forms, so ``was``/``does``/
    ``their`` are present in their own right): ``data/closed-class-anchors.json``. One set serves every parser."""

    text = resources.files("memory_kernel").joinpath(*ANCHORS_FILE).read_text(encoding="utf-8")
    return frozenset(str(form).lower() for form in json.loads(text)["forms"])
