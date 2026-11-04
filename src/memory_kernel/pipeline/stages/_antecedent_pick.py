"""Jev picks a pronoun's antecedent among the admissible candidate NPs, or OPEN.

Ported from a reference-harness module (unreached top-level definitions pruned). Differences: pick_questions takes the
QuestionBatcher (it was a module global).

Nearest-compatible-NP resolution is about half right on real conversations (the horizon, my fam, the floor for
"its" = the cat), so a nearest pick is only a ``candidate``. Here every candidate pronoun role in the
enriched clause pairs (``clause-choice-v3``) becomes ONE finite question per (unit, pronoun):

* options ``a0``..``a2`` = up to three admissible antecedents (``antecedent_candidates``: same unit, then
  earlier units of the same message; clause-mates, the pronoun's own attachment chain, time NPs and
  predicate nominals are already excluded), each shown with its exact text;
* ``open`` = none of these / cannot be determined.

An accepted pick becomes ``{"status": "picked", ...}``; ``open`` or a pick below the Jev acceptance gate
becomes ``{"status": "open"}`` (the referent rule: a canonical node or an OPEN cell, never a guess). Expletive roles
are already bound to their clause and are not asked. Same shared USD2 round ledger.
"""

from memory_kernel.construction.memory_discourse_question_batches import question_batch_policy
from memory_kernel.construction.memory_semantic_jev import seal
from memory_kernel.pipeline.ledger import MODEL
from memory_kernel.pipeline.stages._choice import INSTRUCTIONS
from memory_kernel.pipeline.stages._referents import antecedent_candidates

COMPILED_SCHEMA = "mk-antecedent-pick/v1"

OPEN = "open"

K = 3


def pick_questions(pair, nodes_for, *, batcher):
    """One question per candidate pronoun in a derived clause pair; None when there is nothing to ask.

    ``nodes_for(unit) -> (nodes, context)`` supplies the unit's captured nodes and its referent context.
    """
    questions, targets, units = {}, {}, []
    for unit in pair["body"]["units"]:
        pronouns = sorted(
            {
                element["indices"][0]
                for frame in unit["parsed_evidence"]["native_frame_candidates"]
                for element in frame["frameElements"]
                if element.get("referent", {}).get("status") == "candidate"
            }
        )
        if not pronouns:
            continue
        nodes, context = nodes_for(unit)
        tokens = {t["index"]: t["text"] for t in unit["tokens"]}
        asked = False
        for index in pronouns:
            candidates = antecedent_candidates(index, nodes, context, k=K)
            if not candidates:
                continue
            qid = f"u{unit['unit_id']:02d}p{index:03d}"
            criteria = {f"a{i}": f'The pronoun refers to "{c["text"]}".' for i, c in enumerate(candidates)}
            criteria[OPEN] = "None of the listed phrases; the referent is something else or cannot be determined."
            questions[qid] = {
                "type": "choice",
                "criteria": criteria,
                "instructions": INSTRUCTIONS
                + f'\nIn unit {unit["unit_id"]}, which listed phrase does the pronoun "{tokens.get(index, "")}" '
                f"(token {index}) refer to? Judge from the unit's own text and the earlier message text only.",
            }
            targets[qid] = {"unit_id": unit["unit_id"], "index": index, "candidates": candidates}
            asked = True
        if asked:
            units.append(
                {
                    "unit_id": unit["unit_id"],
                    "text": unit["text"],
                    "tokens": [[t["index"], t["text"]] for t in unit["tokens"]],
                    "earlier_message_units": [c["text"] for c in context if c.get("text")],
                }
            )
    if not questions:
        return None
    state = {"pair_sha256": pair["content_sha256"], "units": units}
    wire = {"model": MODEL, "state": state, "questions": questions}
    batches = batcher.partition(wire, question_batch_policy(), round=0)
    return seal(
        {
            "schema_version": COMPILED_SCHEMA,
            "pair_sha256": pair["content_sha256"],
            "targets": targets,
            "batches": batches,
        }
    )


def read_picks(targets, answers):
    """``{unit_id: {pronoun_index: referent}}``: an accepted candidate pick, otherwise OPEN."""
    out = {}
    for qid, target in targets.items():
        answer = answers[qid]
        if not answer["accepted"]:
            referent = {"status": "open", "pick": "not_accepted"}
        elif answer["choice"] == OPEN:
            referent = {"status": "open", "pick": OPEN}
        else:
            referent = {"status": "picked", **target["candidates"][int(answer["choice"][1:])]}
        out.setdefault(target["unit_id"], {})[target["index"]] = referent
    return out
