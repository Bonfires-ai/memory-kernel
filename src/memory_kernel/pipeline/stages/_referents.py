"""Referents for G1 role fillers: a bare third-person pronoun binds to an antecedent or opens a cell.

Ported from a reference-harness module.

The referent rule: every role binds to a canonical node or an OPEN cell, never to a bare pronoun.
Here that is decided over captured syntax only:

* expletive ``it`` (subject of a copula/predicate that has an extraposed clause after it:
  ``xcomp``/``ccomp``/``advcl``) binds to that clause in the same unit;
* otherwise (as a ``candidate``, never a verified referent) the nearest compatible NP before the pronoun in the same unit, then earlier units of the same
  message, then the previous message: plural pronouns need a plural-headed NP (``NNS``/``NNPS``), ``it``
  needs a non-PERSON, non-pronoun head;
* no compatible NP -> ``{"status": "open"}``.

First/second person is out of scope (envelope deixis); splitter stand-ins never reach this module because
their roles are stitched to clause spans, not single pronoun leaves. Pronoun classes are imported from the
person, number and gender are read from the leaf's PTB tag and ``grammar.pronouns``; gendered pronouns (she/he/her/him) and
demonstratives (that/this/these/those) are deliberately NOT resolved: they stay unresolved rather than guessed.
"""

from memory_kernel.grammar.pronouns import Pronoun, of_leaf

EXTRAPOSED = {"xcomp", "ccomp", "advcl"}

COPULAR_PREDICATE = {"acomp", "attr"}  # Predicates that mark a copula/linking verb

#: Dependency edges that enter a new clause; walks over the head chain stop at them (clause boundary).
CLAUSE_EDGES = {"conj", "ccomp", "advcl", "xcomp", "relcl", "parataxis"}

#: A predicate nominal ("Chess is [a hobby]") describes its subject; it is not a separate referent.
PREDICATE_NOMINAL = {"attr"}

#: A verb reached through these edges sits inside a prepositional phrase ("my way [of greeting]"), not in
#: extraposed-subject position.
PREPOSITIONAL_EDGES = {"prep", "pobj", "pcomp"}


def _leaves(nodes):
    return {n["start"]: n for n in nodes if n["node_type"] == "leaf"}


def _is_pronoun(leaf):
    return (leaf.get("lex_class") or "").startswith("PRP")


def _resolvable(leaf) -> Pronoun | None:
    """The leaf's pronoun when it is a third-person one with no gender ("it", "they" families), else None."""
    pronoun = of_leaf(leaf["string"], leaf.get("lex_class"))
    return pronoun if pronoun and pronoun.ungendered_third else None


def pronoun_roles(frame, nodes):
    leaves = _leaves(nodes)
    out = []
    for position, element in enumerate(frame["frameElements"]):
        if len(element["indices"]) != 1:
            continue
        leaf = leaves.get(element["indices"][0])
        if leaf is not None and _resolvable(leaf) is not None:
            out.append((position, leaf["start"]))
    return out


def _head_chain_reaches(leaf_start, target_head, leaves, by_id, max_depth=50, blocked=frozenset()):
    """Check if a leaf's head chain reaches the target head without crossing a clause boundary
    (or any ``blocked`` edge)."""
    current = leaf_start
    depth = 0
    while depth < max_depth:
        current_leaf = leaves.get(current)
        if current_leaf is None or current_leaf.get("dependency_label") in CLAUSE_EDGES | blocked:
            return False
        current_head = by_id.get(current_leaf.get("dependency_head"))
        if current_head is None:
            return False
        if current_head == target_head:
            return True
        current = current_head
        depth += 1
    return False


def _expletive(index, nodes):
    leaves = _leaves(nodes)
    by_id = {leaf["node_id"]: leaf["start"] for leaf in leaves.values()}
    # Guard: pronoun leaf missing
    if index not in leaves:
        return None
    pronoun = leaves[index]
    resolvable = _resolvable(pronoun)
    if resolvable is None or resolvable.plural or pronoun.get("dependency_label") != "nsubj":
        return None
    # Guard: head missing
    head = by_id.get(pronoun["dependency_head"])
    if head is None:
        return None
    head_leaf = leaves.get(head)
    if head_leaf is None:
        return None
    # Check if head has a copular predicate (acomp or attr dependent)
    is_copular = any(
        by_id.get(leaf["dependency_head"]) == head and leaf.get("dependency_label") in COPULAR_PREDICATE
        for leaf in leaves.values()
    )
    if not is_copular:
        return None
    # Look for EXTRAPOSED dependent of the head
    for leaf in sorted(leaves.values(), key=lambda n: n["start"]):
        if (
            leaf["start"] > head
            and by_id.get(leaf["dependency_head"]) == head
            and leaf.get("dependency_label") in EXTRAPOSED
        ):
            phrase = next(
                (
                    p
                    for p in nodes
                    if p["node_type"] == "phrase"
                    and p["start"] == leaf["start"]
                    and "VP" in (p.get("phrase_types") or ())
                ),
                None,
            )
            end = phrase["end"] if phrase else leaf["end"]
            return list(range(leaf["start"], end))
    # Copula fallback: the parser often mis-labels an extraposed gerund/infinitive ("it was so nice
    # [making tea]" -> making=compound of tea=attr). Take the first VBG or TO+VB after the copula head
    # whose chain reaches the head without a clause or prepositional edge.
    ordered = sorted(leaves.values(), key=lambda n: n["start"])
    for leaf in ordered:
        if leaf["start"] <= head:
            continue
        lex_class = leaf.get("lex_class") or ""
        is_gerund = lex_class == "VBG"
        is_to_infinitive = lex_class == "VB" and leaves.get(leaf["start"] - 1, {}).get("lex_class") == "TO"
        if (is_gerund or is_to_infinitive) and _head_chain_reaches(
            leaf["start"], head, leaves, by_id, blocked=PREPOSITIONAL_EDGES
        ):
            end = max(
                item["start"]
                for item in ordered
                if item["start"] >= leaf["start"]
                and item.get("dependency_label") != "punct"
                and _head_chain_reaches(item["start"], head, leaves, by_id)
            )
            return list(range(leaf["start"], end + 1))
    return None


def _candidates(nodes, *, before=None):
    """NP phrases (innermost-first per start) as (start, span, head leaf), latest first."""
    leaves = _leaves(nodes)
    out = []
    for phrase in nodes:
        if phrase["node_type"] != "phrase" or "NP" not in (phrase.get("phrase_types") or ()):
            continue
        if before is not None and phrase["end"] > before:
            continue
        span = list(range(phrase["start"], phrase["end"]))
        # Guard: head leaf missing
        if span[-1] not in leaves:
            continue
        head = leaves[span[-1]]
        if _is_pronoun(head):
            continue
        # Skip time NPs
        if head.get("named_entity_type") in {"DATE", "TIME"} or head.get("dependency_label") in {"npadvmod", "tmod"}:
            continue
        if head.get("dependency_label") in PREDICATE_NOMINAL:
            continue
        out.append((phrase["start"], span, head))
    out.sort(key=lambda row: (-row[0], len(row[1])))
    return out


def _compatible(plural: bool, head):
    tag = head.get("lex_class") or ""
    if plural:
        return tag in {"NNS", "NNPS"}
    return tag.startswith("NN") and tag not in {"NNS", "NNPS"} and head.get("named_entity_type") != "PERSON"


def _text(nodes, span):
    leaves = _leaves(nodes)
    return " ".join(leaves[i]["string"] for i in span)


def antecedent_candidates(index, nodes, context, *, k=3):
    """Up to ``k`` admissible antecedent NPs for the pronoun at ``index``, nearest first (same unit, then
    the context units in order). Same filters as :func:`resolve_pronoun`; the choice among them is Jev's."""
    leaves = _leaves(nodes)
    pronoun_leaf = leaves.get(index)
    if pronoun_leaf is None:
        return []
    pronoun = of_leaf(pronoun_leaf["string"], pronoun_leaf.get("lex_class"))
    plural = bool(pronoun and pronoun.plural)
    found = []
    scopes = [("same_unit", nodes, index)] + [(c["where"], c["nodes"], None) for c in context]
    for where, scope, before in scopes:
        # For same_unit scope only, compute the pronoun's attachment chain and co-arguments
        pronoun_ancestors = set()
        pronoun_co_args = set()
        if where == "same_unit":
            scope_leaves = _leaves(scope)
            by_id = {leaf["node_id"]: leaf["start"] for leaf in scope_leaves.values()}
            pronoun_scope_leaf = scope_leaves.get(index)
            if pronoun_scope_leaf is not None:
                # Build pronoun's ancestor chain (with depth limit to avoid cycles)
                current_head = by_id.get(pronoun_scope_leaf.get("dependency_head"))
                depth = 0
                while current_head is not None and depth < 50:
                    pronoun_ancestors.add(current_head)
                    current_leaf = scope_leaves.get(current_head)
                    if current_leaf is not None:
                        next_head = by_id.get(current_leaf.get("dependency_head"))
                        if next_head == current_head:  # Self-loop, stop
                            break
                        current_head = next_head
                    else:
                        break
                    depth += 1
                # Clause-mates: dependents of any head on the pronoun's chain inside its own clause
                # (the chain stops at the first clause-entering edge), e.g. "The color adds a lot to it".
                clause_heads = set()
                current = pronoun_scope_leaf
                depth = 0
                while current is not None and current.get("dependency_label") not in CLAUSE_EDGES and depth < 50:
                    governor = by_id.get(current.get("dependency_head"))
                    if governor is None or governor in clause_heads or governor == current["start"]:
                        break
                    clause_heads.add(governor)
                    current = scope_leaves.get(governor)
                    depth += 1
                for leaf in scope_leaves.values():
                    if by_id.get(leaf.get("dependency_head")) in clause_heads:
                        pronoun_co_args.add(leaf["start"])
        for _, span, head in _candidates(scope, before=before):
            if _compatible(plural, head):
                # For same_unit scope, skip clause-mates and ancestors
                if where == "same_unit" and (head["start"] in pronoun_co_args or head["start"] in pronoun_ancestors):
                    continue
                found.append({"where": where, "text": _text(scope, span), "span": span})
                if len(found) == k:
                    return found
    return found


def resolve_pronoun(index, nodes, context):
    clause = _expletive(index, nodes)
    if clause is not None:
        return {"status": "expletive", "span": clause}
    # Nearest-compatible-NP is a candidate, not a verified referent (the referent rule: canonical node or OPEN); Jev
    # picks among antecedent_candidates to confirm one.
    found = antecedent_candidates(index, nodes, context, k=1)
    return {"status": "candidate", **found[0]} if found else {"status": "open"}
