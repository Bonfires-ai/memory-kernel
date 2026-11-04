"""Time role for G1: bind DATE/TIME expressions to the event they modify, as ``argm-tmp``.

Ported from a reference-harness module.

Stock frames carry no modifier roles, so a "when" answer either has no role or is mis-bound (``take.01``
arg0 = "Yesterday"). A time span is a maximal run of DATE/TIME-tagged leaves whose head (the run's leaf
whose dependency head lies outside the run) is nominal or adverbial; determiner/possessive-only runs are
not spans. A span attaches to the frame whose anchor is the nearest ancestor of the span head on the
``dependency_head`` chain, within the segment's tokens when given. A core role whose indices equal the
span exactly is relabelled (the stock misbinding), recorded as ``relabelled_from``. No word lists:
spans come from the parser's entity tags and dependency structure only.
"""

from copy import deepcopy

TIME_TAGS = {"DATE", "TIME"}

HEAD_CLASSES = ("NN", "RB", "CD")

# dependency roles a time expression takes as a modifier; a dobj/attr/nsubj DATE phrase is a core argument
TEMPORAL_DEPS = {"npadvmod", "tmod", "advmod", "pobj"}


def _leaves(nodes):
    return sorted((n for n in nodes if n["node_type"] == "leaf"), key=lambda n: n["start"])


def time_spans(nodes):
    leaves = _leaves(nodes)
    by_id = {leaf["node_id"]: leaf["start"] for leaf in leaves}
    runs, current = [], []
    for leaf in leaves:
        if leaf.get("named_entity_type") in TIME_TAGS:
            if current and leaf["start"] != current[-1]["start"] + 1:
                runs.append(current)
                current = []
            current.append(leaf)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    spans = []
    for run in runs:
        indices = {leaf["start"] for leaf in run}
        heads = [leaf for leaf in run if by_id.get(leaf["dependency_head"]) not in indices]
        if any(
            (leaf.get("lex_class") or "").startswith(HEAD_CLASSES) and leaf.get("dependency_label") in TEMPORAL_DEPS
            for leaf in heads
        ):
            spans.append(sorted(indices))
    return spans


def _ancestors(start, nodes):
    leaves = {leaf["start"]: leaf for leaf in _leaves(nodes)}
    by_id = {leaf["node_id"]: leaf["start"] for leaf in leaves.values()}
    seen, current = [], start
    while current in leaves and current not in seen:
        seen.append(current)
        parent = by_id.get(leaves[current]["dependency_head"])
        if parent is None or parent == current:
            break
        current = parent
    return seen[1:]


def attach_time(frames, nodes, spans, *, segment_tokens=None):
    out = deepcopy(frames)
    anchors = {}
    for position, frame in enumerate(out):
        for index in frame["frameEvokingElement"]["indices"]:
            anchors.setdefault(index, position)
    by_id = {leaf["node_id"]: leaf["start"] for leaf in _leaves(nodes)}
    leaves = {leaf["start"]: leaf for leaf in _leaves(nodes)}
    for span in spans:
        if segment_tokens is not None and not set(span) <= set(segment_tokens):
            continue
        head = next(i for i in span if by_id.get(leaves[i]["dependency_head"]) not in set(span))
        target = None
        for ancestor in _ancestors(head, nodes):
            if segment_tokens is not None and ancestor not in segment_tokens:
                break
            if ancestor in anchors:
                target = anchors[ancestor]
                break
        if target is None:
            continue
        frame = out[target]
        relabel = next((e for e in frame["frameElements"] if sorted(e["indices"]) == span), None)
        if relabel is not None:
            relabel["relabelled_from"] = relabel["feRole"]
            relabel["feRole"] = relabel["feName"] = "argm-tmp"
        else:
            frame["frameElements"].append(
                {"feRole": "argm-tmp", "feName": "argm-tmp", "indices": span, "time_role": "attached"}
            )
    return out
