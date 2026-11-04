"""The stand-in clause segmenter, owned in src (the fallback for clause units).

``segment_unit`` is byte-for-byte the frozen reference-harness segmenter (pinned by a
parity test over all 1,449 units of the development conversation). ``pass_segments`` runs the same segmenter over ONE fenced pass's
subtree for the hybrid fallback: a fence collapsed in that pass renders as the ``it`` stand-in and maps back
to its whole span; charmap entries are ``[char_start, char_end, [original token indices]]``.
``map_segment_frames`` is ``_map_frames`` over that charmap, emitting stock-shape frames on original tokens.
"""

PLACEHOLDER = "it"
LEAD = "It"
CLAUSE = {"S", "SBAR", "SINV", "SQ", "SBARQ"}
AUX = {"aux", "auxpass"}


class StandInTree:
    def __init__(self, nodes):
        self.leaves = sorted((n for n in nodes if n["node_type"] == "leaf"), key=lambda n: n["start"])
        self.phrases = {n["node_id"]: n for n in nodes if n["node_type"] == "phrase"}
        self.children = {}
        for n in nodes:
            self.children.setdefault(n["parent_constituent"], []).append(n)
        for kids in self.children.values():
            kids.sort(key=lambda n: (n["start"], n["node_type"] != "phrase"))
        self.verb = {
            leaf["start"]
            for leaf in self.leaves
            if (leaf.get("lex_class") or "").startswith("VB") and leaf.get("dependency_label") not in AUX
        }
        self.punct = {leaf["start"] for leaf in self.leaves if leaf.get("dependency_label") == "punct"}

    def span(self, node):
        return list(range(node["start"], node["end"])) if node["node_type"] == "phrase" else [node["start"]]

    def verbs(self, tokens):
        return sum(t in self.verb for t in tokens)

    def types(self, node):
        return set(node.get("phrase_types") or ())

    def descendants(self, node):
        for kid in self.children.get(node["node_id"], ()):
            if kid["node_type"] == "phrase":
                yield kid
                yield from self.descendants(kid)


def noun_attached(tree, clause):
    """A clause under an NP ("courage [to resign]", "the help [I've got]") modifies the noun and fills
    no verb role: it still splits off (one verb per segment), but the matrix gets no stand-in. Every other
    split clause fills a verb role and leaves "it" behind ("that" as a stand-in parses empty: clauses-003,
    20/111 empty vs 1/180 for "it")."""
    parent = tree.phrases.get(clause["parent_constituent"])
    return parent is not None and "NP" in tree.types(parent)


def segment(tree, node, prefix, lead=None):
    """Segments over ``node``. ``prefix`` = inherited subject tokens; ``lead`` = the original span of a
    clausal subject that this node's segments open with a pronoun placeholder for."""
    tokens = tree.span(node)

    def seg(toks, **extra):
        return {"tokens": toks, "lead": lead, **extra}

    if tree.verbs(tokens) <= 1:
        return [seg(prefix + tokens)]
    kids = tree.children.get(node["node_id"], [])
    phrases = [k for k in kids if k["node_type"] == "phrase"]
    clauses = [k for k in phrases if tree.types(k) & CLAUSE and tree.verbs(tree.span(k))]
    if len(clauses) >= 2:
        out = [s for k in clauses for s in segment(tree, k, [])]
        out[0]["tokens"] = prefix + out[0]["tokens"]
        out[0]["lead"] = lead
        return out
    vps = [k for k in phrases if "VP" in tree.types(k) and tree.verbs(tree.span(k))]
    coordinated = any(
        k["node_type"] == "leaf" and (k.get("lex_class") == "CC" or k.get("dependency_label") == "cc") for k in kids
    )
    if len(vps) >= 2 and coordinated:
        return [s for k in vps for s in segment(tree, k, list(prefix), lead)]
    holders = [k for k in phrases if tree.verbs(tree.span(k)) == tree.verbs(tokens)]
    if len(holders) == 1:
        holder = holders[0]
        pre = [t for t in tokens if t < holder["start"]]
        post = [t for t in tokens if t >= holder["end"] and t not in tree.punct]
        out = segment(tree, holder, prefix + pre, lead)
        out[-1]["tokens"] = out[-1]["tokens"] + post
        return out
    # clausal subject: "[Walking with friends] is relaxing ..." -> the subject clause, then the
    # predicate opened by a pronoun placeholder that stitches back to the subject clause.
    if clauses and clauses[0]["start"] == node["start"] and not prefix and lead is None:
        subject = clauses[0]
        rest = [k for k in phrases if k["start"] >= subject["end"] and tree.verbs(tree.span(k))]
        remaining = tree.verbs([t for t in tokens if t >= subject["end"]])
        if len(rest) == 1 and remaining and tree.verbs(tree.span(rest[0])) == remaining:
            post = [t for t in tokens if t >= rest[0]["end"] and t not in tree.punct]
            tail = segment(tree, rest[0], [], [subject["start"], subject["end"]])
            tail[-1]["tokens"] = tail[-1]["tokens"] + post
            return [*segment(tree, subject, []), *tail]
    content = [t for t in tokens if t not in tree.punct]
    end = content[-1] + 1 if content else node["end"]
    suffixes = [
        d
        for d in tree.descendants(node)
        if tree.types(d) & CLAUSE
        and d["start"] > node["start"]
        and d["end"] >= end
        and tree.verbs(tree.span(d))
        and tree.verbs([t for t in tokens if t < d["start"]])
    ]
    if suffixes:
        clause = min(suffixes, key=lambda d: d["start"])
        head = prefix + [t for t in tokens if t < clause["start"]]
        if noun_attached(tree, clause):
            matrix = seg(head, stand_in=None)
        else:
            matrix = seg(head, placeholder=[clause["start"], clause["end"]], stand_in=PLACEHOLDER)
        return [matrix, *segment(tree, clause, [])]
    return [seg(prefix + tokens)]


def leaf_offsets(text, leaves):
    """Char span of each leaf in the unit text (sequential find; None when the parser normalised it)."""
    out, cursor = {}, 0
    for leaf in leaves:
        at = text.find(leaf["string"], cursor)
        if at < 0:
            out[leaf["start"]] = None
            continue
        out[leaf["start"]] = (at, at + len(leaf["string"]))
        cursor = at + len(leaf["string"])
    return out


# --- run ------------------------------------------------------------------------------------------


def pass_segments(pass_nodes, full_nodes, text, collapsed_spans):
    """Stand-in segments over one pass tree; a collapsed fence renders as ``it`` -> its whole span.

    Returns ``[{"text", "charmap", "verbs"}]`` (always >= 1 segment: an unsplittable pass is one segment)."""

    tree = StandInTree(pass_nodes)
    root = next(n for n in pass_nodes if n.get("parent_constituent") is None)
    raw = segment(tree, root, []) if root["node_type"] == "phrase" else [{"tokens": [root["start"]], "lead": None}]
    leaves = sorted((n for n in full_nodes if n["node_type"] == "leaf"), key=lambda n: n["start"])
    offsets = leaf_offsets(text, leaves)
    own = {n["start"] for n in pass_nodes if n["node_type"] == "leaf"} - {
        t for lo, hi in collapsed_spans for t in range(lo, hi)
    }
    out = []
    for part in raw:
        pieces, charmap, previous_literal = [], [], None

        def emit(piece, targets, gap=" ", pieces=pieces, charmap=charmap):
            if pieces:
                pieces.append(gap)
            at = sum(map(len, pieces))
            pieces.append(piece)
            charmap.append([at, at + len(piece), sorted(targets)])

        if part.get("lead"):
            emit(LEAD, range(*part["lead"]))
        done_fences = set()
        for token in sorted(set(part["tokens"])):
            fence = next(((lo, hi) for lo, hi in collapsed_spans if lo <= token < hi), None)
            if fence is not None:
                if fence not in done_fences:
                    done_fences.add(fence)
                    emit(PLACEHOLDER, range(*fence))
                    previous_literal = None
                continue
            if token not in own or offsets.get(token) is None:
                continue
            start, end = offsets[token]
            gap = " "
            if previous_literal is not None and token == previous_literal + 1:
                gap = text[offsets[previous_literal][1] : start]
            emit(text[start:end], [token], gap)
            previous_literal = token
        if part.get("placeholder"):
            emit(part.get("stand_in") or PLACEHOLDER, range(*part["placeholder"]))
        seg_text = "".join(pieces).strip()
        if seg_text and any(len(c[2]) == 1 and c[2][0] in own for c in charmap):
            lead_pad = len("".join(pieces)) - len("".join(pieces).lstrip())
            if lead_pad:
                charmap = [[a - lead_pad, b - lead_pad, t] for a, b, t in charmap]
            out.append({"text": seg_text, "charmap": charmap, "verbs": tree.verbs(part["tokens"])})
    return out


def map_segment_frames(segment, jobs, full_nodes, *, pass_id, number):
    """Port of the reference harness's frame mapper over the generalised charmap -> stock-shape frames on the
    unit's original tokens (ids ``<pass>-sNN-<frame id>``)."""

    leaves = {n["start"]: n for n in full_nodes if n["node_type"] == "leaf"}

    def orig(chars):
        out = set()
        for start, end, targets in segment["charmap"]:
            if start < chars[1] and chars[0] < end:
                out |= set(targets)
        return out

    frames = []
    for job in jobs:
        record = job["record"]
        if record["capture_status"] != "framed":
            continue
        unit_text, offset = record["job"]["source"]["text"], record["job"]["span"][0]
        seg_leaves = sorted((n for n in record["job"]["nodes"] if n["node_type"] == "leaf"), key=lambda n: n["start"])
        spans = {k: (v[0] + offset, v[1] + offset) for k, v in leaf_offsets(unit_text, seg_leaves).items() if v}

        def tokens(indices, spans=spans):
            return sorted(set().union(*(orig(spans[i]) for i in indices if i in spans))) if indices else []

        for frame in record["response"]["payload"]["frames"]:
            anchor = tokens(frame["frameEvokingElement"]["indices"])
            if not anchor:
                continue
            elements = []
            for fe in frame.get("frameElements") or []:
                span = tokens(fe["indices"])
                if span:
                    elements.append(
                        {
                            "feName": fe["feName"],
                            "feRole": fe.get("feRole", fe["feName"]),
                            "indices": span,
                            "feString": " ".join(leaves[i]["string"] for i in span if i in leaves),
                            "feLemmas": [leaves[i].get("lemma") for i in span if i in leaves],
                        }
                    )
            frames.append(
                {
                    "id": f"{pass_id}-s{number:02d}-{frame['id']}",
                    "frameName": frame["frameName"],
                    "frameEvokingElement": {
                        "indices": anchor,
                        "lemma": leaves[anchor[0]].get("lemma") if anchor[0] in leaves else None,
                        "felString": " ".join(leaves[i]["string"] for i in anchor if i in leaves),
                    },
                    "frameElements": elements,
                }
            )
    return frames
