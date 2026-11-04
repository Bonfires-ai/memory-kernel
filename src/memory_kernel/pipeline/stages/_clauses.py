"""Clause-split re-parse of serving-budget cutoffs: give the stock parser one clause at a time.

Ported from a reference-harness module (unreached top-level definitions pruned). What is left are the two helpers that map
segment frames back to the unit's original token positions.

Cutoffs track predicate count (1 verb: 2 % cut off, 3 verbs: 55 %, 5+: 93 %), not lexical gaps: each
verb's argument-structure cxns compete over the same phrases. This splits a cutoff unit over its OWN
captured syntax tree, then re-parses each segment at the serving budget (5 s) on the stock resident:

* coordinated clauses (>=2 clause children, or >=2 VP children with a coordinator) become one segment
  each; coordinated VPs inherit the clause's subject tokens;
* a clause that ends its parent (a complement) splits off, and the matrix keeps a pronoun placeholder
  ("it") in the clause's argument position, so a matrix frame still binds that role and the binding is
  stitched back to the clause's original tokens.

Each segment gets a fresh syntax parse (the segment is new text), then the same native request shape as
the capture. The base capture is read-only. Measurement maps segment frames back to original token
indices and compares them with the 10 s teacher pass where one exists.
"""


def _offsets(text, leaves):
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


def _map_frames(segment, segment_row):
    """Segment frames as (frameName, anchor orig tokens, {role: orig token set}); 'P' expands to the clause."""
    charmap = segment["charmap"]
    placeholder = segment["placeholder"]

    def orig(chars):
        out = set()
        for start, end, token in charmap:
            if start < chars[1] and chars[0] < end:
                if token == "P":
                    out |= set(range(*placeholder))
                elif token == "L":
                    out |= set(range(*segment["lead"]))
                else:
                    out.add(token)
        return out

    frames = []
    for job in segment_row["jobs"]:
        record = job["record"]
        if record["capture_status"] != "framed":
            continue
        unit_text, offset = record["job"]["source"]["text"], record["job"]["span"][0]
        leaves = sorted((n for n in record["job"]["nodes"] if n["node_type"] == "leaf"), key=lambda n: n["start"])
        spans = {k: (v[0] + offset, v[1] + offset) for k, v in _offsets(unit_text, leaves).items() if v}

        def tokens(indices, spans=spans):
            return set().union(*(orig(spans[i]) for i in indices if i in spans)) if indices else set()

        for frame in record["response"]["payload"]["frames"]:
            roles = {fe["feName"]: tokens(fe["indices"]) for fe in frame.get("frameElements") or []}
            frames.append((frame["frameName"], tokens(frame["frameEvokingElement"]["indices"]), roles))
    return frames
