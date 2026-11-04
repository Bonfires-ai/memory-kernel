"""Clause units over one stored parse: fences, passes, merge, clauseRef.

Pure: no engine, no I/O. A FENCE is a non-root S/SBAR/SINV/SQ/SBARQ phrase that owns a non-aux VB* leaf and
is finite (carries SBAR, or owns a VBD/VBZ/VBP/MD leaf). A bare S that is an SBAR's only clausal child is
MERGED into it. Non-finite S-VP chains are never fences. Passes: one main-clause pass (top-level fences
collapsed; encoding ``B1`` = childless phrase, ``B2`` = a leaf-level stand-in) and one pass per fence
(nested fences collapsed, fence re-rooted with its original offsets). Dangling dependency heads are rewritten to the
collapsed fence or to the leaf itself.
The reference fence-analysis implementation is the parity reference.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

CLAUSE = frozenset({"S", "SBAR", "SINV", "SQ", "SBARQ"})
AUX = frozenset({"aux", "auxpass"})
FINITE = frozenset({"VBD", "VBZ", "VBP", "MD"})
ENCODINGS = ("B1", "B2")


def _types(node):
    return set(node.get("phrase_types") or ())


def _is_clausal(node):
    return node["node_type"] == "phrase" and bool(_types(node) & CLAUSE)


class FenceTree:
    """Index over one stored tree (snake_case parse nodes)."""

    def __init__(self, nodes):
        self.nodes = list(nodes)
        self.by_id = {n["node_id"]: n for n in self.nodes}
        if len(self.by_id) != len(self.nodes):
            raise ValueError("duplicate node ids")
        self.children = {}
        for n in self.nodes:
            self.children.setdefault(n.get("parent_constituent"), []).append(n)
        for kids in self.children.values():
            kids.sort(key=lambda n: (n["start"], n["node_type"] != "phrase", n["node_id"]))
        roots = self.children.get(None, [])
        if len(roots) != 1:
            raise ValueError(f"expected one root, got {len(roots)}")
        self.root = roots[0]
        self.merged = {n["node_id"] for n in self.nodes if self._merged(n)}

    def _merged(self, node):
        if not _is_clausal(node) or _types(node) != {"S"}:
            return False
        parent = self.by_id.get(node.get("parent_constituent"))
        if parent is None or parent["node_type"] != "phrase" or "SBAR" not in _types(parent):
            return False
        clausal_kids = [k for k in self.children.get(parent["node_id"], ()) if _is_clausal(k)]
        return clausal_kids == [node]

    def subtree(self, node):
        """Node and all descendants, document order."""
        out = [node]
        for kid in self.children.get(node["node_id"], ()):
            out.extend(self.subtree(kid))
        return out

    def own_leaves(self, node):
        """Leaves reachable without crossing a non-merged clause-typed phrase."""
        out = []
        for kid in self.children.get(node["node_id"], ()):
            if kid["node_type"] == "leaf":
                out.append(kid)
            elif _is_clausal(kid) and kid["node_id"] not in self.merged:
                continue
            else:
                out.extend(self.own_leaves(kid))
        return out

    def is_fence(self, node):
        if node is self.root or not _is_clausal(node) or node["node_id"] in self.merged:
            return False
        leaves = self.own_leaves(node)
        has_verb = any(
            (leaf.get("lex_class") or "").startswith("VB") and leaf.get("dependency_label") not in AUX
            for leaf in leaves
        )
        finite = "SBAR" in _types(node) or any((leaf.get("lex_class") or "") in FINITE for leaf in leaves)
        return has_verb and finite

    def fence_parent(self, node):
        """Nearest fence ancestor id, or None (top-level)."""
        current = self.by_id.get(node.get("parent_constituent"))
        while current is not None:
            if current["node_id"] in self.fences:
                return current["node_id"]
            current = self.by_id.get(current.get("parent_constituent"))
        return None

    @property
    def fences(self):
        if not hasattr(self, "_fences"):
            self._fences = {n["node_id"] for n in self.nodes if self.is_fence(n)}
        return self._fences


def clause_head(tree, fence):
    """The leaf inside the fence whose dependency head lies outside the fence's span."""
    lo, hi = fence["start"], fence["end"]
    leaves = sorted((n for n in tree.subtree(fence) if n["node_type"] == "leaf"), key=lambda n: n["start"])
    outward = []
    for leaf in leaves:
        head = tree.by_id.get(leaf.get("dependency_head"))
        if head is None or head is leaf or not lo <= head["start"] < hi:
            outward.append(leaf)
    if not outward:
        return None
    verbs = [leaf for leaf in outward if (leaf.get("lex_class") or "").startswith(("VB", "MD"))]
    return (verbs or outward)[0]


def _collapse(node, encoding, tree):
    if encoding == "B1":
        return {
            k: node[k] for k in ("node_id", "node_type", "parent_constituent", "phrase_types", "start", "end", "string")
        }
    if encoding == "B2":
        head = clause_head(tree, node)
        return {
            "dependency_head": head.get("dependency_head") if head is not None else node["node_id"],
            "dependency_label": head.get("dependency_label") if head is not None else "dep",
            "end": node["end"],
            "lemma": "clause",
            "lex_class": "IN",
            "named_entity_type": "",
            "node_id": node["node_id"],
            "node_type": "leaf",
            "parent_constituent": node["parent_constituent"],
            "phrase_types": list(node.get("phrase_types") or ()),
            "start": node["start"],
            "string": node["string"],
        }
    raise ValueError(f"unknown encoding {encoding!r}")


def _build_pass(tree, top, collapsed_ids, encoding, *, reroot):
    """Nodes of ``top``'s subtree with each collapsed fence kept childless; dangling heads rewritten."""
    keep, owner = [], {}
    removed_owner = {}

    def walk(node):
        if node["node_id"] in collapsed_ids:
            keep.append(_collapse(node, encoding, tree))
            for desc in tree.subtree(node)[1:]:
                removed_owner[desc["node_id"]] = node["node_id"]
            return
        keep.append(deepcopy(node))
        for kid in tree.children.get(node["node_id"], ()):
            walk(kid)

    walk(top)
    ids = {n["node_id"] for n in keep}
    rewrites = []
    for n in keep:
        owner[n["node_id"]] = n
    if reroot:
        keep[0]["parent_constituent"] = None
    for n in keep:
        if n["node_type"] != "leaf" or "dependency_head" not in n:
            continue
        head = n["dependency_head"]
        if head in ids:
            continue
        target = removed_owner.get(head, n["node_id"])
        if target == n["node_id"] and n["node_id"] in collapsed_ids:
            target = n["node_id"]
        rewrites.append({"node_id": n["node_id"], "from": head, "to": target})
        n["dependency_head"] = target
    keep.sort(key=lambda n: n["node_id"])
    return keep, rewrites


def nodes_sha256(nodes):
    return hashlib.sha256(json.dumps(nodes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fence_analysis(nodes, encoding="B1"):
    """Fences, the main-pass nodes and per-fence clause-pass nodes for one stored tree.

    Returns ``{"fences": [...], "passes": [...]}``. ``passes[0]`` is the main-clause pass (``p00``);
    ``passes[i]`` (i >= 1) is the clause pass of ``fences[i-1]`` (document order). Each fence row
    carries its pass id, span, phrase types, parent fence (or None), the pass it is collapsed into,
    and its clause head leaf index. With no fence the single pass is the unchanged tree.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}")
    if not nodes:
        raise ValueError("refusing nil nodes")
    tree = FenceTree(nodes)
    ordered = sorted((tree.by_id[i] for i in tree.fences), key=lambda n: (n["start"], -n["end"], n["node_id"]))
    pass_of = {f["node_id"]: f"p{i + 1:02d}" for i, f in enumerate(ordered)}
    parent_of = {f["node_id"]: tree.fence_parent(f) for f in ordered}
    fences = []
    for f in ordered:
        head = clause_head(tree, f)
        host = parent_of[f["node_id"]]
        fences.append(
            {
                "node_id": f["node_id"],
                "pass": pass_of[f["node_id"]],
                "phrase_types": list(f.get("phrase_types") or ()),
                "span": [f["start"], f["end"]],
                "string": f["string"],
                "parent_fence": host,
                "collapsed_in": "p00" if host is None else pass_of[host],
                "head_index": None if head is None else head["start"],
                "noun_attached": "NP" in _types(tree.by_id.get(f["parent_constituent"]) or {}),
            }
        )
    top_level = {f["node_id"] for f in ordered if parent_of[f["node_id"]] is None}
    main, rewrites = _build_pass(tree, tree.root, top_level, encoding, reroot=False)
    passes = [{"id": "p00", "fence": None, "nodes": main, "rewrites": rewrites, "collapsed": sorted(top_level)}]
    for f in ordered:
        nested = {g["node_id"] for g in ordered if parent_of[g["node_id"]] == f["node_id"]}
        clause, rewrites = _build_pass(tree, f, nested, encoding, reroot=True)
        passes.append(
            {
                "id": pass_of[f["node_id"]],
                "fence": f["node_id"],
                "nodes": clause,
                "rewrites": rewrites,
                "collapsed": sorted(nested),
            }
        )
    for p in passes:
        if not p["nodes"]:
            raise ValueError("refusing a pass with nil nodes")
        p["nodes_sha256"] = nodes_sha256(p["nodes"])
        p["verbs"] = sum(
            1
            for n in p["nodes"]
            if n["node_type"] == "leaf"
            and n.get("lemma") != "clause"
            and (n.get("lex_class") or "").startswith("VB")
            and n.get("dependency_label") not in AUX
        )
    return {"encoding": encoding, "fences": fences, "passes": passes}


def fenceable(nodes):
    return bool(FenceTree(nodes).fences) if nodes else False


# --- merge, clauseRef and the sealed fenced-unit record ------------------------------------

FENCED_UNIT_SCHEMA = "mk-fenced-unit/v1"
TIMEOUT_SYMBOL = "FCG-PROPBANK::TIME-OUT"


def wire_to_py(value):
    kind = value.get("kind")
    if kind == "nil":
        return None
    if kind == "symbol":
        return value["name"]
    if kind == "string":
        return value["value"]
    if kind == "number":
        return value["printed"]
    if kind == "cons":
        out, cur = [], value
        while cur.get("kind") == "cons":
            out.append(wire_to_py(cur["car"]))
            cur = cur["cdr"]
        if cur.get("kind") != "nil":
            out.append(wire_to_py(cur))
        return out
    return None


def snapshot_bindings(native):
    """(verb start, frameName) -> {role: bound unit span list}, from meaning rows + finalUnits SPAN."""
    if not isinstance(native, dict):
        return {}
    spans = {}
    for unit in native.get("finalUnits") or []:
        raw = wire_to_py(unit.get("raw") or {"kind": "nil"})
        if not isinstance(raw, list) or not raw:
            continue
        feats = {f[0]: f[1] for f in raw[1:] if isinstance(f, list) and f and isinstance(f[0], str) and len(f) > 1}
        span = feats.get("SPAN")
        if isinstance(span, list) and len(span) == 2:
            spans[raw[0]] = (int(span[0]), int(span[1]))
    meaning = wire_to_py(native.get("meaning") or {"kind": "nil"}) or []
    verbs, roles = {}, {}
    for row in meaning:
        if not isinstance(row, list):
            continue
        if row[:1] == ["FRAME"] and len(row) == 3 and row[2] in spans:
            verbs[row[2]] = (spans[row[2]][0], str(row[1]).lower())
        elif row[:1] == ["FRAME-ELEMENT"] and len(row) == 4:
            roles.setdefault(row[2], {}).setdefault(str(row[1]).lower(), []).append(row[3])
    out = {}
    for unit, key in verbs.items():
        bound = {}
        for role, units in roles.get(unit, {}).items():
            if len(units) == 1 and units[0] in spans:
                lo, hi = spans[units[0]]
                bound[role] = list(range(lo, hi))
        out[key] = bound
    return out


def pass_status(row):
    """ok | time-out | error | refused-nil. Only the engine's own TIME-OUT symbol counts as a time-out."""
    status = row.get("status")
    if status == "returned":
        return "time-out" if str(row.get("meaningKind") or "").upper() == "TIME-OUT" else "ok"
    if status == "error" and TIMEOUT_SYMBOL in (row.get("message") or ""):
        return "time-out"
    return status or "error"


def pass_rows(analysis, response):
    """Pass rows (analysis order), frames per completed pass (ids ``<pass>-<id>``), SPAN repairs."""
    raw_rows = ((response or {}).get("payload") or {}).get("passes") or []
    by_id = {r["id"]: r for r in raw_rows}
    if len(by_id) != len(raw_rows):
        raise ValueError("duplicate pass ids in phase-A rows")
    rows, frames_by_pass, repairs = [], {}, []
    for p in analysis["passes"]:
        raw = by_id.get(p["id"])
        status = "not-run" if raw is None else pass_status(raw)
        rows.append(
            {
                "id": p["id"],
                "fence": p["fence"],
                "nodes_sha256": p["nodes_sha256"],
                "status": status,
                "seconds": None if raw is None else raw.get("seconds"),
                "native": None if raw is None else raw.get("native"),
                "fallback": None,
            }
        )
        if status != "ok":
            continue
        bound = snapshot_bindings(raw.get("native"))
        out = []
        for frame in raw.get("frames") or []:
            frame = deepcopy(frame)
            frame["id"] = f"{p['id']}-{frame['id']}"
            anchor = (frame.get("frameEvokingElement") or {}).get("indices") or []
            spans = bound.get((anchor[0], frame["frameName"]), {}) if anchor else {}
            for element in frame.get("frameElements") or []:
                span = spans.get(str(element.get("feName", "")).lower())
                if span is not None and span != element.get("indices"):
                    repairs.append(
                        {
                            "pass": p["id"],
                            "frame": frame["id"],
                            "role": element["feName"],
                            "from": element.get("indices"),
                            "to": span,
                        }
                    )
                    element["indices"] = span
            out.append(frame)
        frames_by_pass[p["id"]] = out
    return rows, frames_by_pass, repairs


def attach_clause_refs(frames, fences):
    """Strip every ``clauseRef``, then give a role exactly on a fence collapsed in its pass ONE ref.

    The ref resolves to the frame evoked at the fence's head index in the fence's own pass, whether that
    frame came from phase A or from a phase-B segment (both carry ``<pass>-...`` ids).
    """
    pass_of = {f["node_id"]: f["pass"] for f in fences}
    by_pass = {}
    for frame in frames:
        for element in frame.get("frameElements") or []:
            element.pop("clauseRef", None)
        by_pass.setdefault(frame["id"].split("-", 1)[0], []).append(frame)
    for frame in frames:
        home = frame["id"].split("-", 1)[0]
        for element in frame.get("frameElements") or []:
            for fence in fences:
                if fence["collapsed_in"] != home or element.get("indices") != list(range(*fence["span"])):
                    continue
                target = pass_of[fence["node_id"]]
                head = fence["head_index"]
                hit = next(
                    (
                        g["id"]
                        for g in by_pass.get(target, [])
                        if (g.get("frameEvokingElement") or {}).get("indices") == [head]
                    ),
                    None,
                )
                element["clauseRef"] = {"pass": target, "head_index": head, "frame": hit}
                break
    return frames


def _anchor(frame):
    return tuple((frame.get("frameEvokingElement") or {}).get("indices") or ())


def _dedupe(frames):
    seen, out = set(), []
    for frame in frames:
        key = (frame["id"].split("-", 1)[0], frame.get("frameName"), _anchor(frame))
        if key not in seen:
            seen.add(key)
            out.append(frame)
    return out


_SEGMENT_ID = re.compile(r"^(?P<pass>[^-]+)-(?P<seg>s\d+)-")


def _renumber(frames, pass_order):
    """Deterministic ids: ``<pass>-f<k>`` (phase A) / ``<pass>-sNN-f<k>`` (phase B), k 0-based per group.

    Engine frame ids are global counters (they differ per shard and run), so they never reach a merged
    frame; the raw id survives only inside ``passes[].native``. Order inside a group: sorted anchor
    indices, frameName, sorted (feName, sorted indices) of the elements.
    """
    groups = {}
    for frame in frames:
        pid = frame["id"].split("-", 1)[0]
        m = _SEGMENT_ID.match(frame["id"])
        groups.setdefault((pid, m["seg"] if m else ""), []).append(frame)
    out = []
    for (pid, seg), members in sorted(groups.items(), key=lambda kv: (pass_order[kv[0][0]], kv[0][1])):
        members.sort(
            key=lambda f: (
                sorted(_anchor(f)),
                str(f.get("frameName")),
                sorted((str(e.get("feName")), sorted(e.get("indices") or ())) for e in f.get("frameElements") or []),
            )
        )
        prefix = f"{pid}-{seg}-" if seg else f"{pid}-"
        for k, frame in enumerate(members):
            frame["id"] = f"{prefix}f{k}"
            out.append(frame)
    return out


def fenced_unit_record(job, analysis, phase_a, fallbacks, *, t_pass, deadline_s):
    """The sealed fenced-unit record; byte-identical for any order of pass rows."""
    from memory_kernel.construction.memory_semantic_jev import seal

    rows, frames_by_pass, repairs = pass_rows(analysis, phase_a)
    request_sha256s = {s for s in (phase_a.get("request_sha256s") or []) if s}
    for row in rows:
        fb = (fallbacks or {}).get(row["id"])
        if fb is None:
            continue
        row["fallback"] = {k: deepcopy(fb[k]) for k in ("status", "seconds", "segments")}
        frames_by_pass.setdefault(row["id"], []).extend(deepcopy(fb["frames"]))
        for segment in fb["segments"] or []:
            request_sha256s.update(j["request_sha256"] for j in segment.get("jobs") or [] if j.get("request_sha256"))
    merged = _dedupe([f for p in analysis["passes"] for f in frames_by_pass.get(p["id"], [])])
    order = {p["id"]: i for i, p in enumerate(analysis["passes"])}
    frames = attach_clause_refs(_renumber(merged, order), analysis["fences"])
    timed = any(r["status"] in ("time-out", "not-run") and (r["fallback"] or {}).get("status") != "ok" for r in rows)
    status = "framed" if frames else ("cutoff" if timed or phase_a.get("status") == "deadline" else "empty")
    phase_a_s = sum(r["seconds"] or 0.0 for r in rows)
    fallback_s = sum((r["fallback"] or {}).get("seconds") or 0.0 for r in rows)
    return seal(
        {
            "schema_version": FENCED_UNIT_SCHEMA,
            "job": job,
            "encoding": analysis["encoding"],
            "t_pass": t_pass,
            "deadline_s": deadline_s,
            "request_sha256s": sorted(request_sha256s),
            "repairs": repairs,
            "elapsed_s": round(phase_a_s + fallback_s, 3),
            "engine_s": {"phase_a": round(phase_a_s, 3), "fallback": round(fallback_s, 3)},
            "capture_status": status,
            "response": {
                "status": phase_a.get("status", "error"),
                "payload": {"frames": frames, "passes": rows, "fences": analysis["fences"]},
            },
            "semantic_acceptance": "not_performed",
        }
    )
