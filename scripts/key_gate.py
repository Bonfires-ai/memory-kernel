"""Key gate: project build roots again and compare their search keys by message id.

A build's search keys are what retrieval reads. ``dump`` re-projects each archived root (the projection is pure, from
the stage outputs the root keeps), renders its keys and writes them keyed by message id; ``compare`` says whether two
dumps (two trees, same roots) agree byte for byte; ``persisted`` is informational: a dump against the keys each root
persisted at its own build (``11-index-001/index.json``).

The script reads only modules that exist in every tree it is run under, so one file runs unchanged under both
(``PYTHONPATH=<tree>/src``). Read-only: it writes nothing but ``--out``. Nothing here names a dataset: roots, sample
ids, frame dirs and the purpose come in as arguments.

    key_gate.py dump --workdir DIR --sample-id ID... --frame-dir D... --purpose-file F [--key-render legacy] --out OUT
    key_gate.py compare LEFT RIGHT [--sample-id ID...]      exit 0 only when equal
    key_gate.py persisted DUMP --workdir DIR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SAMPLES_PER_ID = 3


def reader_rows(rows: Mapping) -> tuple[list[dict], list[dict], list[dict]]:
    """(messages, facts, mentions) as ``kg_index.read_key_rows`` returns them, from projection rows.

    Mirrors ``kg_index`` ``MESSAGES_CYPHER`` / ``FACTS_CYPHER`` / ``MENTIONS_CYPHER``."""
    nodes = {n["props"]["uuid"]: n for n in rows["nodes"]}
    saga_of = {e["target"]: e["source"] for e in rows["edges"] if e["type"] == "HAS_EPISODE"}
    msgs, facts, mentions = [], [], []
    for u, n in nodes.items():
        if "Episodic" not in n["labels_neo4j"]:
            continue
        p = n["props"]
        msgs.append(
            {
                "uuid": u,
                "message_id": p.get("message_id"),
                "content": p.get("content"),
                "sender_name": p.get("sender_name"),
                "addressee_name": p.get("addressee_name"),
                "told_at": None if p.get("told_at") is None else str(p["told_at"]),
                "valid_at": None if p.get("valid_at") is None else str(p["valid_at"]),
                "saga": saga_of.get(u),
                "attributes": p.get("attributes"),
            }
        )
    for e in rows["edges"]:
        p = e["props"]
        if e["type"] == "RELATES_TO":
            facts.append(
                {
                    "uuid": p["uuid"],
                    "fact": p.get("fact"),
                    "episodes": p.get("episodes"),
                    "emb": p.get("fact_embedding"),
                    "asserted": p.get("asserted"),
                    "modality": p.get("modality"),
                    "key_text": p.get("key_text"),
                    "attributes": p.get("attributes"),
                    "invalid": p.get("invalid_at") is not None or p.get("expired_at") is not None,
                }
            )
        elif e["type"] == "MENTIONS":
            src, tgt = nodes.get(e["source"]), nodes.get(e["target"])
            if (
                src is None
                or tgt is None
                or "Episodic" not in src["labels_neo4j"]
                or "Entity" not in tgt["labels_neo4j"]
            ):
                continue
            tp = tgt["props"]
            mentions.append(
                {
                    "m": e["source"],
                    "uuid": tp["uuid"],
                    "name": tp.get("name"),
                    "labels": tgt["labels_neo4j"],
                    **{k: tp.get(k) for k in ("asserted", "modality", "key_text", "source_text", "attributes")},
                }
            )
    return msgs, facts, mentions


def root_config(workdir: Path, sample_id: str, frame_dirs: Sequence[Path], purpose: str):
    """The ``BuildConfig`` that projects ``<workdir>/<sample_id>``: only the fields the projection reads are set."""
    from memory_kernel.pipeline.config import BuildConfig

    root = workdir / sample_id
    found = sorted((root / "01-select-001").glob("*-conversation-only.json"))
    if len(found) != 1:
        raise SystemExit(
            f"{sample_id}: expected exactly one *-conversation-only.json under {root}/01-select-001, got {found}"
        )
    index = root / "11-index-001" / "index.json"
    group = json.loads(index.read_text())["group"] if index.is_file() else f"key-gate-{sample_id}"
    return BuildConfig(
        sample_id=sample_id,
        conversation=found[0],
        group=group,
        purpose=purpose,
        workdir=workdir,
        root=workdir,
        frame_dirs=tuple(frame_dirs),
    )


def dump_keys(cfg, key_render: str) -> dict:
    """Project ``cfg``'s root and render its search keys by message id."""
    from memory_kernel.pipeline.runner import Ctx
    from memory_kernel.pipeline.stages import write
    from memory_kernel.search import kg_index

    _, _, rows = write.project(Ctx(cfg=cfg, concurrency=1))
    msgs, facts, mentions = reader_rows(rows)
    by_uuid = kg_index.keys_from_graph(msgs, facts, mentions, key_render=key_render)
    keys: dict[str, str] = {}
    uuids: dict[str, str] = {}
    for m in msgs:
        mid = m["message_id"]
        if mid in keys:
            raise SystemExit(f"{cfg.sample_id}: duplicate message_id {mid!r}")
        if m["uuid"] not in by_uuid:
            raise SystemExit(f"{cfg.sample_id}: message {mid!r} ({m['uuid']}) has no key")
        keys[mid] = by_uuid[m["uuid"]]
        uuids[mid] = m["uuid"]
    return {"recipe_tag": kg_index.recipe_tag(key_render), "keys": keys, "uuids": uuids}


def _diff(left: Mapping[str, str], right: Mapping[str, str]) -> dict:
    changed = sorted(m for m in set(left) & set(right) if left[m] != right[m])
    return {
        "only_left": sorted(set(left) - set(right)),
        "only_right": sorted(set(right) - set(left)),
        "changed": changed,
        "common": len(set(left) & set(right)),
        "samples": [{"message_id": m, "left": left[m], "right": right[m]} for m in changed[:SAMPLES_PER_ID]],
    }


def compare(left: dict, right: dict) -> dict:
    """``{"equal": bool, "problems": [...], "samples": {sid: {only_left, only_right, changed, samples, ...}}}``.

    ``left`` / ``right`` are dump ``samples`` mappings (sample id -> ``dump_keys`` result). Fails when the sample sets
    or the recipe tags differ, or any count is non-zero."""
    problems: list[str] = []
    if set(left) != set(right):
        problems.append(
            f"sample sets differ: only left {sorted(set(left) - set(right))}, only right {sorted(set(right) - set(left))}"
        )
    report: dict[str, Any] = {}
    for sid in sorted(set(left) & set(right)):
        a, b = left[sid], right[sid]
        if a["recipe_tag"] != b["recipe_tag"]:
            problems.append(f"{sid}: recipe tags differ ({a['recipe_tag']!r} vs {b['recipe_tag']!r})")
        d = _diff(a["keys"], b["keys"])
        report[sid] = {
            **d,
            "only_left": len(d["only_left"]),
            "only_right": len(d["only_right"]),
            "changed": len(d["changed"]),
        }
        report[sid]["only_left_ids"], report[sid]["only_right_ids"] = (
            d["only_left"][:SAMPLES_PER_ID],
            d["only_right"][:SAMPLES_PER_ID],
        )
        if any(report[sid][k] for k in ("only_left", "only_right", "changed")):
            problems.append(
                f"{sid}: only_left={report[sid]['only_left']} only_right={report[sid]['only_right']} changed={report[sid]['changed']}"
            )
    return {"equal": not problems, "problems": problems, "samples": report}


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def _tree_state(module_file: str) -> tuple[str | None, int | None]:
    cwd = Path(module_file).resolve().parent
    try:
        return _git(["rev-parse", "HEAD"], cwd).strip(), len(_git(["status", "--porcelain"], cwd).splitlines())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _cmd_dump(args: argparse.Namespace) -> int:
    import memory_kernel
    from memory_kernel.cli import read_purpose

    purpose = read_purpose(Path(args.purpose_file))
    workdir = Path(args.workdir)
    head, dirty = _tree_state(memory_kernel.__file__)
    out: dict[str, Any] = {"memory_kernel": memory_kernel.__file__, "head": head, "dirty": dirty, "samples": {}}
    for sid in args.sample_id:
        cfg = root_config(workdir, sid, [Path(d) for d in args.frame_dir], purpose)
        out["samples"][sid] = dump_keys(cfg, args.key_render)
        print(f"{sid}: {len(out['samples'][sid]['keys'])} keys", file=sys.stderr, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=0, sort_keys=True))
    total = sum(len(s["keys"]) for s in out["samples"].values())
    print(f"{len(out['samples'])} samples, {total} keys, head {head}, dirty {dirty}, {memory_kernel.__file__}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    left, right = _load(args.left)["samples"], _load(args.right)["samples"]
    if args.sample_id:
        left = {k: v for k, v in left.items() if k in args.sample_id}
        right = {k: v for k, v in right.items() if k in args.sample_id}
        missing = [s for s in args.sample_id if s not in left or s not in right]
        if missing:
            print(f"sample ids absent from a dump: {missing}", file=sys.stderr)
            return 1
    result = compare(left, right)
    print(json.dumps(result, indent=1))
    return 0 if result["equal"] else 1


def _cmd_persisted(args: argparse.Namespace) -> int:
    dump = _load(args.dump)["samples"]
    out: dict[str, Any] = {}
    for sid, s in dump.items():
        persisted = json.loads((Path(args.workdir) / sid / "11-index-001" / "index.json").read_text())["keys"]
        uuid_of = {u: m for m, u in s["uuids"].items()}
        mine = {u: s["keys"][m] for u, m in uuid_of.items()}
        d = _diff(persisted, mine)
        d["samples"] = [{**x, "message_id": uuid_of.get(x["message_id"])} for x in d["samples"]]
        out[sid] = {
            **d,
            "only_left": len(d["only_left"]),
            "only_right": len(d["only_right"]),
            "changed_ids": [uuid_of.get(u) for u in d["changed"]],
            "changed": len(d["changed"]),
        }
    print(json.dumps(out, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="project each root and write its keys by message id")
    d.add_argument("--workdir", required=True)
    d.add_argument("--sample-id", nargs="+", required=True)
    d.add_argument("--frame-dir", nargs="+", required=True)
    d.add_argument("--purpose-file", required=True)
    d.add_argument("--key-render", default="legacy")
    d.add_argument("--out", required=True)
    d.set_defaults(fn=_cmd_dump)
    c = sub.add_parser("compare", help="exit 0 only when two dumps' keys are equal")
    c.add_argument("left")
    c.add_argument("right")
    c.add_argument("--sample-id", nargs="*", default=[])
    c.set_defaults(fn=_cmd_compare)
    p = sub.add_parser("persisted", help="informational: a dump against each root's persisted index keys")
    p.add_argument("dump")
    p.add_argument("--workdir", required=True)
    p.set_defaults(fn=_cmd_persisted)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
