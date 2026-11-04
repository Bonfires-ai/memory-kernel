"""The 1.1 input layer is a no-op on the benchmark conversations: prepared batches and sources equal 1.0.0's for every
conversation, the build's message count (``conversation_counts``) equals the turn count, and the boundary plan of a
single conversation (its no-send walk's wire list and plan summary, with and without a one-conversation mapping)
equals 1.0.0's. The boundary walk is computed offline in both trees (no send). The graph side is not covered.

Usage: python scripts/check_input_noop.py <conversations.json> <old-src-dir>
<old-src-dir> holds the 1.0.0 package, e.g. `git archive 2f59b936 src | tar -x -C /tmp/mk100` -> /tmp/mk100/src."""

import json
import subprocess
import sys
from pathlib import Path

PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[2])
from memory_kernel.pipeline.stages._conversation import build_prepared, build_sources, split_conversation
items = json.load(open(sys.argv[1]))
out = {}
for item in items:
    conv_only, _ = split_conversation(item)
    prepared = build_prepared(conv_only)
    sources = build_sources(prepared, {"bonfire_id": "noop", "conversation_id": item["sample_id"]})
    out[item["sample_id"]] = json.dumps([prepared, sources], sort_keys=True, ensure_ascii=True)
print(json.dumps(out))
"""


COUNT_PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[2])
from pathlib import Path
from memory_kernel.pipeline.runner import conversation_counts
items = json.load(open(sys.argv[1]))
print(json.dumps({item["sample_id"]: conversation_counts(Path(sys.argv[1]), item["sample_id"]).messages for item in items}))
"""


WIRES_PROBE = r"""
import inspect, json, sys
sys.path.insert(0, sys.argv[2])
from memory_kernel.pipeline.stages import _boundary_walk as bw
from memory_kernel.pipeline.stages._conversation import build_prepared, build_sources, split_conversation
items = json.load(open(sys.argv[1]))
takes = "conversations" in inspect.signature(bw.walk).parameters
fits = {"fit": False} if "fit" in inspect.signature(bw.walk).parameters else {}  # 1.3: the frozen, untrimmed wires
out = {}
for item in items:
    conv_only, _ = split_conversation(item)
    sources = build_sources(build_prepared(conv_only), {"bonfire_id": "noop", "conversation_id": item["sample_id"]})
    variants = {"default": {}}
    if takes:  # the 1.1 walk: no mapping, and every session mapped to the one conversation
        variants["none"] = {"conversations": None}
        variants["mapped"] = {"conversations": {s["session_id"]: item["sample_id"] for s in sources}}
    for name, extra in variants.items():
        wires = []
        def decide(step, history):
            wires.append([step["index"], step["working_window_start_order"], step["request_sha256"], step["content_sha256"], step["wire"]])
            return {"index": step["index"], "message_id": step["message_id"], "choice": "continue_active"}
        walked = bw.walk(sources, decide, episode_prefix="ep-", window_prefix="win-", **extra, **fits)
        summary = bw.plan(sources, episode_prefix="ep-", window_prefix="win-", **extra)
        summary.pop("stage_cap_exceeded", None)  # 1.3 records a reservation over the cap (1.0.0 stopped live on it)
        out.setdefault(item["sample_id"], {})[name] = json.dumps([wires, walked["rows"], walked["episodes"], summary], sort_keys=True, ensure_ascii=True, default=str)
print(json.dumps(out))
"""


def run(src: Path, data: Path, probe: str = PROBE) -> dict:
    done = subprocess.run([sys.executable, "-I", "-c", probe, str(data), str(src)], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(done.stderr)
    return json.loads(done.stdout)


def main() -> int:
    data, old_src = Path(sys.argv[1]), Path(sys.argv[2])
    new_src = Path(__file__).resolve().parents[1] / "src"
    old, new = run(old_src, data), run(new_src, data)
    bad = 0
    for sample_id in sorted(old):
        same = old[sample_id] == new.get(sample_id)
        bad += not same
        print(f"{sample_id}: {'identical' if same else 'DIFFERS'}")
    counts = run(new_src, data, COUNT_PROBE)
    for item in json.loads(data.read_bytes()):
        conv = item["conversation"]
        turns = sum(len(conv[k]) for k in conv if k.startswith("session_") and isinstance(conv[k], list))
        same = counts[item["sample_id"]] == turns
        bad += not same
        print(f"{item['sample_id']}: messages={counts[item['sample_id']]} turns={turns} {'ok' if same else 'DIFFERS'}")
    old_wires, new_wires = run(old_src, data, WIRES_PROBE), run(new_src, data, WIRES_PROBE)
    for sample_id in sorted(old_wires):
        base = old_wires[sample_id]["default"]
        for name, got in new_wires[sample_id].items():
            same = got == base
            bad += not same
            print(f"{sample_id}: boundary plan ({name}) {'identical' if same else 'DIFFERS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
