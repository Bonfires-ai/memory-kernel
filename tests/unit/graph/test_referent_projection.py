"""``referent_policy="structure"`` keys entities by mention (Unit A): a coordinated role fans out into one referent per
conjunct, and names inside a filler are mentioned by its message. ``phrase`` keeps today's whole-filler keys.
Invented names only; needs WordNet and the spaCy model."""

from __future__ import annotations

from test_forward_resolution import NEEDS_WORDNET

from memory_kernel.graph.g1_mentions import structure_key
from memory_kernel.graph.g1_projection import analyse, project
from memory_kernel.graph.kg_contract import validate_rows

GROUP = "g-test"
LABEL = {
    "be.01": "state",
    "own.01": "state",
    "study.01": "occurrence",
    "visit.01": "occurrence",
    "say.01": "occurrence",
    "fix.01": "occurrence",
}


def rec(n: int, mid: str, text: str, roleset: str, anchor: int, roles: dict[str, tuple[str, list[int]]]) -> dict:
    return {
        "unit": f"{n:04d}/unit-00.json",
        "message_id": mid,
        "session_id": "s1",
        "speaker": "Ana",
        "text": text,
        "timestamp": f"2026-10-17T00:0{n}:00Z",
        "capture_status": "captured",
        "events": [
            {
                "roleset": roleset,
                "anchor": [anchor],
                "roles": {r: {"text": t, "span": s} for r, (t, s) in roles.items()},
            }
        ],
    }


def build(records: list[dict], policy: str) -> dict:
    eventuality = {
        f"{r['unit']}#{i}": {"label": LABEL[ev["roleset"]], "p": {LABEL[ev["roleset"]]: 1.0}}
        for r in records
        for i, ev in enumerate(r["events"])
    }
    out = project(
        records,
        eventuality=eventuality,
        episodes={r["message_id"]: f"ep-{r['message_id']}" for r in records},
        group_id=GROUP,
        created_at="2026-10-17T00:00:00Z",
        referent_policy=policy,
    )
    assert validate_rows(out["nodes"], out["edges"], previous=out["external_endpoints"]) == []
    return out


@NEEDS_WORDNET
def test_structure_fans_out_coordination_and_mentions_inner_names():
    records = [
        rec(1, "m1", "Kira and Milo own Brask Labs.", "own.01", 3,
            {"arg0": ("Kira and Milo", [0, 1, 2]), "arg1": ("Brask Labs", [4, 5])}),
        rec(2, "m2", "Kira studied the Zorvex sample.", "study.01", 1,
            {"arg0": ("Kira", [0]), "arg1": ("the Zorvex sample", [2, 3, 4])}),
        rec(3, "m3", "Kira and Milo visited Brask Labs.", "visit.01", 3,
            {"arg0": ("Kira and Milo", [0, 1, 2]), "arg1": ("Brask Labs", [4, 5])}),
    ]  # fmt: skip

    def names(out: dict) -> dict[str, str]:
        return {n["props"]["uuid"]: n["props"]["name"] for n in out["nodes"] if "Entity" in n["labels_neo4j"]}

    def owns(out: dict) -> list[tuple[str, str]]:
        name = names(out)
        return sorted((name[e["source"]], name[e["target"]]) for e in out["edges"] if e["props"].get("name") == "OWNS")

    out = build(records, "structure")
    name = names(out)
    assert owns(out) == [("Kira", "Brask Labs"), ("Milo", "Brask Labs")]
    m2 = next(
        n["props"]["uuid"]
        for n in out["nodes"]
        if n["props"].get("message_id") == "m2" and "Episodic" in n["labels_neo4j"]
    )
    mentioned = {name.get(e["target"]) for e in out["edges"] if e["type"] == "MENTIONS" and e["source"] == m2}
    assert {"Zorvex sample", "Zorvex"} <= mentioned
    assert "Kira and Milo" not in name.values()
    assert out["dedup"]["referents_split"] == 2
    assert (out["dedup"]["referents_realigned"], out["dedup"]["referents_nonnominal_dropped"]) == (0, 0)
    shifted = build([rec(7, "m7", "Kira and Milo own Brask Labs.", "own.01", 3,
                         {"arg0": ("Kira and Milo", [0, 1, 2]), "arg1": ("Brask Labs", [6, 7])})], "structure")  # fmt: skip
    assert shifted["dedup"]["referents_realigned"] == 1
    visit = next(n["props"]["uuid"] for n in out["nodes"] if n["props"].get("event_key", "").startswith("0003/"))
    agents = sorted(
        name[e["target"]] for e in out["edges"] if e["source"] == visit and e["props"].get("role") == "arg0"
    )
    assert agents == ["Kira", "Milo"]  # one role edge per referent

    # two conjuncts naming one entity write one fact edge and one role edge; "I" is the speaker
    twice = build([
        rec(4, "m4", "Kira and Kira own Brask Labs.", "own.01", 3,
            {"arg0": ("Kira and Kira", [0, 1, 2]), "arg1": ("Brask Labs", [4, 5])}),
        rec(5, "m5", "Kira and Kira visited Brask Labs.", "visit.01", 3,
            {"arg0": ("Kira and Kira", [0, 1, 2]), "arg1": ("Brask Labs", [4, 5])}),
    ], "structure")  # fmt: skip
    assert owns(twice) == [("Kira", "Brask Labs")]
    assert sum(e["props"].get("role") == "arg0" for e in twice["edges"]) == 1
    me = build([rec(6, "m6", "Kira and I own Brask Labs.", "own.01", 3,
                    {"arg0": ("Kira and I", [0, 1, 2]), "arg1": ("Brask Labs", [4, 5])})], "structure")  # fmt: skip
    assert owns(me) == [("Ana", "Brask Labs"), ("Kira", "Brask Labs")]
    # a coordinated copular subject is an identity line of each conjunct's referent (its own mention); a role span
    # shifted off the unit's parse is realigned onto it
    pilots = build([
        rec(8, "m8", "Kira and Milo are pilots.", "be.01", 3,
            {"arg1": ("Kira and Milo", [0, 1, 2]), "arg2": ("pilots", [4])}),
        rec(9, "m9", "Brask Labs is a shipyard.", "be.01", 2,
            {"arg1": ("Brask Labs", [2, 3]), "arg2": ("a shipyard", [3, 4])}),
    ], "structure")  # fmt: skip
    lines = {n["props"]["name"]: n["props"] for n in pilots["nodes"] if n["props"].get("summary_line_ids")}
    assert sorted(lines) == ["Brask Labs", "Kira", "Milo"]
    assert all(p["summary_identity_line_ids"] == p["summary_line_ids"] for p in lines.values())

    out = build(records, "phrase")
    assert "Kira and Milo" in names(out).values()
    assert owns(out) == [("Kira and Milo", "Brask Labs")]
    assert "referents_split" not in out["dedup"]

    assert structure_key("the Brask Labs") == analyse("the Brask Labs")[1]


@NEEDS_WORDNET
def test_structure_events_are_named_by_frame():
    record = rec(1, "m1", "Kira said Milo fixed the device.", "say.01", 1,
                 {"arg0": ("Kira", [0]), "arg1": ("Milo fixed the device", [2, 3, 4, 5])})  # fmt: skip
    record["events"].append(
        {
            "roleset": "fix.01",
            "anchor": [3],
            "roles": {"arg0": {"text": "Milo", "span": [2]}, "arg1": {"text": "the device", "span": [4, 5]}},
        }
    )

    def events(out: dict) -> dict[str, dict]:
        return {n["props"]["event_key"]: n["props"] for n in out["nodes"] if "Event" in n["labels_neo4j"]}

    today = events(build([record], "phrase"))
    assert len({p["name"] for p in today.values()}) == len(today)  # phrase: the rendered sentence, unchanged
    assert all("source_text" not in p for p in today.values())
    out = events(build([record], "structure"))
    assert sorted(p["name"] for p in out.values()) == [
        "fix.01: arg0 Milo; arg1 device",
        "say.01: arg0 Kira; arg1 Milo fixed the device",
    ]
    for key, props in out.items():
        assert props["source_text"] == today[key]["name"]
        assert props["key_text"] == today[key]["key_text"]


def test_answer_context_shows_an_events_source_text():
    from memory_kernel.search.kg_context import ContextConfig, _timeline

    row = {
        "name": "say.01: arg0 Kira; arg1 Milo fixed the device",
        "source_text": "Kira said Milo fixed the device.",
        "told_at": "2026-10-17T00:01:00Z",
    }
    plain = {"name": "Kira said Milo fixed the device.", "told_at": "2026-10-17T00:01:00Z"}
    ((key, line),) = _timeline([row], ContextConfig())
    assert "Kira said Milo fixed the device." in line and "say.01" not in line
    assert (key, line) == _timeline([plain], ContextConfig())[0]

    # a support line: the reader's MENTIONS row (flat props, as the kernel writer stores them) carries source_text
    import asyncio
    import re

    from memory_kernel.search.kg_context import _support_line
    from memory_kernel.search.kg_index import MENTIONS_CYPHER, build_index
    from memory_kernel.search.kg_recipe import search_messages_v2

    props = {**row, "uuid": "e1", "key_text": "Kira say Milo fix the device"}
    mention = {"m": "m1", "labels": ["Entity", "Event"]} | {
        alias: props.get(prop) for prop, alias in re.findall(r"\bn\.(\w+) AS (\w+)", MENTIONS_CYPHER)
    }
    msgs = [{"uuid": "m1", "content": "Kira: Milo fixed the device.", "told_at": row["told_at"], "valid_at": None}]
    idx = build_index(msgs, [], [mention], {"m1": [1.0, 0.0]}, model="m", dim=2)
    res = asyncio.run(search_messages_v2(None, group_id=GROUP, query="q", qvec=[1.0, 0.0], index=idx))
    _, line = _support_line({"kind": "event", **res.items["e1"]})
    assert "Kira said Milo fixed the device." in line and "say.01" not in line
