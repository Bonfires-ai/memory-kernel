"""Joined names project to one person entity (kernel 1.1 B, Task 4).

Fixture: ``project()`` arguments built by hand, the smallest set its body reads. Each record is a ``graph.json``
record (``unit``, ``message_id``, ``session_id``, ``speaker``, ``text``, ``timestamp``, ``capture_status``, ``events``);
each event has a ``roleset``, an ``anchor`` (token index of the predicate) and ``roles`` with ``text`` and ``span``
(token indexes). ``eventuality`` gives every ``unit#index`` a label: be.01 is a ``state``; like.01 is labelled
``occurrence`` so its arg0 becomes a role edge (its arg1 "pears" is not referring, so a state would be a summary
line, not an edge). ``labels``, ``cards``, ``aligned`` are empty; one episode per message; no ``roleset_index``
(``role_themes`` is then not required). Invented names only. Tests whose joined name starts a sentence ("Dee likes
pears") need WordNet: without it every name counts as a common noun, and a sentence-initial one never joins.

Records: m1 Ana "my name is Dee" (be.01 arg1 "my name", arg2 "Dee"); m2 Bo "Dee likes pears" (like.01 arg0 "Dee",
arg1 "pears"); m3 Bo "Dee is Ana" (be.01 arg1 "Dee", arg2 "Ana").
"""

import pytest

from memory_kernel.grammar.dependencies import wordnet_ready
from memory_kernel.graph.g1_projection import project
from memory_kernel.graph.kg_contract import validate_rows

GROUP = "g-test"
EVENTS = {
    "m1": {"roleset": "be.01", "anchor": [2], "roles": {"arg1": ("my name", [0, 1]), "arg2": ("Dee", [3])}},
    "m2": {"roleset": "like.01", "anchor": [1], "roles": {"arg0": ("Dee", [0]), "arg1": ("pears", [2])}},
    "m3": {"roleset": "be.01", "anchor": [1], "roles": {"arg1": ("Dee", [0]), "arg2": ("Ana", [2])}},
}
MESSAGES = [("m1", "Ana", "my name is Dee"), ("m2", "Bo", "Dee likes pears"), ("m3", "Bo", "Dee is Ana")]
LABEL = {"be.01": "state", "like.01": "occurrence", "meet.01": "occurrence", "give.01": "state"}


def _records(without: frozenset[str] = frozenset()) -> list[dict]:
    out = []
    for i, (mid, speaker, text) in enumerate(MESSAGES):
        ev = EVENTS[mid]
        events = (
            []
            if mid in without
            else [
                {
                    "roleset": ev["roleset"],
                    "anchor": ev["anchor"],
                    "roles": {r: {"text": t, "span": s} for r, (t, s) in ev["roles"].items()},
                }
            ]
        )
        out.append(
            {
                "unit": f"0000/unit-{i:02d}.json",
                "message_id": mid,
                "session_id": "s1",
                "speaker": speaker,
                "text": text,
                "timestamp": f"2026-10-14T00:0{i}:00Z",
                "capture_status": "captured",
                "events": events,
            }
        )
    return out


def run(records: list[dict]) -> dict:
    eventuality = {
        f"{r['unit']}#{i}": {"label": LABEL[ev["roleset"]], "p": {LABEL[ev["roleset"]]: 1.0}}
        for r in records
        for i, ev in enumerate(r["events"])
    }
    return project(
        records,
        eventuality=eventuality,
        episodes={r["message_id"]: f"ep-{r['message_id']}" for r in records},
        group_id=GROUP,
        created_at="2026-10-14T00:00:00Z",
    )


def _people(out: dict) -> list[dict]:
    return [n for n in out["nodes"] if "Person" in n["labels_neo4j"]]


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_joined_names_are_one_person_entity():
    out = run(_records())
    assert validate_rows(out["nodes"], out["edges"], previous=out["external_endpoints"]) == []
    people = _people(out)
    ana = [n for n in people if n["props"]["name"] == "Ana"]
    assert len(ana) == 1 and ana[0]["props"]["identity_aliases"] == ["Dee"]
    assert not [n for n in out["nodes"] if n["props"].get("name") == "Dee"]
    uuid = ana[0]["props"]["uuid"]

    # m2's like.01 Event: its arg0 role edge lands on Ana's node; the text keeps the surface "Dee".
    like = [n for n in out["nodes"] if n["props"].get("roleset") == "like.01"]
    assert len(like) == 1
    arg0 = [e for e in out["edges"] if e["source"] == like[0]["props"]["uuid"] and e["props"].get("role") == "arg0"]
    assert [e["target"] for e in arg0] == [uuid]

    # The claim events m1 "my name is Dee" and m3 "Dee is Ana" write nothing: no fact edge (no "Ana's name IS Ana"),
    # no summary line, no entity of their own; their messages remain.
    relates = [e for e in out["edges"] if e["type"] == "RELATES_TO"]
    assert all(e["source"] != e["target"] for e in relates)
    claim_keys = {"0000/unit-00.json#0", "0000/unit-02.json#0"}
    assert not [e for e in relates if claim_keys & set(e["props"].get("event_keys") or [])]
    assert not [e for e in relates if e["props"]["name"] == "IS"]
    assert not [n for n in out["nodes"] if n["props"].get("name") == "Ana's name"]
    assert ana[0]["props"]["summary"] == ""
    assert {n["props"]["name"] for n in out["nodes"] if "Episodic" in n["labels_neo4j"]} == {"m1", "m2", "m3"}
    assert out["dedup"]["identity_claim_events_dropped"] == 2
    assert out["dedup"]["identity_joins"] == 1 and out["dedup"]["identity_conflicts"] == []


def _rec(i: int, mid: str, speaker: str, text: str, roleset: str, anchor: int, roles: dict) -> dict:
    """One record with one event; ``roles`` maps a role to (text, span) or to a full role dict."""
    spec = {r: v if isinstance(v, dict) else {"text": v[0], "span": v[1]} for r, v in roles.items()}
    return {
        "unit": f"0000/unit-{i:02d}.json",
        "message_id": mid,
        "session_id": "s1",
        "speaker": speaker,
        "text": text,
        "timestamp": f"2026-10-14T00:{i:02d}:00Z",
        "capture_status": "captured",
        "events": [{"roleset": roleset, "anchor": [anchor], "roles": spec}],
    }


def _me(speaker: str) -> dict:
    return {"text": "I", "span": [0], "referent": {"status": "deixis", "text": speaker}}


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_a_state_collapsed_by_the_join_keeps_its_summary_line():
    """Bo: "Dee gave Ana a pear" after Ana's "my name is Dee": its two participants are one person now, so the edge
    Dee -> Ana is not written (a self-loop), and the one-participant path writes the line on Ana's node."""
    records = [_records()[0]]
    records.append(
        _rec(
            1,
            "m2",
            "Bo",
            "Dee gave Ana a pear",
            "give.01",
            1,
            {"arg0": ("Dee", [0]), "arg2": ("Ana", [2]), "arg1": ("a pear", [3, 4])},
        )
    )
    out = run(records)
    assert validate_rows(out["nodes"], out["edges"], previous=out["external_endpoints"]) == []
    assert all(e["source"] != e["target"] for e in out["edges"] if e["type"] == "RELATES_TO")
    ana = [n for n in _people(out) if n["props"]["name"] == "Ana"]
    assert len(ana) == 1 and "pear" in ana[0]["props"]["summary"]
    assert out["dedup"]["identity_self_facts_dropped"] == 1


def test_an_unmentioned_join_still_materialises_the_person():
    """Bo: "Dee is Cy" (neither a speaker nor mentioned elsewhere): the claim event writes no Event, fact or line, but
    the joined person exists, with ``identity_aliases``, and the message MENTIONS it."""
    records = [_records()[1], _rec(5, "m6", "Bo", "Dee is Cy", "be.01", 1, {"arg1": ("Dee", [0]), "arg2": ("Cy", [2])})]
    out = run(records)
    assert validate_rows(out["nodes"], out["edges"], previous=out["external_endpoints"]) == []
    joined = [n for n in out["nodes"] if "identity_aliases" in n["props"]]
    assert len(joined) == 1 and "Person" in joined[0]["labels_neo4j"]
    names = {joined[0]["props"]["name"], *joined[0]["props"]["identity_aliases"]}
    assert names == {"Dee", "Cy"} and joined[0]["props"]["summary"] == ""
    episodic = next(
        n["props"]["uuid"]
        for n in out["nodes"]
        if n["props"].get("message_id") == "m6" and "Episodic" in n["labels_neo4j"]
    )
    assert any(
        e["type"] == "MENTIONS" and e["source"] == episodic and e["target"] == joined[0]["props"]["uuid"]
        for e in out["edges"]
    )
    assert not [n for n in out["nodes"] if n["props"].get("roleset") == "be.01"]
    assert not [e for e in out["edges"] if "0000/unit-05.json#0" in (e["props"].get("event_keys") or [])]
    assert out["dedup"]["identity_claim_events_dropped"] == 1


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_a_common_noun_name_joins_only_when_capitalised():
    """After "my name is Rose": "I met Rose" joins; "I like rose" and "I like the roses" do not (rose is a common
    noun in WordNet). "dee" is not, so lowercase "I met dee" joins (the later-mentions test)."""
    records = [
        _rec(0, "m1", "Ana", "my name is Rose", "be.01", 2, {"arg1": ("my name", [0, 1]), "arg2": ("Rose", [3])}),
        _rec(1, "m2", "Bo", "I met Rose", "meet.01", 1, {"arg0": _me("Bo"), "arg1": ("Rose", [2])}),
        _rec(2, "m3", "Bo", "I like rose", "like.01", 1, {"arg0": _me("Bo"), "arg1": ("rose", [2])}),
        _rec(3, "m4", "Bo", "I like the roses", "like.01", 1, {"arg0": _me("Bo"), "arg1": ("the roses", [2, 3])}),
    ]
    out = run(records)
    ana = [n for n in _people(out) if n["props"]["name"] == "Ana"]
    assert len(ana) == 1 and ana[0]["props"]["identity_aliases"] == ["Rose"]
    uuid = ana[0]["props"]["uuid"]
    events = {n["props"]["message_id"]: n["props"]["uuid"] for n in out["nodes"] if "Event" in n["labels_neo4j"]}
    arg1 = {
        mid: [e["target"] for e in out["edges"] if e["source"] == evt and e["props"].get("role") == "arg1"]
        for mid, evt in events.items()
    }
    assert arg1["m2"] == [uuid]
    assert uuid not in arg1["m3"] and uuid not in arg1["m4"] and len(arg1["m4"]) == 1


def test_without_claims_nothing_changes():
    out = run(_records(without=frozenset({"m1", "m3"})))
    assert not [n for n in out["nodes"] if "identity_aliases" in n["props"]]
    # The identity stats keys are written only when joins or conflicts exist (no-claim stats stay as before).
    assert out["dedup"].get("identity_joins", 0) == 0
    assert not {"identity_joins", "identity_conflicts", "identity_self_facts_dropped"} & set(out["dedup"])


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_later_mentions_of_the_alias_land_on_the_joined_entity():
    """Every key form of the alias: lowercase "dee" (exact and lemma key) lands on Ana; the possessive "Dee's" lands
    on the entity "Ana's" gets (the canonical's possessive), whether or not "Ana's" is mentioned."""
    records = _records()
    later = [
        ("m4", "Bo", "I met dee", "dee"),
        ("m5", "Bo", "I visited Dee's", "Dee's"),
        ("m6", "Bo", "I visited Ana's", "Ana's"),
    ]
    for i, (mid, speaker, text, name) in enumerate(later, start=len(records)):
        records.append(
            {
                "unit": f"0000/unit-{i:02d}.json",
                "message_id": mid,
                "session_id": "s1",
                "speaker": speaker,
                "text": text,
                "timestamp": f"2026-10-14T00:0{i}:00Z",
                "capture_status": "captured",
                "events": [
                    {
                        "roleset": "meet.01",
                        "anchor": [1],
                        "roles": {
                            "arg0": {"text": "I", "span": [0], "referent": {"status": "deixis", "text": speaker}},
                            "arg1": {"text": name, "span": [2]},
                        },
                    }
                ],
            }
        )

    def arg1_targets(out: dict) -> dict[str, list[str]]:
        events = {
            n["props"]["message_id"]: n["props"]["uuid"] for n in out["nodes"] if n["props"].get("roleset") == "meet.01"
        }
        return {
            mid: [e["target"] for e in out["edges"] if e["source"] == evt and e["props"].get("role") == "arg1"]
            for mid, evt in events.items()
        }

    out = run(records)
    assert validate_rows(out["nodes"], out["edges"], previous=out["external_endpoints"]) == []
    ana = [n for n in _people(out) if n["props"]["name"] == "Ana"]
    assert len(ana) == 1
    uuid = ana[0]["props"]["uuid"]
    targets = arg1_targets(out)
    assert targets["m4"] == [uuid]
    assert len(targets["m6"]) == 1 and targets["m6"] != [uuid]
    assert targets["m5"] == targets["m6"]  # "Dee's" and "Ana's" are one entity

    # Without "Ana's" in the batch, "Dee's" still lands on the uuid "Ana's" gets, and that node is named "Ana's".
    alone = run(records[:-1])
    assert validate_rows(alone["nodes"], alone["edges"], previous=alone["external_endpoints"]) == []
    assert arg1_targets(alone)["m5"] == targets["m6"]
    assert [n["props"]["name"] for n in alone["nodes"] if [n["props"]["uuid"]] == targets["m6"]] == ["Ana's"]


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_a_refused_claim_event_keeps_its_fact():
    """m4 "Dee is Bo" contradicts Ana's self-declaration: the claim is refused (a conflict), so its event is routed
    as before the joins (a fact edge, here from the joined entity to Bo)."""
    records = _records()
    records.append(
        {
            "unit": "0000/unit-03.json",
            "message_id": "m4",
            "session_id": "s1",
            "speaker": "Cy",
            "text": "Dee is Bo",
            "timestamp": "2026-10-14T00:03:00Z",
            "capture_status": "captured",
            "events": [
                {
                    "roleset": "be.01",
                    "anchor": [1],
                    "roles": {"arg1": {"text": "Dee", "span": [0]}, "arg2": {"text": "Bo", "span": [2]}},
                }
            ],
        }
    )
    out = run(records)
    assert len(out["dedup"]["identity_conflicts"]) == 1 and out["dedup"]["identity_claim_events_dropped"] == 2
    facts = [e for e in out["edges"] if "0000/unit-03.json#0" in (e["props"].get("event_keys") or [])]
    assert [e["props"]["name"] for e in facts] == ["IS"]


def _arg_targets(out: dict, role: str) -> dict[str, list[str]]:
    events = {n["props"]["message_id"]: n["props"]["uuid"] for n in out["nodes"] if "Event" in n["labels_neo4j"]}
    return {
        mid: [e["target"] for e in out["edges"] if e["source"] == evt and e["props"].get("role") == role]
        for mid, evt in events.items()
    }


def _ana(out: dict) -> dict:
    ana = [n for n in _people(out) if n["props"]["name"] == "Ana"]
    assert len(ana) == 1
    return ana[0]


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_a_lowercase_mention_of_a_proper_only_name_joins():
    """WordNet knows "Mary" only as a proper noun (no noun lemma spelled "mary"), so after Ana's "I am Mary",
    "mary likes mangoes" lands on Ana."""
    records = [
        _rec(0, "m1", "Ana", "I am Mary", "be.01", 1, {"arg1": _me("Ana"), "arg2": ("Mary", [2])}),
        _rec(1, "m2", "Bo", "mary likes mangoes", "like.01", 1, {"arg0": ("mary", [0]), "arg1": ("mangoes", [2])}),
    ]
    out = run(records)
    ana = _ana(out)
    assert ana["props"]["identity_aliases"] == ["Mary"]
    assert _arg_targets(out, "arg0")["m2"] == [ana["props"]["uuid"]]


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet corpus not installed (mk setup --component wordnet)")
def test_a_common_noun_alias_joins_only_capitalised_and_not_sentence_initial():
    """After Ana's "my name is Egg": "Who cooked the egg for breakfast?" and "Egg is life." do not land on Ana (egg is
    a common noun; the second is capitalised only because it starts the sentence); "I asked Egg about it" does."""
    records = [
        _rec(0, "m1", "Ana", "my name is Egg", "be.01", 2, {"arg1": ("my name", [0, 1]), "arg2": ("Egg", [3])}),
        _rec(
            1,
            "m2",
            "Bo",
            "Who cooked the egg for breakfast?",
            "like.01",
            1,
            {"arg0": ("Who", [0]), "arg1": ("the egg", [2, 3])},
        ),
        _rec(2, "m3", "Bo", "Egg is life.", "like.01", 1, {"arg0": ("Egg", [0]), "arg1": ("life", [2])}),
        _rec(3, "m4", "Bo", "I asked Egg about it", "meet.01", 1, {"arg0": _me("Bo"), "arg1": ("Egg", [2])}),
    ]
    out = run(records)
    ana = _ana(out)
    assert ana["props"]["identity_aliases"] == ["Egg"]
    uuid = ana["props"]["uuid"]
    arg0, arg1 = _arg_targets(out, "arg0"), _arg_targets(out, "arg1")
    assert uuid not in arg1.get("m2", []) and uuid not in arg0.get("m3", [])
    assert arg1["m4"] == [uuid]


def test_a_picked_referent_of_a_capitalised_alias_joins():
    """A pronoun resolved to an antecedent ("she" -> "Dee") has no occurrence of the name in its own unit: the
    antecedent's own form decides (capitalised, or not a common noun)."""
    records = [
        _records()[0],
        _rec(
            1,
            "m2",
            "Bo",
            "she likes pears",
            "like.01",
            1,
            {
                "arg0": {"text": "she", "span": [0], "referent": {"status": "picked", "text": "Dee"}},
                "arg1": ("pears", [2]),
            },
        ),
    ]
    out = run(records)
    assert _arg_targets(out, "arg0")["m2"] == [_ana(out)["props"]["uuid"]]
