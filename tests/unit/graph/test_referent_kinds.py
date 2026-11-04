"""``referent_policy="structure"`` decides one kind per referent from evidence (Unit B): a roster person, kind nouns the
text supplies, the referent's own head class, then a location role with a location-compatible mention. ``phrase``
keeps today's role-conferred kinds. Invented names only."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import replace

from test_forward_resolution import NEEDS_WORDNET, fold

from memory_kernel.graph.g1_kinds import KindEvidence, decide, frame_evidence, noun_kind
from memory_kernel.graph.g1_mentions import Mention
from memory_kernel.graph.g1_names import ARTIFACT, ORGANISATION, PERSON, PLACE
from memory_kernel.graph.g1_projection import project
from memory_kernel.graph.kg_contract import validate_rows
from memory_kernel.graph.kg_writer import kind_label_clause, write_rows
from memory_kernel.pipeline import chain as chain_mod
from memory_kernel.pipeline.chain import Chain
from memory_kernel.pipeline.stages import write

GROUP = "g-test"
LABEL = {"be.01": "state", "meet.01": "occurrence", "fix.01": "occurrence", "appoint.01": "occurrence"}
ROLE_FUNCTIONS = {
    "be.01": {"arg1": "PPT", "arg2": "PRD"},
    "meet.01": {"argm-loc": "LOC"},
    "appoint.01": {"arg0": "PAG", "arg1": "PPT", "arg2": "PRD"},
}


def test_kind_evidence_order():
    supplied = KindEvidence(supplied=Counter({ORGANISATION: 1}))
    assert decide(KindEvidence(person=True, supplied=Counter({ORGANISATION: 2}))) == PERSON
    assert decide(KindEvidence(supplied=Counter({ORGANISATION: 1}), head=Counter({ARTIFACT: 3}))) == ORGANISATION
    assert decide(KindEvidence(head=Counter({ARTIFACT: 1}), loc_roles=2, located=True)) == ARTIFACT
    assert decide(KindEvidence(loc_roles=1, located=True)) == PLACE
    assert decide(KindEvidence(loc_roles=1, located=False)) is None
    assert decide(KindEvidence()) is None
    assert decide(KindEvidence(supplied=Counter({ARTIFACT: 1, ORGANISATION: 1}))) == ORGANISATION
    assert decide(KindEvidence(supplied=Counter({ARTIFACT: 2, ORGANISATION: 1}))) == ARTIFACT
    ev = KindEvidence(person=True, supplied=Counter({ORGANISATION: 2}), head=Counter({PLACE: 1}), loc_roles=3)
    assert KindEvidence.from_row(ev.to_row()) == ev
    assert list(ev.to_row()) == sorted(ev.to_row())
    supplied.add(KindEvidence(head=Counter({ARTIFACT: 1}), loc_roles=1, located=True))
    assert supplied == KindEvidence(False, Counter({ORGANISATION: 1}), Counter({ARTIFACT: 1}), 1, True)


@NEEDS_WORDNET
def test_organisation_only_under_organization():
    """``noun.group`` gives Organisation only when the noun's first sense is organization.n.01 or below it."""
    assert [noun_kind(w) for w in ("team", "company", "agency")] == [ORGANISATION] * 3
    assert [noun_kind(w) for w in ("meeting", "generation", "audience")] == [None] * 3


@NEEDS_WORDNET
def test_wordnet_reads_the_form_it_is_given():
    """The lookup takes a lemma; WordNet's own morphology reads a plural, with no suffix rule of ours."""
    assert noun_kind("boxes") == noun_kind("box") == ARTIFACT
    assert noun_kind("agencies") == noun_kind("agency") == ORGANISATION


def rec(n: int, text: str, roleset: str, anchor: int, roles: dict[str, tuple[str, list[int]]]) -> dict:
    return {
        "unit": f"{n:04d}/unit-00.json",
        "message_id": f"m{n}",
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


def build(records: list[dict], policy: str, chain: Chain | None = None, number: int = 0, **kw) -> dict:
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
        role_functions=ROLE_FUNCTIONS,
        referent_policy=policy,
        chain=chain,
        run=number,
        conversation_of={r["message_id"]: "k" for r in records} if chain is not None else None,
        **kw,
    )
    assert validate_rows(out["nodes"], out["edges"], previous=out["external_endpoints"]) == []
    return out


@NEEDS_WORDNET
def test_structure_kinds_from_evidence():
    records = [
        rec(1, "Brask is a company.", "be.01", 1, {"arg1": ("Brask", [0]), "arg2": ("a company", [2, 3])}),
        rec(2, "Ana met Kira at Brask.", "meet.01", 1,
            {"arg0": ("Ana", [0]), "arg1": ("Kira", [2]), "argm-loc": ("at Brask", [3, 4])}),
        rec(3, "Ana fixed the device.", "fix.01", 1, {"arg1": ("the device", [2, 3])}),
        rec(4, "Ana met the journalist.", "meet.01", 1, {"arg1": ("the journalist", [2, 3])}),
    ]  # fmt: skip

    def labels(out: dict) -> dict[str, list[str]]:
        return {
            n["props"]["name"]: n["labels_neo4j"]
            for n in out["nodes"]
            if "Entity" in n["labels_neo4j"] and "Event" not in n["labels_neo4j"]
        }

    out = build(records, "structure")
    kinds = labels(out)
    assert kinds["Brask"] == ["Entity", ORGANISATION]  # the PRD kind noun beats the location role
    assert kinds["device"] == ["Entity", ARTIFACT]
    assert kinds["journalist"] == ["Entity", PERSON]
    assert kinds["Ana"] == ["Entity", PERSON]
    minted = {m.display: m.kind for m in out["forward"]["minted"]}
    assert minted["journalist"] == "entity"  # a WordNet Person never joins a roster
    assert minted["Ana"] == "person"
    brask = next(n["props"]["uuid"] for n in out["nodes"] if n["props"]["name"] == "Brask")
    row = out["forward"]["kinds"][brask]
    assert row["supplied"] == {ORGANISATION: 1} and row["loc_roles"] == 1 and row["located"] is False
    entities = {
        n["props"]["uuid"]
        for n in out["nodes"]
        if n["labels_neo4j"][0] == "Entity" and "Event" not in n["labels_neo4j"]
    }
    assert set(out["forward"]["kinds"]) == entities

    phrase = build(records, "phrase")
    assert PLACE in labels(phrase)["Brask"]  # today: the LOC role confers Place
    assert "kinds" not in phrase["forward"]
    assert {m.display for m in phrase["forward"]["minted"] if m.kind == "person"} == {
        m.display for m in out["forward"]["minted"] if m.kind == "person"
    }

    # the PRD kind noun is said of the proto-patient (PPT) only, never of the agent
    director = Mention((4,), "director", "director", "director", False, ())
    tags = {"arg0": "PAG", "arg1": "PPT", "arg2": "PRD"}
    assert frame_evidence(tags, {"arg0": None, "arg1": None, "arg2": director}) == {
        "arg1": KindEvidence(supplied=Counter({PERSON: 1}))
    }
    appoint = build([rec(5, "The board appointed Zelda director.", "appoint.01", 2,
                         {"arg0": ("The board", [0, 1]), "arg1": ("Zelda", [3]), "arg2": ("director", [4])})],
                    "structure")  # fmt: skip
    assert labels(appoint)["Zelda"] == ["Entity", PERSON]
    assert PERSON not in labels(appoint)["board"]
    board = next(n["props"]["uuid"] for n in appoint["nodes"] if n["props"]["name"] == "board")
    assert appoint["forward"]["kinds"][board]["supplied"] == {}

    # an adjective predicate supplies no kind ("rich" is no noun of a group)
    rich = build([rec(6, "Kira is rich.", "be.01", 1, {"arg1": ("Kira", [0]), "arg2": ("rich", [2])})], "structure")
    kira = next(n["props"]["uuid"] for n in rich["nodes"] if n["props"]["name"] == "Kira")
    assert labels(rich)["Kira"] == ["Entity"] and rich["forward"]["kinds"][kira]["supplied"] == {}

    # a name compound is its own referent with its own head class; the name inside is mentioned, supplied nothing
    spy = build([rec(7, "Ana fixed the Zorvex spyware.", "fix.01", 1, {"arg1": ("the Zorvex spyware", [2, 3, 4])})],
                "structure")  # fmt: skip
    assert labels(spy)["Zorvex spyware"] == ["Entity", ARTIFACT] and labels(spy)["Zorvex"] == ["Entity"]
    uuid = {n["props"]["name"]: n["props"]["uuid"] for n in spy["nodes"]}
    assert spy["forward"]["kinds"][uuid["Zorvex"]]["supplied"] == {}
    m7 = next(n["props"]["uuid"] for n in spy["nodes"] if "Episodic" in n["labels_neo4j"])
    assert any(e["type"] == "MENTIONS" and (e["source"], e["target"]) == (m7, uuid["Zorvex"]) for e in spy["edges"])


class _Recorder:
    """A Neo4j driver that records every statement and answers with no rows (``write_rows`` without a database)."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, dict]] = []

    def session(self, **_kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def run(self, query: str, **params):
        self.queries.append((query, params))
        return self

    async def single(self):
        return {"c": 0, "n": 0}

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def node_of(out: dict, uuid: str) -> dict:
    return next(n for n in out["nodes"] if n["props"]["uuid"] == uuid)


def folded(chain: Chain, out: dict, records: list[dict], number: int) -> Chain:
    """``fold`` plus what ``chain.advance`` folds of a structure run's kinds: every row it emits replaces the stored."""
    after = fold(chain, out, records, number)
    if "kinds" not in out["forward"]:
        return after
    return replace(after, kinds={**(chain.kinds or {}), **out["forward"]["kinds"]})


@NEEDS_WORDNET
def test_an_append_recomputes_a_stored_kind():
    """A structure bonfire keeps each referent's kind evidence in the chain: an append's evidence joins the stored
    evidence, one kind is decided again and the writer replaces the stale kind label (Place -> Organisation). Under
    phrase a stored node's labels stay frozen."""
    boot = [
        rec(
            1,
            "Ana met Kira at the city.",
            "meet.01",
            1,
            {"arg0": ("Ana", [0]), "arg1": ("Kira", [2]), "argm-loc": ("at the city", [3, 4, 5])},
        )
    ]
    copula = [rec(2, "The city is a company.", "be.01", 2, {"arg1": ("The city", [0, 1]), "arg2": ("a company", [3, 4])})]  # fmt: skip

    first = build(boot, "structure")
    city = next(n["props"]["uuid"] for n in first["nodes"] if n["props"]["name"] == "city")
    stored = node_of(first, city)
    assert stored["labels_neo4j"] == ["Entity", PLACE]
    chain = folded(Chain(), first, boot, 0)
    view = json.loads(json.dumps(chain_mod.projection_view(chain)))
    assert chain_mod.from_projection_view(view).kinds == chain.kinds  # the run reads the chain through its view

    out = build(copula, "structure", chain_mod.from_projection_view(view), 1)
    row = node_of(out, city)
    assert row["labels_neo4j"] == ["Entity", ORGANISATION] and row["kinds_replace"] is True
    assert city in chain.entities and out["forward"]["minted"] == []
    after = folded(chain, out, copula, 1)
    assert after.kinds is not None and after.kinds[city]["supplied"] == {ORGANISATION: 1}
    assert "labels" not in write._frozen_changes(row, stored)
    assert "labels" in write._frozen_changes({k: v for k, v in row.items() if k != "kinds_replace"}, stored)
    record = write.forward_record(out["forward"], run=1, episodic=[], own=set())
    assert record["kinds"][city] == after.kinds[city]

    remove = kind_label_clause(["Entity", ORGANISATION], True)
    assert remove == f" REMOVE n:{PERSON}:{ARTIFACT}:{PLACE}" and kind_label_clause(["Entity"], False) == ""
    driver = _Recorder()
    rows = {"nodes": [row], "edges": [], "external_endpoints": {}}
    asyncio.run(write_rows(driver, rows, group_id=GROUP, embed=lambda texts: [[0.0]] * len(texts), append=True))
    (query,) = [q for q, p in driver.queries if "MERGE (n:" in q and p["rows"][0]["uuid"] == city]
    assert remove in query

    # a full build reusing a stored Place node: its kind is decided again, and replaces the stored label
    existing = [{"uuid": city, "name": "city", "labels": ["Entity", PLACE], "name_key": "city"}]
    rebuilt = node_of(build(copula, "structure", existing=existing), city)
    assert rebuilt["labels_neo4j"] == ["Entity", ORGANISATION] and rebuilt["kinds_replace"] is True

    # phrase: the stored node's labels are frozen, no flag, no kinds in the chain or its view
    first = build(boot, "phrase")
    chain = folded(Chain(), first, boot, 0)
    assert chain.kinds is None and "kinds" not in chain_mod.projection_view(chain)
    row = node_of(build(copula, "phrase", chain, 1), city)
    assert row["labels_neo4j"] == ["Entity"] and "kinds_replace" not in row
    assert all("kinds_replace" not in n for n in first["nodes"])
