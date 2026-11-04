"""A roleset whose eventuality answer is stored is compiled, checked against its sha and never sent (spec 7.4)."""

from __future__ import annotations

import pytest

from memory_kernel.construction import memory_jev_archive
from memory_kernel.pipeline import llm
from memory_kernel.pipeline.stages import _events_core as ev

YES = ev.YES


class _Ledger:
    def __init__(self, directory, sent):
        self.directory, self.policy, self.sent = directory, None, sent

    def summary(self):
        return {"actual_usd": "0"}

    def evaluate(self, wire):
        self.sent.append(wire["state"]["roleset"])
        return wire

    def close(self):
        pass


class _Shard:
    def __init__(self, directory, sent):
        self.directory, self.sent = directory, sent

    def load(self):
        return {}, {}, None

    def open(self, manifest, provenance):
        return _Ledger(self.directory, self.sent)


@pytest.fixture
def asked(monkeypatch):
    """Every roleset a wire was sent for; the archive reader answers each question 0.9 yes."""
    sent: list[str] = []
    monkeypatch.setattr(
        memory_jev_archive,
        "read_jev_archive",
        lambda wire, archive, policy: {"answers": {q: {"probabilities": {YES: 0.9}} for q in wire["questions"]}},
    )
    return sent


def _row(gloss="to hold", examples=("Ines: The warp holds.",)):
    return {"gloss": gloss, "gloss_source": "chain", "examples": list(examples), "predicates": 2}


def _stored(batcher, roleset, row, label="occurrence"):
    compiled = ev.compile_roleset(roleset, row["gloss"], row["examples"], batcher=batcher)
    p = {k: 0.1 for k in ev.EVENTUALITIES}
    return {"label": label, "p_yes": p, "compiled_sha256": compiled["content_sha256"]}


def test_settled_roleset_sends_no_wire(tmp_path, asked):
    batcher = llm.QuestionBatcher()
    inventory = {"hold.01": _row(), "weave.01": _row("to weave", ("Ines: A new warp.",))}
    settled = {"hold.01": _stored(batcher, "hold.01", inventory["hold.01"], "state")}
    labels, _ = ev.ask_eventuality(_Shard(tmp_path, asked), inventory, live=True, batcher=batcher, settled=settled)
    assert asked == ["weave.01"]  # hold.01 sent nothing
    assert labels["hold.01"] == settled["hold.01"]
    assert set(labels["weave.01"]) == {"label", "p_yes", "compiled_sha256"}
    assert ev.eventuality_wires(tmp_path, inventory, batcher=batcher, settled=settled) == ev.eventuality_wires(
        tmp_path, {"weave.01": inventory["weave.01"]}, batcher=batcher
    )


def test_settled_roleset_with_another_compiled_sha_refuses(tmp_path, asked):
    batcher = llm.QuestionBatcher()
    inventory = {"hold.01": _row()}
    stored = {**_stored(batcher, "hold.01", inventory["hold.01"]), "compiled_sha256": "0" * 64}
    with pytest.raises(ValueError, match=r"hold\.01: settled compiled_sha256 differs"):
        ev.ask_eventuality(_Shard(tmp_path, asked), inventory, live=True, batcher=batcher, settled={"hold.01": stored})
    assert asked == []


def test_frozen_rolesets_without_a_label_still_replay(tmp_path, asked):
    batcher = llm.QuestionBatcher()
    inventory = {"hold.01": _row()}
    frozen = {"hold.01": {"gloss": "to hold", "examples": ["Ines: The warp holds."]}}  # a 1.8 row: no answer stored
    assert ev._settled(frozen) == {}
    labels, _ = ev.ask_eventuality(
        _Shard(tmp_path, asked), inventory, live=True, batcher=batcher, settled=ev._settled(frozen)
    )
    assert asked == ["hold.01"] and labels["hold.01"]["label"] == "occurrence"


def test_new_rolesets_carry_the_settled_answer():
    batcher = llm.QuestionBatcher()
    old = _row()
    stored = _stored(batcher, "hold.01", old, "state")
    frozen = {"hold.01": {**old, **stored}}
    records = [
        {"speaker": "Ines", "text": "The warp holds.", "events": [{"roleset": "hold.01"}, {"roleset": "weave.01"}]}
    ]
    inventory = ev.roleset_inventory(records, [], (), lambda d: str(d), frozen=frozen)
    labels = {
        "hold.01": {k: stored[k] for k in ("label", "p_yes", "compiled_sha256")},
        "weave.01": {"label": "occurrence", "p_yes": {"x": 1}, "compiled_sha256": "ab"},
    }
    new = ev._new_rolesets(inventory, labels, frozen)
    assert list(new) == ["weave.01"]
    assert set(new["weave.01"]) == {"gloss", "gloss_source", "examples", "label", "p_yes", "compiled_sha256"}
    assert new["weave.01"]["label"] == "occurrence" and new["weave.01"]["gloss_source"] == "lemma"


def test_a_sha_mismatch_refuses_before_any_wire_or_prefetch(tmp_path, asked):
    batcher = llm.QuestionBatcher()
    inventory = {"aaa.01": _row("to a", ("Ines: a.",)), "hold.01": _row()}  # the unsettled one comes first
    stored = {**_stored(batcher, "hold.01", inventory["hold.01"]), "compiled_sha256": "0" * 64}
    prefetched = []
    with pytest.raises(ValueError, match=r"hold\.01: settled compiled_sha256 differs"):
        ev.ask_eventuality(
            _Shard(tmp_path, asked),
            inventory,
            live=True,
            batcher=batcher,
            prefetch=lambda ledger, wires: prefetched.append(1),
            settled={"hold.01": stored},
        )
    assert asked == [] and prefetched == []


def test_a_row_with_a_null_answer_part_is_not_settled():
    full = {"label": "state", "p_yes": {"state": 0.1}, "compiled_sha256": "ab"}
    rows = {"ok": {**full}, **{f"no_{k}": {**full, k: None} for k in full}, "bare": {"gloss": "g"}}
    assert list(ev._settled(rows)) == ["ok"]
