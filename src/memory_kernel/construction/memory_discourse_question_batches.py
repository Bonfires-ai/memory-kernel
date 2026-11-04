"""Finite transport partitions; every question retains the full frozen context."""

from collections.abc import Mapping
from copy import deepcopy

from memory_kernel.construction.memory_jev_archive import read_jev_archive
from memory_kernel.construction.memory_jev_wire_ids import MAX_WIRE_BYTES
from memory_kernel.construction.memory_mention_jev import MAX_QUESTIONS
from memory_kernel.construction.memory_semantic_jev import check_seal, seal
from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256

BATCH_WAVE_SCHEMA = "discourse-question-wave/v2"
BATCH_ARCHIVE_SCHEMA = "discourse-question-batch-archive/v1"
MAX_BATCHES = 16


def question_batch_policy():
    return seal({"schema_version": "discourse-question-batch-policy/v1",
        "partition": "sorted_question_ids_greedy_prefix", "context": "complete_frozen_state_per_batch",
        "max_batches": MAX_BATCHES, "max_questions_per_batch": MAX_QUESTIONS,
        "max_wire_bytes": MAX_WIRE_BYTES, "partial_answers": "not_admissible"})


def check_question_batch_policy(policy):
    if policy != question_batch_policy():
        raise ValueError("question batch policy differs from the closed supported policy")


def partition_question_wire(wire, policy, *, round):
    from memory_kernel.construction.memory_discourse_questions import DiscourseQuestionBudgetExhausted

    check_question_batch_policy(policy)
    logical_sha = content_sha256(wire)

    def request(index, questions):
        state = deepcopy(wire["state"])
        state["question_batch"] = {"logical_request_sha256": logical_sha,
            "policy_sha256": policy["content_sha256"], "round": round, "index": index}
        return {"model": wire["model"], "state": state, "questions": deepcopy(questions)}

    batches, pending = [], {}
    for qid in sorted(wire["questions"]):
        trial = {**pending, qid: wire["questions"][qid]}
        if len(trial) > MAX_QUESTIONS or len(canonical_json(request(len(batches), trial)).encode()) > MAX_WIRE_BYTES:
            if pending:
                batches.append(request(len(batches), pending))
            pending = {qid: wire["questions"][qid]}
            if len(canonical_json(request(len(batches), pending)).encode()) > MAX_WIRE_BYTES:
                raise DiscourseQuestionBudgetExhausted("complete context plus one question exceeds batch wire budget")
        else:
            pending = trial
        if len(batches) >= MAX_BATCHES:
            raise DiscourseQuestionBudgetExhausted("question batch count exceeded")
    if pending:
        batches.append(request(len(batches), pending))
    if not batches:
        raise ValueError("question batches cannot be empty")
    return batches


def logical_question_wire(wave):
    if wave["schema_version"] != BATCH_WAVE_SCHEMA:
        return deepcopy(wave["wire"])
    state = deepcopy(wave["batches"][0]["state"])
    state.pop("question_batch")
    return {"model": wave["policy"]["model"], "state": state,
        "questions": {qid: deepcopy(question) for batch in wave["batches"]
                      for qid, question in batch["questions"].items()}}


def question_requests(wave, phase):
    if wave["schema_version"] != BATCH_WAVE_SCHEMA:
        return [(phase, wave["wire"])]
    return [(f"{phase}:{index:02d}", wire) for index, wire in enumerate(wave["batches"])]


def question_archive(wave, archives):
    if wave["schema_version"] != BATCH_WAVE_SCHEMA:
        if len(archives) != 1:
            raise ValueError("single wave requires one archive")
        return deepcopy(archives[0])
    if len(archives) != len(wave["batches"]):
        raise ValueError("question batch archive inventory differs")
    return seal({"schema_version": BATCH_ARCHIVE_SCHEMA, "wave_sha256": wave["content_sha256"],
        "archives": deepcopy(archives)})


def question_archives(wave, archive):
    if wave["schema_version"] != BATCH_WAVE_SCHEMA:
        return [archive]
    check_seal(archive, BATCH_ARCHIVE_SCHEMA)
    if (set(archive) != {"schema_version", "content_sha256", "wave_sha256", "archives"}
            or archive["wave_sha256"] != wave["content_sha256"]
            or type(archive["archives"]) is not list
            or len(archive["archives"]) != len(wave["batches"])):
        raise ValueError("question batch receipt inventory or wave binding differs")
    return archive["archives"]


def read_question_batch_archive(wave, archive):
    evidence = [read_jev_archive(wire, receipt, wave["policy"])
        for wire, receipt in zip(wave["batches"], question_archives(wave, archive), strict=True)]
    answers = {}
    for row in evidence:
        if not isinstance(row["answers"], Mapping) or set(answers) & set(row["answers"]):
            raise ValueError("question batches repeat answers")
        answers.update(row["answers"])
    if set(answers) != set(wave["menus"]):
        raise ValueError("question batches omit finite answers")
    return {"answers": answers, "usage": {key: sum(row["usage"][key] for row in evidence)
            for key in ("input_tokens", "output_tokens", "cost_usd")},
        "batches": [{key: value for key, value in row.items() if key != "answers"} for row in evidence],
        "archive_sha256": archive["content_sha256"], "evidence_mode": wave["policy"]["evidence_mode"]}


MAX_WAVES = 4


def partition_question_waves(wire, policy, *, round):
    """Deterministic wave split for a wire whose batches exceed MAX_BATCHES (byte-identical when it fits)."""
    from memory_kernel.construction.memory_discourse_questions import DiscourseQuestionBudgetExhausted

    try:
        return [partition_question_wire(wire, policy, round=round)]
    except DiscourseQuestionBudgetExhausted as exc:
        if "batch count" not in str(exc):
            raise
    ids = sorted(wire["questions"])
    waves, start = [], 0
    while start < len(ids):
        lo, hi, best = start + 1, len(ids), None
        while lo <= hi:  # longest prefix from ``start`` that partitions within MAX_BATCHES
            mid = (lo + hi) // 2
            sub = {**wire, "questions": {q: wire["questions"][q] for q in ids[start:mid]}}
            try:
                best = (mid, partition_question_wire(sub, policy, round=round))
                lo = mid + 1
            except DiscourseQuestionBudgetExhausted:
                hi = mid - 1
        if best is None:
            raise DiscourseQuestionBudgetExhausted("one question exceeds a whole wave")
        start, batches = best
        waves.append(batches)
        if len(waves) > MAX_WAVES:
            raise DiscourseQuestionBudgetExhausted("question wave count exceeded")
    return waves
