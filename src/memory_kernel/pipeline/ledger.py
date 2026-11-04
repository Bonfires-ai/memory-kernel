"""The round ledger: one fail-closed spending shard per conversation ($2 up to the reference conversation's size).

A shard's sealed caps (``budget_cap_usd``, ``integrated_cap_usd``) bound the new holds of a build that opens it with no
``limit_usd``: the bootstrap and any other plain open. An append run of a bonfire opens it with ``limit_usd``
(``Shard.limit_usd``, ``bonfire.append_limit``): the run's own budget over the shard's total at the run's start, in
place of both sealed caps. A cap binds at ``reserve`` only: a shard whose spend is already past it (appends took it
there) still opens and replays its archived answers; a new hold past it is refused.

Ported from the reference harness:
  * ``IntegratedBudget`` / ``write_new`` / ``MODEL`` / ``JEV_INPUT_PER_MILLION``  <- the harness's Jev budget
  * ``RoundBudget`` / ``CAP``                                                    <- the harness's round budget
  * ``prepare_round_execution`` / ``load_round_execution``                       <- the harness's round run
  * ``make_empty_pairs`` / ``make_execution``                                    <- the harness's ledger

Every ledger file is written by the ported bodies, as the frozen code writes it, plus the manifest's ``ledger_schema`` key. The edits are these
rebindings:
  * the Jev sender is INJECTED (``sender=``, ``llm.JevSender``) and the key is passed (``api_key=``);
    the frozen ``provider_sender()`` (a module loaded by path into a global on the first cache miss) and
    ``os.environ["OPENROUTER_API_KEY"]`` are gone.
  * the stage-cap guard on ``reserve`` and the new-send hook on ``evaluate`` are constructor
    arguments of the instance (``reserve_guard=``, ``on_new_send=``), not class mutation.
  * a prefetch pool serialises ``reserve`` / ``complete`` / ``summary`` through ``serial=`` (an instance
    attribute set for the duration of the prefetch, used by the pool's worker threads only).
  * ``load_round_execution`` opens a shard with ``ledger_schema`` whatever the code bytes; a shard without it checks
    each ``src/memory_kernel/`` pin against ``ACCEPTED_CODE_PINS`` and any other key (a harness-made shard pins the
    harness files) by re-hashing through ``provenance.code_path`` against an explicit ``harness_root``, never against
    ``__file__`` or the data root.

The pair-bearing parts of ``prepare_round_execution`` / ``load_round_execution`` (question pairs of the development conversation) are not
ported: a build's shard is question-blind, its pair manifest is empty (``make_empty_pairs``). A shard that holds
pairs is refused with a clear error; the bench side keeps the harness for it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from memory_kernel.construction.memory_jev_archive import RECEIPT_SCHEMA, read_jev_archive
from memory_kernel.construction.memory_jev_wire_ids import MAX_WIRE_BYTES
from memory_kernel.construction.memory_semantic_jev import check_seal, seal, semantic_policy, sha
from memory_kernel.construction.mined_affordance_contract import canonical_json, content_sha256
from memory_kernel.model_family import model_family_ok
from memory_kernel.pipeline.provenance import SRC_PREFIX, code_path

MODEL = "typesafe/jev-1.13"
CAP = Decimal("2")  # RoundBudget's cap at the reference conversation's size (``make_execution``'s default)
#: The integrated cap a NEW shard records in its manifest (``integrated_cap_usd``): it bounds a build that opens the
#: shard without ``limit_usd`` (the bootstrap's preflight checks its round cap against it, ``round_cap``), never an
#: append.
INTEGRATED_CAP = Decimal("25")
#: What a manifest with no ``integrated_cap_usd`` (a shard made before 1.1) enforces: those shards keep exactly $10.
LEGACY_INTEGRATED_CAP = Decimal("10")
JEV_INPUT_PER_MILLION = Decimal("0.042")
SCHEMA = "mk-execution/v1"
PAIRS_SCHEMA = "mk-pairs-manifest/v1"
#: What a shard made by ``mk`` pins: this package's ledger code and the two
#: frozen src modules. A shard made by the reference harness pins the harness's own key set instead; those sets live in
#: ``<harness root>/harness-code-set.json`` (``HARNESS_CODE_SET``), never in this package.
MK_CODE = (
    "src/memory_kernel/pipeline/ledger.py",
    "src/memory_kernel/construction/memory_discourse_question_batches.py",
    "src/memory_kernel/construction/memory_jev_archive.py",
)
#: The shard layout a new shard records as ``ledger_schema``. A shard that carries it opens whatever the bytes of the
#: ``MK_CODE`` files are; a change to what a shard means raises this number and ``load_round_execution`` branches on it.
LEDGER_SCHEMA = 1
#: ``(path, sha256)`` of the ``MK_CODE`` files a shard made before ``ledger_schema`` pinned: such a shard opens when every
#: pin it holds is listed here.
ACCEPTED_CODE_PINS: frozenset[tuple[str, str]] = frozenset(
    {
        ("src/memory_kernel/pipeline/ledger.py", "bcabda483f6824730cb88e39260c1a1d5609a2ad47e87b746090365c7e617180"),
        (
            "src/memory_kernel/construction/memory_discourse_question_batches.py",
            "fc894f56676ec29462df77f22ca9e14bd7bdfc782a5272a6e777c0e2df4ae615",
        ),
        (
            "src/memory_kernel/construction/memory_jev_archive.py",
            "6f4440e43c34644ecbf39da7fe3633c71bd3bff4d0afff8df7bc323de727f024",
        ),
    }
)
#: ``{"schema": HARNESS_CODE_SET_SCHEMA, "code_sets": [[key, ...], ...]}`` at the harness root: the key sets a
#: harness-made shard may pin.
HARNESS_CODE_SET = "harness-code-set.json"
HARNESS_CODE_SET_SCHEMA = "mk-harness-code-set/v1"
EMPTY_COHORT_SHA = hashlib.sha256(b"mk-empty-cohort: no questions, no pairs").hexdigest()


def write_new(path: Path, value: Any) -> None:
    data = value if isinstance(value, bytes) else canonical_json(value).encode()
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def settle_refused_model(directory: Path, kind: str, reason: str) -> Path:
    """A paid attempt whose response came from another model family, its cost already recorded as spent: note it in
    ``failure.json`` (no reservation retained) and keep the attempt as ``<name>-refused-<n>``, which frees the wire's
    name for a re-ask. Returns the new directory."""
    write_new(
        directory / "failure.json",
        {"type": kind, "reason": reason, "automatic_retry": False, "reservation_retained": False},
    )
    n = 1
    while (target := directory.with_name(f"{directory.name}-refused-{n}")).exists():
        n += 1
    directory.rename(target)
    return target


class JevModelRefused(ValueError):
    """A paid Jev response from another model family; its cost is settled as spent and the wire may be asked again."""


def integrated_cap(manifest: Mapping) -> Decimal:
    """The integrated cap a shard enforces: its own recorded ``integrated_cap_usd``, ``LEGACY_INTEGRATED_CAP`` when its
    manifest has none."""
    try:
        cap = Decimal(str(manifest.get("integrated_cap_usd", LEGACY_INTEGRATED_CAP)))
    except ArithmeticError:
        cap = Decimal(0)
    if not cap.is_finite() or cap <= 0:
        raise ValueError("integrated_cap_usd must be a positive amount")
    return cap


class IntegratedBudget:
    """No automatic retry, no release of an unknown send, one local writer."""

    def __init__(
        self,
        directory: Path,
        manifest: Mapping,
        provenance: Mapping,
        *,
        sender: Callable | None = None,
        api_key: str | None = None,
    ) -> None:
        self.sender, self.api_key = sender, api_key
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = (directory / "execution.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if manifest.get("budget_cap_usd") != "10":
                raise ValueError("integrated budget authorization must be exactly ten dollars")
            for name, value in (("manifest", manifest), ("provenance", provenance)):
                path = directory / (name + ".json")
                if not path.exists():
                    write_new(path, value)
                if path.read_bytes() != canonical_json(value).encode():
                    raise ValueError("immutable integrated budget pin changed")
            self.policy = semantic_policy(
                model=MODEL,
                evidence_mode="provider_receipted",
                provider_run={
                    name: {
                        "path": str((directory / (name + ".json")).resolve()),
                        "file_sha256": sha((directory / (name + ".json")).read_bytes()),
                    }
                    for name in ("manifest", "provenance")
                },
            )
            self.summary()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()

    def summary(self, *, integrated: bool = True) -> dict:
        """The budget dir's totals; ``integrated`` False reads them without holding them against the integrated cap
        (an append's ledger and the read-only ``jev.shard_budget``)."""
        manifest = json.loads((self.directory / "manifest.json").read_bytes())
        inherited = manifest.get("previous_budget")
        if inherited is not None:
            for pin in inherited["pins"]:
                if sha(Path(pin["path"]).read_bytes()) != pin["sha256"]:
                    raise ValueError("inherited budget receipt changed")
            paths = sorted(str(path.resolve()) for path in Path(inherited["directory"]).glob("*/reservation.json"))
            if paths != inherited["reservation_paths"]:
                raise ValueError("previous run made an additional paid attempt")
        cap = integrated_cap(manifest)
        prior = inherited["summary"] if inherited is not None else {}
        reserved = Decimal(prior.get("reserved_usd", "0"))
        actual = Decimal(prior.get("actual_usd", "0"))
        unresolved = Decimal(prior.get("unresolved_reservations_usd", "0"))
        completed = prior.get("completed", 0)
        held_or_spent_by_stage = {key: Decimal(value) for key, value in prior.get("held_or_spent_by_stage", {}).items()}
        reservations = sorted(self.directory.glob("*/reservation.json"))
        for path in reservations:
            row = json.loads(path.read_bytes())
            amount = Decimal(str(row["reserved_usd"]))
            if not amount.is_finite() or amount <= 0 or row["status"] != "reserved":
                raise ValueError("budget reservation is invalid")
            reserved += amount
            exposure = amount
            complete = path.parent / "completed.json"
            if complete.exists():
                receipt = json.loads(complete.read_bytes())
                if any(receipt[key] != value for key, value in row.items() if key not in {"status", "charged_usd"}):
                    raise ValueError("completed budget entry changed its reservation")
                cost = Decimal(str(receipt["receipt"]["cost_usd"]))
                if not cost.is_finite() or not 0 <= cost <= amount or receipt["status"] != "completed":
                    raise ValueError("completed budget usage exceeds hold")
                actual += cost
                exposure = cost
                completed += 1
            else:
                unresolved += amount
            held_or_spent_by_stage[row["stage"]] = held_or_spent_by_stage.get(row["stage"], Decimal(0)) + exposure
        if integrated and actual + unresolved > cap:
            raise ValueError("shared integrated budget exceeded")
        return {
            "cap_usd": str(cap),
            "integrated_cap_usd": str(cap),
            "reserved_usd": str(reserved),
            "actual_usd": str(actual),
            "held_or_spent_usd": str(actual + unresolved),
            "unresolved_reservations_usd": str(unresolved),
            "held_or_spent_by_stage": {key: str(value) for key, value in held_or_spent_by_stage.items()},
            "calls": len(reservations) + prior.get("calls", 0),
            "completed": completed,
            "unknown_or_failed": len(reservations) + prior.get("calls", 0) - completed,
        }

    def reserve(self, *, wire, stage, model, input_cap, output_cap, input_rate, output_rate):
        if (
            stage not in {"jev", "qa_answer", "qa_judge"}
            or type(input_cap) is not int
            or input_cap <= 0
            or type(output_cap) is not int
            or output_cap < 0
        ):
            raise ValueError("budget call bounds differ")
        amount = (Decimal(input_cap) * input_rate + Decimal(output_cap) * output_rate) / Decimal(1_000_000)
        request_sha = content_sha256(wire)
        directory = self.directory / (stage + "-" + request_sha)
        if directory.exists():
            raise ValueError("paid request already attempted; cold replay only, no retry")
        current = self.summary()
        manifest = json.loads((self.directory / "manifest.json").read_bytes())
        cap = integrated_cap(manifest)
        if Decimal(current["held_or_spent_usd"]) + amount > cap:
            raise ValueError(f"shared integrated budget (${cap}) exhausted before send")
        if stage == "jev":
            jev_cap = Decimal(str(manifest.get("jev_reserved_cap_usd", cap)))
            prior = Decimal(current["held_or_spent_by_stage"].get("jev", "0"))
            if prior + amount > jev_cap:
                raise ValueError("Jev allocation exhausted; final QA allocation remains protected")
        pins = self.policy["provider_run"]
        row = {
            "status": "reserved",
            "stage": stage,
            "model": model,
            "request_sha256": request_sha,
            "request_id": stage + "-" + request_sha,
            "reservation_id": "hold-" + request_sha,
            "manifest_sha256": pins["manifest"]["content_sha256"],
            "provenance_sha256": pins["provenance"]["content_sha256"],
            "created_at_unix": time.time(),
            "input_token_cap": input_cap,
            "output_token_cap": output_cap,
            "reserved_usd": float(amount),
            "charged_usd": float(amount),
        }
        directory.mkdir()
        write_new(directory / "wire.json", wire)
        write_new(directory / "reservation.json", row)
        return directory, row

    def complete(self, directory, row, usage):
        if (
            set(usage) != {"input_tokens", "output_tokens", "cost_usd"}
            or any(type(usage[key]) is not int or usage[key] < 0 for key in ("input_tokens", "output_tokens"))
            or type(usage["cost_usd"]) not in {int, float}
            or not math.isfinite(usage["cost_usd"])
            or not 0 <= usage["cost_usd"] <= row["reserved_usd"]
            or usage["input_tokens"] > row["input_token_cap"]
            or usage["output_tokens"] > row["output_token_cap"]
        ):
            raise ValueError("provider usage is missing or exceeds reserved bounds")
        completed = {**row, "status": "completed", "completed_at_unix": time.time(), "receipt": usage}
        write_new(directory / "completed.json", completed)
        return completed

    def evaluate(self, wire, *, semantic=False, sender=None):
        request_sha = content_sha256(wire)
        directory = self.directory / ("jev-" + request_sha)
        if directory.exists():
            path = directory / "archive.json"
            if not path.exists():
                raise ValueError("attempt has no completed archive; automatic retry forbidden")
            archive = json.loads(path.read_bytes())
            generic = dict(archive)
            receipt = json.loads(archive["receipt"])
            receipt.pop("content_sha256")
            receipt["schema_version"] = RECEIPT_SCHEMA
            generic["receipt"] = canonical_json(seal(receipt))
            read_jev_archive(wire, generic, self.policy)
            return archive
        if wire.get("model") != MODEL or not wire.get("questions"):
            raise ValueError("paid Jev request lacks its fixed model or questions")
        if len(canonical_json(wire).encode()) > MAX_WIRE_BYTES:
            raise ValueError("paid Jev request exceeds conservative context bound before reservation")
        if sender is None and self.sender is None:  # the frozen path would load its provider here and KeyError later
            raise ValueError("no Jev sender configured: the ledger is replay-only")
        if self.api_key is None:
            raise ValueError("no OpenRouter key configured: the ledger is replay-only")
        # A byte per token plus framing is deliberately pessimistic. Output is
        # free for this pinned Jev model, but its token count remains bounded.
        directory, reservation = self.reserve(
            wire=wire,
            stage="jev",
            model=MODEL,
            input_cap=len(canonical_json(wire).encode()) + 4096,
            output_cap=4096,
            input_rate=JEV_INPUT_PER_MILLION,
            output_rate=Decimal(0),
        )
        try:
            transport_options = {}
            if sender is None:
                sender = self.sender

                def capture_error(status, raw):
                    write_new(directory / "http-error-body.bin", raw)
                    write_new(directory / "http-error.json", {"status": status, "body_sha256": sha(raw)})

                transport_options["on_http_error"] = capture_error

            def capture(raw, decoded, latency):
                write_new(directory / "raw.json", raw)
                write_new(directory / "transport.json", {"latency_ms": latency, "request_sha256": request_sha})

            raw, _ = sender(
                wire,
                provider="openrouter",
                api_key=self.api_key,
                timeout=120.0,
                on_response=capture,
                **transport_options,
            )
            usage = raw["usage"]
            usage = {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "cost_usd": usage["cost"],
            }
            self.complete(directory, reservation, usage)
            if not model_family_ok(MODEL, str(raw.get("model"))):
                reason = f"response model {raw.get('model')!r} is not {MODEL!r} or a snapshot of it"
                settle_refused_model(directory, "JevModelRefused", reason)
                raise JevModelRefused(reason)
            ledger_bytes = (directory / "completed.json").read_bytes()
            raw_bytes = (directory / "raw.json").read_bytes()
            receipt = seal(
                {
                    "schema_version": RECEIPT_SCHEMA,
                    "evidence_mode": "provider_receipted",
                    "request_sha256": request_sha,
                    "raw_response_sha256": sha(raw_bytes),
                    "ledger_record_sha256": sha(ledger_bytes),
                    "usage": usage,
                }
            )
            archive = {
                "wire": canonical_json(wire),
                "raw": raw_bytes.decode(),
                "receipt": canonical_json(receipt),
                "ledger_record": ledger_bytes.decode(),
            }
            read_jev_archive(wire, archive, self.policy)
            if semantic:
                receipt.pop("content_sha256")
                receipt["schema_version"] = "memory-semantic-send-receipt/v1"
                archive["receipt"] = canonical_json(seal(receipt))
            write_new(directory / "archive.json", archive)
            return archive
        except JevModelRefused:
            raise
        except BaseException as exc:
            write_new(
                directory / "failure.json",
                {
                    "type": type(exc).__name__,
                    "reason": str(exc),
                    "reservation_retained": True,
                    "automatic_retry": False,
                },
            )
            raise


class RoundBudget(IntegratedBudget):
    """Share receipt replay/completion accounting, not the old run's authorization.

    Extra to the frozen class, all optional and all per INSTANCE:
      * ``limit_usd``: the held-or-spent total this ledger may reach, in place of the sealed round cap and the
        integrated cap (an append run's limit, ``bonfire.append_limit``). ``summary()`` reports it as ``cap_usd`` and
        ``integrated_cap_usd`` None, so the prefetch's room (``llm.cap_could_bind``, ``llm.LedgerView``) is the run's.
        Either way the caps bind at ``reserve``, never at the open or in ``summary()``: a shard already past them
        opens and replays, and only a new hold past them is refused;
      * ``reserve_guard(amount_usd)`` runs first in ``reserve`` (the stage-cap meter; ``llm.guard``);
      * ``on_new_send(archive)`` runs after ``evaluate`` returned a wire that was not in the cache before
        (graph_picks' stage meter);
      * ``serial`` (``llm.Serial``) routes ``reserve`` / ``complete`` / ``summary`` of a prefetch pool's worker
        threads through its lock and tickets; any other caller gets the frozen method.
    """

    def __init__(
        self,
        directory,
        manifest,
        provenance,
        *,
        sender: Callable | None = None,
        api_key: str | None = None,
        reserve_guard: Callable[[float], None] | None = None,
        on_new_send: Callable[[dict], None] | None = None,
        limit_usd: Decimal | None = None,
    ):
        try:
            cap = Decimal(manifest.get("budget_cap_usd", ""))
        except ArithmeticError:
            cap = Decimal(0)
        if not cap.is_finite() or cap <= 0 or "previous_budget" in manifest:
            raise ValueError("round budget cap must be a positive amount without inherited runs")
        if limit_usd is not None and (not isinstance(limit_usd, Decimal) or not limit_usd.is_finite() or limit_usd < 0):
            raise ValueError(f"limit_usd {limit_usd!r} must be a finite non-negative Decimal")
        self.limit = limit_usd
        self.cap = cap if limit_usd is None else limit_usd
        #: the shard's integrated cap, held against a new hold of a plain open (None under an append's limit)
        self.integrated = integrated_cap(manifest) if limit_usd is None else None
        self.sender, self.api_key = sender, api_key
        self.reserve_guard, self.on_new_send = reserve_guard, on_new_send
        self.serial: Any = None
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = (directory / "execution.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.frozen = {
                "manifest": canonical_json(manifest).encode(),
                "provenance": canonical_json(provenance).encode(),
            }
            for name, raw in self.frozen.items():
                path = directory / (name + ".json")
                if not path.exists():
                    write_new(path, raw)
                if path.read_bytes() != raw:
                    raise ValueError("immutable round budget pin changed")
            self.policy = semantic_policy(
                model=MODEL,
                evidence_mode="provider_receipted",
                provider_run={
                    name: {"path": str((directory / (name + ".json")).resolve()), "file_sha256": sha(raw)}
                    for name, raw in self.frozen.items()
                },
            )
            self.summary()
        except BaseException:
            self.close()
            raise

    def summary(self):
        if self.serial is not None and self.serial.owns_thread():
            return self.serial.summary(self._summary)
        return self._summary()

    def _summary(self):
        for name, raw in self.frozen.items():
            if (self.directory / (name + ".json")).read_bytes() != raw:
                raise ValueError("immutable round budget pin changed")
        # read, never held against a cap: a shard already past its caps still opens and replays (``_guarded_reserve``)
        result = super().summary(integrated=False)
        out = {**result, "cap_usd": str(self.cap)}
        if self.limit is not None:
            out["integrated_cap_usd"] = None
        return out

    def complete(self, directory, row, usage):
        if self.serial is not None and self.serial.owns_thread():
            return self.serial.complete(super().complete, directory, row, usage)
        return super().complete(directory, row, usage)

    def evaluate(self, wire, *, semantic=False, sender=None):
        hook = self.on_new_send
        if hook is None:
            return super().evaluate(wire, semantic=semantic, sender=sender)
        sent_before = (self.directory / ("jev-" + content_sha256(wire))).exists()
        archive = super().evaluate(wire, semantic=semantic, sender=sender)
        if not sent_before:
            hook(archive)
        return archive

    def reserve(self, *, wire, stage, model, input_cap, output_cap, input_rate, output_rate):
        kwargs = {
            "wire": wire,
            "stage": stage,
            "model": model,
            "input_cap": input_cap,
            "output_cap": output_cap,
            "input_rate": input_rate,
            "output_rate": output_rate,
        }
        if self.serial is not None and self.serial.owns_thread():
            return self.serial.reserve(self._guarded_reserve, kwargs)
        return self._guarded_reserve(**kwargs)

    def _guarded_reserve(self, *, wire, stage, model, input_cap, output_cap, input_rate, output_rate):
        if self.reserve_guard is not None:
            amount = (Decimal(input_cap) * input_rate + Decimal(output_cap) * output_rate) / 1_000_000
            self.reserve_guard(float(amount))
        if (
            stage not in {"jev", "grammar_gap"}
            or type(input_cap) is not int
            or input_cap <= 0
            or type(output_cap) is not int
            or output_cap < 0
        ):
            raise ValueError("round budget call bounds differ")
        if any(not isinstance(rate, Decimal) or not rate.is_finite() or rate < 0 for rate in (input_rate, output_rate)):
            raise ValueError("budget rates must produce a finite positive reservation")
        amount = (Decimal(input_cap) * input_rate + Decimal(output_cap) * output_rate) / 1_000_000
        if amount <= 0:
            raise ValueError("budget rates must produce a finite positive reservation")
        request_sha = content_sha256(wire)
        directory = self.directory / (stage + "-" + request_sha)
        if directory.exists():
            raise ValueError("paid request already attempted; cold replay only, no retry")
        held = Decimal(self.summary()["held_or_spent_usd"])
        if held + amount > self.cap:
            if self.limit is not None:
                raise ValueError(f"append budget exhausted before send (the run's limit ${self.cap})")
            raise ValueError("shared round budget exhausted before send")
        if self.integrated is not None and held + amount > self.integrated:
            raise ValueError(f"shared integrated budget exceeded: ${self.integrated} exhausted before send")
        pins = self.policy["provider_run"]
        row = {
            "status": "reserved",
            "stage": stage,
            "model": model,
            "request_sha256": request_sha,
            "request_id": stage + "-" + request_sha,
            "reservation_id": "hold-" + request_sha,
            "manifest_sha256": pins["manifest"]["content_sha256"],
            "provenance_sha256": pins["provenance"]["content_sha256"],
            "created_at_unix": time.time(),
            "input_token_cap": input_cap,
            "output_token_cap": output_cap,
            "reserved_usd": float(amount),
            "charged_usd": float(amount),
        }
        directory.mkdir()
        write_new(directory / "wire.json", wire)
        write_new(directory / "reservation.json", row)
        return directory, row


# -- the shard --------------------------------------------------------------------------


def prepare_round_execution(
    pairs_dir: Path,
    output: Path,
    *,
    code: Sequence[str] = MK_CODE,
    cap_usd: Decimal = CAP,
    integrated_cap_usd: Decimal = INTEGRATED_CAP,
) -> dict:
    """Pin a pair manifest into a ledger shard. A build's manifest has no pairs (``make_empty_pairs``). The code pins are
    package keys, hashed in the imported package (``provenance.code_path``). ``cap_usd`` is sealed into the shard's
    manifest as ``budget_cap_usd``; ``RoundBudget`` enforces it. ``integrated_cap_usd`` is sealed as the shard's integrated
    cap (``integrated_cap``); a manifest without the field enforces ``LEGACY_INTEGRATED_CAP``."""
    raw = (pairs_dir / "manifest.json").read_bytes()
    pair_manifest = json.loads(raw)
    check_seal(pair_manifest, PAIRS_SCHEMA)
    paths = sorted(pairs_dir.glob("pair-*.json"))
    if len(paths) != len(pair_manifest["pairs"]):
        raise ValueError("pair manifest membership differs")
    if pair_manifest["pairs"]:
        raise ValueError("pair-bearing shards are not promoted: a build's pair manifest is empty (question-blind)")
    rows: list = []
    reserve = Decimal(0)
    manifest = seal(
        {
            "schema_version": SCHEMA,
            "ledger_schema": LEDGER_SCHEMA,
            "budget_cap_usd": str(cap_usd),
            "integrated_cap_usd": str(integrated_cap_usd),
            "round_id": pair_manifest["round_id"],
            "pair_manifest_sha256": sha(raw),
            "cohort_sha256": pair_manifest["cohort_sha256"],
            "rows": rows,
            "jev_max_reservation_usd": str(reserve),
            "provider_retry_limit": 0,
            "gap_order": "pair manifest order, then source unit order",
            "automatic_training": False,
        }
    )
    provenance = {
        "schema_version": "mk-execution-provenance/v1",
        "pair_manifest": pair_manifest,
        "code_pins": {p: sha(code_path(p).read_bytes()) for p in code},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "choices", "dispositions", "gaps"):
        (output / name).mkdir()
    write_new(output / "manifest.json", manifest)
    write_new(output / "provenance.json", provenance)
    return {"pairs": 0, "jev_max_reservation_usd": str(reserve), "all_cutoff_pairs": 0}


def harness_code_sets(harness_root: Path) -> list[set[str]]:
    """The key sets of ``<harness_root>/harness-code-set.json``: what a shard made by the harness at that root pins."""
    path = Path(harness_root) / HARNESS_CODE_SET
    try:
        doc = json.loads(path.read_bytes())
    except FileNotFoundError:
        raise FileNotFoundError(f"{path}: the harness root has no {HARNESS_CODE_SET}") from None
    sets = doc.get("code_sets") if isinstance(doc, dict) else None
    if (
        not isinstance(doc, dict)
        or doc.get("schema") != HARNESS_CODE_SET_SCHEMA
        or not isinstance(sets, list)
        or not sets
        or not all(isinstance(s, list) and s and all(isinstance(k, str) for k in s) for s in sets)
    ):
        raise ValueError(f"{path} is not a {HARNESS_CODE_SET_SCHEMA} document with non-empty code_sets")
    return [set(s) for s in sets]


def load_round_execution(output: Path, *, harness_root: Path | None = None, code: Sequence[str] | None = None):
    """Open a shard: seals and pair inventory checked; the code pins are checked only for a shard without ``ledger_schema``.

    A shard with ``ledger_schema`` opens whatever the bytes of the code it pinned. A shard without it opens when each
    ``src/`` pin is in ``ACCEPTED_CODE_PINS`` (a harness file's pin is re-hashed against ``harness_root``). ``code`` None
    accepts a shard pinned to ``MK_CODE`` and, given ``harness_root``, to a key set of its ``harness-code-set.json``. A
    shard that pins harness files cannot be opened without ``harness_root`` (FileNotFoundError naming the first harness
    key)."""
    manifest = json.loads((output / "manifest.json").read_bytes())
    provenance = json.loads((output / "provenance.json").read_bytes())
    check_seal(manifest, SCHEMA)
    pair_manifest = provenance["pair_manifest"]
    check_seal(pair_manifest, PAIRS_SCHEMA)
    stored = set(provenance["code_pins"])
    harness_keys = sorted(k for k in stored if not k.startswith(SRC_PREFIX))
    if harness_keys and harness_root is None:
        raise FileNotFoundError(f"{harness_keys[0]}: a harness file; pass --harness-root (shard {output})")
    if code is not None:
        allowed = [set(code)]
    else:
        allowed = [set(MK_CODE)]
        if stored not in allowed and harness_root is not None:
            allowed += harness_code_sets(harness_root)
    if (
        stored not in allowed
        or sha(canonical_json(pair_manifest).encode()) != manifest["pair_manifest_sha256"]
        or len(manifest["rows"]) != len(pair_manifest["pairs"])
        or [r["index"] for r in manifest["rows"]] != list(range(len(manifest["rows"])))
        or len({r["pair_sha256"] for r in manifest["rows"]}) != len(manifest["rows"])
    ):
        raise ValueError("round provenance or pair inventory differs")
    schema = manifest.get("ledger_schema")
    if schema is not None and schema != LEDGER_SCHEMA:
        raise ValueError(
            f"ledger schema {schema!r} is newer than this kernel reads ({LEDGER_SCHEMA}): upgrade the kernel"
        )
    if schema is None:
        for path, digest in provenance["code_pins"].items():
            known = (path, digest) in ACCEPTED_CODE_PINS if path.startswith(SRC_PREFIX) else None
            if known is None:
                known = sha(code_path(path, harness_root=harness_root).read_bytes()) == digest
            if not known:
                raise ValueError("round implementation changed: " + path)
    if manifest["rows"]:
        raise ValueError("pair-bearing shards are not promoted: a build's shard holds no question pairs")
    return manifest, provenance, []


def make_empty_pairs(out: Path, *, round_id: str, capture_sha: str, g0_sha: str) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    manifest = seal(
        {
            "schema_version": PAIRS_SCHEMA,
            "round_id": round_id,
            "cohort_sha256": EMPTY_COHORT_SHA,
            "capture_sha256": capture_sha,
            "g0_sha256": g0_sha,
            "pairs": [],
            "unit_status_counts": {"framed": 0, "empty": 0, "cutoff": 0},
            "sense_menu_omissions": {},
            "gold_answers_used": False,
            "gold_evidence_used": False,
        }
    )
    write_new(out / "manifest.json", manifest)
    return manifest


def make_execution(
    out: Path,
    *,
    round_id: str,
    capture_sha: str,
    g0_sha: str,
    code: Sequence[str] = MK_CODE,
    cap_usd: Decimal = CAP,
    integrated_cap_usd: Decimal = INTEGRATED_CAP,
) -> dict:
    pairs = out.parent / (out.name + "-pairs")
    make_empty_pairs(pairs, round_id=round_id, capture_sha=capture_sha, g0_sha=g0_sha)
    return prepare_round_execution(pairs, out, code=code, cap_usd=cap_usd, integrated_cap_usd=integrated_cap_usd)


def round_cap(messages: int) -> Decimal:
    """The round cap a bootstrap seals for a conversation of ``messages``: exactly ``CAP`` up to the reference size (so
    the sealed manifest keeps ``"2"``), proportional above it (``llm.cap_scale``). A round cap above the integrated cap
    (``INTEGRATED_CAP``, what a new shard records) could only fail mid-stage after spend, so it raises ``ValueError``
    (the driver's preflight reports it before any stage). An append never reads it: its budget is per run
    (``bonfire.append_budget``)."""
    from memory_kernel.pipeline.llm import REF_MESSAGES, cap_scale

    scale = cap_scale(messages)
    cap = CAP if scale <= 1 else CAP * Decimal(str(scale))
    if cap > INTEGRATED_CAP:
        most = int(INTEGRATED_CAP / CAP * REF_MESSAGES)
        raise ValueError(
            f"{messages} messages need a round cap of ${cap:.2f}, above the integrated cap ${INTEGRATED_CAP} "
            f"(at most {most} messages, counting each part of a split message)"
        )
    return cap


@dataclass(frozen=True)
class Shard:
    """A round shard directory plus everything a stage needs to open its ledger without module state."""

    path: Path
    #: the root of the harness files a harness-made shard pins (``--harness-root``); None for an ``mk`` shard
    harness_root: Path | None = None
    sender: Callable | None = None
    api_key: str | None = None
    reserve_guard: Callable[[float], None] | None = None
    on_new_send: Callable[[dict], None] | None = None
    code: tuple[str, ...] | None = None
    #: an append run's limit (``RoundBudget``'s ``limit_usd``, ``bonfire.append_limit``); None = the sealed caps
    limit_usd: Decimal | None = None

    @property
    def budget(self) -> Path:
        return self.path / "budget"

    def load(self):
        """``load_round_execution`` for this shard (a shard without ``ledger_schema`` has its code pins checked on every call)."""
        return load_round_execution(self.path, harness_root=self.harness_root, code=self.code)

    def open(self, manifest: Mapping, provenance: Mapping) -> RoundBudget:
        return RoundBudget(
            self.budget,
            manifest,
            provenance,
            sender=self.sender,
            api_key=self.api_key,
            reserve_guard=self.reserve_guard,
            on_new_send=self.on_new_send,
            limit_usd=self.limit_usd,
        )

    def with_hooks(self, **hooks: Any) -> Shard:
        return replace(self, **hooks)
