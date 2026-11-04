"""A shard opens by its recorded schema or by an accepted code digest, never by the live bytes of ledger.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_kernel.construction.memory_semantic_jev import seal
from memory_kernel.pipeline import ledger


def old_style_shard(out: Path, pins: dict[str, str]) -> Path:
    """A shard as made before ``ledger_schema``: no schema in the manifest, the given digests as its code pins."""
    ledger.make_execution(out, round_id="x1-pins", capture_sha="c" * 64, g0_sha="d" * 64)
    manifest = json.loads((out / "manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k not in ("content_sha256", "ledger_schema")}
    provenance = json.loads((out / "provenance.json").read_text())
    provenance["code_pins"] = pins
    for name, doc in (("manifest.json", seal(body)), ("provenance.json", provenance)):
        (out / name).chmod(0o644)
        (out / name).write_text(json.dumps(doc))
    return out


def test_todays_shard_opens_after_ledger_edit(tmp_path):
    pins = {path: digest for path, digest in sorted(ledger.ACCEPTED_CODE_PINS)}
    assert set(pins) == set(ledger.MK_CODE)
    manifest, _provenance, _pairs = ledger.load_round_execution(old_style_shard(tmp_path / "shard", pins))
    assert "ledger_schema" not in manifest


def test_unknown_digest_without_schema_refuses(tmp_path):
    pins = {path: digest for path, digest in sorted(ledger.ACCEPTED_CODE_PINS)}
    pins[ledger.MK_CODE[0]] = "0" * 64
    with pytest.raises(ValueError, match="round implementation changed"):
        ledger.load_round_execution(old_style_shard(tmp_path / "shard", pins))


def test_new_shard_records_the_schema(tmp_path):
    shard = tmp_path / "shard"
    ledger.make_execution(shard, round_id="x1-pins", capture_sha="c" * 64, g0_sha="d" * 64)
    manifest, provenance, _pairs = ledger.load_round_execution(shard)
    assert manifest["ledger_schema"] == ledger.LEDGER_SCHEMA == 1
    assert set(provenance["code_pins"]) == set(ledger.MK_CODE)


def test_a_newer_ledger_schema_refuses(tmp_path):
    shard = tmp_path / "shard"
    ledger.make_execution(shard, round_id="x1-pins", capture_sha="c" * 64, g0_sha="d" * 64)
    manifest = json.loads((shard / "manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "content_sha256"} | {"ledger_schema": ledger.LEDGER_SCHEMA + 1}
    (shard / "manifest.json").chmod(0o644)
    (shard / "manifest.json").write_text(json.dumps(seal(body)))
    with pytest.raises(ValueError, match="newer than this kernel reads"):
        ledger.load_round_execution(shard)
