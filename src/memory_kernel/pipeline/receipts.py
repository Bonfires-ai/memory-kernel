"""Kernel 2.0 dev receipts: ``MK_RECEIPTS=on`` keeps what production throws away, so a 2.0 build can be compared with
a 1.8 one at $0 on replays.

Production keeps no receipts: the append lays its run out in a scratch dir and deletes it. With receipts on, the
append (``pipeline.graph_append``) keeps and replays, under ``<root>/<bonfire id>/``:

* ``shard``: the Jev round-ledger shard. It persists, and its stages replay an archived answer from it for free; the
  run's budget is counted from the shard's held-or-spent total (``bonfire.shard_spend``);
* ``voyage-cache.json``: the events stage's Voyage cache. It persists across appends and is still seeded from the
  stored vectors;
* ``runs/<run:04d>/``: every run dir, kept. A crashed attempt's dir is renamed ``<run>.attempt-<n>``.

``MK_RECEIPTS_DIR`` names the root; ``on`` without it is refused (never a silent default location)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_ON = "MK_RECEIPTS"
ENV_DIR = "MK_RECEIPTS_DIR"
VOYAGE_CACHE_FILE = "voyage-cache.json"
ATTEMPT_MARK = ".attempt-"


@dataclass(frozen=True)
class Receipts:
    """Where one dev receipts root keeps each bonfire's shard, Voyage cache and run dirs."""

    root: Path

    def bonfire(self, bonfire_id: str) -> Path:
        return self.root / bonfire_id

    def shard(self, bonfire_id: str) -> Path:
        return self.bonfire(bonfire_id) / "shard"

    def voyage_cache(self, bonfire_id: str) -> Path:
        return self.bonfire(bonfire_id) / VOYAGE_CACHE_FILE

    def run_dir(self, bonfire_id: str, run: int) -> Path:
        return self.bonfire(bonfire_id) / "runs" / f"{run:04d}"


def receipts_from_env(environ: Mapping[str, str]) -> Receipts | None:
    """The receipts the environment asks for: None unless ``MK_RECEIPTS`` is ``on``; ``on`` without a non-empty
    ``MK_RECEIPTS_DIR`` raises ``ValueError``."""
    if environ.get(ENV_ON, "").strip().lower() != "on":
        return None
    directory = environ.get(ENV_DIR, "").strip()
    if not directory:
        raise ValueError(f"{ENV_ON}=on needs {ENV_DIR}: the directory the receipts are kept in")
    return Receipts(Path(directory).expanduser())
