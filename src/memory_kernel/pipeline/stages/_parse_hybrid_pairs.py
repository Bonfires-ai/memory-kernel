"""The compat-run manifest of a hybrid capture.

Ported from a reference-harness module (unreached top-level definitions pruned; compat_run takes ``root`` instead
of a module global).

A hybrid capture (``_parse_capture.gated_merge``) holds sealed fenced-unit records. ``compat_run`` writes the
write-once ``manifest.json`` of a compat run (``COMPAT_SCHEMA``): the capture's path, relative to ``root`` when
it lies under it, and an empty ``units`` list. The parse stage writes it after the merge and recognises a
finished compat run by that schema.
"""

from __future__ import annotations

import json
from pathlib import Path

COMPAT_SCHEMA = "mk-hybrid-compat-run/v1"


def compat_run(capture: Path, out: Path, root: Path) -> Path:
    out.mkdir(parents=True, exist_ok=False)
    resolved = Path(capture).resolve()
    path = str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)
    with (out / "manifest.json").open("x") as handle:
        json.dump({"schema_version": COMPAT_SCHEMA, "capture": {"path": path}, "units": []}, handle)
    return out
