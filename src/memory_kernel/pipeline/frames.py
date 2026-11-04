"""Where a build reads PropBank frames from, and the path-free label a frame dir gets in a sealed output.

Two frame dirs, in this order (``frame_dirs[0]`` is the upstream source of VerbNet classes and role themes; every dir
feeds the glosses, POS and aliases, first dir wins):

1. the upstream PropBank frames at the pinned commit, which ``mk setup --component propbank-frames`` places under the data
   dir (``<data>/propbank-frames/<commit12>/frames``);
2. the packaged annex (``memory_kernel/data/propbank-frames-annex/frames``): 24 relational-noun rolesets, CC BY-SA 4.0 (see
   the LICENSE and NOTICE beside it).

``--frame-dir`` (repeatable) replaces both.

``gloss_source`` of ``eventuality.json`` names the dir a gloss came from. A dir under the build's ``--vault`` keeps the vault
-relative path (the bytes the benchmarked runs sealed); any other dir gets a label that holds no host path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from memory_kernel.pipeline.provenance import package_dir

ANNEX_REL = "data/propbank-frames-annex/frames"
ANNEX_LABEL = "memory-kernel:propbank-frames-annex"


def upstream_commit12() -> str:
    """The first 12 hex digits of the pinned upstream commit (``runtime-pins.json``)."""
    from memory_kernel import setup_runtime

    return str(setup_runtime.load_pins()["components"]["propbank-frames"]["commit"])[:12]


def upstream_dir(data_dir: Path) -> Path:
    """The ``frames`` dir ``mk setup`` places for the pinned upstream commit (it may not exist yet)."""
    return data_dir / "propbank-frames" / upstream_commit12() / "frames"


def annex_dir() -> Path:
    """The packaged annex ``frames`` dir, in the IMPORTED package (lexical, like every package path)."""
    return package_dir() / ANNEX_REL


def default_frame_dirs(data_dir: Path) -> tuple[Path, Path]:
    return upstream_dir(data_dir), annex_dir()


def frame_label(directory: Path, *, vault: Path | None, data_dir: Path | None) -> str:
    """The ``gloss_source`` label of a frame dir: vault-relative when it lies under ``vault`` (``relative_to``, lexical);
    else ``propbank-frames@<commit12>`` for the setup-managed upstream dir, ``memory-kernel:propbank-frames-annex`` for the
    packaged annex, and ``frames:<12 hex of sha256(path)>`` for anything else. Never the home dir."""
    directory = Path(directory)
    if vault is not None and directory.is_relative_to(vault):
        return str(directory.relative_to(vault))
    if data_dir is not None and directory == upstream_dir(data_dir):
        return f"propbank-frames@{upstream_commit12()}"
    if directory == annex_dir():
        return ANNEX_LABEL
    return "frames:" + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
