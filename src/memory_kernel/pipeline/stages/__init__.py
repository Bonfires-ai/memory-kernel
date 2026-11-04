"""The build's stage functions: ``STAGES[name](ctx, live) -> result`` for every stage of ``runner.ORDER`` through
``seal`` (``config.LAST_STAGE``), ported from the reference harness: ``ingest`` (select, boundary), ``parse``
(capture), ``graph`` (graph_picks, graph_a), ``events`` (events, ctx), ``referents``, ``write`` (build_dry, write,
rewrite) and ``index`` (index, sanity, seal). The evaluation stages belong to the benchmark harness, not to mk. Kernel
1.8 archived the taxonomy and question layer (``config.ARCHIVED_STAGES``): none of its stages is in this map.
No stage imports ``scripts.*``: the runner's import audit proves it for every stage child.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from memory_kernel.pipeline.config import LAST_STAGE
from memory_kernel.pipeline.stages import events, graph, index, ingest, parse, referents, write

if TYPE_CHECKING:
    from memory_kernel.pipeline.runner import Ctx

STAGES: dict[str, Callable[[Ctx, bool], Any]] = {
    "select": ingest.select,
    "boundary": ingest.boundary,
    "capture": parse.capture,
    "graph_picks": graph.graph_picks,
    "graph_a": graph.graph_a,
    "events": events.events,
    "ctx": events.ctx,
    "referents": referents.referents,
    "build_dry": write.build_dry,
    "write": write.write,
    "rewrite": write.rewrite,
    "index": index.index,
    "sanity": index.sanity,
    "seal": index.seal,
}


def _check_complete() -> None:
    from memory_kernel.pipeline.runner import ORDER

    missing = [s for s in ORDER[: ORDER.index(LAST_STAGE) + 1] if s not in STAGES]
    if missing:
        raise RuntimeError(f"stages without a promoted body: {missing}")


_check_complete()
