"""One stage's body, run in whichever process drives it: the build driver (``runner._invoke``, the kernel worker for an
append, ``mk build`` for a bootstrap) or the ``mk _stage`` child (``runner.run_stage_child``). Both call
``stage_body``, so a stage computes and writes the same thing either way.

The stages in ``CHILD_STAGES`` keep running as a child, each for the reason written next to it. ``build_lock`` lets
one build at a time run stages in a process: the worker can overlap two builds (an orphaned job still running while
the next starts), and a build that finds the lock held runs every stage as a child, as before.

**Process-global state.** The worker never restarts, so whatever a stage leaves in the process reaches the next stage
and the next build (maybe of another bonfire, with another data dir, frame dirs or grammar image). A ``mk _stage``
child differed from the driver in four things; in-process each is handled as follows:

* ``MK_DATA_DIR`` (``runner.child_env``). Nothing a non-child stage runs reads it: the stages pass ``cfg.data_dir``
  explicitly (the frame labels, the managed parser), and its one other reader, ``search.kg_context``'s token
  encoder, is reached only by ``serve``. What the child did WITH it is the gap: its ``cli.main`` put
  ``<data>/nltk_data`` first for nltk (``cli.wire_runtime_data``), so WordNet came from the build's data dir.
  ``build_data`` does the same for an in-process build, explicitly: ``nltk.data.path`` gets that directory first, and
  the earlier list comes back when the build ends. The environment is never written, because ``os.environ`` is seen
  by every thread of the worker at once: ``bonfire.host_paths`` (every ``serve`` question) and a busy build's
  ``child_env`` copy would pick up another build's data dir.
* The secrets. A stage reads them from ``cfg`` only (no stage reads the ``MK_*`` secret variables;
  ``run_stage_child`` rebuilds ``cfg`` from them), so in-process they stay out of the environment.
* ``PYTHONPATH`` and ``python -P``. They only decide which code a child imports; in-process the code is the driver's
  own, already imported. No in-process stage starts a Python process (the engine and the parser are capture's).
* The cwd (the child ran in ``cfg.root``). No stage reads a relative path: ``config.lexical_abspath``'s ``Path.cwd()``
  fallback is reached only by the CLI's argument parsing, before a build; the build config's own paths must be absolute
  (``runner.relative_path_problems`` refuses a relative one, and a relative ``MK_STOPWORDS_FILE``, before any stage).

No stage writes ``os.environ`` for longer than a call: ``engine.py`` writes the parser knob only in capture (a child); ``search.kg_context`` swaps
``TIKTOKEN_CACHE_DIR`` and restores it (``serve`` only). ``clear_build_caches`` runs at the start of an in-process
build, after ``build_lock`` is taken (a busy or forced-children build never clears: the lock holder may be mid-stage).

The cache audit (every ``functools`` cache and module memo in ``memory_kernel``). *Pure*: a function of its arguments
and of pinned packages only (spaCy ``en_core_web_sm``, the NLTK stemmer, the code); one keyed by a text the worker
parses is LRU-bounded (4096), since nothing clears a pure cache and the worker never restarts. *Keyed*: the per-build
input is in the key. *Process-constant*: reads the process environment, which no stage writes and which a child
inherits unchanged (``child_env`` adds only ``MK_DATA_DIR``, ``PYTHONPATH`` and the secrets). *Cleared*: cleared per
build by ``clear_build_caches``; each reads the WordNet corpus (the data dir's) or the data dir itself, or memoizes a
load or parse failure. A failure memoized in stage N still degrades the rest of THAT build (a child's ended with the
stage).

==============================================================  ============================================
``grammar.dependencies._wordnet_module``, ``_WORDNET_READY``     cleared (the WordNet corpus and its probe)
``nltk.corpus.wordnet`` (the loaded reader)                      cleared (a fresh lazy loader)
``graph.g1_nlp._models`` (``pipeline``; ``load_spacy_model``     process-constant (one copy per model name);
reads it)                                                        never caches a failure
``grammar.wordnet_semantics._wordnet_*``, ``_candidate_*``       cleared (WordNet)
``grammar.wordnet_semantics._doc``, ``_load_nlp``,
``_porter_stemmer``                                              cleared (memoizes a load/parse failure)
``grammar.wordnet_semantics.get_wordnet_semantics``              pure (stateless service)
``grammar.semantic_operators._operator_profile_cached``          cleared (through the parse)
``grammar.semantic_operators._doc_cached``, ``_load_nlp``        cleared (memoizes a load/parse failure)
``grammar.semantic_operators.get_semantic_operator_service``     pure
``grammar.parse_cache._CACHE``                                   keyed (spaCy model key, text)
``envelope.render._verb_exceptions``, ``_is_verb_base``,
``_irregular_3sg``                                               cleared (WordNet ``verb.exc`` / morphy)
``envelope.render._master``, ``_base_reads_present``             pure (regex / spaCy)
``graph.forward._common``                                        cleared (WordNet, ``identity.is_common_noun``)
``graph.g1_nlp._resolved``, ``lemma_key``                       pure (spaCy; ``lemma_key`` LRU-bounded)
``graph.g1_mentions.bare_light``, ``_decompose_text``             cleared (WordNet)
``graph.g1_mentions._own_doc``                                   pure (spaCy, LRU-bounded)
``graph.g1_kinds.noun_kind``                                     cleared (WordNet)
``graph.g1_mentions.known_names``                                keyed (the doc object, LRU-bounded; spaCy)
``graph.g1_projection._parse``, ``possessors``, ``analyse``,
``_short_name``                                                  pure (spaCy, LRU-bounded)
``graph.g1_render._parse_tokens``                                pure (spaCy)
``knowledge.temporal_deixis.relative_number_value``              pure
``knowledge.propbank_framesets._cached_*_index``                 keyed (frames dir, its mtime)
``stopwords.english_stopwords``                                  process-constant (``MK_STOPWORDS_FILE``; a
                                                                 relative value is refused)
``search.kg_context._ENC``                                       cleared (``MK_DATA_DIR``; ``serve`` only)
``parser_service.registry._load``, ``parser_service.cache``      keyed (model name / parse key; the parser
                                                                 service process, capture only)
``pipeline.llm._TL``                                             per thread of one ``FifoPool`` run (fresh
                                                                 threads; ``pool`` reset when each ends)
``pipeline.engine._tls``                                         re-entrant flock counts, balanced (capture)
``pipeline.bonfire._WARNED_INSTALL_PATHS``                       a log de-duplication (driver side)
``serve.answer._LOCKS``                                          per-path locks (``serve``)
``construction.fcg.penelope_host.DEFAULT_PENELOPE_HOST``         a constant (None)
==============================================================  ============================================
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

from memory_kernel.pipeline import llm

if TYPE_CHECKING:
    from memory_kernel.pipeline.config import BuildConfig

#: Stages that always run as a ``mk _stage`` child. A stage goes here only with its reason.
CHILD_STAGES: frozenset[str] = frozenset(
    {
        # It is the only stage that imports pyfcg at runtime (process-global engine state), and
        # ``_parse_capture._driven`` calls ``signal.signal``, which raises off the main thread (the worker runs an
        # append on a pool thread).
        "capture",
    }
)
#: ``MK_STAGE_CHILDREN=1``: every stage of a build runs as a child (the operator's byte-identity baseline).
STAGE_CHILDREN_ENV = "MK_STAGE_CHILDREN"

#: What ``redact`` writes in place of a secret's value.
REDACTED = "<redacted>"

_BUILD_LOCK = threading.Lock()

#: The ``functools`` caches ``clear_build_caches`` clears, by module: each reads the WordNet corpus (the data dir's
#: ``nltk_data``) or memoizes a spaCy / stemmer load or parse FAILURE (a ``None`` that would otherwise degrade every
#: later build of the worker). A module not imported yet holds nothing and is skipped.
BUILD_CACHES: dict[str, tuple[str, ...]] = {
    "memory_kernel.grammar.dependencies": ("_wordnet_module",),
    "memory_kernel.grammar.wordnet_semantics": (
        "_candidate_wordnet_terms",
        "_wordnet_compound_role_action_terms",
        "_wordnet_role_action_terms_for_pair",
        "_wordnet_taxonomy_profile",
        "_wordnet_taxonomy_metadata_for_text",
        "_candidate_wordnet_taxonomy_terms",
        "_wordnet_direct_hypernyms",
        "_wordnet_term_lexnames",
        "_wordnet_noun_synset_shapes",
        "_wordnet_noun_sense_lexnames",
        "_wordnet_first_noun_sense_closure",
        "_wordnet_noun_sense_closures",
        "_wordnet_verb_lexnames",
        "_wordnet_primary_verb_lexname",
        "_wordnet_verb_lemma",
        "_wordnet_duration_unit_term",
        "_wordnet_derivationally_related_verbs_cached",
        "_wordnet_derivationally_related_nouns_cached",
        "_wordnet_adjective_nominal_terms_cached",
        "_doc",
        "_load_nlp",
        "_porter_stemmer",
    ),
    "memory_kernel.grammar.semantic_operators": (
        "_operator_profile_cached",
        "_doc_cached",
        "_load_nlp",
    ),
    "memory_kernel.envelope.render": ("_verb_exceptions", "_is_verb_base", "_irregular_3sg"),
    "memory_kernel.graph.forward": ("_common",),
    "memory_kernel.graph.g1_mentions": ("bare_light", "_decompose_text"),
    "memory_kernel.graph.g1_kinds": ("noun_kind",),
}


def redact(cfg: BuildConfig, text: str) -> str:
    """``text`` with every non-empty secret value of the build (``cfg.secret_env()``) replaced by ``REDACTED``: what an
    in-process stage's exception may put in a Stop reason or the build log (a child's message stayed in the child)."""
    for value in sorted((v for v in cfg.secret_env().values() if v), key=len, reverse=True):
        text = text.replace(value, REDACTED)
    return text


@contextlib.contextmanager
def build_lock() -> Iterator[bool]:
    """The process-wide in-process lock, taken without waiting. Yields True when this build holds it."""
    held = _BUILD_LOCK.acquire(blocking=False)
    try:
        yield held
    finally:
        if held:
            _BUILD_LOCK.release()


def clear_build_caches() -> None:
    """Forget every cache whose value depends on a build's data dir, and every memoized load or parse failure (the
    audit in the module docstring). First the loaded WordNet reader itself: nltk's ``wordnet`` becomes a fresh lazy
    loader, which finds the corpus on ``nltk.data.path`` at its first use (the old reader object is left intact for any
    thread still holding it); it is swapped BEFORE the memos are cleared, so no memo is refilled from the old reader.
    Then the ``BUILD_CACHES``, the WordNet readiness probe, and the token encoder.
    The loaded spaCy model stays (``graph.g1_nlp.pipeline`` holds it and never caches a failure), as do the
    pure and correctly keyed caches."""
    if (corpus := sys.modules.get("nltk.corpus")) is not None:
        util = importlib.import_module("nltk.corpus.util")
        reader = importlib.import_module("nltk.corpus.reader")
        # as nltk.corpus defines it
        corpus.wordnet = util.LazyCorpusLoader(
            "wordnet",
            reader.WordNetCorpusReader,
            util.LazyCorpusLoader("omw-1.4", reader.CorpusReader, r".*/wn-data-.*\.tab", encoding="utf8"),
        )
    for name, attrs in BUILD_CACHES.items():
        module = sys.modules.get(name)
        if module is not None:
            for attr in attrs:
                getattr(module, attr).cache_clear()
    if (module := sys.modules.get("memory_kernel.grammar.dependencies")) is not None:
        module._WORDNET_READY = None  # noqa: SLF001
    if (module := sys.modules.get("memory_kernel.search.kg_context")) is not None:
        module._ENC = None  # noqa: SLF001


@contextlib.contextmanager
def build_data(cfg: BuildConfig) -> Iterator[None]:
    """An in-process build's view of its data dir, as a ``mk _stage`` child had it: ``<data>/nltk_data`` (``--data-dir``,
    else ``MK_DATA_DIR``, else the user data dir, as ``cli.wire_runtime_data`` resolves it in the child) first on
    ``nltk.data.path`` when it exists, then ``clear_build_caches``. The earlier ``nltk.data.path`` is restored when the
    build ends. The caller holds ``build_lock``."""
    import nltk.data

    from memory_kernel import setup_runtime

    corpus = str(setup_runtime.resolve_data_dir(cfg.data_dir, os.environ) / "nltk_data")
    prior = list(nltk.data.path)
    if os.path.isdir(corpus):
        nltk.data.path[:] = [corpus, *(p for p in prior if p != corpus)]
    try:
        clear_build_caches()
        yield
    finally:
        nltk.data.path[:] = prior


def stage_body(cfg: BuildConfig, stage: str, *, live: bool, lock_fd: int, concurrency: int) -> dict:
    """Run ``STAGES[stage]`` once with a fresh ``Ctx``, log its twin re-asks, write its import audit; return the result
    doc ``{"stage", "mode", "gate", "result"}``. The caller already holds the driver lock (``lock_fd``)."""
    from memory_kernel.pipeline import runner
    from memory_kernel.pipeline.stages import STAGES

    mode = "live" if live else "plan"
    ctx = runner.Ctx(
        cfg=cfg,
        concurrency=concurrency,
        lock_fd=lock_fd,
        batcher=llm.QuestionBatcher(llm.lost_wires(runner.lost_wire_roots(cfg))),
    )
    try:
        out = STAGES[stage](ctx, live)
    finally:
        if ctx.batcher.twins:
            runner.log_event(cfg, "twin-reask", stage, mode=mode, twins=list(ctx.batcher.twins))
    audit = runner.path(cfg, "logs") / "mk" / f"imports-{stage}-{mode}.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(runner.audited_modules(cfg), indent=1) + "\n")
    return {"stage": stage, "mode": mode, "gate": runner._gate(stage, out), "result": out}
