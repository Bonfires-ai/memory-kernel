"""The environment names the package reads are neutral ``MK_*`` names; the retired private names are read nowhere."""

import pathlib

import memory_kernel
from memory_kernel.construction.fcg import penelope_host
from memory_kernel.grammar import parse_cache
from memory_kernel.knowledge import temporal_deixis

SRC = pathlib.Path(memory_kernel.__file__).parent

# The retired subsystem's names (its env prefix and its class prefix) are banned by digest in the hazard ratchet
# (tests/test_hazard_ratchet.py), which scans every tracked file; the retired spelling of the package prefix is built at
# runtime so that this file does not itself spell it.
_RETIRED = ["MEMORY" + "_KERNEL_" + k for k in ("SPACY_MODEL", "ENGLISH_STOPWORDS_FILE")]
_NEW = (
    "MK_BOUNDARY_SHIFT_WINDOW",
    "MK_BOUNDARY_SMART_MASK",
    "MK_RETRIEVAL_EMBED_BATCH_COUNT",
    "MK_GRAMMAR_PARSE_CACHE",
    "MK_TEMPORAL_DEIXIS",
    "MK_FCG_DIALOG_DEBUGGER",
    "MK_FCG_NO_NEIGHBORS_GUARD",
    "MK_FCG_NO_COMPREHEND_CANCEL",
    "MK_FCG_PROVISIONAL_SPAN_WIDTH",
    "MK_FCG_PROVISIONAL_TOKEN_CEILING",
    "MK_GRAMMAR_SPACY_MODEL",
    "MK_SPACY_GPU",
    "MK_STOPWORDS_FILE",
    "MK_SERVE_MAX_RUNTIME_ENGINES",
    "MK_SERVE_RUNTIME_PORTS",
    "MK_SERVE_RESUME_WAIT_S",
    "MK_SERVE_TEACH_WAIT_MS",
)


def _source_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(SRC.rglob("*.py")))


def test_no_retired_name_is_read_or_defined_anywhere_in_the_package():
    text = _source_text()
    assert [n for n in _RETIRED if n in text] == []


def test_the_neutral_names_are_the_ones_the_readers_use():
    text = _source_text()
    assert [n for n in _NEW if n not in text] == []
    assert temporal_deixis.TEMPORAL_DEIXIS_ENV == "MK_TEMPORAL_DEIXIS"
    assert parse_cache._CACHE_ENV == "MK_GRAMMAR_PARSE_CACHE"


def test_the_public_data_classes_have_neutral_names():
    from memory_kernel import ingest

    for name in ("Fact", "Topic", "Episode", "Hypergraph", "Config", "RetrievalIndexBuilder"):
        assert hasattr(ingest, name)


def test_the_parser_host_is_read_from_the_neutral_names(monkeypatch):
    assert penelope_host.PENELOPE_HOST_ENV == "MK_FCG_PARSER_HOST"
    assert penelope_host.PENELOPE_HOST_REQUIRE_REACHABLE_ENV == "MK_FCG_PARSER_HOST_REQUIRE_REACHABLE"
    monkeypatch.setenv("GM_FCG_PENELOPE_HOST", "http://retired.invalid")
    monkeypatch.delenv("MK_FCG_PARSER_HOST", raising=False)
    assert penelope_host.configured_penelope_host() is None
    monkeypatch.setenv("MK_FCG_PARSER_HOST", "http://127.0.0.1:2")
    assert penelope_host.configured_penelope_host() == "http://127.0.0.1:2"


def test_the_runtime_engine_settings_are_the_prefix_and_the_field_name():
    from memory_kernel.serve.runtime import RuntimeSettings

    assert set(RuntimeSettings.env_names().values()) == {
        "MK_SERVE_MAX_RUNTIME_ENGINES",
        "MK_SERVE_RUNTIME_PORTS",
        "MK_SERVE_RESUME_WAIT_S",
        "MK_SERVE_TEACH_WAIT_MS",
    }
