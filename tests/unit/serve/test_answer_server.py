"""``answer.Server`` under ``mk serve``: the query vector runs in a worker thread, so identical misses race; the
vectors live in a process-wide LRU shared by every server of a bonfire's query-cache key and nothing reaches disk; a
server that fails to open closes its driver; and the query frame is the wh-rules' shape alone (no model call, no frame
cache). Offline: fake senders, fake driver, fake retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_kernel.serve import answer
from memory_kernel.serve.answer import ServeConfig, ServeMiss, Server


def config(tmp_path: Path, key: str | None = None, **kw: Any) -> ServeConfig:
    """A config whose query-cache key is its own (the vector LRU is process-wide: a test never meets another's)."""
    return ServeConfig(group="g", query_cache_key=key or f"test-{tmp_path.name}", **kw)


def seed_vector(cfg: ServeConfig, text: str, vector: list[float]) -> None:
    answer._vector_put((cfg.query_cache_key, hashlib.sha256(text.encode()).hexdigest()), vector)


def race(fn, args: list[Any]) -> list[Any]:
    with ThreadPoolExecutor(len(args)) as pool:
        return list(pool.map(fn, args))


# -- the thread hazards ------------------------------------------------------------------------------------------------


def test_question_vector_lru_hits_without_disk(tmp_path, monkeypatch):
    """Threads and two servers of one bonfire (a swap) race on the same questions: each text is a hit after its first
    send; nothing is written to disk (the working directory stays empty); a cache-only server hits what was sent."""
    monkeypatch.chdir(tmp_path)
    sent: list[str] = []

    class FakeVoyage:
        def __init__(self, key: str, input_type: str) -> None:
            assert input_type == "query"

        def __call__(self, texts: list[str]) -> list[list[float]]:
            time.sleep(0.02)
            sent.extend(texts)
            return [[float(len(t))] for t in texts]

    monkeypatch.setattr("memory_kernel.graph.kg_writer.VoyageEmbedder", FakeVoyage)
    one = Server(config(tmp_path, allow_paid=True, voyage_api_key="v"))
    two = Server(config(tmp_path, allow_paid=True, voyage_api_key="v"))  # a swap: same key, same LRU
    texts = [f"question {i}" for i in range(6)]
    jobs = [(s, t) for t in texts for s in (one, two)] + [(one, "same")] * 4
    out = race(lambda job: job[0].query_vector(job[1]), jobs)
    assert out == [[float(len(t))] for _, t in jobs]
    before = len(sent)
    fresh = Server(config(tmp_path))  # cache-only: every vector is a hit
    assert all(fresh.query_vector(t) == [float(len(t))] for t in [*texts, "same"])
    assert len(sent) == before and fresh.misses.voyage == []
    assert list(tmp_path.iterdir()) == []


def test_a_refused_miss_is_not_cached_and_another_key_does_not_share(tmp_path):
    cache_only = Server(config(tmp_path))
    with pytest.raises(ServeMiss):
        cache_only.query_vector("never sent")
    assert cache_only.misses.voyage == [hashlib.sha256(b"never sent").hexdigest()]
    seed_vector(config(tmp_path), "seeded", [1.0])
    assert cache_only.query_vector("seeded") == [1.0]
    with pytest.raises(ServeMiss):
        Server(config(tmp_path, "another-build-config")).query_vector("seeded")


def test_bonfires_without_a_query_cache_key_do_not_share_vectors(tmp_path):
    one = Server(ServeConfig(group=f"{tmp_path.name}-a"))
    other = Server(ServeConfig(group=f"{tmp_path.name}-b"))
    answer._vector_put((f"bonfire:{tmp_path.name}-a", hashlib.sha256(b"q").hexdigest()), [1.0])
    assert one.query_vector("q") == [1.0]
    with pytest.raises(ServeMiss):
        other.query_vector("q")
    got = one.query_vector("q")
    got.append(9.0)  # a caller's mutation never reaches the cache
    assert one.query_vector("q") == [1.0]


def test_the_vector_lru_evicts_the_least_recently_used(monkeypatch):
    monkeypatch.setattr(answer, "QUERY_VECTOR_LRU", 2)
    monkeypatch.setattr(answer, "_VECTORS", answer.OrderedDict())
    answer._vector_put(("k", "a"), [1.0])
    answer._vector_put(("k", "b"), [2.0])
    assert answer._vector_get(("k", "a")) == [1.0]  # a is now the newest
    answer._vector_put(("k", "c"), [3.0])
    assert answer._vector_get(("k", "b")) is None and answer._vector_get(("k", "a")) == [1.0]


def test_the_answer_cache_is_in_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sent: list[dict] = []
    chat = answer.Chat(lambda request: sent.append(request) or {"text": "yes"})
    assert [chat("q?", 10), chat("q?", 10)] == ["yes", "yes"]
    assert (len(sent), chat.calls, chat.hits) == (1, 1, 1) and list(tmp_path.iterdir()) == []


# -- opening -----------------------------------------------------------------------------------------------------------


def test_a_server_whose_index_does_not_load_closes_its_driver(tmp_path, monkeypatch):
    closed: list[bool] = []

    class FakeDriver:
        async def close(self) -> None:
            closed.append(True)

    async def stale(driver: Any, cfg: ServeConfig) -> Any:
        raise ValueError("index stale")

    monkeypatch.setattr(answer, "open_driver", lambda cfg: FakeDriver())
    monkeypatch.setattr(answer, "load_index", stale)

    async def enter() -> None:
        async with Server(config(tmp_path)):
            pass

    with pytest.raises(ValueError, match="index stale"):
        asyncio.run(enter())
    assert closed == [True]


# -- the query frame ----------------------------------------------------------------------------------------------------


@pytest.fixture
def retrieval(monkeypatch):
    """Fake retrieval: the context names the label it was given."""
    seen: list[Any] = []

    async def build_answer_context(driver, question, vec, *, label, **kw):
        seen.append(label)
        return {"text": f"memories for {question} (label {label})", "message_ids": ["D1:1"], "tokens": 9}

    def search(idx, **kw):
        return SimpleNamespace(hits=[], messages={})

    monkeypatch.setattr("memory_kernel.search.kg_reader.build_answer_context", build_answer_context)
    monkeypatch.setattr("memory_kernel.search.kg_recipe.search_index", search)
    return seen


def ready(cfg: ServeConfig, question: str) -> Server:
    srv = Server(cfg)
    srv._driver, srv._idx = object(), object()  # type: ignore[assignment]
    seed_vector(cfg, question, [0.1])
    return srv


Q = "When did Nate win his first tournament?"


def test_context_frame_is_shape_only(tmp_path, retrieval):
    out = asyncio.run(ready(config(tmp_path), Q).context(Q))
    assert out["frame"] == {"shape": "when"} and out["degraded"] is False and "reason" not in out
    assert retrieval == [None]  # the context is assembled without a label


def test_context_frame_ignores_cached_label_frames(tmp_path, retrieval):
    srv = ready(config(tmp_path), Q)
    assert asyncio.run(srv.context(Q))["frame"] == {"shape": "when"}  # a frame is read from the parse, never stored


def test_context_cache_only_needs_no_frame_cache(tmp_path, retrieval):
    q = "Who is Joanna?"  # its query vector is cached by ``ready``; no frame is
    out = asyncio.run(ready(config(tmp_path), q).context(q, paid=False))
    assert out["frame"] == {"shape": "who"} and out["degraded"] is False


def test_the_runtime_readings_shape_is_the_frame(tmp_path):
    srv = Server(config(tmp_path))
    assert srv.query_frame(Q, shape="who") == {"shape": "who"}


# -- serving parses only the question, and assembles off the event loop ------------------------------------------------

ANA = "e-ana"
TOLD = "2023-05-03T10:00:00"
MESSAGES = {
    "m1": {
        "uuid": "m1",
        "message_id": "D1:1",
        "content": "Ana: I build test rigs at Brask Labs.",
        "sender_name": "Ana",
    },
    "m2": {"uuid": "m2", "message_id": "D1:2", "content": "Milo: Kira Dace ships the rigs.", "sender_name": "Milo"},
    # a bare weekday: its note reads the sentence's tense from a parse of the message
    "m3": {"uuid": "m3", "message_id": "D1:3", "content": "Milo: see you Friday at the lab", "sender_name": "Milo"},
}
MESSAGES = {u: {**m, "told_at": TOLD} for u, m in MESSAGES.items()}
SHEET_LINES = ["[Ana, 2023-05-03] Ana is an engineer at Brask Labs.", "[Ana, 2023-05-03] Ana builds test rigs."]


class SheetDriver:
    """Neo4j for ``build_answer_context``: three messages of one saga, and Ana's sheet as line records (an identity
    line among them, decided at write). Every other read is empty."""

    def session(self, **_kw: Any) -> SheetDriver:
        return self

    async def __aenter__(self) -> SheetDriver:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def close(self) -> None:
        return None

    async def run(self, cypher: str, **params: Any) -> Any:
        from memory_kernel.search import kg_reader

        rows: list[dict[str, Any]] = []
        if cypher == kg_reader.WINDOW_ANCHORS_CYPHER:
            rows = [{"anchor": u, "saga": "s1", "session": "session 1", "msg": MESSAGES[u]} for u in params["uuids"]]
        elif cypher == kg_reader.ENTITY_SHEET_CYPHER and ANA in params["uuids"]:
            rows = [
                {
                    "uuid": ANA,
                    "name": "Ana",
                    "summary": "\n".join(SHEET_LINES),
                    "line_ids": ["l1", "l2"],
                    "line_texts": SHEET_LINES,
                    "identity_line_ids": ["l1"],
                }
            ]

        class Rows:
            def __aiter__(self):
                async def gen():
                    for r in rows:
                        yield r

                return gen()

        return Rows()


def sheet_config(tmp_path: Path, question: str, monkeypatch) -> ServeConfig:
    """A bonfire whose run opens over ``SheetDriver`` and a three-message index; the question's vector is cached."""
    from memory_kernel.search.kg_index import KgIndex

    index = KgIndex(
        keys={u: m["content"] for u, m in MESSAGES.items()},
        vectors={"m1": [1.0, 0.0], "m2": [0.8, 0.6], "m3": [0.6, 0.8]},
        msg_facts={u: [] for u in MESSAGES},
        msg_events={u: [] for u in MESSAGES},
        msg_entities={"m1": [{"uuid": ANA, "name": "Ana"}], "m2": [], "m3": []},
        told_at={u: TOLD for u in MESSAGES},
        messages={u: {**m, "saga": "s1"} for u, m in MESSAGES.items()},
        dim=2,
    )

    async def opened(driver: Any, cfg: ServeConfig) -> KgIndex:
        return index

    monkeypatch.setattr(answer, "open_driver", lambda cfg: SheetDriver())
    monkeypatch.setattr(answer, "load_index", opened)
    cfg = config(tmp_path)
    seed_vector(cfg, question, [1.0, 0.0])
    return cfg


def test_one_context_call_parses_only_the_question_and_only_once(tmp_path, monkeypatch):
    """One ``/context`` call parses the question once and nothing else. Ana's sheet is read from its line records (an
    identity line among them); the weekday message's relative-time note was read when the run opened. The pin is 1: no
    reader on this path wants a query-purpose Doc of the question; the one parse is the turn reading the answer shape
    is judged on. A subjectless question that does not open on a verb ("What about Ana?") parses twice, through that
    reading's "you " retry: the same reading, not a second reader. Its newline once made the shape parse it again."""
    import spacy.language

    from memory_kernel.grammar import semantic_operators
    from memory_kernel.search import kg_context
    from memory_kernel.serve.answer import _parsed_shape

    q = "What does Ana do\nat Brask Labs?"
    cfg = sheet_config(tmp_path, q, monkeypatch)
    monkeypatch.setattr(kg_context, "count_tokens", lambda text: len(text.split()))  # no token encoder needed
    parses: list[str] = []
    call, pipe = spacy.language.Language.__call__, spacy.language.Language.pipe

    def counted_call(self, text, *a, **k):
        parses.append(str(text))
        return call(self, text, *a, **k)

    def counted_pipe(self, texts, *a, **k):
        texts = list(texts)
        parses.extend(str(t) for t in texts)
        return pipe(self, texts, *a, **k)

    async def ask() -> dict[str, Any]:
        async with Server(cfg) as srv:  # the run opens: the messages' relative-time notes are read here, once
            for cached in (
                _parsed_shape,
                kg_context.identity_line,
                semantic_operators._doc_cached,
                semantic_operators._operator_profile_cached,
            ):
                cached.cache_clear()
            monkeypatch.setattr(spacy.language.Language, "__call__", counted_call)
            monkeypatch.setattr(spacy.language.Language, "pipe", counted_pipe)
            return await srv.context(q)

    out = asyncio.run(ask())
    assert out["frame"] == {"shape": "what"}
    assert SHEET_LINES[0] in out["text"] and out["retrieved"] == ["D1:1", "D1:2", "D1:3"]
    assert "see you Friday at the lab [resolved relative time: friday = " in out["text"]
    assert parses == [" ".join(q.split())]


def test_a_slow_context_assembly_never_blocks_the_event_loop(tmp_path, monkeypatch):
    """The search over the index, the entity ranking and the assembly are synchronous work: they run in worker
    threads, so another coroutine (``/healthz``, a parallel question) keeps running while they do."""
    from memory_kernel.search import kg_reader, kg_recipe

    q = "What does Ana do at Brask Labs?"
    cfg = sheet_config(tmp_path, q, monkeypatch)
    rank, aliased = kg_recipe.rank_recipe, kg_reader.aliased_uuids

    def slow_rank(*a: Any, **k: Any) -> Any:
        time.sleep(0.2)
        return rank(*a, **k)

    def slow_aliases(*a: Any, **k: Any) -> Any:
        time.sleep(0.2)
        return aliased(*a, **k)

    def slow_assembly(*_a: Any, **_k: Any) -> dict[str, Any]:
        time.sleep(0.5)
        return {"text": "assembled", "message_ids": ["D1:1"], "tokens": 1}

    monkeypatch.setattr(kg_recipe, "rank_recipe", slow_rank)
    monkeypatch.setattr(kg_reader, "aliased_uuids", slow_aliases)
    monkeypatch.setattr(kg_reader, "assemble_context", slow_assembly)

    async def run() -> tuple[dict[str, Any], float, float]:
        async with Server(cfg) as srv:
            srv.query_frame(
                q
            )  # a warm server: the parser is loaded before the clock starts (a model load holds the GIL)
            done, gaps = asyncio.Event(), [0.0]

            async def tick() -> None:
                last = time.perf_counter()
                while not done.is_set():
                    await asyncio.sleep(0.01)
                    now = time.perf_counter()
                    gaps.append(now - last)
                    last = now

            ticker = asyncio.create_task(tick())
            await asyncio.sleep(0)
            start = time.perf_counter()
            out = await srv.context(q)
            took = time.perf_counter() - start
            done.set()
            await ticker
            return out, took, max(gaps)

    out, took, worst = asyncio.run(run())
    assert out["text"] == "assembled" and took >= 0.9  # the slow steps ran while the ticker did
    assert worst < 0.1
