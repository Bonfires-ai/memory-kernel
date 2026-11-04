"""Serve: ``answer(question, cfg)`` -> query frame + answer context + answer.

Ported from the reference harness's frozen answer run: the retrieval and context configs, the query embedder,
the persisted-index loader and context builder, the chat client, the wh-slot, directive, answer-prompt
helpers and the question shape.

Not carried over from the harness: its module-level rebindings, any read
or write of ``os.environ`` (keys arrive in ``ServeConfig``), any ``docker exec`` (Neo4j credentials arrive in
``ServeConfig``), the singleton ``chat()`` (one ``Chat`` per ``Server``) and the module-level ``_SPACY`` /
``_CHAT`` caches (they live on the ``Server``).

Read only. Every Neo4j session is opened with ``default_access_mode=READ`` (``ReadOnlyDriver``); ``kg_index``'s
persist path (``persist_keys``, which SETs) is never called. Nothing is written to disk: the question vectors live in
a process-wide LRU (``QUERY_VECTOR_LRU`` entries, keyed by the bonfire's ``query_cache_key`` and the text's sha256) and
each ``Server``'s answers in memory. A miss sends only when ``allow_paid`` is set (without it: ``ServeMiss``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

#: First-person words of a question, resolved to the asker (``resolve_first_person``), in this order. "you" is left alone:
#: in a question to the agent it is the agent, which the bonfire already knows by its handle.
_FIRST_PERSON: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bI['’]m\b", re.IGNORECASE), "{s} is"),
    (re.compile(r"\bI['’]ve\b", re.IGNORECASE), "{s} has"),
    (re.compile(r"\bI['’]d\b", re.IGNORECASE), "{s} would"),
    (re.compile(r"\bI['’]ll\b", re.IGNORECASE), "{s} will"),
    (re.compile(r"\bmyself\b", re.IGNORECASE), "{s}"),
    (re.compile(r"\b(?:my|mine)\b", re.IGNORECASE), "{s}'s"),
    (re.compile(r"\bme\b", re.IGNORECASE), "{s}"),
    (re.compile(r"\bI\b", re.IGNORECASE), "{s}"),
)


def resolve_first_person(question: str, speaker: str | None) -> str:
    """``question`` with its first-person words (I, me, my, mine, myself, I'm/I've/I'd/I'll) replaced by ``speaker``,
    the asker's name as the bonfire keys people (the platform username). Without a speaker the question is returned
    unchanged: the kernel cannot know who "me" is, and "what do you know about me?" retrieved the agent's own sheet
    while the agent answered with whoever spoke last. A leading "@" on the speaker is dropped."""
    name = (speaker or "").strip().lstrip("@").strip()
    if not name:
        return question
    out = question
    for pattern, replacement in _FIRST_PERSON:
        out = pattern.sub(replacement.replace("{s}", name), out)
    return out


RECIPE_TAG = "legacy|deixis=on|v4"  # the frozen recipe's key tag
MODEL = "openai/gpt-4.1-mini"
MAX_ANSWER_TOKENS = 120
#: the runtime grammar's answer shape of a yes/no question (a contract label, as ``TurnPlan.answer_shape`` writes it)
POLAR = "polar"

ANSWER_PROMPT = """You are a question-answering system. Answer the question using only the memories below.
- Be concise: a short phrase or a single fact, not a paragraph.
- Use the dates written in the memory lines; resolve relative dates against the date of the line that mentions them.
- If the memories do not contain the answer, say "I don't know".

Memories:
{context}

Question: {question}
Answer:"""


class ServeMiss(RuntimeError):
    """A cache miss (answer or query vector) with ``allow_paid`` off: serving never sends."""


#: entries of the process-wide question-vector LRU (``Server.query_vector``)
QUERY_VECTOR_LRU = 4096
_VECTORS: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
_VECTORS_LOCK = threading.Lock()


def _vector_get(key: tuple[str, str]) -> list[float] | None:
    with _VECTORS_LOCK:
        vector = _VECTORS.get(key)
        if vector is not None:
            _VECTORS.move_to_end(key)
        return None if vector is None else list(vector)  # a copy: a caller never mutates the cached list


def _vector_put(key: tuple[str, str], vector: list[float]) -> None:
    with _VECTORS_LOCK:
        _VECTORS[key] = vector
        _VECTORS.move_to_end(key)
        while len(_VECTORS) > QUERY_VECTOR_LRU:
            _VECTORS.popitem(last=False)


@dataclass
class Misses:
    """Every refused send, by kind (``chat``: request shas, ``voyage``: query texts' shas)."""

    chat: list[str] = field(default_factory=list)
    voyage: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.chat) + len(self.voyage)


@dataclass(frozen=True)
class ServeConfig:
    group: str
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    # secrets: never in the repr (a logged config must not carry them)
    neo4j_password: str | None = field(default=None, repr=False)
    openrouter_api_key: str | None = field(default=None, repr=False)
    voyage_api_key: str | None = field(default=None, repr=False)
    allow_paid: bool = False  # False = cache only: a miss raises ServeMiss and is recorded
    concurrency: int = 4
    #: the bonfire's ``BonfireMeta.query_cache_key``: question vectors are cached per key, so bonfires of one build
    #: config share their hits and a bonfire of another does not
    query_cache_key: str = ""


def recipe_configs() -> dict:
    """The freeze's retrieval and context configs (the harness's frozen configs, as constants)."""
    from memory_kernel.search import kg_index
    from memory_kernel.search.kg_context import ContextConfig
    from memory_kernel.search.kg_recipe import SEALED_RECIPE

    if kg_index.recipe_tag(SEALED_RECIPE.key_render) != RECIPE_TAG:
        raise ValueError("SEALED_RECIPE key render differs from the freeze's key recipe tag")
    return {
        "recipe": SEALED_RECIPE,
        "context": ContextConfig(evidence_k=40),
        "directive": True,
        "key_recipe_tag": RECIPE_TAG,
    }


# -- read-only Neo4j ----------------------------------------------------------------------------------------------------


class ReadOnlyDriver:
    """Wraps a neo4j async driver so every session is a READ session (the server rejects a write in one)."""

    def __init__(self, driver: Any) -> None:
        self._driver = driver

    def session(self, **kwargs: Any):
        from neo4j import READ_ACCESS

        kwargs["default_access_mode"] = READ_ACCESS
        return self._driver.session(**kwargs)

    async def close(self) -> None:
        await self._driver.close()


def open_driver(cfg: ServeConfig) -> ReadOnlyDriver:
    from neo4j import AsyncGraphDatabase

    if cfg.neo4j_password is None:
        raise ValueError("no Neo4j password configured (NEO4J_PASSWORD, --env-file or --secrets-from-container)")
    return ReadOnlyDriver(
        AsyncGraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
            connection_timeout=5,  # a hung Neo4j answers a fast 503, not a stalled request
            connection_acquisition_timeout=10,
        )
    )


async def load_index(driver: Any, cfg: ServeConfig) -> Any:
    """The persisted index, refusing a stale or incomplete one. Reads only."""
    from memory_kernel.graph.kg_writer import KEY_EMBED_DIM, KEY_EMBED_MODEL
    from memory_kernel.search import kg_index
    from memory_kernel.search.kg_recipe import SEALED_RECIPE

    got = kg_index.recipe_tag(SEALED_RECIPE.key_render)
    if got != RECIPE_TAG:
        raise ValueError(f"key recipe tag {got!r} differs from the freeze's {RECIPE_TAG!r}")
    idx = await kg_index.load_index(
        driver,
        cfg.group,
        model=KEY_EMBED_MODEL,
        dim=KEY_EMBED_DIM,
        key_render=SEALED_RECIPE.key_render,
        purpose="query",
    )
    if idx.stale or idx.missing:
        raise kg_index.IndexStale(idx.stale, idx.missing)
    return idx


# -- chat ---------------------------------------------------------------------------------------


class Chat:
    """gpt-4.1-mini at temperature 0 with an in-memory cache (request sha -> response text)."""

    def __init__(self, sender: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.sender = sender
        self.calls = 0  # paid (uncached) calls
        self.hits = 0
        self._cache: dict[str, str] = {}

    @staticmethod
    def request(prompt: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

    def key(self, prompt: str, max_tokens: int) -> str:
        return hashlib.sha256(json.dumps(self.request(prompt, max_tokens), sort_keys=True).encode()).hexdigest()

    def __call__(self, prompt: str, max_tokens: int = 200) -> str:
        key = self.key(prompt, max_tokens)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        out = self.sender(self.request(prompt, max_tokens))
        self.calls += 1
        self._cache[key] = out["text"]
        return out["text"]


# -- question analysis ---------------------------------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def _parsed_shape(text: str, nlp: Any) -> str:
    """One parse per question text: the directive and the context frame of a turn share it. A failure is not cached.
    The text is parsed with its whitespace runs collapsed, the text ``read_shape`` reads (a double space or a newline
    misparses), so ``read_shape`` reads this Doc instead of parsing the question a second time."""
    from memory_kernel.runtime.shape import read_shape

    return read_shape(nlp(" ".join(text.split())), ()).shape or POLAR


_PARSE_WARNED: list[bool] = []


def question_shape(text: str, nlp: Any, fallback: str = POLAR) -> str:
    """The answer shape of a question, read from its parse (``runtime.shape.read_shape``, no mined patterns: the shape
    label needs none); a text with no question shape is ``polar``, as a text with no wh word always was. It never
    raises: when the parser fails, the shape is ``fallback`` (the first failure is logged)."""
    try:
        return _parsed_shape(text, nlp)
    except Exception as exc:  # the parser is optional machinery here: a failed parse degrades the shape, not the turn
        if not _PARSE_WARNED:
            _PARSE_WARNED.append(True)
            logging.getLogger(__name__).warning(
                "answer shape: the question parse failed (%s); falling back to %r",
                type(exc).__name__,
                fallback,
            )
        return fallback


def directive_for(question: str, nlp: Any, answer_shape: str | None = None) -> str | None:
    """Production's answer directive, ``directive_preamble()`` + newline + ``answer_directive(shape)``. The shape is the
    runtime reading's ``answer_shape`` when the caller has one (the question is not parsed), else it is read from the
    parse of the question; ``polar`` and a text with no shape get no directive."""
    from memory_kernel.search.recipe_affordances import answer_directive, directive_preamble

    shape = answer_shape if answer_shape is not None else question_shape(question, nlp)
    slot = answer_directive(shape)
    return f"{directive_preamble()}\n{slot}" if slot else None


def answer_prompt(
    question: str, context: str, *, nlp: Any, directive: bool = True, answer_shape: str | None = None
) -> str:
    """The answer prompt; ``answer_shape`` (a runtime reading's, when there is one) picks the directive."""
    text = directive_for(question, nlp, answer_shape) if directive else None
    block = f"ANSWER DIRECTIVE:\n{text}\n\n" if text else ""
    head, tail = ANSWER_PROMPT.split("Question: {question}")
    return (head + block + "Question: {question}" + tail).format(context=context, question=question)


# -- the server ---------------------------------------------------------------------------------------------------------


class Server:
    """One conversation's serving state: driver + index + caches + senders. ``async with Server(cfg) as srv``."""

    def __init__(self, cfg: ServeConfig) -> None:
        self.cfg = cfg
        self.misses = Misses()
        self.rc = recipe_configs()
        self._driver: ReadOnlyDriver | None = None
        self._idx: Any = None
        #: message uuid -> its text with its relative-time note (``kg_context.relative_texts``), computed at open
        self._relative: dict[str, str] | None = None
        self._nlp: Any = None
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self.chat = Chat(self._chat_sender)

    async def __aenter__(self) -> Server:
        from memory_kernel.search.kg_context import relative_texts

        self._driver = open_driver(self.cfg)
        try:
            self._idx = await load_index(self._driver, self.cfg)
            self._relative = await asyncio.to_thread(relative_texts, self._idx.messages)
        except BaseException:  # a server that does not open leaves no driver behind
            driver, self._driver = self._driver, None
            await driver.close()
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._driver is not None:
            await self._driver.close()

    # senders: a miss is recorded and refused unless allow_paid
    def _chat_sender(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.cfg.allow_paid:
            self.misses.chat.append(hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest())
            raise ServeMiss("answer cache miss and serving is cache-only (--allow-paid to send)")
        from memory_kernel.pipeline.llm import OpenRouterChat

        return OpenRouterChat(api_key=self.cfg.openrouter_api_key)(request)

    @property
    def nlp(self) -> Any:
        if self._nlp is None:
            from memory_kernel.graph.g1_nlp import turn_doc

            self._nlp = turn_doc  # the runtime's one question parser, the parse the answer shape was judged on
        return self._nlp

    def query_vector(self, text: str, *, paid: bool = True) -> list[float]:
        """The question's vector from the process-wide LRU (keyed ``(query_cache_key, sha256(text))``); Voyage (query
        input type) only with allow_paid (and ``paid``: False never sends, whatever the config). A refused miss is
        never cached. Thread-safe (``context`` runs it in a worker thread); the send happens outside the LRU's lock."""
        key = (self.cfg.query_cache_key or f"bonfire:{self.cfg.group}", hashlib.sha256(text.encode()).hexdigest())
        if (hit := _vector_get(key)) is not None:
            return hit
        if not (self.cfg.allow_paid and paid):
            self.misses.voyage.append(key[1])
            raise ServeMiss("query embedding cache miss and serving is cache-only (--allow-paid to send)")
        if not self.cfg.voyage_api_key:
            raise ValueError("no Voyage key configured")
        from memory_kernel.graph.kg_writer import VoyageEmbedder

        vector = list(VoyageEmbedder(self.cfg.voyage_api_key, input_type="query")([text])[0])
        _vector_put(key, vector)
        return vector

    def query_frame(self, text: str, *, shape: str | None = None) -> dict[str, Any]:
        """``{"shape"}``: the question's answer shape, ``shape`` (a runtime reading's) when given, else read from the
        question's parse by the wh-rules. No cache, no ledger, no model call."""
        return {"shape": shape if shape is not None else question_shape(text, self.nlp)}

    async def context(self, question: str, *, paid: bool = True, shape: str | None = None) -> dict[str, Any]:
        """The context for one question: frame (``{"shape"}``), text, message_ids, tokens, retrieved, degraded (always
        False: a frame cannot fail). The query vector is read in a worker thread (a miss may send); the search over the
        preloaded index runs once, in a worker thread (a cosine over every key), and its result serves both the context
        and ``retrieved``; the async Neo4j reads stay on this loop and the assembly runs in a worker thread
        (``build_answer_context``). The question is parsed once, for its shape (``_parsed_shape``); no message is
        parsed: their relative-time notes were read when the run opened. ``paid`` False answers from the caches only
        (a query-vector miss raises ``ServeMiss``), whatever ``allow_paid`` says. ``shape`` (the runtime reading's
        answer shape, when the caller has one) is the frame's shape instead of the wh-rule's."""
        from memory_kernel.search.kg_reader import build_answer_context
        from memory_kernel.search.kg_recipe import search_index

        cfg = self.rc
        async with self._sem:
            fr, vec = await asyncio.gather(
                asyncio.to_thread(partial(self.query_frame, question, shape=shape)),
                asyncio.to_thread(partial(self.query_vector, question, paid=paid)),
            )
            res = await asyncio.to_thread(
                partial(search_index, self._idx, qvec=vec, config=cfg["recipe"], query=question)
            )

            async def searched(*_a: Any, **_k: Any) -> Any:  # the question's one search (weights/shape unused)
                return res

            ctx = await build_answer_context(
                self._driver,
                question,
                vec,
                group_id=self.cfg.group,
                shape=fr["shape"],
                label=None,
                search_config=cfg["recipe"],
                context_config=cfg["context"],
                search_fn=searched,
                relative=self._relative,
            )
            return {
                "frame": fr,
                "text": ctx["text"],
                "message_ids": ctx["message_ids"],
                "tokens": ctx["tokens"],
                "retrieved": [str(res.messages[h.message]["message_id"]) for h in res.hits][:20],
                "degraded": False,
            }

    async def entity_sheet(self, *, name: str | None = None, uuid: str | None = None) -> list[dict[str, Any]]:
        """Who is ``name`` (or ``uuid``): the entities' sheets with each line's source message ids
        (``kg_reader.entity_sheet_view``). Neo4j reads only; no model call."""
        from memory_kernel.search.kg_reader import entity_sheet_view

        return await entity_sheet_view(self._driver, self.cfg.group, name=name, uuid=uuid)

    def answer_text(self, question: str, context_text: str, answer_shape: str | None = None) -> str:
        """The answer; ``answer_shape`` (a runtime reading's, when there is one) picks the directive."""
        prompt = answer_prompt(question, context_text, nlp=self.nlp, directive=True, answer_shape=answer_shape)
        return self.chat(prompt, MAX_ANSWER_TOKENS).strip()

    async def ask(self, question: str, answer_shape: str | None = None) -> dict[str, Any]:
        ctx = await self.context(question, shape=answer_shape)
        return {
            "question": question,
            "frame": ctx["frame"],
            "context": ctx,
            "answer": await asyncio.to_thread(self.answer_text, question, ctx["text"], answer_shape),
        }


def answer(question: str, cfg: ServeConfig) -> dict[str, Any]:
    """``{"question", "frame", "context": {...}, "answer"}``: sync wrapper over one ``Server``."""

    async def run() -> dict[str, Any]:
        async with Server(cfg) as srv:
            return await srv.ask(question)

    return asyncio.run(run())


def import_audit(roots: tuple[Path, ...]) -> list[str]:
    """Modules loaded in this process from a ``scripts`` tree (the same rule as the build's per-stage audit)."""
    import sys

    found = set()
    for name, mod in list(sys.modules.items()):
        file = getattr(mod, "__file__", None)
        if (
            name.startswith("scripts.")
            or name == "scripts"
            or name == "integrated_jev_provider"
            or file
            and any(Path(file).is_relative_to(r / "scripts") for r in roots)
        ):
            found.add(name)
    return sorted(found)


__all__ = [
    "RECIPE_TAG",
    "Chat",
    "Misses",
    "ReadOnlyDriver",
    "ServeConfig",
    "ServeMiss",
    "Server",
    "answer",
    "answer_prompt",
    "import_audit",
    "load_index",
    "open_driver",
    "recipe_configs",
]
