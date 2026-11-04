"""The hosted demo: one read-only route over one bonfire, with a rate limit and a daily cap on paid questions.

`mk serve` has no authentication, so it is never exposed. This app is all the internet reaches. It answers through the
same `Server.ask` as `mk ask`, on a read-only Neo4j driver, and it can only ask: it has no route that teaches, appends
or changes anything.

    DEMO_BONFIRE=example-dialogue NEO4J_URI=bolt://localhost:7687 uvicorn app:app --app-dir examples/demo

The secrets (NEO4J_PASSWORD, OPENROUTER_API_KEY, VOYAGE_API_KEY) come from the environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from memory_kernel import api
from memory_kernel.serve.answer import Server
from memory_kernel.stopwords import english_stopwords

HERE = Path(__file__).resolve().parent
SAMPLE_PATH = Path(os.environ.get("DEMO_SAMPLE", HERE.parent / "synthetic_dialogue.json"))
BONFIRE = os.environ.get("DEMO_BONFIRE", "example-dialogue")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
#: questions one address may ask in a minute
PER_MINUTE = int(os.environ.get("DEMO_PER_MINUTE", "6"))
#: paid questions a UTC day, across everyone: the spend cap. A cached question is free and does not count.
DAILY_QUESTIONS = int(os.environ.get("DEMO_DAILY_QUESTIONS", "500"))
MAX_QUESTION_CHARS = 200
#: questions that may wait for the one running; more are told to try again
MAX_WAITING = 8
#: messages shown as the answer's evidence
EVIDENCE_MESSAGES = 3
CACHE_SIZE = 2048

log = logging.getLogger("demo")


def read_sample(path: Path) -> dict[str, Any]:
    """The sample as the page shows it: sessions of messages, plus the questions it ships with."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    conversation = doc["conversation"]
    numbers = sorted(int(m.group(1)) for k in conversation if (m := re.fullmatch(r"session_(\d+)", k)))
    sessions = [
        {
            "when": conversation[f"session_{n}_date_time"],
            "messages": [
                {"id": turn["dia_id"], "speaker": turn["speaker"], "text": turn["text"]}
                for turn in conversation[f"session_{n}"]
            ],
        }
        for n in numbers
    ]
    return {"sessions": sessions, "questions": [q["question"] for q in doc["qa"]]}


def normalise(question: str) -> str:
    return " ".join(question.split()).casefold()


def content_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - english_stopwords()


class Limits:
    """Per-address and daily counts, in memory: a restart resets them, and one process serves the demo."""

    def __init__(self, per_minute: int, daily: int, clock: Any = time.monotonic) -> None:
        self.per_minute, self.daily, self.clock = per_minute, daily, clock
        self.recent: dict[str, deque[float]] = defaultdict(deque)
        self.day = ""
        self.paid = 0

    def allow_address(self, address: str) -> bool:
        now = self.clock()
        seen = self.recent[address]
        while seen and now - seen[0] > 60:
            seen.popleft()
        if len(seen) >= self.per_minute:
            return False
        seen.append(now)
        if len(self.recent) > 10_000:  # a flood of addresses cannot grow this without bound
            self.recent = defaultdict(deque, {address: seen})
        return True

    def take_paid(self) -> bool:
        today = datetime.now(UTC).date().isoformat()
        if today != self.day:
            self.day, self.paid = today, 0
        if self.paid >= self.daily:
            return False
        self.paid += 1
        return True


class Answers:
    """One bonfire's answers: a cache in front of `Server.ask`, one question at a time."""

    def __init__(self, server: Server, sample: dict[str, Any], limits: Limits) -> None:
        self.server, self.limits = server, limits
        self.by_id = {m["id"]: {**m, "when": s["when"]} for s in sample["sessions"] for m in s["messages"]}
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lock = asyncio.Lock()
        self.waiting = 0

    def evidence(self, question: str, answer: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        """The messages behind the answer. The search ranks messages by the question, so the one that holds the answer
        can rank low, and on a conversation this short the context holds nearly all of them. So a message of the context
        is picked by the answer's words, then the question's, then the search's rank. No answer, no evidence."""
        if normalise(answer).startswith("i don't know"):
            return []
        wanted, asked = content_words(answer), content_words(question)
        rank = {message_id.split("#")[0]: i for i, message_id in reversed(list(enumerate(context["retrieved"])))}
        rows = []
        for message_id in dict.fromkeys(m.split("#")[0] for m in context["message_ids"]):  # parts are `<id>#n`
            message = self.by_id.get(message_id)
            if message is None:
                continue
            words = content_words(f"{message['speaker']} {message['text']}")
            if wanted & words:
                rows.append((-len(wanted & words), -len(asked & words), rank.get(message_id, len(rank)), message_id))
        rows.sort()
        best = -rows[0][0] if rows else 0
        return [self.by_id[row[3]] for row in rows[:EVIDENCE_MESSAGES] if -row[0] * 2 >= best]  # a weak match is noise

    async def ask(self, question: str) -> dict[str, Any] | None:
        """The answer, or None when the demo is busy. Raises nothing the route must hide beyond a provider error."""
        key = normalise(question)
        if key in self.cache:
            self.cache.move_to_end(key)
            return {**self.cache[key], "cached": True}
        if self.waiting >= MAX_WAITING:
            return None
        self.waiting += 1
        try:
            async with self.lock:
                if key in self.cache:  # someone ahead in the queue asked the same thing
                    return {**self.cache[key], "cached": True}
                if not self.limits.take_paid():
                    raise DailyCap
                started = time.monotonic()
                out = await self.server.ask(question)
        finally:
            self.waiting -= 1
        result = {
            "answer": out["answer"],
            "shape": out["frame"].get("shape"),
            "evidence": self.evidence(question, out["answer"], out["context"]),
            "context": out["context"]["text"],
            "seconds": round(time.monotonic() - started, 2),
        }
        self.cache[key] = result
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)
        return {**result, "cached": False}


class DailyCap(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    sample = read_sample(SAMPLE_PATH)
    cfg, _meta = await api.bonfire_config(BONFIRE, {"neo4j_uri": NEO4J_URI, "allow_paid": True})
    async with Server(cfg) as server:
        app.state.sample = sample
        app.state.limits = Limits(PER_MINUTE, DAILY_QUESTIONS)
        app.state.answers = Answers(server, sample, app.state.limits)
        for question in sample["questions"]:  # the suggested questions answer at once, and stay free
            try:
                await app.state.answers.ask(question)
            except Exception:  # noqa: BLE001 -- a provider hiccup must not stop the demo from starting
                log.exception("warming a sample question failed")
        yield


app = FastAPI(title="memory-kernel demo", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


def client_address(request: Request) -> str:
    # Caddy sets X-Real-IP from the connection, so a client cannot choose it.
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")


@app.get("/")
async def page() -> FileResponse:
    return FileResponse(HERE / "index.html", media_type="text/html; charset=utf-8")


@app.get("/api/sample")
async def sample(request: Request) -> dict[str, Any]:
    return request.app.state.sample


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask")
async def ask(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except ValueError:
        body = None
    question = body.get("question") if isinstance(body, dict) else None
    if not isinstance(question, str) or not question.strip():
        return JSONResponse({"error": "Ask a question first."}, status_code=422)
    question = " ".join(question.split())
    if len(question) > MAX_QUESTION_CHARS:
        return JSONResponse({"error": f"Keep it under {MAX_QUESTION_CHARS} characters."}, status_code=422)
    state = request.app.state
    cached = normalise(question) in state.answers.cache
    if not cached and not state.limits.allow_address(client_address(request)):
        return JSONResponse({"error": "That's a lot of questions. Try again in a minute."}, status_code=429)
    try:
        result = await state.answers.ask(question)
    except DailyCap:
        return JSONResponse(
            {"error": "The demo has answered all it can for today. The suggested questions still work."},
            status_code=429,
        )
    except Exception:  # noqa: BLE001 -- the page gets a plain message; the log gets the rest
        log.exception("a question failed")
        return JSONResponse({"error": "The demo couldn't answer that just now. Try again."}, status_code=503)
    if result is None:
        return JSONResponse({"error": "The demo is busy. Try again in a moment."}, status_code=429)
    return JSONResponse(result)
