"""The managed syntactic-parser service: the two syntactic-parser routes and ``/health``, nothing else.

The FCG Go engine's de-render asks a parser host for a constituency + dependency analysis of each message
(``POST /spacy-api/syntactic-parser``, body ``{"sentence": ..., "model": ...}``). This app serves that wire from the
copies next to it (``facade``, ``facade_transform``, ``constituency``, ``registry``, ``cache``, ``config``):
the same handler, the same response class (``text/html; charset=utf-8``, which the engine's HTTP client needs to decode
the body), the same shared spaCy pipeline with benepar attached lazily, the same LRU. Both paths are served because
the engine strips the ``/spacy-api`` prefix when its host string contains ``127.0.0.1`` (``facade`` docstring).

``mk build`` starts it as a subprocess (``pipeline.engine.ParserService``); nothing in the package imports this module,
so FastAPI, spaCy, benepar and torch load only in that subprocess. Needs the ``[parser]`` extra and the benepar model
that ``mk setup --component parser-model`` places (``NLTK_DATA``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

#: The two paths of the syntactic-parser wire (prefixed and bare), as the copied facade registers them.
SYNTACTIC_PARSER_PATHS: tuple[str, ...] = ("/spacy-api/syntactic-parser", "/syntactic-parser")


def health() -> dict[str, str]:
    return {"status": "ok"}


def create_app() -> Any:
    """The FastAPI app. The spaCy pipeline is loaded at startup (``/health`` answers once it is loaded); benepar is
    attached on the first parse, as in the copied registry."""
    from fastapi import FastAPI

    from .facade import PenelopeHTMLJSONResponse, syntactic_parser
    from .registry import get_nlp

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        get_nlp()
        yield

    app = FastAPI(title="memory-kernel parser service", lifespan=lifespan)
    for path in SYNTACTIC_PARSER_PATHS:
        app.add_api_route(path, syntactic_parser, methods=["POST"], response_class=PenelopeHTMLJSONResponse)
    app.add_api_route("/health", health, methods=["GET"])
    return app
