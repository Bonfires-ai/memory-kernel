# Derived from an internal parser service, file routes/facade.py, source sha256
# 4d6326c761e6c391cba70f24554c46e5e844e62521ba4e3381f5c8f20c812f62.
# Copied from the origin file; edits: this header, the package-relative imports and a prose-only scrub of docstrings and comments.
"""FCG-Go parser-compatible facade — the wire ``penelope-interface.lisp`` speaks.

TRAP (penelope-interface.lisp:58, ``send-request``): when the configured
host string contains the literal substring "127.0.0.1", the client STRIPS
the "/spacy-api" or "/embedding-api" prefix from the route before sending —
it was written to talk to a bare local Flask app during VUB development, not
through the "/spacy-api" gateway prefix that the hosted parser service
reverse-proxies onto. So every facade route below is registered at BOTH its
prefixed and unprefixed path. Repointing the engine at "http://localhost:PORT"
(not "127.0.0.1") keeps the prefix; either way both paths resolve here.

/embedding-api/* is not implemented (this service has no embedding backend) and
returns 501 with a clear body, at both its prefixed and unprefixed paths.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import facade_transform as ft
from .cache import cached_parse
from .registry import get_nlp


class PenelopeHTMLJSONResponse(JSONResponse):
    """JSONResponse that declares ``Content-Type: text/html; charset=utf-8``.

    This looks like a bug. It is not — it is bit-compatibility with
    the hosted parser service's own wire (Flask behind Apache), which serves this
    exact JSON body under that same "wrong" content type. FCG-Go's Lisp HTTP
    client (Drakma) only string-decodes a response body when the
    Content-Type is ``text/*``; for anything else — INCLUDING the
    standards-correct ``application/json`` — it hands back raw octets
    without decoding, so the JSON is never parsed. The undecoded body then
    reaches the grammar engine as NIL two frames later, surfacing as
    ``FcgError: No applicable methods for POLE-STRUCTURE with args (NIL)`` —
    an error that names neither HTTP nor JSON, which is why it took a
    logging proxy (not code inspection) to trace the first time. Proven by a
    single-variable experiment: only the content-type header needed to
    change for the full FCG gate to pass with a byte-identical grammar
    digest.

    DO NOT "fix" this to ``application/json``. That is the standards-correct
    value and it silently re-breaks FCG-Go comprehension. This class is
    deliberately scoped to the Penelope-facing facade router only — native
    routes consumed by our own Python code keep ``application/json`` (see
    the origin service's native routes).
    """

    media_type = "text/html; charset=utf-8"


router = APIRouter(default_response_class=PenelopeHTMLJSONResponse)


class SentenceRequest(BaseModel):
    sentence: str
    model: str = "en"


class TextRequest(BaseModel):
    text: str
    model: str = "en"


class TextsRequest(BaseModel):
    texts: list[str]
    model: str = "en"


def _add(path: str, endpoint: Any) -> None:
    """Register ``endpoint`` at both "/spacy-api{path}" and the bare "{path}"."""

    router.add_api_route(f"/spacy-api{path}", endpoint, methods=["POST"])
    router.add_api_route(path, endpoint, methods=["POST"])


def _parse_one(sentence: str) -> Any:
    return cached_parse(get_nlp(), sentence)


def _parse_many(texts: list[str]) -> list[Any]:
    nlp = get_nlp()
    return [cached_parse(nlp, text) for text in texts]


# --- single-sentence routes --------------------------------------------------


def tokenize(request: SentenceRequest) -> dict[str, Any]:
    return ft.tokens_payload(_parse_one(request.sentence))


def lemmatize_sentence_payload(sentence: str) -> dict[str, Any]:
    """Facade lemmatize logic, exposed for the origin service's native-route dispatcher.

    ROUTE COLLISION NOTE: the bare path "/lemmatize" is claimed by BOTH
    wires — the native API (``{terms, mode}``) and this facade (``{sentence,
    model}``, reachable here only via the 127.0.0.1-strip trap, since the
    real VUB route is "/spacy-api/lemmatize"). One path can only be
    registered once, so the bare "/lemmatize" handler lives in
    the origin service's native-route module and dispatches on body shape; this module only registers
    the prefixed "/spacy-api/lemmatize" path directly.
    """

    return ft.lemmas_payload(_parse_one(sentence))


def lemmatize(request: SentenceRequest) -> dict[str, Any]:
    return lemmatize_sentence_payload(request.sentence)


def noun_chunks(request: SentenceRequest) -> dict[str, Any]:
    return ft.noun_chunks_payload(_parse_one(request.sentence))


def pos(request: SentenceRequest) -> dict[str, Any]:
    return ft.pos_payload(_parse_one(request.sentence))


def named_entities(request: SentenceRequest) -> dict[str, Any]:
    return ft.named_entities_payload(_parse_one(request.sentence))


def dependencies(request: SentenceRequest) -> dict[str, Any]:
    return ft.dependencies_payload(_parse_one(request.sentence))


def syntactic_parser(request: SentenceRequest) -> dict[str, Any]:
    return ft.syntactic_parser_payload(_parse_one(request.sentence))


def split_sentences(request: TextRequest) -> dict[str, Any]:
    return ft.sentences_payload(_parse_one(request.text))


# --- texts-* batch routes -----------------------------------------------------


def texts_tokenize(request: TextsRequest) -> dict[str, Any]:
    return ft.texts_tokens_payload(_parse_many(request.texts))


def texts_lemmatize(request: TextsRequest) -> dict[str, Any]:
    return ft.texts_lemmas_payload(_parse_many(request.texts))


def texts_noun_chunks(request: TextsRequest) -> dict[str, Any]:
    return ft.texts_noun_chunks_payload(_parse_many(request.texts))


def texts_pos_tags(request: TextsRequest) -> dict[str, Any]:
    return ft.texts_pos_tags_payload(_parse_many(request.texts))


def texts_named_entities(request: TextsRequest) -> dict[str, Any]:
    return ft.texts_named_entities_payload(_parse_many(request.texts))


def texts_dependencies(request: TextsRequest) -> dict[str, Any]:
    return ft.texts_dependencies_payload(_parse_many(request.texts))


def texts_split_sentences(request: TextsRequest) -> dict[str, Any]:
    return ft.texts_sentences_payload(_parse_many(request.texts))


_add("/tokenize", tokenize)
# "/lemmatize" bare path: see lemmatize_sentence_payload's docstring above —
# the origin service's native-route /lemmatize handler dispatches to it, so only the prefixed
# VUB-gateway path is registered here.
router.add_api_route("/spacy-api/lemmatize", lemmatize, methods=["POST"])
_add("/noun-chunks", noun_chunks)
_add("/pos", pos)
_add("/named-entities", named_entities)
_add("/dependencies", dependencies)
_add("/syntactic-parser", syntactic_parser)
_add("/split-sentences", split_sentences)

_add("/texts-tokenize", texts_tokenize)
_add("/texts-lemmatize", texts_lemmatize)
_add("/texts-noun-chunks", texts_noun_chunks)
_add("/texts-pos-tags", texts_pos_tags)
_add("/texts-named-entities", texts_named_entities)
_add("/texts-dependencies", texts_dependencies)
_add("/texts-split-sentences", texts_split_sentences)


# --- /embedding-api/* -- not implemented, explicit 501 ----------------------------
#
# There is no implemented reference payload behind this route to be bit-compatible
# WITH (no embedding backend), so it returns a real error via an explicit JSONResponse
# below rather than the router's PenelopeHTMLJSONResponse default (returning
# a Response instance bypasses default_response_class). Revisit this
# alongside a real embeddings capture of the hosted service.

_EMBEDDING_501_BODY = {
    "error": "not_implemented",
    "detail": (
        "Embeddings are not implemented: this service serves syntactic parses only."
    ),
}


def _embedding_501() -> JSONResponse:
    return JSONResponse(status_code=501, content=_EMBEDDING_501_BODY)


for _path in ("/embedding-api/token_embedding", "/embedding-api/token_embeddings"):
    router.add_api_route(_path, _embedding_501, methods=["POST"])
    router.add_api_route(_path.removeprefix("/embedding-api"), _embedding_501, methods=["POST"])
