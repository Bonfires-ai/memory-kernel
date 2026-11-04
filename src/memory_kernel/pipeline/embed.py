"""Voyage embedders for the write / index stages (sanity is a receipt and embeds nothing), with an optional offline
cache (kernel 1.8 archived the jev2 shortlist's writable cache, ``VoyageCache``, with jev2).

Ported from the reference harness:
  * ``VOYAGE_CAP_USD`` / ``voyage_usd`` / ``CappedEmbed``  <- the harness's build module
  * ``VOYAGE_USD_PER_M``                                   <- the harness's conversation-build module

One rebinding differs: the harness built a ``VoyageEmbedder`` from the container's
Voyage key inside each stage; here ``embedder(cfg, input_type)`` uses the explicit ``cfg.voyage_api_key``
and, when ``cfg.embedding_cache`` is set, a ``CachedEmbedder`` in front of it.

``CachedEmbedder`` is new (an offline-replay feature, like the Jev archive). The cache file is JSON::

    {"schema_version": "mk-embedding-cache/v1", "model": "voyage-3", "dim": 1024,
     "provenance": {"source": "...", "stand_in": {"query": "..."}},
     "document": {"<sha256(text utf-8)>": [floats]}, "query": {"<sha256(text utf-8)>": [floats]},
     "document_requests": {"<request_sha(texts)>": [[floats], ...]}}    # optional, per input type

A ``<input_type>_requests`` entry replays one archived embedding REQUEST (the exact text list, in order) with the
vectors it returned, like the Jev archive replays a wire: Voyage is not bitwise deterministic across requests, so the
same text embedded in two requests (e.g. an entity name and an identical fact text) can carry two vectors, and only
the request-level entry reproduces an archived run exactly. A request that is not archived falls back to the per-text
map. A cache hit costs nothing; only texts that miss go to the fallback Voyage embedder, whose ``tokens`` / ``requests`` are
the ones reported, so a fully cached run records ``paid_usd == 0``. With no fallback a miss raises ``EmbeddingMiss``
(before any Neo4j write: ``write_rows`` embeds before it MERGEs). mk code never fills the cache from a graph.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_kernel.pipeline.config import BuildConfig

#: Voyage list price for voyage-3 per million tokens (assumed; the repo carries no constant).
VOYAGE_USD_PER_M = 0.06
VOYAGE_CAP_USD = 0.50  # the write stage's Voyage cap (the harness build module's value): recorded, never a stop
CACHE_SCHEMA = "mk-embedding-cache/v1"
INPUT_TYPES = ("document", "query")


def voyage_usd(tokens: int) -> float:
    return tokens * VOYAGE_USD_PER_M / 1e6


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def request_sha(texts: list[str]) -> str:
    """Key of one embedding request: the sha of its text list, in order."""
    return hashlib.sha256(json.dumps(list(texts), ensure_ascii=False).encode()).hexdigest()


class EmbeddingMiss(RuntimeError):
    """Texts that are not in the embedding cache and no Voyage key to embed them with."""

    def __init__(self, input_type: str, texts: list[str], *, why: str = "no Voyage key") -> None:
        self.input_type = input_type
        self.count = len(texts)
        self.tokens_est = sum(len(t) for t in texts) // 4
        if not texts:
            super().__init__(f"no Voyage key and no embedding cache for {input_type} embeddings")
            return
        super().__init__(
            f"{self.count} {input_type} texts not cached and {why} (about {self.tokens_est} tokens, "
            f"${voyage_usd(self.tokens_est):.6f} at ${VOYAGE_USD_PER_M}/M)"
        )


class CachedEmbedder:
    """``VoyageEmbedder``'s interface over a text -> vector cache; misses go to ``fallback`` (or raise)."""

    def __init__(self, cache: Path, input_type: str, fallback: Any = None) -> None:
        from memory_kernel.graph.kg_writer import KEY_EMBED_DIM, KEY_EMBED_MODEL

        if input_type not in INPUT_TYPES:
            raise ValueError(f"input_type {input_type!r}: must be one of {INPUT_TYPES}")
        doc = json.loads(Path(cache).read_bytes())
        if doc.get("schema_version") != CACHE_SCHEMA:
            raise ValueError(f"embedding cache {cache}: schema {doc.get('schema_version')!r} != {CACHE_SCHEMA!r}")
        if doc.get("model") != KEY_EMBED_MODEL or int(doc.get("dim") or 0) != KEY_EMBED_DIM:
            raise ValueError(
                f"embedding cache {cache}: model/dim {doc.get('model')!r}/{doc.get('dim')!r} != "
                f"{KEY_EMBED_MODEL!r}/{KEY_EMBED_DIM}"
            )
        self.input_type = input_type
        self.model = str(doc["model"])
        self.name = f"voyage/{self.model}"
        self._vectors: dict[str, list[float]] = dict(doc.get(input_type) or {})
        self._requests: dict[str, list[list[float]]] = dict(doc.get(f"{input_type}_requests") or {})
        self.stand_in: str | None = ((doc.get("provenance") or {}).get("stand_in") or {}).get(input_type)
        self.fallback = fallback
        self.hits = 0
        self.misses = 0
        self.request_hits = 0  # requests served whole from ``<input_type>_requests``

    @property
    def tokens(self) -> int:
        return int(self.fallback.tokens) if self.fallback is not None else 0

    @property
    def requests(self) -> int:
        return int(self.fallback.requests) if self.fallback is not None else 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        archived = self._requests.get(request_sha(texts))
        if archived is not None and len(archived) == len(texts):
            self.request_hits += 1
            self.hits += len(texts)
            return [list(v) for v in archived]
        keys = [text_sha(t) for t in texts]
        missing = sorted({t for t, k in zip(texts, keys, strict=True) if k not in self._vectors})
        self.misses += len(missing)  # distinct texts sent (or refused)
        self.hits += sum(1 for k in keys if k in self._vectors)  # texts served from the cache
        if missing:
            if self.fallback is None:
                raise EmbeddingMiss(self.input_type, missing)
            vectors = self.fallback(missing)
            if len(vectors) != len(missing):
                raise ValueError("fallback embedder returned a different number of vectors than texts")
            self._vectors.update({text_sha(t): list(v) for t, v in zip(missing, vectors, strict=True)})
        return [list(self._vectors[k]) for k in keys]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self(list(texts))

    async def create(self, input_data: str) -> list[float]:
        return (await self.create_batch([input_data]))[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        import asyncio

        return await asyncio.to_thread(self, list(input_data_list))


class CappedEmbed:
    """The Voyage embedder with the write stage's cap, checked after each request. Past ``cap`` the spend is recorded
    (``cap_record``, carried by ``record`` into the stage's output, and a ``stage-cap-exceeded`` event) and the stage
    goes on. Then, given the build's ``cfg``, ``--cap-usd`` is the stop: a request whose charge took the build's paid
    total (the cost ledger's at the crossing: the stage's row is written when it ends) plus the spend over
    ``cfg.cap_usd`` raises ``llm.StopSpend`` (Voyage holds nothing before a request, so the check follows it)."""

    def __init__(self, inner: Any, cap: float = VOYAGE_CAP_USD, *, cfg: Any = None) -> None:
        self.inner, self.cap, self.cfg = inner, cap, cfg
        self.cap_record: dict = {}
        self._paid: float | None = None

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def tokens(self) -> int:
        return int(self.inner.tokens)

    @property
    def requests(self) -> int:
        return int(self.inner.requests)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        from memory_kernel.pipeline import llm, provenance

        vectors = self.inner(texts)
        spend = voyage_usd(self.tokens)
        if spend <= self.cap:
            return vectors
        if not self.cap_record:
            self.cap_record = (
                llm.stage_cap_record(self.cfg, "write", self.cap, spend)
                if self.cfg is not None
                else {"stage_cap_usd": str(self.cap), "stage_cap_spend_usd": str(spend), "stage_cap_exceeded": True}
            )
        if self.cfg is not None:
            if self._paid is None:
                self._paid = provenance.ledger_summary(self.cfg.ledger)["paid_usd"]
            if self._paid + spend > self.cfg.cap_usd:
                raise llm.StopSpend(f"unit cap {self.cfg.cap_usd} crossed by the voyage spend {spend:.4f}")
        return vectors


class RecordingEmbed:
    """A dry stage's embedder (kernel 2.0, ``BuildConfig.graph_commit``): ``inner``'s vectors, each recorded under its
    text (``vectors``), so the commit replays them (``ReplayEmbed``) and never embeds. ``save`` writes the record as
    ``{text: vector}`` JSON. ``model`` / ``tokens`` / ``requests`` (and ``record``) are ``inner``'s."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.vectors: dict[str, list[float]] = {}

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def tokens(self) -> int:
        return int(self.inner.tokens)

    @property
    def requests(self) -> int:
        return int(self.inner.requests)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors = self.inner(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedder returned a different number of vectors than texts")
        self.vectors.update({t: [float(x) for x in v] for t, v in zip(texts, vectors, strict=True)})
        return vectors

    def save(self, path: Path) -> None:
        """Write ``vectors`` to ``path`` through a temp file in its directory (a failure leaves no partial file)."""
        import os

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(self.vectors, sort_keys=True))
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


class ReplayEmbed:
    """The commit's embedder (kernel 2.0): the vectors a dry stage recorded (``RecordingEmbed``), looked up by text.
    It makes no network call: a text it lacks is ``EmbeddingMiss``. ``tokens`` and ``requests`` stay 0."""

    tokens = 0
    requests = 0

    def __init__(self, vectors: Mapping[str, list[float]]) -> None:
        from memory_kernel.graph.kg_writer import KEY_EMBED_MODEL

        self.vectors = dict(vectors)
        self.model = KEY_EMBED_MODEL
        self.hits = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        missing = sorted({t for t in texts if t not in self.vectors})
        if missing:
            raise EmbeddingMiss("document", missing, why="the dry pass recorded no vector to replay")
        self.hits += len(texts)
        return [list(self.vectors[t]) for t in texts]


def token_est(text: str) -> int:
    """Voyage tokens of ``text``, estimated as UTF-8 bytes / 4 (at least 1)."""
    return len(text.encode("utf-8", "surrogatepass")) // 4 + 1


def embedder(cfg: BuildConfig, input_type: str) -> Any:
    """The stage's embedder: the cache (with Voyage behind it when a key is set), else Voyage, else EmbeddingMiss."""
    from memory_kernel.graph.kg_writer import VoyageEmbedder

    voyage = VoyageEmbedder(cfg.voyage_api_key, input_type=input_type) if cfg.voyage_api_key else None
    if cfg.embedding_cache is not None:
        return CachedEmbedder(cfg.embedding_cache, input_type, fallback=voyage)
    if voyage is None:
        raise EmbeddingMiss(input_type, [])
    return voyage


def record(emb: Any) -> dict:
    """``{"kind", "hits", "misses", "request_hits", "stand_in"}`` of an embedder (through a ``CappedEmbed``): a declared addition to
    the write / index / sanity outputs, so a cached or stand-in run can never be mistaken for a live one. A
    ``CappedEmbed`` past its cap adds its ``cap_record``. A ``RecordingEmbed`` reports what it wraps; a
    ``ReplayEmbed`` is kind ``replay`` (no Voyage, no cache)."""
    if isinstance(emb, RecordingEmbed):
        return record(emb.inner)
    if isinstance(emb, ReplayEmbed):
        return {"kind": "replay", "hits": emb.hits, "misses": 0, "request_hits": None, "stand_in": None}
    over = emb.cap_record if isinstance(emb, CappedEmbed) else {}
    inner = emb.inner if isinstance(emb, CappedEmbed) else emb
    if isinstance(inner, CachedEmbedder):
        return {
            "kind": "cache",
            "hits": inner.hits,
            "misses": inner.misses,
            "request_hits": inner.request_hits,
            "stand_in": inner.stand_in,
            **over,
        }
    return {"kind": "voyage", "hits": None, "misses": None, "request_hits": None, "stand_in": None, **over}


class TextVoyageCache:
    """create_batch adapter over kg_writer.VoyageEmbedder (voyage-3 document, the stack default) with a cache.

    The cache file is ``{text: vector}`` JSON. The events stage and the boundary stage share one, so a message embedded
    for its boundary is not paid for again. ``vectors`` is the loaded mapping, read-only: a reader that never embeds
    opens the cache with ``voyage_api_key=None`` and reads it. The Saga-centroid reader in ``write`` passes the key and
    embeds the texts the cache lacks."""

    def __init__(self, cache_path, voyage_api_key):
        self.voyage_api_key = voyage_api_key
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.inner = None
        self.tokens = 0

    @property
    def vectors(self):
        """The cached ``{text: vector}`` mapping, read-only."""
        return MappingProxyType(self.cache)

    def _save(self):
        """Write the cache through a temp file in its own directory, then ``os.replace``: a failure part-way leaves
        the old cache whole (a forward append's cache is the bonfire's, ``<bonfire>/chain/embeddings/``)."""
        import os

        tmp = self.cache_path.with_name(f"{self.cache_path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(self.cache))
            os.replace(tmp, self.cache_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def create_batch(self, texts):
        miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if miss:
            from memory_kernel.graph.kg_writer import VoyageEmbedder

            if self.inner is None:
                if not self.voyage_api_key:
                    raise RuntimeError("an uncached text needs a Voyage key (cfg.voyage_api_key)")
                self.inner = VoyageEmbedder(self.voyage_api_key)
            for i in range(0, len(miss), 64):
                for text, vector in zip(miss[i : i + 64], self.inner(miss[i : i + 64]), strict=True):
                    self.cache[text] = vector
            self.tokens = self.inner.tokens
            self._save()
        return [self.cache[t] for t in texts]


def message_texts(rows: Iterable[Mapping]) -> dict[str, str]:
    """Message id -> the exact string the events stage embeds for that message, from ``select`` source rows.

    Events embeds ``boundary_text`` of a message built from the capture records: the speaker of its first record and
    the text of every record joined, stripped. Capture makes one unit per sentence of each non-blank part (the
    message text, then each attachment caption), and the units of a part tile its text, so the join is the non-blank
    parts concatenated. A row with no text at all has no records, so events never embeds it; it is listed here
    as its speaker alone."""
    from memory_kernel.ingest.episode_boundary import boundary_text

    out = {}
    for row in rows:
        parts = [row["text"], *(a["caption"] for a in row.get("attachments") or ())]
        text = "".join(p for p in parts if p.strip()).strip()
        out[row["message_id"]] = boundary_text({"speaker": row.get("speaker"), "text": text})
    return out
