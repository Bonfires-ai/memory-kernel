"""Saga centroids: every ``:Saga`` stores the L2-normalised mean of its messages' voyage-3 vectors.

The vectors are the events stage's (``embed.TextVoyageCache``, keyed by ``embed.message_texts``), so a Saga lives in
the same vector space as its messages. Stored as extra Saga properties (``centroid_embedding``, ``centroid_model``,
``centroid_messages``) by the write stage; a Saga that grows (a ``walk`` append continuing its chat's open episode) is
recomputed from all its messages, never averaged from its old centroid. Nothing reads them yet.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

#: The vector space of the centroid: the events stage's message vectors (voyage-3, input type document).
CENTROID_MODEL = "voyage-3/document"
#: The Saga properties a centroid adds (``write.SAGA_GROWS``: a grown Saga moves them).
CENTROID_FIELDS = frozenset({"centroid_embedding", "centroid_model", "centroid_messages"})


def saga_members(data: Mapping[str, Any]) -> dict[str, list[str]]:
    """Episode id -> the message ids its ``:Saga`` spans, as ``g1_projection.project`` builds it (``whole``): a reused
    Saga's ``prior`` messages (``data["existing_sagas"]``, first listed per name wins; minus this run's own), then this
    run's messages of that episode in record order. Only episodes with a message in ``data["records"]`` get a Saga."""
    episodes = data["episodes"]
    own: dict[str, list[str]] = {}
    for message_id in dict.fromkeys(r["message_id"] for r in data["records"]):
        own.setdefault(episodes[message_id], []).append(message_id)
    found: dict[str, Mapping[str, Any]] = {}
    for row in data.get("existing_sagas") or ():
        found.setdefault(row["name"], row)
    run_ids = {m for ids in own.values() for m in ids}
    out = {}
    for episode, ids in own.items():
        prior = [m[0] for m in (found.get(episode) or {}).get("prior") or () if m[0] not in run_ids]
        out[episode] = [*dict.fromkeys(prior), *ids]
    return out


def saga_centroids(
    members: Mapping[str, Sequence[str]], texts: Mapping[str, str], cache: Mapping[str, Sequence[float]]
) -> dict[str, dict]:
    """Saga (episode id) -> ``{"centroid_embedding", "centroid_model", "centroid_messages"}``: the mean of its members'
    cached vectors (``cache``: text -> vector; ``texts``: message id -> the text it was embedded as), L2-normalised.
    Pure: no embedding call. ``ValueError`` naming the message ids that have no text or no cached vector."""
    missing = sorted({m for ids in members.values() for m in ids if texts.get(m) not in cache})
    if missing:
        raise ValueError(f"saga centroids: {len(missing)} messages have no cached vector: {missing[:10]}")
    out = {}
    for episode, ids in members.items():
        if not ids:
            raise ValueError(f"saga centroids: episode {episode!r} has no messages")
        vectors = [cache[texts[m]] for m in ids]
        if len({len(v) for v in vectors}) != 1:
            raise ValueError(f"saga centroids: episode {episode!r} mixes vector dimensions")
        mean = [math.fsum(column) / len(vectors) for column in zip(*vectors, strict=True)]
        norm = math.sqrt(math.fsum(x * x for x in mean))
        if not norm:
            raise ValueError(f"saga centroids: episode {episode!r} has a zero mean vector")
        out[episode] = {
            "centroid_embedding": [x / norm for x in mean],
            "centroid_model": CENTROID_MODEL,
            "centroid_messages": len(ids),
        }
    return out
