"""Long-form mode of the embedding segmenter: length-scaled cuts, a message ceiling, cut reasons."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from memory_kernel.ingest.episode_boundary import LongForm, MemoryKernelCentroidTimestampBoundaryService

_START = datetime(2026, 1, 1, tzinfo=UTC)


class _TopicEmbedder:
    """Maps a message text to one of two orthogonal unit vectors by its topic marker."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] if "topic-a" in text else [0.0, 1.0] for text in texts]


def _messages(topics: list[str], *, gap_after: int | None = None) -> list[dict]:
    out = []
    for index, topic in enumerate(topics):
        offset = index + (13 * 3600 if gap_after is not None and index >= gap_after else 0)
        stamp = (_START + timedelta(seconds=offset)).isoformat()
        out.append({"speaker": "Ana", "content": f"{topic} line {index}", "timestamp": stamp})
    return out


def test_long_form_caps_every_segment_and_records_reasons() -> None:
    messages = _messages(["topic-a"] * 200 + ["topic-b"] * 250)
    service = MemoryKernelCentroidTimestampBoundaryService(embedder=_TopicEmbedder(), long_form=LongForm())
    episodes = asyncio.run(service.segment_message_batches([messages]))

    assert all(len(episode) <= 150 for episode in episodes)
    assert sum(len(episode) for episode in episodes) == 450
    flat = [id(message) for episode in episodes for message in episode]
    assert len(flat) == len(set(flat))
    reasons = {cut["reason"] for cut in service.last_cuts}
    assert reasons <= {"shift", "ceiling"}
    shifts = [cut for cut in service.last_cuts if cut["reason"] == "shift"]
    assert shifts and all(abs(cut["at"] - 200) <= 4 for cut in shifts)
    assert all(cut["session_index"] == 0 for cut in service.last_cuts)


def test_short_session_is_one_episode_without_embedding() -> None:
    embedder = _TopicEmbedder()
    service = MemoryKernelCentroidTimestampBoundaryService(embedder=embedder, long_form=LongForm())
    episodes = asyncio.run(service.segment_message_batches([_messages(["topic-a"] * 5)]))

    assert [len(episode) for episode in episodes] == [5]
    assert embedder.calls == 0


def test_default_service_unchanged() -> None:
    sessions = [
        _messages(["topic-a"] * 20 + ["topic-b"] * 20),
        _messages(["topic-a"] * 12 + ["topic-b"] * 12, gap_after=12),
    ]
    service = MemoryKernelCentroidTimestampBoundaryService(embedder=_TopicEmbedder())
    episodes = asyncio.run(service.segment_message_batches(sessions))

    assert [len(episode) for episode in episodes] == [20, 21, 13, 13]
