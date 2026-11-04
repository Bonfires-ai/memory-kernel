"""Kernel-owned episode boundary detection for stack/session ingest."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from inspect import isawaitable
from typing import Any

from memory_kernel.config.flags import env_flag_unless_disabled as _env_bool
from memory_kernel.config.flags import env_float as _env_float
from memory_kernel.config.flags import env_int as _env_int

from .retrieval_index import embed_texts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LongForm:
    """Opt-in settings for cutting a long talk or document instead of a chat session.

    ``target`` is the intended episode size in messages: a session of ``n`` messages may be cut into at most
    ``ceil(n / target)`` episodes. ``ceiling`` is the hard maximum — any episode longer than it is split further.
    """

    target: int = 60
    ceiling: int = 150


class MemoryKernelCentroidTimestampBoundaryService:
    """Fast chronological boundary detector using timestamps and embedding centroids.

    Input stack/session batches remain hard outer boundaries; this service only
    splits inside each batch.
    """

    def __init__(
        self,
        *,
        embedder: Any,
        max_centroids: int | None = None,
        min_segment_messages: int | None = None,
        time_gap_hours: float | None = None,
        improvement_threshold: float | None = None,
        centroid_cosine_threshold: float | None = None,
        shift_window: int | None = None,
        shift_cosine_threshold: float | None = None,
        centroid_improve_radius: int | None = None,
        max_segments_per_session: int | None = None,
        smart_mask: bool | None = None,
        long_form: LongForm | None = None,
    ) -> None:
        """Resolve every detector threshold, preferring an explicit argument over its environment default.

        ``embedder`` is a live, caller-owned client awaited during segmentation; the service holds it but does not
        own its lifecycle. Every other argument is optional and falls back to a ``MK_BOUNDARY_*`` variable
        read once, here — changing the environment after construction has no effect on this instance.

        The numeric settings are clamped to sane floors (and ``max_centroids`` to a ceiling of 3), so a
        misconfigured value degrades the segmentation rather than producing empty or single-message episodes.
        ``time_gap_hours`` is stored as seconds.

        With ``long_form`` the per-session segment cap scales with length (see :class:`LongForm`), no episode may
        exceed the ceiling, the centroid fallback is skipped (where the shift pass finds nothing, only the ceiling
        split cuts), and the smart mask is forced off so each message lands in exactly one episode. Without
        it the behaviour is the chat-tuned default. ``last_cuts`` describes the cuts of the latest segmentation call.
        """
        self._embedder = embedder
        self._long_form = long_form
        self.last_cuts: list[dict[str, Any]] = []
        self._vector_cache: dict[int, list[float]] = {}
        self._max_centroids = max(1, min(3, max_centroids or _env_int("MK_BOUNDARY_MAX_CENTROIDS", 3)))
        self._min_segment_messages = max(2, min_segment_messages or _env_int("MK_BOUNDARY_MIN_SEGMENT_MESSAGES", 4))
        self._shift_window = max(1, shift_window or _env_int("MK_BOUNDARY_SHIFT_WINDOW", 3))
        self._shift_cosine_threshold = (
            shift_cosine_threshold
            if shift_cosine_threshold is not None
            else _env_float("MK_BOUNDARY_SHIFT_COSINE", 0.78)
        )
        self._centroid_improve_radius = max(
            0,
            centroid_improve_radius or _env_int("MK_BOUNDARY_CENTROID_IMPROVE_RADIUS", 2),
        )
        self._max_segments_per_session = max(
            1,
            max_segments_per_session or _env_int("MK_BOUNDARY_MAX_SEGMENTS_PER_SESSION", 3),
        )
        self._time_gap_seconds = 3600.0 * (
            time_gap_hours if time_gap_hours is not None else _env_float("MK_BOUNDARY_TIME_GAP_HOURS", 12.0)
        )
        self._improvement_threshold = (
            improvement_threshold
            if improvement_threshold is not None
            else _env_float("MK_BOUNDARY_CENTROID_IMPROVEMENT", 0.28)
        )
        self._centroid_cosine_threshold = (
            centroid_cosine_threshold
            if centroid_cosine_threshold is not None
            else _env_float("MK_BOUNDARY_CENTROID_MAX_COSINE", 0.88)
        )
        self._smart_mask = smart_mask if smart_mask is not None else _env_bool("MK_BOUNDARY_SMART_MASK", True)
        if long_form is not None:
            self._smart_mask = False

    async def segment_message_batches(
        self,
        message_batches: list[list[Any]],
        *,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> list[list[Any]]:
        """Cut each input batch into episodes, returning the episodes of all batches in input order.

        A batch is a hard outer boundary — no episode ever spans two of them. Inside a batch the cuts are tried
        in order: time gaps first, then a topic-shift pass over each time segment, and only where that finds
        nothing does the more expensive centroid search run. Image sidecars are held out of the boundary
        decision and reattached to their source message's segment afterwards.

        With smart masking on, each episode after the first is prefixed with the last message of the previous
        one, so a message can appear in two consecutive episodes.

        ``last_cuts`` is replaced by each call with one record per cut: ``session_index`` (the batch), ``at`` (the
        index of the first message of the new episode among the batch's boundary messages, i.e. excluding image
        sidecars) and ``reason`` (``time_gap``, ``shift``, ``centroid`` or ``ceiling``).

        Progress events are emitted at start, after each batch and at the end when a callback is supplied.
        """
        total_messages = sum(len(batch) for batch in message_batches)
        await emit_boundary_progress(
            progress_callback,
            stage="boundary_detection",
            status="start",
            mode="centroid_timestamp",
            messages=total_messages,
            input_batches=len(message_batches),
        )
        episodes: list[list[Any]] = []
        self.last_cuts = []
        self._vector_cache = {}
        time_gap_splits = 0
        centroid_splits = 0
        shift_candidates = 0
        local_shift_splits = 0
        for batch_index, batch in enumerate(message_batches, 1):
            boundary_messages, sidecars = _split_boundary_sidecars(batch)
            segment_input = boundary_messages or batch
            primary_segments: list[list[Any]] = []
            session_cuts: list[dict[str, Any]] = []
            session_offset = 0
            for time_segment in _split_on_time_gaps(segment_input, self._time_gap_seconds):
                if len(time_segment) != len(segment_input):
                    time_gap_splits += 1
                if session_offset:
                    session_cuts.append({"session_index": batch_index - 1, "at": session_offset, "reason": "time_gap"})
                cap = self._long_form_cap(len(time_segment))
                segmented, segment_stats = await self._split_by_topic_shifts(time_segment, segment_cap=cap)
                shift_candidates += int(segment_stats.get("shift_candidates", 0))
                local_shift_splits += max(0, len(segmented) - 1)
                reason = "shift"
                # long-form skips the centroid search (exhaustive, hours on a long talk): the ceiling split bounds it
                if len(segmented) == 1 and self._long_form is None:
                    segmented = await self._split_by_centroids(time_segment)
                    centroid_splits += max(0, len(segmented) - 1)
                    reason = "centroid"
                position = session_offset
                for piece in segmented[:-1]:
                    position += len(piece)
                    session_cuts.append({"session_index": batch_index - 1, "at": position, "reason": reason})
                if self._long_form is not None:
                    segmented = await self._split_over_ceiling(segmented, session_offset, batch_index - 1, session_cuts)
                primary_segments.extend(segmented)
                session_offset += len(time_segment)
            self.last_cuts.extend(sorted(session_cuts, key=lambda cut: cut["at"]))
            episodes.extend(_reattach_boundary_sidecars(primary_segments, sidecars, batch))
            await emit_boundary_progress(
                progress_callback,
                stage="boundary_detection",
                status="checked",
                mode="centroid_timestamp",
                index=batch_index,
                total=len(message_batches),
                episodes=len(episodes),
            )

        if self._smart_mask:
            episodes = _add_previous_message_context(episodes)
        await emit_boundary_progress(
            progress_callback,
            stage="boundary_detection",
            status="done",
            mode="centroid_timestamp",
            messages=total_messages,
            episodes=len(episodes),
            time_gap_splits=time_gap_splits,
            shift_candidates=shift_candidates,
            local_shift_splits=local_shift_splits,
            centroid_splits=centroid_splits,
        )
        return episodes

    def _long_form_cap(self, message_count: int) -> int | None:
        """The length-scaled segment cap for ``message_count`` messages, or ``None`` outside long-form mode."""
        if self._long_form is None:
            return None
        return max(1, math.ceil(message_count / max(1, self._long_form.target)))

    async def _vectors_for(self, messages: list[Any]) -> list[list[float]]:
        """Unit-length embeddings of ``messages``; in long-form mode each message is embedded once per call."""
        if self._long_form is None:
            vectors = await embed_texts(self._embedder, [boundary_text(message) for message in messages])
            return [_normalize_vector(vector) for vector in vectors]
        missing = [message for message in messages if id(message) not in self._vector_cache]
        if missing:
            fresh = await embed_texts(self._embedder, [boundary_text(message) for message in missing])
            for message, vector in zip(missing, fresh, strict=True):
                self._vector_cache[id(message)] = _normalize_vector(vector)
        return [self._vector_cache[id(message)] for message in messages]

    async def _split_over_ceiling(
        self,
        segments: list[list[Any]],
        offset: int,
        session_index: int,
        cuts: list[dict[str, Any]],
    ) -> list[list[Any]]:
        """Split every segment longer than the long-form ceiling until all fit, recording each cut as ``ceiling``.

        A segment is cut at its strongest local shift score (when that score is positive), else at its midpoint,
        then each half is checked again. ``offset`` is the session index of the first message of ``segments``.
        """
        assert self._long_form is not None
        ceiling = max(2, self._long_form.ceiling)
        out: list[list[Any]] = []
        position = offset
        for segment in segments:
            pending: list[tuple[int, list[Any]]] = [(position, segment)]
            while pending:
                start, piece = pending.pop(0)
                if len(piece) <= ceiling:
                    out.append(piece)
                    continue
                vectors = await self._vectors_for(piece)
                scored = _local_shift_scores(
                    vectors, window=self._shift_window, min_segment_messages=self._min_segment_messages
                )
                best = max(scored, key=lambda item: item[1], default=None)
                at = best[0] if best is not None and best[1] > 0 else len(piece) // 2
                cuts.append({"session_index": session_index, "at": start + at, "reason": "ceiling"})
                pending[:0] = [(start, piece[:at]), (start + at, piece[at:])]
            position += len(segment)
        return out

    async def _split_by_topic_shifts(
        self, messages: list[Any], *, segment_cap: int | None = None
    ) -> tuple[list[list[Any]], dict[str, int]]:
        """Cut where the conversation turns, returning the segments and a count of candidate turn points.

        A position qualifies when the text before and after it is dissimilar enough and it is a local peak, so a
        gradual drift produces one cut rather than a run of them. Candidates are taken strongest first, never
        closer together than the minimum segment size, and never more than the per-session segment cap allows.
        Chosen cuts are then nudged within a small radius to whichever offset minimises total within-segment
        spread.

        Returns ``([messages], …)`` unchanged — and embeds nothing — when the input is too short to split or the
        cap leaves no room; the reported candidate count is 0 in that case. Requires the embedder. ``segment_cap``
        replaces the configured per-session segment cap (long-form mode passes a length-scaled one).
        """
        if len(messages) < self._min_segment_messages * 2:
            return [messages], {"shift_candidates": 0}
        vectors = await self._vectors_for(messages)
        cap = self._max_segments_per_session if segment_cap is None else segment_cap
        max_boundaries = max(0, min(cap, len(messages) // self._min_segment_messages) - 1)
        if max_boundaries <= 0:
            return [messages], {"shift_candidates": 0}
        scored = _local_shift_scores(
            vectors,
            window=self._shift_window,
            min_segment_messages=self._min_segment_messages,
        )
        candidates = [
            (index, score)
            for index, score in scored
            if score >= 1.0 - self._shift_cosine_threshold and _is_local_shift_peak(scored, index, score)
        ]
        selected: list[int] = []
        for index, _score in sorted(candidates, key=lambda item: item[1], reverse=True):
            if len(selected) >= max_boundaries:
                break
            if all(abs(index - existing) >= self._min_segment_messages for existing in selected):
                selected.append(index)
        if not selected:
            return [messages], {"shift_candidates": len(candidates)}
        splits = [0, *sorted(selected), len(messages)]
        splits = _improve_splits_by_centroid_cost(
            vectors,
            splits,
            radius=self._centroid_improve_radius,
            min_segment_messages=self._min_segment_messages,
        )
        return [messages[start:end] for start, end in zip(splits, splits[1:], strict=False)], {
            "shift_candidates": len(candidates)
        }

    async def _split_by_centroids(self, messages: list[Any]) -> list[list[Any]]:
        """Search for the best contiguous k-way split of ``messages``, k from 2 up to the centroid cap.

        A split is accepted only if it cuts within-segment spread by at least the improvement threshold against
        the unsplit baseline AND leaves no two adjacent segments whose centroids are still too similar — the
        second gate is what stops a numerically better split that does not correspond to a real topic change.
        The last k that passes both wins, so larger k is preferred among acceptable splits.

        Returns ``[messages]`` unchanged when the input is too short, when the baseline spread is already zero,
        or when no k passes. Requires the embedder and is the more expensive of the two content-based passes.
        """
        if len(messages) < self._min_segment_messages * 2:
            return [messages]
        vectors = await self._vectors_for(messages)
        max_k = min(self._max_centroids, len(messages) // self._min_segment_messages)
        baseline = _segmentation_cost(vectors, [0, len(vectors)])
        if baseline <= 0:
            return [messages]

        best_splits: list[int] | None = None
        for k in range(2, max_k + 1):
            splits = _best_contiguous_splits(vectors, k, self._min_segment_messages)
            if not splits:
                continue
            cost = _segmentation_cost(vectors, splits)
            improvement = (baseline - cost) / baseline
            if improvement < self._improvement_threshold:
                continue
            centroids = [_centroid(vectors[start:end]) for start, end in zip(splits, splits[1:], strict=False)]
            adjacent_max = max((_cosine(a, b) for a, b in zip(centroids, centroids[1:], strict=False)), default=1.0)
            if adjacent_max <= self._centroid_cosine_threshold:
                best_splits = splits
        if not best_splits:
            return [messages]
        return [messages[start:end] for start, end in zip(best_splits, best_splits[1:], strict=False)]


def parse_timestamp(value: Any) -> datetime | None:
    """Coerce a message timestamp to a ``datetime``, or ``None`` when there is nothing usable.

    ``datetime`` values pass through; strings are parsed as ISO-8601 with a trailing ``Z`` accepted for UTC.

    Returning ``None`` for an unparseable stamp is the designed behaviour, not a fallback: timestamps arrive from
    external message sources and are parsed across a whole conversation, so one malformed value must not abort
    the batch. The bad value and the parse error are logged at warning level. Every other type, and the empty
    string, also yield ``None`` — silently, with no log line.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            # KEEP: per-item isolation — external message
            # timestamp, parsed across many messages in boundary detection;
            # one malformed timestamp must not abort the whole conversation.
            logger.warning("Unparseable message timestamp %r: %s", value, exc)
            return None
    return None


async def emit_boundary_progress(callback: Callable[[dict[str, Any]], Any] | None, **event: Any) -> None:
    """Deliver one boundary-detection progress event to an optional callback.

    The keyword arguments are collected into a single dict and passed as the callback's one argument. Sync and
    async callbacks are both accepted. A ``None`` callback is a no-op, so instrumentation is optional at every
    call site. Exceptions raised by the callback propagate and will abort segmentation.
    """
    if callback is None:
        return
    result = callback(event)
    if isawaitable(result):
        await result


def _split_on_time_gaps(messages: list[Any], gap_seconds: float) -> list[list[Any]]:
    """Cut a message run wherever consecutive messages are at least ``gap_seconds`` apart.

    Message order is preserved and every message lands in exactly one segment. A cut is only made once the
    current segment has at least two messages, so this never emits a lone message before a gap. A pair whose
    timestamps do not both parse is treated as no gap — an unstamped message never forces a boundary — and a
    non-positive ``gap_seconds`` disables splitting entirely. An empty input gives an empty list, not ``[[]]``.
    """
    if not messages:
        return []
    out: list[list[Any]] = []
    current = [messages[0]]
    for message in messages[1:]:
        previous_time = parse_timestamp(_message_timestamp(current[-1]))
        next_time = parse_timestamp(_message_timestamp(message))
        gap = (next_time - previous_time).total_seconds() if previous_time and next_time else 0.0
        if gap_seconds > 0 and gap >= gap_seconds and len(current) >= 2:
            out.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        out.append(current)
    return out


def _add_previous_message_context(episodes: list[list[Any]]) -> list[list[Any]]:
    """Prefix each episode with the last message of the one before it, so a cut episode keeps its lead-in.

    The overlap is deliberate: that message is then present in two episodes, and downstream extraction may
    attest the same content twice. Skipped when the previous episode is short (five messages or fewer) or when
    the two already share that message. The first episode is never modified, and the input lists are not mutated.
    """
    if not episodes:
        return []
    out = [episodes[0]]
    for episode in episodes[1:]:
        previous = out[-1]
        if previous and len(previous) > 5 and episode and previous[-1] is not episode[0]:
            out.append([previous[-1], *episode])
        else:
            out.append(episode)
    return out


def _split_boundary_sidecars(messages: list[Any]) -> tuple[list[Any], list[Any]]:
    """Separate image-context sidecars from the messages boundaries are decided on, both in original order.

    Sidecars are derived from another message rather than being turns of their own, so letting them vote on
    boundaries would split on the picture rather than the conversation. They are reattached afterwards.
    """
    boundary_messages: list[Any] = []
    sidecars: list[Any] = []
    for message in messages:
        if str(_message_metadata(message).get("source_kind") or "") == "image_context":
            sidecars.append(message)
        else:
            boundary_messages.append(message)
    return boundary_messages, sidecars


def _reattach_boundary_sidecars(
    primary_segments: list[list[Any]],
    sidecars: list[Any],
    original_messages: list[Any],
) -> list[list[Any]]:
    """Put each held-out sidecar back into the segment holding the message it came from.

    A sidecar naming a source message that landed in a segment follows it there; one whose source cannot be
    resolved falls back to the timestamp-nearest segment, and to the first segment when it has no usable
    timestamp either — a sidecar is never dropped. Each segment is finally re-sorted into the original message
    order, so reattachment cannot leave a segment out of sequence.

    Returns ``primary_segments`` as given when there is nothing to reattach; otherwise the segment lists are
    fresh copies.
    """
    if not sidecars or not primary_segments:
        return primary_segments
    original_order = {id(message): index for index, message in enumerate(original_messages)}
    segment_by_message_id: dict[str, int] = {}
    for segment_index, segment in enumerate(primary_segments):
        for message in segment:
            metadata = _message_metadata(message)
            message_id = str(metadata.get("message_id") or metadata.get("id") or "")
            if message_id:
                segment_by_message_id[message_id] = segment_index

    out = [list(segment) for segment in primary_segments]
    for sidecar in sidecars:
        source_id = str(_message_metadata(sidecar).get("source_message_id") or "")
        segment_index = segment_by_message_id.get(source_id)
        if segment_index is None:
            segment_index = _nearest_segment_index(sidecar, primary_segments)
        out[segment_index].append(sidecar)
    for segment in out:
        segment.sort(key=lambda message: original_order.get(id(message), len(original_order)))
    return out


def _nearest_segment_index(message: Any, segments: list[list[Any]]) -> int:
    """Index of the segment holding the message closest in time to ``message``.

    Distance to a segment is to its nearest member, not to its span, so an outlying member can pull a segment
    close. Segments with no parseable timestamp are skipped. Falls back to 0 — never raises, never returns -1 —
    when ``message`` has no usable timestamp or no segment has one, so the caller always has somewhere to put it.
    """
    message_time = parse_timestamp(_message_timestamp(message))
    if message_time is None:
        return 0
    best_index = 0
    best_seconds: float | None = None
    for index, segment in enumerate(segments):
        times = [parse_timestamp(_message_timestamp(item)) for item in segment]
        times = [time for time in times if time is not None]
        if not times:
            continue
        seconds = min(abs((time - message_time).total_seconds()) for time in times)
        if best_seconds is None or seconds < best_seconds:
            best_index = index
            best_seconds = seconds
    return best_index


def boundary_text(message: Any) -> str:
    """The text a message is embedded as for boundary detection: ``"speaker: content"``, or content alone.

    The speaker is included so a change of who is talking contributes to the similarity signal. ``content`` is
    read first and ``text`` accepted as an alternative. Returns ``""`` when the message carries neither.
    """
    speaker = str(_message_value(message, "speaker", "") or "").strip()
    content = str(_message_value(message, "content", _message_value(message, "text", "")) or "").strip()
    return f"{speaker}: {content}" if speaker else content


_boundary_text = boundary_text


def _message_metadata(message: Any) -> Mapping[str, Any]:
    """A message's metadata mapping, or an empty mapping when it has none or it is not mapping-shaped.

    Callers can therefore ``.get`` unconditionally.
    """
    metadata = _message_value(message, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _message_timestamp(message: Any) -> Any:
    """A message's raw ``timestamp`` in whatever form it was stored, or ``None``; parsing is the caller's job."""
    return _message_value(message, "timestamp", None)


def _message_value(message: Any, name: str, default: Any = None) -> Any:
    """Read one field from a message whether it is a mapping or an object, falling back to ``default``.

    This is the single accessor that lets the whole module take dicts and message objects interchangeably.
    """
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def _normalize_vector(vector: list[float]) -> list[float]:
    """Scale to unit length; a zero vector comes back as zeros of the same length rather than raising.

    Every vector in this module is normalized on arrival, which is what lets :func:`_cosine` be a plain dot
    product.
    """
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 0:
        return [0.0 for _value in vector]
    return [float(value) / norm for value in vector]


def _centroid(vectors: list[list[float]]) -> list[float]:
    """The unit-length mean direction of ``vectors``, sized by the first one.

    A longer vector is truncated to that width and a shorter one contributes only the components it has, rather
    than raising. Returns an empty list for an empty input.
    """
    if not vectors:
        return []
    width = len(vectors[0])
    values = [0.0] * width
    for vector in vectors:
        for index, value in enumerate(vector[:width]):
            values[index] += float(value)
    return _normalize_vector([value / len(vectors) for value in values])


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two already-normalized vectors, computed as their dot product.

    Returns 0.0 — treating the pair as unrelated rather than raising — when either side is empty or the lengths
    differ. Inputs that are not unit-length silently give a value outside ``[-1, 1]``.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(float(left) * float(right) for left, right in zip(a, b, strict=False))


def _segmentation_cost(vectors: list[list[float]], splits: list[int]) -> float:
    """Total within-segment spread for a candidate segmentation — lower is a tighter fit.

    ``splits`` are segment start offsets with the total length appended, so ``[0, 5, 12]`` describes two
    segments. Each member contributes its distance from its own segment's centroid, floored at zero. The value
    grows with the number of messages, so it is only comparable between segmentations of the same input.
    """
    cost = 0.0
    for start, end in zip(splits, splits[1:], strict=False):
        segment = vectors[start:end]
        center = _centroid(segment)
        cost += sum(max(0.0, 1.0 - _cosine(vector, center)) for vector in segment)
    return cost


def _local_shift_scores(
    vectors: list[list[float]],
    *,
    window: int,
    min_segment_messages: int,
) -> list[tuple[int, float]]:
    """Score every legal cut position by how unlike the ``window`` messages before it are to those after.

    Scores are ``1 - cosine`` of the two side centroids, so higher means a sharper turn. Positions closer than
    ``min_segment_messages`` to either end are not scored at all, so the result is a contiguous run of interior
    offsets in ascending order and may be empty for a short input. The window is clipped at the ends rather than
    the position being skipped.
    """
    scores: list[tuple[int, float]] = []
    for index in range(min_segment_messages, len(vectors) - min_segment_messages + 1):
        left_start = max(0, index - window)
        right_end = min(len(vectors), index + window)
        left = vectors[left_start:index]
        right = vectors[index:right_end]
        if not left or not right:
            continue
        score = 1.0 - _cosine(_centroid(left), _centroid(right))
        scores.append((index, score))
    return scores


def _is_local_shift_peak(scored: list[tuple[int, float]], index: int, score: float) -> bool:
    """Whether ``score`` at ``index`` is at least as high as its immediate neighbours' scores.

    This is what collapses a gradual drift — where many adjacent positions clear the threshold — into a single
    cut. A position with no scored neighbour on a side counts as a peak on that side, so the ends of the scored
    range always qualify. Ties count as peaks, so a flat run can yield several.
    """
    previous_score = next(
        (candidate_score for candidate_index, candidate_score in scored if candidate_index == index - 1), -1.0
    )
    next_score = next(
        (candidate_score for candidate_index, candidate_score in scored if candidate_index == index + 1), -1.0
    )
    return score >= previous_score and score >= next_score


def _improve_splits_by_centroid_cost(
    vectors: list[list[float]],
    splits: list[int],
    *,
    radius: int,
    min_segment_messages: int,
) -> list[int]:
    """Nudge each interior split up to ``radius`` positions to wherever within-segment spread is lowest.

    The turn-detection pass finds roughly where the conversation changes; this settles exactly where to cut.
    Splits are refined left to right and each one sees its predecessors' new positions, so the result depends on
    that order and is not a global optimum. Candidate positions are constrained to keep every segment at least
    ``min_segment_messages`` long, and a split is left alone when no candidate beats it or no legal window
    exists. Returns ``splits`` unchanged when there is nothing to move.
    """
    if radius <= 0 or len(splits) <= 2:
        return splits
    improved = list(splits)
    for split_index in range(1, len(improved) - 1):
        left_bound = improved[split_index - 1] + min_segment_messages
        right_bound = improved[split_index + 1] - min_segment_messages
        current = improved[split_index]
        start = max(left_bound, current - radius)
        end = min(right_bound, current + radius)
        if start > end:
            continue
        best_split = current
        best_cost = _segmentation_cost(vectors, improved)
        for candidate in range(start, end + 1):
            candidate_splits = [*improved]
            candidate_splits[split_index] = candidate
            cost = _segmentation_cost(vectors, candidate_splits)
            if cost < best_cost:
                best_split = candidate
                best_cost = cost
        improved[split_index] = best_split
    return improved


def _best_contiguous_splits(vectors: list[list[float]], k: int, min_size: int) -> list[int] | None:
    """The lowest-cost way to cut ``vectors`` into exactly ``k`` contiguous runs of at least ``min_size`` each.

    Returns split offsets starting at 0 and ending at the total length, or ``None`` when ``k`` is under 2 or the
    input is too short to hold ``k`` segments of the minimum size.

    The search is exhaustive over every legal placement, so the answer is optimal but the work grows sharply
    with ``k`` and length — this is why the caller caps ``k`` and only reaches this pass when cheaper ones found
    nothing.
    """
    n = len(vectors)
    if k <= 1 or n < k * min_size:
        return None
    best: tuple[float, list[int]] | None = None

    def visit(start: int, remaining: int, splits: list[int]) -> None:
        """Recurse over placements of the ``remaining`` segments from ``start``, keeping the cheapest in ``best``."""
        nonlocal best
        if remaining == 1:
            if n - start < min_size:
                return
            candidate = [*splits, n]
            cost = _segmentation_cost(vectors, candidate)
            if best is None or cost < best[0]:
                best = (cost, candidate)
            return
        max_next = n - (remaining - 1) * min_size
        for next_start in range(start + min_size, max_next + 1):
            visit(next_start, remaining - 1, [*splits, next_start])

    visit(0, k, [0])
    return best[1] if best is not None else None


__all__ = [
    "LongForm",
    "MemoryKernelCentroidTimestampBoundaryService",
    "boundary_text",
    "emit_boundary_progress",
    "parse_timestamp",
]
