"""Saga centroids: the L2-normalised mean of a Saga's cached message vectors, recomputed whole when it grows."""

from __future__ import annotations

import pytest

from memory_kernel.pipeline.saga_vectors import CENTROID_MODEL, saga_centroids


def test_saga_centroid_is_the_normalised_mean_and_grows_whole():
    texts = {"m1": "Ana: the kiln is warm", "m2": "Ben: glaze it tonight", "m3": "Ana: fired at dawn"}
    cache = {texts["m1"]: [1.0, 0.0], texts["m2"]: [0.0, 1.0], texts["m3"]: [1.0, 1.0]}

    first = saga_centroids({"e1": ["m1", "m2"]}, texts, cache)["e1"]
    assert [round(x, 4) for x in first["centroid_embedding"]] == [0.7071, 0.7071]
    assert first["centroid_messages"] == 2 and first["centroid_model"] == CENTROID_MODEL == "voyage-3/document"

    # a growth: the reused Saga's prior messages (m1, m2) and this run's (m3), recomputed whole
    grown = saga_centroids({"e1": ["m1", "m2", "m3"]}, texts, cache)["e1"]
    mean = [2 / 3, 2 / 3]
    norm = sum(x * x for x in mean) ** 0.5
    assert grown["centroid_embedding"] == pytest.approx([x / norm for x in mean])
    assert grown["centroid_messages"] == 3

    del cache[texts["m3"]]
    with pytest.raises(ValueError, match="m3"):
        saga_centroids({"e1": ["m1", "m2", "m3"]}, texts, cache)
