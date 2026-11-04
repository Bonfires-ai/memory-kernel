from memory_kernel.construction.mined_affordance_contract import content_sha256
from memory_kernel.graph.g1_projection import message_rank, order_key
from memory_kernel.pipeline.stages._g1_build import check_episode_map


def test_rank_follows_records_not_id_digits():
    records = [{"message_id": "room-a:7"}, {"message_id": "room-a:7"}, {"message_id": "room-b:3"}]
    rank = message_rank(records)
    assert sorted(["room-b:3", "room-a:7"], key=order_key(rank)) == ["room-a:7", "room-b:3"]


def test_numbered_ids_keep_their_order():
    records = [{"message_id": f"T1:{i}"} for i in (1, 2, 10)]
    assert sorted(["T1:10", "T1:2", "T1:1"], key=order_key(message_rank(records))) == ["T1:1", "T1:2", "T1:10"]


def test_episode_contiguity_uses_source_order():
    # digits order would be b:3 (e2), a:7 (e1), a:9 (e2): e2 is not contiguous; source order is e1, e2, e2
    mapping = {"room-a:7": "e1", "room-b:3": "e2", "room-a:9": "e2"}
    body = {"episode_by_message": mapping}
    doc = {**body, "content_sha256": content_sha256(body)}
    rank = {"room-a:7": 0, "room-b:3": 1, "room-a:9": 2}
    assert check_episode_map(doc, set(mapping), messages=3, episodes=2, rank=rank) == mapping
