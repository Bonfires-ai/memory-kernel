"""Input checks, census and report helpers of the G1 event-graph build (``stages.write``, ``_g1_build_u2``).

Ported from a reference-harness module (unreached top-level definitions pruned). Differences: GROUP / ROOT / P / INPUTS /
EXPECTED_MESSAGES / EXPECTED_EPISODES / JEV1_MESSAGES are not module constants any more (`group`, `inputs` and the
expected counts are arguments of `stages.write` / `_g1_build_u2`); `CREATED_AT` stays the reference value.

This module neither projects nor connects to Neo4j: ``stages.write`` projects through ``_g1_build_u2.project_u2``,
writes the rows and reads them back. The helpers here check the build's inputs and summarise its rows.

Input checks: ``check_episode_map`` (the pinned boundary receipt: its seal, the expected message and episode counts,
one episode per message of the graph, and every episode a contiguous run in message order) and ``check_eventuality``
(the per-predicate CONTEXT eventuality of ``_eventuality_ctx``: sealed, every predicate of the graph labelled or
excluded as a nominal modifier, and the labels reproduce the artifact's own distribution).

Reports: ``envelope_census`` (act and mood over the ``:Episodic`` envelopes; asserted and modality over the content
rows), ``shape`` / ``graph_report`` / ``edge_kind`` (row counts and edge families), and ``row_set_hash`` /
``content_hash`` / ``sha256_file``: ``stages.write`` records these hashes with every input, so the row set can be
regenerated and checked.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from memory_kernel.construction.mined_affordance_contract import content_sha256
from memory_kernel.graph.g1_projection import order_key
from memory_kernel.graph.referents import split_discriminated

#: Identity value of every row's ``created_at``: the reference build's, never the wall clock.
CREATED_AT = "2026-10-07T00:00:00Z"


def _sealed(doc: dict) -> bool:
    return doc.get("content_sha256") == content_sha256({k: v for k, v in doc.items() if k != "content_sha256"})


def check_episode_map(
    doc: dict,
    graph_messages: set[str],
    *,
    messages: int,
    episodes: int,
    rank: Mapping[str, int],
    conversation_of: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The pinned boundary receipt, checked: self-consistent seal, exact counts, one episode per message of
    the graph, and every episode a contiguous run in source message order.

    ``conversation_of`` (message id -> conversation id) checks contiguity within each conversation's subsequence of
    source order instead, since the boundary walks each conversation apart; an episode spanning two conversations is
    refused. None: every message is in one conversation."""
    if not _sealed(doc):
        raise ValueError("episode map content_sha256 is not self-consistent")
    mapping: dict[str, str] = doc["episode_by_message"]
    if len(mapping) != messages:
        raise ValueError(f"episode map has {len(mapping)} messages, expected {messages}")
    if len(set(mapping.values())) != episodes:
        raise ValueError(f"episode map has {len(set(mapping.values()))} episodes, expected {episodes}")
    if set(mapping) != graph_messages:
        raise ValueError("episode map messages differ from the graph's messages")
    runs: dict[str, list[str]] = {}  # conversation -> its episodes in the order they start
    home: dict[str, str] = {}  # episode -> its conversation
    for message_id in sorted(mapping, key=order_key(rank)):
        episode = mapping[message_id]
        conversation = conversation_of[message_id] if conversation_of is not None else ""
        if home.setdefault(episode, conversation) != conversation:
            raise ValueError(f"episode {episode} spans conversations {home[episode]!r} and {conversation!r}")
        seen = runs.setdefault(conversation, [])
        if seen and seen[-1] != episode and episode in seen:
            raise ValueError(f"episode {episode} is not contiguous in message order (resumes at {message_id})")
        if not seen or seen[-1] != episode:
            seen.append(episode)
    return mapping


def check_eventuality(keys: set[str], doc: dict) -> tuple[dict[str, dict], set[str]]:
    """({key: {label, p}}, excluded keys) from the sealed context-eventuality artifact.

    Every predicate key of the graph must be in the artifact (labelled, or excluded as a nominal modifier), and the
    labels must reproduce the artifact's own distribution."""
    if not _sealed(doc):
        raise ValueError("eventuality content_sha256 is not self-consistent")
    preds = doc["predicates"]
    if set(preds) != keys:
        raise ValueError(f"eventuality predicates differ from the graph's ({len(set(preds) ^ keys)} keys)")
    excluded = {k for k, v in preds.items() if v["excluded"]}
    labels = {
        k: {
            "label": v["label"],
            "p": v["p_yes"],
            "fallback": bool(v.get("fallback_prior")),
            "anchor_index": v.get("anchor_index"),
        }
        for k, v in preds.items()
        if not v["excluded"]
    }
    got = dict(Counter(v["label"] for v in labels.values()))
    if got != doc["distribution_predicates"]:
        raise ValueError(f"eventuality distribution {got} != receipt {doc['distribution_predicates']}")
    return labels, excluded


def envelope_census(nodes: list[dict], edges: list[dict]) -> dict:
    """Envelope census: act and mood over the :Episodic envelopes; asserted=false and modality over the content rows."""
    eps = [n["props"] for n in nodes if "Episodic" in n["labels_neo4j"]]
    content_nodes = [n["props"] for n in nodes if "asserted" in n["props"]]
    content_edges = [e["props"] for e in edges if "asserted" in e["props"]]
    return {
        "act": dict(sorted(Counter(p.get("act") for p in eps).items())),
        "mood": dict(sorted(Counter(p.get("mood") for p in eps).items())),
        "reply_to": sum(bool(p.get("reply_to")) for p in eps),
        "addressee": sum(bool(p.get("addressee")) for p in eps),
        "asserted_false": {
            "nodes": sum(p["asserted"] is False for p in content_nodes),
            "edges": sum(p["asserted"] is False for p in content_edges),
        },
        "modality": dict(sorted(Counter(p.get("modality") for p in content_nodes + content_edges).items())),
        "eventuality_source": dict(
            sorted(Counter(p.get("eventuality_source") for p in content_nodes + content_edges).items())
        ),
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def row_set_hash(nodes: list[dict], edges: list[dict]) -> str:
    return _hash([sorted(n["props"]["uuid"] for n in nodes), sorted(e["props"]["uuid"] for e in edges)])


def content_hash(nodes: list[dict], edges: list[dict]) -> str:
    key = lambda r: r["props"]["uuid"]  # noqa: E731
    return _hash(
        [
            [[sorted(n["labels_neo4j"]), n["props"]] for n in sorted(nodes, key=key)],
            [[e["type"], e["source"], e["target"], e["props"]] for e in sorted(edges, key=key)],
        ]
    )


def edge_kind(e: dict) -> str:
    """Report family of an edge: structural type, label mention, or the routed RELATES_TO family."""
    p = e["props"]
    if e["type"] == "MENTIONS":
        return "MENTIONS:label" if "p" in p else "MENTIONS"
    if e["type"] != "RELATES_TO":
        return e["type"]
    if "event_uuid" in p:
        return "RELATES_TO:role"
    return f"RELATES_TO:{p.get('eventuality', 'other')}"


def shape(nodes: list[dict], edges: list[dict]) -> dict:
    kind = edge_kind
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_label": dict(sorted(Counter("+".join(sorted(n["labels_neo4j"])) for n in nodes).items())),
        "edges_by_type": dict(sorted(Counter(e["type"] for e in edges).items())),
        "edges_by_kind": dict(sorted(Counter(kind(e) for e in edges).items())),
    }


def graph_report(nodes: list[dict], edges: list[dict], stored_keys: Mapping[str, str] | None = None) -> dict:
    """Row counts and the duplicate-name check. ``stored_keys`` (an append's chain: uuid -> the key it was minted under)
    tells a REUSED stored instance from its kind: a reused row carries no ``name_key`` (the stored one stays)."""
    stored = stored_keys or {}
    by_uuid = {n["props"]["uuid"]: n for n in nodes}
    plain = [n for n in nodes if "Entity" in n["labels_neo4j"] and "Event" not in n["labels_neo4j"]]
    events = [n for n in nodes if "Event" in n["labels_neo4j"]]
    degree: Counter[str] = Counter()
    for e in edges:
        if e["type"] == "RELATES_TO":
            degree[e["source"]] += 1
            degree[e["target"]] += 1
    top = sorted(((d, u) for u, d in degree.items() if "Event" not in by_uuid[u]["labels_neo4j"]), reverse=True)
    # A discriminated instance (``kiln#<8 hex>``, minted by an append's referents) shares its kind's name by design:
    # it is a distinct referent, never a dedup failure, so the duplicate-name check leaves it out.
    instances = {
        n["props"]["uuid"]
        for n in plain
        if split_discriminated(str(n["props"].get("name_key") or stored.get(n["props"]["uuid"]) or ""))[1]
    }
    lowered = Counter(n["props"]["name"].lower() for n in plain if n["props"]["uuid"] not in instances)
    all_lowered = Counter(n["props"]["name"].lower() for n in nodes if "Entity" in n["labels_neo4j"])
    facts = [e for e in edges if e["type"] == "RELATES_TO" and "event_keys" in e["props"]]
    relates = [e for e in edges if e["type"] == "RELATES_TO"]
    role_sources = {e["source"] for e in edges if e["type"] == "RELATES_TO" and "event_uuid" in e["props"]}
    messages = sum("Episodic" in n["labels_neo4j"] for n in nodes)
    return {
        "nodes_per_message": round(len(nodes) / max(1, messages), 3),
        "edges_per_message": round(len(edges) / max(1, messages), 3),
        "events_with_participant": sum(n["props"]["uuid"] in role_sources for n in events),
        "events_without_participant": sum(n["props"]["uuid"] not in role_sources for n in events),
        "fact_edges_by_name_top20": Counter(e["props"]["name"] for e in facts).most_common(20),
        "relates_to_by_name_top25": Counter(e["props"]["name"] for e in relates).most_common(25),
        "fact_edges_by_name_top25": Counter(e["props"]["name"] for e in facts).most_common(25),
        "fact_edges_multi_episode": sum(len(e["props"]["episodes"]) > 1 for e in facts),
        "entities_with_state_summary": sum(
            bool(n["props"].get("summary")) for n in plain if "TaxonomyLabel" not in n["labels_neo4j"]
        ),
        "edgeless_nodes": len(set(by_uuid) - {u for e in edges for u in (e["source"], e["target"])}),
        "top15_entities_by_relates_to_degree": [
            [by_uuid[u]["props"]["name"], "+".join(sorted(by_uuid[u]["labels_neo4j"])), d] for d, u in top[:15]
        ],
        "entity_name_dups_excl_events": len(plain) - len(instances) - len(lowered),
        "entity_instances_excl_events": len(instances),
        "entity_name_dup_names_excl_events": sorted(k for k, v in lowered.items() if v > 1),
        "entity_name_dups_incl_events": sum(all_lowered.values()) - len(all_lowered),
        "event_names": {
            "events": len(events),
            "distinct": len({n["props"]["name"] for n in events}),
            "distinct_ratio": round(len({n["props"]["name"] for n in events}) / max(1, len(events)), 4),
            "equal_to_message_text": 0,
        },
        "mentions": sum(e["type"] == "MENTIONS" for e in edges),
        "mentions_to_events": sum(
            e["type"] == "MENTIONS" and "Event" in by_uuid[e["target"]]["labels_neo4j"] for e in edges
        ),
        "entities_with_aliases": {
            n["props"]["name"]: n["props"]["aliases"] for n in plain if n["props"].get("aliases")
        },
    }
