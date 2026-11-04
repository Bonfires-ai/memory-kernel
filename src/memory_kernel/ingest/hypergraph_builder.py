"""Hypergraph assembly and validation."""

from __future__ import annotations

from .nlp_models import (
    Episode,
    EpisodeHyperedge,
    Fact,
    FactHyperedge,
    Hypergraph,
    Topic,
)


class HypergraphBuilder:
    """Assemble topic, episode, and fact nodes with bidirectional checks."""

    def build(
        self,
        *,
        topics: list[Topic],
        episodes: list[Episode],
        facts: list[Fact],
        episode_hyperedges: list[EpisodeHyperedge],
        fact_hyperedges: list[FactHyperedge],
        primitive_graph: object | None = None,
    ) -> Hypergraph:
        """Index the nodes and edges by id into a hypergraph, validating before returning it.

        Raises :class:`ValueError` listing every problem found when validation fails — an invalid graph is
        never returned, so callers can treat a returned graph as internally consistent. Items sharing an id
        collapse, last one winning, since each collection becomes a mapping keyed by id.
        """
        graph = Hypergraph(
            topics={topic.id: topic for topic in topics},
            episodes={episode.id: episode for episode in episodes},
            facts={fact.id: fact for fact in facts},
            episode_hyperedges={edge.id: edge for edge in episode_hyperedges},
            fact_hyperedges={edge.id: edge for edge in fact_hyperedges},
            primitive_graph=primitive_graph,
        )
        errors = validate_hypergraph(graph)
        if errors:
            raise ValueError("Invalid hypergraph: " + "; ".join(errors))
        return graph


def validate_hypergraph(graph: Hypergraph) -> list[str]:
    """Report every referential inconsistency in a hypergraph; an empty list means the graph is sound.

    Checks that every edge points at nodes that exist, that no edge carries a weight for something it does
    not link, that a fact hyperedge only links facts that claim its episode, and that topics and facts only
    cite episodes and topics present in the graph. A topic's episodes are also required to be reachable
    through an episode hyperedge, but only once that topic has at least one such edge — a topic with no edges
    at all is not faulted for it.

    Collects all problems rather than stopping at the first, so one call is enough to see the whole picture.
    """
    errors: list[str] = []
    for edge_id, edge in graph.episode_hyperedges.items():
        if edge.topic_id not in graph.topics:
            errors.append(f"episode hyperedge {edge_id} references missing topic {edge.topic_id}")
        for episode_id in edge.relation:
            if episode_id not in graph.episodes:
                errors.append(f"episode hyperedge {edge_id} references missing episode {episode_id}")
        for episode_id in edge.weights:
            if episode_id not in edge.relation:
                errors.append(f"episode hyperedge {edge_id} has weight for unlinked episode {episode_id}")

    for topic_id, topic in graph.topics.items():
        linked = {
            episode_id
            for edge in graph.episode_hyperedges.values()
            if edge.topic_id == topic_id
            for episode_id in edge.relation
        }
        for episode_id in topic.episode_ids:
            if episode_id not in graph.episodes:
                errors.append(f"topic {topic_id} references missing episode {episode_id}")
            if linked and episode_id not in linked:
                errors.append(f"topic {topic_id} episode {episode_id} missing episode hyperedge link")

    for edge_id, edge in graph.fact_hyperedges.items():
        if edge.episode_id not in graph.episodes:
            errors.append(f"fact hyperedge {edge_id} references missing episode {edge.episode_id}")
        for fact_id in edge.relation:
            if fact_id not in graph.facts:
                errors.append(f"fact hyperedge {edge_id} references missing fact {fact_id}")
            elif edge.episode_id not in graph.facts[fact_id].episode_ids:
                errors.append(f"fact hyperedge {edge_id} links fact {fact_id} outside its episode_ids")
        for fact_id in edge.weights:
            if fact_id not in edge.relation:
                errors.append(f"fact hyperedge {edge_id} has weight for unlinked fact {fact_id}")

    for fact_id, fact in graph.facts.items():
        if fact.topic_id not in graph.topics:
            errors.append(f"fact {fact_id} references missing topic {fact.topic_id}")
        for episode_id in fact.episode_ids:
            if episode_id not in graph.episodes:
                errors.append(f"fact {fact_id} references missing episode {episode_id}")
    return errors


__all__ = ["HypergraphBuilder", "validate_hypergraph"]
