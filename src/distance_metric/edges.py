"""Resolve a reproducible, acyclic family of optional graph additions."""

from dataclasses import dataclass
from itertools import combinations
from random import Random

from formula_search.graph import ADMG

from .config import DistanceConfig


@dataclass(frozen=True)
class EdgeFamily:
    topological_order: tuple[str, ...]
    optional_edges: tuple[str, ...]


def _order(graph: ADMG, requested: tuple[str, ...] | None) -> tuple[str, ...]:
    if requested is None:
        return graph.topological_order()
    if len(requested) != len(graph.nodes) or set(requested) != graph.nodes:
        raise ValueError(
            "topological_order must list each graph variable once"
        )
    positions = {node: i for i, node in enumerate(requested)}
    if any(positions[a] >= positions[b] for a, b in graph.directed):
        raise ValueError("topological_order conflicts with directed edges")
    return requested


def resolve_edges(config: DistanceConfig) -> EdgeFamily:
    """Generate missing forward edges, or validate an explicit override.

    Automatic families require an explicit topological order here. The
    estimator samples a fresh order for each chain using rejection sampling.
    Explicit edge tuples, including (), override generation and its flag.
    """
    base = ADMG.parse(config.graph)
    candidates = config.optional_edges
    if candidates is None:
        if config.topological_order is None:
            raise ValueError(
                "Automatic edges need a sampled topological_order; "
                "use estimate_distance to sample orders for all chains"
            )
        order = _order(base, config.topological_order)
        directed = tuple(
            f"{a} -> {b}"
            for a, b in combinations(order, 2)
            if (a, b) not in base.directed
        )
        bidirected = tuple(
            f"{a} <-> {b}"
            for a, b in combinations(sorted(base.nodes), 2)
            if frozenset((a, b)) not in base.bidirected
        )
        candidates = directed + (
            bidirected if config.include_bidirected else ()
        )
    seen: set[str] = set()
    edges = []
    for text in candidates:
        edge = ADMG.parse(text)
        if (
            len(edge.nodes) != 2
            or len(edge.directed) + len(edge.bidirected) != 1
        ):
            raise ValueError(f"Expected exactly one optional edge: {text}")
        if not edge.nodes <= base.nodes:
            raise ValueError(f"Optional edges cannot add variables: {text}")
        if edge.directed & base.directed or edge.bidirected & base.bidirected:
            raise ValueError(f"Optional edge already exists: {text}")
        canonical = str(edge)
        if canonical in seen:
            raise ValueError(f"Duplicate optional edge: {text}")
        seen.add(canonical)
        edges.append(canonical)
    # Acyclic union ensures every prefix of every permutation is acyclic.
    full = ADMG.parse("; ".join((config.graph, *edges)))
    return EdgeFamily(_order(full, config.topological_order), tuple(edges))


def sample_topological_order(
    graph: ADMG, rng: Random
) -> tuple[tuple[str, ...], int]:
    """Reject uniform full-variable permutations until all arrows go forward.

    Every valid permutation has the same acceptance probability. There is
    no retry cap or alternative sampler. Return the accepted order and the
    number of attempted permutations, including the accepted one.
    """
    nodes = sorted(graph.nodes)
    attempts = 0
    while True:
        attempts += 1
        order = tuple(rng.sample(nodes, len(nodes)))
        positions = {node: i for i, node in enumerate(order)}
        if all(positions[a] < positions[b] for a, b in graph.directed):
            return order, attempts
