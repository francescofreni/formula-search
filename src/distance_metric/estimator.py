"""Estimate distance over uniform valid node orders and edge permutations.

Validity decreases along each chain. Binary search finds each formula's
first invalid position, hence the size of its valid prefix. HiProF answers
are probabilistic and restricted to linear-Gaussian models. No verdict
cache is used; only the two base-graph checks are shared across chains.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import perf_counter

from hiprof import HPFalsifier

from formula_search.graph import ADMG

from .config import DistanceConfig
from .edges import resolve_edges, sample_topological_order


@dataclass(frozen=True)
class DistanceResult:
    distance: float
    interval: tuple[float, float]
    n_chains: int
    verifier_calls: int
    verifier_call_budget: int
    elapsed_seconds: float
    sampling_failure_probability: float
    verifier_failure_bound: float
    total_failure_bound: float
    sampling_mode: str = "explicit_edges"
    topological_permutation_attempts: int = 0
    topological_order_counts: dict[tuple[str, ...], int] = field(
        default_factory=dict
    )


def _validate_family(config: DistanceConfig) -> tuple[str, ...]:
    """Require distinct new edges on fixed nodes and an acyclic full graph."""
    edges = resolve_edges(config).optional_edges
    _validate_formulas(config)
    return edges


def _validate_formulas(config: DistanceConfig) -> None:
    if any(
        re.search(r"\bdo\s*\(", formula)
        for formula in (config.formula, config.other_formula)
    ):
        raise ValueError("Both candidate formulas must be observational")


def _first_invalid(m: int, valid: Callable[[int], bool]) -> int:
    """Graph 0 is known valid; m+1 is a sentinel meaning all graphs valid.

    Query only positions 1..m. The returned index also equals the number
    of valid graphs in this chain, counting the starting graph.
    """
    low, high = 0, m + 1
    while high - low > 1:
        middle = (low + high) // 2
        if valid(middle):
            low = middle
        else:
            high = middle
    return high


def _chain_distance(
    config: DistanceConfig,
    order: list[str],
    valid: Callable[[str, str], bool],
) -> float:
    def boundary(formula: str) -> int:
        return _first_invalid(
            len(order),
            lambda k: valid("; ".join((config.graph, *order[:k])), formula),
        )

    first = boundary(config.formula)
    second = boundary(config.other_formula)
    return abs(first - second) / max(first, second)


def _estimate(config: DistanceConfig) -> DistanceResult:
    start = perf_counter()
    _validate_formulas(config)
    base = ADMG.parse(config.graph)
    sampled_orders = (
        config.optional_edges is None and config.topological_order is None
    )
    fixed_family = None if sampled_orders else resolve_edges(config)
    if fixed_family is None:
        # Every accepted order has n*(n-1)/2 forward pairs and includes all
        # existing directed edges. Thus the number of additions is constant.
        pairs = len(base.nodes) * (len(base.nodes) - 1) // 2
        edge_count = pairs - len(base.directed)
        if config.include_bidirected:
            edge_count += pairs - len(base.bidirected)
    else:
        edge_count = len(fixed_family.optional_edges)
    rng = random.Random(f"{config.seed}:chains")
    topo_rng = random.Random(f"{config.seed}:topological_orders")
    attempts = 0
    order_counts: dict[tuple[str, ...], int] = {}
    calls = 0

    def valid(graph: str, formula: str) -> bool:
        nonlocal calls
        calls += 1
        return bool(
            HPFalsifier(graph).check(
                config.target, formula, target_bound=config.target_bound
            )
        )

    for name in ("formula", "other_formula"):
        if not valid(config.graph, getattr(config, name)):
            raise ValueError(f"{name} failed verification on the base graph")

    total = 0.0
    for _ in range(config.n_chains):
        if fixed_family is None:
            topo_order, draws = sample_topological_order(base, topo_rng)
            attempts += draws
            family = resolve_edges(
                replace(config, topological_order=topo_order)
            )
            order_counts[topo_order] = order_counts.get(topo_order, 0) + 1
        else:
            family = fixed_family
        edges = family.optional_edges
        order = rng.sample(edges, len(edges))
        total += _chain_distance(config, order, valid)
    distance = total / config.n_chains
    call_budget = 2 + 2 * config.n_chains * edge_count.bit_length()
    verifier_bound = min(1.0, call_budget * config.target_bound)
    return DistanceResult(
        distance=distance,
        interval=(
            max(0.0, distance - config.epsilon),
            min(1.0, distance + config.epsilon),
        ),
        n_chains=config.n_chains,
        verifier_calls=calls,
        verifier_call_budget=call_budget,
        elapsed_seconds=perf_counter() - start,
        sampling_failure_probability=config.delta,
        verifier_failure_bound=verifier_bound,
        total_failure_bound=min(1.0, config.delta + verifier_bound),
        sampling_mode=(
            "uniform_topological_orders"
            if sampled_orders
            else (
                "explicit_edges"
                if config.optional_edges is not None
                else "fixed_topological_order"
            )
        ),
        topological_permutation_attempts=attempts,
        topological_order_counts=order_counts,
    )


def estimate_distance(config: DistanceConfig) -> DistanceResult:
    """Sample IID chains, binary-search validity boundaries, average distances.

    The interval targets expected chain Jaccard distance, not Jaccard over
    uniformly sampled graphs. Both formulas must pass on the base graph.
    The failure bound combines Hoeffding sampling error with a union bound
    on HiProF false acceptances, within its linear-Gaussian model family.

    By default each chain starts with rejection sampling of a uniform full
    variable permutation until it respects the base graph's arrows. Missing
    forward edges and (by default) missing bidirected edges are then shuffled.
    Explicit edge lists or an explicit order select a fixed-family experiment.

    Topological, edge-order, and verifier draws use separate seeded streams.
    The global Python RNG is restored afterward. Run sequentially: HiProF uses the global RNG,
    so concurrent threads using random would interfere with reproducibility.
    """
    state = random.getstate()
    try:
        random.seed(f"{config.seed}:verification")
        return _estimate(config)
    finally:
        random.setstate(state)
