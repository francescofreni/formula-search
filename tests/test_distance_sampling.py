"""Rejection sampling and the joint node-order/edge-order distribution."""

import itertools
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from distance_metric import estimate_distance, resolve_edges
from distance_metric import estimator as module
from distance_metric.configs import ADJUSTMENT_01_A_VS_B as EXAMPLE
from distance_metric.edges import sample_topological_order
from distance_metric.reporting import (
    print_sampling_config,
    print_sampling_result,
)
from formula_search.graph import ADMG


def test_rejects_full_permutations_until_all_arrows_go_forward():
    graph = ADMG.parse("A -> B; B -> C; A <-> C")
    proposals = iter(
        [
            ["C", "B", "A"],
            ["A", "C", "B"],
            ["A", "B", "C"],
        ]
    )
    calls = []

    def sample(population, k):
        calls.append((population, k))
        return next(proposals)

    order, attempts = sample_topological_order(
        graph, SimpleNamespace(sample=sample)
    )
    assert order == ("A", "B", "C")
    assert attempts == 3
    assert calls == [(["A", "B", "C"], 3)] * 3


def test_accepts_exactly_valid_permutations_without_modifying_them():
    graph = ADMG.parse("A -> B; C")
    permutations = list(itertools.permutations(sorted(graph.nodes)))
    # Put a valid permutation last so every proposal is consumed.
    permutations.sort(key=lambda p: p.index("A") < p.index("B"))
    proposals = iter(permutations)
    rng = SimpleNamespace(sample=lambda population, k: next(proposals))
    accepted, attempts = [], 0
    for _ in range(3):
        order, draws = sample_topological_order(graph, rng)
        accepted.append(order)
        attempts += draws
    assert attempts == 6
    assert accepted == [p for p in permutations if p.index("A") < p.index("B")]


def test_auto_resolution_never_silently_selects_a_fixed_order():
    with pytest.raises(ValueError, match="sampled topological_order"):
        resolve_edges(replace(EXAMPLE, optional_edges=None))


def test_estimate_matches_exact_joint_distribution(monkeypatch, capsys):
    config = replace(
        EXAMPLE,
        graph="A -> B; X -> Y",
        optional_edges=None,
        include_bidirected=False,
    )
    base = ADMG.parse(config.graph)
    valid_orders = [
        p
        for p in itertools.permutations(sorted(base.nodes))
        if all(p.index(a) < p.index(b) for a, b in base.directed)
    ]
    assert len(valid_orders) == 6

    def truth(graph, formula):
        edges = ADMG.parse(graph).directed
        breaking = ("B", "X") if formula == config.formula else ("X", "A")
        return breaking not in edges

    exact_distances = []
    for topo in valid_orders:
        edges = resolve_edges(
            replace(config, topological_order=topo)
        ).optional_edges
        assert len(edges) == 4
        for additions in itertools.permutations(edges):
            graphs = [
                "; ".join((config.graph, *additions[:k])) for k in range(5)
            ]
            left, right = [
                {i for i, graph in enumerate(graphs) if truth(graph, formula)}
                for formula in (config.formula, config.other_formula)
            ]
            exact_distances.append(len(left ^ right) / len(left | right))
    exact = sum(exact_distances) / len(exact_distances)
    calls = []

    def factory(graph):
        def check(target, formula, target_bound):
            calls.append((graph, formula))
            random.getrandbits(32)
            return truth(graph, formula)

        return SimpleNamespace(check=check)

    monkeypatch.setattr(module, "HPFalsifier", factory)
    state = random.getstate()
    result = estimate_distance(config)
    assert random.getstate() == state
    assert abs(result.distance - exact) < config.epsilon
    assert set(result.topological_order_counts) == set(valid_orders)
    assert sum(result.topological_order_counts.values()) == config.n_chains
    assert result.topological_permutation_attempts > config.n_chains
    assert result.sampling_mode == "uniform_topological_orders"
    assert result.verifier_calls == len(calls)
    assert result.verifier_call_budget == 2 + 2 * config.n_chains * 3
    previous_calls = calls.copy()
    calls.clear()
    repeated = estimate_distance(config)
    assert repeated.distance == result.distance
    assert repeated.topological_order_counts == result.topological_order_counts
    assert (
        repeated.topological_permutation_attempts
        == result.topological_permutation_attempts
    )
    assert calls == previous_calls
    # Restricting to one order changes this experiment's expectation.
    fixed = estimate_distance(
        replace(config, topological_order=("A", "B", "X", "Y"))
    )
    assert abs(fixed.distance - result.distance) > config.epsilon
    assert fixed.sampling_mode == "fixed_topological_order"
    assert fixed.topological_permutation_attempts == 0
    print_sampling_config(config)
    print_sampling_result(config, result)
    output = capsys.readouterr().out
    assert "reject" in output and "6 distinct accepted orders" in output
    assert f"{config.n_chains} accepted" in output


def test_real_verifier_automatic_orders_and_no_global_rng_changes():
    config = replace(
        EXAMPLE,
        graph="Z -> X; Z -> Y",
        formula="p(Y)",
        other_formula="sum_{Z} { p(Z) p(Y | X, Z) }",
        optional_edges=None,
        epsilon=0.4,
    )
    state = random.getstate()
    result = estimate_distance(config)
    assert random.getstate() == state
    assert set(result.topological_order_counts) == {
        ("Z", "X", "Y"),
        ("Z", "Y", "X"),
    }
    assert result.verifier_calls <= result.verifier_call_budget
    assert 0 <= result.distance <= 1
