"""Boundary, sampling, graph-family, and real-verifier checks."""

import itertools
import math
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from distance_metric import estimate_distance
from distance_metric import estimator as module
from distance_metric.configs import ADJUSTMENT_01_A_VS_B as EXAMPLE
from distance_metric.configs import ADJUSTMENT_02_SHARED_CONFOUNDING
from formula_search.graph import ADMG
from formula_search.verification import (
    VerificationSettings,
)


def test_shared_config_and_sampling_budget():
    assert isinstance(EXAMPLE, VerificationSettings)
    assert EXAMPLE.n_chains == 1060
    assert EXAMPLE.epsilon == 0.05
    n = EXAMPLE.n_chains
    assert 2 * math.exp(-2 * n * EXAMPLE.epsilon**2) <= EXAMPLE.delta
    assert 2 * math.exp(-2 * (n - 1) * EXAMPLE.epsilon**2) > EXAMPLE.delta


@pytest.mark.parametrize("m", range(17))
def test_binary_search_every_possible_boundary(m):
    for threshold in range(1, m + 2):
        queries = []

        def valid(k, threshold=threshold, queries=queries):
            assert 1 <= k <= m
            queries.append(k)
            return k < threshold

        assert module._first_invalid(m, valid) == threshold
        assert len(queries) <= m.bit_length()


@pytest.mark.parametrize(
    "changes",
    [
        {"epsilon": 0},
        {"epsilon": float("nan")},
        {"epsilon": float("inf")},
        {"delta": -0.1},
        {"delta": 1},
        {"target_bound": 0},
        {"target_bound": float("nan")},
        {"seed": 0.5},
        {"formulas": ("p(Y)", "")},
        {"optional_edges": "A -> X"},
    ],
)
def test_bad_config_values(changes):
    with pytest.raises(ValueError):
        replace(EXAMPLE, **changes)


@pytest.mark.parametrize(
    "edges",
    [
        ("A -> B",),  # Already present.
        ("A <-> B", "B <-> A"),  # Same edge in reverse spelling.
        ("A -> X", "A->X"),
        ("Z -> X",),  # Introduces a new variable.
        ("A -> X; B -> Y",),  # Two edges in one option.
        ("A -> X; B",),  # One edge plus an extra isolated node.
        ("Y -> A",),  # Directed cycle with base edges.
    ],
)
def test_bad_optional_edges(edges):
    with pytest.raises(ValueError):
        module._validate_family(replace(EXAMPLE, optional_edges=edges))


def test_optional_edges_must_coexist_without_cycles():
    config = replace(
        EXAMPLE,
        graph="A; B; X -> Y",
        optional_edges=("A -> B", "B -> A"),
    )
    with pytest.raises(ValueError):
        module._validate_family(config)


def test_candidate_must_be_observational():
    with pytest.raises(ValueError, match="observational"):
        module._validate_family(
            replace(EXAMPLE, formulas=(EXAMPLE.target, EXAMPLE.formulas[1]))
        )


def test_bidirected_optional_edges_are_supported():
    edges = EXAMPLE.optional_edges + ("A <-> Y",)
    assert (
        len(module._validate_family(replace(EXAMPLE, optional_edges=edges)))
        == 3
    )


def test_real_hiprof_known_half_distance():
    result = estimate_distance(replace(EXAMPLE, epsilon=0.4))
    assert result.distance == 0.5
    assert result.verifier_calls == 2 + 3 * result.n_chains
    assert result.verifier_call_budget == 2 + 4 * result.n_chains
    assert result.interval == pytest.approx((0.1, 0.9))
    assert result.verifier_failure_bound == pytest.approx(
        result.verifier_call_budget * EXAMPLE.target_bound
    )
    assert result.total_failure_bound == pytest.approx(
        EXAMPLE.delta + result.verifier_failure_bound
    )
    assert result.elapsed_seconds > 0


def test_real_hiprof_self_distance():
    result = estimate_distance(
        replace(EXAMPLE, formulas=(EXAMPLE.formulas[0],) * 2, epsilon=0.4)
    )
    assert result.distance == 0
    assert result.interval == pytest.approx((0, 0.4))


def test_real_hiprof_rejects_invalid_start():
    with pytest.raises(ValueError, match=r"formulas\[1\] failed"):
        estimate_distance(
            replace(EXAMPLE, formulas=(EXAMPLE.formulas[0], "p(Y | X)"))
        )


def test_empty_edge_family():
    result = estimate_distance(replace(EXAMPLE, optional_edges=()))
    assert result.distance == 0
    assert result.verifier_calls == result.verifier_call_budget == 2


def test_sampled_distance_matches_enumeration_and_is_reproducible(monkeypatch):
    config = replace(
        EXAMPLE, optional_edges=EXAMPLE.optional_edges + ("A <-> Y",)
    )
    calls = []

    def truth(graph, formula):
        directed = ADMG.parse(graph).directed
        a = ("A", "X") in directed
        b = ("B", "Y") in directed
        return not a if formula == config.formulas[0] else not (a and b)

    def factory(graph):
        def check(target, formula, target_bound):
            assert target == config.target
            assert target_bound == config.target_bound
            calls.append((graph, formula))
            random.getrandbits(64)  # Simulate independent verifier randomness.
            return truth(graph, formula)

        return SimpleNamespace(check=check)

    monkeypatch.setattr(module, "HPFalsifier", factory)
    exact_distances = []
    for order in itertools.permutations(config.optional_edges):
        graphs = ["; ".join((config.graph, *order[:k])) for k in range(4)]
        left = {
            i
            for i, graph in enumerate(graphs)
            if truth(graph, config.formulas[0])
        }
        right = {
            i
            for i, graph in enumerate(graphs)
            if truth(graph, config.formulas[1])
        }
        exact_distances.append(len(left ^ right) / len(left | right))
    assert len(set(exact_distances)) > 1
    exact = sum(exact_distances) / len(exact_distances)
    state = random.getstate()
    result = estimate_distance(config)
    assert random.getstate() == state
    assert abs(result.distance - exact) < config.epsilon
    assert result.verifier_calls == len(calls)
    assert result.verifier_calls <= result.verifier_call_budget
    first_calls = calls.copy()
    calls.clear()
    again = estimate_distance(config)
    assert again.distance == result.distance
    assert calls == first_calls
    swapped = estimate_distance(
        replace(config, formulas=tuple(reversed(config.formulas)))
    )
    assert swapped.distance == result.distance


def test_rng_restored_on_verifier_error(monkeypatch):
    def fail(graph):
        random.getrandbits(64)
        raise RuntimeError("verifier error")

    monkeypatch.setattr(module, "HPFalsifier", fail)
    state = random.getstate()
    with pytest.raises(RuntimeError, match="verifier error"):
        estimate_distance(EXAMPLE)
    assert random.getstate() == state


def test_all_valid_prefixes_and_clipped_upper_interval(monkeypatch):
    def factory(graph):
        def check(target, formula, target_bound):
            # First formula works only at the original graph; second always works.
            return formula == EXAMPLE.formulas[1] or graph == EXAMPLE.graph

        return SimpleNamespace(check=check)

    monkeypatch.setattr(module, "HPFalsifier", factory)
    result = estimate_distance(replace(EXAMPLE, epsilon=0.4))
    assert result.distance == pytest.approx(2 / 3)
    assert result.interval[1] == 1


def test_shared_confounding_exact_six_chains():
    config = ADJUSTMENT_02_SHARED_CONFOUNDING
    distances = []
    random_state = random.getstate()
    try:
        random.seed(0)
        for order in itertools.permutations(config.optional_edges):
            graphs = ["; ".join((config.graph, *order[:k])) for k in range(4)]
            valid_sets = [
                {
                    i
                    for i, graph in enumerate(graphs)
                    if module.HPFalsifier(graph).check(
                        config.target,
                        formula,
                        target_bound=config.target_bound,
                    )
                }
                for formula in config.formulas
            ]
            left, right = valid_sets
            distances.append(len(left ^ right) / len(left | right))
    finally:
        random.setstate(random_state)
    assert sorted(distances) == [0, 0, 0.5, 0.5, 0.5, 0.5]
    assert sum(distances) / len(distances) == pytest.approx(1 / 3)
