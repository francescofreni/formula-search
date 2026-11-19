"""Group scores share chain boundaries and preserve the pairwise metric."""

import itertools
from dataclasses import replace
from statistics import mean
from types import SimpleNamespace

import pytest

from distance_metric import estimate_distance
from distance_metric import estimator as module
from distance_metric.configs import ADJUSTMENT_01_A_VS_B as PAIR
from distance_metric.configs import ADJUSTMENT_04_THREE_FORMULAS as GROUP
from formula_search.graph import ADMG


@pytest.mark.parametrize(
    "formulas",
    [
        (),
        [],
        ("p(Y)",),
        ["p(Y)"],
        "p(Y)",
        None,
        ("p(Y)", ""),
        ("p(Y)", 2),
        {"p(Y)", "p(Y | X)"},
    ],
)
def test_rejects_missing_or_malformed_formula_collections(formulas):
    with pytest.raises(ValueError):
        replace(PAIR, formulas=formulas)


def test_list_input_is_copied_to_immutable_tuple():
    formulas = list(GROUP.formulas)
    config = replace(GROUP, formulas=formulas)
    formulas.clear()
    assert config.formulas == GROUP.formulas
    assert isinstance(config.formulas, tuple)
    assert config.n_chains == PAIR.n_chains == 1060
    assert config.n_pairs == 3


@pytest.mark.parametrize("n", [2, 3, 5])
def test_one_boundary_search_per_formula_and_exact_pair_average(
    monkeypatch, n
):
    config = replace(PAIR, formulas=tuple(f"p(F{i})" for i in range(n)))
    thresholds = dict(zip(config.formulas, itertools.cycle((1, 2, 4))))
    edges = ["A -> X", "B -> Y", "X <-> Y"]
    base_search = module._first_invalid
    searches = []

    def boundary(m, valid):
        searches.append(m)
        return base_search(m, valid)

    monkeypatch.setattr(module, "_first_invalid", boundary)

    def valid(graph, formula):
        additions = len(graph.split(";")) - len(config.graph.split(";"))
        return additions < thresholds[formula]

    result = module._chain_distance(config, edges, valid)
    sets = [
        {i for i in range(4) if i < thresholds[f]} for f in config.formulas
    ]
    expected = mean(
        len(a ^ b) / len(a | b) for a, b in itertools.combinations(sets, 2)
    )
    assert result == pytest.approx(expected)
    assert searches == [3] * n


def test_group_estimate_equals_mean_of_pair_estimates(monkeypatch):
    config = replace(GROUP, optional_edges=PAIR.optional_edges)

    def truth(graph, formula):
        edges = ADMG.parse(graph).directed
        if formula == config.formulas[0]:
            return ("A", "X") not in edges
        if formula == config.formulas[1]:
            return ("B", "Y") not in edges
        return True

    def factory(graph):
        return SimpleNamespace(
            check=lambda target, formula, target_bound: truth(graph, formula)
        )

    monkeypatch.setattr(module, "HPFalsifier", factory)
    result = estimate_distance(config)
    pairs = [
        estimate_distance(replace(config, formulas=pair))
        for pair in itertools.combinations(config.formulas, 2)
    ]
    assert result.distance == pytest.approx(mean(r.distance for r in pairs))
    assert result.n_formulas == 3
    assert result.n_pairs == 3
    assert result.n_chains == 1060
    assert result.verifier_call_budget == 3 * (1 + 2 * result.n_chains)
    assert result.verifier_failure_bound == pytest.approx(
        result.verifier_call_budget * config.target_bound
    )
    reordered = estimate_distance(
        replace(config, formulas=tuple(reversed(config.formulas)))
    )
    assert reordered.distance == result.distance


def test_verifier_calls_grow_linearly_not_per_pair(monkeypatch):
    calls = []

    def factory(graph):
        def check(target, formula, target_bound):
            calls.append((graph, formula))
            return True

        return SimpleNamespace(check=check)

    monkeypatch.setattr(module, "HPFalsifier", factory)
    config = replace(PAIR, formulas=PAIR.formulas * 3, epsilon=0.4)
    result = estimate_distance(config)
    assert result.n_formulas == 6
    assert result.n_pairs == 15
    assert result.distance == 0
    assert result.verifier_calls == len(calls) == 6 * (1 + 2 * config.n_chains)
    assert result.verifier_calls == result.verifier_call_budget


def test_real_hiprof_group_and_pair_regression():
    config = replace(GROUP, optional_edges=PAIR.optional_edges, epsilon=0.4)
    result = estimate_distance(config)
    pair_results = [
        estimate_distance(replace(config, formulas=pair))
        for pair in itertools.combinations(config.formulas, 2)
    ]
    assert result.distance == pytest.approx(
        mean(r.distance for r in pair_results)
    )
    assert pair_results[0].distance == 0.5
    assert result.n_formulas == 3
    assert result.n_pairs == 3
    assert 0 <= result.distance <= 1


def test_all_formulas_must_be_observational_and_valid_on_base():
    with pytest.raises(ValueError, match="observational"):
        estimate_distance(
            replace(GROUP, formulas=(*PAIR.formulas, PAIR.target))
        )
    with pytest.raises(ValueError, match=r"formulas\[2\] failed"):
        estimate_distance(
            replace(GROUP, formulas=(*PAIR.formulas, "p(Y | X)"))
        )


def test_empty_family_checks_every_formula_and_repeated_entries_are_retained():
    config = replace(GROUP, optional_edges=(), formulas=PAIR.formulas * 2)
    result = estimate_distance(config)
    assert result.n_formulas == 4
    assert result.n_pairs == 6
    assert result.distance == 0
    assert result.verifier_calls == result.verifier_call_budget == 4
