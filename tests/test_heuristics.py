from __future__ import annotations

import pytest

from formula_search.astar import derive
from formula_search.docalculus import Query
from formula_search.expression import ICD, Product, Sum, Term
from formula_search.heuristics import (
    SearchState,
    conditioning_count,
    conditioning_size,
    expression_size,
    free_proof_bound,
    lookahead_bound,
    max_of,
    summation_count,
    term_count,
    zero,
)
from formula_search.suite import CURATED


def test_max_of_combines_bounds_without_adding_them() -> None:
    query = Query(frozenset({"Y"}), frozenset({"X"}))
    assert max_of(lambda q: 2, lambda q: 5)(query) == 5
    assert max_of()(query) == zero(query) == 0


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_max_of_rejects_invalid_components(value: float) -> None:
    query = Query(frozenset({"Y"}))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        max_of(lambda q: 2, lambda q: value)(query)


def test_free_proof_bound_accounts_for_do_calculus_and_costs() -> None:
    query = Query(frozenset({"Y"}), frozenset({"X"}))
    assert free_proof_bound("X -> Y")(query) == 0
    graph = "Z -> X; Z -> Y; X -> Y"
    costs = {"conditioning": 3, "chain rule": 4, "marginalization": 2}
    assert free_proof_bound(graph, costs=costs)(query) == 2


@pytest.mark.parametrize("name", ["front-door", "napkin"])
def test_lookahead_is_a_lower_bound_on_known_proofs(name: str) -> None:
    task = CURATED[name]
    query = Query(frozenset({task.outcome}), frozenset({task.treatment}))
    optimum = {"front-door": 4, "napkin": 3}[name]
    estimates = [lookahead_bound(task.graph, depth=d)(query) for d in range(3)]
    assert estimates == sorted(estimates)
    assert all(0 < estimate <= optimum for estimate in estimates)
    combined = max_of(
        free_proof_bound(task.graph), lookahead_bound(task.graph, depth=2)
    )
    result = derive(*task, bound=optimum, heuristic=combined, limit=1)
    assert len(result) == 1


def test_weighted_lookahead_and_unidentifiable_dead_end() -> None:
    task = CURATED["napkin"]
    query = Query(frozenset({"Y"}), frozenset({"X"}))
    costs = {"conditioning": 3, "chain rule": 4, "marginalization": 2}
    estimate = lookahead_bound(task.graph, costs=costs)(query)
    assert 2 <= estimate <= 9
    assert lookahead_bound("X -> Y; X <-> Y")(query) >= 1


@pytest.mark.parametrize("depth", [-1, 1.5, True])
def test_invalid_lookahead_depth(depth: int) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        lookahead_bound("X -> Y", depth=depth)


def test_expression_metrics_and_weights() -> None:
    # A deliberately explicit AST: 3 terms, 1 product, 1 sum, 1 ICD.
    expression = Sum(
        frozenset({"Z"}),
        Product(
            (
                Term(frozenset({"Z"})),
                ICD(
                    frozenset({"X"}),
                    Term(frozenset({"X", "Y"}), frozenset({"Z"})),
                ),
                Term(frozenset({"W"}), frozenset({"X", "Y", "Z"})),
            )
        ),
    )
    state = SearchState(Query(expression.outputs), 2, (), expression)
    assert expression_size()(state) == 6
    assert term_count()(state) == 3
    assert term_count(weight=2)(state) == 6
    assert summation_count()(state) == 1
    assert conditioning_count()(state) == 1
    assert conditioning_size()(state) == 3
    assert expression_size(weight=0)(state) == 0


def test_metrics_do_not_depend_on_variable_name_lengths() -> None:
    query = Query(frozenset({"Y"}))
    first = SearchState(query, 0, (), Term(frozenset({"Y"}), frozenset({"X"})))
    second = SearchState(
        query,
        0,
        (),
        Term(
            frozenset({"long_outcome_name"}),
            frozenset({"long_treatment_name"}),
        ),
    )
    for metric in (term_count(), conditioning_size(), expression_size()):
        assert metric(first) == metric(second)


@pytest.mark.parametrize("weight", [-1.0, float("inf"), float("nan")])
def test_invalid_metric_weights(weight: float) -> None:
    with pytest.raises(
        ValueError, match="weights must be finite and nonnegative"
    ):
        term_count(weight=weight)
