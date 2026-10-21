from __future__ import annotations

from collections.abc import Mapping

import pytest
from hiprof import HPFalsifier

from formula_search.astar import derive
from formula_search.derivation import Derivation, rules
from formula_search.derivation import derive as derive_layers
from formula_search.docalculus import Query
from formula_search.evaluator import Archive, Task
from formula_search.graph import ADMG
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
)
from formula_search.suite import CURATED


def proof_cost(d: Derivation, costs: Mapping[str, int]) -> int:
    own = 0 if d.rule in ("base", "do-calculus") else costs.get(d.rule, 1)
    return own + sum(proof_cost(p, costs) for p in d.premises)


@pytest.mark.parametrize(
    ("task", "bound"),
    [
        (Task("X -> Y", "X", "Y"), 1),
        (Task("Z -> X; X -> Y", "X", "Y"), 1),
        (Task("Z -> X; Z -> Y; X -> Y", "X", "Y"), 2),
        (CURATED["napkin"], 3),
        (CURATED["front-door"], 4),
    ],
)
def test_astar_recovers_the_bounded_formula_classes(
    task: Task, bound: int
) -> None:
    expected = derive_layers(*task, bound=bound)
    actual = derive(*task, bound=bound)
    archive = Archive(task)
    graph = ADMG.parse(task.graph)
    for d in actual:
        assert d.verify(graph)
        assert archive.submit(d.formula, d).outcome != "duplicate"
    assert len(actual) == len(expected)
    for d in expected:
        assert archive.submit(d.formula, d).outcome == "duplicate"
    costs = [proof_cost(d, {}) for d in actual]
    assert costs == sorted(costs)
    assert all(cost <= bound for cost in costs)


def test_simple_output_is_deterministic_and_can_be_limited() -> None:
    task = Task("Z -> X; X -> Y", "X", "Y")
    expected = ["p(Y | X)", "sum_{Z} { p(Z) p(Y | X, Z) }"]
    for _ in range(2):
        assert [d.formula for d in derive(*task, bound=1)] == expected
    assert [d.formula for d in derive(*task, bound=1, limit=1)] == expected[:1]


def test_weighted_costs_and_multiple_premises() -> None:
    task = CURATED["napkin"]
    costs = {"marginalization": 2, "conditioning": 3, "chain rule": 4}
    assert derive(*task, bound=8, costs=costs) == []
    actual = derive(*task, bound=9, costs=costs)
    assert len(actual) == 2
    assert all(proof_cost(d, costs) == 9 for d in actual)
    assert all(d.verify(ADMG.parse(task.graph)) for d in actual)


def test_admissible_heuristic_handles_free_equivalent_queries() -> None:
    from formula_search.docalculus import equivalent_queries

    task = CURATED["napkin"]
    graph = ADMG.parse(task.graph)
    evaluated = []

    def heuristic(query: Query) -> float:
        evaluated.append(query)
        # If no equivalent query is observational, at least one positive
        # probability-rule step is required. This is a lower bound.
        return float(
            all(q.treatments for q in equivalent_queries(graph, query))
        )

    actual = derive(*task, bound=3, heuristic=heuristic, limit=1)
    assert evaluated
    assert len(evaluated) == len(set(evaluated))
    assert len(actual) == 1
    assert proof_cost(actual[0], {}) == 3
    assert actual[0].verify(graph)


def test_inadmissible_estimate_does_not_prune_valid_proofs() -> None:
    assert [
        d.formula
        for d in derive("X -> Y", "X", "Y", bound=0, heuristic=lambda q: 99)
    ] == ["p(Y | X)"]


def test_empty_bounded_search() -> None:
    assert derive(*CURATED["napkin"], bound=0) == []


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_invalid_heuristic_estimates(value: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        derive("X -> Y", "X", "Y", bound=0, heuristic=lambda q: value)


@pytest.mark.parametrize("bound", [-1, 1.5, True])
def test_invalid_bounds(bound: int) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        derive("X -> Y", "X", "Y", bound=bound)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_limits(limit: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        derive("X -> Y", "X", "Y", bound=0, limit=limit)


def test_rule_cost_validation_is_shared() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        derive("X -> Y", "X", "Y", bound=0, costs={"conditioning": 0})
    with pytest.raises(ValueError, match="Unknown probability rule"):
        derive("X -> Y", "X", "Y", bound=0, costs={"do-calculus": 1})


def test_disabling_tiebreakers_preserves_default_order() -> None:
    task = Task("Z -> X; X -> Y", "X", "Y")
    assert derive(*task, bound=1, tiebreaker=[]) == derive(*task, bound=1)
    assert derive(*task, bound=1, tiebreaker=iter(())) == derive(
        *task, bound=1
    )


def test_tiebreakers_compose_and_preserve_formula_classes_and_cost_order() -> (
    None
):
    task = Task("Z -> X; X -> Y", "X", "Y")
    actual = derive(
        *task,
        bound=2,
        tiebreaker=[
            term_count(),
            summation_count(weight=2),
            conditioning_size(),
        ],
    )
    expected = derive(*task, bound=2)
    archive = Archive(task)
    for d in actual:
        assert d.verify(ADMG.parse(task.graph))
        archive.submit(d.formula, d)
    assert len(actual) == len(expected)
    assert all(
        archive.submit(d.formula, d).outcome == "duplicate" for d in expected
    )
    costs = [proof_cost(d, {}) for d in actual]
    assert costs == sorted(costs)


def test_completed_transport_weights_are_scored_individually() -> None:
    seen = []

    def tiebreaker(state: SearchState) -> float:
        seen.append(state)
        if state.derivation and "p(Z | X)" in state.derivation.formula:
            return -1
        return 0

    task = CURATED["napkin"]
    actual = derive(*task, bound=3, tiebreaker=tiebreaker, limit=1)
    assert len(actual) == 1
    assert "p(Z | X)" in actual[0].formula
    assert any(state.pending for state in seen)
    assert any(state.derivation is not None for state in seen)
    assert all(
        state.expression == state.derivation.expression
        for state in seen
        if state.derivation is not None
    )


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_invalid_tiebreaker_scores(value: float) -> None:
    with pytest.raises(ValueError, match="Tie-breaker scores must be finite"):
        derive("X -> Y", "X", "Y", bound=0, tiebreaker=lambda state: value)


def test_tiebreaker_scores_are_summed_and_finite() -> None:
    with pytest.raises(ValueError, match="Tie-breaker scores must be finite"):
        derive(
            "X -> Y",
            "X",
            "Y",
            bound=0,
            tiebreaker=[lambda state: 1e308, lambda state: 1e308],
        )


def assert_matches_layers(
    graph: str,
    treatment: str,
    outcomes: str | tuple[str, ...],
    bound: int,
    costs: Mapping[str, int],
    actual: list[Derivation],
) -> None:
    """Compare classes and their minimum costs, allowing different proofs."""
    expected = derive_layers(
        graph, treatment, outcomes, bound=bound, costs=costs
    )
    admg = ADMG.parse(graph)
    complete = HPFalsifier(str(admg.complete()))
    assert expected
    assert len(actual) == len(expected)
    assert all(d.verify(admg) for d in actual)
    actual_costs = [proof_cost(d, costs) for d in actual]
    assert actual_costs == sorted(actual_costs)
    assert all(cost <= bound for cost in actual_costs)
    for reference in expected:
        matches = [
            d for d in actual if complete.check(reference.formula, d.formula)
        ]
        assert len(matches) == 1
        assert proof_cost(matches[0], costs) == proof_cost(reference, costs)


@pytest.mark.parametrize(
    ("task", "bound", "costs"),
    [
        (
            Task("Z -> X; X -> Y", "X", "Y"),
            4,
            {"marginalization": 2, "conditioning": 1, "chain rule": 3},
        ),
        (
            CURATED["front-door"],
            8,
            {"marginalization": 2, "conditioning": 3, "chain rule": 2},
        ),
        (
            CURATED["napkin"],
            9,
            {"marginalization": 2, "conditioning": 3, "chain rule": 4},
        ),
    ],
)
def test_combined_guidance_preserves_classes_and_minimum_weighted_costs(
    task: Task, bound: int, costs: Mapping[str, int]
) -> None:
    heuristic = max_of(
        free_proof_bound(task.graph, costs=costs),
        lookahead_bound(task.graph, depth=2, costs=costs),
    )
    tiebreakers = [
        expression_size(),
        term_count(weight=2),
        summation_count(),
        conditioning_count(),
        conditioning_size(),
    ]
    actual = derive(
        *task,
        bound=bound,
        costs=costs,
        heuristic=heuristic,
        tiebreaker=tiebreakers,
    )
    assert_matches_layers(
        task.graph, task.treatment, task.outcome, bound, costs, actual
    )
    # Early stopping must also preserve the minimum proof cost, even when
    # tie-breakers choose a different class or representative at that cost.
    first = derive(
        *task,
        bound=bound,
        costs=costs,
        heuristic=heuristic,
        tiebreaker=tiebreakers,
        limit=1,
    )
    assert len(first) == 1
    assert proof_cost(first[0], costs) == proof_cost(actual[0], costs)


@pytest.mark.parametrize(
    ("graph", "outcomes", "bound"),
    [
        ("Z -> X; X -> Y; Y -> W", ("Y", "W"), 1),
        (
            "X -> Y; X -> Z; X -> W; Y -> Z; Y -> W; Z -> W",
            ("Y", "Z", "W"),
            2,
        ),
    ],
)
def test_multi_outcome_targets_and_nested_partial_expressions(
    graph: str, outcomes: tuple[str, ...], bound: int
) -> None:
    seen: list[SearchState] = []
    metric = expression_size()

    def tiebreaker(state: SearchState) -> float:
        seen.append(state)
        return metric(state)

    actual = derive(
        graph,
        "X",
        outcomes,
        bound=bound,
        heuristic=lookahead_bound(graph, depth=1),
        tiebreaker=tiebreaker,
    )
    assert_matches_layers(graph, "X", outcomes, bound, {}, actual)
    target_outputs = frozenset(outcomes)
    assert all(d.expression.outputs == target_outputs for d in actual)
    assert all(state.expression.outputs == target_outputs for state in seen)
    assert all(state.expression.inputs <= state.query.inputs for state in seen)
    assert any(state.pending for state in seen)
    if len(outcomes) == 3:
        # Exercise two nested chain-rule splits rather than merely passing
        # several outcomes to a search that immediately stops at a base.
        assert any(len(state.pending) == 3 for state in seen)

        def chain_steps(d: Derivation) -> int:
            return int(d.rule == "chain rule") + sum(
                chain_steps(p) for p in d.premises
            )

        nested = [
            state.derivation
            for state in seen
            if state.derivation is not None
            and chain_steps(state.derivation) >= 2
        ]
        assert nested
        assert all(d.verify(ADMG.parse(graph)) for d in nested)


@pytest.mark.parametrize("use_tiebreaker", [False, True])
def test_admissible_inconsistent_heuristic_preserves_minimum_costs(
    use_tiebreaker: bool,
) -> None:
    task = CURATED["napkin"]
    target = task.query
    graph = ADMG.parse(task.graph)
    # Establish the target's true minimum independently. The heuristic is
    # exact only at this query and zero everywhere else, hence admissible.
    assert derive_layers(*task, bound=2) == []
    assert derive_layers(*task, bound=3)
    evaluated = []

    def heuristic(query: Query) -> float:
        evaluated.append(query)
        return 3.0 if query == target else 0.0

    # Exhibit an actual backward rule where consistency fails:
    # h(parent) > rule cost + sum h(premises).
    _, premises = next(rules(graph, target))
    assert heuristic(target) > 1 + sum(heuristic(p) for p in premises)
    evaluated.clear()
    actual = derive(
        *task,
        bound=3,
        heuristic=heuristic,
        tiebreaker=(
            [term_count(), summation_count()] if use_tiebreaker else None
        ),
    )
    assert target in evaluated
    assert any(query != target for query in evaluated)
    assert_matches_layers(*task, 3, {}, actual)
    first = derive(
        *task,
        bound=3,
        heuristic=heuristic,
        limit=1,
        tiebreaker=expression_size() if use_tiebreaker else None,
    )
    assert len(first) == 1
    assert proof_cost(first[0], {}) == 3
