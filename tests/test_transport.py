from __future__ import annotations

from collections.abc import Callable
from functools import cache
from itertools import combinations, product

import pytest
from hiprof import HPFalsifier

from formula_search.astar import derive as astar
from formula_search.derivation import (
    Derivation,
    SearchLimitExceeded,
    derive,
    rules,
)
from formula_search.docalculus import Query, equivalent_queries
from formula_search.graph import ADMG
from formula_search.heuristics import lookahead_bound, term_count
from formula_search.suite import CURATED


def q(y: str, x: str = "", w: str = "") -> Query:
    return Query(*(frozenset(value.split()) for value in (y, x, w)))


def napkin_proofs() -> tuple[Derivation, Derivation]:
    """Independent certificates: single transport and composed transport."""
    graph = ADMG.parse(CURATED["napkin"].graph)

    def transport(
        d: Derivation, target: Query, weight: str = ""
    ) -> Derivation:
        path = equivalent_queries(graph, target)[d.query]
        return Derivation(
            target, "do-calculus", (d,), path, frozenset(weight.split())
        )

    w = transport(Derivation(q("W"), "base"), q("W", "Z"))
    xy = transport(Derivation(q("X Y", w="W Z"), "base"), q("X Y", "Z", "W"))
    joint = Derivation(q("W X Y", "Z"), "chain rule", (w, xy))
    conditional = Derivation(q("W Y", "Z", "X"), "conditioning", (joint,))
    kernel = Derivation(q("Y", "Z", "X"), "marginalization", (conditional,))
    first = transport(kernel, q("Y", "X"))
    intermediate = transport(kernel, q("Y", "W X"), "W")
    composed = transport(intermediate, q("Y", "X"), "X")
    assert first.verify(graph) and composed.verify(graph)
    return first, composed


@pytest.mark.parametrize("engine", [derive, astar])
def test_composed_napkin_transport_is_found_without_probability_cost(
    engine: Callable,
) -> None:
    first, composed = napkin_proofs()
    graph = ADMG.parse(CURATED["napkin"].graph)
    complete = HPFalsifier(str(graph.complete()))
    assert first.cost() == composed.cost() == 3
    assert not complete.check(first.formula, composed.formula)
    actual = engine(*CURATED["napkin"], bound=3, limit=3, max_expansions=3000)
    assert len(actual) == 3
    assert any(complete.check(d.formula, composed.formula) for d in actual)
    assert all(d.cost() == 3 and d.verify(graph) for d in actual)


@pytest.mark.parametrize("engine", [derive, astar])
def test_positive_transport_cost_bounds_composition(engine: Callable) -> None:
    _, composed = napkin_proofs()
    task = CURATED["napkin"]
    costs = {"transport": 1}
    complete = HPFalsifier(str(ADMG.parse(task.graph).complete()))
    assert composed.cost(costs) == 7
    before = engine(*task, bound=6, costs=costs)
    after = engine(*task, bound=7, costs=costs)
    assert len(before) == 2
    assert not any(complete.check(d.formula, composed.formula) for d in before)
    matches = [d for d in after if complete.check(d.formula, composed.formula)]
    assert len(matches) == 1 and matches[0].cost(costs) == 7
    assert [d.cost(costs) for d in after] == sorted(
        d.cost(costs) for d in after
    )


def test_composition_with_lookahead_and_tiebreakers() -> None:
    _, composed = napkin_proofs()
    task = CURATED["napkin"]
    costs = {"transport": 1}
    actual = astar(
        *task,
        bound=7,
        costs=costs,
        heuristic=lookahead_bound(task.graph, costs=costs),
        tiebreaker=term_count(),
    )
    complete = HPFalsifier(str(ADMG.parse(task.graph).complete()))
    assert any(complete.check(d.formula, composed.formula) for d in actual)
    assert [d.cost(costs) for d in actual] == [6, 6, 7, 7]
    assert all(d.verify(ADMG.parse(task.graph)) for d in actual)


@pytest.mark.parametrize("engine", [derive, astar])
def test_zero_cost_budget_exhaustion_is_explicit(engine: Callable) -> None:
    with pytest.raises(
        SearchLimitExceeded, match="Results are incomplete"
    ) as info:
        engine(*CURATED["napkin"], bound=3, max_expansions=500)
    assert info.value.expanded == 500
    graph = ADMG.parse(CURATED["napkin"].graph)
    assert all(d.verify(graph) for d in info.value.derivations)


@pytest.mark.parametrize("engine", [derive, astar])
def test_redundant_free_cycles_terminate_and_empty_search_can_exhaust(
    engine: Callable,
) -> None:
    assert [d.formula for d in engine("X -> Y", "X", "Y", bound=0)] == [
        "p(Y | X)"
    ]
    assert engine(*CURATED["bow arc"], bound=0) == []


@pytest.mark.parametrize("value", [-1, 0.5, True, "0"])
def test_invalid_transport_costs(value: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        derive("X -> Y", "X", "Y", bound=0, costs={"transport": value})  # type: ignore[dict-item]


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_expansion_limits(value: int) -> None:
    with pytest.raises(ValueError, match="Max expansions"):
        derive("X -> Y", "X", "Y", bound=0, max_expansions=value)


def brute_transports(
    graph: ADMG,
    query: Query,
    budget: int,
    trees: Callable[[Query, int], tuple[Derivation, ...]],
) -> list[Derivation]:
    proofs = []
    for member, path in equivalent_queries(graph, query).items():
        if member == query:
            continue
        for child in trees(member, budget - 1):
            inputs = sorted(query.inputs)
            extra = child.expression.inputs - query.inputs
            for size in range(len(inputs) + 1 if extra else 1):
                for weight in combinations(inputs, size):
                    proofs.append(
                        Derivation(
                            query,
                            "do-calculus",
                            (child,),
                            path,
                            frozenset(weight),
                        )
                    )
    return proofs


def brute_force(graph: ADMG, target: Query, bound: int) -> list[Derivation]:
    """Independent finite oracle: enumerate trees with all rule costs one.

    No search agenda, class memoization, or shared transport helper is used.
    Positive transport cost makes recursion strictly decrease the budget.
    """
    costs = {"transport": 1}

    @cache
    def trees(query: Query, budget: int) -> tuple[Derivation, ...]:
        proofs = [] if query.treatments else [Derivation(query, "base")]
        if budget == 0:
            return tuple(proofs)
        for rule, premises in rules(graph, query):
            for children in product(*(trees(p, budget - 1) for p in premises)):
                d = Derivation(query, rule, children)
                if d.cost(costs) <= budget:
                    proofs.append(d)
        proofs.extend(brute_transports(graph, query, budget, trees))
        return tuple(proofs)

    return sorted(trees(target, bound), key=lambda d: d.cost(costs))


@pytest.mark.parametrize("guided", [False, True])
def test_agenda_matches_independent_tree_enumeration(guided: bool) -> None:
    # A smaller graph allows exhaustive proof-tree enumeration in the test.
    graph = ADMG.parse("Z -> X; X -> Y")
    target = q("Y", "X")
    costs = {"transport": 1}
    expected = brute_force(graph, target, 2)
    complete = HPFalsifier(str(graph.complete()))
    classes = []
    for d in expected:
        if not any(
            complete.check(previous.formula, d.formula) for previous in classes
        ):
            classes.append(d)
    actual = astar(
        str(graph),
        "X",
        "Y",
        bound=2,
        costs=costs,
        heuristic=lookahead_bound(graph, costs=costs) if guided else None,
        tiebreaker=term_count() if guided else None,
    )
    assert len(classes) == len(actual) == 2
    for reference in classes:
        matches = [
            d for d in actual if complete.check(reference.formula, d.formula)
        ]
        assert len(matches) == 1
        assert matches[0].cost(costs) == reference.cost(costs)
        assert matches[0].verify(graph)


def test_cheaper_proofs_reopen_a_previously_admitted_class() -> None:
    from formula_search.derivation import _Proof, _Search

    graph = ADMG.parse("X -> Y")
    target = q("Y")
    expensive = Derivation(
        target, "marginalization", (Derivation(q("X Y"), "base"),)
    )
    cheap = Derivation(target, "base")
    assert expensive.verify(graph) and cheap.verify(graph)
    search = _Search(graph)
    assert search.admit(_Proof(expensive, expensive.cost()), target)
    assert not search.admit(_Proof(cheap, cheap.cost()), target)
    assert search.found == [cheap]
    assert search.classes[target][-1].cost == 0


def test_free_composition_has_more_than_four_cost_three_classes() -> None:
    actual = derive(*CURATED["napkin"], bound=3, limit=5, max_expansions=3000)
    graph = ADMG.parse(CURATED["napkin"].graph)
    complete = HPFalsifier(str(graph.complete()))
    assert len(actual) == 5
    assert all(d.cost() == 3 and d.verify(graph) for d in actual)
    assert all(
        not complete.check(d.formula, previous.formula)
        for index, d in enumerate(actual)
        for previous in actual[:index]
    )


@pytest.mark.parametrize("engine", [derive, astar])
def test_expansion_guard_can_be_explicitly_disabled_for_a_finite_search(
    engine: Callable,
) -> None:
    actual = engine(
        "X -> Y",
        "X",
        "Y",
        bound=1,
        costs={"transport": 1},
        max_expansions=None,
    )
    assert len(actual) == 1
    assert actual[0].cost({"transport": 1}) == 1
