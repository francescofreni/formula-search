"""History-based tie ordering must stay optional, fresh and cost-secondary."""

from dataclasses import replace
from heapq import heappop

import pytest

from formula_search.astar import SearchStats, _Plan, _Search, derive
from formula_search.docalculus import Query
from formula_search.evaluator import Archive, Task
from formula_search.expression import Product, Term
from formula_search.graph import ADMG
from formula_search.heuristics import SearchState, less_explored, new_terms


def query(name):
    return Query(frozenset({name}))


def test_term_novelty_is_fractional_and_distinct():
    a, b = Term(frozenset({"A"})), Term(frozenset({"B"}))
    state = SearchState(query("A"), 0, (), a, discovered_terms=frozenset({a}))
    score = new_terms()
    assert score(state) == 0
    assert score(replace(state, expression=b)) == -1
    assert score(replace(state, expression=Product((a, b)))) == -0.5
    assert score(replace(state, expression=Product((a, a, b)))) == -0.5
    assert new_terms(weight=2)(replace(state, expression=b)) == -2


def test_less_explored_counts_next_query_not_tree_size():
    a, b = query("A"), query("B")
    state = SearchState(
        a, 0, (a, b), Term(a.outcomes), expansion_counts={a: 4, b: 1}
    )
    score = less_explored()
    assert score(state) == 4
    assert score(replace(state, pending=(b, a))) == 1
    assert score(replace(state, pending=())) == 0


@pytest.mark.parametrize("factory", [new_terms, less_explored])
@pytest.mark.parametrize("weight", [-1, float("inf"), float("nan")])
def test_invalid_weights(factory, weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        factory(weight=weight)


def test_queued_terms_are_rescored_after_discovery():
    search = _Search(ADMG.parse("A -> B"), tiebreaker=new_terms())
    a, b = _Plan(query("A")), _Plan(query("B"))
    frontier = [
        (0.0, search.tie_score(a, 0), 0, 0, a),
        (0.0, search.tie_score(b, 0), 1, 0, b),
    ]
    search.discovered_terms.add(Term(a.query.outcomes))
    search.refresh_scores(frontier)
    assert heappop(frontier)[-1] == b


def test_queued_subproblems_refresh_without_overriding_primary_cost():
    search = _Search(ADMG.parse("A -> B"), tiebreaker=less_explored())
    a, b = _Plan(query("A")), _Plan(query("B"))
    frontier = [(0.0, 0.0, 0, 0, a), (0.0, 0.0, 1, 0, b)]
    search.record_expansion(a)
    search.refresh_scores(frontier)
    assert heappop(frontier)[-1] == b
    frontier = [(0.0, 0.0, 0, 0, a), (1.0, 0.0, 1, 1, b)]
    search.refresh_scores(frontier)
    assert heappop(frontier)[-1] == a


@pytest.mark.parametrize(
    "scorers",
    [[new_terms()], [less_explored()], [new_terms(), less_explored()]],
)
def test_novelty_keeps_bounded_classes_and_replayable_proofs(scorers):
    task = Task("Z -> X; X -> Y", "X", "Y")
    reference = Archive(task)
    for proof in derive(*task, bound=1):
        reference.submit(proof.formula, proof)
    results = derive(*task, bound=1, tiebreaker=scorers)
    assert len(results) == len(reference.classes)
    for proof in results:
        assert proof.verify(ADMG.parse(task.graph))
        assert reference.submit(proof.formula, proof).outcome == "duplicate"
    # Reusing the same scorer objects never leaks one run's history into another.
    assert results == derive(*task, bound=1, tiebreaker=scorers)


def test_dynamic_refresh_does_not_spend_extra_pop_budget(monkeypatch):
    import formula_search.astar as module

    actual = 0
    pop = module.heappop

    def count(queue):
        nonlocal actual
        actual += 1
        return pop(queue)

    monkeypatch.setattr(module, "heappop", count)
    stats = SearchStats()
    derive(
        "Z -> X; X -> Y",
        "X",
        "Y",
        bound=None,
        max_pops=30,
        tiebreaker=[new_terms(), less_explored()],
        stats=stats,
    )
    assert actual == stats.queue_pops == 30
    assert stats.stop_reason == "pop_limit"


def test_custom_priority_replaces_cost_order_and_refreshes():
    search = _Search(ADMG.parse("A -> B"), priority=new_terms())
    a, b = _Plan(query("A")), _Plan(query("B"))
    frontier = [
        (search.primary_score(a, 0), 0.0, 0, 0, a),
        (search.primary_score(b, 5), 0.0, 1, 5, b),
    ]
    search.discovered_terms.add(Term(a.query.outcomes))
    search.refresh_scores(frontier)
    assert heappop(frontier)[-1] == b  # Novelty may select a costlier state.


def test_custom_priority_receives_estimate_and_read_only_history():
    search = _Search(
        ADMG.parse("A -> B"),
        heuristic=lambda q: 2,
        priority=lambda s: s.cost + s.estimated_remaining_cost,
    )
    plan = _Plan(query("A"))
    assert search.primary_score(plan, 3) == 5
    state = search.state(plan, 3)
    with pytest.raises(TypeError):
        state.expansion_counts[query("A")] = 9
    search.record_expansion(plan)
    assert not state.expansion_counts  # Snapshot, not a live mutable view.


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_custom_priority_must_be_finite(value):
    with pytest.raises(ValueError, match="Priority scores must be finite"):
        derive("X -> Y", "X", "Y", bound=1, priority=lambda s: value)


def test_priority_search_preserves_rules_bound_and_exact_pop_cap(monkeypatch):
    import formula_search.astar as module

    pop = module.heappop
    actual = 0

    def count(queue):
        nonlocal actual
        actual += 1
        return pop(queue)

    monkeypatch.setattr(module, "heappop", count)
    task = Task("Z -> X; X -> Y", "X", "Y")
    stats = SearchStats()
    proofs = derive(
        *task, bound=1, max_pops=30, priority=new_terms(), stats=stats
    )
    assert stats.queue_pops == actual <= 30
    assert all(p.verify(ADMG.parse(task.graph)) for p in proofs)
    # An exhaustive bounded custom-priority run keeps the same class set.
    reference = Archive(task)
    for proof in derive(*task, bound=1):
        reference.submit(proof.formula, proof)
    found = derive(*task, bound=1, priority=new_terms())
    assert len(found) == len(reference.classes)
    assert all(
        reference.submit(p.formula, p).outcome == "duplicate" for p in found
    )
