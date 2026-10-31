"""Composable cost bounds, expression-size and novelty tie-breakers for A*.

Cost heuristics take a Query and lower-bound the remaining probability-rule
cost, including free do-calculus moves. Their maximum remains admissible;
their sum need not be. Factories must use the same rule costs as the search.

Tie-breakers take a SearchState and score its expression tree. An unfinished
state uses kernel placeholders for unresolved queries and unconditional
weights for unresolved transport choices. These expressions only estimate
size; they are not identifying formulas. A completed candidate uses its
actual simplified expression. Size scores guide ties, without guaranteeing
that the globally shortest equation is returned first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from math import isfinite
from types import MappingProxyType

from .derivation import Derivation, _rule_costs, rules
from .docalculus import Query, equivalent_queries
from .expression import ICD, Expression, Product, Sum, Term
from .graph import ADMG

Heuristic = Callable[[Query], float]
Tiebreaker = Callable[["SearchState"], float]


@dataclass(frozen=True)
class SearchState:
    """Read-only input to a tie-breaker.

    ``cost`` is committed probability-rule cost, and ``pending`` lists all
    outstanding premises. ``expression`` estimates an unfinished proof's
    size; only when ``derivation`` is present is it an actual formula.
    For history-dependent scorers, ``discovered_terms`` contains terms from
    classes found by this run; ``expansion_counts`` records actual expansions
    of each next unresolved query. The engine supplies read-only snapshots.
    """

    query: Query
    cost: int
    pending: tuple[Query, ...]
    expression: Expression
    derivation: Derivation | None = None
    discovered_terms: frozenset[Term] = frozenset()
    expansion_counts: Mapping[Query, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    estimated_remaining_cost: float = 0.0


def zero(query: Query) -> float:
    """The zero lower bound: uniform-cost search."""
    return 0.0


def max_of(*heuristics: Heuristic) -> Heuristic:
    """Take the strongest supplied lower bound, or zero if none are given.

    Each heuristic estimates the cost of solving the SAME query, rather
    than the cost of a separate piece of work. If two admissible heuristics
    return 2 and 5, we know the remaining cost is at least 5. We do not know
    that it is at least 7: both estimates may account for the same rule
    applications. If the true minimum cost is 5, adding them overestimates
    it, whereas taking their maximum is still a valid lower bound.

    In general, if every h_i(q) <= optimal_cost(q), their maximum also
    satisfies that inequality. This preserves A*'s minimum-cost discovery
    guarantee when the individual heuristics are admissible. Taking a max
    does not make an inadmissible heuristic admissible; the checks below
    only validate finite, nonnegative values.

    Actual rule costs ARE added along a proof, and the search adds bounds
    for separate required premises. Here we combine alternative estimates
    for one query. Summing those estimates would require a separate proof
    that they charge disjoint portions of the remaining cost.
    """

    def combined(query: Query) -> float:
        values = [heuristic(query) for heuristic in heuristics]
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError(
                "Heuristic estimates must be finite and nonnegative"
            )
        return max(values, default=0.0)

    return combined


def free_proof_bound(
    graph: str | ADMG, *, costs: Mapping[str, int] | None = None
) -> Heuristic:
    """Return zero for free proofs, otherwise the cheapest rule cost."""
    return lookahead_bound(graph, depth=0, costs=costs)


def lookahead_bound(
    graph: str | ADMG,
    *,
    depth: int = 2,
    costs: Mapping[str, int] | None = None,
) -> Heuristic:
    """Build an admissible bound by looking ahead probability-rule steps.

    At depth zero, queries with no free proof need at least the cheapest
    rule. Each further level minimizes rule cost plus bounds for all its
    premises, across every do-calculus-equivalent query. A dead end gets a
    finite lower bound rather than infinity. Results are cached by query
    and depth; small depths are recommended.
    """
    if type(depth) is not int or depth < 0:
        raise ValueError("Depth must be a nonnegative integer")
    admg = ADMG.parse(graph) if isinstance(graph, str) else graph
    rule_costs = _rule_costs(costs)
    cheapest = min(rule_costs.values())

    @cache
    def component(query: Query) -> tuple[Query, ...]:
        return tuple(equivalent_queries(admg, query))

    @cache
    def estimate(query: Query, remaining: int) -> float:
        members = component(query)
        if any(not member.treatments for member in members):
            return 0.0
        if remaining == 0:
            return float(cheapest)
        return min(
            (
                rule_costs[rule]
                + sum(estimate(premise, remaining - 1) for premise in premises)
                for member in members
                for rule, premises in rules(admg, member)
            ),
            default=float(cheapest),
        )

    def heuristic(query: Query) -> float:
        return estimate(query, depth)

    return heuristic


def _nodes(expression: Expression) -> tuple[Expression, ...]:
    """Walk expression nodes independently of rendered variable names."""
    if isinstance(expression, Term):
        return (expression,)
    children = (
        expression.factors
        if isinstance(expression, Product)
        else (expression.body,)
    )
    return (expression,) + tuple(
        node for child in children for node in _nodes(child)
    )


def _weighted(
    measure: Callable[[Expression], int], weight: float
) -> Tiebreaker:
    if not isfinite(weight) or weight < 0:
        raise ValueError("Tie-breaker weights must be finite and nonnegative")

    def score(state: SearchState) -> float:
        return weight * measure(state.expression)

    return score


def expression_size(*, weight: float = 1) -> Tiebreaker:
    """Count all nodes in the simplified or provisional expression tree."""
    return _weighted(lambda expression: len(_nodes(expression)), weight)


def term_count(*, weight: float = 1) -> Tiebreaker:
    """Count probability terms, including placeholders in unfinished proofs."""
    return _weighted(
        lambda expression: sum(
            isinstance(n, Term) for n in _nodes(expression)
        ),
        weight,
    )


def summation_count(*, weight: float = 1) -> Tiebreaker:
    """Count summation nodes, rather than the number of summed variables."""
    return _weighted(
        lambda expression: sum(isinstance(n, Sum) for n in _nodes(expression)),
        weight,
    )


def conditioning_count(*, weight: float = 1) -> Tiebreaker:
    """Count internal conditional-division nodes."""
    return _weighted(
        lambda expression: sum(isinstance(n, ICD) for n in _nodes(expression)),
        weight,
    )


def conditioning_size(*, weight: float = 1) -> Tiebreaker:
    """Count the largest input set of a probability term or ICD node."""
    return _weighted(
        lambda expression: max(
            (
                len(n.inputs)
                for n in _nodes(expression)
                if isinstance(n, (Term, ICD))
            ),
            default=0,
        ),
        weight,
    )


@dataclass(frozen=True)
class _HistoryTiebreaker:
    """Marker for scores that must refresh as search history changes."""

    score: Tiebreaker

    def __call__(self, state: SearchState) -> float:
        return self.score(state)


def _terms(expression: Expression) -> frozenset[Term]:
    return frozenset(n for n in _nodes(expression) if isinstance(n, Term))


def new_terms(*, weight: float = 1) -> Tiebreaker:
    """Favor a larger fraction of terms absent from discovered classes.

    Count distinct probability terms, not repeated occurrences. Using a
    fraction avoids rewarding a formula merely for having more terms.
    Partial expressions include provisional terms: this is a novelty hint,
    not an equivalence test or a remaining-cost lower bound.
    """
    if not isfinite(weight) or weight < 0:
        raise ValueError("Tie-breaker weights must be finite and nonnegative")

    def score(state: SearchState) -> float:
        terms = _terms(state.expression)
        return (
            -weight * len(terms - state.discovered_terms) / max(1, len(terms))
        )

    return _HistoryTiebreaker(score)


def less_explored(*, weight: float = 1) -> Tiebreaker:
    """Favor a next subproblem with fewer previous expansions.

    Count actual expansions of the first pending Query, including repeated
    appearances in different proof trees. Completed candidates score zero.
    This discourages revisiting busy regions without pruning any proofs.
    """
    if not isfinite(weight) or weight < 0:
        raise ValueError("Tie-breaker weights must be finite and nonnegative")

    def score(state: SearchState) -> float:
        return (
            weight * state.expansion_counts.get(state.pending[0], 0)
            if state.pending
            else 0.0
        )

    return _HistoryTiebreaker(score)
