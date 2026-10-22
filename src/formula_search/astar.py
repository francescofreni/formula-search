"""A* search over unfinished derivations of identifying formulas.

A state is a proof tree with unresolved queries at its leaves. Expanding
one leaf chooses an equivalent query and either an observational base term
or a rule of probability with new premises. Do-calculus paths are folded
into these choices, so free steps cannot create an infinite search branch.

The frontier is ordered by the cost of the chosen probability rules plus
an estimate for every unresolved premise. The default estimate is zero,
which gives uniform-cost search. An admissible heuristic lower-bounds the
cheapest derivation of each query, including free do-calculus moves. With
such a heuristic, the first derivation retained for each class has minimum
cost. The heuristic need not be consistent, since partial trees are not
merged or closed on their first visit.

Completed trees are turned into replayable Derivations, including each
weight allowed by do-calculus transport. HiProF on the complete DAG keeps
one representative per formula class; validity comes from the proof rules.
Unlike the cost-layer engine, this engine does not deduplicate subproofs.
Its frontier can grow rapidly: start with small graphs and bounds. An empty
bounded search does not establish non-identifiability.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from heapq import heappop, heappush
from itertools import count
from itertools import product as cartesian_product
from math import isfinite

from hiprof import HPFalsifier

from .baselines.id import parse_query
from .derivation import Derivation, Rule, _rule_costs, _transport, rules
from .docalculus import Query, equivalent_queries
from .expression import Expression, Term, condition, marginal, product
from .graph import ADMG
from .heuristics import Heuristic, SearchState, Tiebreaker


@dataclass(frozen=True)
class _Plan:
    """An unresolved query, or a chosen rule with unfinished premises."""

    query: Query
    member: Query | None = None
    rule: Rule | None = None
    premises: tuple[_Plan, ...] = ()
    path: tuple[Query, ...] = ()

    @cached_property
    def pending(self) -> tuple[Query, ...]:
        """The unresolved queries, in deterministic expansion order."""
        if self.rule is None:
            return (self.query,)
        return tuple(q for p in self.premises for q in p.pending)

    @cached_property
    def expression(self) -> Expression:
        """Estimate size with kernel placeholders and unconditional weights."""
        if self.rule is None:
            return Term(self.query.outcomes, self.query.inputs)
        assert self.member is not None
        if self.rule == "base":
            expression: Expression = Term(
                self.member.outcomes, self.member.conditions
            )
        elif self.rule == "chain rule":
            expression = product(p.expression for p in self.premises)
        else:
            (premise,) = self.premises
            expression = premise.expression
            if self.rule == "marginalization":
                expression = marginal(
                    expression, premise.query.outcomes - self.member.outcomes
                )
            else:
                expression = condition(
                    expression,
                    self.member.conditions - premise.query.conditions,
                )
        extra = expression.inputs - self.query.inputs
        if extra:
            expression = marginal(product([Term(extra), expression]), extra)
        return expression

    def derivations(self) -> Iterator[Derivation]:
        """Complete the proof, enumerating its allowed transport weights."""
        if self.rule is None or self.member is None:
            raise ValueError("Cannot render an unfinished derivation")
        for premises in cartesian_product(
            *(premise.derivations() for premise in self.premises)
        ):
            direct = Derivation(self.member, self.rule, premises)
            yield from _transport(direct, self.query, self.path)


def derive(
    graph: str,
    treatments: str | Iterable[str],
    outcomes: str | Iterable[str],
    bound: int,
    *,
    costs: Mapping[str, int] | None = None,
    heuristic: Heuristic | None = None,
    tiebreaker: Tiebreaker | Iterable[Tiebreaker] | None = None,
    limit: int | None = None,
) -> list[Derivation]:
    """Search for formulas for ``p(outcomes | do(treatments))`` with A*.

    :param bound: Maximum total probability-rule cost, a nonnegative integer.
    :param costs: Positive integer costs by probability-rule name. Omitted
        rules cost one; base terms and do-calculus steps are free.
    :param heuristic: A finite nonnegative estimate of the remaining cost
        for a Query, defaulting to zero. For minimum-cost discovery order,
        this must lower-bound every proof of that query, including proofs
        through equivalent queries. Queries with free proofs need zero.
        Estimates order the frontier; only actual cost prunes the bound.
    :param tiebreaker: A SearchState scorer, or an iterable of scorers whose
        scores are summed. Lower scores break equal A* priorities. Unfinished
        expressions are size estimates; completed candidates use their actual
        expressions. Scores must be finite. None preserves insertion order.
    :param limit: Stop after this many distinct formula classes, or search
        the whole bound if None. Must be a positive integer when supplied.
    :returns: One derivation per discovered class. With an admissible
        heuristic, discovery is in nondecreasing minimum cost; ties use the
        supplied scorers, then deterministic insertion order. An arbitrary heuristic may discover
        a more expensive representative first.
    :raises ValueError: For invalid graph/query arguments, costs, bounds,
        limits, heuristic estimates, or tie-breaker scores.
    """
    if type(bound) is not int or bound < 0:
        raise ValueError("Bound must be a nonnegative integer")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Limit must be a positive integer")
    admg = ADMG.parse(graph)
    x, y, _ = parse_query(admg, treatments, outcomes)
    search = _Search(
        admg, costs=costs, heuristic=heuristic, tiebreaker=tiebreaker
    )
    found = []
    for derivation in search.run(Query(y, x), bound):
        found.append(derivation)
        if limit is not None and len(found) == limit:
            break
    return found


class _Search:
    """A priority queue of partial proofs, scored by committed cost plus h."""

    def __init__(
        self,
        graph: ADMG,
        *,
        costs: Mapping[str, int] | None = None,
        heuristic: Heuristic | None = None,
        tiebreaker: Tiebreaker | Iterable[Tiebreaker] | None = None,
    ) -> None:
        self.graph = graph
        self.costs = _rule_costs(costs)
        self.heuristic = heuristic
        self.tiebreakers = (
            ()
            if tiebreaker is None
            else (tiebreaker,) if callable(tiebreaker) else tuple(tiebreaker)
        )
        self.estimates: dict[Query, float] = {}
        self.components: dict[Query, dict[Query, tuple[Query, ...]]] = {}
        self.complete = HPFalsifier(str(graph.complete()))
        self.classes: list[str] = []

    def estimate(self, plan: _Plan) -> float:
        """Sum the estimates of all premises still needed by the proof."""
        for query in plan.pending:
            if query not in self.estimates:
                value = (
                    0.0 if self.heuristic is None else self.heuristic(query)
                )
                if not isfinite(value) or value < 0:
                    raise ValueError(
                        "Heuristic estimates must be finite and nonnegative"
                    )
                self.estimates[query] = value
        total = sum(self.estimates[q] for q in plan.pending)
        if not isfinite(total):
            raise ValueError("Total heuristic estimate must be finite")
        return total

    def tie_score(self, item: _Plan | Derivation, cost: int) -> float:
        """Score a public snapshot without exposing mutable search internals."""
        if not self.tiebreakers:
            return 0.0
        if isinstance(item, Derivation):
            state = SearchState(item.query, cost, (), item.expression, item)
        else:
            state = SearchState(
                item.query, cost, item.pending, item.expression
            )
        values = [score(state) for score in self.tiebreakers]
        total = sum(values)
        if any(not isfinite(value) for value in values) or not isfinite(total):
            raise ValueError("Tie-breaker scores must be finite")
        return total

    def run(self, query: Query, bound: int) -> Iterator[Derivation]:
        """Expand by f, then tie-breaker score, then stable insertion order."""
        serial = count()
        initial = _Plan(query)
        frontier: list[tuple[float, float, int, int, _Plan | Derivation]] = []

        def push(item: _Plan | Derivation, cost: int) -> None:
            estimate = self.estimate(item) if isinstance(item, _Plan) else 0.0
            heappush(
                frontier,
                (
                    cost + estimate,
                    self.tie_score(item, cost),
                    next(serial),
                    cost,
                    item,
                ),
            )

        push(initial, 0)
        while frontier:
            _, _, _, cost, item = heappop(frontier)
            if isinstance(item, Derivation):
                candidates: Iterable[Derivation] = (item,)
            elif not item.pending:
                if self.tiebreakers:
                    # Score each transport weight's actual expression before
                    # accepting a formula, rather than emitting a whole batch.
                    for derivation in item.derivations():
                        push(derivation, cost)
                    continue
                candidates = item.derivations()
            else:
                for added, expanded in self.expand(item, bound - cost):
                    push(expanded, cost + added)
                continue
            for derivation in candidates:
                if not any(
                    self.complete.check(formula, derivation.formula)
                    for formula in self.classes
                ):
                    self.classes.append(derivation.formula)
                    yield derivation

    def expand(
        self, plan: _Plan, budget: int | None
    ) -> Iterator[tuple[int, _Plan]]:
        """Resolve the first leaf; None offers every rule regardless of cost."""
        if plan.rule is not None:
            for index, premise in enumerate(plan.premises):
                if premise.pending:
                    for added, expanded in self.expand(premise, budget):
                        updated = (
                            plan.premises[:index]
                            + (expanded,)
                            + plan.premises[index + 1 :]
                        )
                        yield added, replace(plan, premises=updated)
                    return
            return
        query = plan.query
        if query not in self.components:
            self.components[query] = equivalent_queries(self.graph, query)
        for member, path in self.components[query].items():
            if not member.treatments:
                yield 0, _Plan(query, member, "base", path=path)
            for rule, premises in rules(self.graph, member):
                added = self.costs[rule]
                if budget is None or added <= budget:
                    yield added, _Plan(
                        query,
                        member,
                        rule,
                        tuple(_Plan(premise) for premise in premises),
                        path,
                    )
