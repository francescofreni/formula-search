"""Search for identifying formulas with replayable AND/OR derivations.

Probability rules are marginalization, conditioning, and chain rule. A
transport proves an equivalent query by a do-calculus path and averages out
extra inputs with a normalized observational weight. Transports are explicit
inferences and can compose, including when their cost is zero.

An agenda combines admitted subproof classes and transports them repeatedly.
HiProF compares classes on a complete DAG; the proof rules establish
validity in the task graph. Minimum-cost representatives can substitute in
all parent inferences, so class deduplication applies to subproofs as well
as target formulas. This is verification support relative to HiProF's
checks, not an exact symbolic equality decision procedure.

Probability steps default to cost one, transports to zero, and bases are
free. Zero-cost composition can generate infinitely many classes within a
finite cost bound. A class limit requests a prefix; an expansion-budget stop
raises SearchLimitExceeded with explicitly incomplete results. Positive
transport cost makes bounded proof enumeration finite, although it can
still be expensive. Empty bounded search is not a non-identifiability proof.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from heapq import heappop, heappush
from itertools import combinations, count, pairwise
from itertools import product as cartesian_product
from math import isfinite
from typing import Literal

from hiprof import HPFalsifier

from .baselines.id import parse_query
from .docalculus import Query, equivalent_queries, neighbours
from .expression import Expression, Term, condition, marginal, product, render
from .graph import ADMG

Rule = Literal[
    "base", "marginalization", "conditioning", "chain rule", "do-calculus"
]


_DEFAULT_COSTS = {
    "marginalization": 1,
    "conditioning": 1,
    "chain rule": 1,
    "transport": 0,
}


def _rule_costs(costs: Mapping[str, int] | None) -> dict[str, int]:
    """Validate positive probability costs and nonnegative transport cost."""
    result = _DEFAULT_COSTS.copy()
    if costs is not None:
        for rule, cost in costs.items():
            if rule not in result:
                raise ValueError(f"Unknown probability rule: {rule!r}")
            if type(cost) is not int or cost < (
                0 if rule == "transport" else 1
            ):
                raise ValueError(
                    f"Cost for {rule!r} must be a "
                    + ("nonnegative" if rule == "transport" else "positive")
                    + " integer"
                )
            result[rule] = cost
    return result


@dataclass(frozen=True)
class Derivation:
    """A derivation of ``query = expression``, as a tree of rules.

    :param query: The query derived.
    :param rule: The last rule. ``"base"`` derives a query without
        interventions as its base term. A rule of probability derives it from
        the queries of ``premises``, and ``"do-calculus"`` from the query of
        its one premise, reached by ``path``.
    :param premises: Derivations of the queries that the rule uses.
    :param path: For ``"do-calculus"``, the queries from ``query`` to that
        of the premise, each one rule of do-calculus from the last.
    :param weight: For ``"do-calculus"``, the inputs of the weight that
        averages out the inputs of the premise that the query lacks.
    """

    query: Query
    rule: Rule
    premises: tuple[Derivation, ...] = ()
    path: tuple[Query, ...] = ()
    weight: frozenset[str] = frozenset()

    @cached_property
    def expression(self) -> Expression:
        """The observational expression derived for the query."""
        query = self.query
        if self.rule == "base":
            return Term(query.outcomes, query.conditions)
        if self.rule == "chain rule":
            return product(premise.expression for premise in self.premises)

        (premise,) = self.premises
        expression = premise.expression
        if self.rule == "marginalization":
            return marginal(
                expression, premise.query.outcomes - query.outcomes
            )
        if self.rule == "conditioning":
            given = query.conditions - premise.query.conditions
            return condition(expression, given)
        if extra := expression.inputs - query.inputs:
            weighted = product([Term(extra, self.weight), expression])
            return marginal(weighted, extra)
        return expression

    @cached_property
    def formula(self) -> str:
        """The expression in hiprof's syntax."""
        return render(self.expression, reserved=self.query.inputs)

    def cost(self, costs: Mapping[str, int] | None = None) -> int:
        """Sum rule costs; each do-calculus transport wrapper is one step.

        A transport may contain a multi-edge do-calculus path. Its cost is
        per wrapper, not per edge. Base terms cost zero.
        """
        values = _rule_costs(costs)

        def total(d: Derivation) -> int:
            own = (
                0
                if d.rule == "base"
                else values["transport" if d.rule == "do-calculus" else d.rule]
            )
            return own + sum(total(p) for p in d.premises)

        return total(self)

    def verify(self, graph: ADMG) -> bool:
        """Return whether every step of the derivation is valid in ``graph``.

        Each step must apply its rule to the queries of its premises, with
        the side conditions of do-calculus tested again along its path, and
        derive an expression of the type of its query.
        """
        queries = tuple(premise.query for premise in self.premises)
        if self.rule == "base":
            valid = not self.query.treatments
        elif self.rule == "do-calculus":
            valid = (
                self.path[:1] == (self.query,)
                and self.path[-1:] == queries
                and all(
                    after in neighbours(graph, before)
                    for before, after in pairwise(self.path)
                )
            )
        else:
            valid = (self.rule, queries) in rules(graph, self.query)
        return (
            valid
            and self.expression.outputs == self.query.outcomes
            and self.expression.inputs <= self.query.inputs
            and all(premise.verify(graph) for premise in self.premises)
        )


def derive(
    graph: str,
    treatments: str | Iterable[str],
    outcomes: str | Iterable[str],
    bound: int,
    *,
    costs: Mapping[str, int] | None = None,
    limit: int | None = None,
    max_expansions: int | None = 10000,
) -> list[Derivation]:
    """Derive formulas for ``p(outcomes | do(treatments))``.

    :param graph: ADMG in hiprof syntax.
    :param treatments: Treatment variable, or variables.
    :param outcomes: Outcome variable, or variables.
    :param bound: Maximum total rule cost, a nonnegative integer.
    :param costs: Positive integer probability-rule costs, default one.
        ``transport`` is nonnegative, default zero. It is charged once per
        transport wrapper, not per edge in its do-calculus path. Bases are
        free. Composing transports adds their costs in the usual way.
    :param limit: Stop after this many target classes; None requests all.
    :param max_expansions: Budget for cataloged queries and popped agenda
        items, default 10000. None disables this guard. Free composition
        can make an exhaustive finite-bound search infinite.
    :returns: Minimum-cost representatives in nondecreasing cost order.
        A supplied class limit requests only a prefix of the result stream.
    :raises ValueError: For invalid graph, query, cost, or budget arguments.
    :raises SearchLimitExceeded: On expansion-budget exhaustion, with
        ``derivations`` found so far and ``expanded`` count. Partial results
        are never silently reported as an exhausted bounded search.
    """
    admg = ADMG.parse(graph)
    x, y, _ = parse_query(admg, treatments, outcomes)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Limit must be a positive integer")
    search = _Search(admg, costs=costs, max_expansions=max_expansions)
    found = []
    for derivation in search.run(Query(y, x), bound):
        found.append(derivation)
        if limit is not None and len(found) == limit:
            break
    return search.found


def rules(
    graph: ADMG,
    query: Query,
) -> Iterator[tuple[Rule, tuple[Query, ...]]]:
    """Yield the rules of probability that derive ``query``, and premises.

    Marginalization and conditioning act on one variable, which suffices, as
    they compose; the chain rule splits the outcomes in every order.
    """
    a, x, b = query.outcomes, query.treatments, query.conditions
    for z in sorted(graph.nodes - a - x - b):
        yield "marginalization", (Query(a | {z}, x, b),)
    for c in sorted(b):
        yield "conditioning", (Query(a | {c}, x, b - {c}),)
    for size in range(1, len(a)):
        for first in map(frozenset, combinations(sorted(a), size)):
            yield "chain rule", (
                Query(first, x, b),
                Query(a - first, x, b | first),
            )


def _transport(
    derivation: Derivation,
    query: Query,
    path: tuple[Query, ...],
) -> Iterator[Derivation]:
    """Yield derivations of ``query`` from one of the query ending ``path``.

    Each set of inputs of ``query`` gives a weight for the inputs of the
    derivation that ``query`` lacks; without such inputs, no weight is
    needed.
    """
    if len(path) == 1:
        yield derivation
        return
    inputs = sorted(query.inputs)
    extra = derivation.expression.inputs - query.inputs
    for size in range(len(inputs) + 1 if extra else 1):
        for weight in combinations(inputs, size):
            yield Derivation(
                query, "do-calculus", (derivation,), path, frozenset(weight)
            )


class SearchLimitExceeded(RuntimeError):
    """The expansion budget was reached; partial results are not exhaustive."""

    def __init__(self, derivations: list[Derivation], expanded: int) -> None:
        self.derivations = list(derivations)
        self.expanded = expanded
        super().__init__(
            f"Search reached {expanded} expansions; "
            f"{len(derivations)} classes found so far. Results are incomplete."
        )


@dataclass(frozen=True)
class _Inference:
    """A probability rule or one transport between equivalent queries."""

    query: Query
    rule: Rule
    premises: tuple[Query, ...]
    path: tuple[Query, ...] = ()


@dataclass(frozen=True)
class _Proof:
    derivation: Derivation
    cost: int


@dataclass(frozen=True)
class _Partial:
    """An inference whose first premises have been proved."""

    inference: int
    proofs: tuple[int, ...] = ()


Score = Callable[
    [Query, int, tuple[Query, ...], Expression, Derivation | None], float
]


class _Search:
    """Infer distinct proof classes with an agenda and premise wait lists.

    Backward reachability builds a finite query/rule graph within the cost
    bound. Forward inference then admits observational bases, combines
    proved premises, and transports admitted proofs repeatedly. Zero-cost
    cycles with no new formula classes stop; cycles generating new classes
    can run forever, so the expansion budget is checked explicitly.

    The agenda uses inside cost plus a lower bound on the outside context
    needed to reach the target. Outside bounds are shortest paths through
    backward rules, charging the supplied inside heuristic for siblings.
    An admissible inside heuristic therefore preserves minimum-cost class
    discovery, even when inconsistent. Queries are never closed merely
    because one of their formula classes has been proved.
    """

    def __init__(
        self,
        graph: ADMG,
        *,
        costs: Mapping[str, int] | None = None,
        heuristic: Callable[[Query], float] | None = None,
        score: Score | None = None,
        max_expansions: int | None = 10000,
    ) -> None:
        if max_expansions is not None and (
            type(max_expansions) is not int or max_expansions <= 0
        ):
            raise ValueError(
                "Max expansions must be a positive integer or None"
            )
        self.graph = graph
        self.costs = _rule_costs(costs)
        self.heuristic = heuristic
        self.score = score
        self.max_expansions = max_expansions
        self.expanded = 0
        self.found: list[Derivation] = []
        self.complete = HPFalsifier(str(graph.complete()))
        self.components: dict[Query, dict[Query, tuple[Query, ...]]] = {}
        self.classes: dict[Query, list[_Proof]] = defaultdict(list)
        self.inferences: list[_Inference] = []
        self.context: dict[Query, int] = {}
        self.outside: dict[Query, float] = {}
        self.estimates: dict[Query, float] = {}
        self.frontier: list[tuple[float, float, int, _Proof | _Partial]] = []
        self.scheduled: set[_Partial] = set()
        self.waiting: dict[Query, list[_Partial]] = defaultdict(list)
        self.serial = count()
        self.bound = 0

    def tick(self) -> None:
        """Count catalog/agenda expansions, failing rather than truncating."""
        if (
            self.max_expansions is not None
            and self.expanded >= self.max_expansions
        ):
            raise SearchLimitExceeded(self.found, self.expanded)
        self.expanded += 1

    def estimate(self, query: Query) -> float:
        if query not in self.estimates:
            value = 0.0 if self.heuristic is None else self.heuristic(query)
            if not isfinite(value) or value < 0:
                raise ValueError(
                    "Heuristic estimates must be finite and nonnegative"
                )
            self.estimates[query] = value
        return self.estimates[query]

    def rule_cost(self, rule: Rule) -> int:
        return self.costs["transport" if rule == "do-calculus" else rule]

    def catalog(self, query: Query, bound: int) -> None:
        """Find queries reachable within a relaxed outside cost budget."""
        serial = count()
        pending = [(0, next(serial), query)]
        while pending:
            distance, _, current = heappop(pending)
            if current in self.context:
                continue
            self.tick()
            self.context[current] = distance
            self.components[current] = equivalent_queries(self.graph, current)
            choices = [
                _Inference(current, rule, premises)
                for rule, premises in rules(self.graph, current)
            ] + [
                _Inference(current, "do-calculus", (member,), path)
                for member, path in self.components[current].items()
                if member != current
            ]
            for inference in choices:
                cost = self.rule_cost(inference.rule)
                if distance + cost > bound:
                    continue
                self.inferences.append(inference)
                for premise in inference.premises:
                    if premise not in self.context:
                        heappush(
                            pending, (distance + cost, next(serial), premise)
                        )

    def outside_bounds(self, target: Query) -> None:
        """Lower-bound each subproof's cheapest context up to the target."""
        by_head: dict[Query, list[_Inference]] = defaultdict(list)
        for inference in self.inferences:
            by_head[inference.query].append(inference)
        serial = count()
        pending = [(0.0, next(serial), target)]
        for query in self.context:
            self.estimate(query)
        while pending:
            distance, _, current = heappop(pending)
            if current in self.outside:
                continue
            self.outside[current] = distance
            for inference in by_head[current]:
                for index, premise in enumerate(inference.premises):
                    siblings = sum(
                        self.estimate(q)
                        for i, q in enumerate(inference.premises)
                        if i != index
                    )
                    value = (
                        distance + self.rule_cost(inference.rule) + siblings
                    )
                    if not isfinite(value):
                        raise ValueError(
                            "Total heuristic estimate must be finite"
                        )
                    if premise not in self.outside:
                        heappush(pending, (value, next(serial), premise))

    def partial_state(
        self, partial: _Partial
    ) -> tuple[Query, int, tuple[Query, ...], Expression]:
        inference = self.inferences[partial.inference]
        selected = [
            self.classes[q][index]
            for q, index in zip(
                inference.premises, partial.proofs, strict=False
            )
        ]
        cost = self.rule_cost(inference.rule) + sum(p.cost for p in selected)
        pending = inference.premises[len(selected) :]
        expressions = [p.derivation.expression for p in selected] + [
            Term(q.outcomes, q.inputs) for q in pending
        ]
        if inference.rule == "chain rule":
            expression = product(expressions)
        else:
            (expression,) = expressions
            (premise,) = inference.premises
            if inference.rule == "marginalization":
                expression = marginal(
                    expression, premise.outcomes - inference.query.outcomes
                )
            elif inference.rule == "conditioning":
                expression = condition(
                    expression, inference.query.conditions - premise.conditions
                )
            elif extra := expression.inputs - inference.query.inputs:
                expression = marginal(
                    product([Term(extra), expression]), extra
                )
        return inference.query, cost, pending, expression

    def push(self, item: _Proof | _Partial) -> None:
        if isinstance(item, _Proof):
            d: Derivation | None = item.derivation
            head = item.derivation.query
            cost = item.cost
            pending: tuple[Query, ...] = ()
            expression = item.derivation.expression
        else:
            if item in self.scheduled:
                return
            self.scheduled.add(item)
            head, cost, pending, expression = self.partial_state(item)
            d = None
        if cost + self.context[head] > self.bound:
            return
        estimate = sum(self.estimate(q) for q in pending)
        priority = cost + estimate + self.outside[head]
        if not isfinite(priority):
            raise ValueError("Total heuristic estimate must be finite")
        score = (
            0.0
            if self.score is None
            else self.score(head, cost, pending, expression, d)
        )
        if not isfinite(score):
            raise ValueError("Tie-breaker scores must be finite")
        heappush(self.frontier, (priority, score, next(self.serial), item))

    def advance(self, partial: _Partial) -> None:
        """Wait for the next premise, or finish a fully proved inference."""
        inference = self.inferences[partial.inference]
        if len(partial.proofs) < len(inference.premises):
            premise = inference.premises[len(partial.proofs)]
            self.waiting[premise].append(partial)
            for index in range(len(self.classes[premise])):
                self.push(
                    _Partial(partial.inference, partial.proofs + (index,))
                )
            return
        proofs = tuple(
            self.classes[q][index]
            for q, index in zip(
                inference.premises, partial.proofs, strict=True
            )
        )
        cost = self.rule_cost(inference.rule) + sum(p.cost for p in proofs)
        if inference.rule == "do-calculus":
            candidates = _transport(
                proofs[0].derivation, inference.query, inference.path
            )
        else:
            candidates = iter(
                [
                    Derivation(
                        inference.query,
                        inference.rule,
                        tuple(p.derivation for p in proofs),
                    )
                ]
            )
        for d in candidates:
            self.push(_Proof(d, cost))

    def admit(self, proof: _Proof, target: Query) -> bool:
        """Propagate new or cheaper classes; reopen on improved cost.

        Admissible estimates settle each class at minimum cost. Arbitrary
        estimates can admit a costlier proof first; retaining improvements
        is necessary to avoid losing combinations within the actual bound.
        """
        d = proof.derivation
        known = self.classes[d.query]
        matches = [
            p
            for p in known
            if self.complete.check(p.derivation.formula, d.formula)
        ]
        if any(p.cost <= proof.cost for p in matches):
            return False
        index = len(known)
        known.append(proof)
        for partial in self.waiting[d.query]:
            self.push(_Partial(partial.inference, partial.proofs + (index,)))
        if d.query != target:
            return False
        if matches:
            index = next(
                i
                for i, previous in enumerate(self.found)
                if any(previous is p.derivation for p in matches)
            )
            self.found[index] = d
            return False
        self.found.append(d)
        return True

    def run(self, query: Query, bound: int) -> Iterator[Derivation]:
        """Yield new classes until the agenda exhausts or reaches its budget."""
        if type(bound) is not int or bound < 0:
            raise ValueError("Bound must be a nonnegative integer")
        self.bound = bound
        self.catalog(query, bound)
        self.outside_bounds(query)
        for current in self.context:
            if not current.treatments:
                self.push(_Proof(Derivation(current, "base"), 0))
        for index in range(len(self.inferences)):
            self.push(_Partial(index))
        while self.frontier:
            self.tick()
            _, _, _, item = heappop(self.frontier)
            if isinstance(item, _Partial):
                self.advance(item)
            elif self.admit(item, query):
                yield item.derivation

    def layer(self, query: Query, cost: int) -> list[Derivation]:
        """Debug helper: classes whose cheapest proof has exactly this cost."""
        search = _Search(
            self.graph, costs=self.costs, max_expansions=self.max_expansions
        )
        return [
            d for d in search.run(query, cost) if d.cost(self.costs) == cost
        ]

    def direct(self, query: Query, cost: int) -> list[Derivation]:
        """Debug helper: probability steps with premises of the given cost."""
        if cost == 0:
            return [] if query.treatments else [Derivation(query, "base")]
        return [
            Derivation(query, rule, combination)
            for rule, premises in rules(self.graph, query)
            for remaining in (cost - self.costs[rule],)
            if remaining >= 0
            for costs in cartesian_product(
                range(remaining + 1), repeat=len(premises)
            )
            if sum(costs) == remaining
            for combination in cartesian_product(
                *map(self.layer, premises, costs)
            )
        ]
