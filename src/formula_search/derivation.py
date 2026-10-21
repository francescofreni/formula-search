"""Derivation search for identifying formulas.

The derivation graph of Yvernes et al. (2026) joins the queries
``p(a | do(x), b)`` that one rule of do-calculus proves equal. The search adds
three rules of probability, each of which writes a query in terms of others:

- marginalization, ``p(a | do(x), b) = sum_{z} { p(a, z | do(x), b) }``;
- conditioning, ``p(a | do(x), b, c) = icd_{c | b, x} { p(a, c | do(x), b) }``;
- the chain rule, ``p(a, c | do(x), b) = p(a | do(x), b) p(c | do(x), a, b)``.

As in HyperTree Proof Search (Lample et al., 2022), the queries form an
AND/OR graph: a query is derived by any one rule, and the chain rule needs
both of its factors. A query without interventions is derived as its base
term. Queries equal by do-calculus share their derivations, and the inputs
that a derivation has but its query lacks, as ``z`` in
``p(y | do(x)) = p(y | do(x, z))``, are averaged out with a weight
``p(z | c)``, one for each set ``c`` of inputs of the query.

The expressions derived are infinitely many (Yvernes et al., Section 6), but
the queries are finitely many. The search derives every query breadth-first
in total probability-rule cost, and keeps one derivation per class of
each query: two expressions are in one class if HiProF finds them equal on
the complete DAG, as in the evaluator, which checks only expressions with the
same :mod:`~formula_search.fingerprints`. This loses no class of the target,
since expressions of one class combine into expressions of one class.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations, pairwise
from itertools import product as cartesian_product
from typing import Literal

from hiprof import HPFalsifier

from .baselines.id import parse_query
from .docalculus import Query, equivalent_queries, neighbours
from .expression import Expression, Term, condition, marginal, product, render
from .fingerprints import Fingerprint, FixedModel
from .graph import ADMG

Rule = Literal[
    "base", "marginalization", "conditioning", "chain rule", "do-calculus"
]


_DEFAULT_COSTS = {
    "marginalization": 1,
    "conditioning": 1,
    "chain rule": 1,
}


def _rule_costs(costs: Mapping[str, int] | None) -> dict[str, int]:
    """Validate overrides and return the probability-rule costs."""
    result = _DEFAULT_COSTS.copy()
    if costs is not None:
        for rule, cost in costs.items():
            if rule not in result:
                raise ValueError(f"Unknown probability rule: {rule!r}")
            if type(cost) is not int or cost <= 0:
                raise ValueError(
                    f"Cost for {rule!r} must be a positive integer"
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
) -> list[Derivation]:
    """Derive formulas for ``p(outcomes | do(treatments))``.

    :param graph: Acyclic directed mixed graph in hiprof's syntax.
    :param treatments: Treatment variable, or variables.
    :param outcomes: Outcome variable, or variables.
    :param bound: The largest total probability-rule cost in a derivation.
    :param costs: Positive integer costs for marginalization, conditioning,
        and chain rule. Omitted rules cost one. Base terms and do-calculus
        steps remain free.
    :returns: One derivation for each class of formulas found, in the order
        of their total probability-rule cost.
    :raises ValueError: If the graph is invalid, the outcomes are empty, or
        the variables overlap or are not nodes of the graph, or rule costs
        have unknown names or are not positive integers.
    """
    admg = ADMG.parse(graph)
    x, y, _ = parse_query(admg, treatments, outcomes)
    search = _Search(admg, costs=costs)
    return [
        derivation
        for cost in range(bound + 1)
        for derivation in search.layer(Query(y, x), cost)
    ]


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


class _Search:
    """Derivations of the queries of a graph, by rules of probability."""

    def __init__(
        self, graph: ADMG, *, costs: Mapping[str, int] | None = None
    ) -> None:
        self.costs = _rule_costs(costs)
        self.graph = graph
        self.complete = HPFalsifier(str(graph.complete()))
        self.model = FixedModel(graph.nodes)
        self.layers: dict[tuple[Query, int], list[Derivation]] = {}
        self.direct_steps: dict[tuple[Query, int], list[Derivation]] = {}
        self.components: dict[Query, dict[Query, tuple[Query, ...]]] = {}
        self.classes: dict[Query, dict[Fingerprint, list[str]]] = defaultdict(
            dict
        )

    def layer(self, query: Query, cost: int) -> list[Derivation]:
        """Return derivations of ``query`` with total rule cost ``cost``.

        There is one for each class that no derivation with fewer reaches.
        """
        if (query, cost) not in self.layers:
            for fewer in range(cost):
                self.layer(query, fewer)
            if query not in self.components:
                self.components[query] = equivalent_queries(self.graph, query)
            candidates = (
                transported
                for member, path in self.components[query].items()
                for derivation in self.direct(member, cost)
                for transported in _transport(derivation, query, path)
            )
            self.layers[query, cost] = [
                derivation
                for derivation in candidates
                if self._new_class(derivation)
            ]
        return self.layers[query, cost]

    def direct(self, query: Query, cost: int) -> list[Derivation]:
        """Return derivations of ``query`` whose last rule applies to it."""
        if (query, cost) not in self.direct_steps:
            if cost == 0:
                base = [] if query.treatments else [Derivation(query, "base")]
                self.direct_steps[query, cost] = base
            else:  # the premises share the cost of all but this last rule
                self.direct_steps[query, cost] = [
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
        return self.direct_steps[query, cost]

    def _new_class(self, derivation: Derivation) -> bool:
        """Record the class of a derivation, and return whether it is new."""
        query, formula = derivation.query, derivation.formula
        fingerprint = self.model.fingerprint(
            formula, query.outcomes, query.inputs
        )
        group = self.classes[query].setdefault(fingerprint, [])
        if formula in group or any(
            self.complete.check(known, formula) for known in group
        ):
            return False
        group.append(formula)
        return True


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
