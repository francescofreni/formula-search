"""A* inference of identifying formulas, including composed transports.

The shared AND/OR agenda admits distinct subproof classes, combines them
with probability rules, and transports them between equivalent queries.
A query heuristic estimates its cheapest remaining proof (inside cost).
The engine uses these estimates for sibling premises to lower-bound each
subproof's context up to the target (outside cost). Agenda priority is
committed inside cost plus unresolved-premise bounds plus outside cost.
Admissible query estimates preserve minimum-cost discovery, including when
inconsistent. A zero heuristic gives a uniform-cost relaxation of contexts.

Tie-breakers score partial inferences with kernel placeholders, or completed
proofs with their actual simplified expressions. They guide equal priorities
without guaranteeing globally smallest formulas. HiProF on the complete DAG
supports class deduplication; validity comes from replayable proof rules.

Transport defaults to zero cost. Composition can produce infinitely many
classes at a finite cost bound. A class limit stops intentionally; an
expansion-budget stop raises SearchLimitExceeded with explicitly incomplete
results. Positive transport cost makes a finite cost bound exhaustive when
the agenda finishes. Large finite searches can still exhaust resources.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite

from .baselines.id import parse_query
from .derivation import Derivation, SearchLimitExceeded, _Search
from .docalculus import Query
from .expression import Expression
from .graph import ADMG
from .heuristics import Heuristic, SearchState, Tiebreaker

__all__ = ["SearchLimitExceeded", "derive"]


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
    max_expansions: int | None = 10000,
) -> list[Derivation]:
    """Search for formulas for ``p(outcomes | do(treatments))`` with A*.

    :param bound: Maximum total rule cost, a nonnegative integer.
    :param costs: Probability-rule costs are positive integers, default one.
        ``transport`` is a nonnegative integer, default zero, charged once
        per transport wrapper, not per edge in its do-calculus path.
    :param heuristic: A finite nonnegative lower bound on a Query's proof
        cost. It must account for equivalent queries to be admissible.
        The default is zero. Only actual costs prune the bound.
    :param tiebreaker: A SearchState scorer or iterable of scorers to sum.
        Partial states describe a local inference, not the entire target
        tree. Scores must be finite; lower scores break equal priorities.
    :param limit: Stop after this many target classes, or None to exhaust.
    :param max_expansions: Budget for cataloged queries and popped agenda
        items, default 10000; None disables the guard. Free transport can
        give infinitely many classes within the bound. Budget exhaustion
        raises SearchLimitExceeded; it never silently returns a full result.
    :returns: One derivation per class, in minimum-cost order for admissible
        heuristics, with ties broken by scorers and stable insertion order.
    :raises ValueError: For invalid arguments, costs, or scoring values.
    :raises SearchLimitExceeded: With partial derivations and expansion count
        when the budget is reached before exhaustion or the class limit.
    """
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Limit must be a positive integer")
    scorers = (
        ()
        if tiebreaker is None
        else (tiebreaker,) if callable(tiebreaker) else tuple(tiebreaker)
    )

    def score(
        query: Query,
        cost: int,
        pending: tuple[Query, ...],
        expression: Expression,
        derivation: Derivation | None,
    ) -> float:
        state = SearchState(query, cost, pending, expression, derivation)
        values = [scorer(state) for scorer in scorers]
        total = sum(values)
        if any(not isfinite(value) for value in values) or not isfinite(total):
            raise ValueError("Tie-breaker scores must be finite")
        return total

    admg = ADMG.parse(graph)
    x, y, _ = parse_query(admg, treatments, outcomes)
    search = _Search(
        admg,
        costs=costs,
        heuristic=heuristic,
        score=score if scorers else None,
        max_expansions=max_expansions,
    )
    found = []
    for derivation in search.run(Query(y, x), bound):
        found.append(derivation)
        if limit is not None and len(found) == limit:
            break
    return search.found
