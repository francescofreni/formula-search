"""Rules 2 and 3 of do-calculus (Pearl, 1995), as graphical tests.

Rule 1 follows from Rules 2 and 3 (Huang and Valtorta, 2006, Lemma 4;
Yvernes et al., 2026, Theorem 2), so these two suffice. Applied to one
variable at a time, they join the queries that they prove equal into the
connected components of the derivation graph of Yvernes et al., whose edges
are single applications of the rules; a rule applied to a set of variables
decomposes into such applications (their Theorems 3 and 4).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Set
from dataclasses import dataclass

from .expression import render_query
from .graph import ADMG


@dataclass(frozen=True)
class Query:
    """The query ``p(outcomes | do(treatments), conditions)``."""

    outcomes: frozenset[str]
    treatments: frozenset[str] = frozenset()
    conditions: frozenset[str] = frozenset()

    @property
    def inputs(self) -> frozenset[str]:
        return self.treatments | self.conditions

    def __str__(self) -> str:
        return render_query(self.outcomes, self.treatments, self.conditions)


def rule2(
    graph: ADMG,
    y: Set[str],
    x: Set[str],
    z: Set[str],
    w: Set[str],
) -> bool:
    """Return whether ``p(y | do(x, z), w) = p(y | do(x), z, w)`` by Rule 2."""
    mutilated = graph.mutilated(into=x, out_of=z)
    return mutilated.d_separated(y, z, x | w)


def rule3(
    graph: ADMG,
    y: Set[str],
    x: Set[str],
    z: Set[str],
    w: Set[str],
) -> bool:
    """Return whether ``p(y | do(x, z), w) = p(y | do(x), w)`` by Rule 3."""
    mutilated = graph.mutilated(into=x)
    mutilated = mutilated.mutilated(into=z - mutilated.ancestors(w))
    return mutilated.d_separated(y, z, x | w)


def neighbours(graph: ADMG, query: Query) -> Iterator[Query]:
    """Yield the queries that one rule on one variable equates with ``query``.

    For each variable ``v``, Rule 2 equates ``p(y | do(x, v), w)`` with
    ``p(y | do(x), v, w)``, and Rule 3 equates it with ``p(y | do(x), w)``.
    """
    y = query.outcomes
    for v in sorted(graph.nodes - y):
        x, w = query.treatments - {v}, query.conditions - {v}
        if v in query.treatments:  # observe v by Rule 2, or drop it by Rule 3
            if rule2(graph, y, x, {v}, w):
                yield Query(y, x, w | {v})
            if rule3(graph, y, x, {v}, w):
                yield Query(y, x, w)
        elif v in query.conditions:  # intervene on v by Rule 2
            if rule2(graph, y, x, {v}, w):
                yield Query(y, x | {v}, w)
        elif rule3(graph, y, x, {v}, w):  # intervene on v by Rule 3
            yield Query(y, x | {v}, w)


def equivalent_queries(
    graph: ADMG,
    query: Query,
) -> dict[Query, tuple[Query, ...]]:
    """Return the queries that do-calculus proves equal to ``query``.

    :returns: For each such query, a shortest path to it from ``query``,
        each step one of its :func:`neighbours`. The first is ``query``.
    """
    paths: dict[Query, tuple[Query, ...]] = {query: (query,)}
    pending = deque([query])
    while pending:
        current = pending.popleft()
        for neighbour in neighbours(graph, current):
            if neighbour not in paths:
                paths[neighbour] = paths[current] + (neighbour,)
                pending.append(neighbour)
    return paths
