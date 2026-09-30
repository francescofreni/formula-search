"""ID applied to the queries equivalent to the target (Yvernes et al., 2026).

The queries ``p(y | do(x), w)`` that do-calculus turns into one another form
a connected component of the derivation graph of Yvernes et al., whose edges
are single applications of the rules. Since Rule 1 follows from Rules 2 and 3
(their Theorem 2), the component is explored with these two rules, one
variable at a time: Rule 2 exchanges an intervention and an observation, and
Rule 3 inserts or deletes an intervention. IDC identifies every query of the
component, which yields several identifying formulas for the same target.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..docalculus import rule2, rule3
from ..expression import average, render, render_query
from ..graph import ADMG
from .id import idc, parse_query

Query = tuple[frozenset[str], frozenset[str]]


def identify_equivalent(
    graph: str,
    treatments: str | Iterable[str],
    outcomes: str | Iterable[str],
    conditions: str | Iterable[str] = (),
) -> dict[str, str]:
    """Identify every query equivalent to the target.

    The target is ``p(outcomes | do(treatments), conditions)``, and each
    equivalent query is identified with IDC. Inputs of the resulting formula
    that the target does not have are averaged out, so that it identifies the
    target, just as the query does.

    :param graph: Acyclic directed mixed graph in hiprof's syntax.
    :param treatments: Treatment variable, or variables.
    :param outcomes: Outcome variable, or variables.
    :param conditions: Conditioning variable, or variables.
    :returns: The formula obtained from each equivalent query, keyed by the
        query, both in hiprof's syntax, starting with the target; empty if
        the target is not identifiable.
    :raises ValueError: If the graph is invalid, the outcomes are empty, or
        the variables overlap or are not nodes of the graph.
    """
    admg = ADMG.parse(graph)
    x, y, w = parse_query(admg, treatments, outcomes, conditions)
    formulas = {}
    for do, given in equivalent_queries(admg, y, x, w):
        formula = idc(admg, y, do, given)
        if formula is not None:
            query = render_query(y, do, given)
            formula = average(formula, formula.inputs - x - w)
            formulas[query] = render(formula, reserved=x | w)
    return formulas


def equivalent_queries(
    graph: ADMG,
    y: frozenset[str],
    x: frozenset[str],
    w: frozenset[str] = frozenset(),
) -> list[Query]:
    """Return the queries equivalent to ``p(y | do(x), w)``.

    :returns: The interventions and conditions of the queries, starting with
        ``(x, w)``.
    """
    queries = {(x, w): None}  # an ordered set
    pending = [(x, w)]
    while pending:
        for query in _neighbours(graph, y, *pending.pop()):
            if query not in queries:
                queries[query] = None
                pending.append(query)
    return list(queries)


def _neighbours(
    graph: ADMG,
    y: frozenset[str],
    x: frozenset[str],
    w: frozenset[str],
) -> Iterator[Query]:
    """Yield the queries one rule away from ``p(y | do(x), w)``.

    For each variable ``v``, Rule 2 equates ``p(y | do(x', v), w')`` with
    ``p(y | do(x'), v, w')``, and Rule 3 equates it with ``p(y | do(x'), w')``.
    """
    for v in sorted(graph.nodes - y):
        others, given = x - {v}, w - {v}
        if v in x:  # observe v by Rule 2, or drop it by Rule 3
            if rule2(graph, y, others, {v}, given):
                yield others, given | {v}
            if rule3(graph, y, others, {v}, given):
                yield others, given
        elif v in w:  # intervene on v by Rule 2
            if rule2(graph, y, others, {v}, given):
                yield others | {v}, given
        elif rule3(graph, y, others, {v}, given):  # intervene on v by Rule 3
            yield others | {v}, given
