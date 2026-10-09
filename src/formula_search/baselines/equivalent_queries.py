"""ID applied to the queries equivalent to the target (Yvernes et al., 2026).

The queries ``p(y | do(x), w)`` that do-calculus turns into one another form
a connected component of the derivation graph of Yvernes et al., which
:func:`~formula_search.docalculus.equivalent_queries` explores. IDC
identifies every query of the component, which yields several identifying
formulas for the same target.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..docalculus import Query, equivalent_queries
from ..expression import average, render
from ..graph import ADMG
from .id import idc, parse_query


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
    for query in equivalent_queries(admg, Query(y, x, w)):
        formula = idc(admg, y, query.treatments, query.conditions)
        if formula is not None:
            formula = average(formula, formula.inputs - x - w)
            formulas[str(query)] = render(formula, reserved=x | w)
    return formulas
