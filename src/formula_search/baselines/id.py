"""The ID and IDC algorithms of Shpitser and Pearl, in hiprof's grammar.

Lines 1-4 of ID (Shpitser and Pearl, 2006) reduce the target to the
districts ``D`` of ``G[R]``, where ``R`` holds the ancestors of the outcomes
``Y`` once the treatments ``T`` are removed (Richardson et al., 2023,
Theorem 48):

    p(y | do(t)) = sum_{R \\ Y} prod_D Q[D].

Lines 5-7 identify each district kernel ``Q[D]`` recursively and carry the
current distribution as a kernel expression, so every expression they build
is admissible in hiprof's grammar. Two steps of the proof of Proposition 1 of
Freni et al. (2026) turn the product over districts into an admissible
formula of type ``Y | T``:

- a kernel may depend on variables outside ``T`` and ``R``, which line 3 of
  ID treats as intervened on; it is averaged over them;
- the product need not be sequential, since ``Q[D]`` may depend on variables
  of a district that in turn depends on ``D`` (their Example 7); it is then
  split into one factor per variable of ``R``, each averaged over the inputs
  it may not use.

Averaging leaves a kernel unchanged under the model, as it does not depend
on these inputs.

IDC (Shpitser and Pearl, 2008, Figure 7) identifies a conditional target
``p(y | do(t), w)``: Rule 2 turns conditions into interventions while it
applies, and ID identifies the joint kernel of the outcomes and the remaining
conditions, which is then conditioned on these conditions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..docalculus import rule2
from ..expression import (
    Expression,
    Term,
    average,
    condition,
    marginal,
    product,
    render,
)
from ..graph import ADMG


class _Hedge(Exception):
    """Raised on line 5 of ID, which finds a hedge for the target."""


def identify(
    graph: str,
    treatments: str | Iterable[str],
    outcomes: str | Iterable[str],
    conditions: str | Iterable[str] = (),
) -> str | None:
    """Identify ``p(outcomes | do(treatments), conditions)``.

    The target is identified with ID, or with IDC if there are conditions.

    :param graph: Acyclic directed mixed graph in hiprof's syntax, such as
        ``"T -> M; M -> Y; T <-> Y"``.
    :param treatments: Treatment variable, or variables.
    :param outcomes: Outcome variable, or variables.
    :param conditions: Conditioning variable, or variables.
    :returns: An identifying formula in hiprof's grammar, or ``None`` if the
        target is not identifiable. Its inputs are treatments and conditions,
        but it may omit those that the target does not depend on.
    :raises ValueError: If the graph is invalid, the outcomes are empty, or
        the variables overlap or are not nodes of the graph.
    """
    admg = ADMG.parse(graph)
    x, y, w = parse_query(admg, treatments, outcomes, conditions)
    formula = idc(admg, y, x, w)
    return None if formula is None else render(formula, reserved=x | w)


def parse_query(
    graph: ADMG,
    treatments: str | Iterable[str],
    outcomes: str | Iterable[str],
    conditions: str | Iterable[str] = (),
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Return the treatments, outcomes, and conditions of a query as sets.

    :raises ValueError: If the outcomes are empty, or the variables overlap
        or are not nodes of the graph.
    """
    x, y, w = map(_as_set, (treatments, outcomes, conditions))
    if not y:
        raise ValueError("Outcomes must not be empty.")
    if unknown := (x | y | w) - graph.nodes:
        raise ValueError(
            "Variables must be nodes of the graph; "
            f"not nodes: {', '.join(sorted(unknown))}."
        )
    if len(x) + len(y) + len(w) > len(x | y | w):
        raise ValueError(
            "Treatments, outcomes, and conditions must be disjoint."
        )
    return x, y, w


def idc(
    graph: ADMG,
    y: frozenset[str],
    x: frozenset[str],
    w: frozenset[str] = frozenset(),
) -> Expression | None:
    """Identify ``p(y | do(x), w)`` with IDC, as an expression.

    :returns: The identifying formula, or ``None`` if the target is not
        identifiable.
    """
    for z in sorted(w):  # line 1
        if rule2(graph, y, x, {z}, w - {z}):
            return idc(graph, y, x | {z}, w - {z})

    try:  # line 2
        joint = _id(graph, y | w, x)
    except _Hedge:
        return None
    return condition(joint, w)


def _id(graph: ADMG, y: frozenset[str], x: frozenset[str]) -> Expression:
    """Identify ``p(y | do(x))`` with ID.

    :raises _Hedge: If the target is not identifiable.
    """
    order = graph.topological_order()

    graph = graph.subgraph(graph.ancestors(y))  # line 2
    x &= graph.nodes
    if not x:  # line 1
        return Term(y)

    r = graph.subgraph(graph.nodes - x).ancestors(y)  # line 3
    districts = sorted(  # line 4
        graph.subgraph(r).districts(),
        key=lambda district: min(map(order.index, district)),
    )
    kernels = [
        _kernel(district, Term(graph.nodes), graph, order)
        for district in districts
    ]
    return marginal(product(_factors(kernels, x, order)), r - y)


def _kernel(
    district: frozenset[str],
    kernel: Expression,
    graph: ADMG,
    order: Sequence[str],
) -> Expression:
    """Identify ``Q[district]`` from the kernel over the nodes of the graph.

    This is the call ``ID(district, V \\ district, kernel, graph)`` of
    line 4, for which lines 3 and 4 have no effect.
    """
    if district == graph.nodes:  # line 1
        return kernel

    ancestors = graph.ancestors(district)
    if ancestors != graph.nodes:  # line 2
        return _kernel(
            district,
            marginal(kernel, graph.nodes - ancestors),
            graph.subgraph(ancestors),
            order,
        )

    districts = graph.districts()
    if districts == {graph.nodes}:  # line 5
        raise _Hedge

    enclosing = next(other for other in districts if district <= other)
    factors = product(
        _conditional(kernel, node, order)
        for node in order
        if node in enclosing
    )
    if enclosing == district:  # line 6
        return factors
    return _kernel(district, factors, graph.subgraph(enclosing), order)


def _factors(
    kernels: list[Expression],
    treatments: frozenset[str],
    order: Sequence[str],
) -> list[Expression]:
    """Order the district kernels into the factors of a sequential product.

    The inputs of each factor are treatments or outputs of earlier factors.
    """
    r = frozenset().union(*(q.outputs for q in kernels))
    whole = _in_sequence(
        [average(q, q.inputs - treatments - r) for q in kernels], treatments
    )
    if whole is not None:
        return whole

    # Steps (a)-(c) of the proof of Proposition 1 of Freni et al. (2026).
    factors = []
    for index, node in enumerate(order):
        if node in r:
            kernel = next(q for q in kernels if node in q.outputs)
            factor = _conditional(kernel, node, order)
            allowed = treatments | (r & set(order[:index]))
            factors.append(average(factor, factor.inputs - allowed))
    return factors


def _in_sequence(
    factors: list[Expression],
    inputs: frozenset[str],
) -> list[Expression] | None:
    """Order factors into a sequential product with the given inputs.

    :returns: The ordered factors, or ``None`` if no such order exists.
    """
    pending, ordered, available = list(factors), [], set(inputs)
    while pending:
        factor = next((f for f in pending if f.inputs <= available), None)
        if factor is None:
            return None
        pending.remove(factor)
        ordered.append(factor)
        available |= factor.outputs
    return ordered


def _conditional(
    kernel: Expression,
    node: str,
    order: Sequence[str],
) -> Expression:
    """Return the kernel's conditional of ``node`` given its earlier outputs."""
    earlier = kernel.outputs & set(order[: order.index(node)])
    return condition(
        marginal(kernel, kernel.outputs - earlier - {node}),
        earlier,
    )


def _as_set(variables: str | Iterable[str]) -> frozenset[str]:
    return frozenset([variables] if isinstance(variables, str) else variables)
