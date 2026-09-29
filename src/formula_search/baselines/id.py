"""The ID algorithm of Shpitser and Pearl (2006), in hiprof's grammar.

Lines 1-4 of ID reduce the target to the districts ``D`` of ``G[R]``, where
``R`` holds the ancestors of the outcomes ``Y`` once the treatments ``T`` are
removed (Richardson et al., 2023, Theorem 48):

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
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

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
) -> str | None:
    """Identify ``p(outcomes | do(treatments))`` with the ID algorithm.

    :param graph: Acyclic directed mixed graph in hiprof's syntax, such as
        ``"T -> M; M -> Y; T <-> Y"``.
    :param treatments: Treatment variable, or variables.
    :param outcomes: Outcome variable, or variables.
    :returns: An identifying formula in hiprof's grammar, or ``None`` if the
        target is not identifiable. Its inputs are treatments, but it may
        omit treatments that the target does not depend on; declare these as
        ``redundant_inputs`` when checking the formula with hiprof.
    :raises ValueError: If the graph is invalid, or the treatments and
        outcomes are empty, overlap, or are not nodes of the graph.
    """
    admg = ADMG.parse(graph)
    t = _variables(treatments, "Treatments", admg)
    y = _variables(outcomes, "Outcomes", admg)
    if t & y:
        raise ValueError("Treatments and outcomes must be disjoint.")
    order = admg.topological_order()

    admg = admg.subgraph(admg.ancestors(y))  # line 2
    x = t & admg.nodes
    if not x:  # line 1
        return render(Term(y))

    r = admg.subgraph(admg.nodes - x).ancestors(y)  # line 3
    districts = sorted(  # line 4
        admg.subgraph(r).districts(),
        key=lambda district: min(map(order.index, district)),
    )
    try:
        kernels = [
            _kernel(district, Term(admg.nodes), admg, order)
            for district in districts
        ]
    except _Hedge:
        return None

    factors = _factors(kernels, x, order)
    return render(marginal(product(factors), r - y), reserved=t)


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


def _variables(
    variables: str | Iterable[str],
    name: str,
    graph: ADMG,
) -> frozenset[str]:
    names = frozenset([variables] if isinstance(variables, str) else variables)
    if not names:
        raise ValueError(f"{name} must not be empty.")
    if unknown := names - graph.nodes:
        raise ValueError(
            f"{name} must be nodes of the graph; "
            f"not nodes: {', '.join(sorted(unknown))}."
        )
    return names
