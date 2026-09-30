from __future__ import annotations

import random

import pytest

from formula_search.baselines import identify
from formula_search.docalculus import rule2
from formula_search.graph import ADMG

from .helpers import assert_identifying, random_graph

IDENTIFIABLE = [
    pytest.param("T -> Y", "T", "Y", "p(Y | T)", id="direct effect"),
    pytest.param("Y -> T", "T", "Y", "p(Y)", id="no causal path"),
    pytest.param(
        "C -> T; C -> Y; T -> Y",
        "T",
        "Y",
        "sum_{C} { p(C) p(Y | C, T) }",
        id="back-door",
    ),
    pytest.param(
        "T -> M; M -> Y; T <-> Y",
        "T",
        "Y",
        "sum_{M} { p(M | T) sum_{T'} { p(T') p(Y | M, T') } }",
        id="front-door",
    ),
    # Freni et al. (2026), Example 3: ID returns phi_1 on G1, phi_2 on G2.
    pytest.param(
        "T -> M; M -> Y; C -> T; C -> Y",
        "T",
        "Y",
        "sum_{C} { p(C) p(Y | C, T) }",
        id="Freni et al. G1",
    ),
    pytest.param(
        "T -> M; M -> Y; T -> C; Y -> C",
        "T",
        "Y",
        "p(Y | T)",
        id="Freni et al. G2",
    ),
    # Line 3 of ID intervenes on U, which the formula averages out.
    pytest.param(
        "U -> T; C -> T; C -> Y; T -> Y",
        "T",
        "Y",
        "sum_{C} { p(C) p(Y | C, T) }",
        id="instrument",
    ),
    # y0 0.2.11 returns p(Y | X) on the napkin graph, which is invalid.
    pytest.param(
        "X -> Y; W -> Z; Z -> X; X <-> W; W <-> Y",
        "X",
        "Y",
        "sum_{Z} { p(Z | X) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }",
        id="napkin",
    ),
    # Freni et al. (2026), Example 7: the product over districts is split.
    pytest.param(
        "T -> A; A -> B; B -> C; C -> Y; T <-> B; B <-> Y",
        "T",
        "Y",
        "sum_{A, B, C} { p(A | T) sum_{T'} { p(T') p(B | A, T') } "
        "p(C | A, B, T) icd_{B | A, C} { sum_{T'} { p(T') p(B | A, T') "
        "p(Y | A, B, C, T') } } }",
        id="Freni et al. Example 7",
    ),
    # Shpitser and Pearl (2006), Figure 1(a).
    pytest.param(
        "W1 -> X; X -> Y1; W2 -> Y2; W1 <-> Y1; W1 <-> W2; W1 <-> Y2",
        "X",
        ("Y1", "Y2"),
        "p(Y2) sum_{W1} { p(W1) p(Y1 | W1, X) }",
        id="joint outcomes",
    ),
    pytest.param(
        "T1 -> T2; T2 -> Y; T1 <-> Y",
        ("T1", "T2"),
        "Y",
        "sum_{T1'} { p(T1') p(Y | T1', T2) }",
        id="joint treatments",
    ),
]

CONDITIONAL = [
    # Rule 2 turns the condition into an intervention, as in Figure 6(a) of
    # Shpitser and Pearl (2008), where p(Y | do(X)) is not identifiable.
    pytest.param(
        "X -> Z; Z -> Y; X <-> Z",
        "X",
        "Y",
        "Z",
        "p(Y | X, Z)",
        id="confounded mediator",
    ),
    pytest.param(
        "T -> M; M -> Y; T <-> Y",
        "T",
        "Y",
        "M",
        "sum_{T'} { p(T') p(Y | M, T') }",
        id="front-door mediator",
    ),
    pytest.param(
        "T -> M; M -> Y; T <-> Y; Y -> C",
        "T",
        "Y",
        "C",
        "icd_{C | T} { sum_{M} { p(M | T) sum_{T'} { p(T') p(Y | M, T') } "
        "p(C | M, T, Y) } }",
        id="child of the outcome",
    ),
    pytest.param("A -> B; A <-> B", (), "B", "A", "p(B | A)", id="no action"),
]

NOT_IDENTIFIABLE = [
    pytest.param("T -> Y; T <-> Y", (), id="bow arc"),
    pytest.param("T -> M; M -> Y; T <-> M", (), id="confounded mediator"),
    pytest.param(
        "T -> M; M -> Y; T <-> Y; M <-> Y", (), id="confounded outcome"
    ),
    pytest.param(
        "Z -> T; Z -> Y; T -> Y; T <-> Z; Z <-> Y", (), id="confounder"
    ),
    # Conditioning on a collider, as in Figure 6(b) of Shpitser and Pearl
    # (2008), where p(Y | do(T)) = p(Y | T).
    pytest.param("T -> Y; T -> Z; Y -> Z; T <-> Z", "Z", id="collider"),
]


@pytest.mark.parametrize(
    ("graph", "treatments", "outcomes", "expected"), IDENTIFIABLE
)
def test_identify_returns_identifying_formulas(
    graph: str,
    treatments: str | tuple[str, ...],
    outcomes: str | tuple[str, ...],
    expected: str,
) -> None:
    formula = identify(graph, treatments, outcomes)

    assert formula == expected
    assert_identifying(graph, treatments, outcomes, (), formula)


@pytest.mark.parametrize(
    ("graph", "treatments", "outcomes", "conditions", "expected"),
    CONDITIONAL,
)
def test_identify_returns_formulas_for_conditional_targets(
    graph: str,
    treatments: str | tuple[str, ...],
    outcomes: str,
    conditions: str,
    expected: str,
) -> None:
    formula = identify(graph, treatments, outcomes, conditions)

    assert formula == expected
    assert_identifying(graph, treatments, outcomes, conditions, formula)


@pytest.mark.parametrize(("graph", "conditions"), NOT_IDENTIFIABLE)
def test_identify_returns_none_for_non_identifiable_targets(
    graph: str,
    conditions: str | tuple[str, ...],
) -> None:
    assert identify(graph, "T", "Y", conditions) is None


def test_identify_agrees_with_fixing_on_random_graphs() -> None:
    rng = random.Random(0)

    for _ in range(300):
        graph, nodes = random_graph(rng)
        nodes = rng.sample(nodes, len(nodes))
        split = rng.randint(1, min(2, len(nodes)))
        y, others = frozenset(nodes[:split]), nodes[split:]
        x = frozenset(node for node in others if rng.random() < 0.3)
        w = frozenset(n for n in others if n not in x and rng.random() < 0.3)
        formula = identify(graph, x, y, w)

        expected = identifiable(ADMG.parse(graph), x, y, w)
        assert (formula is not None) == expected, (graph, x, y, w)
        if formula is not None:
            assert_identifying(graph, x, y, w, formula)


@pytest.mark.parametrize(
    ("outcomes", "conditions", "message"),
    [
        ((), (), "must not be empty"),
        ("Z", (), "not nodes: Z"),
        ("T", (), "must be disjoint"),
        ("Y", "Y", "must be disjoint"),
    ],
)
def test_identify_rejects_invalid_queries(
    outcomes: str | tuple[str, ...],
    conditions: str | tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        identify("T -> Y", "T", outcomes, conditions)


def identifiable(
    graph: ADMG,
    treatments: frozenset[str],
    outcomes: frozenset[str],
    conditions: frozenset[str],
) -> bool:
    """Decide identifiability with the fixing criterion.

    Rule 2 first turns every condition that it can into an intervention
    (Shpitser and Pearl, 2008, Theorems 20 and 21), and the joint target of
    the outcomes and the other conditions is then identifiable if its
    districts are reachable (Richardson et al., 2023, Theorem 48).
    """
    movable = {
        z
        for z in conditions
        if rule2(graph, outcomes, treatments, {z}, conditions - {z})
    }
    x, y = treatments | movable, outcomes | (conditions - movable)
    r = graph.subgraph(graph.nodes - x).ancestors(y)
    return all(
        reachable(graph, district)
        for district in graph.subgraph(r).districts()
    )


def reachable(graph: ADMG, target: frozenset[str]) -> bool:
    """Whether the nodes outside ``target`` can be fixed one by one."""
    nodes = graph.nodes
    while nodes != target:
        current = graph.subgraph(nodes)
        fixable = [
            node
            for node in sorted(nodes - target)
            if current.district(node) & _descendants(current, node) == {node}
        ]
        if not fixable:
            return False
        nodes -= {fixable[0]}
    return True


def _descendants(graph: ADMG, node: str) -> frozenset[str]:
    return frozenset(
        other for other in graph.nodes if node in graph.ancestors({other})
    )
