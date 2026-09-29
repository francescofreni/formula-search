from __future__ import annotations

import random
from collections.abc import Iterable
from itertools import combinations

import pytest
from hiprof import HPFalsifier
from hiprof.formula import parse_and_validate

from formula_search.baselines import identify
from formula_search.graph import ADMG

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

NOT_IDENTIFIABLE = [
    pytest.param("T -> Y; T <-> Y", id="bow arc"),
    pytest.param("T -> M; M -> Y; T <-> M", id="confounded mediator"),
    pytest.param("T -> M; M -> Y; T <-> Y; M <-> Y", id="confounded outcome"),
    pytest.param("Z -> T; Z -> Y; T -> Y; T <-> Z; Z <-> Y", id="confounder"),
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
    assert_identifying(graph, _set(treatments), _set(outcomes), formula)


@pytest.mark.parametrize("graph", NOT_IDENTIFIABLE)
def test_identify_returns_none_for_non_identifiable_targets(
    graph: str,
) -> None:
    assert identify(graph, "T", "Y") is None


def test_identify_agrees_with_fixing_on_random_graphs() -> None:
    rng = random.Random(0)

    for _ in range(300):
        graph, treatments, outcomes = random_query(rng)
        formula = identify(graph, treatments, outcomes)

        admg = ADMG.parse(graph)
        assert (formula is not None) == identifiable(
            admg, treatments, outcomes
        )
        if formula is not None:
            assert_identifying(graph, treatments, outcomes, formula)


@pytest.mark.parametrize(
    ("treatments", "outcomes", "message"),
    [
        ((), "Y", "must not be empty"),
        ("T", "Z", "not nodes: Z"),
        ("T", "T", "must be disjoint"),
    ],
)
def test_identify_rejects_invalid_queries(
    treatments: str | tuple[str, ...],
    outcomes: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        identify("T -> Y", treatments, outcomes)


def assert_identifying(
    graph: str,
    treatments: frozenset[str],
    outcomes: frozenset[str],
    formula: str,
) -> None:
    """Check the type of the formula and accept it with hiprof."""
    signature = parse_and_validate(formula).signature
    inputs = {str(variable) for variable in signature.inputs}
    assert {str(variable) for variable in signature.outputs} == outcomes
    assert inputs <= treatments

    target = f"p({_names(outcomes)} | do({_names(treatments)}))"
    result = HPFalsifier(graph).check(
        target,
        formula,
        redundant_inputs=sorted(treatments - inputs) or None,
    )
    assert result.accepted


def identifiable(
    graph: ADMG,
    treatments: frozenset[str],
    outcomes: frozenset[str],
) -> bool:
    """Decide identifiability by fixing (Richardson et al., 2023, Thm. 48)."""
    r = graph.subgraph(graph.nodes - treatments).ancestors(outcomes)
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


def random_query(
    rng: random.Random,
) -> tuple[str, frozenset[str], frozenset[str]]:
    nodes = rng.sample("ABCDEFG", rng.randint(2, 7))
    statements = list(nodes)
    for first, second in combinations(nodes, 2):
        if rng.random() < 0.4:
            statements.append(f"{first} -> {second}")
        if rng.random() < 0.25:
            statements.append(f"{first} <-> {second}")

    treatments = rng.sample(nodes, rng.randint(1, min(2, len(nodes) - 1)))
    others = [node for node in nodes if node not in treatments]
    outcomes = rng.sample(others, rng.randint(1, min(2, len(others))))
    return "; ".join(statements), frozenset(treatments), frozenset(outcomes)


def _descendants(graph: ADMG, node: str) -> frozenset[str]:
    return frozenset(
        other for other in graph.nodes if node in graph.ancestors({other})
    )


def _set(variables: str | Iterable[str]) -> frozenset[str]:
    return frozenset([variables] if isinstance(variables, str) else variables)


def _names(variables: Iterable[str]) -> str:
    return ", ".join(sorted(variables))
