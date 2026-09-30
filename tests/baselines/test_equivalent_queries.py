from __future__ import annotations

import random

import pytest
from hiprof import HPFalsifier

from formula_search.baselines import identify_equivalent
from formula_search.baselines.equivalent_queries import equivalent_queries
from formula_search.baselines.id import parse_query
from formula_search.expression import render_query
from formula_search.graph import ADMG

from .helpers import assert_identifying, random_graph

NAPKIN = "W -> Z; Z -> X; X -> Y; W <-> X; W <-> Y"
FIGURE_15 = (
    "X -> Z4; Z4 -> Y; Z1 -> X; Z2 -> Z1; Z3 -> Z2; Z3 -> X; Z5 -> Z1; "
    "Z5 -> Z4; X <-> Z2; Z3 <-> Z2; Z2 <-> Y; Z4 <-> Y; Z5 <-> Z4"
)
SACHS = (
    "PKA -> JNK; PKA -> P38; PKA -> MEK; PKA -> RAF; PKA -> ERK; "
    "PKA -> AKT; PKC -> JNK; PKC -> P38; PKC -> MEK; PKC -> RAF; "
    "PKC -> PKA; PIP3 -> AKT; MEK -> ERK; RAF -> MEK; ERK -> AKT; "
    "PIP2 -> PKC; PIP3 -> PIP2; PLCG -> PKC; PLCG -> PIP2; PLCG -> PIP3"
)

# Connected components of derivation graphs in Yvernes et al. (2026).
COMPONENTS = [
    pytest.param(
        NAPKIN,
        "X",
        "Y",
        (),
        {
            "p(Y | do(X))",
            "p(Y | do(X, Z))",
            "p(Y | do(W, X))",
            "p(Y | do(W, X), Z)",
            "p(Y | do(W), X)",
            "p(Y | do(W, X, Z))",
            "p(Y | do(W, Z), X)",
            "p(Y | do(W), X, Z)",
            "p(Y | do(Z), X)",
        },
        id="Figure 4",
    ),
    pytest.param(
        "W -> Z; Z -> X; X -> Y; W <-> Y",
        "W",
        "Y",
        "Z",
        {"p(Y | do(W), Z)", "p(Y | do(W, Z))", "p(Y | do(Z))"},
        id="Figure 13",
    ),
    pytest.param(
        "A; B; C",
        "C",
        ("A", "B"),
        (),
        {"p(A, B | do(C))", "p(A, B)", "p(A, B | C)"},
        id="Example 2, empty graph",
    ),
    pytest.param(
        "A -> B; B -> C",
        (),
        "B",
        "A",
        {"p(B | A)", "p(B | do(C), A)", "p(B | do(A, C))", "p(B | do(A))"},
        id="Example 2, chain",
    ),
    pytest.param(
        FIGURE_15,
        "X",
        "Y",
        (),
        {
            "p(Y | do(X), Z3)",
            "p(Y | do(X, Z3))",
            "p(Y | do(X, Z2), Z3)",
            "p(Y | do(X))",
            "p(Y | do(X, Z2))",
            "p(Y | do(X, Z2, Z3))",
            "p(Y | do(X, Z1), Z3)",
            "p(Y | do(X, Z1, Z3))",
            "p(Y | do(X, Z1, Z2), Z3)",
            "p(Y | do(X, Z1))",
            "p(Y | do(X, Z1, Z2))",
            "p(Y | do(X, Z1, Z2, Z3))",
            "p(Y | do(Z1), X, Z3)",
            "p(Y | do(Z1, Z3), X)",
            "p(Y | do(Z1, Z2), X, Z3)",
            "p(Y | do(Z1), X)",
            "p(Y | do(Z1, Z2), X)",
            "p(Y | do(Z1, Z2, Z3), X)",
        },
        id="Figure 15",
    ),
]


@pytest.mark.parametrize(
    ("graph", "treatments", "outcomes", "conditions", "expected"),
    COMPONENTS,
)
def test_identify_equivalent_identifies_the_component_of_the_target(
    graph: str,
    treatments: str | tuple[str, ...],
    outcomes: str | tuple[str, ...],
    conditions: str | tuple[str, ...],
    expected: set[str],
) -> None:
    formulas = identify_equivalent(graph, treatments, outcomes, conditions)

    x, y, w = parse_query(ADMG.parse(graph), treatments, outcomes, conditions)
    assert list(formulas)[0] == render_query(y, x, w)
    assert set(formulas) == expected
    for formula in formulas.values():
        assert_identifying(graph, x, y, w, formula)


def test_identify_equivalent_finds_the_null_effect_in_the_sachs_graph() -> (
    None
):
    # Yvernes et al. (2026), Section 5.2: 32 queries equal p(P38 | do(MEK)).
    formulas = identify_equivalent(SACHS, "MEK", "P38")

    assert len(formulas) == 32
    assert set(formulas.values()) == {"p(P38)"}


def test_identify_equivalent_is_empty_for_non_identifiable_targets() -> None:
    assert identify_equivalent("T -> Y; T <-> Y", "T", "Y") == {}


def test_equivalent_queries_equal_the_target_on_random_graphs() -> None:
    rng = random.Random(0)

    for _ in range(100):
        graph, nodes = random_graph(rng)
        outcome, *others = rng.sample(nodes, len(nodes))
        y = frozenset({outcome})
        x = frozenset(node for node in others if rng.random() < 0.4)
        w = frozenset(n for n in others if n not in x and rng.random() < 0.2)
        target = render_query(y, x, w)

        falsifier = HPFalsifier(graph)
        for do, given in equivalent_queries(ADMG.parse(graph), y, x, w):
            assert falsifier.check(target, render_query(y, do, given)).accepted
        for formula in identify_equivalent(graph, x, y, w).values():
            assert_identifying(graph, x, y, w, formula)
