from __future__ import annotations

import random
from collections.abc import Iterable
from itertools import combinations

from hiprof import HPFalsifier
from hiprof.formula import parse_and_validate

from formula_search.baselines.id import parse_query
from formula_search.expression import render_query
from formula_search.graph import ADMG


def assert_identifying(
    graph: str,
    treatments: str | Iterable[str],
    outcomes: str | Iterable[str],
    conditions: str | Iterable[str],
    formula: str | None,
) -> None:
    """Check that an observational formula for the target identifies it."""
    x, y, w = parse_query(ADMG.parse(graph), treatments, outcomes, conditions)
    assert formula is not None and "do(" not in formula
    signature = parse_and_validate(formula).signature
    assert {str(variable) for variable in signature.outputs} == y
    assert {str(variable) for variable in signature.inputs} <= x | w
    assert HPFalsifier(graph).check(render_query(y, x, w), formula).accepted


def random_graph(rng: random.Random) -> tuple[str, list[str]]:
    """Return a random graph on two to seven nodes, and its nodes."""
    nodes = rng.sample("ABCDEFG", rng.randint(2, 7))
    statements = list(nodes)
    for first, second in combinations(nodes, 2):
        if rng.random() < 0.4:
            statements.append(f"{first} -> {second}")
        if rng.random() < 0.25:
            statements.append(f"{first} <-> {second}")
    return "; ".join(statements), nodes
