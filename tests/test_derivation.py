from __future__ import annotations

import random

import pytest
from hiprof import HPFalsifier

from formula_search.baselines import identify, identify_equivalent
from formula_search.derivation import Derivation, derive, rules
from formula_search.docalculus import Query, neighbours
from formula_search.evaluator import Archive
from formula_search.graph import ADMG
from formula_search.suite import CURATED, random_tasks


def query(outcomes: str, treatments: str = "", conditions: str = "") -> Query:
    return Query(
        *(frozenset(v.split()) for v in (outcomes, treatments, conditions))
    )


def test_derive_finds_the_front_door_formula() -> None:
    task = CURATED["front-door"]

    (derivation,) = derive(*task, bound=4)

    assert derivation.formula == (
        "sum_{M} { p(M | T) sum_{T'} { p(T') p(Y | M, T') } }"
    )
    assert derivation.verify(ADMG.parse(task.graph))


def test_derive_finds_both_napkin_formulas() -> None:
    task = CURATED["napkin"]

    derivations = derive(*task, bound=3)

    # The Verma constraint lets either weight average Z out.
    assert [derivation.formula for derivation in derivations] == [
        "sum_{Z} { p(Z) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }",
        "sum_{Z} { p(Z | X) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }",
    ]
    assert all(d.verify(ADMG.parse(task.graph)) for d in derivations)


@pytest.mark.parametrize(
    ("name", "bound"),
    [
        ("front-door", 4),
        ("napkin", 3),
        ("confounder and mediator", 2),
        ("collider and mediator", 0),
    ],
)
def test_derive_recovers_the_classes_of_id_on_equivalent_queries(
    name: str,
    bound: int,
) -> None:
    task = CURATED[name]
    archive = Archive(task)
    for derivation in derive(*task, bound=bound):
        archive.submit(derivation.formula, derivation)

    for formula in {identify(*task), *identify_equivalent(*task).values()}:
        assert formula is not None
        assert archive.submit(formula).outcome == "duplicate"


def test_verify_rejects_invalid_steps() -> None:
    graph = ADMG.parse(CURATED["front-door"].graph)
    base = Derivation(query("M", conditions="T"), "base")

    assert base.verify(graph)
    assert not Derivation(query("M", "T"), "base").verify(graph)
    # p(M | do(T)) = p(M | T) by Rule 2, but p(Y | do(T)) != p(Y | T).
    path = (query("M", "T"), query("M", conditions="T"))
    assert Derivation(path[0], "do-calculus", (base,), path).verify(graph)
    wrong = (query("Y", "T"), query("Y", conditions="T"))
    premise = Derivation(wrong[1], "base")
    assert not Derivation(wrong[0], "do-calculus", (premise,), wrong).verify(
        graph
    )


def test_derivations_on_random_graphs_identify_their_target() -> None:
    # Independences multiply the classes, which already cost seconds at
    # bound 2 on some graphs of four nodes.
    for task in random_tasks(8, size=5, density=0.4, seed=0):
        graph = ADMG.parse(task.graph)
        falsifier = HPFalsifier(task.graph)
        for derivation in derive(*task, bound=1):
            assert derivation.verify(graph)
            assert falsifier.check(task.target, derivation.formula)


def test_every_step_holds_on_random_graphs() -> None:
    rng = random.Random(0)

    for task in random_tasks(30, size=5, density=0.4, seed=1):
        graph = ADMG.parse(task.graph)
        falsifier = HPFalsifier(task.graph)
        for _ in range(3):
            roles = {node: rng.choice("axb.") for node in graph.nodes}
            roles[rng.choice(sorted(graph.nodes))] = "a"
            target = Query(
                *(
                    frozenset(n for n, r in roles.items() if r == role)
                    for role in "axb"
                )
            )
            for rule, premises in rules(graph, target):
                rhs = written(target, rule, premises)
                assert falsifier.check(str(target), rhs), (task, target, rhs)
            for neighbour in neighbours(graph, target):
                assert falsifier.check(str(target), str(neighbour))


def written(target: Query, rule: str, premises: tuple[Query, ...]) -> str:
    """Write the side of a rule of probability that has the premises."""
    if rule == "chain rule":
        return " ".join(map(str, premises))
    (premise,) = premises
    if rule == "marginalization":
        (z,) = premise.outcomes - target.outcomes
        return f"sum_{{{z}}} {{ {premise} }}"
    (c,) = target.conditions - premise.conditions
    inputs = ", ".join(sorted(premise.inputs))
    return f"icd_{{{c} | {inputs}}} {{ {premise} }}"
