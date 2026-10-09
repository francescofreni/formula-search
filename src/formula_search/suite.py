"""The graph suite: curated and random tasks for one treatment and one outcome.

The curated tasks include a negative control, the bow arc, whose target is
not identifiable; graphs that imply no equality constraints, such as the
front-door graph, have a single class of identifying formulas
(:meth:`~formula_search.graph.ADMG.implies_constraints`).
"""

from __future__ import annotations

import random
from itertools import combinations
from string import ascii_uppercase

from .baselines.id import identify
from .evaluator import Task
from .graph import ADMG

CURATED = {
    "front-door": Task("T -> M; M -> Y; T <-> Y", "T", "Y"),
    "napkin": Task("W -> Z; Z -> X; X -> Y; W <-> X; W <-> Y", "X", "Y"),
    "two adjustment sets": Task("Z1 -> T; Z1 -> Y; T -> Y; Z2 -> Y", "T", "Y"),
    # Freni et al. (2026), Figure 1: an adjustment and a front-door formula
    # on the first graph, p(Y | T) and the front-door formula on the second.
    "confounder and mediator": Task(
        "C -> T; C -> Y; T -> M; M -> Y", "T", "Y"
    ),
    "collider and mediator": Task("T -> M; M -> Y; T -> C; Y -> C", "T", "Y"),
    "adjustment-rich": Task(
        "A -> T; A -> Y; B -> A; B -> Y; C -> T; D -> Y; E -> D; T -> Y; "
        "E <-> C",
        "T",
        "Y",
    ),
    # ADMGs (ii) to (iv) of hiprof-eq/examples/frontdoor.py, after Figure 1
    # of Wienöbst et al. (2024).
    "front-door (ii)": Task(
        "X -> A; A -> B; A -> C; B -> D; C -> D; D -> Y; X <-> Y", "X", "Y"
    ),
    "front-door (iii)": Task(
        "X -> A; A -> Y; X <-> Y; B -> Y; C -> Y; D -> B; D -> C", "X", "Y"
    ),
    "front-door (iv)": Task(
        "X -> A; A -> B; B -> C; B -> Y; C -> Y; D -> Y; E -> C; F -> E; "
        "F -> G; G -> Y; X <-> Y; A <-> D",
        "X",
        "Y",
    ),
    # Freni et al. (2026), Example 7, where ID splits its product.
    "split product": Task(
        "T -> A; A -> B; B -> C; C -> Y; T <-> B; B <-> Y", "T", "Y"
    ),
    # Yvernes et al. (2026), Figure 15 and Section 5.2.
    "equivalent queries": Task(
        "X -> Z4; Z4 -> Y; Z1 -> X; Z2 -> Z1; Z3 -> Z2; Z3 -> X; Z5 -> Z1; "
        "Z5 -> Z4; X <-> Z2; Z3 <-> Z2; Z2 <-> Y; Z4 <-> Y; Z5 <-> Z4",
        "X",
        "Y",
    ),
    "Sachs": Task(
        "PKA -> JNK; PKA -> P38; PKA -> MEK; PKA -> RAF; PKA -> ERK; "
        "PKA -> AKT; PKC -> JNK; PKC -> P38; PKC -> MEK; PKC -> RAF; "
        "PKC -> PKA; PIP3 -> AKT; MEK -> ERK; RAF -> MEK; ERK -> AKT; "
        "PIP2 -> PKC; PIP3 -> PIP2; PLCG -> PKC; PLCG -> PIP2; PLCG -> PIP3",
        "MEK",
        "P38",
    ),
    "bow arc": Task("T -> Y; T <-> Y", "T", "Y"),
}


def random_tasks(
    count: int,
    size: int,
    density: float,
    seed: int,
) -> list[Task]:
    """Return random tasks whose target is identifiable and not ``p(Y)``.

    Each graph orders ``size`` nodes at random, including ``T`` and ``Y``,
    and has each directed edge along the order with probability ``density``
    and each bidirected edge with probability ``density / 2``. Graphs are
    kept if ``T`` is an ancestor of ``Y`` and ID identifies the target.
    """
    rng = random.Random(seed)
    names = ["T", "Y", *(c for c in ascii_uppercase if c not in "TY")]
    tasks: list[Task] = []
    while len(tasks) < count:
        order = rng.sample(names[:size], size)
        pairs = list(combinations(order, 2))
        graph = ADMG(
            frozenset(order),
            frozenset(pair for pair in pairs if rng.random() < density),
            frozenset(
                frozenset(pair) for pair in pairs if rng.random() < density / 2
            ),
        )
        task = Task(str(graph), "T", "Y")
        if "T" in graph.ancestors({"Y"}) and identify(*task) is not None:
            tasks.append(task)
    return tasks
