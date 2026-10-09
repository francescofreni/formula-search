from __future__ import annotations

from formula_search.baselines import identify
from formula_search.graph import ADMG
from formula_search.suite import CURATED, random_tasks


def test_random_tasks_are_identifiable_and_reproducible() -> None:
    tasks = random_tasks(20, size=6, density=0.4, seed=0)

    assert tasks == random_tasks(20, size=6, density=0.4, seed=0)
    for task in tasks:
        graph = ADMG.parse(task.graph)
        assert len(graph.nodes) == 6
        assert task.treatment in graph.ancestors({task.outcome})
        assert identify(*task) is not None


def test_curated_tasks_include_controls() -> None:
    assert identify(*CURATED["bow arc"]) is None
    assert not ADMG.parse(CURATED["front-door"].graph).implies_constraints()
    assert ADMG.parse(CURATED["napkin"].graph).implies_constraints()
