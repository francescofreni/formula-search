from __future__ import annotations

from formula_search.baselines import template_formulas
from formula_search.evaluator import Archive, Task
from formula_search.suite import CURATED


def archive_templates(task: Task, *templates: str) -> Archive:
    """Archive the formulas of the templates, or of all if none are named."""
    archive = Archive(task)
    for formula, source in template_formulas(*task).items():
        if not templates or source.split(":")[0] in templates:
            archive.submit(formula, source)
    return archive


def test_empty_slots_drop_their_sums_and_factors() -> None:
    formulas = template_formulas("T -> M; M -> Y; T <-> Y", "T", "Y")

    assert formulas == {
        "p(Y | T)": "adjustment: S = {}",
        "sum_{M} { p(M) p(Y | M, T) }": "adjustment: S = {M}",
        "p(Y)": "front-door: S = {}",
        "sum_{M} { p(M | T) sum_{T'} { p(T') p(Y | M, T') } }": (
            "front-door: S = {M}"
        ),
    }


def test_two_slots_take_disjoint_sets() -> None:
    formulas = template_formulas("Z -> X; X -> Y; W -> Y", "X", "Y")

    assert formulas[
        "sum_{Z} { p(Z) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }"
    ] == ("napkin: Z = {Z}, W = {W}")


def test_the_front_door_graph_has_one_class() -> None:
    archive = archive_templates(CURATED["front-door"])

    assert [known.formula for known in archive.classes] == [
        "sum_{M} { p(M | T) sum_{T'} { p(T') p(Y | M, T') } }"
    ]


def test_front_door_formulas_give_48_classes_on_admg_iv() -> None:
    archive = archive_templates(CURATED["front-door (iv)"], "front-door")

    assert len(archive.classes) == 48


def test_adjustment_formulas_give_14_classes_on_the_rich_graph() -> None:
    archive = archive_templates(CURATED["adjustment-rich"], "adjustment")

    assert len(archive.classes) == 14


def test_repeated_runs_give_the_same_archive() -> None:
    first, second = (
        archive_templates(CURATED["two adjustment sets"]) for _ in range(2)
    )

    assert first.classes == second.classes
