from __future__ import annotations

import json
from pathlib import Path

import pytest

from formula_search.derivation import derive
from formula_search.evaluator import Archive, Features, Task, features
from formula_search.suite import CURATED

FRONT_DOOR = "sum_{M} { p(M | T) sum_{T'} { p(Y | M, T') p(T') } }"
NAPKIN = "sum_{Z} { icd_{X | Z} { sum_{W} { p(Y, X | Z, W) p(W) } } p(Z) }"


def test_submit_accepts_identifying_formulas_and_falsifies_others() -> None:
    archive = Archive(CURATED["front-door"])

    assert archive.submit(FRONT_DOOR).outcome == "new"
    assert archive.submit("p(Y | T)").outcome == "falsified"
    assert [known.formula for known in archive.classes] == [FRONT_DOOR]
    assert [verdict.outcome for verdict in archive.log] == ["new", "falsified"]
    assert archive.checks > 2


def test_rewrites_of_a_formula_are_duplicates() -> None:
    archive = Archive(CURATED["front-door"])
    archive.submit(FRONT_DOOR, source="ID")
    quotient = FRONT_DOOR.replace("p(M | T)", "(p(M, T) / p(T))")

    verdict = archive.submit(quotient, source="rewrite")

    assert (verdict.outcome, verdict.class_index) == ("duplicate", 0)


def test_a_duplicate_is_checked_against_its_own_class_only() -> None:
    archive = Archive(CURATED["two adjustment sets"])
    for formula in [
        "sum_{Z1} { p(Z1) p(Y | T, Z1) }",
        "sum_{Z1, Z2} { p(Z1) p(Z2) p(Y | T, Z1, Z2) }",
        "sum_{Z1, Z2} { p(Z1, Z2) p(Y | T, Z1, Z2) }",
    ]:
        archive.submit(formula)
    checks = archive.checks

    verdict = archive.submit(
        "sum_{Z1, Z2} { p(Z1) p(Z2 | Z1) p(Y | T, Z1, Z2) }"
    )

    assert (verdict.outcome, verdict.class_index) == ("duplicate", 2)
    # One check of validity, and one against the class of its fingerprint.
    assert archive.checks == checks + 2


def test_adjustment_sets_fall_in_classes_with_their_assumptions() -> None:
    archive = Archive(CURATED["two adjustment sets"])
    one = archive.submit("sum_{Z1} { p(Y | T, Z1) p(Z1) }")
    both = archive.submit("sum_{Z1, Z2} { p(Y | T, Z1, Z2) p(Z1, Z2) }")

    assert (one.outcome, both.outcome) == ("new", "new")
    first, second = archive.classes
    assert "Z2 -> T" in first.broken_by
    assert "Z2 -> T" not in second.broken_by


@pytest.mark.parametrize(
    ("formula", "reason"),
    [
        pytest.param("p(Y | T", "Invalid formula syntax", id="syntax"),
        pytest.param(
            "sum_{Q} { p(Q) p(Y | Q, T) }",
            "observed variables",
            id="unknown variable",
        ),
        pytest.param("p(Y | do(T))", "observational", id="intervention"),
        pytest.param("p(Y | M, T)", "of type Y | T", id="extra input"),
        pytest.param("p(M | T)", "of type Y | T", id="other outcome"),
    ],
)
def test_malformed_formulas_record_why(formula: str, reason: str) -> None:
    verdict = Archive(CURATED["front-door"]).submit(formula)

    assert verdict.outcome == "malformed"
    assert reason in verdict.reason


def test_formulas_submitted_with_their_derivation_are_certified() -> None:
    task = CURATED["front-door"]
    archive = Archive(task)
    (derivation,) = derive(*task, bound=4)

    assert archive.submit(derivation.formula, derivation).certified
    assert not archive.submit(FRONT_DOOR, "ID").certified
    with pytest.raises(ValueError, match="does not derive"):
        archive.submit("p(Y | T)", derivation)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        pytest.param(
            FRONT_DOOR,
            Features(3, ("M", "T'"), 2, quotient=False, icd=False),
            id="front-door",
        ),
        pytest.param(
            FRONT_DOOR.replace("p(M | T)", "(p(M, T) / p(T))"),
            Features(4, ("M", "T'"), 2, quotient=True, icd=False),
            id="quotient",
        ),
        pytest.param(
            NAPKIN,
            Features(3, ("Z", "W"), 2, quotient=False, icd=True),
            id="napkin",
        ),
    ],
)
def test_features_describe_the_formula_as_written(
    formula: str,
    expected: Features,
) -> None:
    assert features(formula) == expected


def test_save_writes_the_classes_and_every_verdict(tmp_path: Path) -> None:
    task = CURATED["front-door"]
    archive = Archive(task)
    (derivation,) = derive(*task, bound=4)
    archive.submit(derivation.formula, derivation)
    archive.submit("p(Y | T)", "adjustment: S = {}")
    archive.submit("p(Y | T", "a typo")

    archive.save(tmp_path / "front-door.json")
    saved = json.loads((tmp_path / "front-door.json").read_text())

    assert saved["task"] == {
        "graph": task.graph,
        "treatment": "T",
        "outcome": "Y",
    }
    assert saved["checks"] == archive.checks
    (record,) = saved["classes"]
    assert record["formula"] == derivation.formula
    assert record["evidence"]["false_acceptance_bound"] < 1e-14
    assert record["broken_by"] == ["M <-> T", "M <-> Y", "T -> Y"]
    assert record["features"]["summed"] == ["M", "T'"]
    # The derivation is a tree of rules over queries in hiprof's syntax.
    assert record["source"]["query"] == "p(Y | do(T))"
    assert record["source"]["rule"] == "marginalization"
    assert [entry["outcome"] for entry in saved["log"]] == [
        "new",
        "falsified",
        "malformed",
    ]
    assert "class_index" not in saved["log"][1]
    assert "Invalid formula syntax" in saved["log"][2]["reason"]


@pytest.mark.parametrize(
    ("task", "formula", "valid"),
    [
        # hiprof-eq/examples/cross_graph_verification.py
        pytest.param(
            CURATED["confounder and mediator"],
            "sum_{C} { p(Y | C, T) p(C) }",
            True,
            id="adjustment, first graph",
        ),
        pytest.param(
            CURATED["confounder and mediator"],
            "p(Y | T)",
            False,
            id="conditional, first graph",
        ),
        pytest.param(
            CURATED["collider and mediator"],
            "sum_{C} { p(Y | C, T) p(C) }",
            False,
            id="adjustment, second graph",
        ),
        pytest.param(
            CURATED["collider and mediator"],
            "p(Y | T)",
            True,
            id="conditional, second graph",
        ),
        pytest.param(
            CURATED["collider and mediator"],
            FRONT_DOOR,
            True,
            id="front-door, second graph",
        ),
        # hiprof-eq/examples/napkin.py
        pytest.param(CURATED["napkin"], NAPKIN, True, id="napkin"),
        pytest.param(
            CURATED["napkin"], "p(Y | X)", False, id="napkin, p(Y|X)"
        ),
        # hiprof-eq/examples/frontdoor.py, ADMG (iv)
        pytest.param(
            CURATED["front-door (iv)"],
            "sum_{A, C, D} { p(A, C, D | X) "
            "sum_{X'} { p(Y | X', A, C, D) p(X') } }",
            True,
            id="front-door (iv), {A, C, D}",
        ),
        pytest.param(
            CURATED["front-door (iv)"],
            "sum_{A} { p(A | X) sum_{X'} { p(Y | X', A) p(X') } }",
            False,
            id="front-door (iv), {A}",
        ),
    ],
)
def test_examples_of_hiprof_reproduce(
    task: Task,
    formula: str,
    valid: bool,
) -> None:
    verdict = Archive(task).submit(formula)

    assert verdict.outcome == ("new" if valid else "falsified")
