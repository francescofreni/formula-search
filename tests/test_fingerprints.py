from __future__ import annotations

from formula_search.fingerprints import FixedModel

VARIABLES = {"T", "Y", "Z1", "Z2"}


def test_rewrites_share_a_fingerprint_and_other_functions_do_not() -> None:
    model = FixedModel(VARIABLES)
    fingerprints = [
        model.fingerprint(formula, {"Y"}, {"T"})
        for formula in [
            "sum_{Z1} { p(Z1) p(Y | T, Z1) }",
            "sum_{Z1, Z2} { p(Z1) p(Z2) p(Y | T, Z1, Z2) }",
            "sum_{Z1, Z2} { p(Z1, Z2) p(Y | T, Z1, Z2) }",
            "sum_{Z1, Z2} { p(Z1) p(Z2 | Z1) p(Y | T, Z1, Z2) }",
        ]
    ]

    assert fingerprints[2] == fingerprints[3]
    assert len(set(fingerprints)) == 3


def test_inputs_that_a_formula_lacks_count_as_unused() -> None:
    model = FixedModel(VARIABLES)

    marginal = model.fingerprint("p(Y)", {"Y"}, {"T"})

    assert marginal == model.fingerprint(
        "sum_{T'} { p(T') p(Y | T') }", {"Y"}, {"T"}
    )
    assert marginal != model.fingerprint("p(Y | T)", {"Y"}, {"T"})


def test_the_model_over_the_same_variables_is_fixed() -> None:
    formula = "sum_{Z1} { p(Z1) p(Y | T, Z1) }"

    assert FixedModel(VARIABLES).fingerprint(
        formula, {"Y"}, {"T"}
    ) == FixedModel(sorted(VARIABLES, reverse=True)).fingerprint(
        formula, {"Y"}, {"T"}
    )
