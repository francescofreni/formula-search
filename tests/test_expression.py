from __future__ import annotations

from hiprof.formula import parse_and_validate

from formula_search.expression import (
    ICD,
    Product,
    Sum,
    Term,
    average,
    condition,
    marginal,
    product,
    render,
)


def p(outputs: str, inputs: str = "") -> Term:
    return Term(frozenset(outputs.split()), frozenset(inputs.split()))


# The napkin kernel of Y given X and Z (Freni et al., 2026, Section S2.1).
NAPKIN = ICD(
    frozenset({"X"}),
    Sum(frozenset({"W"}), Product((p("W"), p("X Y", "W Z")))),
)


def test_product_merges_base_terms_by_the_chain_rule() -> None:
    assert product([p("A", "T"), p("B", "A T")]) == p("A B", "T")
    assert product([p("A"), Product((p("B", "A"), p("C", "B")))]) == Product(
        (p("A B"), p("C", "B"))
    )


def test_product_keeps_terms_with_other_inputs() -> None:
    assert product([p("T"), p("Y", "M T")]) == Product((p("T"), p("Y", "M T")))


def test_marginal_of_a_base_term_is_a_base_term() -> None:
    assert marginal(p("A B", "C"), {"B"}) == p("A", "C")


def test_marginal_drops_factors_that_integrate_to_one() -> None:
    # Summing Y out of the kernel Q of Freni et al., Example 7.
    kernel = Product((p("T"), p("B", "A T"), p("Y", "A B C T")))

    assert marginal(kernel, {"Y"}) == Product((p("T"), p("B", "A T")))
    assert marginal(kernel, {"T", "Y"}) == Sum(
        frozenset({"T"}), Product((p("T"), p("B", "A T")))
    )


def test_marginal_of_a_sum_is_one_sum() -> None:
    kernel = Sum(frozenset({"W"}), Product((p("W"), p("X Y", "W Z"))))

    assert marginal(kernel, {"Y"}) == Sum(
        frozenset({"W"}), Product((p("W"), p("X", "W Z")))
    )


def test_condition_on_leading_factors_keeps_the_others() -> None:
    assert condition(p("A B", "C"), {"A"}) == p("B", "A C")
    assert condition(Product((p("W"), p("X", "W Z"))), {"W"}) == p("X", "W Z")


def test_condition_of_a_marginal_is_an_internal_conditional_division() -> None:
    kernel = NAPKIN.body

    assert condition(kernel, {"X"}) == NAPKIN
    assert NAPKIN.outputs == {"Y"}
    assert NAPKIN.inputs == {"X", "Z"}


def test_average_over_inputs_of_a_base_term_drops_them() -> None:
    assert average(p("Y", "C T U"), {"U"}) == p("Y", "C T")


def test_average_weights_by_the_conditional_of_the_averaged_inputs() -> None:
    assert average(NAPKIN, {"Z"}) == Sum(
        frozenset({"Z"}), Product((p("Z", "X"), NAPKIN))
    )


def test_render_primes_summed_variables_that_are_in_use() -> None:
    front_door = Sum(
        frozenset({"M"}),
        Product(
            (
                p("M", "T"),
                Sum(frozenset({"T"}), Product((p("T"), p("Y", "M T")))),
            )
        ),
    )
    joint = Sum(frozenset({"T1"}), Product((p("T1"), p("Y", "T1 T2"))))

    assert render(front_door) == (
        "sum_{M} { p(M | T) sum_{T'} { p(T') p(Y | M, T') } }"
    )
    assert render(joint, reserved={"T1"}) == (
        "sum_{T1'} { p(T1') p(Y | T1', T2) }"
    )


def test_render_writes_internal_conditional_divisions() -> None:
    assert render(ICD(frozenset({"A"}), p("A B"))) == "icd_{A |} { p(A, B) }"
    assert render(NAPKIN) == (
        "icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } }"
    )


def test_rendered_expressions_are_admissible_with_the_same_type() -> None:
    expression = average(NAPKIN, {"Z"})
    signature = parse_and_validate(render(expression)).signature

    assert {str(v) for v in signature.outputs} == expression.outputs
    assert {str(v) for v in signature.inputs} == expression.inputs
