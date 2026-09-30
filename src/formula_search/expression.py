"""Kernel expressions in hiprof's formula grammar.

Every expression denotes a kernel, a conditional density over its
``outputs`` given its ``inputs`` (Freni et al., 2026). The
constructors :func:`product`, :func:`marginal`, :func:`condition`, and
:func:`average` build admissible expressions and simplify them with
identities that hold for every positive observational law.
:func:`render` writes an expression in hiprof's syntax, and
:func:`render_query` writes a query ``p(y | do(x), w)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Set
from dataclasses import dataclass
from itertools import chain, combinations, count
from typing import TypeAlias


@dataclass(frozen=True)
class Term:
    """The base term ``p(outputs | inputs)``."""

    outputs: frozenset[str]
    inputs: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Product:
    """A sequential product.

    The inputs of each factor are inputs of the product or outputs of
    earlier factors.
    """

    factors: tuple[Expression, ...]

    @property
    def outputs(self) -> frozenset[str]:
        return _outputs(self.factors)

    @property
    def inputs(self) -> frozenset[str]:
        return _inputs(self.factors) - self.outputs


@dataclass(frozen=True)
class Sum:
    """The marginalisation ``sum_{variables} { body }``."""

    variables: frozenset[str]
    body: Expression

    @property
    def outputs(self) -> frozenset[str]:
        return self.body.outputs - self.variables

    @property
    def inputs(self) -> frozenset[str]:
        return self.body.inputs


@dataclass(frozen=True)
class ICD:
    """The internal conditional division ``icd_{given | inputs} { body }``.

    It denotes the kernel of the other outputs of ``body`` given ``given``
    and the inputs of ``body``.
    """

    given: frozenset[str]
    body: Expression

    @property
    def outputs(self) -> frozenset[str]:
        return self.body.outputs - self.given

    @property
    def inputs(self) -> frozenset[str]:
        return self.body.inputs | self.given


Expression: TypeAlias = Term | Product | Sum | ICD


def product(factors: Iterable[Expression]) -> Expression:
    """Multiply factors that are listed in a sequential order.

    Nested products are flattened, and base terms ``p(A | B) p(C | A, B)``
    are merged into ``p(A, C | B)``.
    """
    flat: list[Expression] = []
    for factor in factors:
        flat.extend(
            factor.factors if isinstance(factor, Product) else [factor]
        )

    for (i, first), (j, second) in combinations(enumerate(flat), 2):
        if (
            isinstance(first, Term)
            and isinstance(second, Term)
            and second.inputs == first.outputs | first.inputs
        ):
            merged = Term(first.outputs | second.outputs, first.inputs)
            return product(
                flat[:i] + [merged] + flat[i + 1 : j] + flat[j + 1 :]
            )

    return flat[0] if len(flat) == 1 else Product(tuple(flat))


def marginal(expression: Expression, variables: Iterable[str]) -> Expression:
    """Sum ``variables``, a proper subset of the outputs, out of a kernel.

    In a product, a variable that no other factor uses is summed out of the
    factor that outputs it, and a factor whose outputs are all summed out is
    dropped, since it integrates to one.
    """
    variables = frozenset(variables)
    if not variables:
        return expression

    if isinstance(expression, Term):
        return Term(expression.outputs - variables, expression.inputs)

    if isinstance(expression, Sum):
        return marginal(expression.body, expression.variables | variables)

    if isinstance(expression, ICD):
        return condition(
            marginal(expression.body, variables), expression.given
        )

    for index, factor in enumerate(expression.factors):
        others = list(expression.factors)
        del others[index]
        local = (variables & factor.outputs) - _inputs(others)
        if local:
            if local != factor.outputs:  # else the factor sums to one
                others.insert(index, marginal(factor, local))
            return marginal(product(others), variables - local)

    return Sum(variables, expression)


def condition(expression: Expression, given: Iterable[str]) -> Expression:
    """Condition a kernel on ``given``, a proper subset of its outputs.

    The leading factors of a product that output only given variables drop
    out, since the remaining factors carry the conditional.
    """
    given = frozenset(given)
    if not given:
        return expression

    if isinstance(expression, Term):
        return Term(expression.outputs - given, expression.inputs | given)

    if isinstance(expression, ICD):
        return condition(expression.body, expression.given | given)

    if isinstance(expression, Product):
        leading: list[Expression] = []
        rest: list[Expression] = []
        for factor in expression.factors:
            if factor.outputs <= given and not factor.inputs & _outputs(rest):
                leading.append(factor)
            else:
                rest.append(factor)
        if leading:
            return condition(product(rest), given - _outputs(leading))

    return ICD(given, expression)


def average(expression: Expression, variables: Iterable[str]) -> Expression:
    """Average inputs out of a kernel that does not depend on them.

    The result ``sum_{S} { p(S | C) E }``, where ``C`` holds the remaining
    inputs of ``E``, equals ``E`` at any value of ``S`` when ``E`` is
    invariant in ``S``. It reduces to ``p(A | C)`` when ``E`` is the base
    term ``p(A | C, S)``.
    """
    variables = frozenset(variables)
    if not variables:
        return expression

    weights = Term(variables, expression.inputs - variables)
    return marginal(product([weights, expression]), variables)


def render(expression: Expression, reserved: Iterable[str] = ()) -> str:
    """Write an expression in hiprof's syntax.

    A summed variable whose name is already in use is renamed to a primed
    copy, such as ``T'``.

    :param expression: Expression to write.
    :param reserved: Names in use besides the free variables of the
        expression, such as treatments that it does not depend on.
    """
    used = expression.outputs | expression.inputs | frozenset(reserved)
    return _render(expression, {}, used)


def _render(
    expression: Expression,
    names: dict[str, str],
    used: frozenset[str],
) -> str:
    if isinstance(expression, Term):
        outputs = _join(expression.outputs, names)
        if not expression.inputs:
            return f"p({outputs})"
        return f"p({outputs} | {_join(expression.inputs, names)})"

    if isinstance(expression, Product):
        return " ".join(
            _render(factor, names, used) for factor in expression.factors
        )

    if isinstance(expression, Sum):
        names = dict(names)
        for variable in sorted(expression.variables):
            names[variable] = _unused_copy(variable, used)
            used |= {names[variable]}
        variables = _join(expression.variables, names)
        body = _render(expression.body, names, used)
        return f"sum_{{{variables}}} {{ {body} }}"

    given = _join(expression.given, names)
    inputs = _join(expression.body.inputs, names)
    bar = f"{given} | {inputs}" if inputs else f"{given} |"
    body = _render(expression.body, names, used)
    return f"icd_{{{bar}}} {{ {body} }}"


def render_query(
    outcomes: Set[str],
    treatments: Set[str] = frozenset(),
    conditions: Set[str] = frozenset(),
) -> str:
    """Write the query ``p(outcomes | do(treatments), conditions)``."""
    given = sorted(conditions)
    if treatments:
        given.insert(0, f"do({_join(treatments, {})})")
    if not given:
        return f"p({_join(outcomes, {})})"
    return f"p({_join(outcomes, {})} | {', '.join(given)})"


def _join(variables: Iterable[str], names: dict[str, str]) -> str:
    return ", ".join(
        sorted(names.get(variable, variable) for variable in variables)
    )


def _unused_copy(variable: str, used: frozenset[str]) -> str:
    copies = chain(
        (variable, f"{variable}'"),
        (f"{variable}'{index}" for index in count(1)),
    )
    return next(copy for copy in copies if copy not in used)


def _outputs(factors: Iterable[Expression]) -> frozenset[str]:
    return frozenset().union(*(factor.outputs for factor in factors))


def _inputs(factors: Iterable[Expression]) -> frozenset[str]:
    return frozenset().union(*(factor.inputs for factor in factors))
