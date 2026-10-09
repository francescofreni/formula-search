"""Formulas from templates, such as adjustment and front-door formulas.

A template is a formula with slots, each of which takes a set of the
variables other than the treatments ``T`` and the outcomes ``Y``:

- adjustment, ``sum_{S} { p(S) p(Y | T, S) }``;
- front-door, ``sum_{S} { p(S | T) sum_{T'} { p(T') p(Y | T', S) } }``;
- front-door with adjustment, ``sum_{C} { p(C) sum_{S} { p(S | T, C)
  sum_{T'} { p(T' | C) p(Y | T', S, C) } } }``;
- napkin, ``sum_{Z} { p(Z) icd_{T | Z} { sum_{W} { p(W) p(T, Y | W, Z) } } }``.

An empty slot drops the sums and the factors over it, so that adjustment for
the empty set gives ``p(Y | T)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import product as cartesian_product

from ..expression import Expression, Term, condition, marginal, product, render
from ..graph import ADMG
from .id import parse_query

Slots = frozenset[str]


def template_formulas(
    graph: str,
    treatments: str | Iterable[str],
    outcomes: str | Iterable[str],
) -> dict[str, str]:
    """Return the formulas of every template, for every filling of its slots.

    The slots of a template take disjoint sets, in every way.

    :param graph: Acyclic directed mixed graph in hiprof's syntax.
    :param treatments: Treatment variable, or variables.
    :param outcomes: Outcome variable, or variables.
    :returns: For each formula, the first template and slots that give it,
        such as ``"front-door: S = {M}"``.
    :raises ValueError: If the graph is invalid, the outcomes are empty, or
        the variables overlap or are not nodes of the graph.
    """
    admg = ADMG.parse(graph)
    t, y, _ = parse_query(admg, treatments, outcomes)
    others = sorted(admg.nodes - t - y)
    formulas: dict[str, str] = {}
    for name, (slots, template) in TEMPLATES.items():
        for sets in _disjoint_sets(others, len(slots)):
            formula = render(template(t, y, *sets), reserved=t)
            filling = ", ".join(
                f"{slot} = {{{', '.join(sorted(variables))}}}"
                for slot, variables in zip(slots, sets, strict=True)
            )
            formulas.setdefault(formula, f"{name}: {filling}")
    return formulas


def _adjustment(t: Slots, y: Slots, s: Slots) -> Expression:
    return marginal(product([*_base(s), Term(y, t | s)]), s)


def _front_door(t: Slots, y: Slots, s: Slots) -> Expression:
    return _front_door_with_adjustment(t, y, frozenset(), s)


def _front_door_with_adjustment(
    t: Slots,
    y: Slots,
    c: Slots,
    s: Slots,
) -> Expression:
    effect = marginal(product([Term(t, c), Term(y, t | s | c)]), t)
    mediated = marginal(product([*_base(s, t | c), effect]), s)
    return marginal(product([*_base(c), mediated]), c)


def _napkin(t: Slots, y: Slots, z: Slots, w: Slots) -> Expression:
    joint = marginal(product([*_base(w), Term(t | y, z | w)]), w)
    return marginal(product([*_base(z), condition(joint, t)]), z)


TEMPLATES: dict[str, tuple[tuple[str, ...], Callable[..., Expression]]] = {
    "adjustment": (("S",), _adjustment),
    "front-door": (("S",), _front_door),
    "front-door with adjustment": (("C", "S"), _front_door_with_adjustment),
    "napkin": (("Z", "W"), _napkin),
}


def _base(outputs: Slots, inputs: Slots = frozenset()) -> list[Expression]:
    """Return the factors of the base term, none if it has no outputs."""
    return [Term(outputs, inputs)] if outputs else []


def _disjoint_sets(
    variables: list[str],
    count: int,
) -> Iterator[tuple[Slots, ...]]:
    """Yield every tuple of ``count`` disjoint sets of the variables.

    Each variable is placed in one of the sets, or in none, the set 0.
    """
    for places in cartesian_product(range(count + 1), repeat=len(variables)):
        sets: list[set[str]] = [set() for _ in range(count + 1)]
        for variable, place in zip(variables, places, strict=True):
            sets[place].add(variable)
        yield tuple(map(frozenset, sets[1:]))
