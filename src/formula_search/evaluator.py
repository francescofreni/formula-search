"""The evaluator: an archive of distinct valid formulas for each task.

A task is a graph with a target ``p(y | do(t))``. :meth:`Archive.submit`
judges every formula proposed for it, with checks that run cheapest first:

1. the formula must parse in hiprof's grammar, be observational, and be a
   kernel of ``y`` given at most ``t``;
2. HiProF must accept it as equal to the target in the graph;
3. it joins the class of a formula already archived if HiProF accepts the
   two as equal on the complete DAG over the same variables, whose model
   imposes no constraint, so that formulas of different classes are
   different functions of the observed law; only the classes with its
   fingerprint (:mod:`~formula_search.fingerprints`) are checked, and an
   identical formula needs no check;
4. the first formula of a new class records the single edges whose addition
   to the graph breaks it, and its :func:`features`.

A rejection by HiProF is exact, as it exhibits a model on which the two sides
differ; an acceptance is relative to its linear Gaussian models and wrong
with probability at most 1e-14 per check, so a union bound over the
``checks`` of an archive bounds its false acceptances. A formula submitted
with a :class:`~formula_search.derivation.Derivation` that derives it is
derivation-certified, and any other valid formula verification-supported.
:meth:`Archive.save` writes the classes and every verdict as JSON.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias

from hiprof import CheckResult, HPFalsifier
from hiprof.formula import parse_and_validate
from hiprof.formula.formula import (
    BaseKernel,
    BaseQuotient,
    Formula,
    InternalConditionalDivision,
    Marginalisation,
    Product,
)
from hiprof.formula.parser import parse

from .derivation import Derivation
from .docalculus import Query
from .fingerprints import Fingerprint, FixedModel
from .graph import ADMG

Source: TypeAlias = str | Derivation | None


class Task(NamedTuple):
    """The target ``p(outcome | do(treatment))`` in a graph."""

    graph: str
    treatment: str
    outcome: str

    @property
    def query(self) -> Query:
        return Query(frozenset({self.outcome}), frozenset({self.treatment}))

    @property
    def target(self) -> str:
        return str(self.query)


@dataclass(frozen=True)
class Features:
    """Features of a formula, as written, that bear on its estimation.

    :param kernels: The number of base terms ``p(A | B)``, two for each
        quotient of base terms.
    :param summed: The variables summed out, in the order written.
    :param largest_conditioning_set: The largest number of inputs ``B`` of a
        base term.
    :param quotient: Whether the formula has a quotient of base terms.
    :param icd: Whether it has an internal conditional division.
    """

    kernels: int
    summed: tuple[str, ...]
    largest_conditioning_set: int
    quotient: bool
    icd: bool


@dataclass(frozen=True)
class FormulaClass:
    """Valid formulas that are one function of the observed law.

    :param formula: The first formula found in the class.
    :param source: How that formula was found.
    :param evidence: HiProF's acceptance of the formula in the task's graph.
    :param broken_by: The edges whose addition to the graph makes the
        formula invalid.
    :param features: The features of the formula.
    """

    formula: str
    source: Source
    evidence: CheckResult
    broken_by: frozenset[str]
    features: Features


@dataclass(frozen=True)
class Verdict:
    """The evaluator's verdict on a formula proposed for a task.

    :param formula: The formula.
    :param source: How the formula was found, such as a template with its
        slots, or a derivation.
    :param outcome: Whether the formula is ``"malformed"`` for the task,
        ``"falsified"``, or valid, and then a ``"duplicate"`` of an archived
        class or in a ``"new"`` one.
    :param class_index: The index of the class of a valid formula.
    :param reason: Why the formula is malformed.
    """

    formula: str
    source: Source
    outcome: Literal["malformed", "falsified", "duplicate", "new"]
    class_index: int | None = None
    reason: str = ""

    @property
    def certified(self) -> bool:
        """Whether the formula is derivation-certified."""
        return isinstance(self.source, Derivation)


class Archive:
    """The classes of valid formulas found for a task.

    :ivar classes: The classes, in the order found.
    :ivar log: The verdict on every formula submitted, in order.
    :ivar checks: The number of checks run by HiProF.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        self.classes: list[FormulaClass] = []
        self.log: list[Verdict] = []
        self.checks = 0
        self._graph = ADMG.parse(task.graph)
        self._model = FixedModel(self._graph.nodes)
        self._groups: defaultdict[Fingerprint, list[int]] = defaultdict(list)

    def submit(self, formula: str, source: Source = None) -> Verdict:
        """Judge a formula proposed for the task, and log the verdict.

        :raises ValueError: If ``source`` is a derivation that does not
            derive ``formula`` for the target in the task's graph.
        """
        if isinstance(source, Derivation) and not (
            (source.query, source.formula) == (self.task.query, formula)
            and source.verify(self._graph)
        ):
            raise ValueError("The derivation does not derive the formula.")
        verdict = self._judge(formula, source)
        self.log.append(verdict)
        return verdict

    def save(self, path: str | Path) -> None:
        """Write the task, the checks, the classes, and the log as JSON.

        Fields left at their default are omitted, sets are written as sorted
        lists, queries in hiprof's syntax, and bounds as floats.
        """
        archive = {
            "task": self.task._asdict(),
            "checks": self.checks,
            "classes": self.classes,
            "log": self.log,
        }
        Path(path).write_text(json.dumps(_json(archive), indent=2) + "\n")

    def _judge(self, formula: str, source: Source) -> Verdict:
        task = self.task
        try:
            signature = parse_and_validate(formula).signature
            if re.search(r"\bdo\b", formula) or (
                {str(v) for v in signature.outputs} != {task.outcome}
                or not {str(v) for v in signature.inputs} <= {task.treatment}
            ):
                raise ValueError(
                    "The formula must be observational and of type "
                    f"{task.outcome} | {task.treatment}."
                )
            evidence = self._check(task.graph, task.target, formula)
        except ValueError as error:
            return Verdict(formula, source, "malformed", reason=str(error))
        if not evidence:
            return Verdict(formula, source, "falsified")

        group = self._groups[
            self._model.fingerprint(formula, {task.outcome}, {task.treatment})
        ]
        complete = str(self._graph.complete())
        for index in group:
            known = self.classes[index].formula
            if known == formula or self._check(complete, known, formula):
                return Verdict(formula, source, "duplicate", index)

        broken_by = frozenset(
            edge
            for edge in self._additions()
            if not self._check(f"{task.graph}; {edge}", task.target, formula)
        )
        group.append(len(self.classes))
        self.classes.append(
            FormulaClass(
                formula, source, evidence, broken_by, features(formula)
            )
        )
        return Verdict(formula, source, "new", len(self.classes) - 1)

    def _check(self, graph: str, lhs: str, rhs: str) -> CheckResult:
        self.checks += 1
        return HPFalsifier(graph).check(lhs, rhs)

    def _additions(self) -> Iterator[str]:
        """Yield the edges that the graph can gain and stay acyclic."""
        graph = self._graph
        for parent, child in permutations(sorted(graph.nodes), 2):
            cyclic = child in graph.ancestors({parent})
            if (parent, child) not in graph.directed and not cyclic:
                yield f"{parent} -> {child}"
        for first, second in combinations(sorted(graph.nodes), 2):
            if frozenset({first, second}) not in graph.bidirected:
                yield f"{first} <-> {second}"


def features(formula: str) -> Features:
    """Return the features of a formula, as it is written.

    :raises ValueError: If the formula does not parse.
    """
    nodes = list(_nodes(parse(formula)))
    terms = [node for node in nodes if isinstance(node, BaseKernel)]
    return Features(
        kernels=len(terms),
        summed=tuple(
            str(variable)
            for node in nodes
            if isinstance(node, Marginalisation)
            for variable in node.variables
        ),
        largest_conditioning_set=max(len(term.inputs) for term in terms),
        quotient=any(isinstance(node, BaseQuotient) for node in nodes),
        icd=any(
            isinstance(node, InternalConditionalDivision) for node in nodes
        ),
    )


def _nodes(formula: Formula) -> Iterator[Formula]:
    """Yield the nodes of a formula's syntax tree, parents first."""
    yield formula
    if isinstance(formula, Product):
        children: tuple[Formula, ...] = formula.factors
    elif isinstance(formula, BaseQuotient):
        children = (formula.numerator, formula.denominator)
    elif isinstance(formula, Marginalisation | InternalConditionalDivision):
        children = (formula.body,)
    else:
        children = ()
    for child in children:
        yield from _nodes(child)


def _json(value: object) -> object:
    """Convert records into JSON values.

    A dataclass becomes its fields that differ from their default, a query
    its hiprof syntax, a set a sorted list, and a fraction a float.
    """
    if isinstance(value, Query):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: _json(getattr(value, field.name))
            for field in fields(value)
            if getattr(value, field.name) != field.default
        }
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, list | tuple):
        return [_json(item) for item in value]
    if isinstance(value, Fraction):
        return float(value)
    return value
