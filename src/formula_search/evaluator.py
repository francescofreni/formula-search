"""The evaluator: an archive of distinct valid formulas for each task.

A task is a graph with a target ``p(y | do(t))``. :meth:`Archive.submit`
judges every formula proposed for it, with checks that run cheapest first:

1. the formula must parse in hiprof's grammar, be observational, and be a
   kernel of ``y`` given at most ``t``;
2. HiProF must accept it as equal to the target in the graph;
3. it joins the class of a formula already archived if HiProF accepts the
   two as equal on the complete DAG over the same variables, whose model
   imposes no constraint, so that formulas of different classes are
   different functions of the observed law;
4. the first formula of a new class records the single edges whose addition
   to the graph breaks it.

A rejection by HiProF is exact, as it exhibits a model on which the two sides
differ; an acceptance is relative to its linear Gaussian models and wrong
with probability at most 1e-14 per check, so a union bound over the
``checks`` of an archive bounds its false acceptances. A formula submitted
with a :class:`~formula_search.derivation.Derivation` that derives it is
derivation-certified, and any other valid formula verification-supported.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Literal, NamedTuple

from hiprof import CheckResult, HPFalsifier
from hiprof.formula import parse_and_validate

from .derivation import Derivation
from .docalculus import Query
from .graph import ADMG


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
class FormulaClass:
    """Valid formulas that are one function of the observed law.

    :param formula: The first formula found in the class.
    :param evidence: HiProF's acceptance of the formula in the task's graph.
    :param broken_by: The edges whose addition to the graph makes the
        formula invalid.
    """

    formula: str
    evidence: CheckResult
    broken_by: frozenset[str]


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
    source: object
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

    def submit(self, formula: str, source: object = None) -> Verdict:
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

    def _judge(self, formula: str, source: object) -> Verdict:
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

        complete = str(self._graph.complete())
        for index, known in enumerate(self.classes):
            if self._check(complete, known.formula, formula):
                return Verdict(formula, source, "duplicate", index)

        broken_by = frozenset(
            edge
            for edge in self._additions()
            if not self._check(f"{task.graph}; {edge}", task.target, formula)
        )
        self.classes.append(FormulaClass(formula, evidence, broken_by))
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
