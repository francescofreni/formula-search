"""Shared legal moves, kernel identity, and session interface.

Traversal policy belongs in depth_first.py or beam.py. This module does
not score candidates or choose which proof branches survive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..astar import _Plan, _Search
from ..baselines.id import parse_query
from ..derivation import Derivation
from ..evaluator import Archive, Task
from ..expression import Term, render
from ..graph import ADMG
from .decisions import DecisionStats, EventSink, advance
from .prompts import PromptTemplate


def _integer(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class _Move:
    id: int
    cost: int
    item: _Plan | Derivation
    description: dict[str, Any]


def _partial(plan: _Plan) -> str:
    """Render a proof template without pretending holes are observations.

    Kernel operators avoid ambiguous dummy-variable scope in unfinished
    trees. Transport weights are selected only when a plan is complete.
    """
    if plan.rule is None:
        return f"UNRESOLVED[{plan.query}]"
    assert plan.member is not None
    if plan.rule == "base":
        body = render(Term(plan.member.outcomes, plan.member.conditions))
    elif plan.rule == "chain rule":
        body = " * ".join(f"({_partial(p)})" for p in plan.premises)
    else:
        premise = plan.premises[0]
        if plan.rule == "marginalization":
            variables = premise.query.outcomes - plan.member.outcomes
            operator = "Marginalize"
        else:
            variables = plan.member.conditions - premise.query.conditions
            operator = "Condition"
        body = (
            f"{operator}[{', '.join(sorted(variables))}]({_partial(premise)})"
        )
    if plan.member != plan.query:
        body = f"Transport[{plan.member} -> {plan.query}]({body})"
    return body


def _kernel_key(plan: _Plan) -> tuple[Any, ...]:
    """Structural kernel identity, retaining holes and deferred transports.

    Only adjacent marginalizations are normalized: summation order does not
    affect the kernel. Do not use _Plan.expression: it replaces unresolved
    interventions with observational placeholders and provisional weights.
    """
    if plan.rule is None:
        return ("unresolved", plan.query)
    assert plan.member is not None
    if plan.rule == "base":
        key: tuple[Any, ...] = (
            "base",
            plan.member.outcomes,
            plan.member.conditions,
        )
    elif plan.rule == "chain rule":
        key = ("product", tuple(_kernel_key(p) for p in plan.premises))
    else:
        child = plan.premises[0]
        body = _kernel_key(child)
        if plan.rule == "marginalization":
            variables = child.query.outcomes - plan.member.outcomes
            if body[0] == "marginalization":
                variables = variables | body[1]
                body = body[2]
            key = ("marginalization", variables, body)
        else:
            key = (
                "conditioning",
                plan.member.conditions - child.query.conditions,
                body,
            )
    if plan.member != plan.query:
        key = ("transport", plan.query, plan.member, key)
    return key


def _state_key(item: _Plan | Derivation) -> tuple[Any, ...]:
    if isinstance(item, Derivation):
        return ("formula", item.query, item.formula)
    # Preserve the order of unresolved work, not just the displayed algebra.
    return ("plan", item.query, item.pending, _kernel_key(item))


def _chosen_leaf(plan: _Plan) -> _Plan:
    """Locate the first unresolved premise that expand() will replace."""
    while plan.rule is not None:
        plan = next(p for p in plan.premises if p.pending)
    return plan


def _replacement(before: _Plan, after: _Plan) -> _Plan:
    while before.rule is not None:
        index = next(i for i, p in enumerate(before.premises) if p.pending)
        before, after = before.premises[index], after.premises[index]
    return after


class SearchSession(ABC):
    """Common callback interface; each step applies one move or beam layer."""

    prompt_template: PromptTemplate

    def __init__(
        self,
        task: Task,
        *,
        costs: Mapping[str, int] | None = None,
        prune_duplicates: bool = True,
    ) -> None:
        graph = ADMG.parse(task.graph)
        parse_query(graph, task.treatment, task.outcome)
        self.task = task
        self.archive = Archive(task)
        self.history: list[dict[str, Any]] = []
        self.decision_stats = DecisionStats()
        self._search = _Search(graph, costs=costs)
        self.prune_duplicates = prune_duplicates
        self.pruned: list[dict[str, Any]] = []
        self._best_cost: dict[tuple[Any, ...], int] = {}
        self._serial = 0

    @property
    @abstractmethod
    def exhausted(self) -> bool:
        """Whether the strategy has no remaining candidates."""

    @abstractmethod
    def prompt(self) -> str:
        """Describe the next deterministic choice set."""

    @abstractmethod
    def step(self, response: str) -> dict[str, Any]:
        """Validate a model selection, then apply it."""

    @abstractmethod
    def validate(self, response: str) -> None:
        """Reject an invalid selection without changing any search state."""

    def forced_response(self) -> str | None:
        """Return the sole valid selection, or None when a choice is needed."""
        return None

    def _moves(self, plan: _Plan, cost: int) -> list[_Move]:
        moves = []
        if plan.pending:
            query = _chosen_leaf(plan).query
            for added, expanded in self._search.expand(plan, None):
                leaf = _replacement(plan, expanded)
                description = {
                    "query": str(query),
                    "rule": leaf.rule,
                    "equivalent_query": str(leaf.member),
                    "do_calculus_path": [str(q) for q in leaf.path],
                    "premises": [str(p.query) for p in leaf.premises],
                    "pending": [str(q) for q in expanded.pending],
                }
                moves.append(self._move(expanded, cost + added, description))
        else:
            for proof in plan.derivations():
                moves.append(
                    self._move(
                        proof,
                        cost,
                        {
                            "rule": "submit",
                            "formula": proof.formula,
                        },
                    )
                )
        return moves

    def _move(
        self, item: _Plan | Derivation, cost: int, description: dict[str, Any]
    ) -> _Move:
        move = _Move(self._serial, cost, item, description)
        self._serial += 1
        return move

    def _equation(self, item: _Plan | Derivation) -> str:
        rhs = _partial(item) if isinstance(item, _Plan) else item.formula
        return f"{self.task.query} = {rhs}"

    def run(
        self,
        complete: Callable[[str], str],
        *,
        max_steps: int = 20,
        max_decision_retries: int = 2,
        on_event: EventSink | None = None,
    ) -> Archive:
        """Apply at most max_steps moves/layers, including forced selections.

        Invalid selections and truncated responses receive bounded corrective
        retries without advancing the search. Exhausted retries propagate;
        no heuristic fallback chooses branches. HTTP retries are independent.
        """
        _integer(max_steps, "max_steps", 1)
        _integer(max_decision_retries, "max_decision_retries", 0)
        for _ in range(max_steps):
            if self.exhausted:
                break
            advance(
                self,
                complete,
                max_decision_retries=max_decision_retries,
                on_event=on_event,
            )
        return self.archive
