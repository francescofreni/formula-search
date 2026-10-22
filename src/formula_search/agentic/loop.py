"""Model-chosen proof-tree traversal with deterministic legal moves."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..astar import _Plan, _Search
from ..baselines.id import parse_query
from ..derivation import Derivation
from ..evaluator import Archive, Task
from ..expression import Term, render
from ..graph import ADMG


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


@dataclass
class _Frame:
    plan: _Plan
    cost: int
    path: tuple[dict[str, Any], ...]
    moves: list[_Move]


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


class Session:
    """Depth-first search whose branch order is entirely model-selected.

    Legal moves reuse A*'s deterministic expansion, never its priority queue.
    Resolve the first unfinished premise, keeping other premises in the proof.
    Unchosen siblings survive on a stack. Completed proofs and dead ends cause
    automatic backtracking. Final transport-weight variants are also choices.
    Costs are recorded only; no heuristic ranks or selects moves.

    This replaces the earlier configuration-level Session API. All state is
    in memory; run() can be called again to continue after a decision budget.
    """

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
        self._search = _Search(graph, costs=costs)
        self.prune_duplicates = prune_duplicates
        self.pruned: list[dict[str, Any]] = []
        self._best_cost: dict[tuple[Any, ...], int] = {}
        self._serial = 0
        self._stack: list[_Frame] = []
        self._descend(_Plan(task.query), 0)

    @property
    def exhausted(self) -> bool:
        """Whether no unexplored branches remain. Unbounded search may never exhaust."""
        return not self._stack

    def _descend(
        self,
        plan: _Plan,
        cost: int,
        path: tuple[dict[str, Any], ...] = (),
    ) -> None:
        if self.prune_duplicates:
            key = _state_key(plan)
            previous = self._best_cost.get(key)
            if previous is not None and previous <= cost:
                self._backtrack()
                return
            self._best_cost[key] = cost
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
        if moves:
            self._stack.append(_Frame(plan, cost, path, moves))
        self._backtrack()

    def _move(
        self, item: _Plan | Derivation, cost: int, description: dict[str, Any]
    ) -> _Move:
        move = _Move(self._serial, cost, item, description)
        self._serial += 1
        return move

    def _backtrack(self) -> None:
        while self._stack:
            frame = self._stack[-1]
            if self.prune_duplicates:
                kept = []
                for move in frame.moves:
                    previous = self._best_cost.get(_state_key(move.item))
                    if previous is not None and previous <= move.cost:
                        self.pruned.append(
                            {
                                "move": move.id,
                                "cost": move.cost,
                                "previous_cost": previous,
                                "reason": "duplicate_equation",
                                "equation": self._equation(move.item),
                            }
                        )
                    else:
                        kept.append(move)
                frame.moves = kept
            if frame.moves:
                return
            self._stack.pop()

    def _equation(self, item: _Plan | Derivation) -> str:
        rhs = _partial(item) if isinstance(item, _Plan) else item.formula
        return f"{self.task.query} = {rhs}"

    def prompt(self) -> str:
        """Show only the active branch and its available next equations."""
        frame = self._stack[-1] if self._stack else None
        current_cost = frame.cost if frame else 0
        return json.dumps(
            {
                "instructions": (
                    "Which offered move should we apply next? "
                    'Return only {"move": INTEGER_ID}. '
                    "There is no proof-cost bound; costs are recorded only. "
                    "Choose one local step; you do not need to solve the "
                    "whole proof. All offered moves are legal. "
                    "UNRESOLVED marks a kernel still to derive. "
                    "Marginalize[V] sums out V; Condition[V] conditions on V. "
                    "Transport converts equivalent queries; its averaging "
                    "weights remain deferred until a submit option. "
                    "Equations containing these operators are proof templates. "
                    "All unresolved terms must eventually be resolved."
                ),
                "graph": self.task.graph,
                "target": str(self.task.query),
                "current_equation": (
                    self._equation(frame.plan) if frame else None
                ),
                "branch_path": list(frame.path) if frame else [],
                "term_to_expand": (
                    str(_chosen_leaf(frame.plan).query)
                    if frame and frame.plan.pending
                    else None
                ),
                "current_cost": current_cost,
                "exhausted": self.exhausted,
                "known_formulas": [c.formula for c in self.archive.classes],
                "options": [
                    {
                        "id": m.id,
                        "cost": m.cost,
                        "added_cost": m.cost - current_cost,
                        **m.description,
                        "resulting_equation": self._equation(m.item),
                    }
                    for m in (frame.moves if frame else [])
                ],
            },
            sort_keys=True,
        )

    def step(self, response: str) -> dict[str, Any]:
        """Apply exactly one offered move. Invalid output leaves state intact."""
        if self.exhausted:
            raise ValueError("Search is exhausted")
        action = json.loads(response)
        if not isinstance(action, dict) or set(action) != {"move"}:
            raise ValueError('Expected {"move": INTEGER_ID}')
        move_id = _integer(action["move"], "move", 0)
        frame = self._stack[-1]
        options = frame.moves
        move = next((m for m in options if m.id == move_id), None)
        if move is None:
            raise ValueError("Move is not offered at the current node")
        record: dict[str, Any] = {
            "move": move.id,
            "cost": move.cost,
            **move.description,
            "from_equation": self._equation(frame.plan),
            "resulting_equation": self._equation(move.item),
        }
        if isinstance(move.item, Derivation):
            verdict = self.archive.submit(move.item.formula, move.item)
            record.update(
                outcome=verdict.outcome, class_index=verdict.class_index
            )
            if self.prune_duplicates:
                self._best_cost[_state_key(move.item)] = move.cost
        options.remove(move)
        if isinstance(move.item, _Plan):
            self._descend(move.item, move.cost, frame.path + (dict(record),))
        else:
            self._backtrack()
        record.update(
            total_classes=len(self.archive.classes), exhausted=self.exhausted
        )
        self.history.append(record)
        return record

    def run(
        self, complete: Callable[[str], str], *, max_steps: int = 20
    ) -> Archive:
        """Make at most max_steps model decisions; preserve the frontier.

        Errors propagate without a fallback choice. This is a decision cap,
        not a wall-time or memory cap. Even a singleton option is selected
        through the callback. No minimum-cost discovery order is promised.
        """
        _integer(max_steps, "max_steps", 1)
        for _ in range(max_steps):
            if self.exhausted:
                break
            self.step(complete(self.prompt()))
        return self.archive
