"""LLM-guided depth-first traversal with retained alternatives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..astar import _Plan
from ..derivation import Derivation
from ..evaluator import Task
from .common import (
    DISCOVERY_OBJECTIVE,
    SearchSession,
    _chosen_leaf,
    _integer,
    _Move,
    _state_key,
)


@dataclass
class _Frame:
    plan: _Plan
    cost: int
    path: tuple[dict[str, Any], ...]
    moves: list[_Move]


class DepthFirstSession(SearchSession):
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
        super().__init__(task, costs=costs, prune_duplicates=prune_duplicates)
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
        moves = self._moves(plan, cost)
        if moves:
            self._stack.append(_Frame(plan, cost, path, moves))
        self._backtrack()

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

    def prompt(self) -> str:
        """Show only the active branch and its available next equations."""
        frame = self._stack[-1] if self._stack else None
        current_cost = frame.cost if frame else 0
        return json.dumps(
            {
                "strategy": "depth-first",
                "instructions": DISCOVERY_OBJECTIVE
                + (
                    "Which offered move should we apply next? "
                    'Return only {"move": INTEGER_ID}. '
                    "There is no proof-cost bound; costs are recorded only. "
                    "Choose one local step. All offered moves are legal. "
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
