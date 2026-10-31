"""Layer-wise beam search: the model selects survivors, not legal moves."""

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
    _integer,
    _Move,
    _state_key,
)


@dataclass(frozen=True)
class _Branch:
    plan: _Plan
    cost: int
    path: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _Candidate:
    move: _Move
    parent: _Branch


class BeamSession(SearchSession):
    """Expand every retained branch once, then ask the model to keep k.

    k = min(beam_width, candidate count). Complete plans offer their final
    proof/weight variants in a separate submit layer, as in depth-first
    search. Selected proofs enter the archive and leave the beam; selected
    partial plans become the next frontier. Rejected candidates are not
    retained for backtracking. No heuristic or cost bound ranks candidates.

    Width limits survivors, not generated options, memory, or prompt size.
    A step is one selection call for a whole layer, not one proof expansion.
    """

    def __init__(
        self,
        task: Task,
        *,
        beam_width: int = 3,
        costs: Mapping[str, int] | None = None,
        prune_duplicates: bool = True,
    ) -> None:
        self.beam_width = _integer(beam_width, "beam_width", 1)
        super().__init__(task, costs=costs, prune_duplicates=prune_duplicates)
        self.layer = 0
        self._frontier = [_Branch(_Plan(task.query), 0)]
        if prune_duplicates:
            self._best_cost[_state_key(self._frontier[0].plan)] = 0
        self._candidates: list[_Candidate] = []
        self._prepare()

    @property
    def exhausted(self) -> bool:
        """No candidates remain in this beam; discarded branches may exist."""
        return not self._candidates

    def _prepare(self) -> None:
        candidates = [
            _Candidate(move, branch)
            for branch in self._frontier
            for move in self._moves(branch.plan, branch.cost)
        ]
        if not self.prune_duplicates:
            self._candidates = candidates
            return
        # Choose the cheapest representative of exact structural duplicates.
        # Only selected states become visited; beam rejection is not visiting.
        representatives: dict[tuple[Any, ...], _Candidate] = {}
        for candidate in candidates:
            move = candidate.move
            key = _state_key(move.item)
            old = representatives.get(key)
            if old is None or move.cost < old.move.cost:
                representatives[key] = candidate
        kept = []
        for candidate in candidates:
            move = candidate.move
            key = _state_key(move.item)
            previous = self._best_cost.get(key)
            representative = representatives[key]
            if previous is not None and previous <= move.cost:
                reason = "duplicate_equation"
            elif representative is not candidate:
                previous = representative.move.cost
                reason = "duplicate_in_layer"
            else:
                kept.append(candidate)
                continue
            self.pruned.append(
                {
                    "move": move.id,
                    "cost": move.cost,
                    "previous_cost": previous,
                    "reason": reason,
                    "equation": self._equation(move.item),
                }
            )
        self._candidates = kept

    def _describe(self, candidate: _Candidate) -> dict[str, Any]:
        move, parent = candidate.move, candidate.parent
        return {
            "id": move.id,
            "cost": move.cost,
            "added_cost": move.cost - parent.cost,
            **move.description,
            "from_equation": self._equation(parent.plan),
            "resulting_equation": self._equation(move.item),
            "branch_path": list(parent.path),
        }

    def prompt(self) -> str:
        return json.dumps(
            {
                "strategy": "beam",
                "instructions": DISCOVERY_OBJECTIVE
                + (
                    'Return only {"keep": [INTEGER_ID, ...]}. Select exactly '
                    "selection_count distinct IDs from options, across all "
                    "parents together. Unselected candidates are "
                    "discarded. All offered moves are legal. Costs are "
                    "recorded, not a budget or ranking. UNRESOLVED is a kernel "
                    "still to derive; Marginalize sums out variables; "
                    "Condition conditions a kernel. Transport converts "
                    "equivalent queries with final weights deferred until "
                    "submit. Selected submit options are verified and archived "
                    "and do not occupy a slot in the next frontier."
                ),
                "graph": self.task.graph,
                "target": str(self.task.query),
                "layer": self.layer + 1,
                "beam_width": self.beam_width,
                "selection_count": min(self.beam_width, len(self._candidates)),
                "known_formulas": [c.formula for c in self.archive.classes],
                "exhausted": self.exhausted,
                "options": [self._describe(c) for c in self._candidates],
            },
            sort_keys=True,
        )

    def _selection(self, response: str) -> list[_Candidate]:
        action = json.loads(response)
        if not isinstance(action, dict) or set(action) != {"keep"}:
            raise ValueError('Expected {"keep": [INTEGER_ID, ...]}')
        ids = action["keep"]
        count = min(self.beam_width, len(self._candidates))
        if not isinstance(ids, list) or len(ids) != count:
            raise ValueError(f"keep must contain exactly {count} IDs")
        for move_id in ids:
            _integer(move_id, "candidate ID", 0)
        if len(set(ids)) != len(ids):
            raise ValueError("keep must contain distinct IDs")
        options = {c.move.id: c for c in self._candidates}
        if any(move_id not in options for move_id in ids):
            raise ValueError("Candidate is not offered in the current layer")
        # Selection is a set; provider list order must not reorder enumeration.
        return [c for c in self._candidates if c.move.id in ids]

    def step(self, response: str) -> dict[str, Any]:
        """Validate the entire selection before changing the beam or archive."""
        if self.exhausted:
            raise ValueError("Search is exhausted")
        selected = self._selection(response)
        records = []
        frontier = []
        for candidate in selected:
            move, parent = candidate.move, candidate.parent
            record = self._describe(candidate)
            record.pop("branch_path")
            if isinstance(move.item, Derivation):
                verdict = self.archive.submit(move.item.formula, move.item)
                record.update(
                    outcome=verdict.outcome, class_index=verdict.class_index
                )
            else:
                frontier.append(
                    _Branch(
                        move.item, move.cost, parent.path + (dict(record),)
                    )
                )
            if self.prune_duplicates:
                self._best_cost[_state_key(move.item)] = move.cost
            records.append(record)
        ids = {c.move.id for c in selected}
        discarded = [
            self._describe(c) for c in self._candidates if c.move.id not in ids
        ]
        self.layer += 1
        self._frontier = frontier
        self._prepare()
        result = {
            "layer": self.layer,
            "beam_width": self.beam_width,
            "selected": records,
            "discarded": discarded,
            "frontier_size": len(frontier),
            "total_classes": len(self.archive.classes),
            "exhausted": self.exhausted,
        }
        self.history.append(result)
        return result
