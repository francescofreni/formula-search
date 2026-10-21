"""A JSON decision loop over fresh A* searches and one shared archive.

The caller supplies a model callback, so no SDK, credentials, network calls,
or provider assumptions live here. Model output selects known options only;
it is never executed as code. Search limits bound formula cost and output
count, not wall time or expansions. Use small tasks and bounds.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..astar import derive
from ..evaluator import Archive, Task
from ..heuristics import expression_size, free_proof_bound, term_count

_INSTRUCTIONS = """Discover distinct causal identifying formulas for the task.
Choose one action as a JSON object, with no markdown:
{"action": "search", "bound": 1, "limit": 2,
 "heuristic": "zero", "tiebreaker": "none"}
or {"action": "stop"}.
Search requires integer bound and limit within the supplied limits.
Optional heuristic: zero (default) or free.
Optional tiebreaker: none (default), expression_size, or term_count.
Each action starts a fresh A* search; the archive survives across actions.
Limit counts classes returned by that search, including already known ones.
Use feedback to change your next search; identical actions repeat work.
Only replayable engine proofs enter the archive. Class equality is checked
by HiProF on a complete DAG. An empty search does not prove that the target
is unidentifiable. Stopping does not assert that all formulas were found.
"""


def _integer(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class Session:
    """One task, a verified archive, and JSON-serializable action feedback.

    ``run`` accepts a callable from prompt text to response text, typically
    a thin wrapper around an LLM. Alternatively use ``prompt`` and ``step``
    manually. Invalid responses raise ValueError without launching a search;
    model/engine errors propagate. History and proofs are kept in memory.
    """

    def __init__(
        self, task: Task, *, max_bound: int = 3, max_limit: int = 10
    ) -> None:
        self.max_bound = _integer(max_bound, "max_bound", 0)
        self.max_limit = _integer(max_limit, "max_limit", 1)
        self.task = task
        self.archive = Archive(task)
        self.history: list[dict[str, Any]] = []
        self.stopped = False

    def prompt(self) -> str:
        """Describe allowed actions, the task, and all previous feedback."""
        return (
            _INSTRUCTIONS
            + "\n"
            + json.dumps(
                {
                    "task": self.task._asdict(),
                    "max_bound": self.max_bound,
                    "max_limit": self.max_limit,
                    "formulas": [c.formula for c in self.archive.classes],
                    "history": self.history,
                },
                sort_keys=True,
            )
        )

    def step(self, response: str) -> dict[str, Any]:
        """Validate one model response, execute it, and record feedback.

        A search hitting its output limit is conservatively marked
        ``class_limit``. ``exhausted`` refers only to that engine's bounded
        search space, never all possible identifying equations.
        """
        if self.stopped:
            raise ValueError("Session has stopped")
        action = json.loads(response)
        if not isinstance(action, dict):
            raise ValueError("Expected a JSON object")
        if action == {"action": "stop"}:
            self.stopped = True
            record: dict[str, Any] = {"action": action, "status": "stopped"}
        else:
            record = self._search(action)
        self.history.append(record)
        return record

    def _search(self, action: dict[str, Any]) -> dict[str, Any]:
        allowed = {"action", "bound", "limit", "heuristic", "tiebreaker"}
        if action.get("action") != "search" or action.keys() - allowed:
            raise ValueError("Expected search or stop with supported fields")
        bound = _integer(action.get("bound"), "bound", 0)
        limit = _integer(action.get("limit"), "limit", 1)
        if bound > self.max_bound or limit > self.max_limit:
            raise ValueError("Action exceeds session limits")
        heuristic = action.get("heuristic", "zero")
        tie = action.get("tiebreaker", "none")
        if heuristic not in ("zero", "free"):
            raise ValueError("Unknown heuristic")
        if tie not in ("none", "expression_size", "term_count"):
            raise ValueError("Unknown tiebreaker")
        scorers = {
            "none": None,
            "expression_size": expression_size(),
            "term_count": term_count(),
        }
        derivations = derive(
            *self.task,
            bound=bound,
            limit=limit,
            heuristic=(
                free_proof_bound(self.task.graph)
                if heuristic == "free"
                else None
            ),
            tiebreaker=scorers[tie],
        )
        verdicts = [self.archive.submit(d.formula, d) for d in derivations]
        return {
            "action": action,
            "status": (
                "class_limit" if len(derivations) == limit else "exhausted"
            ),
            "new_classes": sum(v.outcome == "new" for v in verdicts),
            "total_classes": len(self.archive.classes),
            "results": [
                {
                    "formula": v.formula,
                    "outcome": v.outcome,
                    "class_index": v.class_index,
                }
                for v in verdicts
            ],
        }

    def run(
        self, complete: Callable[[str], str], *, max_rounds: int = 5
    ) -> Archive:
        """Ask the model for at most max_rounds actions and return the archive.

        Reaching max_rounds leaves the session open for further calls. It
        does not mark search exhausted. Proofs remain in ``archive.log``.
        """
        _integer(max_rounds, "max_rounds", 1)
        for _ in range(max_rounds):
            if self.stopped:
                break
            self.step(complete(self.prompt()))
        return self.archive
