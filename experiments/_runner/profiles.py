"""Controlled queue-order ablations; all scores are lower-is-better."""

from __future__ import annotations

from dataclasses import dataclass

from formula_search.evaluator import Task
from formula_search.heuristics import (
    Heuristic,
    SearchState,
    Tiebreaker,
    _HistoryTiebreaker,
    expression_size,
    less_explored,
    lookahead_bound,
    new_terms,
)
from formula_search.suite import CURATED

TASKS = {
    "adjustment": Task("A -> B; B -> X; A -> Y; X -> Y", "X", "Y"),
    "mediation": Task("Z -> X; Z -> Y; X -> M; M -> Y", "X", "Y"),
    "front-door": CURATED["front-door"],
    "napkin": CURATED["napkin"],
}
COSTS = {"marginalization": 1, "conditioning": 1, "chain rule": 1}


@dataclass(frozen=True)
class Profile:
    name: str
    primary: str
    secondary: str
    rationale: str
    lookahead_depth: int | None = 2
    minimum_cost_order: bool = True


PROFILES = (
    Profile("uniform", "g", "0", "Uniform-cost control.", None),
    Profile("lookahead", "f", "0", "Measure lookahead without novelty."),
    Profile(
        "concise_ties", "f", "size", "Test concise equations, not novelty."
    ),
    Profile(
        "new_term_ties", "f", "-N", "Novelty among equal estimated costs."
    ),
    Profile(
        "exploration_ties", "f", "-E", "Less-used next queries on cost ties."
    ),
    Profile(
        "diverse_ties",
        "f",
        "-D",
        "Combine novelty and exploration on equal scales.",
    ),
    Profile(
        "new_terms_first",
        "-N",
        "f",
        "Can novelty justify more expensive routes?",
        minimum_cost_order=False,
    ),
    Profile(
        "exploration_first",
        "-E",
        "f",
        "Can broader query coverage improve discovery?",
        minimum_cost_order=False,
    ),
    Profile(
        "diversity_first",
        "-D",
        "f",
        "Remove cost's dominance; cost only breaks ties.",
        minimum_cost_order=False,
    ),
    Profile(
        "mixed_1",
        "f-D",
        "0",
        "At most one rule-cost unit of diversity reward.",
        minimum_cost_order=False,
    ),
    Profile(
        "mixed_4",
        "f-4D",
        "0",
        "Allow up to four cost units for diversity.",
        minimum_cost_order=False,
    ),
    Profile(
        "finish_first",
        "h",
        "-D",
        "Ignore sunk cost and favor near-complete proofs.",
        minimum_cost_order=False,
    ),
)


def configured(
    graph: str,
    profile: Profile,
) -> tuple[Heuristic | None, Tiebreaker | None, Tiebreaker]:
    heuristic = (
        lookahead_bound(graph, depth=profile.lookahead_depth, costs=COSTS)
        if profile.lookahead_depth is not None
        else None
    )
    novelty, visits, size = new_terms(), less_explored(), expression_size()

    def values(state: SearchState) -> dict[str, float]:
        g, h = float(state.cost), state.estimated_remaining_cost
        n = -novelty(
            state
        )  # Fraction of distinct terms absent from known classes.
        e = 1.0 / (
            1.0 + visits(state)
        )  # In [0,1], avoiding raw-count dominance.
        d = (n + e) / 2.0
        return {
            "0": 0.0,
            "g": g,
            "f": g + h,
            "h": h,
            "size": size(state),
            "-N": -n,
            "-E": -e,
            "-D": -d,
            "f-D": g + h - d,
            "f-4D": g + h - 4 * d,
        }

    priority = (
        None
        if profile.primary in ("f", "g")
        else lambda state: values(state)[profile.primary]
    )

    def secondary(state: SearchState) -> float:
        return values(state)[profile.secondary]

    tie: Tiebreaker = secondary
    if profile.secondary in ("-N", "-E", "-D"):
        tie = _HistoryTiebreaker(tie)
    # Even controls get an explicit zero scorer: all profiles queue final
    # transport-weight proof candidates individually and count their pops.
    return heuristic, priority, tie
