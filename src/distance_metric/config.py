"""Inputs for a mean chain-Jaccard distance, with a Hoeffding guarantee."""

from dataclasses import dataclass
from math import ceil, isfinite, log

from formula_search.verification import VerificationConfig


@dataclass(frozen=True)
class DistanceConfig(VerificationConfig):
    other_formula: str
    optional_edges: tuple[str, ...] | None = None
    epsilon: float = 0.05
    delta: float = 0.01
    include_bidirected: bool = True
    topological_order: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.other_formula, str)
            or not self.other_formula.strip()
        ):
            raise ValueError("other_formula must be a nonempty string")
        if self.optional_edges is not None and (
            not isinstance(self.optional_edges, tuple)
            or not all(
                isinstance(edge, str) and edge.strip()
                for edge in self.optional_edges
            )
        ):
            raise ValueError(
                "optional_edges must be None or a tuple of edge strings"
            )
        if not isinstance(self.include_bidirected, bool):
            raise ValueError("include_bidirected must be a boolean")
        if self.topological_order is not None and (
            not isinstance(self.topological_order, tuple)
            or not all(
                isinstance(node, str) for node in self.topological_order
            )
        ):
            raise ValueError(
                "topological_order must be None or a tuple of names"
            )
        for name in ("epsilon", "delta"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between 0 and 1")

    @property
    def n_chains(self) -> int:
        """P(|estimate - true mean chain distance| >= epsilon) <= delta."""
        return ceil(log(2 / self.delta) / (2 * self.epsilon**2))
