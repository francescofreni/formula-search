"""Shared input settings for graph-specific HiProF equation checks."""

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class VerificationSettings:
    graph: str
    seed: int = field(default=0, kw_only=True)
    target_bound: float = field(default=1e-14, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, str) or not self.graph.strip():
            raise ValueError("graph must be a nonempty string")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not isfinite(self.target_bound) or not 0 < self.target_bound < 1:
            raise ValueError("target_bound must lie strictly between 0 and 1")


@dataclass(frozen=True)
class VerificationConfig(VerificationSettings):
    formula: str
    target: str

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("formula", "target"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
