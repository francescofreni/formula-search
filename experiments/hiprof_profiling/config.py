"""Profiling settings extend the shared graph/formula verification config."""

from dataclasses import dataclass

from formula_search.verification import VerificationConfig


@dataclass(frozen=True)
class Config(VerificationConfig):
    n: int = 30

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.n) is not int or self.n < 1:
            raise ValueError("n must be a positive integer")
