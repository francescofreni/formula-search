"""Config type; named presets and the selected CONFIG live in configs.py."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    graph: str
    formula: str
    target: str
    n: int = 30
    seed: int = 0
    target_bound: float = 1e-14

    def __post_init__(self) -> None:
        if type(self.n) is not int or self.n < 1:
            raise ValueError("n must be a positive integer")
