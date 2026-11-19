"""Repeated profiling wraps a distance config without duplicating its fields."""

from dataclasses import dataclass
from math import isfinite

from distance_metric import DistanceConfig


@dataclass(frozen=True)
class Config:
    distance: DistanceConfig
    n: int = 5
    expected_distance: float | None = None

    def __post_init__(self) -> None:
        if type(self.n) is not int or self.n < 1:
            raise ValueError("n must be a positive integer")
        if self.expected_distance is not None and (
            not isfinite(self.expected_distance)
            or not 0 <= self.expected_distance <= 1
        ):
            raise ValueError("expected_distance must lie between 0 and 1")
