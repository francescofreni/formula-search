"""Repeat a distance estimate with successive seeds and summarize performance."""

from dataclasses import replace
from statistics import mean, median

from distance_metric import DistanceResult, estimate_distance
from distance_metric.reporting import (
    print_sampling_config,
    print_sampling_result,
)

from .config import Config
from .configs import CONFIG


def main(config: Config) -> list[DistanceResult]:
    print(config)
    print_sampling_config(config.distance)
    results = []
    for i in range(config.n):
        run = replace(config.distance, seed=config.distance.seed + i)
        result = estimate_distance(run)
        results.append(result)
        print_sampling_result(run, result)
        print(
            f"{i + 1}/{config.n}: seed={run.seed}; "
            f"distance={result.distance:.6f}; "
            f"{result.elapsed_seconds:.3f} s; "
            f"{result.verifier_calls} HiProF calls",
            flush=True,
        )
    times = [r.elapsed_seconds for r in results]
    distances = [r.distance for r in results]
    print(f"Completed: {len(results)}/{config.n}")
    print(f"Mean time: {mean(times):.3f} s; median: {median(times):.3f} s")
    print(
        f"Mean distance: {mean(distances):.6f}; "
        f"range: [{min(distances):.6f}, {max(distances):.6f}]"
    )
    print(f"Mean HiProF calls: {mean(r.verifier_calls for r in results):.1f}")
    if config.expected_distance is not None:
        hits = sum(
            abs(d - config.expected_distance) < config.distance.epsilon
            for d in distances
        )
        print(f"Expected distance: {config.expected_distance:.6f}")
        print(f"Within epsilon: {hits}/{config.n}")
    return results


if __name__ == "__main__":
    main(CONFIG)
