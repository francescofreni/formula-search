"""Time n fresh checks, including verifier construction, excluding printing."""

import random
from statistics import mean, median
from time import perf_counter

from hiprof import HPFalsifier

from .config import Config
from .configs import CONFIG


def main(config: Config) -> None:
    random.seed(config.seed)
    times = []
    successes = 0
    print(config)
    for i in range(config.n):
        start = perf_counter()
        result = HPFalsifier(config.graph).check(
            config.target, config.formula, target_bound=config.target_bound
        )
        elapsed = perf_counter() - start
        times.append(elapsed)
        successes += bool(result)
        print(f"{i + 1}/{config.n}: {elapsed * 1000:.3f} ms; {result}")
    print(f"Accepted: {successes}/{len(times)}")
    print(f"Mean: {mean(times) * 1000:.3f} ms")
    print(f"Median: {median(times) * 1000:.3f} ms")


if __name__ == "__main__":
    main(CONFIG)
