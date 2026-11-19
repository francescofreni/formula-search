"""Run the selected distance experiment and print its estimate and timing."""

from .config import DistanceConfig
from .configs import CONFIG
from .estimator import estimate_distance
from .reporting import print_sampling_config, print_sampling_result


def main(config: DistanceConfig) -> None:
    print(config)
    print_sampling_config(config)
    print(f"Sampling {config.n_chains} chains...", flush=True)
    result = estimate_distance(config)
    print_sampling_result(config, result)
    print(f"Estimated distance: {result.distance:.6f}")
    print(f"Interval: [{result.interval[0]:.6f}, {result.interval[1]:.6f}]")
    print(f"Sampling failure probability <= {config.delta:g}")
    print(f"Verifier failure bound <= {result.verifier_failure_bound:.3e}")
    print(f"Combined failure bound <= {result.total_failure_bound:.12g}")
    print(
        f"HiProF calls: {result.verifier_calls} "
        f"(worst-case budget: {result.verifier_call_budget})"
    )
    print(f"Elapsed: {result.elapsed_seconds:.3f} s")
    print("Distance and guarantees apply within linear-Gaussian models.")


if __name__ == "__main__":
    main(CONFIG)
