"""Make the sampling distribution and rejection work visible in run logs."""

from dataclasses import replace

from .config import DistanceConfig
from .edges import resolve_edges
from .estimator import DistanceResult


def print_sampling_config(config: DistanceConfig) -> None:
    print(
        f"Score: mean distance over {config.n_pairs} pairs "
        f"from {len(config.formulas)} formula entries."
    )
    if config.optional_edges is None and config.topological_order is None:
        print(
            "Sampling: uniform random permutations of all variables; reject "
            "until every existing arrow points forward. Repeat per chain."
        )
        print(
            "Then generate missing forward edges and shuffle additions; "
            f"include new bidirected edges: {config.include_bidirected}."
        )
    else:
        print("Sampling: fixed edge family; shuffle additions per chain.")
        print("Resolved edge family:", resolve_edges(config))


def print_sampling_result(
    config: DistanceConfig, result: DistanceResult
) -> None:
    if result.sampling_mode != "uniform_topological_orders":
        return
    accepted = sum(result.topological_order_counts.values())
    attempts = result.topological_permutation_attempts
    print(
        f"Topological permutations: {attempts} attempted, {accepted} "
        f"accepted, {attempts - accepted} rejected; "
        f"{len(result.topological_order_counts)} distinct accepted orders."
    )
    for order, count in sorted(result.topological_order_counts.items()):
        family = resolve_edges(replace(config, topological_order=order))
        print(
            f"  {count} chains: order={order}; edges={family.optional_edges}"
        )
