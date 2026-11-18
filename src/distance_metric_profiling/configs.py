"""Named benchmarks grouped by graph; select CONFIG at the bottom.

GRAPH_NN_DESCRIPTION orders scenarios within a group, not measured runtime.
All use epsilon=0.05, delta=0.01, and five estimates unless overridden.
"""

from dataclasses import replace

from distance_metric import DistanceConfig
from distance_metric import configs as examples
from formula_search.verification import VerificationConfig
from hiprof_profiling import configs as formulas

from .config import Config


def _pair(
    first: VerificationConfig,
    second: VerificationConfig,
) -> Config:
    if (first.graph, first.target) != (second.graph, second.target):
        raise ValueError("Formula pairs must share the same graph and target")
    return Config(
        distance=DistanceConfig(
            graph=first.graph,
            target=first.target,
            formula=first.formula,
            other_formula=second.formula,
            seed=first.seed,
            target_bound=first.target_bound,
        )
    )


# Adjustment: controls, then a larger optional-edge family.

ADJUSTMENT_01_IDENTICAL = Config(
    distance=replace(
        examples.ADJUSTMENT_03_AUTOMATIC,
        other_formula=examples.ADJUSTMENT_01_A_VS_B.formula,
    ),
    expected_distance=0.0,
)

ADJUSTMENT_02_HALF = Config(
    distance=examples.ADJUSTMENT_01_A_VS_B,
    expected_distance=0.5,
)

ADJUSTMENT_03_ONE_THIRD = Config(
    distance=examples.ADJUSTMENT_02_SHARED_CONFOUNDING,
    expected_distance=1 / 3,
)

ADJUSTMENT_04_EIGHT_EDGES = Config(
    distance=examples.ADJUSTMENT_03_AUTOMATIC,
)


# Mediation: automatically generated edges, increasing formula complexity.

MEDIATION_01_ADJUSTMENT_VS_FACTORIZATION = _pair(
    formulas.MEDIATION_01_ADJUST_Z,
    formulas.MEDIATION_02_MEDIATOR_FACTORIZATION,
)

MEDIATION_02_ADJUSTMENT_VS_NESTED_DIVISIONS = _pair(
    formulas.MEDIATION_01_ADJUST_Z,
    formulas.MEDIATION_05_NESTED_CONDITIONAL_DIVISIONS,
)


# Front-door: self-distance control on a confounded graph (expected zero).

FRONT_DOOR_01_IDENTICAL = replace(
    _pair(
        formulas.FRONT_DOOR_01_STANDARD,
        formulas.FRONT_DOOR_01_STANDARD,
    ),
    expected_distance=0.0,
)


# Napkin: compare the two archived formulas under seven optional edges.

NAPKIN_01_MARGINAL_VS_CONDITIONAL_WEIGHTS = _pair(
    formulas.NAPKIN_01_MARGINAL_WEIGHTS,
    formulas.NAPKIN_02_CONDITIONAL_WEIGHTS,
)

# n counts full distance estimates; each currently samples 1,060 chains.
CONFIG = ADJUSTMENT_04_EIGHT_EDGES
