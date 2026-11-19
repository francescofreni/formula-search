"""Select a distance example here; all edges are additions to the base graph."""

from dataclasses import replace

from .config import DistanceConfig

# Either optional edge breaks one adjustment formula, but not the other.
# Both possible chains have distance 1/2, including the original graph.
ADJUSTMENT_01_A_VS_B = DistanceConfig(
    graph="A -> B; B -> X; A -> Y; X -> Y",
    target="p(Y | do(X))",
    formulas=(
        "sum_{A} { p(A) p(Y | A, X) }",
        "sum_{B} { p(B) p(Y | B, X) }",
    ),
    optional_edges=("A -> X", "B -> Y"),
    epsilon=0.05,
    delta=0.01,
    seed=0,
)

# X <-> Y invalidates both formulas. If it is first, distance is 0;
# in the other four orders distance is 1/2. Mean over six orders: 1/3.
ADJUSTMENT_02_SHARED_CONFOUNDING = replace(
    ADJUSTMENT_01_A_VS_B,
    optional_edges=("A -> X", "B -> Y", "X <-> Y"),
)

# Normal experiment: derive all optional edges from sampled variable orders.
# The two preceding presets are fixed-family, known-distance controls.
ADJUSTMENT_03_AUTOMATIC = replace(ADJUSTMENT_01_A_VS_B, optional_edges=None)

# Three valid adjustment formulas on the same base graph.
ADJUSTMENT_04_THREE_FORMULAS = replace(
    ADJUSTMENT_03_AUTOMATIC,
    formulas=(
        *ADJUSTMENT_03_AUTOMATIC.formulas,
        "sum_{A, B} { p(A, B) p(Y | A, B, X) }",
    ),
)

CONFIG = ADJUSTMENT_03_AUTOMATIC
