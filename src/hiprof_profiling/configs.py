"""Actual search outputs, grouped by graph and expression complexity.

Names: GRAPH_NN_DESCRIPTION. Rank by probability-term count, then number
of sum/icd/quotient operators; ties are allowed. This does not rank runtime.
SOURCES.md records exact saved files and zero-based formula indices.
Presets inherit graph, target, n, seed, and tolerance from their group's 01.
"""

from dataclasses import replace

from .config import Config

# ADJUSTMENT: A -> B; B -> X; A -> Y; X -> Y.

# astar-adjustment: classes[0]; 2 probability terms, 1 operators.
ADJUSTMENT_01_ADJUST_A = Config(
    graph="A -> B; B -> X; A -> Y; X -> Y",
    target="p(Y | do(X))",
    formula="sum_{A} { p(A) p(Y | A, X) }",
)

# full-adjustment: formulas[1]; 3 probability terms, 2 operators.
ADJUSTMENT_02_NESTED_ADJUSTMENT = replace(
    ADJUSTMENT_01_ADJUST_A,
    formula="sum_{A} { p(A) sum_{B} { p(B) p(Y | A, B, X) } }",
)

# full-adjustment: formulas[5]; 3 probability terms, 3 operators.
ADJUSTMENT_03_CONDITIONAL_DIVISION = replace(
    ADJUSTMENT_01_ADJUST_A,
    formula="sum_{B} { p(B) icd_{X | B} { sum_{A} { p(A) p(X, Y | A, B) } } }",
)

# full-adjustment: formulas[8]; 4 probability terms, 2 operators.
ADJUSTMENT_04_TREATMENT_AVERAGING = replace(
    ADJUSTMENT_01_ADJUST_A,
    formula=(
        "sum_{A, B} { p(B) sum_{X'} { p(X') p(A | B, X') } p(Y | A, B, X) " "}"
    ),
)

# full-adjustment: formulas[65]; 6 probability terms, 4 operators.
ADJUSTMENT_05_NESTED_REWEIGHTING = replace(
    ADJUSTMENT_01_ADJUST_A,
    formula=(
        "sum_{A, B} { p(B) sum_{Y'} { p(Y' | B, X) sum_{B'} { p(B') "
        "sum_{X'} { p(X' | B', Y') p(A | B', X') } } } p(Y | A, X) }"
    ),
)


# MEDIATION: Z -> X; Z -> Y; X -> M; M -> Y.

# full-mediation: formulas[0]; 2 probability terms, 1 operators.
MEDIATION_01_ADJUST_Z = Config(
    graph="Z -> X; Z -> Y; X -> M; M -> Y",
    target="p(Y | do(X))",
    formula="sum_{Z} { p(Z) p(Y | X, Z) }",
)

# full-mediation: formulas[2]; 3 probability terms, 1 operators.
MEDIATION_02_MEDIATOR_FACTORIZATION = replace(
    MEDIATION_01_ADJUST_Z,
    formula="sum_{M, Z} { p(M | X) p(Z) p(Y | M, Z) }",
)

# beam-mediation: classes[5]; 3 probability terms, 3 operators.
MEDIATION_03_CONDITIONAL_DIVISION = replace(
    MEDIATION_01_ADJUST_Z,
    formula=(
        "sum_{M} { p(M | X) icd_{M | X} { sum_{Z} { p(Z) p(M, Y | X, Z) } "
        "} }"
    ),
)

# beam-mediation: classes[4]; 4 probability terms, 3 operators.
MEDIATION_04_SEPARATE_ADJUSTMENTS = replace(
    MEDIATION_01_ADJUST_Z,
    formula=(
        "sum_{M} { sum_{Z} { p(Z) p(M | X, Z) } sum_{Z} { p(Z) p(Y | M, Z) "
        "} }"
    ),
)

# beam-mediation: classes[23]; 7 probability terms, 6 operators.
MEDIATION_05_NESTED_CONDITIONAL_DIVISIONS = replace(
    MEDIATION_01_ADJUST_Z,
    formula=(
        "sum_{M, Z} { p(M | X) icd_{M | X} { sum_{Y'} { p(M | X) sum_{X'} "
        "{ p(X') p(Y' | M, X') } icd_{Y' | M} { sum_{X'} { p(X') p(Y', Z | "
        "M, X') } } } } p(Y | M, Z) }"
    ),
)


# FRONT_DOOR: T -> M; M -> Y; T <-> Y.

# full-front-door: formulas[0]; 3 probability terms, 2 operators.
FRONT_DOOR_01_STANDARD = Config(
    graph="T -> M; M -> Y; T <-> Y",
    target="p(Y | do(T))",
    formula="sum_{M} { p(M | T) sum_{T'} { p(T') p(Y | M, T') } }",
)


# NAPKIN: W -> Z; Z -> X; X -> Y; W <-> X; W <-> Y.

# full-napkin: formulas[0]; 3 probability terms, 3 operators.
NAPKIN_01_MARGINAL_WEIGHTS = Config(
    graph="W -> Z; Z -> X; X -> Y; W <-> X; W <-> Y",
    target="p(Y | do(X))",
    formula="sum_{Z} { p(Z) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } } }",
)

# full-napkin: formulas[1]; 3 probability terms, 3 operators.
NAPKIN_02_CONDITIONAL_WEIGHTS = replace(
    NAPKIN_01_MARGINAL_WEIGHTS,
    formula=(
        "sum_{Z} { p(Z | X) icd_{X | Z} { sum_{W} { p(W) p(X, Y | W, Z) } "
        "} }"
    ),
)

# Select a preset. Override repeats with replace(..., n=100).
CONFIG = ADJUSTMENT_01_ADJUST_A
