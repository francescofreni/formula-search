"""Rules 2 and 3 of do-calculus (Pearl, 1995), as graphical tests.

Rule 1 follows from Rules 2 and 3 (Huang and Valtorta, 2006, Lemma 4;
Yvernes et al., 2026, Theorem 2), so these two suffice.
"""

from __future__ import annotations

from collections.abc import Set

from .graph import ADMG


def rule2(
    graph: ADMG,
    y: Set[str],
    x: Set[str],
    z: Set[str],
    w: Set[str],
) -> bool:
    """Return whether ``p(y | do(x, z), w) = p(y | do(x), z, w)`` by Rule 2."""
    mutilated = graph.mutilated(into=x, out_of=z)
    return mutilated.d_separated(y, z, x | w)


def rule3(
    graph: ADMG,
    y: Set[str],
    x: Set[str],
    z: Set[str],
    w: Set[str],
) -> bool:
    """Return whether ``p(y | do(x, z), w) = p(y | do(x), w)`` by Rule 3."""
    mutilated = graph.mutilated(into=x)
    mutilated = mutilated.mutilated(into=z - mutilated.ancestors(w))
    return mutilated.d_separated(y, z, x | w)
