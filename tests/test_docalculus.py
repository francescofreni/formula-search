from __future__ import annotations

from collections.abc import Callable, Set

import pytest

from formula_search.docalculus import rule2, rule3
from formula_search.graph import ADMG

Rule = Callable[[ADMG, Set[str], Set[str], Set[str], Set[str]], bool]

NAPKIN = ADMG.parse("W -> Z; Z -> X; X -> Y; W <-> X; W <-> Y")


# Rule applications to p(y | do(x, z), w) in the napkin graph, from
# Examples 3 to 8 of Yvernes et al. (2026).
@pytest.mark.parametrize(
    ("rule", "x", "z", "w", "valid"),
    [
        pytest.param(
            rule2, "W", "Z", "X", True, id="p(y|do(w,z),x)=p(y|do(w),x,z)"
        ),
        pytest.param(
            rule2, "W", "X", "Z", True, id="p(y|do(w,x),z)=p(y|do(w),x,z)"
        ),
        pytest.param(
            rule2, "W X", "Z", "", True, id="p(y|do(w,x,z))=p(y|do(w,x),z)"
        ),
        pytest.param(
            rule2, "Z", "X", "", True, id="p(y|do(x,z))=p(y|do(z),x)"
        ),
        pytest.param(rule2, "", "X", "", False, id="p(y|do(x))!=p(y|x)"),
        pytest.param(
            rule3, "W", "Z", "X", True, id="p(y|do(w,z),x)=p(y|do(w),x)"
        ),
        pytest.param(rule3, "X", "W", "", True, id="p(y|do(w,x))=p(y|do(x))"),
        pytest.param(
            rule3, "Z", "W", "X", True, id="p(y|do(w,z),x)=p(y|do(z),x)"
        ),
        pytest.param(rule3, "", "W", "X", False, id="p(y|do(w),x)!=p(y|x)"),
        pytest.param(rule3, "", "Z", "X", False, id="p(y|do(z),x)!=p(y|x)"),
    ],
)
def test_rules_in_the_napkin_graph(
    rule: Rule,
    x: str,
    z: str,
    w: str,
    valid: bool,
) -> None:
    assert (
        rule(NAPKIN, {"Y"}, set(x.split()), set(z.split()), set(w.split()))
        is valid
    )
