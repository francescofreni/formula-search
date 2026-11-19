"""Automatic families preserve nodes, acyclicity, and explicit overrides."""

from dataclasses import replace

import pytest

from distance_metric import DistanceConfig, estimate_distance, resolve_edges
from distance_metric.configs import ADJUSTMENT_01_A_VS_B as EXAMPLE
from formula_search.graph import ADMG


def test_automatic_default_and_directed_only():
    config = DistanceConfig(
        graph=EXAMPLE.graph,
        target=EXAMPLE.target,
        formulas=EXAMPLE.formulas,
    )
    assert config.optional_edges is None
    config = replace(config, topological_order=("A", "B", "X", "Y"))
    family = resolve_edges(config)
    assert family.topological_order == ("A", "B", "X", "Y")
    assert family.optional_edges == (
        "A -> X",
        "B -> Y",
        "A <-> B",
        "A <-> X",
        "A <-> Y",
        "B <-> X",
        "B <-> Y",
        "X <-> Y",
    )
    directed = resolve_edges(replace(config, include_bidirected=False))
    assert directed.optional_edges == EXAMPLE.optional_edges
    assert (
        ADMG.parse("; ".join((config.graph, *family.optional_edges))).nodes
        == ADMG.parse(config.graph).nodes
    )


def test_existing_bidirected_edges_are_preserved_not_added_twice():
    config = replace(
        EXAMPLE,
        graph="X -> Y; X <-> Y",
        optional_edges=None,
        topological_order=("X", "Y"),
    )
    for flag in (True, False):
        assert (
            resolve_edges(
                replace(config, include_bidirected=flag)
            ).optional_edges
            == ()
        )
    assert "X <-> Y" in config.graph


def test_automatic_order_is_deterministic_and_can_be_overridden():
    config = replace(
        EXAMPLE,
        graph="B; A; X -> Y",
        topological_order=("A", "B", "X", "Y"),
        optional_edges=None,
        include_bidirected=False,
    )
    assert resolve_edges(config) == resolve_edges(
        replace(config, graph="X -> Y; A; B")
    )
    alternate = resolve_edges(
        replace(config, topological_order=("B", "A", "X", "Y"))
    )
    assert "B -> A" in alternate.optional_edges
    assert "A -> B" not in alternate.optional_edges
    full = ADMG.parse("; ".join((config.graph, *alternate.optional_edges)))
    assert full.topological_order() == alternate.topological_order


@pytest.mark.parametrize(
    "order",
    [
        ("A", "B", "X"),
        ("A", "B", "X", "X"),
        ("A", "B", "X", "Q"),
        ("Y", "X", "B", "A"),
    ],
)
def test_invalid_order_rejected(order):
    with pytest.raises(ValueError, match="topological_order"):
        resolve_edges(
            replace(EXAMPLE, optional_edges=None, topological_order=order)
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"include_bidirected": "false"},
        {"include_bidirected": 0},
        {"topological_order": "ABXY"},
    ],
)
def test_invalid_generation_settings_rejected(changes):
    with pytest.raises(ValueError):
        replace(EXAMPLE, **changes)


def test_explicit_edges_override_flag_and_default_order():
    config = replace(
        EXAMPLE,
        graph="A; B; X -> Y",
        optional_edges=("B -> A", "X <-> Y"),
        include_bidirected=False,
    )
    assert resolve_edges(config).optional_edges == config.optional_edges
    assert resolve_edges(config).topological_order == ("B", "A", "X", "Y")
    assert (
        resolve_edges(replace(config, optional_edges=())).optional_edges == ()
    )
    with pytest.raises(ValueError, match="conflicts"):
        resolve_edges(replace(config, topological_order=("A", "B", "X", "Y")))


def test_real_automatic_estimate_matches_explicit_edges():
    config = replace(EXAMPLE, optional_edges=None, epsilon=0.4)
    auto = estimate_distance(config)
    explicit = estimate_distance(
        replace(
            config,
            optional_edges=resolve_edges(
                replace(config, topological_order=("A", "B", "X", "Y"))
            ).optional_edges,
        )
    )
    assert auto.distance == explicit.distance
    assert auto.verifier_calls == explicit.verifier_calls
    assert auto.verifier_call_budget == explicit.verifier_call_budget
    directed = estimate_distance(replace(config, include_bidirected=False))
    assert directed.distance == 0.5
