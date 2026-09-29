from __future__ import annotations

import pytest

from formula_search.graph import ADMG

NAPKIN = "X -> Y; W -> Z; Z -> X; X <-> W; W <-> Y"


def test_parse_reads_directed_bidirected_and_isolated_nodes() -> None:
    graph = ADMG.parse("T -> M; M -> Y; T <-> Y; C")

    assert graph.nodes == {"C", "M", "T", "Y"}
    assert graph.directed == {("T", "M"), ("M", "Y")}
    assert graph.bidirected == {frozenset({"T", "Y"})}


def test_parse_rejects_directed_cycles() -> None:
    with pytest.raises(ValueError, match="directed cycle"):
        ADMG.parse("A -> B; B -> A")


def test_subgraph_keeps_the_edges_between_its_nodes() -> None:
    graph = ADMG.parse(NAPKIN).subgraph({"W", "X", "Y"})

    assert graph.nodes == {"W", "X", "Y"}
    assert graph.directed == {("X", "Y")}
    assert graph.bidirected == {frozenset({"W", "X"}), frozenset({"W", "Y"})}


def test_ancestors_follow_directed_edges_only() -> None:
    graph = ADMG.parse(NAPKIN)

    assert graph.ancestors({"X"}) == {"W", "X", "Z"}
    assert graph.subgraph({"W", "Y", "Z"}).ancestors({"Y"}) == {"Y"}


def test_districts_follow_bidirected_edges() -> None:
    assert ADMG.parse(NAPKIN).districts() == {
        frozenset({"W", "X", "Y"}),
        frozenset({"Z"}),
    }


def test_topological_order_places_ties_alphabetically() -> None:
    graph = ADMG.parse("D -> A; C -> A; B")

    assert graph.topological_order() == ("B", "C", "D", "A")
