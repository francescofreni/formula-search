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


def test_mutilated_removes_the_edges_into_and_out_of_nodes() -> None:
    graph = ADMG.parse(NAPKIN).mutilated(into={"X"}, out_of={"Z"})

    assert graph.nodes == {"W", "X", "Y", "Z"}
    assert graph.directed == {("W", "Z"), ("X", "Y")}
    assert graph.bidirected == {frozenset({"W", "Y"})}


def test_parents_of_a_set_of_nodes() -> None:
    assert ADMG.parse(NAPKIN).parents({"X", "Y"}) == {"X", "Z"}


def test_ancestors_follow_directed_edges_only() -> None:
    graph = ADMG.parse(NAPKIN)

    assert graph.ancestors({"X"}) == {"W", "X", "Z"}
    assert graph.subgraph({"W", "Y", "Z"}).ancestors({"Y"}) == {"Y"}


def test_districts_follow_bidirected_edges() -> None:
    assert ADMG.parse(NAPKIN).districts() == {
        frozenset({"W", "X", "Y"}),
        frozenset({"Z"}),
    }


@pytest.mark.parametrize(
    ("graph", "first", "second", "given", "separated"),
    [
        pytest.param("A -> B; B -> C", "A", "C", "", False, id="chain"),
        pytest.param(
            "A -> B; B -> C", "A", "C", "B", True, id="blocked chain"
        ),
        pytest.param("A -> B; C -> B", "A", "C", "", True, id="collider"),
        pytest.param(
            "A -> B; C -> B", "A", "C", "B", False, id="open collider"
        ),
        pytest.param(
            "A -> B; C -> B; B -> D", "A", "C", "D", False, id="descendant"
        ),
        pytest.param("A <-> B; B <-> C", "A", "C", "", True, id="bidirected"),
        pytest.param("A <-> B; B <-> C", "A", "C", "B", False, id="open"),
        # Z -> X <-> W <-> Y is open given its colliders X and W.
        pytest.param(NAPKIN, "Z", "Y", "W X", False, id="napkin"),
    ],
)
def test_d_separated_follows_colliders_and_bidirected_edges(
    graph: str,
    first: str,
    second: str,
    given: str,
    separated: bool,
) -> None:
    admg = ADMG.parse(graph)

    assert admg.d_separated(first.split(), second.split(), given.split()) is (
        separated
    )


def test_d_separated_rejects_overlapping_sets() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        ADMG.parse("A -> B").d_separated({"A"}, {"B"}, {"A"})


def test_topological_order_places_ties_alphabetically() -> None:
    graph = ADMG.parse("D -> A; C -> A; B")

    assert graph.topological_order() == ("B", "C", "D", "A")


def test_str_writes_the_graph_in_hiprofs_syntax() -> None:
    graph = ADMG.parse("T -> M; M -> Y; T <-> Y; C")

    assert str(graph) == "C; M -> Y; T -> M; T <-> Y"
    assert ADMG.parse(str(graph)) == graph


def test_complete_joins_every_pair_along_the_topological_order() -> None:
    graph = ADMG.parse("T -> M; M -> Y; T <-> Y").complete()

    assert graph.directed == {("T", "M"), ("T", "Y"), ("M", "Y")}
    assert not graph.bidirected


def test_closure_fixes_every_other_node_that_it_can() -> None:
    graph = ADMG.parse(NAPKIN)

    assert graph.closure({"Y"}) == {"Y"}
    # Neither W nor X is fixable while Y is not fixed.
    assert graph.closure({"Y", "Z"}) == {"W", "X", "Y", "Z"}


@pytest.mark.parametrize(
    ("graph", "constrained"),
    [
        pytest.param("T -> M; M -> Y; T <-> Y", False, id="front-door"),
        pytest.param("A -> B; B -> C", True, id="chain"),
        pytest.param(NAPKIN, True, id="napkin"),
        # A and Y are not adjacent, yet A is a parent of the closure of Y.
        pytest.param("A -> M; M -> Y; M <-> Y", False, id="instrument"),
    ],
)
def test_implies_constraints_unless_every_pair_is_adjacent_once_projected(
    graph: str,
    constrained: bool,
) -> None:
    assert ADMG.parse(graph).implies_constraints() is constrained
