"""Acyclic directed mixed graphs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hiprof.graph import parse_graph


@dataclass(frozen=True)
class ADMG:
    """An acyclic directed mixed graph over observed variables.

    :param nodes: Variable names.
    :param directed: Directed edges, as ``(parent, child)`` pairs.
    :param bidirected: Bidirected edges, as sets of their two endpoints.
    """

    nodes: frozenset[str]
    directed: frozenset[tuple[str, str]] = frozenset()
    bidirected: frozenset[frozenset[str]] = frozenset()

    @classmethod
    def parse(cls, text: str) -> ADMG:
        """Parse a graph in hiprof's syntax, such as ``"T -> Y; T <-> Y"``.

        :param text: Graph specification accepted by hiprof.
        :returns: The graph.
        :raises ValueError: If the specification is invalid or the graph
            contains a directed cycle.
        """
        nodes = parse_graph(text).nodes.values()
        # hiprof represents each bidirected edge by a latent common parent.
        return cls(
            nodes=frozenset(node.name for node in nodes if node.observed),
            directed=frozenset(
                (node.name, child.name)
                for node in nodes
                if node.observed
                for child in node.children
            ),
            bidirected=frozenset(
                frozenset(child.name for child in node.children)
                for node in nodes
                if not node.observed
            ),
        )

    def subgraph(self, nodes: Iterable[str]) -> ADMG:
        """Return the subgraph induced by ``nodes``."""
        nodes = frozenset(nodes)
        return ADMG(
            nodes=nodes,
            directed=frozenset(
                edge for edge in self.directed if set(edge) <= nodes
            ),
            bidirected=frozenset(
                edge for edge in self.bidirected if edge <= nodes
            ),
        )

    def ancestors(self, nodes: Iterable[str]) -> frozenset[str]:
        """Return the nodes with a directed path into ``nodes``.

        Every node is its own ancestor.
        """
        ancestors = frozenset(nodes)
        while True:
            parents = {
                parent for parent, child in self.directed if child in ancestors
            }
            if parents <= ancestors:
                return ancestors
            ancestors |= parents

    def district(self, node: str) -> frozenset[str]:
        """Return the nodes joined to ``node`` by bidirected paths."""
        district = frozenset({node})
        while True:
            edges = [edge for edge in self.bidirected if edge & district]
            siblings = frozenset().union(*edges)
            if siblings <= district:
                return district
            district |= siblings

    def districts(self) -> frozenset[frozenset[str]]:
        """Return the maximal sets of nodes joined by bidirected paths."""
        return frozenset(self.district(node) for node in self.nodes)

    def topological_order(self) -> tuple[str, ...]:
        """Return a topological order that places ties alphabetically."""
        order: list[str] = []
        while unplaced := self.nodes - set(order):
            children = {
                child for parent, child in self.directed if parent in unplaced
            }
            order.append(min(unplaced - children))
        return tuple(order)
