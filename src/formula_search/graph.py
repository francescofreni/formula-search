"""Acyclic directed mixed graphs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

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

    def __str__(self) -> str:
        """Write the graph in hiprof's syntax, as ``"T -> Y; T <-> Y"``."""
        linked = frozenset().union(*self.directed, *self.bidirected)
        return "; ".join(
            sorted(self.nodes - linked)
            + sorted(f"{parent} -> {child}" for parent, child in self.directed)
            + sorted(" <-> ".join(sorted(edge)) for edge in self.bidirected)
        )

    def complete(self) -> ADMG:
        """Return the complete DAG over the nodes, in topological order.

        Its model imposes no constraint on the observed law (Bhattacharya et
        al., 2022, p. 10).
        """
        order = self.topological_order()
        return ADMG(self.nodes, frozenset(combinations(order, 2)))

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

    def mutilated(
        self,
        into: Iterable[str] = (),
        out_of: Iterable[str] = (),
    ) -> ADMG:
        """Return the graph without the edges into ``into`` or out of ``out_of``.

        Edges into a node have an arrowhead at it, so these include the
        bidirected edges at ``into``.
        """
        into, out_of = frozenset(into), frozenset(out_of)
        return ADMG(
            nodes=self.nodes,
            directed=frozenset(
                (parent, child)
                for parent, child in self.directed
                if child not in into and parent not in out_of
            ),
            bidirected=frozenset(
                edge for edge in self.bidirected if not edge & into
            ),
        )

    def parents(self, nodes: Iterable[str]) -> frozenset[str]:
        """Return the nodes with a directed edge into ``nodes``."""
        nodes = frozenset(nodes)
        return frozenset(
            parent for parent, child in self.directed if child in nodes
        )

    def ancestors(self, nodes: Iterable[str]) -> frozenset[str]:
        """Return the nodes with a directed path into ``nodes``.

        Every node is its own ancestor.
        """
        ancestors = frozenset(nodes)
        while parents := self.parents(ancestors) - ancestors:
            ancestors |= parents
        return ancestors

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

    def d_separated(
        self,
        first: Iterable[str],
        second: Iterable[str],
        given: Iterable[str] = (),
    ) -> bool:
        """Return whether ``given`` d-separates ``first`` from ``second``.

        In an ADMG, this is m-separation, that is, d-separation once every
        bidirected edge is replaced by a latent common parent. It holds if
        no path joins the sets outside ``given`` in the augmented graph of
        their ancestors, where each district and its parents form a clique
        (Richardson, 2003, Theorem 1).

        :raises ValueError: If the sets overlap.
        """
        first, second, given = map(frozenset, (first, second, given))
        if first & second or given & (first | second):
            raise ValueError("The sets must be disjoint.")
        ancestral = self.subgraph(self.ancestors(first | second | given))
        neighbours: dict[str, set[str]] = {n: set() for n in ancestral.nodes}
        for district in ancestral.districts():
            clique = district | ancestral.parents(district)
            for node in clique:
                neighbours[node] |= clique

        reached, pending = set(first), list(first)
        while pending:
            for node in neighbours[pending.pop()] - given - reached:
                reached.add(node)
                pending.append(node)
        return not reached & second

    def topological_order(self) -> tuple[str, ...]:
        """Return a topological order that places ties alphabetically."""
        order: list[str] = []
        while unplaced := self.nodes - set(order):
            children = {
                child for parent, child in self.directed if parent in unplaced
            }
            order.append(min(unplaced - children))
        return tuple(order)

    def closure(self, nodes: Iterable[str]) -> frozenset[str]:
        """Return the reachable closure of ``nodes``.

        The other nodes are fixed while one of them is fixable, that is, has
        no descendant in its district but itself (Richardson et al., 2023,
        Definition 17). Fixing removes the edges into a node, so the nodes
        left unfixed induce a subgraph, and the order does not matter, since
        fixing a node never makes another unfixable (their Lemma 22).
        """
        nodes, graph = frozenset(nodes), self
        while fixable := [
            node
            for node in sorted(graph.nodes - nodes)
            if not any(
                node in graph.ancestors({other})
                for other in graph.district(node) - {node}
            )
        ]:
            graph = graph.subgraph(graph.nodes - {fixable[0]})
        return graph.nodes

    def implies_constraints(self) -> bool:
        """Return whether the graph implies equality constraints.

        These are the conditional independences and Verma constraints that
        the graph implies on the observed law. By Algorithm 1 and Theorem 1
        of Bhattacharya et al. (2022), the graph implies some exactly when
        two nodes are not adjacent in its maximal arid projection: neither
        is a parent of the closure of the other, and the closure of both is
        not a single district.
        """
        for first, second in combinations(sorted(self.nodes), 2):
            closures = self.closure({first}), self.closure({second})
            joint = self.closure({first, second})
            if not (
                first in self.parents(closures[1]) - closures[1]
                or second in self.parents(closures[0]) - closures[0]
                or self.subgraph(joint).district(first) == joint
            ):
                return True
        return False
