"""Descriptive statistics for a directed weighted graph.

These numbers are how we will later check that a random control graph is
comparable to the fly graph (same node count, similar edge count, similar
degrees), so every one is defined precisely here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
import numpy as np

from flyconnectome_mri.connectome.loader import ConnectomeGraph


@dataclass(frozen=True)
class GraphStatistics:
    """Summary statistics of a directed weighted graph.

    Attributes:
        num_nodes: N, number of nodes.
        num_edges: E, number of directed edges (self-loops included, if kept).
        num_self_loops: Edges from a node to itself.
        density: E / (N * (N - 1)), the fraction of possible directed edges
            between distinct nodes that exist. Can slightly exceed 1 only if
            self-loops are kept (NetworkX convention).
        mean_in_degree: Average number of incoming edges per node (= E / N).
        mean_out_degree: Average number of outgoing edges per node (= E / N).
        max_in_degree: Most incoming edges of any single node.
        max_out_degree: Most outgoing edges of any single node.
        max_degree: Largest in-degree + out-degree of any single node.
        num_weakly_connected_components: Groups of nodes connected when edge
            direction is ignored.
        largest_weakly_connected_component: Size of the biggest such group.
        num_strongly_connected_components: Groups where every node can reach
            every other node following edge directions.
        largest_strongly_connected_component: Size of the biggest such group.
        total_weight, mean_weight, min_weight, max_weight: Edge-weight summary.
    """

    num_nodes: int
    num_edges: int
    num_self_loops: int
    density: float
    mean_in_degree: float
    mean_out_degree: float
    max_in_degree: int
    max_out_degree: int
    max_degree: int
    num_weakly_connected_components: int
    largest_weakly_connected_component: int
    num_strongly_connected_components: int
    largest_strongly_connected_component: int
    total_weight: float
    mean_weight: float
    min_weight: float
    max_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_graph_statistics(connectome: ConnectomeGraph) -> GraphStatistics:
    """Compute :class:`GraphStatistics` for a loaded graph."""
    graph = connectome.graph
    n = connectome.num_nodes
    e = connectome.num_edges

    # Degrees from the sparse adjacency: row sums of a 0/1 pattern = out-degree,
    # column sums = in-degree. (A self-loop adds 1 to both.)
    pattern = (connectome.adjacency != 0).astype(np.int64)
    out_degree = np.asarray(pattern.sum(axis=1)).ravel()
    in_degree = np.asarray(pattern.sum(axis=0)).ravel()

    weak = [len(c) for c in nx.weakly_connected_components(graph)]
    strong = [len(c) for c in nx.strongly_connected_components(graph)]
    weights = connectome.weights

    return GraphStatistics(
        num_nodes=n,
        num_edges=e,
        num_self_loops=nx.number_of_selfloops(graph),
        density=nx.density(graph),
        # Every edge has exactly one sender and one receiver, so the total
        # in-degree equals the total out-degree (= E), and both means are E / N.
        mean_in_degree=float(in_degree.mean()),
        mean_out_degree=float(out_degree.mean()),
        max_in_degree=int(in_degree.max()),
        max_out_degree=int(out_degree.max()),
        max_degree=int((in_degree + out_degree).max()),
        num_weakly_connected_components=len(weak),
        largest_weakly_connected_component=max(weak),
        num_strongly_connected_components=len(strong),
        largest_strongly_connected_component=max(strong),
        total_weight=float(weights.sum()),
        mean_weight=float(weights.mean()),
        min_weight=float(weights.min()),
        max_weight=float(weights.max()),
    )


def format_statistics(stats: GraphStatistics) -> str:
    """Return a human-readable, aligned multi-line summary."""
    rows = [
        ("Nodes", f"{stats.num_nodes}"),
        ("Edges", f"{stats.num_edges}"),
        ("Self-loops", f"{stats.num_self_loops}"),
        ("Density", f"{stats.density:.6f}"),
        ("Mean in-degree", f"{stats.mean_in_degree:.4f}"),
        ("Mean out-degree", f"{stats.mean_out_degree:.4f}"),
        ("Max in-degree", f"{stats.max_in_degree}"),
        ("Max out-degree", f"{stats.max_out_degree}"),
        ("Max degree (in+out)", f"{stats.max_degree}"),
        ("Weakly connected components", f"{stats.num_weakly_connected_components}"
                                        f" (largest: {stats.largest_weakly_connected_component} nodes)"),
        ("Strongly connected components", f"{stats.num_strongly_connected_components}"
                                          f" (largest: {stats.largest_strongly_connected_component} nodes)"),
        ("Total weight", f"{stats.total_weight:g}"),
        ("Weight mean / min / max", f"{stats.mean_weight:g} / {stats.min_weight:g} / {stats.max_weight:g}"),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}} : {value}" for label, value in rows)
