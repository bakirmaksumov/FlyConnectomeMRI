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
        num_isolated_nodes: Nodes with no incoming and no outgoing edges.
        density: E / (N * (N - 1)), the fraction of possible directed edges
            between distinct nodes that exist. Can slightly exceed 1 only if
            self-loops are kept (NetworkX convention).
        mean_in_degree: Average number of incoming edges per node (= E / N).
        mean_out_degree: Average number of outgoing edges per node (= E / N).
        max_in_degree: Most incoming edges of any single node.
        max_out_degree: Most outgoing edges of any single node.
        max_degree: Largest in-degree + out-degree of any single node.
        in_degree_std, out_degree_std: Standard deviation of the degrees
            (population std). Small = every node similar; large = hubs exist.
        reciprocity: Fraction of edges A->B whose reverse B->A also exists.
        average_clustering: Mean directed clustering coefficient (unweighted,
            NetworkX definition): how often a node's neighbours are also
            connected to each other. High in small-world graphs.
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
    num_isolated_nodes: int
    density: float
    mean_in_degree: float
    mean_out_degree: float
    max_in_degree: int
    max_out_degree: int
    max_degree: int
    in_degree_std: float
    out_degree_std: float
    reciprocity: float
    average_clustering: float
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
        num_isolated_nodes=int(((in_degree + out_degree) == 0).sum()),
        density=nx.density(graph),
        # Every edge has exactly one sender and one receiver, so the total
        # in-degree equals the total out-degree (= E), and both means are E / N.
        mean_in_degree=float(in_degree.mean()),
        mean_out_degree=float(out_degree.mean()),
        max_in_degree=int(in_degree.max()),
        max_out_degree=int(out_degree.max()),
        max_degree=int((in_degree + out_degree).max()),
        in_degree_std=float(in_degree.std()),
        out_degree_std=float(out_degree.std()),
        reciprocity=_reciprocity(connectome),
        average_clustering=float(nx.average_clustering(graph)),
        num_weakly_connected_components=len(weak),
        largest_weakly_connected_component=max(weak),
        num_strongly_connected_components=len(strong),
        largest_strongly_connected_component=max(strong),
        total_weight=float(weights.sum()),
        mean_weight=float(weights.mean()),
        min_weight=float(weights.min()),
        max_weight=float(weights.max()),
    )


def _reciprocity(connectome: ConnectomeGraph) -> float:
    """Fraction of non-self-loop edges A->B whose reverse B->A also exists."""
    pattern = (connectome.adjacency != 0).astype(np.int8)
    pattern.setdiag(0)
    pattern.eliminate_zeros()
    if pattern.nnz == 0:
        return 0.0
    # Element-wise product with the transpose keeps exactly the reciprocated edges.
    return float(pattern.multiply(pattern.T).nnz / pattern.nnz)


def format_statistics(stats: GraphStatistics) -> str:
    """Return a human-readable, aligned multi-line summary."""
    rows = [
        ("Nodes", f"{stats.num_nodes}"),
        ("Edges", f"{stats.num_edges}"),
        ("Self-loops", f"{stats.num_self_loops}"),
        ("Isolated nodes", f"{stats.num_isolated_nodes}"),
        ("Density", f"{stats.density:.6f}"),
        ("Mean in-degree", f"{stats.mean_in_degree:.4f}"),
        ("Mean out-degree", f"{stats.mean_out_degree:.4f}"),
        ("Max in-degree", f"{stats.max_in_degree}"),
        ("Max out-degree", f"{stats.max_out_degree}"),
        ("Max degree (in+out)", f"{stats.max_degree}"),
        ("In/out-degree std", f"{stats.in_degree_std:.4f} / {stats.out_degree_std:.4f}"),
        ("Reciprocity", f"{stats.reciprocity:.4f}"),
        ("Average clustering", f"{stats.average_clustering:.4f}"),
        ("Weakly connected components", f"{stats.num_weakly_connected_components}"
                                        f" (largest: {stats.largest_weakly_connected_component} nodes)"),
        ("Strongly connected components", f"{stats.num_strongly_connected_components}"
                                          f" (largest: {stats.largest_strongly_connected_component} nodes)"),
        ("Total weight", f"{stats.total_weight:g}"),
        ("Weight mean / min / max", f"{stats.mean_weight:g} / {stats.min_weight:g} / {stats.max_weight:g}"),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}} : {value}" for label, value in rows)


# (label, attribute, format) rows for side-by-side comparison tables.
_COMPARISON_ROWS = (
    ("Nodes", "num_nodes", "{:d}"),
    ("Edges", "num_edges", "{:d}"),
    ("Self-loops", "num_self_loops", "{:d}"),
    ("Isolated nodes", "num_isolated_nodes", "{:d}"),
    ("Density", "density", "{:.4f}"),
    ("Mean in/out-degree", "mean_out_degree", "{:.2f}"),
    ("Max in-degree", "max_in_degree", "{:d}"),
    ("Max out-degree", "max_out_degree", "{:d}"),
    ("Max degree (in+out)", "max_degree", "{:d}"),
    ("In-degree std", "in_degree_std", "{:.2f}"),
    ("Out-degree std", "out_degree_std", "{:.2f}"),
    ("Reciprocity", "reciprocity", "{:.3f}"),
    ("Avg clustering", "average_clustering", "{:.3f}"),
    ("Weak components", "num_weakly_connected_components", "{:d}"),
    ("Largest weak comp.", "largest_weakly_connected_component", "{:d}"),
    ("Strong components", "num_strongly_connected_components", "{:d}"),
    ("Largest strong comp.", "largest_strongly_connected_component", "{:d}"),
    ("Total weight", "total_weight", "{:.0f}"),
    ("Mean weight", "mean_weight", "{:.2f}"),
)


def format_comparison(graphs: dict[str, ConnectomeGraph]) -> str:
    """Return a side-by-side statistics table, one column per graph."""
    stats = {name: compute_graph_statistics(g) for name, g in graphs.items()}
    label_width = max(len(label) for label, _, _ in _COMPARISON_ROWS)
    widths = {name: max(len(name), 10) for name in stats}
    header = " " * label_width + " | " + " | ".join(f"{name:>{widths[name]}}" for name in stats)
    lines = [header, "-" * len(header)]
    for label, attr, fmt in _COMPARISON_ROWS:
        cells = " | ".join(f"{fmt.format(getattr(s, attr)):>{widths[n]}}" for n, s in stats.items())
        lines.append(f"{label:<{label_width}} | {cells}")
    return "\n".join(lines)
