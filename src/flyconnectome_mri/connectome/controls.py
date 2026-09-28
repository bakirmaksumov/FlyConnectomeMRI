"""Generate, save and validate control graphs matched to a biological graph.

Determinism: each (control type, seed) pair gets its own independent random
stream, ``numpy.random.default_rng([seed, CONTROL_STREAM[control]])``. The same
seed therefore always gives the same graph, and changing one control type never
changes another.

Node IDs in the saved files:
    er / ws / ba : 0..n-1. These are abstract slots, NOT neurons. Slot k is
                   aligned with the k-th biological node (sorted bodyId) only
                   so that later phases can use the same feature->node mapping.
    rewired      : the biological bodyIds, because every neuron keeps its own
                   in/out-degree; only its partners are randomised.

Weights: ER/WS/BA receive a random permutation of the biological weights (same
weight distribution, random placement). In the rewired graph each weight stays
with its source neuron (see ``degree_preserving_rewire``).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flyconnectome_mri.connectome.drosophila import write_edge_csv
from flyconnectome_mri.connectome.loader import ConnectomeGraph, load_edge_list, load_node_ids
from flyconnectome_mri.connectome.random_graphs import (
    barabasi_albert_directed,
    degree_preserving_rewire,
    edge_overlap,
    erdos_renyi_directed,
    watts_strogatz_directed,
)
from flyconnectome_mri.connectome.statistics import compute_graph_statistics
from flyconnectome_mri.utils.config import ControlsConfig
from flyconnectome_mri.utils.provenance import git_commit, sha256_file, utc_now_iso

# Fixed stream IDs: never renumber, or every saved control graph changes.
CONTROL_STREAM = {"er": 1, "ws": 2, "ba": 3, "rewired": 4}

DESCRIPTIONS = {
    "er": "Directed Erdos-Renyi G(n, m): same node and edge count, edges placed uniformly at random.",
    "ws": "Directed Watts-Strogatz variant: ring lattice with exact edge count, each edge rewired with prob p.",
    "ba": "Directed Barabasi-Albert variant: preferential attachment, random edge directions, exact edge count.",
    "rewired": "Degree-preserving rewiring of the biological graph (directed double-edge swaps): every "
               "neuron keeps its exact in- and out-degree; which neurons connect is randomised.",
}


def control_paths(output_dir: Path, control: str, label: str, seed: int) -> dict[str, Path]:
    stem = f"{control}_{label}_seed{seed}"
    return {
        "edges": output_dir / f"{stem}.csv",
        "nodes": output_dir / f"{stem}_nodes.csv",
        "metadata": output_dir / f"{stem}_metadata.json",
    }


def make_rng(control: str, seed: int) -> np.random.Generator:
    """Independent, reproducible random stream for one (control, seed) pair."""
    return np.random.default_rng([seed, CONTROL_STREAM[control]])


def generate_control_edges(
    control: str, bio: ConnectomeGraph, seed: int, ws_rewire_prob: float, swaps_per_edge: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (edges in index space, weights, generator details) for one control graph."""
    rng = make_rng(control, seed)
    n, m = bio.num_nodes, bio.num_edges
    details: dict[str, Any] = {}
    if control == "er":
        edges = erdos_renyi_directed(n, m, rng)
    elif control == "ws":
        edges = watts_strogatz_directed(n, m, ws_rewire_prob, rng)
        details["rewire_prob"] = ws_rewire_prob
    elif control == "ba":
        edges = barabasi_albert_directed(n, m, rng)
    elif control == "rewired":
        result = degree_preserving_rewire(bio.edges, bio.weights, swaps_per_edge, rng)
        details.update(swaps_per_edge=swaps_per_edge, attempted_swaps=result.attempted_swaps,
                       accepted_swaps=result.accepted_swaps)
        return result.edges, result.weights, details
    else:
        raise ValueError(f"unknown control {control!r}")
    weights = rng.permutation(bio.weights)
    return edges, weights, details


def generate_controls(config: ControlsConfig, root: Path, overwrite: bool = False) -> dict[tuple[str, int], ConnectomeGraph]:
    """Generate every (control, seed) in ``config``; returns {(control, seed): graph}."""
    edges_path = root / config.source_edges
    nodes_path = root / config.source_nodes
    output_dir = root / config.output_dir

    bio_ids = load_node_ids(nodes_path, column=config.source_node_column)
    bio = load_edge_list(edges_path, node_ids=bio_ids)
    if bio.num_nodes != len(bio_ids):
        raise RuntimeError("biological node list and loaded graph disagree")

    jobs = [(c, s) for c in config.controls for s in config.seeds]
    existing = [str(p) for c, s in jobs for p in control_paths(output_dir, c, config.label, s).values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file(s): {existing[:5]}... Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for control, seed in jobs:
        edges, weights, details = generate_control_edges(
            control, bio, seed, config.ws_rewire_prob, config.rewire_swaps_per_edge
        )
        node_ids = bio.node_ids if control == "rewired" else np.arange(bio.num_nodes, dtype=np.int64)
        paths = control_paths(output_dir, control, config.label, seed)
        table = pd.DataFrame({"source": node_ids[edges[:, 0]], "target": node_ids[edges[:, 1]], "weight": weights})
        write_edge_csv(table, paths["edges"])
        pd.DataFrame({"node_id": node_ids}).to_csv(paths["nodes"], index=False, lineterminator="\n")

        graph = load_edge_list(paths["edges"], node_ids=node_ids)  # validate what was written
        metadata = {
            "control": control,
            "description": DESCRIPTIONS[control],
            "seed": seed,
            "random_stream": f"numpy.random.default_rng([{seed}, {CONTROL_STREAM[control]}])",
            "generator_details": details,
            "node_ids": "biological bodyIds" if control == "rewired"
                        else "abstract slots 0..n-1 (not neurons)",
            "matched_to": {
                "edges": {"name": edges_path.name, "sha256": sha256_file(edges_path)},
                "nodes": {"name": nodes_path.name, "sha256": sha256_file(nodes_path)},
                "num_nodes": bio.num_nodes,
                "num_edges": bio.num_edges,
            },
            "edge_overlap_with_biological_graph": edge_overlap(edges, bio.edges),
            "statistics": asdict(compute_graph_statistics(graph)),
            "files": {k: {"name": p.name, "sha256": sha256_file(p)} for k, p in paths.items() if k != "metadata"},
            "created_utc": utc_now_iso(),
            "code": {"git_commit": git_commit()},
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        results[(control, seed)] = graph
    return results
