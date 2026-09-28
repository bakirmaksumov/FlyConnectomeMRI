"""Derive weight-threshold variants of a raw biological graph.

Filtering rule: keep an edge if and only if its synapse count (weight) is
>= the variant's threshold. Nothing else is used: no MRI data, no model
performance, no neuron annotations. Directions and original weights are kept.

Every neuron of the raw selection stays in the variant's node list, even if the
filter removes all of its connections (it then becomes an isolated node).

The raw files are never modified. Before deriving anything, their SHA-256
checksums are compared with the ones recorded at download time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flyconnectome_mri.connectome.drosophila import write_edge_csv
from flyconnectome_mri.connectome.loader import ConnectomeGraph, load_edge_list, load_node_ids
from flyconnectome_mri.connectome.statistics import compute_graph_statistics
from flyconnectome_mri.utils.config import GraphVariantsConfig
from flyconnectome_mri.utils.provenance import git_commit, sha256_file, utc_now_iso


class ChecksumMismatchError(RuntimeError):
    """Raised when a source file no longer matches its recorded checksum."""


def verify_checksum(path: Path, expected: str) -> None:
    """Raise :class:`ChecksumMismatchError` if ``path`` does not have SHA-256 ``expected``."""
    actual = sha256_file(path)
    if actual != expected:
        raise ChecksumMismatchError(
            f"{path.name} has changed since it was downloaded "
            f"(expected sha256 {expected[:12]}..., found {actual[:12]}...). Refusing to derive from it."
        )


def filter_edges_by_weight(edges: pd.DataFrame, min_weight: float) -> pd.DataFrame:
    """Keep rows with ``weight >= min_weight``; direction, weights and order unchanged."""
    return edges.loc[edges["weight"] >= min_weight].reset_index(drop=True)


def graph_to_edge_table(graph: ConnectomeGraph) -> pd.DataFrame:
    """Convert a loaded graph back to a ``source, target, weight`` table using original IDs."""
    return pd.DataFrame({
        "source": graph.node_ids[graph.edges[:, 0]],
        "target": graph.node_ids[graph.edges[:, 1]],
        "weight": graph.weights,
    })


def variant_paths(output_dir: Path, source_edges: Path, variant: str) -> dict[str, Path]:
    """e.g. drosophila_128.csv + "w20" -> drosophila_128_w20.csv / _nodes.csv / _metadata.json."""
    stem = f"{source_edges.stem}_{variant}"
    return {
        "edges": output_dir / f"{stem}.csv",
        "nodes": output_dir / f"{stem}_nodes.csv",
        "metadata": output_dir / f"{stem}_metadata.json",
    }


def derive_weight_variants(
    config: GraphVariantsConfig, root: Path, overwrite: bool = False
) -> dict[str, tuple[ConnectomeGraph, dict[str, Any]]]:
    """Create every variant in ``config``; returns {variant: (graph, metadata)}.

    Args:
        config: Validated variants configuration.
        root: Folder that relative paths in ``config`` are resolved against.
        overwrite: Replace existing derived files (the raw files are never touched).
    """
    edges_path = root / config.source_edges
    nodes_path = root / config.source_nodes
    metadata_path = root / config.source_metadata
    output_dir = root / config.output_dir

    source_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    verify_checksum(edges_path, source_meta["files"]["edges"]["sha256"])
    verify_checksum(nodes_path, source_meta["files"]["nodes"]["sha256"])

    node_table = pd.read_csv(nodes_path)
    node_ids = load_node_ids(nodes_path)
    raw = load_edge_list(edges_path, node_ids=node_ids)  # full validation of the raw data
    raw_edges = graph_to_edge_table(raw)

    all_paths = {v: variant_paths(output_dir, edges_path, v) for v in config.variants}
    existing = [str(p) for paths in all_paths.values() for p in paths.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file(s): {existing}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for variant, threshold in config.variants.items():
        paths = all_paths[variant]
        kept = filter_edges_by_weight(raw_edges, threshold)
        if kept.empty:
            raise ValueError(f"Variant {variant!r} (weight >= {threshold}) keeps no edges")
        write_edge_csv(kept, paths["edges"])
        _write_variant_nodes(node_table, kept, paths["nodes"])

        graph = load_edge_list(paths["edges"], node_ids=node_ids)
        stats = compute_graph_statistics(graph)
        is_primary = variant == config.primary_variant
        metadata = {
            "variant": variant,
            "role": "PRIMARY (pre-declared)" if is_primary else "sensitivity analysis only",
            "filter": {
                "rule": f"keep edge if weight >= {threshold}",
                "min_weight": threshold,
                "based_only_on": "biological edge weight (synapse count)",
            },
            "primary_variant": config.primary_variant,
            "primary_declared_on": config.primary_declared_on,
            "primary_declaration": (
                f"'{config.primary_variant}' was chosen on {config.primary_declared_on}, before any MRI "
                "training, and must not be tuned based on MRI performance."
            ),
            "label": source_meta.get("label"),
            "is_visual_system_selection": source_meta.get("is_visual_system_selection"),
            "source": {
                "edges": {"name": edges_path.name, "sha256": sha256_file(edges_path)},
                "nodes": {"name": nodes_path.name, "sha256": sha256_file(nodes_path)},
                "metadata": {"name": metadata_path.name, "sha256": sha256_file(metadata_path)},
                "dataset": source_meta.get("source", {}).get("dataset"),
                "database": source_meta.get("source", {}).get("database"),
            },
            "counts": {
                "nodes": graph.num_nodes,
                "edges_in_source": len(raw_edges),
                "edges_kept": len(kept),
                "edges_removed": len(raw_edges) - len(kept),
                "isolated_nodes": [int(i) for i in _isolated_ids(graph)],
            },
            "statistics": stats.to_dict(),
            "files": {
                kind: {"name": p.name, "sha256": sha256_file(p)}
                for kind, p in paths.items() if kind != "metadata"
            },
            "created_utc": utc_now_iso(),
            "code": {"git_commit": git_commit()},
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        results[variant] = (graph, metadata)
    return results


def _write_variant_nodes(node_table: pd.DataFrame, kept: pd.DataFrame, path: Path) -> None:
    """Copy ALL source neurons, adding this variant's degrees and an isolation flag."""
    out = node_table.copy()
    out_degree = kept["source"].value_counts()
    in_degree = kept["target"].value_counts()
    out["out_degree"] = out["bodyId"].map(out_degree).fillna(0).astype("int64")
    out["in_degree"] = out["bodyId"].map(in_degree).fillna(0).astype("int64")
    out["isolated"] = (out["out_degree"] + out["in_degree"]) == 0
    out.to_csv(path, index=False, lineterminator="\n")


def _isolated_ids(graph: ConnectomeGraph) -> np.ndarray:
    pattern = graph.adjacency != 0
    degree = np.asarray(pattern.sum(axis=0)).ravel() + np.asarray(pattern.sum(axis=1)).ravel()
    return graph.node_ids[degree == 0]

