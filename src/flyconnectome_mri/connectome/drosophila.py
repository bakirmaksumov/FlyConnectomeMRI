"""Extract a small Drosophila subgraph from neuPrint (Janelia MaleCNS) and save it
in the canonical ``source,target,weight`` format, with provenance metadata.

Programmer analogy: neuPrint is a graph database of the fly's nervous system.
Each neuron is a node with an integer ``bodyId``; each ``ConnectsTo`` relation
says "neuron A makes ``weight`` synapses onto neuron B". We ask the database for
a small, precisely defined set of neurons and all connections among them.

Selection method ``roi_top_synapses`` (generic, anatomical):
    1. Take every neuron that has synapses inside one ROI (brain region).
    2. Rank them by synapse count inside that ROI (pre + post), descending;
       ties broken by smaller bodyId, so the ranking is deterministic.
    3. Keep the top ``num_nodes``.
    4. Fetch all connections whose source AND target are both selected.

This is NOT a functional circuit and NOT a visual-system selection, even if the
ROI happens to be a visual region. It is labelled "generic" in the metadata.

Security: the API token is read from an environment variable, passed straight
to the neuPrint client, and never logged, printed or written to any file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import neuprint
import numpy as np
import pandas as pd

from flyconnectome_mri.connectome.loader import ConnectomeGraph, load_edge_list
from flyconnectome_mri.utils.config import DrosophilaFetchConfig
from flyconnectome_mri.utils.logging import get_logger
from flyconnectome_mri.utils.provenance import git_commit, sha256_file, utc_now_iso

MALECNS_REFERENCE = {
    "dataset_homepage": "https://male-cns.janelia.org/",
    "preprint": "https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2",
    "preprint_doi": "10.1101/2025.10.09.680999",
    "license": "CC-BY 4.0 (per https://male-cns.janelia.org/)",
}
# Neuron annotation columns copied to the node table when neuPrint provides them.
ANNOTATION_COLUMNS = ("type", "instance", "superclass", "class", "subclass", "status")
# neuPrint metadata keys recorded (when present) to pin the exact database state.
_META_KEYS = ("lastDatabaseEdit", "uuid", "latestMutationId")


class NeuprintTokenError(RuntimeError):
    """Raised when the neuPrint token environment variable is missing."""


# ---------------------------------------------------------------------------
# Connecting
# ---------------------------------------------------------------------------


def read_token(env_var: str) -> str:
    """Return the API token from environment variable ``env_var`` (never logged)."""
    token = os.environ.get(env_var, "").strip()
    if not token:
        raise NeuprintTokenError(
            f"Environment variable {env_var} is not set. Get a token from your account "
            f"page at https://neuprint.janelia.org and set it, e.g. in PowerShell: "
            f"$env:{env_var} = '<your token>'"
        )
    return token


def connect(config: DrosophilaFetchConfig) -> neuprint.Client:
    """Create an authenticated neuPrint client for ``config.dataset``."""
    token = read_token(config.token_env_var)
    try:
        return neuprint.Client(config.server, dataset=config.dataset, token=token, progress=False)
    except Exception as exc:  # noqa: BLE001 - re-raise with the token scrubbed from the message
        raise RuntimeError(
            f"Could not connect to {config.server} dataset {config.dataset}: "
            f"{type(exc).__name__}: {_scrub(str(exc), token)}"
        ) from None


def _scrub(text: str, secret: str) -> str:
    """Remove ``secret`` from ``text`` so it can never leak through an error message."""
    return text.replace(secret, "***") if secret else text


# ---------------------------------------------------------------------------
# Selecting neurons and edges
# ---------------------------------------------------------------------------


def select_roi_top_neurons(client: Any, roi: str, num_nodes: int) -> pd.DataFrame:
    """Select the ``num_nodes`` neurons with the most synapses inside ``roi``.

    Returns:
        Node table sorted by rank with columns ``rank, bodyId, roi_pre, roi_post,
        roi_synapses`` plus any available annotation columns (type, instance, ...).
        ``roi_pre`` = output synapses (T-bars) of the neuron inside the ROI,
        ``roi_post`` = input synapses (PSDs) of the neuron inside the ROI.
    """
    available = set(client.all_rois)
    if roi not in available:
        raise ValueError(
            f"ROI {roi!r} does not exist in this dataset. Primary ROIs: {client.primary_rois}"
        )

    neurons_df, roi_counts_df = neuprint.fetch_neurons(neuprint.NeuronCriteria(rois=[roi]), client=client)
    counts = roi_counts_df.loc[roi_counts_df["roi"] == roi, ["bodyId", "pre", "post"]]
    counts = counts.rename(columns={"pre": "roi_pre", "post": "roi_post"}).astype("int64")
    counts["roi_synapses"] = counts["roi_pre"] + counts["roi_post"]

    if len(counts) < num_nodes:
        raise ValueError(f"ROI {roi!r} has only {len(counts)} neurons; {num_nodes} were requested")

    # Deterministic ranking: most synapses first, then smaller bodyId first.
    counts = counts.sort_values(["roi_synapses", "bodyId"], ascending=[False, True], kind="mergesort")
    selected = counts.head(num_nodes).reset_index(drop=True)
    selected.insert(0, "rank", np.arange(1, num_nodes + 1))

    annotations = [c for c in ANNOTATION_COLUMNS if c in neurons_df.columns]
    return selected.merge(neurons_df[["bodyId", *annotations]], on="bodyId", how="left", validate="one_to_one")


def fetch_edges_among(client: Any, body_ids: list[int], min_total_weight: int) -> pd.DataFrame:
    """Fetch every connection whose source and target are both in ``body_ids``.

    ``weight`` is the total number of synapses from source to target, summed over
    the whole nervous system (``omit_rois=True`` gives one row per neuron pair).

    Returns:
        Canonical edge table ``source, target, weight`` sorted by source, target.
    """
    _, conn_df = neuprint.fetch_adjacencies(
        sources=body_ids,
        targets=body_ids,
        min_total_weight=min_total_weight,
        omit_rois=True,
        weight_props=["weight"],
        properties=["type", "instance"],
        client=client,
    )
    edges = conn_df.rename(columns={"bodyId_pre": "source", "bodyId_post": "target"})
    edges = edges[["source", "target", "weight"]].astype("int64")

    outside = ~(edges["source"].isin(body_ids) & edges["target"].isin(body_ids))
    if outside.any():
        raise RuntimeError(f"neuPrint returned {int(outside.sum())} edge(s) outside the selected neurons")
    return edges.sort_values(["source", "target"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# End-to-end: fetch, save, validate
# ---------------------------------------------------------------------------


def output_paths(output_dir: Path, num_nodes: int, prefix: str = "drosophila") -> dict[str, Path]:
    """File names for one subgraph, e.g. drosophila_128.csv / _nodes.csv / _metadata.json."""
    stem = f"{prefix}_{num_nodes}"
    return {
        "edges": output_dir / f"{stem}.csv",
        "nodes": output_dir / f"{stem}_nodes.csv",
        "metadata": output_dir / f"{stem}_metadata.json",
    }


def fetch_malecns_subgraph(
    config: DrosophilaFetchConfig, client: Any, output_dir: Path, overwrite: bool = False
) -> tuple[ConnectomeGraph, dict[str, Any]]:
    """Select neurons, fetch their connections, save CSV + metadata, validate.

    Args:
        config: Validated fetch configuration.
        client: A connected ``neuprint.Client`` (or a test double).
        output_dir: Folder to write the files to.
        overwrite: Replace existing files. Off by default so a dataset used in
            earlier experiments is never silently replaced.

    Returns:
        The validated graph (loaded back with the Phase 2 loader) and the metadata dict.
    """
    paths = output_paths(output_dir, config.num_nodes)
    _check_can_write(paths, overwrite)
    logger = get_logger()

    logger.info("Selecting top %d neurons by synapse count in ROI %s (%s)",
                config.num_nodes, config.roi, config.dataset)
    nodes = select_roi_top_neurons(client, config.roi, config.num_nodes)
    body_ids = nodes["bodyId"].astype("int64").tolist()

    logger.info("Fetching connections among the %d selected neurons", len(body_ids))
    edges = fetch_edges_among(client, body_ids, config.min_total_weight)
    if edges.empty:
        raise RuntimeError("No connections found among the selected neurons")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_edge_csv(edges, paths["edges"])
    nodes.to_csv(paths["nodes"], index=False, lineterminator="\n")

    graph = load_edge_list(paths["edges"], self_loops=config.self_loops)
    connected = set(edges["source"]) | set(edges["target"])
    isolated = [b for b in body_ids if b not in connected]

    metadata = {
        "label": (
            f"GENERIC ROI-based subgraph: top {config.num_nodes} neurons by synapse count in ROI "
            f"'{config.roi}'. Not a functional circuit and not a visual-system selection."
        ),
        "is_visual_system_selection": False,
        "source": {
            "mode": "neuprint",
            "server": config.server,
            "dataset": config.dataset,
            "neuprint_python_version": importlib_metadata.version("neuprint-python"),
            "database": _database_info(client),
        },
        "reference": MALECNS_REFERENCE,
        "extraction_date_utc": utc_now_iso(),
        "selection": {
            "method": config.selection_method,
            "roi": config.roi,
            "num_nodes_requested": config.num_nodes,
            "candidates": f"all :Neuron bodies with synapses in ROI '{config.roi}' "
                          f"(neuprint.NeuronCriteria(rois=['{config.roi}']))",
            "ranking": "roi_pre + roi_post inside the ROI, descending; ties by bodyId ascending",
            "roi_synapses_of_last_selected": int(nodes["roi_synapses"].iloc[-1]),
        },
        "edges": {
            "query": "neuprint.fetch_adjacencies(sources=selected, targets=selected, omit_rois=True)",
            "weight_definition": "total synapse count from source to target across the whole dataset",
            "min_total_weight": config.min_total_weight,
            "direction": "source = presynaptic neuron, target = postsynaptic neuron",
        },
        "counts": {
            "neurons_selected": len(body_ids),
            "neurons_in_edge_list": len(connected),
            "isolated_selected_neurons": isolated,
            "edges_in_csv": len(edges),
            "self_loops_in_csv": int((edges["source"] == edges["target"]).sum()),
        },
        "validation": {
            "self_loop_policy": config.self_loops,
            "report": asdict(graph.report),
        },
        "files": _file_info(paths),
        "code": {"git_commit": git_commit()},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return graph, metadata


def import_local_edge_list(
    source_csv: Path, output_dir: Path, self_loops: str, overwrite: bool = False
) -> tuple[ConnectomeGraph, dict[str, Any]]:
    """Validate a local edge-list CSV and save it as ``drosophila_<N>.csv`` with metadata.

    The selection method of a user-provided file is unknown, so it is labelled
    as such; nothing about its biological meaning is assumed.
    """
    graph = load_edge_list(source_csv, self_loops=self_loops)
    paths = output_paths(output_dir, graph.num_nodes)
    _check_can_write(paths, overwrite)
    paths.pop("nodes")  # no neuron annotations are available for a local file

    edges = pd.DataFrame({
        "source": graph.node_ids[graph.edges[:, 0]],
        "target": graph.node_ids[graph.edges[:, 1]],
        "weight": graph.weights,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_edge_csv(edges, paths["edges"])

    metadata = {
        "label": "User-provided local edge list; selection method unknown (not verified as any circuit).",
        "is_visual_system_selection": False,
        "source": {"mode": "local_csv", "path": str(source_csv.resolve()), "sha256": sha256_file(source_csv)},
        "extraction_date_utc": utc_now_iso(),
        "counts": {"nodes": graph.num_nodes, "edges": graph.num_edges},
        "validation": {"self_loop_policy": self_loops, "report": asdict(graph.report)},
        "files": _file_info(paths),
        "code": {"git_commit": git_commit()},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return graph, metadata


def write_edge_csv(edges: pd.DataFrame, path: Path) -> None:
    """Write ``source,target,weight`` with LF line endings; integer weights stay integers."""
    out = edges[["source", "target", "weight"]].copy()
    if len(out) and np.all(np.mod(out["weight"].to_numpy(dtype=float), 1) == 0):
        out["weight"] = out["weight"].astype("int64")
    out.to_csv(path, index=False, lineterminator="\n")


def _check_can_write(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file(s): {existing}. Use --overwrite.")


def _file_info(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {
        kind: {"name": p.name, "sha256": sha256_file(p)}
        for kind, p in paths.items()
        if kind != "metadata" and p.exists()
    }


def _database_info(client: Any) -> dict[str, Any]:
    """Record identifying fields of the server database, when available."""
    try:
        meta = client.meta
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return {}
    return {k: meta[k] for k in _META_KEYS if k in meta}
