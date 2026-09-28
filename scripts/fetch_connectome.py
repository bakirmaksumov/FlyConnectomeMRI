"""Fetch a small Drosophila subgraph from neuPrint, or import a local edge list.

Usage:
    # Mode 1: query neuPrint (needs the NEUPRINT_TOKEN environment variable)
    python scripts/fetch_connectome.py --config configs/malecns_fetch.yaml
    python scripts/fetch_connectome.py --config configs/malecns_fetch.yaml --num-nodes 256

    # Mode 2: validate and import a local source,target,weight CSV
    python scripts/fetch_connectome.py --config configs/malecns_fetch.yaml --from-csv my_edges.csv

Output (in data/connectome/ by default):
    drosophila_<N>.csv            canonical edge list: source,target,weight
    drosophila_<N>_nodes.csv      selected neurons with rank and annotations (neuPrint mode)
    drosophila_<N>_metadata.json  provenance: dataset, date, selection criteria, counts, checksums
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flyconnectome_mri.connectome.drosophila import connect, fetch_malecns_subgraph, import_local_edge_list
from flyconnectome_mri.connectome.statistics import compute_graph_statistics, format_statistics
from flyconnectome_mri.utils.config import load_fetch_config
from flyconnectome_mri.utils.logging import setup_logging
from flyconnectome_mri.utils.provenance import PROJECT_ROOT

NUM_EDGES_TO_SHOW = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or import a Drosophila connectome subgraph.")
    parser.add_argument("--config", required=True, help="Fetch config YAML (e.g. configs/malecns_fetch.yaml).")
    parser.add_argument("--num-nodes", type=int, default=None, help="Override num_nodes from the config.")
    parser.add_argument("--from-csv", default=None, help="Import this local edge list instead of querying neuPrint.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from the config.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = setup_logging()
    config = load_fetch_config(args.config, num_nodes=args.num_nodes)

    output_dir = Path(args.output_dir or config.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if args.from_csv:
        graph, metadata = import_local_edge_list(Path(args.from_csv), output_dir, config.self_loops, args.overwrite)
    else:
        client = connect(config)
        graph, metadata = fetch_malecns_subgraph(config, client, output_dir, args.overwrite)

    logger.info("Label: %s", metadata["label"])
    for kind, info in metadata["files"].items():
        logger.info("Saved %-5s -> %s (sha256 %s...)", kind, output_dir / info["name"], info["sha256"][:12])

    print("\nGraph statistics (validated with the Phase 2 loader):")
    print(format_statistics(compute_graph_statistics(graph)))

    print(f"\nFirst {NUM_EDGES_TO_SHOW} edges (source -> target : weight):")
    for (s, t), w in list(zip(graph.edges, graph.weights))[:NUM_EDGES_TO_SHOW]:
        print(f"  {graph.node_ids[s]} -> {graph.node_ids[t]} : {w:g}")


if __name__ == "__main__":
    main()
