"""Load an edge-list CSV, validate it, and print graph statistics.

Usage:
    python scripts/inspect_connectome.py --edges tests/data/tiny_graph.csv
    python scripts/inspect_connectome.py --edges data/connectome/x.csv --self-loops drop --json stats.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from flyconnectome_mri.connectome.loader import load_edge_list
from flyconnectome_mri.connectome.statistics import compute_graph_statistics, format_statistics
from flyconnectome_mri.utils.logging import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an edge list and print graph statistics.")
    parser.add_argument("--edges", required=True, help="CSV file with columns source,target,weight.")
    parser.add_argument("--self-loops", choices=["error", "drop", "keep"], default="error",
                        help="How to handle source == target edges (default: error).")
    parser.add_argument("--duplicates", choices=["error", "sum"], default="error",
                        help="How to handle repeated (source, target) pairs (default: error).")
    parser.add_argument("--json", default=None, help="Optionally save the statistics to this JSON file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = setup_logging()
    connectome = load_edge_list(args.edges, self_loops=args.self_loops, duplicates=args.duplicates)
    stats = compute_graph_statistics(connectome)

    logger.info("Loaded %s", Path(args.edges).resolve())
    logger.info("Validation report: %s", connectome.report)
    print(format_statistics(stats))

    if args.json:
        payload = {"edges_file": str(Path(args.edges).resolve()), "report": asdict(connectome.report),
                   "statistics": stats.to_dict()}
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Statistics saved to %s", args.json)


if __name__ == "__main__":
    main()
