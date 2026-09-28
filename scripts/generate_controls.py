"""Generate random control graphs (ER / WS / BA / degree-preserving rewired).

Usage:
    python scripts/generate_controls.py --config configs/controls.yaml

Prints the biological graph and the seed-0 controls side by side, plus how much
each control's edges overlap with the biological edges, across all seeds.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from flyconnectome_mri.connectome.controls import control_paths, generate_controls
from flyconnectome_mri.connectome.loader import load_edge_list, load_node_ids
from flyconnectome_mri.connectome.statistics import format_comparison
from flyconnectome_mri.utils.config import load_controls_config
from flyconnectome_mri.utils.logging import setup_logging
from flyconnectome_mri.utils.provenance import PROJECT_ROOT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate random control graphs.")
    parser.add_argument("--config", required=True, help="Controls YAML (e.g. configs/controls.yaml).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing control files.")
    args = parser.parse_args(argv)

    logger = setup_logging()
    config = load_controls_config(args.config)
    graphs = generate_controls(config, PROJECT_ROOT, overwrite=args.overwrite)
    logger.info("Generated %d control graphs in %s", len(graphs), config.output_dir)

    bio_nodes = load_node_ids(PROJECT_ROOT / config.source_nodes, column=config.source_node_column)
    bio = load_edge_list(PROJECT_ROOT / config.source_edges, node_ids=bio_nodes)
    first_seed = config.seeds[0]
    columns = {f"drosophila {config.label}": bio}
    columns.update({f"{c} seed{first_seed}": graphs[(c, first_seed)] for c in config.controls})
    print(format_comparison(columns))

    output_dir = PROJECT_ROOT / config.output_dir
    print(f"\nEdge overlap with the biological graph across seeds {config.seeds} "
          "(fraction of control edges that are also biological edges):")
    for control in config.controls:
        overlaps = []
        extra = ""
        for seed in config.seeds:
            meta = json.loads(control_paths(output_dir, control, config.label, seed)["metadata"]
                              .read_text(encoding="utf-8"))
            overlaps.append(meta["edge_overlap_with_biological_graph"])
            if control == "rewired" and seed == first_seed:
                d = meta["generator_details"]
                extra = f"  (seed {seed}: {d['accepted_swaps']}/{d['attempted_swaps']} swaps accepted)"
        print(f"  {control:<8} mean {np.mean(overlaps):.3f}  min {np.min(overlaps):.3f}  "
              f"max {np.max(overlaps):.3f}{extra}")


if __name__ == "__main__":
    main()
