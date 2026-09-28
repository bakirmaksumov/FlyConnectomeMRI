"""Derive weight-threshold variants (raw / w5 / w10 / w20) from the raw downloaded graph.

Usage:
    python scripts/derive_graph_variants.py --config configs/graph_variants.yaml
"""

from __future__ import annotations

import argparse

from flyconnectome_mri.connectome.filtering import derive_weight_variants
from flyconnectome_mri.connectome.statistics import format_comparison
from flyconnectome_mri.utils.config import load_graph_variants_config
from flyconnectome_mri.utils.logging import setup_logging
from flyconnectome_mri.utils.provenance import PROJECT_ROOT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Derive weight-threshold graph variants.")
    parser.add_argument("--config", required=True, help="Variants YAML (e.g. configs/graph_variants.yaml).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing derived files.")
    args = parser.parse_args(argv)

    logger = setup_logging()
    config = load_graph_variants_config(args.config)
    results = derive_weight_variants(config, PROJECT_ROOT, overwrite=args.overwrite)

    for variant, (_, meta) in results.items():
        logger.info("%-4s %-28s -> %s (%d edges kept, %d isolated nodes)", variant, meta["role"],
                    meta["files"]["edges"]["name"], meta["counts"]["edges_kept"],
                    len(meta["counts"]["isolated_nodes"]))
    columns = {
        f"{v} (w>={config.variants[v]}){' *' if v == config.primary_variant else ''}": graph
        for v, (graph, _) in results.items()
    }
    print(format_comparison(columns))
    print(f"\n* primary graph, pre-declared on {config.primary_declared_on} before any MRI training.")


if __name__ == "__main__":
    main()
