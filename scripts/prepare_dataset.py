"""Scan an image dataset, split it leak-free, and write manifest.csv + dataset_metadata.json.

Usage:
    python scripts/prepare_dataset.py --config configs/datasets/synthetic_demo.yaml
"""

from __future__ import annotations

import argparse

from flyconnectome_mri.datasets.prepare import prepare_dataset
from flyconnectome_mri.utils.config import load_dataset_config
from flyconnectome_mri.utils.logging import setup_logging
from flyconnectome_mri.utils.provenance import PROJECT_ROOT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare a leak-free train/val/test split.")
    parser.add_argument("--config", required=True, help="Dataset YAML (e.g. configs/datasets/synthetic_demo.yaml).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing manifest/metadata.")
    args = parser.parse_args(argv)

    logger = setup_logging()
    config = load_dataset_config(args.config)
    manifest, meta = prepare_dataset(config, PROJECT_ROOT, overwrite=args.overwrite)

    counts = meta["counts"]
    print(f"\nDataset       : {meta['dataset']['name']}  (source: {meta['dataset']['source']}, "
          f"license: {meta['dataset']['license']})")
    print(f"Images        : {counts['images']}   classes: {counts['class_counts']}")
    print(f"Patients      : {counts['patients'] if counts['patients'] is not None else 'UNAVAILABLE'}"
          f"   groups: {counts['groups']}   images with duplicate content: "
          f"{counts['images_with_exact_duplicate_content']}")
    print(f"Split seed    : {meta['split']['seed']}   fractions: {meta['split']['fractions']}")
    print(f"\n{'split':<6} {'images':>7} {'patients':>9} {'groups':>7}   class counts")
    for split, info in meta["split"]["per_split"].items():
        patients = info["patients"] if info["patients"] is not None else "n/a"
        print(f"{split:<6} {info['images']:>7} {patients:>9} {info['groups']:>7}   {info['class_counts']}")
    print(f"\nLeakage check : passed (no patient_id, group or identical file shared between splits)")
    print(f"Normalization : mean {meta['preprocessing']['normalize_mean']} "
          f"std {meta['preprocessing']['normalize_std']} (train split only)")
    for limitation in meta["limitations"]:
        logger.warning("LIMITATION: %s", limitation)


if __name__ == "__main__":
    main()
