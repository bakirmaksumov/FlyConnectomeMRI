"""Prepare a dataset: scan -> patient IDs -> groups -> split -> leakage check -> save.

Outputs (in ``config.output_dir``):
    manifest.csv           one row per image: image_path, label, label_index,
                           patient_id ("NA" = unavailable), group_id, split,
                           sha256, width, height, mode
    dataset_metadata.json  provenance, counts, split sizes and class balance,
                           split seed, train-only normalization statistics,
                           limitations (e.g. missing patient IDs)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from flyconnectome_mri.datasets.manifest import SUPPORTED_EXTENSIONS, DatasetError, assign_patient_ids, scan_image_folder
from flyconnectome_mri.datasets.mri_dataset import MANIFEST_NAME, METADATA_NAME
from flyconnectome_mri.datasets.splits import SPLITS, build_groups, check_no_leakage, split_groups, split_summary
from flyconnectome_mri.datasets.transforms import compute_channel_stats
from flyconnectome_mri.utils.config import DatasetConfig
from flyconnectome_mri.utils.logging import get_logger
from flyconnectome_mri.utils.provenance import git_commit, sha256_file, utc_now_iso

NO_PATIENT_ID_WARNING = (
    "Patient IDs are unavailable: images are split individually (exact duplicates are still kept "
    "together). Different slices of the same patient may appear in different splits, so "
    "patient-level leakage CANNOT be ruled out and results may be optimistic."
)
MANIFEST_COLUMNS = ["image_path", "label", "label_index", "patient_id", "group_id", "split",
                    "sha256", "width", "height", "mode"]


def prepare_dataset(config: DatasetConfig, project_root: Path, overwrite: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build, check and save the manifest and metadata. Returns (manifest, metadata)."""
    logger = get_logger()
    root = Path(config.root)
    if not root.is_absolute():
        root = project_root / root
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    manifest_path, metadata_path = output_dir / MANIFEST_NAME, output_dir / METADATA_NAME
    if (manifest_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(f"{output_dir} already contains a prepared dataset. Use --overwrite.")

    scan = scan_image_folder(root, config.classes, config.on_invalid_image)
    manifest = scan.records
    manifest["patient_id"] = assign_patient_ids(manifest, config, root)
    has_patient_ids = config.patient_id_source != "none"
    if not has_patient_ids:
        logger.warning(NO_PATIENT_ID_WARNING)

    manifest["group_id"] = build_groups(manifest)
    manifest["split"] = split_groups(
        manifest, (config.train_fraction, config.val_fraction, config.test_fraction), config.split_seed
    )
    empty = [s for s in SPLITS if not (manifest["split"] == s).any()]
    if empty:
        raise DatasetError(f"Split(s) {empty} received no images: the dataset has too few groups for these fractions")
    checked = check_no_leakage(manifest)
    manifest = manifest[MANIFEST_COLUMNS]

    summary = split_summary(manifest)
    for split, info in summary.items():
        missing_classes = sorted(set(config.classes) - set(info["class_counts"]))
        if missing_classes:
            logger.warning("Split %r has no images of class(es) %s", split, missing_classes)

    train_paths = [root / p for p in manifest.loc[manifest["split"] == "train", "image_path"]]
    mean, std = compute_channel_stats(train_paths, config.image_size, config.grayscale)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, na_rep="NA", lineterminator="\n")

    duplicate_images = int(manifest["sha256"].duplicated(keep=False).sum())
    metadata: dict[str, Any] = {
        "dataset": {"name": config.name, "source": config.source, "license": config.license},
        "dataset_root": config.root,  # as configured; relative = relative to project root
        "created_utc": utc_now_iso(),
        "task": {
            "type": "binary classification",
            "classes": config.classes,
            "positive_class": config.positive_class,
        },
        "counts": {
            "images": len(manifest),
            "class_counts": {str(k): int(v) for k, v in manifest["label"].value_counts().sort_index().items()},
            "patients": checked["patient_id"] if has_patient_ids else None,
            "groups": checked["group_id"],
            "images_with_exact_duplicate_content": duplicate_images,
        },
        "patient_ids": {
            "available": has_patient_ids,
            "source": config.patient_id_source,
            "regex": config.patient_id_regex,
            "csv": config.patient_id_csv,
        },
        "split": {
            "seed": config.split_seed,
            "fractions": {"train": config.train_fraction, "val": config.val_fraction, "test": config.test_fraction},
            "method": "stratified by label; whole groups (same patient or identical content) assigned together",
            "per_split": summary,
            "leakage_check": {
                "passed": True,
                "no_shared": ["patient_id", "group_id", "sha256"],
                "distinct_values_checked": checked,
            },
        },
        "limitations": [] if has_patient_ids else [NO_PATIENT_ID_WARNING],
        "preprocessing": {
            "grayscale": config.grayscale,
            "image_size": config.image_size,
            "resize": "bilinear, antialias=True, to image_size x image_size (aspect ratio not preserved)",
            "normalize_mean": mean,
            "normalize_std": std,
            "normalization_source": "train split only",
            "augmentation": "none",
        },
        "skipped": {
            "supported_extensions": list(SUPPORTED_EXTENSIONS),
            "unsupported_files": scan.unsupported_files,
            "invalid_images": [{"image_path": p, "reason": r} for p, r in scan.invalid_images],
            "ignored_top_level_files": scan.ignored_top_level_files,
        },
        "files": {"manifest": {"name": MANIFEST_NAME, "sha256": sha256_file(manifest_path)}},
        "code": {"git_commit": git_commit()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return manifest, metadata
