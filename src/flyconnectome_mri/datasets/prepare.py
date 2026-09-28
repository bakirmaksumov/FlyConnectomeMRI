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
import re
from pathlib import Path
from typing import Any

import pandas as pd

from flyconnectome_mri.datasets.figshare import (
    load_cv_folds,
    map_folds_to_patients,
    patients_with_multiple_labels,
    scan_figshare,
)
from flyconnectome_mri.datasets.manifest import (
    SUPPORTED_EXTENSIONS,
    DatasetError,
    ScanResult,
    assign_patient_ids,
    scan_image_folder,
)
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
# Written after MANIFEST_COLUMNS when present (format-specific provenance).
OPTIONAL_COLUMNS = ["patient_link_key", "source_filename", "image_number", "original_label_code", "cv_fold"]


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

    format_info: dict[str, Any] = {"format": config.format}
    if config.format == "figshare_mat":
        records, exclusions = scan_figshare(root, config.figshare_label_codes, config.classes)
        scan = ScanResult(records=records)
        manifest = scan.records
        mixed = patients_with_multiple_labels(manifest)
        if mixed:
            logger.warning("%d patient(s) have slices with different labels: %s", len(mixed), mixed[:10])
        format_info.update(label_codes=config.figshare_label_codes, source_counts=exclusions,
                           patients_with_multiple_labels=mixed)
        if config.cv_indices_file is not None:
            cv_path = Path(config.cv_indices_file)
            cv_path = cv_path if cv_path.is_absolute() else project_root / cv_path
            folds = load_cv_folds(cv_path, exclusions["all_slices"])
            manifest["cv_fold"], split_patients = map_folds_to_patients(manifest, folds)
            format_info["official_cv_folds"] = {
                "file": cv_path.name,
                "sha256": sha256_file(cv_path),
                "maps_cleanly_to_patients": not split_patients,
                "patients_split_across_folds": split_patients,
                "slices_per_fold": {int(k): int(v) for k, v in manifest["cv_fold"].value_counts().sort_index().items()},
                "patients_per_fold": {int(k): int(v) for k, v in
                                      manifest.groupby("cv_fold")["patient_id"].nunique().sort_index().items()},
                "note": "stored for later experiments; the train/val/test split below does not use them",
            }
    else:
        scan = scan_image_folder(root, config.classes, config.on_invalid_image)
        manifest = scan.records
        manifest["patient_id"] = assign_patient_ids(manifest, config, root)
    has_patient_ids = config.patient_id_source != "none"
    linked: dict[str, list[str]] = {}
    if config.patient_link_regex is not None:
        manifest["patient_link_key"], linked = link_patient_ids(manifest["patient_id"], config.patient_link_regex)
        logger.info("patient_link_regex links %d patient ID(s) into %d shared key(s)",
                    sum(len(v) for v in linked.values()), len(linked))
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
    manifest = manifest[MANIFEST_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in manifest.columns]]

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
        "raw_data_provenance": _read_provenance(config.provenance_file, project_root),
        "format": format_info,
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
            "link_regex": config.patient_link_regex,
            "linked_ids": linked,
            "distinct_link_keys": checked.get("patient_link_key"),
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


def _read_provenance(provenance_file: str | None, project_root: Path) -> dict[str, Any] | None:
    """Load an optional provenance JSON (e.g. download_provenance.json) to embed in the metadata."""
    if provenance_file is None:
        return None
    path = Path(provenance_file)
    path = path if path.is_absolute() else project_root / path
    if not path.is_file():
        raise DatasetError(f"provenance_file not found: {path}")
    return {"file": path.name, "sha256": sha256_file(path), "content": json.loads(path.read_text(encoding="utf-8"))}


def link_patient_ids(patient_ids: pd.Series, link_regex: str) -> tuple[pd.Series, dict[str, list[str]]]:
    """Map each patient ID to its link key (the regex ``base`` if it fully matches, else itself).

    Returns:
        (link key per row, {key: [original IDs]} for keys shared by 2+ distinct IDs).
    """
    pattern = re.compile(link_regex)

    def key(pid: str | None) -> str | None:
        if pid is None:
            return None
        match = pattern.fullmatch(pid)
        return match.group("base") if match else pid

    keys = patient_ids.map(key)
    members = pd.DataFrame({"key": keys, "pid": patient_ids}).dropna().drop_duplicates()
    shared = members.groupby("key")["pid"].agg(sorted)
    return keys, {k: v for k, v in shared.items() if len(v) > 1}
