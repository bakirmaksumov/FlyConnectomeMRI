"""Prepare the 5 leakage-safe CV fold datasets (official Figshare outer test folds) + pre-run report.

Usage:
    python scripts/prepare_cv_folds.py
    python scripts/prepare_cv_folds.py --source data/splits/figshare_glioma_meningioma \\
        --output data/splits/figshare_cv_leakage_safe

For each outer fold k writes <output>/fold{k}/:
    manifest.csv            retained slices; split = train/val/test; cv_fold unchanged
    dataset_metadata.json   copy of the source metadata with the fold split, counts,
                            exclusions and TRAIN-fold-only normalisation statistics
    exclusions.csv          removed linked-family patients (reason linked_to_test / linked_to_validation)
and <output>/cv_prerun_report.json + counts.csv + exclusions_all.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from flyconnectome_mri.datasets.cv_folds import N_FOLDS, build_outer_fold, check_oof_coverage, validation_fold
from flyconnectome_mri.datasets.mri_dataset import MANIFEST_NAME, METADATA_NAME, read_manifest
from flyconnectome_mri.datasets.splits import split_summary
from flyconnectome_mri.datasets.transforms import compute_channel_stats
from flyconnectome_mri.training.train import compute_pos_weight
from flyconnectome_mri.utils.logging import setup_logging
from flyconnectome_mri.utils.provenance import PROJECT_ROOT, git_commit, sha256_file, utc_now_iso

DESCRIPTION = "Leakage-safe cross-validation using the official Figshare outer test folds"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--source", default="data/splits/figshare_glioma_meningioma")
    parser.add_argument("--output", default="data/splits/figshare_cv_leakage_safe")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logger = setup_logging()

    source, output = resolve(args.source), resolve(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output} is not empty; use --overwrite")
    base_meta = json.loads((source / METADATA_NAME).read_text(encoding="utf-8"))
    manifest = read_manifest(source / MANIFEST_NAME)
    root = resolve(base_meta["dataset_root"])
    pre = base_meta["preprocessing"]

    folds, counts, exclusions, report_folds = {}, [], [], {}
    for k in range(1, N_FOLDS + 1):
        fold = build_outer_fold(manifest, k)
        folds[k] = fold.manifest
        counts.append(fold.counts)
        exclusions.append(fold.exclusions)
        fold_dir = output / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold.manifest.to_csv(fold_dir / MANIFEST_NAME, index=False, na_rep="NA", lineterminator="\n")
        fold.exclusions.to_csv(fold_dir / "exclusions.csv", index=False, lineterminator="\n")

        train = fold.manifest[fold.manifest["split"] == "train"]
        logger.info("fold %d: computing train-only normalisation from %d slices", k, len(train))
        mean, std = compute_channel_stats([root / p for p in train["image_path"]], pre["image_size"], pre["grayscale"])
        pos_weight = compute_pos_weight(train["label_index"].to_numpy())
        meta = dict(base_meta)
        meta["preprocessing"] = {**pre, "normalize_mean": mean, "normalize_std": std,
                                 "normalization_source": f"train split of outer fold {k} only"}
        meta["split"] = {
            "method": DESCRIPTION,
            "outer_test_fold": k,
            "validation_fold": validation_fold(k),
            "training_folds": [f for f in range(1, N_FOLDS + 1) if f not in (k, validation_fold(k))],
            "rule": "test = official fold k unchanged; val = official fold k%5+1 minus relatives of test; "
                    "train = other 3 folds minus relatives of test or cleaned val (patient_link_key)",
            "per_split": split_summary(fold.manifest),
            "excluded_patients": fold.exclusions["removed_patient_id"].tolist(),
            "source_manifest_sha256": sha256_file(source / MANIFEST_NAME),
        }
        meta["cv"] = {"outer_fold": k, "validation_fold": validation_fold(k), "train_pos_weight": pos_weight}
        meta["files"] = {"manifest": {"name": MANIFEST_NAME, "sha256": sha256_file(fold_dir / MANIFEST_NAME)}}
        meta["created_utc"] = utc_now_iso()
        meta["code"] = {"git_commit": git_commit()}
        (fold_dir / METADATA_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        report_folds[k] = {"validation_fold": validation_fold(k), "train_pos_weight": pos_weight,
                           "normalize_mean": mean, "normalize_std": std,
                           "manifest_sha256": meta["files"]["manifest"]["sha256"],
                           "overlaps": _overlaps(fold.manifest)}

    coverage = check_oof_coverage(manifest, folds)
    counts_df = pd.concat(counts, ignore_index=True)
    exclusions_df = pd.concat(exclusions, ignore_index=True)
    counts_df.to_csv(output / "counts.csv", index=False, lineterminator="\n")
    exclusions_df.to_csv(output / "exclusions_all.csv", index=False, lineterminator="\n")
    # The official cv_fold of every retained image must equal its value in the source manifest.
    official = manifest.set_index("image_path")["cv_fold"]
    unchanged = all(bool((f.set_index("image_path")["cv_fold"] == official.loc[f["image_path"]].to_numpy()).all())
                    for f in folds.values())
    report = {"description": DESCRIPTION, "source_manifest_sha256": sha256_file(source / MANIFEST_NAME),
              "official_cv_fold_values_unchanged": unchanged, "oof_coverage": coverage, "folds": report_folds,
              "created_utc": utc_now_iso(), "git_commit": git_commit()}
    (output / "cv_prerun_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"oof_coverage": coverage, "cv_fold_unchanged": unchanged}, indent=2))


def _overlaps(fold_manifest: pd.DataFrame) -> dict[str, int]:
    """Size of every pairwise intersection (must all be 0)."""
    result = {}
    for column in ("patient_id", "patient_link_key"):
        sets = {s: set(fold_manifest.loc[fold_manifest["split"] == s, column]) for s in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            result[f"{column}:{a}&{b}"] = len(sets[a] & sets[b])
    return result


if __name__ == "__main__":
    main()
