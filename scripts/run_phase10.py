"""Phase 10: leakage-safe cross-validation using the official Figshare outer test folds.

4 models x 5 outer folds x 5 seeds = 100 runs (protocol: docs/RESEARCH_PLAN.md, Phase 10).

Usage:
    python scripts/run_phase10.py --dry-run          # verify folds + write/print the 100 planned runs; no training
    python scripts/run_phase10.py                    # train (only after the pre-run report is approved)
    python scripts/run_phase10.py --resume results/phase10_<timestamp>

Before any training the five fold datasets are re-checked: zero patient_id and
zero patient_link_key overlap between train/val/test, and every patient in
exactly one outer test fold. Any violation stops the batch immediately.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from flyconnectome_mri.datasets.cv_folds import N_FOLDS, check_oof_coverage
from flyconnectome_mri.datasets.mri_dataset import MANIFEST_NAME, read_manifest
from flyconnectome_mri.datasets.splits import check_no_leakage
from flyconnectome_mri.evaluation.batch import Job, new_batch_dir, run_jobs
from flyconnectome_mri.utils.config import load_config
from flyconnectome_mri.utils.provenance import PROJECT_ROOT

MODELS = {
    "cnn_baseline": "configs/cnn_baseline.yaml",
    "visual_drosophila": "configs/visual_drosophila.yaml",
    "visual_rewired": "configs/visual_rewired.yaml",
    "visual_ws": "configs/visual_ws.yaml",
}
FOLDS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2, 3, 4]
SOURCE_MANIFEST = "data/splits/figshare_glioma_meningioma/manifest.csv"
FOLD_DIR = "data/splits/figshare_cv_leakage_safe/fold{fold}"


def planned_jobs() -> list[Job]:
    """Order: fold, then seed, then model (a partial batch stays paired within fold x seed)."""
    return [Job(model, seed, MODELS[model], fold, FOLD_DIR.format(fold=fold))
            for fold in FOLDS for seed in SEEDS for model in MODELS]


def verify_folds() -> dict[str, int]:
    """Fail immediately on any patient_id / patient_link_key overlap or incomplete OOF coverage."""
    folds = {}
    for fold in FOLDS:
        manifest = read_manifest(PROJECT_ROOT / FOLD_DIR.format(fold=fold) / MANIFEST_NAME)
        check_no_leakage(manifest)  # raises LeakageError on any overlap
        tested = set(manifest.loc[manifest["split"] == "test", "patient_id"])
        cv_folds = set(manifest.loc[manifest["split"] == "test", "cv_fold"])
        if cv_folds != {fold}:
            raise RuntimeError(f"fold {fold}: test set is not exactly official fold {fold}")
        folds[fold] = manifest
        assert tested
    return check_oof_coverage(read_manifest(PROJECT_ROOT / SOURCE_MANIFEST), folds)


def plan_rows(jobs: list[Job]) -> list[dict]:
    rows = []
    for number, job in enumerate(jobs, start=1):
        config = load_config(PROJECT_ROOT / job.config, seed=job.seed, prepared_dir=job.prepared_dir)
        rows.append({"run": number, "model": job.model, "fold": job.fold, "seed": job.seed,
                     "graph_file": config.graph.edges_file or "-", "prepared_dir": job.prepared_dir,
                     "config": job.config})
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 10: leakage-safe CV (100 runs).")
    parser.add_argument("--dry-run", action="store_true", help="Verify folds and list the planned runs only.")
    parser.add_argument("--resume", default=None, help="Existing batch folder: start only runs never started.")
    args = parser.parse_args(argv)

    coverage = verify_folds()
    print(f"Fold verification passed: {coverage}")
    jobs = planned_jobs()
    rows = plan_rows(jobs)
    if args.dry_run:
        out = PROJECT_ROOT / "data" / "splits" / "figshare_cv_leakage_safe" / "planned_runs.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)} planned runs written to {out} (dry run: nothing trained)")
        return
    batch = Path(args.resume) if args.resume else new_batch_dir("phase10")
    run_jobs(batch, jobs)


if __name__ == "__main__":
    assert len(FOLDS) == N_FOLDS
    main()
