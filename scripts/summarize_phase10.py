"""Pre-registered Phase 10 summary (written and tested before any Phase 10 result existed).

Usage:
    python scripts/summarize_phase10.py --batch results/phase10_<timestamp>

Outputs:
    <batch>/summary/: summary.md, run_level_metrics.csv, run_level_summary.csv,
        fold_level_metrics.csv, fold_level_summary.csv, paired_drosophila_rewired.csv,
        paired_drosophila_ws.csv, paired_drosophila_cnn.csv, paired_summary.csv,
        paired_fold_means.csv, oof_metrics.csv, oof_summary.csv, run_index.csv,
        checksums.json, fig1..fig9 PNGs
    <batch>/oof/<model>/oof_predictions_seed{s}.csv (patient level) and
        oof_slice_predictions_seed{s}.csv (slice level)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from flyconnectome_mri.datasets.mri_dataset import read_manifest
from flyconnectome_mri.evaluation import cv_plots
from flyconnectome_mri.evaluation.comparison import to_markdown
from flyconnectome_mri.evaluation.cv_summary import (
    DISPLAY,
    MODELS,
    PRIMARY,
    SEEDS,
    build_oof,
    collect_cv_runs,
    fold_level_summary,
    fold_means,
    oof_metrics,
    oof_summary,
    paired,
    paired_fold_means,
    paired_summary,
    run_level_summary,
)
from flyconnectome_mri.utils.provenance import PROJECT_ROOT, sha256_file

COHORT_MANIFEST = "data/splits/figshare_glioma_meningioma/manifest.csv"
TABLE_METRICS = ["patient_roc_auc", "slice_roc_auc", "patient_balanced_accuracy", "slice_balanced_accuracy",
                 "patient_sensitivity", "patient_specificity", "patient_f1", "patient_precision"]
RUN_TABLE_COLUMNS = ["model", "outer_fold", "seed", "graph_seed", "status", "best_epoch", "patient_roc_auc",
                     "slice_roc_auc", "patient_balanced_accuracy", "slice_balanced_accuracy", "patient_sensitivity",
                     "patient_specificity", "patient_f1", "patient_precision", "patient_recall", "slice_sensitivity",
                     "slice_specificity", "slice_f1", "slice_precision", "training_seconds", "wall_seconds",
                     "inference_ms_per_slice", "train_pos_weight", "parameters", "checkpoint_sha256", "run_dir"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise the Phase 10 batch (pre-registered).")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--cohort-manifest", default=COHORT_MANIFEST,
                        help="Manifest listing every eligible patient (each must be tested exactly once).")
    args = parser.parse_args(argv)
    batch = Path(args.batch)
    out, oof_dir = batch / "summary", batch / "oof"
    out.mkdir(exist_ok=True)
    cohort_path = Path(args.cohort_manifest)
    cohort_path = cohort_path if cohort_path.is_absolute() else PROJECT_ROOT / cohort_path
    cohort = set(read_manifest(cohort_path)["patient_id"])

    index = pd.read_csv(batch / "runs_index.csv", dtype={"run_dir": str})
    shutil.copy(batch / "runs_index.csv", out / "run_index.csv")
    runs = collect_cv_runs(index)
    runs[[c for c in RUN_TABLE_COLUMNS if c in runs.columns]].to_csv(out / "run_level_metrics.csv", index=False)
    run_level_summary(runs).to_csv(out / "run_level_summary.csv", index=False)
    folds = fold_means(runs)
    folds.to_csv(out / "fold_level_metrics.csv", index=False)
    fold_summary = fold_level_summary(folds)
    fold_summary.to_csv(out / "fold_level_summary.csv", index=False)

    diffs = {name: paired(runs, "visual_drosophila", other)
             for name, other in (("rewired", "visual_rewired"), ("ws", "visual_ws"), ("cnn", "cnn_baseline"))}
    for name, diff in diffs.items():
        diff.to_csv(out / f"paired_drosophila_{name}.csv", index=False)
    pd.DataFrame([paired_summary(d, m) for d in diffs.values() for m in TABLE_METRICS]).to_csv(
        out / "paired_summary.csv", index=False)
    pd.concat([paired_fold_means(d).assign(comparison=d["comparison"].iloc[0]) for d in diffs.values()]).to_csv(
        out / "paired_fold_means.csv", index=False)

    oof_rows = []
    for model in MODELS:
        (oof_dir / model).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            patients, slices, status = build_oof(runs, model, seed, cohort)
            if patients is not None:
                patients.to_csv(oof_dir / model / f"oof_predictions_seed{seed}.csv", index=False)
                slices.to_csv(oof_dir / model / f"oof_slice_predictions_seed{seed}.csv", index=False)
            oof_rows.append({"model": model, "seed": seed, "status": status,
                             "patients": len(patients) if patients is not None else 0, **oof_metrics(patients)})
    oof_table = pd.DataFrame(oof_rows)
    oof_table.to_csv(out / "oof_metrics.csv", index=False)
    oof_sum = oof_summary(oof_table)
    oof_sum.to_csv(out / "oof_summary.csv", index=False)

    cv_plots.metric_by_fold(runs, PRIMARY, "Test patient ROC AUC by outer fold", out / "fig1_patient_auc_by_fold.png")
    cv_plots.fold_means_by_model(folds, PRIMARY, "Fold-level mean patient ROC AUC", out / "fig2_fold_level_patient_auc.png")
    cv_plots.oof_by_seed(oof_table, "roc_auc", "Pooled OOF patient ROC AUC", out / "fig3_oof_patient_auc.png")
    cv_plots.paired_by_fold(diffs["rewired"], "Visual Drosophila − Visual rewired, patient ROC AUC, by outer fold",
                            out / "fig4_paired_drosophila_rewired_by_fold.png")
    cv_plots.paired_all_runs(diffs["rewired"], "Visual Drosophila − Visual rewired, patient ROC AUC",
                             out / "fig5_paired_drosophila_rewired_all_runs.png")
    cv_plots.metric_by_fold(runs, "patient_balanced_accuracy", "Test patient balanced accuracy by outer fold",
                            out / "fig6_balanced_accuracy_by_fold.png")
    cv_plots.runs_by_model(runs, [("patient_sensitivity", "patient sensitivity"),
                                  ("patient_specificity", "patient specificity")],
                           "Sensitivity and specificity", out / "fig7_sensitivity_specificity.png")
    cv_plots.runs_by_model(runs, [("best_epoch", "best validation epoch"), ("val_roc_auc_best", "validation slice ROC AUC at best epoch")],
                           "Training stability", out / "fig8_training_stability.png")
    cv_plots.runs_by_model(runs, [("training_seconds", "training time (s)"), ("inference_ms_per_slice", "inference (ms per slice)")],
                           "Runtime", out / "fig9_runtime.png")

    write_markdown(runs, folds, fold_summary, diffs, oof_table, oof_sum, out / "summary.md")
    checksums = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "checksums.json"}
    checksums.update({f"oof/{p.parent.name}/{p.name}": sha256_file(p) for p in sorted(oof_dir.rglob("*.csv"))})
    (out / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print((out / "summary.md").read_text(encoding="utf-8"))


def _fmt(mean: float, sd: float) -> str:
    return "n/a" if pd.isna(mean) else (f"{mean:.3f} ± {sd:.3f}" if not pd.isna(sd) else f"{mean:.3f}")


def write_markdown(runs, folds, fold_summary, diffs, oof_table, oof_sum, path: Path) -> None:
    lines = ["# Phase 10 summary: leakage-safe cross-validation using the official Figshare outer test folds", "",
             "Pre-registered analysis. PRIMARY metric: patient-level ROC AUC. Fold-level summary (seeds averaged "
             "within each fold first) is the main robustness summary; pooled OOF patient ROC AUC is the main "
             "practical robustness metric. Run-level values are 25 fold×seed runs per model, NOT 25 independent "
             "experiments. No significance or clinical claims; possible site confound applies. "
             "Visual Drosophila vs CNN is an architectural comparison, not a pure topology comparison.", ""]
    status = runs.groupby(["model", "status"]).size().unstack(fill_value=0)
    lines += ["## Run status (all 100 planned runs)", "", to_markdown(status, index=True), ""]

    lines += ["## Pooled out-of-fold (OOF) patient metrics: mean ± SD across seeds", "",
              "| Model | seeds ok | " + " | ".join(m.replace("oof_patient_", "") for m in oof_sum["metric"].unique()) + " |",
              "|" + "---|" * (2 + oof_sum["metric"].nunique())]
    for model in MODELS:
        part = oof_sum[oof_sum["model"] == model]
        lines.append(f"| {DISPLAY[model]} | {part['seeds_ok'].iloc[0]} | "
                     + " | ".join(_fmt(r["mean"], r["sd"]) for _, r in part.iterrows()) + " |")
    lines += ["", to_markdown(oof_table[["model", "seed", "status", "patients", "roc_auc", "balanced_accuracy",
                                         "sensitivity", "specificity", "f1", "precision", "recall"]], ".3f"), ""]

    lines += ["## Fold-level summary (seeds averaged within fold; then across the 5 folds)", ""]
    for metric in TABLE_METRICS[:4]:
        view = fold_summary[fold_summary["metric"] == metric][["model", "n", "mean", "sd", "median", "min", "max"]]
        lines += [f"### {metric}", "", to_markdown(view, ".3f"), ""]
    pivot = folds.pivot_table(index="model", columns="outer_fold", values="patient_roc_auc")
    lines += ["### patient_roc_auc per fold (mean over seeds)", "", to_markdown(pivot, ".3f", index=True), ""]

    lines += ["## Paired comparisons (matched by outer fold + seed)", ""]
    for name, diff in diffs.items():
        label = "PRIMARY topology comparison" if name == "rewired" else (
            "architectural comparison, NOT a pure topology comparison" if name == "cnn" else "secondary comparison")
        summary = paired_summary(diff)
        lines += [f"### {diff['comparison'].iloc[0]} ({label})", "",
                  f"patient ROC AUC: mean {summary['mean']:+.3f} ± {summary['sd']:.3f} over {summary['pairs']} pairs "
                  f"({summary['pairs_na']} NA); positive {summary['positive']}, negative {summary['negative']}, "
                  f"tied {summary['tied']}; fold-level mean {summary['fold_level_mean']:+.3f} ± "
                  f"{summary['fold_level_sd']:.3f}", "",
                  to_markdown(paired_fold_means(diff), "+.3f"), "",
                  to_markdown(diff[["outer_fold", "seed", "both_ok", PRIMARY, "slice_roc_auc",
                                    "patient_balanced_accuracy"]], "+.3f"), ""]

    lines += ["## Run level (all 100 runs)", "",
              to_markdown(runs[["model", "outer_fold", "seed", "graph_seed", "status", "best_epoch", *TABLE_METRICS,
                                "training_seconds"]], ".3f"), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
