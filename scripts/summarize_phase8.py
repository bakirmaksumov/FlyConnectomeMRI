"""Summarise a Phase 8 batch: tables (CSV + Markdown) and figures.

Usage:
    python scripts/summarize_phase8.py --batch results/phase8_<timestamp>

Outputs in <batch>/summary/:
    runs.csv                   one row per run (all runs, including failed ones)
    summary_by_model.csv       mean and SD per model (successful runs)
    paired_differences.csv     per-seed Drosophila - null for every null model
    paired_summary.csv         mean/SD/sign counts of those differences (no p-values)
    summary.md                 human-readable report
    fig1_patient_auc_by_seed.png, fig2_slice_auc_by_seed.png, fig3_mean_sd.png,
    fig4_paired_drosophila_minus_rewired.png, fig5_training_stability.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from flyconnectome_mri.evaluation.comparison import (
    DISPLAY_NAMES,
    PRIMARY_NULL,
    PRIMARY_REFERENCE,
    SECONDARY_NULLS,
    SUMMARY_METRICS,
    collect_runs,
    format_mean_sd,
    metric_column,
    paired_differences,
    summarize,
    summarize_differences,
    to_markdown,
)
from flyconnectome_mri.evaluation.plots import (
    plot_mean_sd,
    plot_metric_by_seed,
    plot_paired_differences,
    plot_training_stability,
)

TABLE_METRICS = [("slice", "roc_auc"), ("patient", "roc_auc"), ("slice", "balanced_accuracy"),
                 ("patient", "balanced_accuracy"), ("slice", "sensitivity"), ("patient", "sensitivity"),
                 ("slice", "specificity"), ("patient", "specificity"), ("slice", "f1"), ("patient", "f1")]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise a Phase 8 batch.")
    parser.add_argument("--batch", required=True)
    args = parser.parse_args(argv)
    batch = Path(args.batch)
    out = batch / "summary"
    out.mkdir(exist_ok=True)

    with open(batch / "runs_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    runs = collect_runs([Path(row["run_dir"]) for row in index if row["run_dir"]])
    # Runs that crashed before writing metrics.json are still reported.
    crashed = [row for row in index if not row["run_dir"] or not (Path(row["run_dir"]) / "metrics.json").exists()]
    runs.to_csv(out / "runs.csv", index=False)

    summary = summarize(runs)
    summary.to_csv(out / "summary_by_model.csv", index=False)
    diffs = [paired_differences(runs, PRIMARY_REFERENCE, other) for other in [PRIMARY_NULL, *SECONDARY_NULLS]]
    pd.concat(diffs).to_csv(out / "paired_differences.csv", index=False)
    diff_summary = pd.concat([summarize_differences(d) for d in diffs if len(d)])
    diff_summary.to_csv(out / "paired_summary.csv", index=False)

    plot_metric_by_seed(runs, "patient_roc_auc", "Test patient-level ROC AUC", out / "fig1_patient_auc_by_seed.png")
    plot_metric_by_seed(runs, "slice_roc_auc", "Test slice-level ROC AUC", out / "fig2_slice_auc_by_seed.png")
    plot_mean_sd(summary, TABLE_METRICS[:6] + TABLE_METRICS[8:], out / "fig3_mean_sd.png")
    plot_paired_differences(diffs[0], TABLE_METRICS, "Drosophila w20 − degree-preserving rewired (paired by seed)",
                            out / "fig4_paired_drosophila_minus_rewired.png")
    plot_training_stability(runs, out / "fig5_training_stability.png")

    write_markdown(runs, summary, diffs, diff_summary, crashed, out / "summary.md")
    # Windows consoles may use a legacy code page (e.g. cp1251) without "±" or "−".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print((out / "summary.md").read_text(encoding="utf-8"))


def write_markdown(runs, summary, diffs, diff_summary, crashed, path: Path) -> None:
    lines = ["# Phase 8 summary", "",
             "Descriptive only: 5 training seeds, one fixed test split of 21 patients. No significance claims, "
             "no clinical claims. CNN baseline vs graph models is not a pure topology comparison "
             "(41,921 vs 48,527 parameters).", ""]
    status = runs.groupby(["model", "status"], observed=True).size().unstack(fill_value=0)
    lines += ["## Run status", "", to_markdown(status, index=True), ""]
    if crashed:
        lines += [f"Runs without metrics (crashed): {[(r['model'], r['seed']) for r in crashed]}", ""]

    lines += ["## Mean ± SD over seeds (successful runs)", ""]
    header = "| Model | runs | " + " | ".join(f"{lvl} {m.replace('_', ' ')}" for lvl, m in TABLE_METRICS) + " |"
    lines += [header, "|" + "---|" * (len(TABLE_METRICS) + 2)]
    for _, row in summary.iterrows():
        cells = [format_mean_sd(row[f"{lvl}_{m}_mean"], row[f"{lvl}_{m}_sd"]) for lvl, m in TABLE_METRICS]
        lines.append(f"| {row['display_name']} | {row['runs_ok']} | " + " | ".join(cells) + " |")

    lines += ["", "## Individual seeds", ""]
    cols = ["model", "seed", "graph_seed", "status", "best_epoch", "val_roc_auc_best", "slice_roc_auc",
            "patient_roc_auc", "slice_balanced_accuracy", "patient_balanced_accuracy", "slice_sensitivity",
            "slice_specificity", "patient_sensitivity", "patient_specificity"]
    view = runs[[c for c in cols if c in runs.columns]].copy()
    view["model"] = view["model"].map(lambda m: DISPLAY_NAMES.get(m, m))
    lines += [to_markdown(view, ".3f"), ""]

    lines += ["## Paired differences: Drosophila w20 − null (per seed)", ""]
    for diff in diffs:
        if not len(diff):
            continue
        view = diff[["seed"] + [metric_column(lvl, m) for lvl, m in TABLE_METRICS]]
        lines += [f"### {diff['comparison'].iloc[0]}", "", to_markdown(view, "+.3f"), ""]
    lines += ["## Summary of paired differences (descriptive; no p-values)", "",
              to_markdown(diff_summary, "+.3f"), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
