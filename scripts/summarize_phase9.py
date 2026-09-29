"""Summarise the Phase 9 visual-graph batch (tables + figures).

Usage:
    python scripts/summarize_phase9.py --batch results/phase9_<timestamp>

The CNN baseline is NOT retrained: its frozen Phase 8 results
(docs/results/phase8/runs.csv) are shown as a reference. EB vs visual is a
descriptive side-by-side only.

Outputs in <batch>/summary/: runs.csv, summary_by_model.csv, paired_differences.csv,
paired_summary.csv, eb_vs_visual.csv, summary.md and fig1..fig6 PNGs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from flyconnectome_mri.evaluation.comparison import (
    DISPLAY_NAMES,
    MODEL_ORDER,
    collect_runs,
    format_mean_sd,
    metric_column,
    paired_differences,
    summarize,
    summarize_differences,
    to_markdown,
)
from flyconnectome_mri.evaluation.plots import (
    plot_graph_set_comparison,
    plot_mean_sd,
    plot_metric_by_seed,
    plot_paired_differences,
    plot_training_stability,
)
from flyconnectome_mri.utils.provenance import PROJECT_ROOT

PHASE8_RUNS = PROJECT_ROOT / "docs" / "results" / "phase8" / "runs.csv"
REFERENCE = "graph_visual_drosophila"
PRIMARY_NULL = "graph_visual_rewired"
SECONDARY_NULLS = ["graph_visual_er", "graph_visual_ws", "graph_visual_ba"]
# PRIMARY metric first (patient ROC AUC), then the declared secondary metrics.
METRICS = [("patient", "roc_auc"), ("slice", "roc_auc"), ("patient", "balanced_accuracy"),
           ("slice", "balanced_accuracy"), ("patient", "sensitivity"), ("patient", "specificity"),
           ("patient", "f1"), ("slice", "f1"), ("patient", "precision"), ("patient", "recall"),
           ("slice", "precision"), ("slice", "recall")]
FAMILIES = [("Drosophila", "graph_drosophila", "graph_visual_drosophila"),
            ("Rewired", "graph_rewired", "graph_visual_rewired"),
            ("ER", "graph_er", "graph_visual_er"), ("WS", "graph_ws", "graph_visual_ws"),
            ("BA", "graph_ba", "graph_visual_ba")]


def with_order(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    runs["model"] = pd.Categorical(runs["model"].astype(str),
                                   categories=[m for m in MODEL_ORDER if m in set(runs["model"].astype(str))],
                                   ordered=True)
    return runs.sort_values(["model", "seed"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise the Phase 9 batch.")
    parser.add_argument("--batch", required=True)
    args = parser.parse_args(argv)
    batch = Path(args.batch)
    out = batch / "summary"
    out.mkdir(exist_ok=True)

    with open(batch / "runs_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    visual = collect_runs([Path(row["run_dir"]) for row in index if row["run_dir"]])
    crashed = [(r["model"], r["seed"]) for r in index
               if not r["run_dir"] or not (Path(r["run_dir"]) / "metrics.json").exists()]
    phase8 = pd.read_csv(PHASE8_RUNS)
    cnn_ref = phase8[phase8["model"] == "cnn_baseline"]
    eb = with_order(phase8[phase8["model"] != "cnn_baseline"])

    visual.to_csv(out / "runs.csv", index=False)
    combined = with_order(pd.concat([cnn_ref, visual.astype({"model": str})], ignore_index=True))
    summary = summarize(combined, METRICS)
    summary.to_csv(out / "summary_by_model.csv", index=False)

    diffs = [paired_differences(visual, REFERENCE, other, METRICS) for other in [PRIMARY_NULL, *SECONDARY_NULLS]]
    pd.concat(diffs).to_csv(out / "paired_differences.csv", index=False)
    diff_summary = pd.concat([summarize_differences(d, METRICS) for d in diffs if len(d)])
    diff_summary.to_csv(out / "paired_summary.csv", index=False)

    eb_summary = summarize(eb, METRICS).set_index("model")
    vis_summary = summarize(with_order(visual.astype({"model": str})), METRICS).set_index("model")
    rows = []
    for label, eb_model, vis_model in FAMILIES:
        for level, metric in METRICS[:4]:
            key = f"{level}_{metric}"
            rows.append({"family": label, "metric": key,
                         "eb_w20_mean": eb_summary.loc[eb_model, f"{key}_mean"],
                         "eb_w20_sd": eb_summary.loc[eb_model, f"{key}_sd"],
                         "visual_raw_mean": vis_summary.loc[vis_model, f"{key}_mean"],
                         "visual_raw_sd": vis_summary.loc[vis_model, f"{key}_sd"]})
    eb_vs_visual = pd.DataFrame(rows)
    eb_vs_visual.to_csv(out / "eb_vs_visual.csv", index=False)

    plot_metric_by_seed(combined, "patient_roc_auc", "Test patient-level ROC AUC (PRIMARY; CNN = Phase 8 reference)",
                        out / "fig1_patient_auc_by_seed.png")
    plot_metric_by_seed(combined, "slice_roc_auc", "Test slice-level ROC AUC (CNN = Phase 8 reference)",
                        out / "fig2_slice_auc_by_seed.png")
    plot_mean_sd(summary, METRICS[:6] + METRICS[6:8], out / "fig3_mean_sd.png")
    plot_paired_differences(diffs[0], METRICS[:8], "Visual Drosophila − visual degree-preserving rewired (paired by seed)",
                            out / "fig4_paired_visual_drosophila_minus_rewired.png")
    plot_training_stability(with_order(visual.astype({"model": str})), out / "fig5_training_stability.png")
    plot_graph_set_comparison(eb_summary, vis_summary, FAMILIES, [("patient", "roc_auc"), ("slice", "roc_auc")],
                              out / "fig6_eb_vs_visual_descriptive.png")

    write_markdown(visual, summary, diffs, diff_summary, eb_vs_visual, crashed, out / "summary.md")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print((out / "summary.md").read_text(encoding="utf-8"))


def write_markdown(visual, summary, diffs, diff_summary, eb_vs_visual, crashed, path: Path) -> None:
    lines = ["# Phase 9 summary: visual-system graph", "",
             "Descriptive only: 5 training seeds, one fixed 21-patient test split. No significance or clinical "
             "claims. PRIMARY metric: patient-level ROC AUC. The CNN baseline was NOT retrained (frozen Phase 8 "
             "reference). EB vs visual is descriptive, not a controlled comparison.", ""]
    status = visual.groupby(["model", "status"], observed=True).size().unstack(fill_value=0)
    lines += ["## Run status (Phase 9 runs)", "", to_markdown(status, index=True), ""]
    if crashed:
        lines += [f"Runs without metrics (crashed): {crashed}", ""]
    lines += ["## Mean ± SD over seeds (successful runs)", ""]
    header = "| Model | runs | " + " | ".join(f"{lvl} {m.replace('_', ' ')}" for lvl, m in METRICS) + " |"
    lines += [header, "|" + "---|" * (len(METRICS) + 2)]
    for _, row in summary.iterrows():
        name = row["display_name"] + (" (Phase 8 ref.)" if row["model"] == "cnn_baseline" else "")
        cells = [format_mean_sd(row[f"{lvl}_{m}_mean"], row[f"{lvl}_{m}_sd"]) for lvl, m in METRICS]
        lines.append(f"| {name} | {row['runs_ok']} | " + " | ".join(cells) + " |")
    cols = ["model", "seed", "graph_seed", "status", "best_epoch", "val_roc_auc_best", "patient_roc_auc",
            "slice_roc_auc", "patient_balanced_accuracy", "slice_balanced_accuracy", "patient_sensitivity",
            "patient_specificity", "diag_graph_layer2_dead_nodes_frac", "diag_logit_std"]
    view = visual[[c for c in cols if c in visual.columns]].copy()
    view["model"] = view["model"].map(lambda m: DISPLAY_NAMES.get(m, m))
    lines += ["", "## Individual seeds (Phase 9 runs)", "", to_markdown(view, ".3f"), ""]
    lines += ["## Paired differences: Visual Drosophila − control (per seed)", ""]
    for diff in diffs:
        if len(diff):
            view = diff[["seed"] + [metric_column(lvl, m) for lvl, m in METRICS[:8]]]
            lines += [f"### {diff['comparison'].iloc[0]}", "", to_markdown(view, "+.3f"), ""]
    lines += ["## Summary of paired differences (descriptive; no p-values)", "",
              to_markdown(diff_summary, "+.3f"), "",
              "## EB-128 w20 (Phase 8) vs Visual-128 raw (Phase 9): descriptive only", "",
              to_markdown(eb_vs_visual, ".3f"), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
