"""Collect many training runs and compare models across seeds.

Statistics are deliberately descriptive: mean, sample standard deviation
(ddof=1), every individual seed, and paired per-seed differences. With only a
handful of training seeds no p-values or significance claims are produced.

Failed runs (status != "ok") are kept in the run table and counted, but are
excluded from means and from the paired differences of their seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Display order and names (order = colour slot order in the plots).
MODEL_ORDER = ["cnn_baseline", "graph_drosophila", "graph_rewired", "graph_er", "graph_ws", "graph_ba",
               "graph_visual_drosophila", "graph_visual_rewired", "graph_visual_er", "graph_visual_ws",
               "graph_visual_ba"]
DISPLAY_NAMES = {
    "cnn_baseline": "CNN baseline",
    "graph_drosophila": "Drosophila w20",
    "graph_rewired": "Rewired (deg.-pres.)",
    "graph_er": "ER",
    "graph_ws": "WS",
    "graph_ba": "BA",
    "graph_visual_drosophila": "Visual Drosophila",
    "graph_visual_rewired": "Visual rewired (deg.-pres.)",
    "graph_visual_er": "Visual ER",
    "graph_visual_ws": "Visual WS",
    "graph_visual_ba": "Visual BA",
}
SUMMARY_METRICS = [
    ("slice", "roc_auc"), ("patient", "roc_auc"),
    ("slice", "balanced_accuracy"), ("patient", "balanced_accuracy"),
    ("slice", "sensitivity"), ("patient", "sensitivity"),
    ("slice", "specificity"), ("patient", "specificity"),
    ("slice", "f1"), ("patient", "f1"),
]
PRIMARY_REFERENCE = "graph_drosophila"
PRIMARY_NULL = "graph_rewired"
SECONDARY_NULLS = ["graph_er", "graph_ws", "graph_ba"]


def metric_column(level: str, metric: str) -> str:
    return f"{level}_{metric}"


def collect_runs(run_dirs: list[Path]) -> pd.DataFrame:
    """One row per run with everything needed for the Phase 8 tables."""
    rows = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "model": m["model"],
            "seed": m["seed"],
            "graph_seed": m.get("graph_seed"),
            "status": m.get("status", "ok"),
            "parameters": m.get("parameters"),
            "best_epoch": m["training"]["best_epoch"],
            "epochs_run": m["training"]["epochs_run"],
            "training_seconds": m["training"]["training_seconds"],
            "val_roc_auc_best": m["training"].get("best_val_score"),
            "run_dir": str(run_dir),
        }
        diagnostics = m.get("validation_diagnostics") or {}
        for key in ("logit_std", "graph_layer1_std", "graph_layer2_std",
                    "graph_layer1_dead_nodes_frac", "graph_layer2_dead_nodes_frac"):
            row[f"diag_{key}"] = diagnostics.get(key)
        if m.get("test") is not None:
            row["inference_ms_per_slice"] = m["inference"]["forward_ms_per_slice"]
            row["val_slice_roc_auc"] = m["validation"]["slice"]["roc_auc"]
            row["val_patient_roc_auc"] = m["validation"]["patient"]["roc_auc"]
            for level in ("slice", "patient"):
                test = m["test"][level]
                for metric in ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1",
                               "precision", "recall", "accuracy"):
                    row[metric_column(level, metric)] = test[metric]
                for cell, value in test["confusion_matrix"].items():
                    row[f"{level}_{cell}"] = value
        rows.append(row)
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs
    runs["model"] = pd.Categorical(runs["model"], categories=[m for m in MODEL_ORDER if m in set(runs["model"])],
                                   ordered=True)
    return runs.sort_values(["model", "seed"]).reset_index(drop=True)


def summarize(runs: pd.DataFrame, metrics: list[tuple[str, str]] = SUMMARY_METRICS) -> pd.DataFrame:
    """Mean and sample SD (ddof=1) per model over successful runs, plus run counts."""
    ok = runs[runs["status"] == "ok"]
    rows = []
    for model in runs["model"].cat.categories:
        part = ok[ok["model"] == model]
        row: dict[str, Any] = {
            "model": model,
            "display_name": DISPLAY_NAMES.get(model, model),
            "runs_ok": len(part),
            "runs_failed": int(((runs["model"] == model) & (runs["status"] != "ok")).sum()),
            "parameters": int(part["parameters"].iloc[0]) if len(part) else None,
        }
        for level, metric in metrics:
            values = part[metric_column(level, metric)].astype(float)
            row[f"{level}_{metric}_mean"] = values.mean()
            row[f"{level}_{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else float("nan")
        row["best_epoch_mean"] = part["best_epoch"].mean()
        row["training_seconds_mean"] = part["training_seconds"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(runs: pd.DataFrame, reference: str, other: str,
                       metrics: list[tuple[str, str]] = SUMMARY_METRICS) -> pd.DataFrame:
    """Per-seed ``reference - other`` for every metric (seeds where both runs are ok)."""
    ok = runs[runs["status"] == "ok"]
    columns = [metric_column(level, metric) for level, metric in metrics]
    ref = ok[ok["model"] == reference].set_index("seed")[columns]
    oth = ok[ok["model"] == other].set_index("seed")[columns]
    seeds = sorted(set(ref.index) & set(oth.index))
    diff = (ref.loc[seeds] - oth.loc[seeds]).reset_index()
    diff.insert(0, "comparison", f"{reference} - {other}")
    return diff


def summarize_differences(diff: pd.DataFrame, metrics: list[tuple[str, str]] = SUMMARY_METRICS) -> pd.DataFrame:
    """Mean, SD and sign counts of paired differences (descriptive only; no p-values)."""
    rows = []
    for level, metric in metrics:
        col = metric_column(level, metric)
        values = diff[col].astype(float).dropna()
        rows.append({
            "comparison": diff["comparison"].iloc[0] if len(diff) else None,
            "metric": col,
            "n_seeds": len(values),
            "mean_difference": values.mean(),
            "sd_difference": values.std(ddof=1) if len(values) > 1 else float("nan"),
            "seeds_reference_higher": int((values > 0).sum()),
            "seeds_reference_lower": int((values < 0).sum()),
            "seeds_equal": int((values == 0).sum()),
        })
    return pd.DataFrame(rows)


def format_mean_sd(mean: float, sd: float) -> str:
    if mean is None or np.isnan(mean):
        return "n/a"
    return f"{mean:.3f} ± {sd:.3f}" if not np.isnan(sd) else f"{mean:.3f}"


def to_markdown(table: pd.DataFrame, floatfmt: str = ".3f", index: bool = False) -> str:
    """Minimal Markdown table (avoids the optional 'tabulate' dependency)."""
    frame = table.reset_index() if index else table

    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "n/a" if np.isnan(value) else format(value, floatfmt)
        return str(value)

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])
