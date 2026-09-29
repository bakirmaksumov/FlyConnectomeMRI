"""Phase 10 aggregation (pre-registered before any Phase 10 result existed).

Levels of summary, never mixed up:
    run level   one row per (model, outer fold, seed); 25 per model. These are
                NOT 25 independent experiments.
    fold level  per model and outer fold: mean over the seeds that ran
                successfully; then mean / SD / median / min / max across the 5
                folds. This is the main robustness summary.
    OOF         per model and seed: the five outer-test folds are concatenated
                into one full-cohort patient table (each patient exactly once);
                metrics are computed on it; then mean ± SD across seeds. Seeds
                are never pooled together.

Failed runs are always kept (status != "ok", metrics NA). A seed is never
replaced. Paired comparisons match runs strictly by (outer fold, seed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flyconnectome_mri.training.metrics import binary_metrics

MODELS = ["cnn_baseline", "visual_drosophila", "visual_rewired", "visual_ws"]
DISPLAY = {"cnn_baseline": "CNN baseline", "visual_drosophila": "Visual Drosophila",
           "visual_rewired": "Visual rewired", "visual_ws": "Visual WS"}
FOLDS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2, 3, 4]
METRIC_NAMES = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "precision", "recall"]
RUN_METRICS = [f"patient_{m}" for m in METRIC_NAMES] + [f"slice_{m}" for m in METRIC_NAMES]
PRIMARY = "patient_roc_auc"


def id_format(patient_id: str) -> str:
    """Patient-ID format, kept for the later (not yet run) numeric-ID subgroup analysis."""
    return "MR-prefixed" if str(patient_id).startswith("MR") else "numeric"


# ---------------------------------------------------------------------------
# Run level
# ---------------------------------------------------------------------------


def collect_cv_runs(index: pd.DataFrame) -> pd.DataFrame:
    """One row per PLANNED (model, fold, seed); missing or failed runs keep NA metrics."""
    rows = []
    by_key = {(r.model, int(r.fold), int(r.seed)): r for r in index.itertuples()}
    for model in MODELS:
        for fold in FOLDS:
            for seed in SEEDS:
                row: dict[str, Any] = {"model": model, "outer_fold": fold, "seed": seed,
                                       "graph_seed": seed if model in ("visual_rewired", "visual_ws") else None,
                                       "status": "not_run", **{m: np.nan for m in RUN_METRICS}}
                entry = by_key.get((model, fold, seed))
                if entry is not None:
                    row["wall_seconds"] = float(entry.wall_seconds)
                    row["run_dir"] = entry.run_dir if isinstance(entry.run_dir, str) else ""
                    row.update(_read_metrics(row["run_dir"], int(entry.returncode)))
                rows.append(row)
    return pd.DataFrame(rows)


def _read_metrics(run_dir: str, returncode: int) -> dict[str, Any]:
    path = Path(run_dir) / "metrics.json" if run_dir else None
    if path is None or not path.exists():
        return {"status": f"failed_exit_{returncode}" if returncode else "failed_no_metrics"}
    m = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "status": m.get("status", "ok") if returncode == 0 else f"failed_exit_{returncode}",
        "best_epoch": m["training"].get("best_epoch"),
        "epochs_run": m["training"].get("epochs_run"),
        "training_seconds": m["training"].get("training_seconds"),
        "parameters": m.get("parameters"),
        "train_pos_weight": (m.get("class_weighting") or {}).get("pos_weight"),
        "checkpoint_sha256": m.get("checkpoint_sha256"),
    }
    if out["status"] == "ok" and m.get("test"):
        out["inference_ms_per_slice"] = m["inference"]["forward_ms_per_slice"]
        out["val_roc_auc_best"] = m["training"].get("best_val_score")
        for level in ("patient", "slice"):
            for name in METRIC_NAMES:
                out[f"{level}_{name}"] = m["test"][level][name]
            for cell, value in m["test"][level]["confusion_matrix"].items():
                out[f"{level}_{cell}"] = value
    elif out["status"] == "ok":
        out["status"] = "failed_no_test"
    return out


def describe(values: pd.Series) -> dict[str, float]:
    v = values.astype(float).dropna()
    return {"n": len(v), "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
            "median": v.median(), "min": v.min(), "max": v.max()}


def run_level_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Per model: descriptive statistics over its 25 fold x seed runs (failed runs counted, not used)."""
    rows = []
    for model in MODELS:
        part = runs[runs["model"] == model]
        for metric in RUN_METRICS:
            rows.append({"model": model, "metric": metric, "runs_ok": int((part["status"] == "ok").sum()),
                         "runs_not_ok": int((part["status"] != "ok").sum()), **describe(part[metric])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fold level
# ---------------------------------------------------------------------------


def fold_means(runs: pd.DataFrame) -> pd.DataFrame:
    """Per model and outer fold: mean over successful seeds (seeds are averaged FIRST)."""
    ok = runs[runs["status"] == "ok"]
    table = ok.groupby(["model", "outer_fold"])[RUN_METRICS].mean().reset_index()
    counts = runs.assign(ok=runs["status"] == "ok").groupby(["model", "outer_fold"])["ok"].sum().reset_index()
    table = counts.rename(columns={"ok": "seeds_ok"}).merge(table, on=["model", "outer_fold"], how="left")
    return table


def fold_level_summary(fold_table: pd.DataFrame) -> pd.DataFrame:
    """Across the 5 fold means of each model: mean, SD, median, min, max."""
    rows = []
    for model in MODELS:
        part = fold_table[fold_table["model"] == model]
        for metric in RUN_METRICS:
            rows.append({"model": model, "metric": metric, **describe(part[metric])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired comparisons (strictly by outer fold + seed)
# ---------------------------------------------------------------------------


def paired(runs: pd.DataFrame, reference: str, other: str, metrics: list[str] = RUN_METRICS) -> pd.DataFrame:
    """All 25 (fold, seed) pairs: reference - other. NA if either run is not ok."""
    key = ["outer_fold", "seed"]
    ref = runs[runs["model"] == reference].set_index(key)
    oth = runs[runs["model"] == other].set_index(key)
    both_ok = (ref["status"] == "ok") & (oth["status"] == "ok")
    diff = (ref[metrics] - oth[metrics]).where(both_ok)
    diff.insert(0, "both_ok", both_ok)
    diff.insert(0, "comparison", f"{reference} - {other}")
    return diff.reset_index()


def paired_fold_means(diff: pd.DataFrame, metric: str = PRIMARY) -> pd.DataFrame:
    """Five fold-level paired differences: mean over seeds within each fold."""
    return diff.groupby("outer_fold").agg(mean_difference=(metric, "mean"),
                                          pairs_used=(metric, lambda s: int(s.notna().sum()))).reset_index()


def paired_summary(diff: pd.DataFrame, metric: str = PRIMARY) -> dict[str, Any]:
    values = diff[metric].dropna()
    folds = paired_fold_means(diff, metric)["mean_difference"].dropna()
    return {"comparison": diff["comparison"].iloc[0], "metric": metric, "pairs": len(values),
            "pairs_na": int(diff[metric].isna().sum()),
            "mean": values.mean(), "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
            "positive": int((values > 0).sum()), "negative": int((values < 0).sum()), "tied": int((values == 0).sum()),
            "fold_level_mean": folds.mean(), "fold_level_sd": folds.std(ddof=1) if len(folds) > 1 else np.nan}


# ---------------------------------------------------------------------------
# Pooled out-of-fold (OOF) predictions
# ---------------------------------------------------------------------------


def build_oof(runs: pd.DataFrame, model: str, seed: int, cohort: set[str], threshold: float = 0.5
              ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """Concatenate the five outer-test folds of one model + seed.

    Returns:
        (patient table, slice table, status). status is "ok" only if all five
        fold runs succeeded AND every cohort patient appears exactly once.
    """
    parts = runs[(runs["model"] == model) & (runs["seed"] == seed)].sort_values("outer_fold")
    if (parts["status"] != "ok").any() or len(parts) != len(FOLDS):
        return None, None, "incomplete_failed_fold"
    patients, slices = [], []
    for row in parts.itertuples():
        p = pd.read_csv(Path(row.run_dir) / "test_predictions_patient.csv", dtype={"patient_id": str})
        s = pd.read_csv(Path(row.run_dir) / "test_predictions_slice.csv", dtype={"patient_id": str})
        patients.append(p.assign(outer_fold=row.outer_fold))
        slices.append(s.assign(outer_fold=row.outer_fold, model=model, seed=seed))
    pat = pd.concat(patients, ignore_index=True)
    counts = pat["patient_id"].value_counts()
    if (counts > 1).any():
        raise ValueError(f"{model} seed {seed}: patient(s) tested in more than one fold: {list(counts[counts > 1].index[:5])}")
    missing = cohort - set(pat["patient_id"])
    if missing or set(pat["patient_id"]) - cohort:
        raise ValueError(f"{model} seed {seed}: OOF does not match the cohort (missing {sorted(missing)[:5]})")
    link = pat["patient_link_key"] if "patient_link_key" in pat.columns else pat["patient_id"]
    oof = pd.DataFrame({
        "patient_id": pat["patient_id"], "patient_link_key": link, "id_format": pat["patient_id"].map(id_format),
        "true_label": pat["label_index"].astype(int), "label": pat["label"], "outer_fold": pat["outer_fold"],
        "n_slices": pat["n_slices"], "mean_probability": pat["prob"],
        "predicted_label": (pat["prob"] >= threshold).astype(int), "model": model, "seed": seed,
    }).sort_values("patient_id").reset_index(drop=True)
    return oof, pd.concat(slices, ignore_index=True), "ok"


def oof_metrics(oof: pd.DataFrame | None, threshold: float = 0.5) -> dict[str, float]:
    if oof is None:
        return {name: np.nan for name in METRIC_NAMES}
    m = binary_metrics(oof["true_label"].to_numpy(), oof["mean_probability"].to_numpy(), threshold)
    return {name: m[name] for name in METRIC_NAMES} | {f"cm_{k}": v for k, v in m["confusion_matrix"].items()}


def oof_summary(oof_table: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD of each OOF metric across seeds (one value per seed, never pooled over seeds)."""
    rows = []
    for model in MODELS:
        part = oof_table[oof_table["model"] == model]
        for name in METRIC_NAMES:
            rows.append({"model": model, "metric": f"oof_patient_{name}",
                         "seeds_ok": int((part["status"] == "ok").sum()), **describe(part[name])})
    return pd.DataFrame(rows)
