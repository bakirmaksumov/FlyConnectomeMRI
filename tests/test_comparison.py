"""Tests for Phase 8 run collection, summaries, paired differences, tables and figures."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from flyconnectome_mri.evaluation.comparison import (
    collect_runs,
    paired_differences,
    summarize,
    summarize_differences,
    to_markdown,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fake_metrics(model: str, seed: int, auc: float, status: str = "ok") -> dict:
    level = {"roc_auc": auc, "balanced_accuracy": auc - 0.1, "sensitivity": 0.6, "specificity": 0.9, "f1": 0.7,
             "precision": 0.9, "recall": 0.6, "accuracy": 0.7,
             "confusion_matrix": {"tn": 9, "fp": 1, "fn": 4, "tp": 6}}
    metrics = {
        "model": model, "seed": seed, "graph_seed": seed if model not in ("cnn_baseline", "graph_drosophila") else None,
        "status": status, "parameters": 48527,
        "training": {"best_epoch": 5 + seed, "epochs_run": 12, "training_seconds": 100.0, "best_val_score": 0.9},
        "validation_diagnostics": {"logit_std": 2.0, "graph_layer2_dead_nodes_frac": 0.05},
        "test": None,
    }
    if status == "ok":
        metrics["test"] = {"slice": dict(level), "patient": dict(level, roc_auc=auc + 0.05)}
        metrics["validation"] = {"slice": {"roc_auc": 0.9}, "patient": {"roc_auc": 0.95}}
        metrics["inference"] = {"forward_ms_per_slice": 2.0}
    return metrics


@pytest.fixture
def batch(tmp_path: Path) -> Path:
    """Drosophila AUCs 0.80..0.84, rewired 0.78..0.82 except seed 2 failed, baseline 0.70."""
    runs = []
    for seed in range(5):
        for model, auc, status in (("cnn_baseline", 0.70, "ok"), ("graph_drosophila", 0.80 + 0.01 * seed, "ok"),
                                   ("graph_rewired", 0.78 + 0.01 * seed, "collapsed_on_validation" if seed == 2 else "ok")):
            run_dir = tmp_path / "runs" / model / f"seed{seed}"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(json.dumps(fake_metrics(model, seed, auc, status)), encoding="utf-8")
            runs.append((model, seed, run_dir))
    with open(tmp_path / "runs_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", "config", "returncode", "wall_seconds", "run_dir", "log", "started_utc"])
        for model, seed, run_dir in runs:
            writer.writerow([model, seed, "c.yaml", 0, 1.0, str(run_dir), "log", "t"])
    return tmp_path


def test_collect_runs_keeps_failed_runs(batch: Path) -> None:
    runs = collect_runs(sorted((batch / "runs").glob("*/*")))
    assert len(runs) == 15
    assert list(runs["model"].cat.categories) == ["cnn_baseline", "graph_drosophila", "graph_rewired"]
    failed = runs[runs["status"] != "ok"]
    assert failed[["model", "seed"]].values.tolist() == [["graph_rewired", 2]]
    assert np.isnan(failed["slice_roc_auc"].iloc[0])


def test_summary_mean_and_sample_sd(batch: Path) -> None:
    summary = summarize(collect_runs(sorted((batch / "runs").glob("*/*")))).set_index("model")
    dros = summary.loc["graph_drosophila"]
    assert dros["slice_roc_auc_mean"] == pytest.approx(0.82)
    assert dros["slice_roc_auc_sd"] == pytest.approx(np.std([0.80, 0.81, 0.82, 0.83, 0.84], ddof=1))
    assert dros["patient_roc_auc_mean"] == pytest.approx(0.87)
    rew = summary.loc["graph_rewired"]
    assert (rew["runs_ok"], rew["runs_failed"]) == (4, 1)
    assert rew["slice_roc_auc_mean"] == pytest.approx(np.mean([0.78, 0.79, 0.81, 0.82]))  # seed 2 excluded


def test_paired_differences_are_per_seed_and_skip_failed_seeds(batch: Path) -> None:
    runs = collect_runs(sorted((batch / "runs").glob("*/*")))
    diff = paired_differences(runs, "graph_drosophila", "graph_rewired")
    assert diff["seed"].tolist() == [0, 1, 3, 4]              # seed 2 has no valid rewired run
    assert diff["slice_roc_auc"].tolist() == pytest.approx([0.02] * 4)
    summary = summarize_differences(diff).set_index("metric")
    assert summary.loc["slice_roc_auc", "seeds_reference_higher"] == 4
    assert summary.loc["slice_roc_auc", "mean_difference"] == pytest.approx(0.02)
    assert summary.loc["slice_specificity", "seeds_equal"] == 4
    assert "p_value" not in summary.columns                    # descriptive only


def test_to_markdown() -> None:
    import pandas as pd

    table = to_markdown(pd.DataFrame({"a": [1, 2], "b": [0.12345, float("nan")]}))
    assert table.splitlines() == ["| a | b |", "|---|---|", "| 1 | 0.123 |", "| 2 | n/a |"]


def test_summarize_script_writes_tables_and_figures(batch: Path) -> None:
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "summarize_phase8.py"), "--batch", str(batch)],
                   check=True, capture_output=True, text=True)
    out = batch / "summary"
    for name in ("runs.csv", "summary_by_model.csv", "paired_differences.csv", "paired_summary.csv", "summary.md",
                 "fig1_patient_auc_by_seed.png", "fig2_slice_auc_by_seed.png", "fig3_mean_sd.png",
                 "fig4_paired_drosophila_minus_rewired.png", "fig5_training_stability.png"):
        assert (out / name).is_file(), name
    report = (out / "summary.md").read_text(encoding="utf-8")
    assert "collapsed_on_validation" in report
    assert "No significance claims" in report
