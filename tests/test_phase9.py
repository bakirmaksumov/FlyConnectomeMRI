"""Tests for Phase 9 infrastructure: graph labels, visual configs, batch summary (fake runs)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from flyconnectome_mri.training.experiment import graph_seed, model_name
from flyconnectome_mri.utils.config import ConfigError, GraphConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUAL = ["visual_drosophila", "visual_rewired", "visual_er", "visual_ws", "visual_ba"]


@pytest.mark.parametrize("name", VISUAL)
def test_visual_configs(name: str) -> None:
    config = load_config(PROJECT_ROOT / "configs" / f"{name}.yaml", seed=3)
    kind = name.removeprefix("visual_")
    assert model_name(config) == f"graph_visual_{kind}"
    assert config.graph.label == "visual"
    if kind == "drosophila":
        assert config.graph.edges_file == "data/connectome/drosophila_visual_128_raw.csv"
        assert graph_seed(config) is None
    else:
        assert config.graph.edges_file == f"data/connectome/{kind}_visual128_raw_seed3.csv"
        assert graph_seed(config) == 3


def test_eb_model_names_are_unchanged() -> None:
    """Frozen Phase 8 outputs use these names."""
    assert model_name(load_config(PROJECT_ROOT / "configs" / "drosophila.yaml")) == "graph_drosophila"
    assert model_name(load_config(PROJECT_ROOT / "configs" / "rewired.yaml")) == "graph_rewired"
    assert model_name(load_config(PROJECT_ROOT / "configs" / "cnn_baseline.yaml")) == "cnn_baseline"


def test_graph_label_validation() -> None:
    with pytest.raises(ConfigError, match="label"):
        GraphConfig(type="er", edges_file="a", nodes_file="b", label="bad label")


def test_phase9_runner_models_and_seeds() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        import run_phase9
    finally:
        sys.path.pop(0)
    assert list(run_phase9.MODELS) == VISUAL
    assert run_phase9.SEEDS == [0, 1, 2, 3, 4]
    assert "cnn_baseline" not in run_phase9.MODELS  # CNN baseline is NOT retrained


def fake_metrics(model: str, seed: int, auc: float) -> dict:
    level = {"roc_auc": auc, "balanced_accuracy": auc - 0.1, "sensitivity": 0.6, "specificity": 0.9, "f1": 0.7,
             "precision": 0.9, "recall": 0.6, "accuracy": 0.7,
             "confusion_matrix": {"tn": 9, "fp": 1, "fn": 4, "tp": 6}}
    return {"model": model, "seed": seed, "graph_seed": seed, "status": "ok", "parameters": 48527,
            "training": {"best_epoch": 5, "epochs_run": 12, "training_seconds": 90.0, "best_val_score": 0.9},
            "validation_diagnostics": {"logit_std": 2.0, "graph_layer2_dead_nodes_frac": 0.05},
            "validation": {"slice": {"roc_auc": 0.9}, "patient": {"roc_auc": 0.95}},
            "inference": {"forward_ms_per_slice": 2.0},
            "test": {"slice": dict(level), "patient": dict(level, roc_auc=auc + 0.02)}}


def test_summarize_phase9_on_fake_batch(tmp_path: Path) -> None:
    rows = []
    for seed in range(5):
        for i, model in enumerate(f"graph_{m}" for m in VISUAL):
            run_dir = tmp_path / "runs" / model / f"s{seed}"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(json.dumps(fake_metrics(model, seed, 0.80 + 0.01 * i)), encoding="utf-8")
            rows.append([model, seed, "c", 0, 1, str(run_dir), "log", "t"])
    with open(tmp_path / "runs_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", "config", "returncode", "wall_seconds", "run_dir", "log", "started_utc"])
        writer.writerows(rows)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "summarize_phase9.py"), "--batch", str(tmp_path)],
                   check=True, capture_output=True, text=True)
    out = tmp_path / "summary"
    for i in range(1, 7):
        assert len(list(out.glob(f"fig{i}_*.png"))) == 1
    summary = pd.read_csv(out / "summary_by_model.csv")
    assert summary["model"].tolist()[0] == "cnn_baseline"          # Phase 8 reference, not retrained
    assert summary.set_index("model").loc["cnn_baseline", "runs_ok"] == 5
    diff = pd.read_csv(out / "paired_differences.csv")
    primary = diff[diff["comparison"] == "graph_visual_drosophila - graph_visual_rewired"]
    assert primary["patient_roc_auc"].tolist() == pytest.approx([-0.01] * 5)
    eb_vs_visual = pd.read_csv(out / "eb_vs_visual.csv")
    assert set(eb_vs_visual["family"]) == {"Drosophila", "Rewired", "ER", "WS", "BA"}
    report = (out / "summary.md").read_text(encoding="utf-8")
    assert "PRIMARY metric: patient-level ROC AUC" in report and "not a controlled comparison" in report
