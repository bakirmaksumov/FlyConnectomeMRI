"""Tests for the pre-registered Phase 10 aggregation, on synthetic/mock run outputs only."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyconnectome_mri.evaluation.cv_summary import (
    FOLDS,
    MODELS,
    SEEDS,
    build_oof,
    collect_cv_runs,
    fold_level_summary,
    fold_means,
    oof_metrics,
    paired,
    paired_fold_means,
    paired_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 10 patients, 2 per official fold: one glioma (label 1) and one meningioma (label 0).
COHORT = {f"P{f}{c}": (f, c) for f in FOLDS for c in (0, 1)}
COHORT["MR91"] = (1, 1)  # an MR-prefixed patient in fold 1 (id_format check)
FAILED = ("visual_ws", 1, 4)      # collapsed on validation: metrics.json without test results
CRASHED = ("visual_rewired", 5, 4)  # crashed: no metrics.json at all


def auc_for(model: str, fold: int, seed: int) -> float:
    """Known patient AUC per run: model offset + fold offset + seed offset."""
    return 0.70 + 0.05 * MODELS.index(model) + 0.01 * fold + 0.002 * seed


def write_run(run_dir: Path, model: str, fold: int, seed: int, status: str = "ok") -> None:
    run_dir.mkdir(parents=True)
    level = lambda auc: {"roc_auc": auc, "balanced_accuracy": auc - 0.1, "sensitivity": 0.6, "specificity": 0.9,
                         "f1": 0.7, "precision": 0.8, "recall": 0.6, "accuracy": 0.7,
                         "confusion_matrix": {"tn": 1, "fp": 0, "fn": 0, "tp": 1}}
    auc = auc_for(model, fold, seed)
    metrics = {"model": model, "seed": seed, "status": status, "parameters": 44411, "checkpoint_sha256": "abc",
               "class_weighting": {"pos_weight": 0.7},
               "training": {"best_epoch": 10 + seed, "epochs_run": 17, "training_seconds": 100.0 + fold,
                            "best_val_score": 0.9}, "test": None}
    if status == "ok":
        metrics["test"] = {"patient": level(auc), "slice": level(auc - 0.05)}
        metrics["inference"] = {"forward_ms_per_slice": 2.0}
        patients = [p for p, (f, _) in COHORT.items() if f == fold]
        rows = [{"patient_id": p, "patient_link_key": p, "label": "glioma" if COHORT[p][1] else "meningioma",
                 "label_index": COHORT[p][1], "n_slices": 2,
                 "prob": 0.8 if COHORT[p][1] else 0.2 + 0.01 * seed, "pred": COHORT[p][1]} for p in patients]
        pd.DataFrame(rows).to_csv(run_dir / "test_predictions_patient.csv", index=False)
        slices = [{"image_path": f"{r['patient_id']}_{i}.mat", "patient_id": r["patient_id"],
                   "label_index": r["label_index"], "prob": r["prob"], "pred": r["pred"]} for r in rows for i in range(2)]
        pd.DataFrame(slices).to_csv(run_dir / "test_predictions_slice.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


@pytest.fixture
def batch(tmp_path: Path) -> Path:
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            for model in MODELS:
                key = (model, fold, seed)
                run_dir = tmp_path / "runs" / model / f"fold{fold}" / f"seed{seed}"
                if key == CRASHED:
                    rows.append([model, seed, fold, "c", "p", 1, 5.0, "", "log", "t"])
                    continue
                write_run(run_dir, model, fold, seed, "collapsed_on_validation" if key == FAILED else "ok")
                rows.append([model, seed, fold, "c", "p", 0, 120.0, str(run_dir), "log", "t"])
    index = pd.DataFrame(rows, columns=["model", "seed", "fold", "config", "prepared_dir", "returncode",
                                        "wall_seconds", "run_dir", "log", "started_utc"])
    index.sample(frac=1, random_state=0).to_csv(tmp_path / "runs_index.csv", index=False)  # shuffled order
    manifest = pd.DataFrame([{"patient_id": p, "image_path": f"{p}_{i}.mat", "sha256": "x"} for p in COHORT for i in range(2)])
    manifest.to_csv(tmp_path / "cohort_manifest.csv", index=False)
    return tmp_path


def runs_of(batch: Path) -> pd.DataFrame:
    return collect_cv_runs(pd.read_csv(batch / "runs_index.csv", dtype={"run_dir": str}))


def test_all_100_planned_runs_are_indexed_including_failures(batch: Path) -> None:
    runs = runs_of(batch)
    assert len(runs) == 100 and len(runs.drop_duplicates(["model", "outer_fold", "seed"])) == 100
    status = runs.set_index(["model", "outer_fold", "seed"])["status"]
    assert status[FAILED] == "collapsed_on_validation"
    assert status[CRASHED] == "failed_exit_1"
    assert (status == "ok").sum() == 98
    failed = runs.set_index(["model", "outer_fold", "seed"]).loc[[FAILED, CRASHED]]
    assert failed["patient_roc_auc"].isna().all()          # NA, not dropped and not zero
    assert runs.set_index(["model", "outer_fold", "seed"]).loc[("visual_rewired", 1, 1), "graph_seed"] == 1
    assert pd.isna(runs.set_index(["model", "outer_fold", "seed"]).loc[("visual_drosophila", 1, 1), "graph_seed"])


def test_fold_means_average_seeds_first(batch: Path) -> None:
    runs = runs_of(batch)
    folds = fold_means(runs).set_index(["model", "outer_fold"])
    # visual_ws fold 1: seeds 0-3 ok (seed 4 failed)
    expected = np.mean([auc_for("visual_ws", 1, s) for s in (0, 1, 2, 3)])
    assert folds.loc[("visual_ws", 1), "patient_roc_auc"] == pytest.approx(expected)
    assert folds.loc[("visual_ws", 1), "seeds_ok"] == 4
    summary = fold_level_summary(fold_means(runs)).set_index(["model", "metric"])
    ws = summary.loc[("visual_ws", "patient_roc_auc")]
    fold_values = [np.mean([auc_for("visual_ws", f, s) for s in SEEDS if (f, s) != (1, 4)]) for f in FOLDS]
    assert ws["n"] == 5 and ws["mean"] == pytest.approx(np.mean(fold_values))
    assert ws["sd"] == pytest.approx(np.std(fold_values, ddof=1))
    assert ws["median"] == pytest.approx(np.median(fold_values)) and ws["min"] == pytest.approx(min(fold_values))


def test_fold_level_mean_is_not_a_pooled_run_mean(batch: Path) -> None:
    """With an unequal number of ok seeds per fold, mean-of-fold-means != mean of all runs."""
    runs = runs_of(batch)
    ws_runs = runs[(runs["model"] == "visual_ws") & (runs["status"] == "ok")]["patient_roc_auc"]
    fold_level = fold_level_summary(fold_means(runs)).set_index(["model", "metric"]).loc[("visual_ws", "patient_roc_auc"), "mean"]
    assert fold_level != pytest.approx(ws_runs.mean(), abs=1e-12)


def test_paired_comparison_matches_fold_and_seed(batch: Path) -> None:
    runs = runs_of(batch)
    diff = paired(runs, "visual_drosophila", "visual_rewired").set_index(["outer_fold", "seed"])
    assert len(diff) == 25
    # AUC offsets: drosophila index 1, rewired index 2 -> difference exactly -0.05 for every matched pair
    ok = diff["patient_roc_auc"].dropna()
    assert len(ok) == 24 and ok.to_numpy() == pytest.approx(np.full(24, -0.05))
    assert pd.isna(diff.loc[(5, 4), "patient_roc_auc"]) and not diff.loc[(5, 4), "both_ok"]
    folds = paired_fold_means(diff.reset_index()).set_index("outer_fold")
    assert folds.loc[5, "pairs_used"] == 4 and folds.loc[1, "pairs_used"] == 5
    summary = paired_summary(diff.reset_index())
    assert (summary["pairs"], summary["pairs_na"], summary["negative"], summary["positive"]) == (24, 1, 24, 0)


def test_oof_merge_has_each_patient_exactly_once(batch: Path) -> None:
    runs = runs_of(batch)
    oof, slices, status = build_oof(runs, "visual_drosophila", 0, set(COHORT))
    assert status == "ok"
    assert len(oof) == len(COHORT) and oof["patient_id"].is_unique
    assert set(oof["patient_id"]) == set(COHORT)
    assert oof.set_index("patient_id").loc["P31", "outer_fold"] == 3
    assert oof.set_index("patient_id").loc["MR91", "id_format"] == "MR-prefixed"
    assert oof.set_index("patient_id").loc["P10", "id_format"] == "numeric"
    assert set(oof.columns) >= {"patient_id", "patient_link_key", "true_label", "outer_fold", "mean_probability",
                                "predicted_label", "model", "seed"}
    assert len(slices) == 2 * len(COHORT)
    metrics = oof_metrics(oof)
    assert metrics["roc_auc"] == 1.0        # gliomas 0.8 vs meningiomas 0.2: computed over the 11 pooled patients
    assert metrics["cm_tp"] + metrics["cm_fn"] == 6   # 6 gliomas, each counted once (not once per run)


def test_oof_is_incomplete_when_a_fold_run_failed(batch: Path) -> None:
    runs = runs_of(batch)
    oof, slices, status = build_oof(runs, "visual_ws", 4, set(COHORT))
    assert oof is None and status == "incomplete_failed_fold"
    assert np.isnan(oof_metrics(None)["roc_auc"])


def test_oof_rejects_duplicate_or_missing_patients(batch: Path) -> None:
    runs = runs_of(batch)
    with pytest.raises(ValueError, match="missing"):
        build_oof(runs, "visual_drosophila", 1, set(COHORT) | {"GHOST"})
    run_dir = Path(runs[(runs["model"] == "visual_drosophila") & (runs["outer_fold"] == 2) & (runs["seed"] == 1)]["run_dir"].iloc[0])
    table = pd.read_csv(run_dir / "test_predictions_patient.csv")
    extra = table.iloc[[0]].assign(patient_id="P10")  # P10 belongs to fold 1
    pd.concat([table, extra]).to_csv(run_dir / "test_predictions_patient.csv", index=False)
    with pytest.raises(ValueError, match="more than one fold"):
        build_oof(runs, "visual_drosophila", 1, set(COHORT))


def test_summary_script_writes_every_predefined_output(batch: Path) -> None:
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "summarize_phase10.py"), "--batch", str(batch),
                    "--cohort-manifest", str(batch / "cohort_manifest.csv")], check=True, capture_output=True, text=True)
    out = batch / "summary"
    for name in ("summary.md", "run_level_metrics.csv", "fold_level_metrics.csv", "paired_drosophila_rewired.csv",
                 "paired_drosophila_ws.csv", "paired_drosophila_cnn.csv", "oof_metrics.csv", "run_index.csv",
                 "checksums.json"):
        assert (out / name).is_file(), name
    for i in range(1, 10):
        assert len(list(out.glob(f"fig{i}_*.png"))) == 1, i
    runs = pd.read_csv(out / "run_level_metrics.csv")
    assert len(runs) == 100 and (runs["status"] != "ok").sum() == 2
    oof = pd.read_csv(out / "oof_metrics.csv")
    assert len(oof) == 20 and (oof["status"] != "ok").sum() == 2   # ws seed 4 and rewired seed 4 incomplete
    assert (batch / "oof" / "visual_drosophila" / "oof_predictions_seed0.csv").is_file()
    assert not (batch / "oof" / "visual_ws" / "oof_predictions_seed4.csv").exists()
    checksums = json.loads((out / "checksums.json").read_text(encoding="utf-8"))
    assert "summary.md" in checksums and "oof/visual_drosophila/oof_predictions_seed0.csv" in checksums
    report = (out / "summary.md").read_text(encoding="utf-8")
    assert "NOT a pure topology comparison" in report and "NOT 25 independent" in report
    assert "collapsed_on_validation" in report and "failed_exit_1" in report
