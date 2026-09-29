"""Tests for leakage-safe CV using the official outer test folds (Phase 10)."""

from pathlib import Path

import pandas as pd
import pytest

from flyconnectome_mri.datasets.cv_folds import build_outer_fold, check_oof_coverage, validation_fold
from flyconnectome_mri.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# patient -> (link key, official fold, label). Family "MR1" spans folds 1, 2 and 3;
# family "MR2" spans folds 3 and 4; everything else is unlinked.
PATIENTS = {
    "MR1B": ("MR1", 1, "glioma"), "MR1C": ("MR1", 2, "glioma"), "MR1": ("MR1", 3, "glioma"),
    "MR2": ("MR2", 3, "glioma"), "MR2B": ("MR2", 4, "glioma"),
    "p1": ("p1", 1, "meningioma"), "p2": ("p2", 2, "meningioma"), "p3": ("p3", 3, "meningioma"),
    "p4": ("p4", 4, "meningioma"), "p5": ("p5", 5, "meningioma"), "g5": ("g5", 5, "glioma"),
    "g1": ("g1", 1, "glioma"), "g2": ("g2", 2, "glioma"), "g4": ("g4", 4, "glioma"),
}


@pytest.fixture
def manifest() -> pd.DataFrame:
    rows = []
    for i, (pid, (key, fold, label)) in enumerate(PATIENTS.items()):
        for s in range(2):  # two slices per patient
            rows.append({"image_path": f"{i}_{s}.mat", "patient_id": pid, "patient_link_key": key, "cv_fold": fold,
                         "label": label, "label_index": int(label == "glioma"), "group_id": f"g{key}",
                         "sha256": f"h{i}_{s}", "split": "train"})
    return pd.DataFrame(rows)


def test_validation_fold_rule_is_cyclic() -> None:
    assert [validation_fold(k) for k in range(1, 6)] == [2, 3, 4, 5, 1]
    with pytest.raises(ValueError):
        validation_fold(6)


def test_test_fold_is_the_unchanged_official_fold(manifest: pd.DataFrame) -> None:
    for k in range(1, 6):
        fold = build_outer_fold(manifest, k).manifest
        tested = set(fold.loc[fold["split"] == "test", "patient_id"])
        assert tested == {p for p, (_, f, _) in PATIENTS.items() if f == k}  # nothing removed from test
        assert set(fold.loc[fold["split"] == "test", "cv_fold"]) == {k}


def test_relatives_are_removed_from_validation_and_training(manifest: pd.DataFrame) -> None:
    # outer fold 1: test = fold 1 (MR1B, p1, g1); val = fold 2 (MR1C removed: relative of MR1B)
    # train = folds 3, 4, 5: MR1 removed (relative of test); MR2 and MR2B both stay (no relative in test/val)
    fold = build_outer_fold(manifest, 1)
    splits = fold.manifest.drop_duplicates("patient_id").set_index("patient_id")["split"].to_dict()
    assert "MR1C" not in splits and "MR1" not in splits
    assert splits["MR2"] == "train" and splits["MR2B"] == "train"
    ex = fold.exclusions.set_index("removed_patient_id")
    assert ex.loc["MR1C", "removed_from"] == "validation" and ex.loc["MR1C", "reason"] == "linked_to_test"
    assert ex.loc["MR1", "removed_from"] == "train" and ex.loc["MR1", "reason"] == "linked_to_test"
    assert ex.loc["MR1", "original_cv_fold"] == 3


def test_training_relatives_of_validation_are_removed(manifest: pd.DataFrame) -> None:
    # outer fold 3: test = fold 3 (MR1, MR2, p3); val = fold 4 (MR2B removed: relative of test MR2)
    # outer fold 5: test = fold 5; val = fold 1 (MR1B kept); train = folds 2-4: MR1C, MR1 linked_to_validation
    fold5 = build_outer_fold(manifest, 5)
    ex = fold5.exclusions.set_index("removed_patient_id")
    assert set(ex.index) == {"MR1C", "MR1"}
    assert set(ex["reason"]) == {"linked_to_validation"} and set(ex["removed_from"]) == {"train"}


def test_zero_overlap_of_patient_and_link_key_in_every_fold(manifest: pd.DataFrame) -> None:
    for k in range(1, 6):
        fold = build_outer_fold(manifest, k).manifest
        for column in ("patient_id", "patient_link_key"):
            sets = {s: set(fold.loc[fold["split"] == s, column]) for s in ("train", "val", "test")}
            assert not sets["train"] & sets["val"]
            assert not sets["train"] & sets["test"]
            assert not sets["val"] & sets["test"]


def test_every_patient_is_tested_exactly_once(manifest: pd.DataFrame) -> None:
    folds = {k: build_outer_fold(manifest, k).manifest for k in range(1, 6)}
    coverage = check_oof_coverage(manifest, folds)
    assert coverage == {"patients": 14, "tested_exactly_once": 14, "tested_zero_times": 0, "tested_more_than_once": 0}
    broken = dict(folds)
    broken[2] = broken[2][broken[2]["split"] != "test"]
    with pytest.raises(ValueError, match="never tested"):
        check_oof_coverage(manifest, broken)


def test_official_cv_fold_values_are_never_changed(manifest: pd.DataFrame) -> None:
    original = manifest.set_index("image_path")["cv_fold"]
    for k in range(1, 6):
        fold = build_outer_fold(manifest, k).manifest.set_index("image_path")["cv_fold"]
        assert (fold == original.loc[fold.index]).all()


def test_counts_report_original_retained_removed(manifest: pd.DataFrame) -> None:
    counts = build_outer_fold(manifest, 1).counts.set_index(["split", "class"])
    val = counts.loc[("val", "all")]
    assert (val["original_patients"], val["retained_patients"], val["removed_patients"]) == (3, 2, 1)
    assert counts.loc[("val", "glioma"), "removed_patients"] == 1
    assert counts.loc[("val", "meningioma"), "removed_patients"] == 0
    assert counts.loc[("test", "all"), "removed_patients"] == 0
    assert counts.loc[("train", "all"), "removed_slices"] == 2


def test_patient_spanning_two_folds_is_rejected(manifest: pd.DataFrame) -> None:
    manifest.loc[0, "cv_fold"] = 5
    with pytest.raises(ValueError, match="spans several official folds"):
        build_outer_fold(manifest, 1)


def test_prepared_dir_override_in_config() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "visual_drosophila.yaml", seed=2,
                         prepared_dir="data/splits/figshare_cv_leakage_safe/fold3")
    assert config.data.prepared_dir == "data/splits/figshare_cv_leakage_safe/fold3"
    assert config.experiment.seed == 2


def test_phase10_plan_has_exactly_100_runs() -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        import run_phase10
    finally:
        sys.path.pop(0)
    jobs = run_phase10.planned_jobs()
    assert len(jobs) == 100 and len({job.key for job in jobs}) == 100
    assert set(run_phase10.MODELS) == {"cnn_baseline", "visual_drosophila", "visual_rewired", "visual_ws"}
    assert {job.fold for job in jobs} == {1, 2, 3, 4, 5} and {job.seed for job in jobs} == {0, 1, 2, 3, 4}
    rows = run_phase10.plan_rows(jobs)
    rewired = [r for r in rows if r["model"] == "visual_rewired" and r["seed"] == 4]
    assert all(r["graph_file"] == "data/connectome/rewired_visual128_raw_seed4.csv" for r in rewired)
    drosophila = {r["graph_file"] for r in rows if r["model"] == "visual_drosophila"}
    assert drosophila == {"data/connectome/drosophila_visual_128_raw.csv"}  # same graph for every run
