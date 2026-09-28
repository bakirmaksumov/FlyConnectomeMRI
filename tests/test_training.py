"""Tests for the CNN baseline, metrics, class weighting and the training protocol."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from flyconnectome_mri.datasets.prepare import prepare_dataset
from flyconnectome_mri.datasets.synthetic import make_synthetic_dataset
from flyconnectome_mri.models.cnn_baseline import CNNBaseline, count_parameters
from flyconnectome_mri.models.encoder import CNNEncoder
from flyconnectome_mri.training import experiment as experiment_module
from flyconnectome_mri.training.metrics import aggregate_by_patient, binary_metrics, format_metrics
from flyconnectome_mri.training.train import _is_better, compute_pos_weight
from flyconnectome_mri.utils.config import (
    Config,
    ConfigError,
    DataConfig,
    DatasetConfig,
    ExperimentConfig,
    TrainingConfig,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Metrics (hand-calculated)
# ---------------------------------------------------------------------------


def test_binary_metrics_hand_example() -> None:
    # true  : 1    1    1    0    0
    # prob  : .9   .6   .4   .3   .7   -> pred 1 1 0 0 1  => TP=2 FN=1 TN=1 FP=1
    m = binary_metrics(np.array([1, 1, 1, 0, 0]), np.array([0.9, 0.6, 0.4, 0.3, 0.7]))
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert m["sensitivity"] == pytest.approx(2 / 3) == m["recall"]
    assert m["specificity"] == pytest.approx(1 / 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m["accuracy"] == pytest.approx(3 / 5)
    # AUC: of the 3x2 (glioma, meningioma) pairs, the glioma scores higher in 4 -> 4/6
    assert m["roc_auc"] == pytest.approx(4 / 6)


def test_threshold_is_inclusive() -> None:
    assert binary_metrics(np.array([1, 0]), np.array([0.5, 0.49]))["confusion_matrix"]["tp"] == 1


def test_undefined_metrics_are_none_not_zero() -> None:
    single_class = binary_metrics(np.array([1, 1]), np.array([0.2, 0.9]))
    assert single_class["roc_auc"] is None and single_class["specificity"] is None
    no_positive_calls = binary_metrics(np.array([1, 0]), np.array([0.1, 0.2]))
    assert no_positive_calls["precision"] is None and no_positive_calls["f1"] is None
    assert "undefined" in format_metrics(no_positive_calls)


@pytest.mark.parametrize("y, p", [([0, 2], [0.1, 0.2]), ([0, 1], [0.1, 1.5]), ([0, 1], [0.1]),
                                  ([0, 1], [0.1, float("nan")])])
def test_invalid_metric_inputs_are_rejected(y, p) -> None:
    with pytest.raises(ValueError):
        binary_metrics(np.array(y), np.array(p))


def test_patient_aggregation_averages_slices() -> None:
    slices = pd.DataFrame({
        "patient_id": ["A", "A", "A", "B", "B"],
        "label_index": [1, 1, 1, 0, 0],
        "prob": [0.2, 0.5, 0.8, 0.1, 0.4],
    })
    patients = aggregate_by_patient(slices)
    assert patients["patient_id"].tolist() == ["A", "B"]
    assert patients["prob"].tolist() == pytest.approx([0.5, 0.25])
    assert patients["n_slices"].tolist() == [3, 2]
    assert patients["label_index"].tolist() == [1, 0]


def test_patient_with_mixed_labels_cannot_be_aggregated() -> None:
    slices = pd.DataFrame({"patient_id": ["A", "A"], "label_index": [1, 0], "prob": [0.2, 0.5]})
    with pytest.raises(ValueError, match="mixed labels"):
        aggregate_by_patient(slices)


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------


def test_pos_weight_is_negatives_over_positives() -> None:
    assert compute_pos_weight(np.array([1, 1, 1, 0])) == pytest.approx(1 / 3)
    assert compute_pos_weight(np.array([1, 0, 0, 0])) == pytest.approx(3.0)
    # the Figshare train split: 496 meningioma, 1010 glioma
    assert compute_pos_weight(np.array([0] * 496 + [1] * 1010)) == pytest.approx(496 / 1010)


def test_pos_weight_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        compute_pos_weight(np.array([1, 1]))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_model_shapes() -> None:
    x = torch.randn(4, 1, 128, 128)
    assert CNNEncoder()(x).shape == (4, 128)
    assert CNNBaseline()(x).shape == (4,)


def test_parameter_count_hand_calculated() -> None:
    # conv1: 1*16*3*3 + 16 = 160; conv2: 16*32*9 + 32 = 4640;
    # conv3: 32*128*9 + 128 = 36992; linear: 128 + 1 = 129  -> 41921
    assert count_parameters(CNNBaseline()) == 41921


def test_same_seed_gives_same_initial_weights() -> None:
    torch.manual_seed(0)
    a = CNNBaseline().state_dict()
    torch.manual_seed(0)
    b = CNNBaseline().state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_selection_rule_keeps_earlier_epoch_on_ties() -> None:
    assert _is_better(0.8, float("nan"), "val_roc_auc")
    assert not _is_better(0.8, 0.8, "val_roc_auc")
    assert _is_better(0.81, 0.8, "val_roc_auc")
    assert _is_better(0.4, 0.5, "val_loss") and not _is_better(0.5, 0.5, "val_loss")


# ---------------------------------------------------------------------------
# Training protocol
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("train")
    root = make_synthetic_dataset(tmp / "raw", patients_per_class=8, slices_per_patient=3, image_size=32)
    prepare_dataset(DatasetConfig(
        name="t", source="synthetic", license="n/a", root=str(root), classes={"normal": 0, "tumor": 1},
        positive_class="tumor", split_seed=0, output_dir=str(tmp / "prepared"),
        patient_id_source="filename_regex", patient_id_regex=r"^(?P<patient>P\d+)_s\d+", image_size=32,
        train_fraction=0.5, val_fraction=0.25, test_fraction=0.25,
    ), PROJECT_ROOT)
    return tmp / "prepared"


def make_experiment_config(prepared_dir: Path, **training) -> Config:
    return Config(
        experiment=ExperimentConfig(name="t", seed=0, device="cpu"),
        data=DataConfig(prepared_dir=str(prepared_dir)),
        training=TrainingConfig(**{"epochs": 3, "batch_size": 8, **training}),
    )


def test_test_split_is_loaded_only_after_training(prepared: Path, tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    original_from_prepared = experiment_module.MRIDataset.from_prepared
    original_fit = experiment_module.fit

    def recording_from_prepared(prepared_dir, split, **kwargs):
        events.append(f"load:{split}")
        return original_from_prepared(prepared_dir, split, **kwargs)

    def recording_fit(*args, **kwargs):
        events.append("fit:start")
        result = original_fit(*args, **kwargs)
        events.append("fit:end")
        return result

    monkeypatch.setattr(experiment_module.MRIDataset, "from_prepared", staticmethod(recording_from_prepared))
    monkeypatch.setattr(experiment_module, "fit", recording_fit)
    experiment_module.run_training_experiment(make_experiment_config(prepared), tmp_path, torch.device("cpu"),
                                              logging.getLogger("test"))
    assert events == ["load:train", "load:val", "fit:start", "fit:end", "load:test"]


def test_pos_weight_comes_from_train_split_only(prepared: Path, tmp_path: Path) -> None:
    metrics = experiment_module.run_training_experiment(make_experiment_config(prepared), tmp_path,
                                                        torch.device("cpu"), logging.getLogger("test"))
    manifest = pd.read_csv(prepared / "manifest.csv")
    train = manifest[manifest["split"] == "train"]["label_index"]
    assert metrics["class_weighting"]["pos_weight"] == pytest.approx((train == 0).sum() / (train == 1).sum())


def test_checkpoint_is_the_best_validation_epoch(prepared: Path, tmp_path: Path) -> None:
    metrics = experiment_module.run_training_experiment(make_experiment_config(prepared, epochs=4), tmp_path,
                                                        torch.device("cpu"), logging.getLogger("test"))
    history = pd.read_csv(tmp_path / "history.csv")
    best = int(history["val_roc_auc"].idxmax()) + 1  # first epoch with the maximum (ties -> earlier)
    assert metrics["training"]["best_epoch"] == best
    checkpoint = torch.load(tmp_path / "best_model.pt")
    assert checkpoint["epoch"] == best
    # Re-evaluating the saved checkpoint on validation reproduces the recorded best score.
    assert metrics["validation"]["slice"]["roc_auc"] == pytest.approx(history["val_roc_auc"][best - 1])


def test_early_stopping(prepared: Path, tmp_path: Path) -> None:
    metrics = experiment_module.run_training_experiment(
        make_experiment_config(prepared, epochs=30, early_stopping_patience=1, selection_metric="val_loss"),
        tmp_path, torch.device("cpu"), logging.getLogger("test"))
    assert metrics["training"]["epochs_run"] <= 30
    if metrics["training"]["stopped_early"]:
        assert metrics["training"]["epochs_run"] == metrics["training"]["best_epoch"] + 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_baseline_config_matches_base_except_name() -> None:
    base = load_config(PROJECT_ROOT / "configs" / "base.yaml").to_dict()
    baseline = load_config(PROJECT_ROOT / "configs" / "cnn_baseline.yaml").to_dict()
    base["experiment"]["name"] = "cnn_baseline"
    assert baseline == base
    assert baseline["data"]["prepared_dir"] == "data/splits/figshare_glioma_meningioma"


@pytest.mark.parametrize(
    "training",
    [{"epochs": 0}, {"learning_rate": 0.0}, {"decision_threshold": 1.0}, {"selection_metric": "test_auc"},
     {"class_weighting": "from_test"}],
)
def test_invalid_training_configs_are_rejected(training: dict) -> None:
    with pytest.raises(ConfigError):
        TrainingConfig(**training)
