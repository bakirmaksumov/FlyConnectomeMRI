"""End-to-end tests of scripts/run_experiment.py on a tiny synthetic dataset (2 epochs)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from flyconnectome_mri.datasets.prepare import prepare_dataset
from flyconnectome_mri.datasets.synthetic import make_synthetic_dataset
from flyconnectome_mri.utils.config import DatasetConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_experiment.py"
EXPECTED_FILES = ("config.yaml", "environment.json", "seed_fingerprint.json", "run.log", "best_model.pt",
                  "history.csv", "training_curves.png", "val_predictions_slice.csv",
                  "test_predictions_slice.csv", "test_predictions_patient.csv", "metrics.json")


@pytest.fixture(scope="module")
def experiment_config(tmp_path_factory) -> Path:
    """Synthetic data (NOT medical) prepared with patient IDs, and a 2-epoch config."""
    tmp = tmp_path_factory.mktemp("exp")
    root = make_synthetic_dataset(tmp / "raw", patients_per_class=8, slices_per_patient=3, image_size=32)
    prepare_dataset(DatasetConfig(
        name="synthetic_test", source="synthetic", license="n/a", root=str(root),
        classes={"normal": 0, "tumor": 1}, positive_class="tumor", split_seed=0,
        output_dir=str(tmp / "prepared"), patient_id_source="filename_regex",
        patient_id_regex=r"^(?P<patient>P\d+)_s\d+", image_size=32,
        train_fraction=0.5, val_fraction=0.25, test_fraction=0.25,
    ), PROJECT_ROOT)
    config = {
        "inherit": str(PROJECT_ROOT / "configs" / "base.yaml"),
        "experiment": {"name": "pipeline_test", "device": "cpu"},
        "data": {"prepared_dir": str(tmp / "prepared")},
        "training": {"epochs": 2, "batch_size": 8},
    }
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def run(config: Path, output_dir: Path, *extra: str) -> Path:
    """Run the script in a fresh Python process and return the created run folder."""
    before = set(output_dir.iterdir()) if output_dir.exists() else set()
    subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config), "--output-dir", str(output_dir), *extra],
        check=True, capture_output=True, text=True,
    )
    created = set(output_dir.iterdir()) - before
    assert len(created) == 1
    return created.pop()


def test_run_creates_all_outputs(experiment_config: Path, tmp_path: Path) -> None:
    run_dir = run(experiment_config, tmp_path)
    for name in EXPECTED_FILES:
        assert (run_dir / name).is_file(), name

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["seed"] == 0
    assert metrics["parameters"] > 0
    assert set(metrics["test"]) >= {"slice", "patient"}
    assert metrics["class_weighting"]["computed_from"] == "train split only"

    slices = pd.read_csv(run_dir / "test_predictions_slice.csv")
    assert {"image_path", "patient_id", "label_index", "prob", "pred"} <= set(slices.columns)
    patients = pd.read_csv(run_dir / "test_predictions_patient.csv")
    assert len(patients) == slices["patient_id"].nunique()
    assert metrics["test"]["patient"]["n"] == len(patients)

    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved["training"]["epochs"] == 2  # full config saved next to the checkpoint


def test_same_seed_reproduces_results_across_processes(experiment_config: Path, tmp_path: Path) -> None:
    a = run(experiment_config, tmp_path, "--seed", "3")
    b = run(experiment_config, tmp_path, "--seed", "3")
    c = run(experiment_config, tmp_path, "--seed", "4")
    fp = lambda d: json.loads((d / "seed_fingerprint.json").read_text(encoding="utf-8"))
    probs = lambda d: pd.read_csv(d / "test_predictions_slice.csv")["prob"].tolist()
    assert a != b
    assert fp(a) == fp(b) and fp(a) != fp(c)
    assert probs(a) == probs(b)  # bit-identical predictions on CPU
    assert probs(a) != probs(c)
