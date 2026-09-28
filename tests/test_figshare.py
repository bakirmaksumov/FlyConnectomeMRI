"""Tests for the Figshare .mat reader, extraction, CV folds, patient linking and download.

Fake .mat files are written with h5py using the same layout as the real
MATLAB v7.3 files (struct "cjdata"; arrays stored transposed; PID as char codes).
"""

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.io
import torch

from flyconnectome_mri.datasets.figshare import (
    extract_archives,
    load_cv_folds,
    load_mat_image,
    map_folds_to_patients,
    read_cjdata,
    scan_figshare,
)
from flyconnectome_mri.datasets.figshare_download import ChecksumError, download_file
from flyconnectome_mri.datasets.manifest import DatasetError
from flyconnectome_mri.datasets.mri_dataset import MRIDataset
from flyconnectome_mri.datasets.prepare import link_patient_ids, prepare_dataset
from flyconnectome_mri.utils.config import ConfigError, DatasetConfig, load_dataset_config
from flyconnectome_mri.utils.provenance import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODES = {1: "meningioma", 2: "glioma", 3: "pituitary"}
BINARY = {"meningioma": 0, "glioma": 1}


def write_cjdata(path: Path, label: int, pid: str, image: np.ndarray) -> None:
    """Write a MATLAB-v7.3-style cjdata struct (arrays transposed, as MATLAB does)."""
    with h5py.File(path, "w") as f:
        cj = f.create_group("cjdata")
        cj.create_dataset("label", data=np.array([[float(label)]]))
        cj.create_dataset("PID", data=np.array([[ord(c)] for c in pid], dtype=np.uint16))
        cj.create_dataset("image", data=image.astype(np.int16).T)
        cj.create_dataset("tumorMask", data=np.zeros(image.shape, dtype=np.uint8).T)


# (file number, label code, patient ID): 4 meningioma patients, 5 glioma "patients"
# (two of which, MR000001B and MR000001C, share a base number), 1 pituitary patient.
LAYOUT = [
    (1, 1, "1001"), (2, 1, "1001"), (3, 1, "1002"), (4, 1, "1003"), (5, 1, "1003"), (6, 1, "1004"),
    (7, 2, "2001"), (8, 2, "2001"), (9, 2, "2002"), (10, 2, "MR000001B"), (11, 2, "MR000001C"),
    (12, 2, "MR000001B"), (13, 2, "2003"), (14, 3, "3001"), (15, 3, "3001"),
    (16, 1, "1005"), (17, 2, "2004"), (18, 1, "1006"), (19, 2, "2005"), (20, 2, "2006"),
]


@pytest.fixture
def mat_root(tmp_path: Path) -> Path:
    root = tmp_path / "mat"
    root.mkdir()
    rng = np.random.default_rng(0)
    for number, code, pid in LAYOUT:
        size = 32 if number != 5 else 16  # one smaller image, like the real 256x256 slices
        image = rng.integers(0, 3000, (size, size))
        image[0, 0], image[0, 1] = 0, 3000  # known min/max
        write_cjdata(root / f"{number}.mat", code, pid, image)
    return root


def make_config(root: Path, **overrides) -> DatasetConfig:
    values = dict(
        name="fig", source="test", license="CC BY 4.0", root=str(root), format="figshare_mat",
        figshare_label_codes=dict(CODES), classes=dict(BINARY), positive_class="glioma",
        patient_id_source="figshare_pid", split_seed=0, output_dir=str(root.parent / "prepared"),
        image_size=16, train_fraction=0.5, val_fraction=0.25, test_fraction=0.25,
    )
    values.update(overrides)
    return DatasetConfig(**values)


# ---------------------------------------------------------------------------
# Reading .mat files
# ---------------------------------------------------------------------------


def test_read_cjdata_fields_and_orientation(tmp_path: Path) -> None:
    image = np.arange(12).reshape(3, 4)  # non-square: orientation errors would show
    write_cjdata(tmp_path / "1.mat", 2, "MR024780B", image)
    cj = read_cjdata(tmp_path / "1.mat")
    assert cj.label_code == 2
    assert cj.patient_id == "MR024780B"
    assert (cj.height, cj.width) == (3, 4)
    assert np.array_equal(cj.image, image)


def test_mat_image_is_min_max_scaled_float(tmp_path: Path) -> None:
    write_cjdata(tmp_path / "1.mat", 1, "7", np.array([[10, 20], [30, 110]]))
    pil = load_mat_image(tmp_path / "1.mat")
    assert pil.mode == "F"
    values = np.asarray(pil)
    assert values.min() == 0.0 and values.max() == 1.0
    assert values[0, 1] == pytest.approx(0.1)


def test_constant_or_broken_mat_files_are_errors(tmp_path: Path) -> None:
    write_cjdata(tmp_path / "1.mat", 1, "7", np.full((4, 4), 5))
    with pytest.raises(DatasetError, match="constant"):
        load_mat_image(tmp_path / "1.mat")
    (tmp_path / "2.mat").write_bytes(b"not hdf5")
    with pytest.raises(DatasetError, match="not a readable"):
        read_cjdata(tmp_path / "2.mat")


# ---------------------------------------------------------------------------
# Scanning and class selection
# ---------------------------------------------------------------------------


def test_scan_keeps_only_selected_classes_and_counts_exclusions(mat_root: Path) -> None:
    records, info = scan_figshare(mat_root, CODES, BINARY)
    assert set(records["label"]) == {"meningioma", "glioma"}
    assert info["excluded_classes"] == ["pituitary"]
    assert info["excluded_slices"] == 2
    assert info["all_slices"] == 20
    assert info["per_original_class"]["pituitary"] == {"slices": 2, "patients": 1}
    row = records[records["source_filename"] == "10.mat"].iloc[0]
    assert (row["patient_id"], row["label"], row["label_index"], row["original_label_code"]) == (
        "MR000001B", "glioma", 1, 2)
    assert records["image_number"].tolist() == sorted(records["image_number"])  # numeric, not "10" < "2"


def test_unknown_label_code_is_an_error(mat_root: Path) -> None:
    write_cjdata(mat_root / "21.mat", 9, "x", np.eye(4))
    with pytest.raises(DatasetError, match="unknown label code 9"):
        scan_figshare(mat_root, CODES, BINARY)


def test_source_files_are_not_modified_by_preparation(mat_root: Path) -> None:
    before = {p.name: sha256_file(p) for p in mat_root.glob("*.mat")}
    prepare_dataset(make_config(mat_root), PROJECT_ROOT)
    assert {p.name: sha256_file(p) for p in mat_root.glob("*.mat")} == before


# ---------------------------------------------------------------------------
# Patient-level splitting on .mat data
# ---------------------------------------------------------------------------


def test_prepared_manifest_has_zero_patient_overlap(mat_root: Path) -> None:
    for seed in range(15):
        manifest, _ = prepare_dataset(make_config(mat_root, split_seed=seed), PROJECT_ROOT, overwrite=True)
        assert manifest.groupby("patient_id")["split"].nunique().max() == 1
        assert "pituitary" not in set(manifest["label"])


def test_link_regex_keeps_related_ids_together(mat_root: Path) -> None:
    config = make_config(mat_root, patient_link_regex=r"^(?P<base>MR\d{6})[A-Z]?$")
    for seed in range(15):
        manifest, meta = prepare_dataset(
            DatasetConfig(**{**config.__dict__, "split_seed": seed}), PROJECT_ROOT, overwrite=True)
        linked = manifest[manifest["patient_id"].str.startswith("MR000001")]
        assert linked["split"].nunique() == 1, f"seed {seed}"
    assert meta["patient_ids"]["linked_ids"] == {"MR000001": ["MR000001B", "MR000001C"]}
    assert set(manifest["patient_id"]) >= {"MR000001B", "MR000001C"}  # originals unchanged


def test_link_patient_ids_function() -> None:
    keys, linked = link_patient_ids(pd.Series(["MR1B", "MR1C", "MR1", "MR2B", "123"]), r"^(?P<base>MR\d)[A-Z]?$")
    assert keys.tolist() == ["MR1", "MR1", "MR1", "MR2", "123"]
    assert linked == {"MR1": ["MR1", "MR1B", "MR1C"]}


def test_dataset_loads_mat_slices_as_normalized_tensors(mat_root: Path) -> None:
    prepare_dataset(make_config(mat_root), PROJECT_ROOT)
    dataset = MRIDataset.from_prepared(mat_root.parent / "prepared", "train")
    image, label = dataset[0]
    assert image.shape == (1, 16, 16) and image.dtype == torch.float32
    assert label in (0, 1)


# ---------------------------------------------------------------------------
# Official CV folds
# ---------------------------------------------------------------------------


def test_cv_folds_map_cleanly_and_are_stored(mat_root: Path, tmp_path: Path) -> None:
    folds = np.array([1 + (i % 5) for i in range(20)])  # per image number 1..20
    pid_by_number = {n: pid for n, _, pid in LAYOUT}
    for number in range(1, 21):  # same fold for every slice of a patient -> clean mapping
        first = min(n for n, p in pid_by_number.items() if p == pid_by_number[number])
        folds[number - 1] = folds[first - 1]
    scipy.io.savemat(tmp_path / "cvind.mat", {"cvind": folds.reshape(-1, 1).astype(float)})
    manifest, meta = prepare_dataset(make_config(mat_root, cv_indices_file=str(tmp_path / "cvind.mat")), PROJECT_ROOT)
    info = meta["format"]["official_cv_folds"]
    assert info["maps_cleanly_to_patients"] is True
    assert info["patients_split_across_folds"] == []
    assert manifest.loc[manifest["source_filename"] == "7.mat", "cv_fold"].item() == folds[6]


def test_cv_folds_that_split_a_patient_are_flagged(mat_root: Path) -> None:
    records, _ = scan_figshare(mat_root, CODES, BINARY)
    folds = np.ones(20, dtype=int)
    folds[1] = 2  # 2.mat belongs to patient 1001, like 1.mat, but gets another fold
    _, split_patients = map_folds_to_patients(records, folds)
    assert split_patients == ["1001"]


def test_cv_file_with_wrong_length_is_an_error(tmp_path: Path) -> None:
    scipy.io.savemat(tmp_path / "cvind.mat", {"cvind": np.ones((7, 1))})
    with pytest.raises(DatasetError, match="expected 20"):
        load_cv_folds(tmp_path / "cvind.mat", 20)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)


def test_extract_is_flat_read_only_and_idempotent(tmp_path: Path) -> None:
    make_zip(tmp_path / "a.zip", {"1.mat": b"one", "2.mat": b"two"})
    make_zip(tmp_path / "b.zip", {"10.mat": b"ten"})
    out = tmp_path / "out"
    table = extract_archives(tmp_path, out)
    assert table["file"].tolist() == ["1.mat", "2.mat", "10.mat"]
    assert table["archive"].tolist() == ["a.zip", "a.zip", "b.zip"]
    assert not os.stat(out / "1.mat").st_mode & stat.S_IWRITE
    assert extract_archives(tmp_path, out).equals(table)  # second run verifies, same result


def test_extract_refuses_unsafe_paths_and_conflicts(tmp_path: Path) -> None:
    make_zip(tmp_path / "evil.zip", {"../escape.mat": b"x"})
    with pytest.raises(DatasetError, match="Unsafe path"):
        extract_archives(tmp_path, tmp_path / "out")
    (tmp_path / "evil.zip").unlink()
    make_zip(tmp_path / "a.zip", {"1.mat": b"one"})
    make_zip(tmp_path / "b.zip", {"1.mat": b"other"})
    with pytest.raises(DatasetError, match="appears in both"):
        extract_archives(tmp_path, tmp_path / "out2")


def test_extract_detects_modified_extracted_file(tmp_path: Path) -> None:
    make_zip(tmp_path / "a.zip", {"1.mat": b"one"})
    out = tmp_path / "out"
    extract_archives(tmp_path, out)
    os.chmod(out / "1.mat", stat.S_IWRITE | stat.S_IREAD)
    (out / "1.mat").write_bytes(b"tampered")
    with pytest.raises(DatasetError, match="differs"):
        extract_archives(tmp_path, out)


# ---------------------------------------------------------------------------
# Download with MD5 verification (local file:// URLs, no network)
# ---------------------------------------------------------------------------


def test_download_verifies_md5(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * 5000)
    good = hashlib.md5(source.read_bytes()).hexdigest()
    dest = tmp_path / "dl" / "file.bin"
    dest.parent.mkdir()
    download_file(source.as_uri(), dest, good, 5000)
    assert dest.read_bytes() == source.read_bytes()
    assert not dest.with_name("file.bin.part").exists()


def test_download_with_wrong_md5_is_rejected_and_removed(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"data")
    dest = tmp_path / "file.bin"
    with pytest.raises(ChecksumError, match="does not match"):
        download_file(source.as_uri(), dest, "0" * 32)
    assert not dest.exists() and not dest.with_name("file.bin.part").exists()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"patient_id_source": "none"},                                # figshare needs PIDs
        {"figshare_label_codes": None},
        {"figshare_label_codes": {1: "meningioma", 2: "meningioma", 3: "pituitary"}},
        {"classes": {"normal": 0, "glioma": 1}},                      # "normal" is not in the codes
        {"grayscale": False},
        {"patient_link_regex": r"^(MR\d+)"},                          # no (?P<base>)
    ],
)
def test_invalid_figshare_configs_are_rejected(tmp_path: Path, overrides: dict) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, **overrides)


def test_figshare_options_rejected_for_image_folders(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, format="image_folder")  # still has figshare_pid / label codes


def test_shipped_figshare_config() -> None:
    config = load_dataset_config(PROJECT_ROOT / "configs" / "datasets" / "figshare_glioma_meningioma.yaml")
    assert config.classes == {"meningioma": 0, "glioma": 1}
    assert config.positive_class == "glioma"
    assert config.figshare_label_codes == {1: "meningioma", 2: "glioma", 3: "pituitary"}
    assert config.patient_id_source == "figshare_pid"
