"""Tests for the MRI dataset pipeline: scanning, patient IDs, splitting, leakage, loading."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from flyconnectome_mri.datasets.manifest import DatasetError, scan_image_folder
from flyconnectome_mri.datasets.mri_dataset import MRIDataset, read_manifest
from flyconnectome_mri.datasets.prepare import NO_PATIENT_ID_WARNING, prepare_dataset
from flyconnectome_mri.datasets.splits import LeakageError, build_groups, check_no_leakage
from flyconnectome_mri.datasets.synthetic import make_synthetic_dataset
from flyconnectome_mri.datasets.transforms import build_transform, compute_channel_stats
from flyconnectome_mri.utils.config import ConfigError, DatasetConfig, load_dataset_config
from flyconnectome_mri.utils.provenance import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"normal": 0, "tumor": 1}
REGEX = r"^(?P<patient>P\d+)_s\d+"


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """12 synthetic patients per class, 3 slices each (72 images, 16x16)."""
    return make_synthetic_dataset(tmp_path / "ds", patients_per_class=12, slices_per_patient=3, image_size=16)


def make_config(root: Path, **overrides) -> DatasetConfig:
    values = dict(
        name="test", source="synthetic", license="n/a", root=str(root), classes=dict(CLASSES),
        positive_class="tumor", split_seed=0, output_dir=str(root.parent / "prepared"),
        patient_id_source="filename_regex", patient_id_regex=REGEX, image_size=16,
    )
    values.update(overrides)
    return DatasetConfig(**values)


def prepare(root: Path, **overrides) -> tuple[pd.DataFrame, dict]:
    return prepare_dataset(make_config(root, **overrides), PROJECT_ROOT, overwrite=True)


def save_png(path: Path, value: int = 100, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size), value, dtype=np.uint8), mode="L").save(path)


# ---------------------------------------------------------------------------
# Scanning, labels and extensions
# ---------------------------------------------------------------------------


def test_labels_come_from_top_level_class_folder(data_root: Path) -> None:
    save_png(data_root / "tumor" / "extra" / "deep" / "P0099_s00.png")
    records = scan_image_folder(data_root, CLASSES).records
    deep = records[records["image_path"] == "tumor/extra/deep/P0099_s00.png"].iloc[0]
    assert deep["label"] == "tumor" and deep["label_index"] == 1
    assert set(records.loc[records["image_path"].str.startswith("normal/"), "label_index"]) == {0}
    assert set(records.loc[records["image_path"].str.startswith("tumor/"), "label_index"]) == {1}


def test_supported_extensions_are_case_insensitive(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    for name in ("a.png", "b.PNG", "c.jpg", "d.JPEG"):
        image = Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L")
        (root / "normal").mkdir(parents=True, exist_ok=True)
        image.save(root / "normal" / name, format="PNG" if name.lower().endswith("png") else "JPEG")
    save_png(root / "tumor" / "e.png")
    records = scan_image_folder(root, CLASSES).records
    assert sorted(records["image_path"]) == ["normal/a.png", "normal/b.PNG", "normal/c.jpg", "normal/d.JPEG",
                                             "tumor/e.png"]


def test_unsupported_and_junk_files_are_skipped_and_reported(data_root: Path) -> None:
    (data_root / "normal" / "notes.txt").write_text("x", encoding="utf-8")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(data_root / "tumor" / "scan.bmp")
    (data_root / "tumor" / "Thumbs.db").write_bytes(b"junk")
    scan = scan_image_folder(data_root, CLASSES)
    assert scan.unsupported_files == ["normal/notes.txt", "tumor/scan.bmp"]
    assert not scan.records["image_path"].str.contains("Thumbs").any()
    assert scan.ignored_top_level_files == ["README.txt"]


def test_missing_root_or_class_folder_is_an_error(tmp_path: Path, data_root: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        scan_image_folder(tmp_path / "nope", CLASSES)
    shutil.rmtree(data_root / "tumor")
    with pytest.raises(DatasetError, match=r"missing.*tumor"):
        scan_image_folder(data_root, CLASSES)


def test_unexpected_class_folder_is_an_error(data_root: Path) -> None:
    save_png(data_root / "glioma" / "x.png")
    with pytest.raises(DatasetError, match="Unexpected folder"):
        scan_image_folder(data_root, CLASSES)


def test_dataset_without_images_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "ds" / "normal").mkdir(parents=True)
    (tmp_path / "ds" / "tumor").mkdir()
    with pytest.raises(DatasetError, match="No valid"):
        scan_image_folder(tmp_path / "ds", CLASSES)


# ---------------------------------------------------------------------------
# Invalid images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [b"this is not an image", b"", None],  # None -> truncated real PNG
    ids=["garbage", "empty", "truncated"],
)
def test_invalid_images_are_an_error_by_default(data_root: Path, content) -> None:
    bad = data_root / "tumor" / "P0050_s00.png"
    if content is None:
        save_png(bad, size=64)
        bad.write_bytes(bad.read_bytes()[:60])
    else:
        bad.write_bytes(content)
    with pytest.raises(DatasetError, match=r"Invalid image tumor/P0050_s00.png"):
        scan_image_folder(data_root, CLASSES)


def test_invalid_images_can_be_skipped_and_are_recorded(data_root: Path) -> None:
    (data_root / "tumor" / "P0050_s00.png").write_bytes(b"garbage")
    _, meta = prepare(data_root, on_invalid_image="skip")
    assert [x["image_path"] for x in meta["skipped"]["invalid_images"]] == ["tumor/P0050_s00.png"]
    assert meta["counts"]["images"] == 72


# ---------------------------------------------------------------------------
# Patient IDs
# ---------------------------------------------------------------------------


def test_patient_ids_from_filename_regex(data_root: Path) -> None:
    manifest, meta = prepare(data_root)
    assert manifest.loc[manifest["image_path"] == "normal/P0001_s02.png", "patient_id"].item() == "P0001"
    assert meta["counts"]["patients"] == 24
    assert meta["patient_ids"]["available"] is True
    assert meta["limitations"] == []


def test_unmatched_file_name_is_an_error_not_a_guess(data_root: Path) -> None:
    save_png(data_root / "tumor" / "scan_without_id.png")
    with pytest.raises(DatasetError, match="did not match 1 file"):
        prepare(data_root)


def test_patient_ids_from_csv(data_root: Path) -> None:
    records = scan_image_folder(data_root, CLASSES).records
    mapping = pd.DataFrame({"image_path": records["image_path"],
                            "patient_id": "X" + records["image_path"].str.extract(r"P(\d+)_")[0]})
    mapping.to_csv(data_root / "patients.csv", index=False)
    manifest, meta = prepare(data_root, patient_id_source="csv", patient_id_csv="patients.csv",
                             patient_id_regex=None)
    assert manifest["patient_id"].str.startswith("X").all()
    assert meta["counts"]["patients"] == 24


def test_csv_with_missing_or_duplicate_images_is_an_error(data_root: Path) -> None:
    records = scan_image_folder(data_root, CLASSES).records
    pd.DataFrame({"image_path": records["image_path"][1:], "patient_id": "p"}).to_csv(
        data_root / "partial.csv", index=False)
    with pytest.raises(DatasetError, match="have no patient ID"):
        prepare(data_root, patient_id_source="csv", patient_id_csv="partial.csv", patient_id_regex=None)
    doubled = pd.concat([records["image_path"], records["image_path"][:1]])
    pd.DataFrame({"image_path": doubled, "patient_id": "p"}).to_csv(data_root / "dup.csv", index=False)
    with pytest.raises(DatasetError, match="more than once"):
        prepare(data_root, patient_id_source="csv", patient_id_csv="dup.csv", patient_id_regex=None)


def test_missing_patient_ids_are_explicit_null_and_warned(data_root: Path) -> None:
    manifest, meta = prepare(data_root, patient_id_source="none", patient_id_regex=None)
    assert manifest["patient_id"].isna().all()  # no invented IDs
    assert meta["patient_ids"]["available"] is False
    assert meta["counts"]["patients"] is None
    assert meta["limitations"] == [NO_PATIENT_ID_WARNING]
    raw_csv = (Path(meta["dataset_root"]).parent / "prepared" / "manifest.csv").read_text(encoding="utf-8")
    assert ",NA," in raw_csv.splitlines()[1]
    assert read_manifest(Path(meta["dataset_root"]).parent / "prepared" / "manifest.csv")["patient_id"].isna().all()


# ---------------------------------------------------------------------------
# Grouping, splitting and leakage
# ---------------------------------------------------------------------------


def test_zero_patient_leakage_across_many_seeds(tmp_path: Path) -> None:
    """Patients with different slice counts, 30 seeds: no patient may ever span two splits."""
    root = tmp_path / "ds"
    rng = np.random.default_rng(0)
    for p in range(1, 41):
        label = "tumor" if p % 2 else "normal"
        for s in range(int(rng.integers(1, 7))):
            save_png(root / label / f"P{p:04d}_s{s:02d}.png", value=(p * 7 + s * 3) % 250)
    for seed in range(30):
        manifest, _ = prepare(root, split_seed=seed)
        splits_per_patient = manifest.groupby("patient_id")["split"].nunique()
        assert splits_per_patient.max() == 1, f"seed {seed}"
        split_sets = [set(manifest.loc[manifest["split"] == s, "patient_id"]) for s in ("train", "val", "test")]
        assert not (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2])


def test_identical_images_are_kept_in_one_split(data_root: Path) -> None:
    """An exact duplicate under another patient's name must be grouped with the original."""
    shutil.copy(data_root / "normal" / "P0001_s00.png", data_root / "normal" / "P0077_s00.png")
    manifest, meta = prepare(data_root)
    rows = manifest[manifest["patient_id"].isin(["P0001", "P0077"])]
    assert rows["group_id"].nunique() == 1
    assert rows["split"].nunique() == 1
    assert meta["counts"]["images_with_exact_duplicate_content"] == 2


def test_build_groups_links_patients_and_duplicates() -> None:
    records = pd.DataFrame({
        "patient_id": ["A", "A", "B", "C", None, None],
        "sha256": ["h1", "h2", "h2", "h3", "h4", "h3"],
    })
    groups = build_groups(records).tolist()
    assert groups == ["g00000", "g00000", "g00000", "g00001", "g00002", "g00001"]


def test_check_no_leakage_detects_patient_in_two_splits() -> None:
    manifest = pd.DataFrame({
        "patient_id": ["A", "A", "B"], "group_id": ["g1", "g2", "g3"],
        "sha256": ["x", "y", "z"], "split": ["train", "test", "val"],
    })
    with pytest.raises(LeakageError, match="patient_id"):
        check_no_leakage(manifest)


def test_splits_are_stratified_by_label(data_root: Path) -> None:
    manifest, meta = prepare(data_root)
    for split, info in meta["split"]["per_split"].items():
        assert info["class_counts"]["normal"] == info["class_counts"]["tumor"], split


def test_too_few_patients_for_three_splits_is_an_error(tmp_path: Path) -> None:
    root = make_synthetic_dataset(tmp_path / "ds", patients_per_class=1, slices_per_patient=2, image_size=8)
    with pytest.raises(DatasetError, match="received no images"):
        prepare(root)


# ---------------------------------------------------------------------------
# Determinism and reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_manifest(data_root: Path) -> None:
    prepare(data_root, split_seed=3)
    first = sha256_file(data_root.parent / "prepared" / "manifest.csv")
    prepare(data_root, split_seed=3)
    assert sha256_file(data_root.parent / "prepared" / "manifest.csv") == first


def test_different_seed_gives_different_split(data_root: Path) -> None:
    a, _ = prepare(data_root, split_seed=1)
    b, _ = prepare(data_root, split_seed=2)
    assert not a["split"].equals(b["split"])


def test_synthetic_generator_is_deterministic(tmp_path: Path) -> None:
    a = make_synthetic_dataset(tmp_path / "a", 2, 2, 16, seed=5)
    b = make_synthetic_dataset(tmp_path / "b", 2, 2, 16, seed=5)
    files = sorted(p.relative_to(a).as_posix() for p in a.rglob("*.png"))
    assert files == sorted(p.relative_to(b).as_posix() for p in b.rglob("*.png"))
    assert all(sha256_file(a / f) == sha256_file(b / f) for f in files)


def test_metadata_records_required_fields(data_root: Path) -> None:
    _, meta = prepare(data_root)
    assert meta["dataset"] == {"name": "test", "source": "synthetic", "license": "n/a"}
    assert meta["counts"]["images"] == 72
    assert meta["counts"]["class_counts"] == {"normal": 36, "tumor": 36}
    assert meta["split"]["seed"] == 0
    assert set(meta["split"]["per_split"]) == {"train", "val", "test"}
    assert sum(s["images"] for s in meta["split"]["per_split"].values()) == 72
    assert meta["split"]["leakage_check"]["passed"] is True
    assert meta["files"]["manifest"]["sha256"] == sha256_file(data_root.parent / "prepared" / "manifest.csv")
    assert meta["preprocessing"]["normalization_source"] == "train split only"


def test_existing_prepared_dataset_is_not_overwritten(data_root: Path) -> None:
    config = make_config(data_root)
    prepare_dataset(config, PROJECT_ROOT)
    with pytest.raises(FileExistsError):
        prepare_dataset(config, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Preprocessing and the PyTorch dataset
# ---------------------------------------------------------------------------


def test_normalization_statistics_use_only_the_train_split(data_root: Path) -> None:
    manifest, meta = prepare(data_root)
    test_image = data_root / manifest.loc[manifest["split"] == "test", "image_path"].iloc[0]
    save_png(test_image, value=255)  # change a TEST image drastically
    manifest2, meta2 = prepare(data_root)  # same seed -> same split
    assert manifest2["split"].equals(manifest["split"])
    assert meta2["preprocessing"]["normalize_mean"] == meta["preprocessing"]["normalize_mean"]


def test_transform_shapes_and_normalization(data_root: Path) -> None:
    path = next((data_root / "normal").glob("*.png"))
    from flyconnectome_mri.datasets.transforms import load_image

    gray = build_transform(32)(load_image(path, grayscale=True))
    rgb = build_transform(32)(load_image(path, grayscale=False))
    assert gray.shape == (1, 32, 32) and rgb.shape == (3, 32, 32)
    assert 0.0 <= gray.min() and gray.max() <= 1.0
    mean, std = compute_channel_stats([path], 32, grayscale=True)
    normalized = build_transform(32, mean, std)(load_image(path, grayscale=True))
    assert normalized.mean().item() == pytest.approx(0.0, abs=1e-4)
    assert normalized.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-3)


def test_dataset_returns_tensors_and_labels(data_root: Path) -> None:
    prepare(data_root)
    dataset = MRIDataset.from_prepared(data_root.parent / "prepared", "train")
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor) and image.shape == (1, 16, 16) and image.dtype == torch.float32
    assert label in (0, 1)
    assert sum(len(MRIDataset.from_prepared(data_root.parent / "prepared", s)) for s in ("train", "val", "test")) == 72


def test_dataset_loading_is_deterministic(data_root: Path) -> None:
    prepare(data_root)
    a = MRIDataset.from_prepared(data_root.parent / "prepared", "val")
    b = MRIDataset.from_prepared(data_root.parent / "prepared", "val")
    assert all(torch.equal(a[i][0], b[i][0]) and a[i][1] == b[i][1] for i in range(len(a)))


def test_missing_image_file_is_detected(data_root: Path) -> None:
    manifest, _ = prepare(data_root)
    (data_root / manifest.loc[manifest["split"] == "train", "image_path"].iloc[0]).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        MRIDataset.from_prepared(data_root.parent / "prepared", "train")


def test_changed_image_is_detected_with_checksum_verification(data_root: Path) -> None:
    manifest, _ = prepare(data_root)
    save_png(data_root / manifest.loc[manifest["split"] == "val", "image_path"].iloc[0], value=1)
    MRIDataset.from_prepared(data_root.parent / "prepared", "val")  # fine without verification
    with pytest.raises(RuntimeError, match="changed since the manifest"):
        MRIDataset.from_prepared(data_root.parent / "prepared", "val", verify_checksums=True)


def test_dataset_refuses_a_leaking_manifest(data_root: Path) -> None:
    manifest, _ = prepare(data_root)
    leaking = manifest.copy()
    leaking.loc[leaking.index[0], "split"] = "test" if leaking.loc[leaking.index[0], "split"] != "test" else "train"
    with pytest.raises(LeakageError):
        MRIDataset(leaking, data_root, "train", build_transform(16))


def test_invalid_split_name_is_rejected(data_root: Path) -> None:
    prepare(data_root)
    with pytest.raises(ValueError, match="split must be"):
        MRIDataset.from_prepared(data_root.parent / "prepared", "validation")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"classes": {"normal": 0, "tumor": 1, "other": 2}},
        {"classes": {"normal": 0, "tumor": 0}},
        {"positive_class": "normal"},
        {"train_fraction": 0.8},                       # fractions no longer sum to 1
        {"val_fraction": 0},
        {"patient_id_regex": r"^(P\d+)"},              # no named group
        {"patient_id_source": "csv", "patient_id_csv": None},
        {"patient_id_source": "guess"},
        {"on_invalid_image": "ignore"},
        {"license": ""},
    ],
)
def test_invalid_dataset_configs_are_rejected(tmp_path: Path, overrides: dict) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, **overrides)


def test_shipped_dataset_configs_load() -> None:
    demo = load_dataset_config(PROJECT_ROOT / "configs" / "datasets" / "synthetic_demo.yaml")
    assert demo.patient_id_source == "filename_regex"
    template = load_dataset_config(PROJECT_ROOT / "configs" / "datasets" / "TEMPLATE_real_dataset.yaml")
    assert template.patient_id_source == "none"
