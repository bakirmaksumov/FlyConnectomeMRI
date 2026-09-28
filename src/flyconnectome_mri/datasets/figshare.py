"""Read the original Cheng et al. Figshare brain tumor dataset (.mat, MATLAB v7.3 / HDF5).

Dataset: https://doi.org/10.6084/m9.figshare.1512427 (CC BY 4.0).
Per the official README, each ``<n>.mat`` file (n = 1..3064) holds a struct
``cjdata`` with:
    label        1 = meningioma, 2 = glioma, 3 = pituitary tumor
    PID          patient ID (stored as MATLAB char codes)
    image        the MRI slice (int16, usually 512 x 512)
    tumorBorder  manually delineated border points  (not used here yet)
    tumorMask    binary tumor mask                  (not used here yet)

MATLAB stores arrays column-major; h5py returns them transposed, so images are
transposed back (``.T``) to match MATLAB's own orientation.

Intensity scaling follows the dataset authors' conversion code: each image is
min-max scaled by its OWN minimum and maximum (to [0, 1] here). This uses no
information from other images, so it cannot leak across splits.

The original files are never modified: archives are extracted to a separate
folder, extracted files are made read-only, and every file's SHA-256 is recorded.
"""

from __future__ import annotations

import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
from PIL import Image

from flyconnectome_mri.datasets.manifest import DatasetError
from flyconnectome_mri.utils.provenance import sha256_file


@dataclass(frozen=True)
class CjData:
    """Fields of one ``cjdata`` struct (image optional, to allow fast metadata scans)."""

    label_code: int
    patient_id: str
    height: int
    width: int
    dtype: str
    image: np.ndarray | None = None


def read_cjdata(path: str | Path, with_image: bool = True) -> CjData:
    """Read label, patient ID and (optionally) the image from one Figshare ``.mat`` file."""
    try:
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            label = cj["label"][()].ravel()
            pid_codes = cj["PID"][()].ravel()
            image_ds = cj["image"]
            height, width = image_ds.shape[1], image_ds.shape[0]  # transposed on disk
            image = image_ds[()].T if with_image else None
            dtype = str(image_ds.dtype)
    except (OSError, KeyError) as exc:
        raise DatasetError(f"{Path(path).name} is not a readable Figshare cjdata .mat file: {exc}") from exc
    if label.size != 1 or float(label[0]) != int(label[0]):
        raise DatasetError(f"{Path(path).name}: cjdata.label is not a single integer ({label})")
    patient_id = "".join(chr(int(c)) for c in pid_codes).strip()
    if not patient_id:
        raise DatasetError(f"{Path(path).name}: cjdata.PID is empty")
    return CjData(int(label[0]), patient_id, height, width, dtype, image)


def load_mat_image(path: str | Path) -> Image.Image:
    """Return the slice as a PIL float image ("F" mode) min-max scaled to [0, 1]."""
    image = read_cjdata(path).image.astype(np.float32)
    low, high = float(image.min()), float(image.max())
    if high <= low:
        raise DatasetError(f"{Path(path).name}: constant image, cannot min-max scale")
    return Image.fromarray((image - low) / (high - low), mode="F")


def extract_archives(archive_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Extract every ``*.zip`` in ``archive_dir`` into ``output_dir`` (flat), read-only.

    Existing extracted files are verified against the archive instead of being
    rewritten. Returns a table: file, archive, sha256.

    Raises:
        DatasetError: On unsafe member paths, duplicate names across archives,
            or an existing file whose content differs from the archive.
    """
    archives = sorted(archive_dir.glob("*.zip"))
    if not archives:
        raise DatasetError(f"No .zip archives in {archive_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, seen = [], {}
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if Path(info.filename).is_absolute() or ".." in Path(info.filename).parts or not name:
                    raise DatasetError(f"Unsafe path {info.filename!r} in {archive.name}")
                if name in seen:
                    raise DatasetError(f"{name} appears in both {seen[name]} and {archive.name}")
                seen[name] = archive.name
                target = output_dir / name
                data = z.read(info)
                if target.exists():
                    if target.read_bytes() != data:
                        raise DatasetError(f"Extracted {name} differs from {archive.name}; refusing to continue")
                else:
                    target.write_bytes(data)
                    os.chmod(target, stat.S_IREAD)  # read-only: protects the original data
                rows.append({"file": name, "archive": archive.name, "sha256": sha256_file(target)})
    return pd.DataFrame(rows).sort_values("file", key=lambda s: s.map(_file_number)).reset_index(drop=True)


def _file_number(name: str) -> int:
    """'123.mat' -> 123 (the global image index used by cvind.mat)."""
    stem = Path(name).stem
    if not stem.isdigit():
        raise DatasetError(f"Unexpected Figshare file name {name!r} (expected <number>.mat)")
    return int(stem)


def scan_figshare(root: Path, label_codes: dict[int, str], classes: dict[str, int]) -> tuple[pd.DataFrame, dict]:
    """Read every ``<n>.mat`` under ``root``; keep only classes listed in ``classes``.

    Returns:
        (records, exclusions): records has one row per KEPT slice with columns
        image_path, source_filename, image_number, label, label_index,
        original_label_code, patient_id, sha256, width, height, mode.
        exclusions counts all slices and patients per original class, and which
        classes were excluded.
    """
    files = sorted(root.rglob("*.mat"), key=lambda p: _file_number(p.name))
    files = [f for f in files if f.name.lower() != "cvind.mat"]
    if not files:
        raise DatasetError(f"No .mat files found under {root}")
    all_rows = []
    for path in files:
        cj = read_cjdata(path, with_image=False)
        if cj.label_code not in label_codes:
            raise DatasetError(f"{path.name}: unknown label code {cj.label_code}; known {sorted(label_codes)}")
        all_rows.append({
            "image_path": path.relative_to(root).as_posix(),
            "source_filename": path.name,
            "image_number": _file_number(path.name),
            "label": label_codes[cj.label_code],
            "original_label_code": cj.label_code,
            "patient_id": cj.patient_id,
            "width": cj.width,
            "height": cj.height,
            "mode": cj.dtype,
        })
    everything = pd.DataFrame(all_rows)
    per_class = {
        name: {"slices": int((everything["label"] == name).sum()),
               "patients": int(everything.loc[everything["label"] == name, "patient_id"].nunique())}
        for name in label_codes.values()
    }
    exclusions = {
        "all_slices": len(everything),
        "all_patients": int(everything["patient_id"].nunique()),
        "per_original_class": per_class,
        "excluded_classes": sorted(set(label_codes.values()) - set(classes)),
        "excluded_slices": int((~everything["label"].isin(list(classes))).sum()),
    }
    kept = everything[everything["label"].isin(list(classes))].copy()
    kept["label_index"] = kept["label"].map(classes)
    kept["sha256"] = [sha256_file(root / p) for p in kept["image_path"]]
    return kept.reset_index(drop=True), exclusions


def patients_with_multiple_labels(records: pd.DataFrame) -> list[str]:
    """Patient IDs whose slices carry more than one label (should be none)."""
    counts = records.groupby("patient_id")["label"].nunique()
    return sorted(counts[counts > 1].index)


def load_cv_folds(cvind_path: Path, num_images: int) -> np.ndarray:
    """Return the official fold number (1..5) for image numbers 1..num_images.

    ``cvind.mat`` stores one fold index per image, in image-number order.
    Handles both MATLAB v5 (scipy) and v7.3 (HDF5) files.
    """
    try:
        values = next(v for k, v in scipy.io.loadmat(cvind_path).items() if not k.startswith("__"))
    except NotImplementedError:  # v7.3 file
        with h5py.File(cvind_path, "r") as f:
            values = f[next(iter(f.keys()))][()]
    folds = np.asarray(values).ravel().astype(np.int64)
    if len(folds) != num_images:
        raise DatasetError(f"cvind has {len(folds)} entries, expected {num_images}")
    return folds


def map_folds_to_patients(records: pd.DataFrame, folds: np.ndarray) -> tuple[pd.Series, list[str]]:
    """Attach each slice's official fold; list patients whose slices span several folds.

    Returns:
        (fold per record, patients split across folds). The folds "map cleanly"
        to patients only if the second list is empty.
    """
    fold = records["image_number"].map(lambda n: int(folds[n - 1]))
    per_patient = fold.groupby(records["patient_id"]).nunique()
    return fold, sorted(per_patient[per_patient > 1].index)
