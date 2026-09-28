"""Scan an image-folder dataset into a manifest: one row per image.

Expected layout::

    <root>/
        normal/   (any sub-folders allowed)  *.png *.jpg *.jpeg
        tumor/    ...

The label comes ONLY from the top-level class folder. Every image is fully
decoded once, so corrupt or truncated files are found now, not in the middle of
training. Each image's SHA-256 is recorded (for provenance and to detect exact
duplicates, a common hidden leakage source in public MRI datasets).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from flyconnectome_mri.utils.config import DatasetConfig
from flyconnectome_mri.utils.provenance import sha256_file

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")
# Operating-system junk files that are skipped silently.
IGNORED_FILE_NAMES = {"thumbs.db", ".ds_store", "desktop.ini"}


class DatasetError(ValueError):
    """Raised when the dataset folder or its patient-ID information is invalid."""


@dataclass
class ScanResult:
    """Output of :func:`scan_image_folder`.

    Attributes:
        records: One row per valid image: image_path (relative to root, "/"-separated),
            label, label_index, sha256, width, height, mode.
        unsupported_files: Files skipped because of their extension.
        invalid_images: (image_path, reason) for unreadable images skipped with
            ``on_invalid_image="skip"``.
        ignored_top_level_files: Files directly in the root (e.g. a README).
    """

    records: pd.DataFrame
    unsupported_files: list[str] = field(default_factory=list)
    invalid_images: list[tuple[str, str]] = field(default_factory=list)
    ignored_top_level_files: list[str] = field(default_factory=list)


def scan_image_folder(root: Path, classes: dict[str, int], on_invalid_image: str = "error") -> ScanResult:
    """Find, validate and describe every image under ``root/<class folder>/``.

    Raises:
        DatasetError: Missing root or class folder, unexpected top-level folder,
            no images, or an invalid image with ``on_invalid_image="error"``.
    """
    if not root.is_dir():
        raise DatasetError(f"Dataset folder not found: {root}")
    top_dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    unexpected = [d for d in top_dirs if d not in classes]
    if unexpected:
        raise DatasetError(f"Unexpected folder(s) in {root}: {unexpected}. Expected only {sorted(classes)}.")
    missing = [c for c in classes if c not in top_dirs]
    if missing:
        raise DatasetError(f"Class folder(s) missing in {root}: {missing}")

    result = ScanResult(records=pd.DataFrame())
    result.ignored_top_level_files = sorted(p.name for p in root.iterdir() if p.is_file())
    rows = []
    for class_name in sorted(classes):
        for path in sorted((root / class_name).rglob("*")):  # sorted: deterministic order
            if not path.is_file() or path.name.lower() in IGNORED_FILE_NAMES:
                continue
            rel = path.relative_to(root).as_posix()
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                result.unsupported_files.append(rel)
                continue
            info = _inspect_image(path)
            if isinstance(info, str):  # error message
                if on_invalid_image == "error":
                    raise DatasetError(f"Invalid image {rel}: {info}. Fix/remove it or set on_invalid_image: skip")
                result.invalid_images.append((rel, info))
                continue
            width, height, mode = info
            rows.append({
                "image_path": rel,
                "label": class_name,
                "label_index": classes[class_name],
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
                "mode": mode,
            })
    if not rows:
        raise DatasetError(f"No valid {'/'.join(SUPPORTED_EXTENSIONS)} images found in {root}")
    result.records = pd.DataFrame(rows)
    return result


def _inspect_image(path: Path) -> tuple[int, int, str] | str:
    """Fully decode an image; return (width, height, mode) or an error message."""
    try:
        with Image.open(path) as image:
            image.load()  # forces a full decode: catches truncated files that verify() misses
            return image.width, image.height, image.mode
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return f"{type(exc).__name__}: {exc}"


def assign_patient_ids(records: pd.DataFrame, config: DatasetConfig, root: Path) -> pd.Series:
    """Return one patient ID per image, or None everywhere when IDs are unavailable.

    IDs are never invented: a file whose patient cannot be determined is an error
    (with "filename_regex" or "csv"), not a guess.
    """
    if config.patient_id_source == "none":
        return pd.Series([None] * len(records), index=records.index, dtype=object)

    if config.patient_id_source == "filename_regex":
        pattern = re.compile(config.patient_id_regex)
        ids, unmatched = [], []
        for rel in records["image_path"]:
            match = pattern.search(Path(rel).name)
            if match is None or not match.group("patient"):
                unmatched.append(rel)
                ids.append(None)
            else:
                ids.append(match.group("patient"))
        if unmatched:
            raise DatasetError(
                f"patient_id_regex did not match {len(unmatched)} file(s), e.g. {unmatched[:5]}"
            )
        return pd.Series(ids, index=records.index, dtype=object)

    # config.patient_id_source == "csv"
    csv_path = Path(config.patient_id_csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.is_file():
        raise DatasetError(f"Patient-ID CSV not found: {csv_path}")
    image_col, patient_col = config.patient_id_csv_image_column, config.patient_id_csv_patient_column
    mapping = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for col in (image_col, patient_col):
        if col not in mapping.columns:
            raise DatasetError(f"Patient-ID CSV {csv_path.name} has no column {col!r}")
    mapping[image_col] = mapping[image_col].str.replace("\\", "/", regex=False).str.strip()
    mapping[patient_col] = mapping[patient_col].str.strip()
    duplicated = mapping[image_col][mapping[image_col].duplicated()].tolist()
    if duplicated:
        raise DatasetError(f"Patient-ID CSV lists some images more than once, e.g. {duplicated[:5]}")
    lookup = dict(zip(mapping[image_col], mapping[patient_col]))
    ids = records["image_path"].map(lookup)
    missing = records.loc[ids.isna() | (ids == ""), "image_path"].tolist()
    if missing:
        raise DatasetError(f"{len(missing)} image(s) have no patient ID in {csv_path.name}, e.g. {missing[:5]}")
    return ids.astype(object)
