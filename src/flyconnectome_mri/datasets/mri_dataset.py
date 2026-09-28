"""PyTorch Dataset that reads one split of a prepared manifest.

Usage::

    train_set = MRIDataset.from_prepared("data/splits/synthetic_demo", split="train")
    image, label = train_set[0]      # image: float tensor (C, H, W); label: 0 or 1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from flyconnectome_mri.datasets.splits import SPLITS, check_no_leakage
from flyconnectome_mri.datasets.transforms import build_transform, load_image
from flyconnectome_mri.utils.provenance import PROJECT_ROOT, sha256_file

MANIFEST_NAME = "manifest.csv"
METADATA_NAME = "dataset_metadata.json"


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest CSV. Missing patient IDs (written as "NA") become None."""
    manifest = pd.read_csv(path, dtype={"patient_id": str, "image_path": str, "sha256": str})
    manifest["patient_id"] = manifest["patient_id"].astype(object).where(manifest["patient_id"].notna(), None)
    return manifest


class MRIDataset(Dataset):
    """Images and binary labels (0/1) of one split.

    Args:
        manifest: Manifest table (see ``prepare_dataset``).
        dataset_root: Folder that ``image_path`` values are relative to.
        split: "train", "val" or "test".
        transform: Callable applied to each PIL image (e.g. ``build_transform``).
        grayscale: Load images as 1 channel (True) or RGB (False).
        verify_checksums: Re-hash every file on construction and fail if any
            changed since the manifest was made (slower; off by default).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_root: str | Path,
        split: str,
        transform: Callable,
        grayscale: bool = True,
        verify_checksums: bool = False,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        check_no_leakage(manifest)  # refuse to serve data from a leaking manifest
        self.root = Path(dataset_root)
        self.records = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.records.empty:
            raise ValueError(f"split {split!r} is empty")
        self.transform = transform
        self.grayscale = grayscale

        missing = [p for p in self.records["image_path"] if not (self.root / p).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image(s) listed in the manifest are missing, e.g. {missing[:5]}")
        if verify_checksums:
            changed = [p for p, h in zip(self.records["image_path"], self.records["sha256"])
                       if sha256_file(self.root / p) != h]
            if changed:
                raise RuntimeError(f"{len(changed)} image(s) changed since the manifest was made, e.g. {changed[:5]}")

    @classmethod
    def from_prepared(cls, prepared_dir: str | Path, split: str, verify_checksums: bool = False) -> "MRIDataset":
        """Build the dataset from ``prepare_dataset`` output, using the TRAIN-split normalization."""
        prepared_dir = Path(prepared_dir)
        if not prepared_dir.is_absolute():
            prepared_dir = PROJECT_ROOT / prepared_dir
        metadata = json.loads((prepared_dir / METADATA_NAME).read_text(encoding="utf-8"))
        pre = metadata["preprocessing"]
        transform = build_transform(pre["image_size"], pre["normalize_mean"], pre["normalize_std"])
        dataset_root = Path(metadata["dataset_root"])
        if not dataset_root.is_absolute():
            dataset_root = PROJECT_ROOT / dataset_root
        return cls(
            read_manifest(prepared_dir / MANIFEST_NAME),
            dataset_root=dataset_root,
            split=split,
            transform=transform,
            grayscale=pre["grayscale"],
            verify_checksums=verify_checksums,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.records.iloc[index]
        image = load_image(self.root / row["image_path"], self.grayscale)
        return self.transform(image), int(row["label_index"])

    def class_counts(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.records["label"].value_counts().sort_index().items()}
