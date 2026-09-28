"""Leak-free, deterministic train / validation / test splitting.

Key idea: we never split individual images. We split GROUPS of images that must
stay together, then every image follows its group.

Two images belong to the same group if
  - they have the same patient ID (slices of one patient), or
  - they have identical file content (same SHA-256): public MRI collections
    often contain exact duplicates, which would otherwise leak across splits.
Groups are connected components of these links ("union-find").

Without patient IDs only the duplicate rule applies, so different slices of the
same (unknown) patient CAN still end up in different splits. That limitation is
reported loudly and written into the dataset metadata.

Splitting is stratified by label: groups are split separately per label (a
group whose images have mixed labels forms its own "mixed" stratum), so each
split gets roughly the same class balance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


class LeakageError(RuntimeError):
    """Raised when a patient, group or identical image appears in more than one split."""


def build_groups(records: pd.DataFrame) -> pd.Series:
    """Return a group ID ("g00000", ...) per image; see module docstring for the rules.

    Group IDs are numbered in order of each group's first image in ``records``,
    so they are deterministic for a deterministic ``records`` order.
    """
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)

    for column in ("patient_id", "sha256"):
        first_seen: dict[str, int] = {}
        for position, key in enumerate(records[column].tolist()):
            if key is None or (isinstance(key, float) and np.isnan(key)):
                continue  # missing patient ID links nothing
            if key in first_seen:
                union(position, first_seen[key])
            else:
                first_seen[key] = position

    roots = [find(i) for i in range(len(records))]
    numbering: dict[int, str] = {}
    for root in roots:
        numbering.setdefault(root, f"g{len(numbering):05d}")
    return pd.Series([numbering[r] for r in roots], index=records.index)


def split_groups(
    records: pd.DataFrame, fractions: tuple[float, float, float], seed: int
) -> pd.Series:
    """Assign every image to "train", "val" or "test" by assigning whole groups.

    Per stratum (label, or "mixed"): shuffle the group IDs with the seed, then
    walk through them filling train until it holds its target share of the
    stratum's images, then val, and the remainder goes to test.

    Args:
        records: Must contain ``group_id`` and ``label`` columns.
        fractions: (train, val, test), summing to 1.
        seed: Split seed; the same seed and records always give the same split.
    """
    group_label = records.groupby("group_id")["label"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
    group_size = records.groupby("group_id").size()
    rng = np.random.default_rng(seed)

    assignment: dict[str, str] = {}
    for stratum in sorted(group_label.unique()):
        groups = sorted(group_label.index[group_label == stratum])
        groups = [groups[i] for i in rng.permutation(len(groups))]
        total = int(group_size[groups].sum())
        target_train = fractions[0] * total
        target_val = (fractions[0] + fractions[1]) * total
        filled = 0
        for group in groups:
            # Decide by the share filled BEFORE adding this group.
            split = "train" if filled < target_train else "val" if filled < target_val else "test"
            assignment[group] = split
            filled += int(group_size[group])
    return records["group_id"].map(assignment)


def check_no_leakage(manifest: pd.DataFrame) -> dict[str, int]:
    """Raise :class:`LeakageError` if any patient, group or identical file spans two splits.

    Returns:
        Number of distinct patient IDs, groups and file hashes that were checked.
    """
    checked = {}
    for column in ("patient_id", "group_id", "sha256"):
        values = manifest.dropna(subset=[column])
        splits_per_value = values.groupby(column)["split"].nunique()
        leaking = splits_per_value[splits_per_value > 1]
        if len(leaking):
            raise LeakageError(
                f"{len(leaking)} {column} value(s) appear in more than one split, e.g. {list(leaking.index[:5])}"
            )
        checked[column] = int(values[column].nunique())
    return checked


def split_summary(manifest: pd.DataFrame) -> dict[str, dict]:
    """Images, patients and class counts per split."""
    summary = {}
    for split in SPLITS:
        part = manifest[manifest["split"] == split]
        summary[split] = {
            "images": len(part),
            "patients": int(part["patient_id"].nunique()) if part["patient_id"].notna().any() else None,
            "groups": int(part["group_id"].nunique()),
            "class_counts": {str(k): int(v) for k, v in part["label"].value_counts().sort_index().items()},
        }
    return summary
