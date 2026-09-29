"""Leakage-safe cross-validation using the official Figshare outer test folds.

This is NOT an exact reproduction of the dataset authors' five-fold protocol.

For outer fold k (official ``cv_fold`` values are never changed):
    test       = every patient of official fold k (unchanged, never reduced)
    validation = official fold VAL(k) = k % 5 + 1  (1->2, 2->3, 3->4, 4->5, 5->1),
                 minus any patient whose patient_link_key also occurs in test
    training   = the remaining three official folds,
                 minus any patient whose patient_link_key occurs in test or in
                 the cleaned validation set

Why: the official folds were built from the published patient_id, while our
conservative patient_link_key grouping (a leakage-prevention ASSUMPTION, not a
known fact) treats some letter-suffixed IDs as possibly the same person. Such
linked examinations can span official folds; they are therefore excluded from
the corresponding training/validation sets. No patient is moved to another
fold and no replacement patients are added. Every patient is still in exactly
one outer TEST fold, so pooled out-of-fold predictions cover the whole cohort.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from flyconnectome_mri.datasets.splits import check_no_leakage

N_FOLDS = 5


def validation_fold(test_fold: int, n_folds: int = N_FOLDS) -> int:
    """Pre-declared cyclic rule: the validation fold is the next official fold."""
    if not 1 <= test_fold <= n_folds:
        raise ValueError(f"test_fold must be in 1..{n_folds}, got {test_fold}")
    return test_fold % n_folds + 1


@dataclass
class OuterFold:
    """Result of :func:`build_outer_fold`.

    Attributes:
        manifest: Slice manifest restricted to the retained train/val/test
            patients, with ``split`` reassigned; ``cv_fold`` unchanged.
        exclusions: One row per removed patient (outer_fold, removed_patient_id,
            patient_link_key, original_cv_fold, removed_from, reason).
        counts: Before/after patient, link-key and slice counts per split and class.
    """

    manifest: pd.DataFrame
    exclusions: pd.DataFrame
    counts: pd.DataFrame


def build_outer_fold(manifest: pd.DataFrame, test_fold: int, n_folds: int = N_FOLDS) -> OuterFold:
    """Assign official folds to train/val/test for one outer fold, excluding linked relatives."""
    required = {"patient_id", "patient_link_key", "cv_fold", "label"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest lacks column(s) {sorted(missing)}")
    val_fold = validation_fold(test_fold, n_folds)
    patients = manifest.drop_duplicates("patient_id")[["patient_id", "patient_link_key", "cv_fold", "label"]]
    if (manifest.groupby("patient_id")["cv_fold"].nunique() > 1).any():
        raise ValueError("a patient_id spans several official folds")

    test = patients[patients["cv_fold"] == test_fold]
    val_all = patients[patients["cv_fold"] == val_fold]
    train_all = patients[~patients["cv_fold"].isin([test_fold, val_fold])]

    test_keys = set(test["patient_link_key"])
    val_removed = val_all[val_all["patient_link_key"].isin(test_keys)]
    val = val_all[~val_all["patient_link_key"].isin(test_keys)]
    val_keys = set(val["patient_link_key"])
    train_removed = train_all[train_all["patient_link_key"].isin(test_keys | val_keys)]
    train = train_all[~train_all["patient_link_key"].isin(test_keys | val_keys)]

    exclusions = pd.concat([
        _exclusion_rows(val_removed, test_fold, "validation", lambda key: "linked_to_test"),
        _exclusion_rows(train_removed, test_fold, "train",
                        lambda key: "linked_to_test" if key in test_keys else "linked_to_validation"),
    ], ignore_index=True)

    split_of = {**{p: "train" for p in train["patient_id"]}, **{p: "val" for p in val["patient_id"]},
                **{p: "test" for p in test["patient_id"]}}
    fold_manifest = manifest[manifest["patient_id"].isin(split_of)].copy()
    fold_manifest["split"] = fold_manifest["patient_id"].map(split_of)
    check_no_leakage(fold_manifest)  # patient_id, patient_link_key, group_id, sha256
    counts = _counts(manifest, test_fold, {"train": train_all, "val": val_all, "test": test},
                     {"train": train, "val": val, "test": test})
    return OuterFold(fold_manifest.reset_index(drop=True), exclusions, counts)


def _exclusion_rows(removed: pd.DataFrame, test_fold: int, removed_from: str, reason) -> pd.DataFrame:
    return pd.DataFrame({
        "outer_fold": test_fold,
        "removed_patient_id": removed["patient_id"].to_numpy(),
        "patient_link_key": removed["patient_link_key"].to_numpy(),
        "original_cv_fold": removed["cv_fold"].to_numpy(),
        "label": removed["label"].to_numpy(),
        "removed_from": removed_from,
        "reason": [reason(k) for k in removed["patient_link_key"]],
    })


def _counts(manifest: pd.DataFrame, test_fold: int, before: dict, after: dict) -> pd.DataFrame:
    """Original / retained / removed counts per split, overall and per class."""
    rows = []
    for split in ("train", "val", "test"):
        for label in (None, *sorted(manifest["label"].unique())):
            def count(frame: pd.DataFrame) -> dict[str, int]:
                part = frame if label is None else frame[frame["label"] == label]
                slices = manifest[manifest["patient_id"].isin(part["patient_id"])]
                return {"patients": len(part), "link_keys": part["patient_link_key"].nunique(), "slices": len(slices)}
            original, retained = count(before[split]), count(after[split])
            rows.append({"outer_fold": test_fold, "split": split, "class": label or "all",
                         **{f"original_{k}": v for k, v in original.items()},
                         **{f"retained_{k}": v for k, v in retained.items()},
                         **{f"removed_{k}": original[k] - retained[k] for k in original}})
    return pd.DataFrame(rows)


def check_oof_coverage(manifest: pd.DataFrame, fold_manifests: dict[int, pd.DataFrame]) -> dict[str, int]:
    """Every cohort patient must be in exactly one outer TEST set (raises otherwise)."""
    cohort = set(manifest["patient_id"])
    test_count = pd.Series(0, index=sorted(cohort))
    for fold_manifest in fold_manifests.values():
        tested = fold_manifest.loc[fold_manifest["split"] == "test", "patient_id"].unique()
        test_count[tested] += 1
    zero, many = int((test_count == 0).sum()), int((test_count > 1).sum())
    if zero or many:
        raise ValueError(f"OOF coverage broken: {zero} patients never tested, {many} tested more than once")
    return {"patients": len(cohort), "tested_exactly_once": int((test_count == 1).sum()),
            "tested_zero_times": zero, "tested_more_than_once": many}
