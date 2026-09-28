"""Binary classification metrics at slice level and patient level.

Positive class = 1 (glioma), negative = 0 (meningioma).

    sensitivity (= recall) = TP / (TP + FN)   share of gliomas correctly found
    specificity            = TN / (TN + FP)   share of meningiomas correctly found
    precision              = TP / (TP + FP)   share of "glioma" calls that are right
    balanced accuracy      = (sensitivity + specificity) / 2
    F1                     = 2 * precision * recall / (precision + recall)
    ROC AUC                = probability that a random glioma gets a higher score
                             than a random meningioma (threshold-free)

Accuracy is also reported but is NOT a primary metric: with ~67% glioma slices,
always answering "glioma" already scores ~67-70%.

Patient level: a patient's probability is the MEAN of the probabilities of all
of that patient's slices; the same metrics are then computed over patients.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PRIMARY_METRICS = ("roc_auc", "sensitivity", "specificity", "balanced_accuracy", "f1", "precision", "recall")


def _safe_div(numerator: float, denominator: float) -> float | None:
    """Division that returns None (reported as "undefined") instead of dividing by zero."""
    return None if denominator == 0 else numerator / denominator


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Compute all metrics; prediction = 1 if prob >= threshold.

    Returns:
        Dict with the primary metrics, accuracy, the confusion matrix
        ({"tn","fp","fn","tp"} and [[tn, fp], [fn, tp]]), counts and threshold.
        A metric that is undefined for the given data (e.g. AUC with one class)
        is None.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    if y_true.shape != prob.shape or y_true.ndim != 1:
        raise ValueError("y_true and prob must be 1-D arrays of equal length")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("y_true must contain only 0 and 1")
    if np.any((prob < 0) | (prob > 1)) or np.any(~np.isfinite(prob)):
        raise ValueError("prob must be finite and within [0, 1]")

    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = None if precision is None or sensitivity is None or precision + sensitivity == 0 \
        else 2 * precision * sensitivity / (precision + sensitivity)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)) if both_classes else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": None if sensitivity is None or specificity is None
        else (sensitivity + specificity) / 2,
        "f1": f1,
        "precision": precision,
        "recall": sensitivity,
        "accuracy": (tp + tn) / len(y_true),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "confusion_matrix_rows_true_cols_pred": [[tn, fp], [fn, tp]],
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "threshold": threshold,
    }


def aggregate_by_patient(slices: pd.DataFrame) -> pd.DataFrame:
    """Average slice probabilities per patient.

    Args:
        slices: Columns patient_id, label_index (true label), prob.

    Returns:
        One row per patient: patient_id, label_index, prob (mean over slices), n_slices.

    Raises:
        ValueError: If a patient's slices have different true labels.
    """
    labels_per_patient = slices.groupby("patient_id")["label_index"].nunique()
    mixed = labels_per_patient[labels_per_patient > 1]
    if len(mixed):
        raise ValueError(f"patients with mixed labels cannot be aggregated: {list(mixed.index[:5])}")
    grouped = slices.groupby("patient_id", sort=True)
    return pd.DataFrame({
        "label_index": grouped["label_index"].first(),
        "prob": grouped["prob"].mean(),
        "n_slices": grouped.size(),
    }).reset_index()


def format_metrics(metrics: dict[str, Any]) -> str:
    """Readable multi-line summary: primary metrics first, then accuracy and the confusion matrix."""
    def fmt(value: float | None) -> str:
        return "undefined" if value is None else f"{value:.4f}"

    lines = [f"  {name:<18} {fmt(metrics[name])}" for name in PRIMARY_METRICS]
    lines.append(f"  {'accuracy':<18} {fmt(metrics['accuracy'])}   (secondary: class-imbalanced)")
    cm = metrics["confusion_matrix"]
    lines += [
        "  confusion matrix (rows = true, columns = predicted):",
        f"                     pred meningioma  pred glioma",
        f"    true meningioma  {cm['tn']:>15}  {cm['fp']:>11}",
        f"    true glioma      {cm['fn']:>15}  {cm['tp']:>11}",
        f"  n = {metrics['n']} ({metrics['n_positive']} glioma, {metrics['n_negative']} meningioma), "
        f"threshold {metrics['threshold']}",
    ]
    return "\n".join(lines)
