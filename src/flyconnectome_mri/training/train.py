"""Training loop with validation-based checkpoint selection and early stopping.

The test set is never touched here. Validation data is used only to pick the
best epoch (``selection_metric``) and to decide when to stop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from flyconnectome_mri.training.evaluate import collect_diagnostics, predict
from flyconnectome_mri.training.metrics import binary_metrics
from flyconnectome_mri.utils.config import TrainingConfig
from flyconnectome_mri.utils.logging import get_logger


def compute_pos_weight(train_labels: np.ndarray) -> float:
    """pos_weight for BCEWithLogitsLoss = #negatives / #positives, from TRAIN labels only.

    Each positive (glioma) example's loss is multiplied by this value. With more
    positives than negatives it is < 1, so both classes contribute equally in
    total: #positives x pos_weight = #negatives.
    """
    labels = np.asarray(train_labels).astype(int)
    positives, negatives = int(labels.sum()), int((1 - labels).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("training labels must contain both classes to compute pos_weight")
    return negatives / positives


@dataclass
class FitResult:
    """Outcome of :func:`fit`."""

    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = 0
    best_score: float = float("nan")
    stopped_early: bool = False
    training_seconds: float = 0.0
    failure: str | None = None  # set if training produced non-finite values


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module, device: torch.device) -> float:
    """One pass over the training data; returns the mean loss per example."""
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


def _format_diagnostics(diagnostics: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4g}" for k, v in diagnostics.items())


def _is_better(score: float, best: float, metric: str) -> bool:
    if np.isnan(best):
        return True
    return score > best if metric == "val_roc_auc" else score < best


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_path: Path,
) -> FitResult:
    """Train, evaluate on validation after every epoch, keep the best checkpoint.

    The best epoch is the one with the highest validation ROC AUC (or lowest
    validation loss). Ties keep the EARLIER epoch. Training stops after
    ``early_stopping_patience`` epochs without improvement.
    """
    logger = get_logger()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    result = FitResult()
    start = time.perf_counter()
    since_best = 0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val = predict(model, val_loader, device, loss_fn)
        if not (np.isfinite(train_loss) and np.isfinite(val.loss) and np.all(np.isfinite(val.probs))):
            result.failure = f"non-finite loss or predictions at epoch {epoch}"
            result.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val.loss,
                                   "epoch_seconds": time.perf_counter() - epoch_start, "is_best": False})
            logger.error("epoch %2d | %s: stopping", epoch, result.failure)
            break
        val_metrics = binary_metrics(val.labels, val.probs, config.decision_threshold)
        score = val_metrics["roc_auc"] if config.selection_metric == "val_roc_auc" else val.loss
        improved = _is_better(score, result.best_score, config.selection_metric)
        if improved:
            result.best_score, result.best_epoch, since_best = score, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "selection_metric": config.selection_metric, "score": score}, checkpoint_path)
        else:
            since_best += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val.loss,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "is_best": improved,
        }
        diagnostics = collect_diagnostics(model, val_loader, device)
        row.update({f"diag_{k}": v for k, v in diagnostics.items()})
        result.history.append(row)
        logger.info(
            "epoch %2d | train loss %.4f | val loss %.4f | val AUC %.4f | val bal.acc %.4f%s",
            epoch, train_loss, val.loss, val_metrics["roc_auc"], val_metrics["balanced_accuracy"],
            "  <- best" if improved else "",
        )
        logger.info("         diagnostics (val): %s", _format_diagnostics(diagnostics))
        if since_best >= config.early_stopping_patience:
            result.stopped_early = True
            logger.info("Early stopping: no improvement in %s for %d epochs",
                        config.selection_metric, config.early_stopping_patience)
            break
    result.training_seconds = time.perf_counter() - start
    return result
