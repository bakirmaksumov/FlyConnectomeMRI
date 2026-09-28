"""Run a model over a data loader and collect probabilities (plus loss and timing)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class Predictions:
    """Model outputs for every example of a loader, in loader order.

    Attributes:
        labels: True labels (0/1).
        probs: sigmoid(logit) = predicted probability of class 1.
        loss: Mean loss per example (NaN if no loss function was given).
        seconds: Wall-clock time of the model forward passes only (data loading
            and preprocessing are excluded).
    """

    labels: np.ndarray
    probs: np.ndarray
    loss: float
    seconds: float


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module | None = None) -> Predictions:
    """Evaluate ``model`` on ``loader`` (eval mode, no gradients).

    ``seconds`` measures only the model forward passes (and loss), not data
    loading: with cached tensors this is the model's inference cost.
    """
    model.eval()
    labels, probs = [], []
    total_loss, count, seconds = 0.0, 0, 0.0
    for images, batch_labels in loader:
        images = images.to(device)
        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - start
        if loss_fn is not None:
            total_loss += loss_fn(logits, batch_labels.float().to(device)).item() * len(batch_labels)
        count += len(batch_labels)
        labels.append(batch_labels.numpy())
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return Predictions(
        labels=np.concatenate(labels).astype(int),
        probs=np.concatenate(probs).astype(float),
        loss=total_loss / count if loss_fn is not None else float("nan"),
        seconds=seconds,
    )


@torch.no_grad()
def collect_diagnostics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Run ``model.diagnostics`` on ALL examples of ``loader`` at once (eval mode)."""
    model.eval()
    images = torch.cat([batch for batch, _ in loader]).to(device)
    return model.diagnostics(images)
