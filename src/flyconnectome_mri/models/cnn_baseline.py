"""Conventional CNN baseline: encoder -> 128-dim features -> one linear layer -> 1 logit.

Output is a single logit per image. sigmoid(logit) = predicted probability of
the positive class (glioma = 1). No graph, no pretrained weights.
"""

from __future__ import annotations

import torch
from torch import nn

from flyconnectome_mri.models.encoder import CNNEncoder


class CNNBaseline(nn.Module):
    """Small CNN for binary classification (returns logits of shape (B,))."""

    def __init__(self, in_channels: int = 1, channels: tuple[int, int] = (16, 32), feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels, channels, feature_dim)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x)).squeeze(-1)

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> dict[str, float]:
        """Across-image std of encoder features and logits (see GraphClassifier.diagnostics)."""
        features = self.encoder(x)
        return {
            "encoder_feature_std": float(features.std(dim=0).mean()),
            "logit_std": float(self.classifier(features).squeeze(-1).std()),
        }


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) scalar parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
