"""Small convolutional image encoder: 1 x 128 x 128 MRI slice -> 128-dim feature vector.

The same encoder is used by the CNN baseline (Phase 6) and, later, by the
connectome-constrained models (Phase 7), so any difference between them comes
from what happens AFTER the feature vector.

Architecture (default channels 16, 32, 128), shapes for a 128 x 128 input:

    input                       (B,   1, 128, 128)
    Conv 3x3 (1->16)  + ReLU    (B,  16, 128, 128)
    MaxPool 2x2                 (B,  16,  64,  64)
    Conv 3x3 (16->32) + ReLU    (B,  32,  64,  64)
    MaxPool 2x2                 (B,  32,  32,  32)
    Conv 3x3 (32->128) + ReLU   (B, 128,  32,  32)
    AdaptiveAvgPool -> 1 x 1    (B, 128,   1,   1)
    Flatten                     (B, 128)          <- feature vector

No batch normalisation or dropout: deliberately minimal.
"""

from __future__ import annotations

import torch
from torch import nn


class CNNEncoder(nn.Module):
    """Deliberately small conv encoder producing a ``feature_dim``-dimensional vector.

    Args:
        in_channels: Image channels (1 for grayscale MRI).
        channels: Output channels of the first two conv blocks.
        feature_dim: Output channels of the third conv = feature vector size.
    """

    def __init__(self, in_channels: int = 1, channels: tuple[int, int] = (16, 32), feature_dim: int = 128) -> None:
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # average each feature map to one number: size-independent
            nn.Flatten(),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
