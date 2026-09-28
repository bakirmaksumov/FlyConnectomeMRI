"""Image preprocessing: grayscale/RGB -> resize -> tensor -> normalize.

No augmentation (random flips, crops, rotations...) at this stage: every image
is processed identically every time, so preprocessing is fully deterministic.

Normalization uses the mean/std of the TRAINING split only (computed by
``prepare_dataset``). Using statistics from validation or test images would
leak information about them into training.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_image(path: str | Path, grayscale: bool) -> Image.Image:
    """Open an image and convert it to 1-channel ("L") or 3-channel ("RGB") mode."""
    with Image.open(path) as image:
        return image.convert("L" if grayscale else "RGB")


def build_transform(
    image_size: int, mean: Sequence[float] | None = None, std: Sequence[float] | None = None
) -> transforms.Compose:
    """Resize to (image_size, image_size), convert to a float tensor in [0, 1], then normalize.

    Args:
        image_size: Output height and width in pixels.
        mean, std: Per-channel normalization, output = (x - mean) / std. Pass
            None for both to skip normalization (used to compute the statistics).
    """
    if (mean is None) != (std is None):
        raise ValueError("pass both mean and std, or neither")
    steps: list = [
        # Bilinear with antialiasing: smooth, deterministic downscaling.
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),  # uint8 [0, 255] -> float32 [0, 1], shape (channels, H, W)
    ]
    if mean is not None:
        if any(s <= 0 for s in std):
            raise ValueError(f"std must be positive, got {list(std)}")
        steps.append(transforms.Normalize(mean=list(mean), std=list(std)))
    return transforms.Compose(steps)


def compute_channel_stats(paths: Sequence[Path], image_size: int, grayscale: bool) -> tuple[list[float], list[float]]:
    """Per-channel pixel mean and std over ``paths`` after resize, in [0, 1] units.

    Uses running sums (float64), so memory stays small for large datasets.
    """
    if not paths:
        raise ValueError("no images to compute statistics from")
    to_tensor = build_transform(image_size)
    total = total_sq = None
    count = 0
    for path in paths:
        x = to_tensor(load_image(path, grayscale)).double()
        flat = x.reshape(x.shape[0], -1)
        total = flat.sum(dim=1) if total is None else total + flat.sum(dim=1)
        total_sq = (flat**2).sum(dim=1) if total_sq is None else total_sq + (flat**2).sum(dim=1)
        count += flat.shape[1]
    mean = total / count
    std = torch.sqrt(torch.clamp(total_sq / count - mean**2, min=0.0))
    return [round(float(m), 6) for m in mean], [round(float(s), 6) for s in std]
