"""Reproducibility helpers: seeding, device selection, environment capture.

Why this matters: neural network training uses randomness everywhere (weight
initialisation, data shuffling, dropout). To compare a fly-connectome graph
fairly against a random graph, both must see *the same* randomness, and a run
must be repeatable later. Seeding fixes the starting point of every random
number generator, so the "random" sequence is identical each time.

Known limits (see README):
- CPU runs with the same seed, code and library versions should repeat exactly.
- Some CUDA operations have no deterministic implementation. With
  ``warn_only=True`` PyTorch warns instead of crashing, so GPU runs may differ
  very slightly.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from importlib import metadata
from typing import Any

import numpy as np
import torch

# Libraries whose versions are recorded with every run.
_TRACKED_PACKAGES = (
    "torch", "torchvision", "numpy", "pandas", "scipy",
    "scikit-learn", "networkx", "matplotlib", "pyyaml",
)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs).

    Args:
        seed: Non-negative integer seed.
        deterministic: If True, also ask PyTorch/cuDNN to prefer deterministic
            algorithms (slightly slower on GPU, no effect on speed on CPU).
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device

    # PYTHONHASHSEED only affects *new* Python processes (hash randomisation of
    # the current process is fixed at startup). Setting it keeps any DataLoader
    # worker processes consistent.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Required by cuBLAS for deterministic matrix multiplies on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def make_generator(seed: int) -> torch.Generator:
    """Return a PyTorch random generator with its own fixed seed.

    Pass it to ``DataLoader(generator=...)`` or ``torch.randperm(generator=...)``
    so shuffling is reproducible and independent of other random calls.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """``DataLoader(worker_init_fn=seed_worker)``: give each worker a reproducible seed.

    PyTorch already seeds each worker's torch RNG from the DataLoader's
    generator. This copies that seed into NumPy and Python ``random`` as well,
    which PyTorch does not do on its own.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Turn a config device string into a ``torch.device``.

    Args:
        requested: "cpu", "cuda", or "auto" (CUDA if available, else CPU).

    Raises:
        RuntimeError: If "cuda" is requested but no CUDA GPU is available.
            Failing loudly is better than silently running a GPU experiment on CPU.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is not available")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device {requested!r}; expected 'auto', 'cpu' or 'cuda'")


def collect_environment_info(device: torch.device) -> dict[str, Any]:
    """Collect software/hardware information to save alongside a run."""
    packages: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "packages": packages,
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
    return info


def seed_fingerprint(n: int = 3) -> dict[str, list[float]]:
    """Draw ``n`` numbers from each seeded generator (call right after :func:`set_seed`).

    Same seed gives the same fingerprint. Saving it with a run gives a quick
    check that seeding actually happened and behaves the same on another machine.
    Note: this consumes random numbers, so later draws in the same process shift.
    """
    return {
        "python_random": [round(random.random(), 8) for _ in range(n)],
        "numpy": [round(float(x), 8) for x in np.random.rand(n)],
        "torch": [round(float(x), 8) for x in torch.rand(n)],
    }
