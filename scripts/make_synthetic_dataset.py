"""Create the small SYNTHETIC demo dataset (NOT medical data) used to exercise the pipeline.

Usage:
    python scripts/make_synthetic_dataset.py --output data/raw/synthetic_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flyconnectome_mri.datasets.synthetic import make_synthetic_dataset
from flyconnectome_mri.utils.provenance import PROJECT_ROOT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic (non-medical) image dataset.")
    parser.add_argument("--output", default="data/raw/synthetic_demo")
    parser.add_argument("--patients-per-class", type=int, default=10)
    parser.add_argument("--slices-per-patient", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty; choose another --output or delete it first.")
    make_synthetic_dataset(output, args.patients_per_class, args.slices_per_patient, args.image_size, args.seed)
    print(f"Synthetic dataset written to {output} (NOT medical data)")


if __name__ == "__main__":
    main()
