"""Phase 9: visual-graph topology comparison (25 graph-model runs; CNN baseline NOT retrained).

Usage:
    python scripts/run_phase9.py
    python scripts/run_phase9.py --resume results/phase9_<timestamp>

Protocol: docs/RESEARCH_PLAN.md, "Phase 9 protocol". Afterwards the batch is
summarised by scripts/summarize_phase9.py (with the frozen Phase 8 results as
reference).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from flyconnectome_mri.evaluation.batch import new_batch_dir, run_batch
from flyconnectome_mri.utils.provenance import PROJECT_ROOT

MODELS = {
    "visual_drosophila": "configs/visual_drosophila.yaml",
    "visual_rewired": "configs/visual_rewired.yaml",
    "visual_er": "configs/visual_er.yaml",
    "visual_ws": "configs/visual_ws.yaml",
    "visual_ba": "configs/visual_ba.yaml",
}
SEEDS = [0, 1, 2, 3, 4]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 9 visual-graph comparison (25 runs).")
    parser.add_argument("--resume", default=None, help="Existing batch folder: start only runs never started.")
    args = parser.parse_args(argv)
    batch = Path(args.resume) if args.resume else new_batch_dir("phase9")
    run_batch(batch, MODELS, SEEDS)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "summarize_phase9.py"), "--batch", str(batch)],
                   cwd=PROJECT_ROOT, check=False)


if __name__ == "__main__":
    main()
