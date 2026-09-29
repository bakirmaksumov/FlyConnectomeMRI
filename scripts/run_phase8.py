"""Phase 8: run every model with every seed (frozen settings), then summarise.

Usage:
    python scripts/run_phase8.py                       # new batch folder results/phase8_<timestamp>
    python scripts/run_phase8.py --resume results/phase8_20260929-000000

Each run is a separate Python process (``scripts/run_experiment.py``), so runs
cannot influence each other. Runs are interleaved by seed (seed 0 for all
models, then seed 1, ...) so a partial batch is still paired.

Failed-run policy (docs/RESEARCH_PLAN.md): a failed run is recorded and its
log kept; it is never replaced, retried or rerun with other settings.
``--resume`` only starts runs that have not been started at all.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from flyconnectome_mri.evaluation.batch import new_batch_dir, run_batch
from flyconnectome_mri.utils.provenance import PROJECT_ROOT

MODELS = {  # model name -> config file
    "cnn_baseline": "configs/cnn_baseline.yaml",
    "drosophila": "configs/drosophila.yaml",
    "rewired": "configs/rewired.yaml",
    "er": "configs/er.yaml",
    "ws": "configs/ws.yaml",
    "ba": "configs/ba.yaml",
}
SEEDS = [0, 1, 2, 3, 4]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 8 topology comparison (30 runs).")
    parser.add_argument("--resume", default=None, help="Existing batch folder: start only runs never started.")
    args = parser.parse_args(argv)
    batch = Path(args.resume) if args.resume else new_batch_dir("phase8")
    run_batch(batch, MODELS, SEEDS)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "summarize_phase8.py"), "--batch", str(batch)],
                   cwd=PROJECT_ROOT, check=False)


if __name__ == "__main__":
    main()
