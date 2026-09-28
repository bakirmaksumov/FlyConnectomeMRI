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
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from flyconnectome_mri.utils.provenance import PROJECT_ROOT, git_commit

MODELS = {  # model name -> config file
    "cnn_baseline": "configs/cnn_baseline.yaml",
    "drosophila": "configs/drosophila.yaml",
    "rewired": "configs/rewired.yaml",
    "er": "configs/er.yaml",
    "ws": "configs/ws.yaml",
    "ba": "configs/ba.yaml",
}
SEEDS = [0, 1, 2, 3, 4]
INDEX_COLUMNS = ["model", "seed", "config", "returncode", "wall_seconds", "run_dir", "log", "started_utc"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 8 topology comparison (30 runs).")
    parser.add_argument("--resume", default=None, help="Existing batch folder: start only runs never started.")
    args = parser.parse_args(argv)

    if args.resume:
        batch = Path(args.resume)
    else:
        batch = PROJECT_ROOT / "results" / f"phase8_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        batch.mkdir(parents=True)
        (batch / "batch_info.txt").write_text(
            f"git_commit: {git_commit()}\nmodels: {list(MODELS)}\nseeds: {SEEDS}\n", encoding="utf-8")
    (batch / "logs").mkdir(exist_ok=True)
    index_path = batch / "runs_index.csv"
    started = set()
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            started = {(row["model"], int(row["seed"])) for row in csv.DictReader(f)}
    else:
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(INDEX_COLUMNS)

    jobs = [(model, seed) for seed in SEEDS for model in MODELS]
    for number, (model, seed) in enumerate(jobs, start=1):
        if (model, seed) in started:
            print(f"[{number:2d}/{len(jobs)}] {model} seed {seed}: already started, not rerun")
            continue
        out_dir = batch / "runs" / model
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = batch / "logs" / f"{model}_seed{seed}.log"
        before = set(out_dir.iterdir())
        start_utc = datetime.utcnow().isoformat(timespec="seconds")
        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "run_experiment.py"), "--config", MODELS[model],
                 "--seed", str(seed), "--output-dir", str(out_dir)],
                cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
        created = sorted(set(out_dir.iterdir()) - before)
        run_dir = str(created[0]) if created else ""
        wall = time.perf_counter() - start
        with open(index_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([model, seed, MODELS[model], proc.returncode, f"{wall:.1f}", run_dir,
                                    str(log_path), start_utc])
        print(f"[{number:2d}/{len(jobs)}] {model} seed {seed}: exit {proc.returncode} in {wall:.0f}s", flush=True)

    print(f"Batch complete: {batch}")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "summarize_phase8.py"), "--batch", str(batch)],
                   cwd=PROJECT_ROOT, check=False)


if __name__ == "__main__":
    main()
