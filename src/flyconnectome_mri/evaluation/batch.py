"""Run a batch of (model config, seed) experiments, one separate process each.

Failed-run policy (docs/RESEARCH_PLAN.md): a failed run is recorded and its log
kept; it is never replaced, retried or rerun with other settings. Resuming a
batch only starts runs that were never started.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from flyconnectome_mri.utils.provenance import PROJECT_ROOT, git_commit

INDEX_COLUMNS = ["model", "seed", "config", "returncode", "wall_seconds", "run_dir", "log", "started_utc"]


def new_batch_dir(prefix: str) -> Path:
    batch = PROJECT_ROOT / "results" / f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    batch.mkdir(parents=True)
    return batch


def run_batch(batch: Path, models: dict[str, str], seeds: list[int]) -> Path:
    """Run every (model, seed) not yet started, interleaved by seed; returns the index CSV path.

    Args:
        batch: Batch folder (created by ``new_batch_dir`` or an existing one to resume).
        models: Short model name -> config file (relative to the project root).
        seeds: Training seeds.
    """
    info = batch / "batch_info.txt"
    if not info.exists():
        info.write_text(f"git_commit: {git_commit()}\nmodels: {list(models)}\nseeds: {seeds}\n", encoding="utf-8")
    (batch / "logs").mkdir(exist_ok=True)
    index_path = batch / "runs_index.csv"
    started = set()
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            started = {(row["model"], int(row["seed"])) for row in csv.DictReader(f)}
    else:
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(INDEX_COLUMNS)

    jobs = [(model, seed) for seed in seeds for model in models]
    for number, (model, seed) in enumerate(jobs, start=1):
        if (model, seed) in started:
            print(f"[{number:2d}/{len(jobs)}] {model} seed {seed}: already started, not rerun")
            continue
        out_dir = batch / "runs" / model
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = batch / "logs" / f"{model}_seed{seed}.log"
        before = set(out_dir.iterdir())
        start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "run_experiment.py"), "--config", models[model],
                 "--seed", str(seed), "--output-dir", str(out_dir)],
                cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
        created = sorted(set(out_dir.iterdir()) - before)
        run_dir = str(created[0]) if created else ""
        wall = time.perf_counter() - start
        with open(index_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([model, seed, models[model], proc.returncode, f"{wall:.1f}", run_dir,
                                    str(log_path), start_utc])
        print(f"[{number:2d}/{len(jobs)}] {model} seed {seed}: exit {proc.returncode} in {wall:.0f}s", flush=True)
    print(f"Batch complete: {batch}")
    return index_path
