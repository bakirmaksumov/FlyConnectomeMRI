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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from flyconnectome_mri.utils.provenance import PROJECT_ROOT, git_commit

INDEX_COLUMNS = ["model", "seed", "fold", "config", "prepared_dir", "returncode", "wall_seconds", "run_dir", "log",
                 "started_utc"]


@dataclass(frozen=True)
class Job:
    """One training run: model config + seed (+ optional CV fold dataset)."""

    model: str
    seed: int
    config: str
    fold: int | None = None
    prepared_dir: str | None = None

    @property
    def key(self) -> tuple[str, int, str]:
        return self.model, self.seed, "" if self.fold is None else str(self.fold)

    @property
    def name(self) -> str:
        return f"{self.model}_seed{self.seed}" + ("" if self.fold is None else f"_fold{self.fold}")


def new_batch_dir(prefix: str) -> Path:
    batch = PROJECT_ROOT / "results" / f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    batch.mkdir(parents=True)
    return batch


def run_batch(batch: Path, models: dict[str, str], seeds: list[int]) -> Path:
    """Run every (model, seed), interleaved by seed (Phases 8 and 9); returns the index CSV path."""
    return run_jobs(batch, [Job(model, seed, models[model]) for seed in seeds for model in models])


def run_jobs(batch: Path, jobs: list[Job]) -> Path:
    """Run every job not yet started, in the given order; returns the index CSV path.

    Each job is a separate ``scripts/run_experiment.py`` process. A failed job is
    recorded (return code + log) and never retried.
    """
    info = batch / "batch_info.txt"
    if not info.exists():
        models = sorted({job.model for job in jobs})
        seeds = sorted({job.seed for job in jobs})
        folds = sorted({job.fold for job in jobs if job.fold is not None})
        info.write_text(f"git_commit: {git_commit()}\nmodels: {models}\nseeds: {seeds}\nfolds: {folds}\n"
                        f"jobs: {len(jobs)}\n", encoding="utf-8")
    (batch / "logs").mkdir(exist_ok=True)
    index_path = batch / "runs_index.csv"
    started = set()
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            started = {(row["model"], int(row["seed"]), row.get("fold", "")) for row in csv.DictReader(f)}
    else:
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(INDEX_COLUMNS)

    for number, job in enumerate(jobs, start=1):
        if job.key in started:
            print(f"[{number:3d}/{len(jobs)}] {job.name}: already started, not rerun")
            continue
        out_dir = batch / "runs" / job.model / ("" if job.fold is None else f"fold{job.fold}")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = batch / "logs" / f"{job.name}.log"
        before = set(out_dir.iterdir())
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_experiment.py"), "--config", job.config,
                   "--seed", str(job.seed), "--output-dir", str(out_dir)]
        if job.prepared_dir is not None:
            command += ["--prepared-dir", job.prepared_dir]
        start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        created = sorted(set(out_dir.iterdir()) - before)
        run_dir = str(created[0]) if created else ""
        wall = time.perf_counter() - start
        with open(index_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([job.model, job.seed, "" if job.fold is None else job.fold, job.config,
                                    job.prepared_dir or "", proc.returncode, f"{wall:.1f}", run_dir, str(log_path),
                                    start_utc])
        print(f"[{number:3d}/{len(jobs)}] {job.name}: exit {proc.returncode} in {wall:.0f}s", flush=True)
    print(f"Batch complete: {batch}")
    return index_path
