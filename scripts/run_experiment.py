"""Run one experiment from a YAML config.

Usage:
    python scripts/run_experiment.py --config configs/base.yaml
    python scripts/run_experiment.py --config configs/drosophila.yaml --seed 3

Steps:
    load + validate config -> seed everything -> choose device ->
    create a run folder (resolved config, environment info, seed fingerprint, log)
    -> train on train, select on validation, evaluate ONCE on test
    (see flyconnectome_mri/training/experiment.py).

Run folder contents: config.yaml, environment.json, seed_fingerprint.json,
run.log, best_model.pt, history.csv, training_curves.png,
val_predictions_slice.csv, test_predictions_slice.csv,
test_predictions_patient.csv, metrics.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from flyconnectome_mri.training.experiment import run_training_experiment
from flyconnectome_mri.training.metrics import format_metrics
from flyconnectome_mri.training.reproducibility import (
    collect_environment_info,
    resolve_device,
    seed_fingerprint,
    set_seed,
)
from flyconnectome_mri.utils.config import Config, load_config, save_config
from flyconnectome_mri.utils.logging import setup_logging
from flyconnectome_mri.utils.provenance import git_commit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FlyConnectomeMRI experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override experiment.output_dir (where the run folder is created).",
    )
    return parser.parse_args(argv)


def create_run_dir(config: Config, output_dir: str | None) -> Path:
    """Create a unique folder ``<output_dir>/<name>_seed<seed>_<timestamp>``."""
    base = Path(output_dir or config.experiment.output_dir)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{config.experiment.name}_seed{config.experiment.seed}_{stamp}"

    run_dir = base / stem
    suffix = 1
    while run_dir.exists():  # two runs in the same second must not overwrite each other
        run_dir = base / f"{stem}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    config = load_config(args.config, seed=args.seed)

    run_dir = create_run_dir(config, args.output_dir)
    logger = setup_logging(run_dir / "run.log")
    logger.info("Config file : %s", Path(args.config).resolve())
    logger.info("Run folder  : %s", run_dir)

    set_seed(config.experiment.seed, deterministic=config.experiment.deterministic)
    fingerprint = seed_fingerprint()
    device = resolve_device(config.experiment.device)
    logger.info("Seed        : %d (deterministic=%s)", config.experiment.seed, config.experiment.deterministic)
    logger.info("Device      : %s", device)
    logger.info("Graph       : type=%s, num_nodes=%d", config.graph.type, config.graph.num_nodes)
    logger.info("Seed fingerprint (torch): %s", fingerprint["torch"])

    save_config(config, run_dir / "config.yaml")
    environment = collect_environment_info(device)
    environment["seed"] = config.experiment.seed
    environment["git_commit"] = git_commit()
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (run_dir / "seed_fingerprint.json").write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

    metrics = run_training_experiment(config, run_dir, device, logger)
    for level in ("slice", "patient"):
        logger.info("TEST %s-level metrics:\n%s", level.upper(), format_metrics(metrics["test"][level]))
    logger.info("Parameters: %d | training %.1f s | test inference %.2f ms/slice",
                metrics["parameters"], metrics["training"]["training_seconds"],
                metrics["inference"]["forward_ms_per_slice"])
    logger.info("All outputs in %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
