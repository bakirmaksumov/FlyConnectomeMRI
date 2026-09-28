"""End-to-end test of scripts/run_experiment.py (Phase 1 behaviour)."""

import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_experiment.py"


def run(config: str, output_dir: Path, *extra: str) -> Path:
    """Run the script in a fresh Python process and return the created run folder."""
    before = set(output_dir.iterdir()) if output_dir.exists() else set()
    subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(PROJECT_ROOT / config),
         "--output-dir", str(output_dir), *extra],
        check=True, capture_output=True, text=True,
    )
    created = set(output_dir.iterdir()) - before
    assert len(created) == 1
    return created.pop()


def test_run_creates_expected_files(tmp_path: Path) -> None:
    run_dir = run("configs/base.yaml", tmp_path)
    for name in ("config.yaml", "environment.json", "seed_fingerprint.json", "run.log"):
        assert (run_dir / name).is_file(), name

    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved["experiment"]["name"] == "base"
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert env["seed"] == 0
    assert "torch" in env["packages"]


def test_same_seed_reproduces_across_processes(tmp_path: Path) -> None:
    """Two separate Python processes with the same seed must draw the same numbers."""
    a = run("configs/er.yaml", tmp_path, "--seed", "3")
    b = run("configs/er.yaml", tmp_path, "--seed", "3")
    c = run("configs/er.yaml", tmp_path, "--seed", "4")
    fp = lambda d: json.loads((d / "seed_fingerprint.json").read_text(encoding="utf-8"))
    assert a != b  # distinct folders, nothing overwritten
    assert fp(a) == fp(b)
    assert fp(a) != fp(c)
