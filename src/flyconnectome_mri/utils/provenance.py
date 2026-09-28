"""Provenance helpers: record where data and code came from."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def git_commit() -> str | None:
    """Return the current git commit hash, or None outside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum of a file (hex string)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string, e.g. '2026-09-28T11:22:33+00:00'."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
