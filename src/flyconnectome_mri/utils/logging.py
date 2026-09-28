"""Logging setup: messages go to the console and, optionally, to a run's log file."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "flyconnectome_mri"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger.

    Calling this again replaces the previous handlers, so repeated calls (for
    example one per run in a test session) do not duplicate every message.

    Args:
        log_file: If given, messages are also written to this file (UTF-8).
        level: Minimum level to record, e.g. ``logging.INFO``.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the project logger (call :func:`setup_logging` first to add handlers)."""
    return logging.getLogger(LOGGER_NAME)
