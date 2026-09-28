"""Training-curve figure (static PNG).

Two panels sharing the epoch axis: loss (train vs validation) and validation
ROC AUC. The selected (best-validation) epoch is marked. Colors: categorical
slots 1 (blue, train) and 2 (orange, validation) of the project palette.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt  # noqa: E402

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
TEXT_SECONDARY = "#5f5e5a"
GRID_COLOR = "#e6e4dc"


def plot_training_curves(history: list[dict[str, Any]], best_epoch: int, path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax_loss.plot(epochs, [r["train_loss"] for r in history], color=TRAIN_COLOR, lw=2, marker="o", ms=4, label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], color=VAL_COLOR, lw=2, marker="o", ms=4, label="validation")
    ax_loss.set_title("Loss (weighted BCE)", loc="left", fontsize=11)
    ax_loss.legend(frameon=False)
    ax_auc.plot(epochs, [r["val_roc_auc"] for r in history], color=VAL_COLOR, lw=2, marker="o", ms=4)
    ax_auc.set_title("Validation ROC AUC (slice level)", loc="left", fontsize=11)
    for ax in (ax_loss, ax_auc):
        ax.axvline(best_epoch, color=TEXT_SECONDARY, lw=1, ls="--")
        ax.annotate(f"selected epoch {best_epoch}", xy=(best_epoch, 0), xycoords=("data", "axes fraction"),
                    xytext=(4, 6), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlabel("epoch")
        ax.grid(True, color=GRID_COLOR, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=TEXT_SECONDARY)
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)
