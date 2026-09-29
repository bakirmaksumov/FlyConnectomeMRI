"""Phase 8 figures (static PNG).

Colour follows the model (fixed categorical slots, validated with the dataviz
palette checker); model identity is ALWAYS also given by the x-axis label, so
colour is never the only cue. Single y-axis per panel, recessive grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flyconnectome_mri.evaluation.comparison import DISPLAY_NAMES, metric_column  # noqa: E402

MODEL_COLORS = {
    "graph_drosophila": "#2a78d6",  # slot 1 blue
    "graph_rewired": "#eb6834",     # slot 2 orange
    "graph_er": "#1baf7a",          # slot 3 aqua
    "graph_ws": "#eda100",          # slot 4 yellow
    "graph_ba": "#e87ba4",          # slot 5 magenta
    "cnn_baseline": "#008300",      # slot 6 green
}
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e6e4dc"
NEUTRAL = "#5f5e5a"


def _style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY)


def _models(runs: pd.DataFrame) -> list[str]:
    return [m for m in runs["model"].cat.categories if (runs["model"] == m).any()]


def plot_metric_by_seed(runs: pd.DataFrame, column: str, title: str, path: Path) -> None:
    """One column of dots per model (one dot per seed, labelled with the seed), plus the mean as a bar."""
    ok = runs[runs["status"] == "ok"]
    models = _models(runs)
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    offsets = np.linspace(-0.22, 0.22, max(len(ok["seed"].unique()), 1))
    seeds = sorted(ok["seed"].unique())
    for x, model in enumerate(models):
        part = ok[ok["model"] == model].set_index("seed")[column]
        color = MODEL_COLORS.get(model, NEUTRAL)
        ax.hlines(part.mean(), x - 0.32, x + 0.32, color=color, lw=2.5, zorder=3)
        for offset, seed in zip(offsets, seeds):
            if seed in part.index:
                ax.scatter(x + offset, part[seed], s=46, color=color, edgecolor="white", linewidth=1.2, zorder=4)
                ax.annotate(str(seed), (x + offset, part[seed]), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=7, color=TEXT_SECONDARY)
        failed = runs[(runs["model"] == model) & (runs["status"] != "ok")]
        if len(failed):
            ax.annotate(f"{len(failed)} failed", (x, 0), xycoords=("data", "axes fraction"), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(models)), [DISPLAY_NAMES.get(m, m) for m in models])
    ax.set_title(f"{title}  (dots = seeds, labelled; bar = mean)", loc="left", fontsize=11)
    ax.set_xlim(-0.6, len(models) - 0.4)
    _style(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mean_sd(summary: pd.DataFrame, metrics: list[tuple[str, str]], path: Path) -> None:
    """Small multiples: one panel per metric, mean ± SD (error bar) per model."""
    cols = 3
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.3 * rows), constrained_layout=True, squeeze=False)
    names = [DISPLAY_NAMES.get(m, m) for m in summary["model"]]
    for ax, (level, metric) in zip(axes.flat, metrics):
        for x, (_, row) in enumerate(summary.iterrows()):
            color = MODEL_COLORS.get(row["model"], NEUTRAL)
            ax.errorbar(x, row[f"{level}_{metric}_mean"], yerr=row[f"{level}_{metric}_sd"], fmt="o", ms=7,
                        color=color, ecolor=color, elinewidth=2, capsize=4)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{level} {metric.replace('_', ' ')}", loc="left", fontsize=10)
        _style(ax)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle("Mean ± SD over seeds (successful runs)", x=0.01, ha="left", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_paired_differences(diff: pd.DataFrame, metrics: list[tuple[str, str]], title: str, path: Path) -> None:
    """Per-seed paired differences (reference - other) per metric; 0 line = no difference."""
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    labels = []
    for x, (level, metric) in enumerate(metrics):
        values = diff.set_index("seed")[metric_column(level, metric)]
        ax.hlines(values.mean(), x - 0.3, x + 0.3, color=MODEL_COLORS["graph_drosophila"], lw=2.5, zorder=3)
        offsets = np.linspace(-0.2, 0.2, max(len(values), 1))
        for offset, (seed, value) in zip(offsets, values.items()):
            ax.scatter(x + offset, value, s=42, color=MODEL_COLORS["graph_drosophila"], edgecolor="white",
                       linewidth=1.2, zorder=4)
            ax.annotate(str(seed), (x + offset, value), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color=TEXT_SECONDARY)
        labels.append(f"{level}\n{metric.replace('balanced_accuracy', 'bal. acc.').replace('_', ' ')}")
    ax.axhline(0, color=TEXT_SECONDARY, lw=1)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("difference (reference − null)")
    ax.set_title(f"{title}  (dots = seeds, labelled; bar = mean; above 0 = Drosophila higher)",
                 loc="left", fontsize=11)
    _style(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_stability(runs: pd.DataFrame, path: Path) -> None:
    """Best epoch, validation AUC at the best epoch, and training time per model and seed."""
    ok = runs[runs["status"] == "ok"]
    models = _models(runs)
    panels = [("best_epoch", "best validation epoch"), ("val_roc_auc_best", "validation slice ROC AUC at best epoch"),
              ("training_seconds", "training time (s)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (column, title) in zip(axes, panels):
        for x, model in enumerate(models):
            values = ok.loc[ok["model"] == model, column].astype(float)
            color = MODEL_COLORS.get(model, NEUTRAL)
            ax.scatter(np.full(len(values), x) + np.linspace(-0.15, 0.15, len(values)), values, s=36, color=color,
                       edgecolor="white", linewidth=1.1, zorder=4)
            ax.hlines(values.mean(), x - 0.3, x + 0.3, color=color, lw=2.5, zorder=3)
        ax.set_xticks(range(len(models)), [DISPLAY_NAMES.get(m, m) for m in models], rotation=30, ha="right",
                      fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        _style(ax)
    fig.suptitle("Training stability (dots = seeds; bar = mean)", x=0.01, ha="left", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
