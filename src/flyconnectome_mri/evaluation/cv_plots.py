"""The nine pre-registered Phase 10 figures (defined before any Phase 10 result existed).

Colour = model (validated categorical slots): Visual Drosophila blue, Visual
rewired orange, Visual WS yellow, CNN baseline green. Model identity is always
also written on an axis or in a legend. Dots = individual runs/seeds/folds
(labelled where few), bars = means. Failed runs are shown as counts, never hidden.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flyconnectome_mri.evaluation.cv_summary import DISPLAY, FOLDS, MODELS, PRIMARY  # noqa: E402

COLORS = {"visual_drosophila": "#2a78d6", "visual_rewired": "#eb6834", "visual_ws": "#eda100",
          "cnn_baseline": "#008300"}
TEXT = "#5f5e5a"
GRID = "#e6e4dc"


def _style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TEXT)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _failed_line(runs: pd.DataFrame) -> str:
    """Second title line naming runs that are not ok (they are excluded from means, never hidden)."""
    failed = runs[runs["status"] != "ok"]
    if not len(failed):
        return ""
    text = ", ".join(f"{DISPLAY[m]}: {n}" for m, n in failed.groupby("model").size().items())
    return f"\nruns not ok (excluded from means): {text}"


def metric_by_fold(runs: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    """Figures 1 and 6: per outer fold, one dot per seed for each model (dodged), bar = fold mean."""
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    width = 0.8 / len(MODELS)
    for m_index, model in enumerate(MODELS):
        color = COLORS[model]
        for fold in FOLDS:
            part = runs[(runs["model"] == model) & (runs["outer_fold"] == fold) & (runs["status"] == "ok")]
            x = fold - 0.4 + width * (m_index + 0.5)
            ax.scatter(np.full(len(part), x), part[metric], s=22, color=color, edgecolor="white", linewidth=0.8,
                       zorder=4, label=DISPLAY[model] if fold == FOLDS[0] else None)
            if len(part):
                ax.hlines(part[metric].mean(), x - width * 0.4, x + width * 0.4, color=color, lw=2.5, zorder=3)
    ax.set_xticks(FOLDS, [f"outer fold {f}" for f in FOLDS])
    ax.set_title(f"{title}  (dots = seeds; bar = mean over seeds){_failed_line(runs)}", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _style(ax)
    _save(fig, path)


def fold_means_by_model(fold_table: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    """Figure 2: the five fold means per model (labelled with the fold), bar = mean across folds."""
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    for x, model in enumerate(MODELS):
        part = fold_table[fold_table["model"] == model].dropna(subset=[metric])
        offsets = np.linspace(-0.2, 0.2, max(len(part), 1))
        for offset, row in zip(offsets, part.itertuples()):
            value = getattr(row, metric)
            ax.scatter(x + offset, value, s=46, color=COLORS[model], edgecolor="white", linewidth=1.2, zorder=4)
            ax.annotate(str(row.outer_fold), (x + offset, value), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color=TEXT)
        if len(part):
            ax.hlines(part[metric].mean(), x - 0.3, x + 0.3, color=COLORS[model], lw=2.5, zorder=3)
    ax.set_xticks(range(len(MODELS)), [DISPLAY[m] for m in MODELS])
    ax.set_title(f"{title}  (dots = outer folds, labelled; bar = mean across folds)", loc="left", fontsize=11)
    _style(ax)
    _save(fig, path)


def oof_by_seed(oof_table: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    """Figure 3: pooled OOF metric per model, one dot per training seed (labelled), bar = mean."""
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    for x, model in enumerate(MODELS):
        part = oof_table[(oof_table["model"] == model)].dropna(subset=[metric])
        offsets = np.linspace(-0.2, 0.2, max(len(part), 1))
        for offset, row in zip(offsets, part.itertuples()):
            value = getattr(row, metric)
            ax.scatter(x + offset, value, s=46, color=COLORS[model], edgecolor="white", linewidth=1.2, zorder=4)
            ax.annotate(str(row.seed), (x + offset, value), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=7, color=TEXT)
        if len(part):
            ax.hlines(part[metric].mean(), x - 0.3, x + 0.3, color=COLORS[model], lw=2.5, zorder=3)
        not_ok = int((oof_table[oof_table["model"] == model]["status"] != "ok").sum())
        if not_ok:
            ax.annotate(f"{not_ok} seed(s) incomplete", (x, 0), xycoords=("data", "axes fraction"), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(range(len(MODELS)), [DISPLAY[m] for m in MODELS])
    ax.set_title(f"{title}  (dots = training seeds, labelled; bar = mean)", loc="left", fontsize=11)
    _style(ax)
    _save(fig, path)


def paired_by_fold(diff: pd.DataFrame, title: str, path: Path, metric: str = PRIMARY) -> None:
    """Figure 4: Drosophila - rewired per outer fold; dots = seeds (labelled), bar = fold mean."""
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    for fold in FOLDS:
        part = diff[diff["outer_fold"] == fold].dropna(subset=[metric])
        offsets = np.linspace(-0.2, 0.2, max(len(part), 1))
        for offset, row in zip(offsets, part.itertuples()):
            value = getattr(row, metric)
            ax.scatter(fold + offset, value, s=42, color=COLORS["visual_drosophila"], edgecolor="white",
                       linewidth=1.2, zorder=4)
            ax.annotate(str(row.seed), (fold + offset, value), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color=TEXT)
        if len(part):
            ax.hlines(part[metric].mean(), fold - 0.3, fold + 0.3, color=COLORS["visual_drosophila"], lw=2.5, zorder=3)
    ax.axhline(0, color=TEXT, lw=1)
    ax.set_xticks(FOLDS, [f"outer fold {f}" for f in FOLDS])
    ax.set_ylabel("difference")
    ax.set_title(f"{title}\ndots = seeds (labelled); bar = fold mean; above 0 = Drosophila higher", loc="left",
                 fontsize=11)
    _style(ax)
    _save(fig, path)


def paired_all_runs(diff: pd.DataFrame, title: str, path: Path, metric: str = PRIMARY) -> None:
    """Figure 5: all 25 paired differences in fixed (fold, seed) order; bar = mean of all pairs."""
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ordered = diff.sort_values(["outer_fold", "seed"]).reset_index(drop=True)
    x = np.arange(len(ordered))
    values = ordered[metric]
    ax.scatter(x, values, s=40, color=COLORS["visual_drosophila"], edgecolor="white", linewidth=1.2, zorder=4)
    missing = values.isna()
    if missing.any():
        ax.scatter(x[missing], np.zeros(missing.sum()), marker="x", s=40, color=TEXT, zorder=4, label="pair not ok (NA)")
        ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color=TEXT, lw=1)
    if values.notna().any():
        ax.axhline(values.mean(), color=COLORS["visual_drosophila"], lw=2, ls="--")
    ax.set_xticks(x, [f"f{f}s{s}" for f, s in zip(ordered["outer_fold"], ordered["seed"])], fontsize=7, rotation=90)
    ax.set_ylabel("difference")
    ax.set_title(f"{title}\nall fold×seed pairs (f = outer fold, s = seed); dashed = mean; above 0 = Drosophila higher",
                 loc="left", fontsize=11)
    _style(ax)
    _save(fig, path)


def runs_by_model(runs: pd.DataFrame, panels: list[tuple[str, str]], title: str, path: Path) -> None:
    """Figures 7-9: one panel per column; dots = all fold x seed runs of each model, bar = mean."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2), constrained_layout=True, squeeze=False)
    ok = runs[runs["status"] == "ok"]
    for ax, (column, label) in zip(axes.flat, panels):
        for x, model in enumerate(MODELS):
            values = ok.loc[ok["model"] == model, column].astype(float).dropna()
            jitter = np.linspace(-0.22, 0.22, max(len(values), 1))
            ax.scatter(x + jitter[: len(values)], values, s=18, color=COLORS[model], edgecolor="white", linewidth=0.6,
                       zorder=4)
            if len(values):
                ax.hlines(values.mean(), x - 0.3, x + 0.3, color=COLORS[model], lw=2.5, zorder=3)
        ax.set_xticks(range(len(MODELS)), [DISPLAY[m] for m in MODELS], rotation=20, ha="right", fontsize=8)
        ax.set_title(label, loc="left", fontsize=10)
        _style(ax)
    fig.suptitle(f"{title}  (dots = fold×seed runs; bar = mean){_failed_line(runs)}", x=0.01, ha="left", fontsize=12)
    _save(fig, path)
