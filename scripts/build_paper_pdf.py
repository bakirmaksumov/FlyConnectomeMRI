"""Build docs/paper/paper_draft.pdf from docs/paper/paper_draft.md.

Usage:
    python scripts/build_paper_pdf.py

Requires reportlab (requirements-docs.txt). Nothing is recomputed: numbers come
from the Markdown draft (itself taken from frozen files in docs/results/), the
result figures are the frozen PNGs, and the three explanatory figures (model
overview, CV design, adjacency matrices) are drawn from frozen input files.

Supported Markdown subset: # / ## / ### headings, paragraphs, "- " bullets
(one nesting level), "1. " numbered items, tables, "---" rules, images
![caption](path), display maths on a line "$$...$$" (numbered) and inline
maths "$...$" (rendered with matplotlib mathtext), **bold**, *italic*, `code`.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"
FIGURES = PAPER / "figures"
MATH_DIR = PAPER / "build" / "math"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
TEXT_WIDTH = A4[0] - 4.0 * cm

INK, MUTED, GRID = "#2b2b28", "#5f5e5a", "#e6e4dc"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
TYPE_ORDER = ["L1", "L2", "L3", "Mi1", "Mi4", "Mi9", "C3", "Tm1", "Tm2", "Tm4", "Tm9"]


# ---------------------------------------------------------------------------
# Explanatory figures drawn from frozen input files
# ---------------------------------------------------------------------------


def figure_adjacency(path: Path) -> None:
    nodes = pd.read_csv(ROOT / "data/connectome/drosophila_visual_128_raw_nodes.csv")
    nodes["type_rank"] = nodes["type"].map({t: i for i, t in enumerate(TYPE_ORDER)})
    nodes = nodes.sort_values(["type_rank", "hex1", "hex2", "bodyId"]).reset_index(drop=True)
    position = {int(b): i for i, b in enumerate(nodes["bodyId"])}
    panels = [("Drosophila visual graph (989 edges)", "data/connectome/drosophila_visual_128_raw.csv"),
              ("Degree-preserving rewired, seed 0 (989 edges)", "data/connectome/rewired_visual128_raw_seed0.csv")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), constrained_layout=True)
    bounds = nodes.groupby("type_rank").size().cumsum().to_numpy()
    starts = np.concatenate([[0], bounds[:-1]])
    for ax, (title, file) in zip(axes, panels):
        edges = pd.read_csv(ROOT / file)
        rows = edges["source"].map(position)
        cols = edges["target"].map(position)
        ax.scatter(cols, rows, s=2.2, marker="s", color=INK, linewidths=0)
        for b in bounds[:-1]:
            ax.axhline(b - 0.5, color=GRID, lw=0.7)
            ax.axvline(b - 0.5, color=GRID, lw=0.7)
        centres = (starts + bounds) / 2 - 0.5
        ax.set_xticks(centres, TYPE_ORDER, fontsize=7, rotation=90)
        ax.set_yticks(centres, TYPE_ORDER, fontsize=7)
        ax.set_xlim(-0.5, 127.5)
        ax.set_ylim(127.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("postsynaptic (target)", fontsize=8, color=MUTED)
        ax.set_ylabel("presynaptic (source)", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=10, loc="left")
        ax.tick_params(colors=MUTED, length=0)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure_cv_design(path: Path) -> None:
    counts = pd.read_csv(ROOT / "docs/results/phase10_prerun/counts.csv")
    removed = counts[counts["class"] == "all"].groupby("outer_fold")["removed_patients"].sum()
    fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    for k in range(1, 6):
        val = k % 5 + 1
        for f in range(1, 6):
            role, color = ("test", BLUE) if f == k else ("val", ORANGE) if f == val else ("train", GREY)
            ax.add_patch(plt.Rectangle((f - 0.46, -k - 0.4), 0.92, 0.8, color=color, lw=0))
            ax.text(f, -k, role, ha="center", va="center", fontsize=9,
                    color="white" if role != "train" else INK)
        ax.text(5.7, -k, f"{int(removed[k])} linked patients excluded from train/val", va="center", fontsize=8,
                color=MUTED)
    ax.set_xticks(range(1, 6), [f"official fold {f}" for f in range(1, 6)], fontsize=8)
    ax.set_yticks([-k for k in range(1, 6)], [f"outer fold {k}" for k in range(1, 6)], fontsize=8)
    ax.set_xlim(0.4, 8.9)
    ax.set_ylim(-5.6, -0.4)
    ax.tick_params(length=0, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Test = official fold k (unchanged); validation = fold k%5+1; every patient is tested exactly once",
                 fontsize=9, loc="left", color=INK)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 35)
    ax.axis("off")

    def box(x, y, w, h, text, face="#ffffff", edge=INK, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=face, ec=edge, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, color=INK,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1, color=MUTED))

    box(1, 13, 11, 8, "MRI slice\n1×128×128")
    box(16, 13, 16, 8, "shared CNN encoder\n3 conv blocks + GAP\n→ 128 features", bold=True)
    arrow(12.5, 17, 15.5, 17)
    box(37, 24, 10, 7, "BatchNorm1d\n(128)")
    box(51, 24, 15, 7, "connectome layer 1\nfeature i → node i", face="#e9f1fb", edge=BLUE)
    box(70, 24, 13, 7, "connectome\nlayer 2", face="#e9f1fb", edge=BLUE)
    box(87, 24, 11, 7, "Linear(128→1)\nall node states")
    arrow(32.5, 19, 36.5, 27)
    arrow(47.5, 27.5, 50.5, 27.5)
    arrow(66.5, 27.5, 69.5, 27.5)
    arrow(83.5, 27.5, 86.5, 27.5)
    box(51, 3, 15, 7, "Linear(128→1)", face="#eef7ee", edge="#008300")
    arrow(32.5, 15, 50.5, 7)
    ax.text(67.5, 33.3, "graph-constrained model (44,411 parameters): one trainable weight per existing directed edge;"
            " no skip connection", ha="center", fontsize=7, color=BLUE)
    ax.text(70, 5.5, "CNN baseline (41,921 parameters)", fontsize=7, color="#008300", va="center")
    ax.text(92.5, 18.5, "P(glioma)", ha="center", fontsize=7.6, color=INK)
    arrow(92.5, 23.3, 92.5, 20.5)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Maths rendering (matplotlib mathtext -> PNG)
# ---------------------------------------------------------------------------


def render_math(tex: str, fontsize: float) -> tuple[Path, float, float]:
    """Render ``tex`` to a PNG; return (path, width_pt, height_pt) at the rendered size."""
    MATH_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{fontsize}:{tex}".encode()).hexdigest()[:16]
    out = MATH_DIR / f"{key}.png"
    dpi = 300
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=fontsize, color=INK)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    from PIL import Image as PILImage

    with PILImage.open(out) as image:
        w, h = image.size
    return out, w * 72 / dpi, h * 72 / dpi


# ---------------------------------------------------------------------------
# Markdown -> reportlab
# ---------------------------------------------------------------------------


def register_fonts() -> None:
    for name, file in (("Serif", "DejaVuSerif.ttf"), ("Serif-Bold", "DejaVuSerif-Bold.ttf"),
                       ("Serif-Italic", "DejaVuSerif-Italic.ttf"), ("Serif-BoldItalic", "DejaVuSerif-BoldItalic.ttf"),
                       ("Mono", "DejaVuSansMono.ttf"), ("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic",
                                  boldItalic="Serif-BoldItalic")


def styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("body", fontName="Serif", fontSize=9.4, leading=12.6, alignment=TA_JUSTIFY,
                          textColor=colors.HexColor(INK), spaceAfter=5)
    return {
        "body": base,
        "title": ParagraphStyle("title", parent=base, fontName="Serif-Bold", fontSize=15, leading=19,
                                alignment=TA_CENTER, spaceAfter=8),
        "note": ParagraphStyle("note", parent=base, fontSize=7.8, leading=10, alignment=TA_CENTER,
                               textColor=colors.HexColor(MUTED), spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base, fontName="Sans-Bold", fontSize=11.5, leading=15, spaceBefore=10,
                             spaceAfter=5, alignment=0),
        "h2": ParagraphStyle("h2", parent=base, fontName="Sans-Bold", fontSize=9.8, leading=13, spaceBefore=7,
                             spaceAfter=3, alignment=0),
        "bullet": ParagraphStyle("bullet", parent=base, leftIndent=14, bulletIndent=4, spaceAfter=2),
        "bullet2": ParagraphStyle("bullet2", parent=base, leftIndent=28, bulletIndent=18, spaceAfter=2),
        "caption": ParagraphStyle("caption", parent=base, fontSize=8, leading=10.4, spaceBefore=3, spaceAfter=10,
                                  alignment=0),
        "cell": ParagraphStyle("cell", parent=base, fontSize=7.2, leading=9, alignment=0, spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", parent=base, fontName="Serif-Bold", fontSize=7.2, leading=9,
                                   alignment=0, spaceAfter=0),
        "eqnum": ParagraphStyle("eqnum", parent=base, alignment=2),
    }


def inline(text: str, style: ParagraphStyle) -> str:
    """Convert inline Markdown (+ inline maths) to reportlab paragraph markup."""
    protected: list[str] = []

    def protect(markup: str) -> str:
        protected.append(markup)
        return f"\x00{len(protected) - 1}\x00"

    def math(match: re.Match) -> str:
        path, w, h = render_math(match.group(1), style.fontSize * 1.05)
        return protect(f'<img src="{path.as_posix()}" width="{w:.1f}" height="{h:.1f}" valign="{-h * 0.28:.1f}"/>')

    def code(match: re.Match) -> str:
        return protect(f'<font face="Mono" size="{style.fontSize - 1:.1f}">{escape(match.group(1))}</font>')

    text = re.sub(r"`([^`]+)`", code, text)
    text = re.sub(r"(?<!\$)\$([^$]+)\$(?!\$)", math, text)
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", text)
    return re.sub("\x00(\\d+)\x00", lambda m: protected[int(m.group(1))], text)


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_table(rows: list[str], st: dict[str, ParagraphStyle]) -> Table:
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows
             if not re.fullmatch(r"\|?[\s:\-|]+\|?", row.strip())]
    data = [[Paragraph(inline(c, st["cellhead" if r == 0 else "cell"]), st["cellhead" if r == 0 else "cell"])
             for c in row] for r, row in enumerate(cells)]
    lengths = np.array([[len(re.sub(r"[*`$]", "", c)) for c in row] for row in cells], dtype=float)
    # Column weight: longest cell, but never narrower than the longest single word of the header.
    header_words = np.array([max((len(w) for w in re.sub(r"[*`$]", "", c).split()), default=1)
                             for c in cells[0]], dtype=float)
    content = lengths[1:].max(axis=0) if len(cells) > 1 else lengths[0]
    weights = np.clip(np.maximum(content, header_words) + 3, 7, 40)
    widths = list(TEXT_WIDTH * weights / weights.sum())
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor(INK)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor(INK)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor(INK)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def build_pdf(markdown: Path, out: Path) -> dict[str, int]:
    register_fonts()
    st = styles()
    story: list = []
    counters = {"eq": 0, "fig": 0, "tables": 0}
    lines = markdown.read_text(encoding="utf-8").splitlines()
    i = 0
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph), st["body"]), st["body"]))
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped == "---":
            flush()
            story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor(GRID), spaceBefore=4,
                                    spaceAfter=6))
        elif line.startswith("# "):
            flush()
            story.append(Paragraph(inline(line[2:], st["title"]), st["title"]))
        elif line.startswith("## "):
            flush()
            story.append(Paragraph(inline(line[3:], st["h1"]), st["h1"]))
        elif line.startswith("### "):
            flush()
            story.append(Paragraph(inline(line[4:], st["h2"]), st["h2"]))
        elif stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            flush()
            counters["eq"] += 1
            path, w, h = render_math(stripped[2:-2], 11)
            scale = min(1.0, (TEXT_WIDTH - 40) / w)
            eq = Table([[Image(str(path), width=w * scale, height=h * scale),
                         Paragraph(f"({counters['eq']})", st["eqnum"])]],
                       colWidths=[TEXT_WIDTH - 40, 40])
            eq.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (0, 0), "CENTER")]))
            story.append(Spacer(1, 3))
            story.append(eq)
            story.append(Spacer(1, 4))
        elif stripped.startswith("!["):
            flush()
            match = re.fullmatch(r"!\[(.*)\]\((.+)\)", stripped)
            caption, rel = match.group(1), match.group(2)
            path = (PAPER / rel).resolve()
            from PIL import Image as PILImage

            with PILImage.open(path) as image:
                pw, ph = image.size
            width = TEXT_WIDTH
            height = width * ph / pw
            if height > 10 * cm:
                height = 10 * cm
                width = height * pw / ph
            counters["fig"] += 1
            story.append(KeepTogether([
                Spacer(1, 4),
                Image(str(path), width=width, height=height),
                Paragraph(f"<b>Fig. {counters['fig']}.</b> " + inline(caption, st["caption"]), st["caption"]),
            ]))
        elif stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            counters["tables"] += 1
            block = [build_table(rows, st), Spacer(1, 8)]
            # Keep a "TABLE ..." caption paragraph on the same page as its table.
            if story and isinstance(story[-1], Paragraph) and story[-1].text.startswith("<b>TABLE"):
                block.insert(0, story.pop())
            story.append(KeepTogether(block))
            continue
        elif re.match(r"^\s{2,}- ", line):
            flush()
            story.append(Paragraph(inline(line.strip()[2:], st["bullet2"]), st["bullet2"], bulletText="–"))
        elif line.startswith("- "):
            flush()
            story.append(Paragraph(inline(line[2:], st["bullet"]), st["bullet"], bulletText="•"))
        elif re.match(r"^\d+\. ", line):
            flush()
            number, text = line.split(". ", 1)
            story.append(Paragraph(inline(text, st["bullet"]), st["bullet"], bulletText=f"{number}."))
        elif stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**") and not story[1:]:
            flush()
            story.append(Paragraph(inline(stripped, st["note"]), st["note"]))
        else:
            paragraph.append(stripped)
        i += 1
    flush()

    def page(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Sans", 7)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(2 * cm, 1.2 * cm, "FlyConnectomeMRI: first draft generated from frozen results. Not peer reviewed.")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Connectome-Constrained Neural Networks for Brain MRI "
                            "Classification (draft)", author="FlyConnectomeMRI")
    doc.build(story, onFirstPage=page, onLaterPages=page)
    return counters


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_pipeline(FIGURES / "fig_pipeline.png")
    figure_cv_design(FIGURES / "fig_cv_design.png")
    figure_adjacency(FIGURES / "fig_adjacency.png")
    out = PAPER / "paper_draft.pdf"
    counters = build_pdf(PAPER / "paper_draft.md", out)
    print(f"Written {out} ({counters['eq']} equations, {counters['fig']} figures, {counters['tables']} tables)")


if __name__ == "__main__":
    main()
