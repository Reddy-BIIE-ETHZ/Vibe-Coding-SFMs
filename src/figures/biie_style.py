"""biie_style.py — Shared figure-style template for all SFM paper figures.

Drop-in matplotlib style for Vibe-coding SFM paper figures. Every chat
producing figures (encoder, decoder, scaling, etc.) imports from here so
all figures share consistent typography, dimensions, palette, and PDF
output convention.

Usage:
    from calm.figures.biie_style import apply_style, BIIE, save_figure

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    axes[0].bar(targets, values_id, color=BIIE.BLUE_PURPLE)
    axes[1].bar(targets, values_ood, color="white",
                edgecolor=BIIE.BLUE_PURPLE, linewidth=1.5)
    save_figure(fig, "fig2_retrieval_bidirectional")  # → dtSFM-Figures/*.pdf + *.png

Design conventions (locked 2026-05-10 for the V-SFM paper portfolio):
  - DejaVu Sans / Arial / Helvetica fallback chain, 9 pt body
  - 2-panel ≈ 6.5 × 3.0 in at 300 DPI; 3-panel ≈ 9.0 × 3.0 in
  - Solid fill = ID / case study / STRONG / Class A
  - Open bar (white fill, colored edge) = OOD / control / WEAK / Class C
  - BIIE palette extracted from the institute logo gradient
  - PDF output with embedded fonts (fonttype 42 = TrueType embedded)
  - Numerical bar labels above bars, 0-1 decimals depending on range
  - Top/right spines hidden; bottom/left at 0.8 pt black; outward ticks
  - LEGENDS ARE AVOIDED. Prefer direct in-axis labels (panel titles,
    colored category text next to bars, or descriptive caption text).
    Use the `direct_label` helper. Only fall back to ax.legend() when
    direct labelling truly cannot fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# BIIE palette (from logo gradient)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Palette:
    # Primary palette — bold versions (locked 2026-05-10, replacing the
    # original BIIE-logo gradient which read as anemic at print scale).
    # Material-Design-style saturated tones that hold up on white.
    BLUE:        str = "#1565C0"   # bold deep blue   — STRONG / decoder / ID
    GREEN:       str = "#2E7D32"   # bold deep green  — MODERATE / anchor
    PURPLE:      str = "#6A1B9A"   # bold deep purple — WEAK / accent
    ALERT:       str = "#C62828"   # bold deep red    — thresholds / warnings
    GOLD:        str = "#F59E0B"   # bold amber       — extra series accent
                                    #                    when blue/green/purple
                                    #                    don't separate cleanly

    # Backward-compat aliases (old code that imported by historical names).
    TEAL:        str = "#2E7D32"   # = GREEN
    BLUE_PURPLE: str = "#1565C0"   # = BLUE
    MAGENTA:     str = "#6A1B9A"   # = PURPLE

    # Neutrals
    BLACK:       str = "#000000"
    GREY_DARK:   str = "#4D4D4D"
    GREY_MID:    str = "#888888"
    GREY_LIGHT:  str = "#DDDDDD"
    WHITE:       str = "#FFFFFF"

    # Semantic shortcuts (verdict / categorical mappings)
    STRONG:      str = "#1565C0"   # = BLUE
    MODERATE:    str = "#2E7D32"   # = GREEN
    WEAK:        str = "#6A1B9A"   # = PURPLE
    NEGATIVE:    str = "#888888"   # = GREY_MID — for OFF-TARGET-RISK / failures

    # ID vs OOD convention (for retrieval / pool-512 figures)
    ID_FILL:     str = "#1565C0"   # solid bold blue for in-distribution
    OOD_FILL:    str = "#FFFFFF"   # white fill ("open") for OOD
    OOD_EDGE:    str = "#1565C0"   # bold blue edge for OOD

    # 3-stop gradient for heatmaps / continuous data (cool → warm)
    GRADIENT:    tuple = ("#1565C0", "#2E7D32", "#C62828")


BIIE = _Palette()


# --------------------------------------------------------------------------- #
# Continuous colormap from the BIIE gradient (for heatmaps, density plots)
# --------------------------------------------------------------------------- #
def biie_colormap(name: str = "biie_gradient"):
    """Return a Matplotlib LinearSegmentedColormap interpolating the BIIE gradient."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(BIIE.GRADIENT))


# --------------------------------------------------------------------------- #
# Style application
# --------------------------------------------------------------------------- #
_RC_PARAMS = {
    # Typography
    "font.family":      ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size":          9.0,
    "axes.labelsize":     9.0,
    "axes.titlesize":    10.0,
    "axes.titleweight":  "bold",
    "xtick.labelsize":    8.0,
    "ytick.labelsize":    8.0,
    "legend.fontsize":    8.0,
    "figure.titlesize":  10.0,
    # Lines / spines
    "axes.linewidth":     0.8,
    "axes.edgecolor":     BIIE.BLACK,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    # Ticks
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.color":        BIIE.GREY_DARK,
    "ytick.color":        BIIE.GREY_DARK,
    # Lines / patches
    "lines.linewidth":    1.5,
    "patch.edgecolor":    BIIE.BLACK,
    "patch.linewidth":    0.8,
    # Grid
    "axes.grid":          False,
    "grid.color":         BIIE.GREY_LIGHT,
    "grid.linewidth":     0.5,
    # Figure
    "figure.dpi":         100,    # display dpi; save_figure uses 300
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "figure.facecolor":   BIIE.WHITE,
    "axes.facecolor":     BIIE.WHITE,
    # PDF font embedding (TrueType — type 42)
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    # Legend
    "legend.frameon":      False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.5,
}


def apply_style():
    """Set BIIE figure style globally on the current matplotlib session."""
    mpl.rcParams.update(_RC_PARAMS)


# --------------------------------------------------------------------------- #
# Standard figure-size shortcuts
# --------------------------------------------------------------------------- #
SIZE_1PANEL    = (3.5, 3.0)    # single-panel
SIZE_2PANEL    = (6.5, 3.0)    # two horizontal panels
SIZE_3PANEL    = (9.0, 3.0)    # three horizontal panels
SIZE_4PANEL_2x2 = (6.5, 6.0)   # 2x2 grid
SIZE_SQUARE     = (3.5, 3.5)   # square (heatmap / scatter)
SIZE_TALL       = (3.5, 5.0)   # tall single column
SIZE_WIDE       = (9.0, 4.5)   # wide single panel (for chemical-structure grid)


# --------------------------------------------------------------------------- #
# Save helper — exports both PDF and PNG with correct settings
# --------------------------------------------------------------------------- #
def save_figure(
    fig,
    stem: str,
    output_dir: Path | str = "dtSFM-Figures",
    formats: tuple = ("pdf", "png"),
    transparent: bool = False,
):
    """Save figure to <output_dir>/<stem>.{pdf,png} with paper-grade settings.

    PDF gets embedded TrueType fonts (pdf.fonttype = 42 in rcParams).
    PNG at 300 DPI from savefig.dpi.

    Returns dict mapping format → Path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt, transparent=transparent)
        paths[fmt] = p
    return paths


# --------------------------------------------------------------------------- #
# Bar-label helper — value labels above bars, matching Vibe-coding Fig 2 style
# --------------------------------------------------------------------------- #
def label_bars(ax, bars, values, fmt: str = "{:.0f}", padding: float = 1.0,
               fontsize: float = 8.0, color: str = "#000000"):
    """Place value labels above each bar.

    fmt examples:
        "{:.0f}"       → "65"
        "{:.0f} %"     → "65 %"
        "{:.2f}"       → "0.95"
    """
    for bar, val in zip(bars, values):
        if val is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + padding,
            fmt.format(val),
            ha="center", va="bottom",
            fontsize=fontsize, color=color,
        )


# --------------------------------------------------------------------------- #
# Direct-label helper — preferred replacement for ax.legend()
# --------------------------------------------------------------------------- #
def direct_label(ax, x, y, text: str, color: str, fontsize: float = 8.0,
                 weight: str = "bold", ha: str = "left", va: str = "center"):
    """Place a colored category label inline at (x, y) in data coordinates.

    Use this instead of ax.legend() to keep figures legend-free. Color the
    text in the same hue as the data series it describes.

    Example:
        direct_label(ax, x=7.2, y=4.0, text="STRONG",   color=BIIE.STRONG)
        direct_label(ax, x=7.2, y=2.5, text="MODERATE", color=BIIE.MODERATE)
    """
    ax.text(x, y, text, color=color, fontsize=fontsize, weight=weight,
            ha=ha, va=va)


# --------------------------------------------------------------------------- #
# Quick demo (run as `python -m calm.figures.biie_style` to verify install)
# --------------------------------------------------------------------------- #
def _demo():
    apply_style()
    import numpy as np

    targets = ["FLT3", "ALK", "MAP2K1", "MAP2K2", "F11", "CDK4", "CDK6", "PARP1"]
    strong   = [5, 2, 2, 3, 0, 0, 0, 0]
    moderate = [0, 2, 3, 0, 0, 0, 0, 0]
    weak     = [0, 1, 0, 0, 5, 0, 0, 1]
    risk     = [0, 0, 0, 2, 0, 5, 5, 4]

    x = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=SIZE_3PANEL)
    bottom = np.zeros(len(targets))
    series = [
        ("STRONG",          strong,   BIIE.STRONG),
        ("MODERATE",        moderate, BIIE.MODERATE),
        ("WEAK",            weak,     BIIE.MAGENTA),
        ("OFF-TARGET RISK", risk,     BIIE.NEGATIVE),
    ]
    for label, vals, color in series:
        ax.bar(x, vals, bottom=bottom, color=color,
               edgecolor=BIIE.BLACK, linewidth=0.5)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=0)
    ax.set_ylabel("Number of candidates")
    ax.set_title("§F.5.3 verdicts per target (n = 5 each)")
    ax.set_ylim(0, 6.5)

    # Direct labels (no legend): colored category text along the right edge.
    label_x = len(targets) - 0.4
    for i, (label, _, color) in enumerate(series):
        direct_label(ax, x=label_x, y=5.7 - 0.6 * i, text=label, color=color)

    paths = save_figure(fig, "biie_style_demo", output_dir="/tmp")
    print("Demo saved to:")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    _demo()
