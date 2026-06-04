"""Combined Figure 3 — training dynamics + physics signatures.

Layout:
   Panel A (top, full width)    : 2x3 small multiples of training
                                   convergence curves, one per SFM
   Panel B (lower-left)         : val − train accuracy gap at best epoch
                                   (bars, +ve = inverted gap = physics signature)
   Panel C (lower-right)        : agent → target R@1 vs OOD threshold
                                   stringency, 4 SFMs with multi-threshold data

Style: blue (#3667a6) for validation curve / inverted-gap bars / eSFM line;
other colours from the existing palette for the C line plot.
"""

from __future__ import annotations
import csv
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent

# Palette
COL_TRAIN = "#3f3f3f"
COL_VAL   = "#3667a6"
COL_MARK  = "#3667a6"
COL_INV   = "#3667a6"     # blue for inverted-gap bars (physics signature)
COL_STD   = "#ffffff"     # white for standard-DL gap
EDGE      = "black"


# =============================================================================
# Panel A data: per-epoch training logs (fold 0, identity_100)
# =============================================================================

@dataclass
class SfmTrain:
    name:  str
    log:   Path
    total_epochs: int
    best_epoch:   int   = 0
    best_train:   float = 0.0
    best_val:     float = 0.0
    epochs:    list[int]   = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc:   list[float] = field(default_factory=list)


SFMS: list[SfmTrain] = [
    SfmTrain("tSFM",      ROOT / "audit/tsfm/archival_logs/tsfm_full_64763913_0.out",      100),
    SfmTrain("eSFM",      ROOT / "logs/esfm/esfm_full_62858369_0.out",                     100),
    SfmTrain("mhcSFM",    ROOT / "audit/mhcsfm/archival_logs/mhcsfm_full_62515934_0.out",  100),
    SfmTrain("crisprSFM", ROOT / "audit/crisprSFM/archival_logs/crispr_hn_train_62512408.out", 100),
    SfmTrain("mir-SFM",   ROOT / "audit/mirsfm/archival_logs/mirsfm_full_62523321_0.out",  100),
    SfmTrain("dtSFM",     ROOT / "audit/dtsfm_v2/archival_logs/dtsfm2_full_63170839_0.out", 100),
]

LINE_RE = re.compile(
    r"Epoch\s+(\d+)\s+(train|val)[^:]*:\s.*Acc_avg=([0-9.]+)",
    re.IGNORECASE,
)


def parse_log(path: Path) -> tuple[list[int], list[float], list[float]]:
    """Return (epochs, train_acc%, val_acc%) for the first training run."""
    train: dict[int, float] = {}
    val:   dict[int, float] = {}
    seen_zero = False
    with path.open() as fh:
        for line in fh:
            m = LINE_RE.search(line)
            if not m: continue
            ep = int(m.group(1)); split = m.group(2).lower()
            acc = float(m.group(3)) * 100.0
            if split == "train" and ep == 0:
                if seen_zero: break
                seen_zero = True
            (train if split == "train" else val)[ep] = acc
    epochs = sorted(set(train) & set(val))
    return epochs, [train[e] for e in epochs], [val[e] for e in epochs]


def load_panel_A() -> None:
    for s in SFMS:
        if not s.log.exists():
            print(f"WARN: missing log for {s.name}")
            continue
        s.epochs, s.train_acc, s.val_acc = parse_log(s.log)
        best_idx = max(range(len(s.val_acc)), key=lambda i: s.val_acc[i])
        s.best_epoch = s.epochs[best_idx]
        s.best_train = s.train_acc[best_idx]
        s.best_val   = s.val_acc[best_idx]


# =============================================================================
# Panel C data: OOD decay curves (4 SFMs × 4 thresholds)
# =============================================================================

@dataclass
class CurvePoint:
    threshold: str
    r1: float
    sd: float | None
    lv_pass: bool


@dataclass
class SfmCurve:
    name: str
    points: list[CurvePoint]
    color: str


X_ORDER_C = ["ID", "80%", "60%", "40%"]

CURVES = [
    SfmCurve("tSFM", [
        CurvePoint("ID",  83.1, 3.0, True),
        CurvePoint("80%", 54.7, 5.1, True),
        CurvePoint("60%", 53.7, 4.9, True),
        CurvePoint("40%", 48.4, 4.3, True),
    ], color="#2b2b2b"),
    SfmCurve("eSFM", [
        CurvePoint("ID",  86.3, 1.5, True),
        CurvePoint("80%", 85.0, 1.5, True),
        CurvePoint("60%", 81.0, 1.9, False),
        CurvePoint("40%", 66.4, 3.3, False),
    ], color="#3667a6"),
    SfmCurve("crisprSFM", [
        CurvePoint("ID",  95.4, 6.5, True),
        CurvePoint("80%", 86.9, 6.3, True),
        CurvePoint("60%", 89.0, 5.5, False),
        CurvePoint("40%", 89.6, 7.8, False),
    ], color="#c4762b"),
    SfmCurve("dtSFM", [
        CurvePoint("ID",  27.7, 3.7, True),
        CurvePoint("80%", 46.0, None,True),
        CurvePoint("60%", 41.8, 6.8, True),
        CurvePoint("40%", 41.3, 21.1,True),
    ], color="#7a3b8f"),
]


# =============================================================================
# Helpers
# =============================================================================

def plain_axes(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(width=0.5, labelsize=7.5)
    ax.yaxis.grid(True, color="#d8d8d8", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


# =============================================================================
# Panel drawing
# =============================================================================

def draw_panel_A(fig, gs_top) -> None:
    """2x3 small-multiples grid of convergence curves."""
    inner = gs_top.subgridspec(2, 3, hspace=0.55, wspace=0.35)
    for i, s in enumerate(SFMS):
        ax = fig.add_subplot(inner[i // 3, i % 3])
        budget = 100.0 * np.array(s.epochs) / s.total_epochs
        ax.plot(budget, s.train_acc, color=COL_TRAIN,
                linewidth=1.1, linestyle="-", zorder=2)
        ax.plot(budget, s.val_acc, color=COL_VAL,
                linewidth=1.3, linestyle=":", zorder=3)
        x_best = 100.0 * s.best_epoch / s.total_epochs
        ax.plot([x_best], [s.best_val], marker="o",
                markersize=5.0, markerfacecolor="white",
                markeredgecolor=COL_MARK, markeredgewidth=1.2, zorder=4)
        ax.text(x_best, s.best_val + 7, f"ep {s.best_epoch}",
                ha="center", va="bottom", fontsize=6.5, color=COL_MARK)

        ax.set_title(s.name, fontsize=9, pad=3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([0, 50, 100])
        if i % 3 == 0:
            ax.set_ylabel("Batch prediction\naccuracy (%)",
                          fontsize=8, linespacing=1.2)
        if i // 3 == 1:
            ax.set_xlabel("Training budget (%)", fontsize=8)
        plain_axes(ax)


def draw_panel_B(ax) -> None:
    """Val − Train accuracy gap at best epoch."""
    names = [s.name for s in SFMS]
    gaps  = [s.best_val - s.best_train for s in SFMS]
    centres = np.arange(len(names))
    for i, (s, gap) in enumerate(zip(SFMS, gaps)):
        col = COL_INV if gap > 0 else COL_STD
        ax.bar(centres[i], gap, width=0.62,
               color=col, edgecolor=EDGE, linewidth=0.7, zorder=3)
        offset = 3.5 if gap >= 0 else -3.5
        va = "bottom" if gap >= 0 else "top"
        ax.text(centres[i], gap + offset, f"{gap:+.1f}",
                ha="center", va=va, fontsize=7.5, color=EDGE)

    ax.axhline(0, color=EDGE, linewidth=0.7, zorder=2)
    ax.set_xticks(centres)
    ax.set_xticklabels(names, fontsize=8.5, rotation=30, ha="right")
    ax.set_ylabel("Val − Train accuracy (pp)", fontsize=9.5)
    ax.set_ylim(-95, 35)
    plain_axes(ax)


def draw_panel_C(ax) -> None:
    x_pos = np.arange(len(X_ORDER_C))
    for c in CURVES:
        xs = [X_ORDER_C.index(p.threshold) for p in c.points]
        ys = [p.r1 for p in c.points]
        ax.plot(xs, ys, color=c.color, linestyle="-",
                linewidth=1.3, zorder=3, label=c.name)
        for p, x, y in zip(c.points, xs, ys):
            face = c.color if p.lv_pass else "white"
            ax.plot(x, y, marker="o", markersize=6,
                    markerfacecolor=face, markeredgecolor=c.color,
                    markeredgewidth=1.1, zorder=4)
            if p.sd is not None:
                ax.errorbar(x, y, yerr=p.sd, fmt="none",
                            ecolor=c.color, elinewidth=0.7,
                            capsize=2.4, capthick=0.7, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(X_ORDER_C, fontsize=9)
    ax.set_xlim(-0.3, len(X_ORDER_C) - 0.7)
    ax.set_xlabel("Clustering threshold\n(identity_100 → MMseqs / Hamming OOD)",
                  fontsize=8.5, linespacing=1.2)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Agent → Target  R@1 (%)", fontsize=9.5)
    plain_axes(ax)

    handles = []
    for c in CURVES:
        handles.append(mlines.Line2D(
            [], [], color=c.color, linestyle="-",
            marker="o", markersize=6,
            markerfacecolor=c.color, markeredgecolor=c.color,
            linewidth=1.3, label=c.name,
        ))
    handles.append(mlines.Line2D([], [], linestyle="none", marker="o",
                                 markersize=6, markerfacecolor="black",
                                 markeredgecolor="black", label="LV PASS"))
    handles.append(mlines.Line2D([], [], linestyle="none", marker="o",
                                 markersize=6, markerfacecolor="white",
                                 markeredgecolor="black",
                                 markeredgewidth=1.1, label="LV FAIL"))
    ax.legend(handles=handles, loc="lower right",
              bbox_to_anchor=(0.99, 0.02), fontsize=6.5, frameon=False,
              handlelength=2.0, handleheight=0.8, labelspacing=0.3,
              ncol=2, columnspacing=1.2)


# =============================================================================
# Master assembly
# =============================================================================

def panel_label(fig, x, y, label):
    fig.text(x, y, label, fontsize=12, fontweight="bold",
             ha="left", va="top")


def build_figure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(9.5, 9.5))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.55, 1.0],
        width_ratios=[1.0, 1.05],
        hspace=0.55, wspace=0.30,
        left=0.085, right=0.98, top=0.93, bottom=0.10,
    )

    # Panel A spans both columns of row 0
    gs_A = gs[0, :]
    draw_panel_A(fig, gs_A)

    # Panels B and C in row 1
    ax_B = fig.add_subplot(gs[1, 0])
    ax_C = fig.add_subplot(gs[1, 1])
    draw_panel_B(ax_B)
    draw_panel_C(ax_C)

    # Top-level legend for Panel A curves
    legend_handles = [
        mlines.Line2D([], [], color=COL_TRAIN, linewidth=1.4,
                      linestyle="-", label="Train"),
        mlines.Line2D([], [], color=COL_VAL, linewidth=1.6,
                      linestyle=":", label="Validation"),
        mlines.Line2D([], [], color="none", marker="o",
                      markerfacecolor="white", markeredgecolor=COL_MARK,
                      markersize=6, label="Best-val epoch"),
    ]
    fig.legend(handles=legend_handles,
               loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=3, frameon=False, fontsize=9,
               handlelength=1.6, columnspacing=2.0)

    # Panel labels (figure coords)
    panel_label(fig, 0.015, 0.945, "A")
    panel_label(fig, 0.015, 0.435, "B")
    panel_label(fig, 0.515, 0.435, "C")

    fig.savefig(OUT / "fig3_combined.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_combined.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig3_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "SFM", "x_value", "y_value", "sd",
                    "extra", "source"])
        # Panel A: per-epoch
        for s in SFMS:
            for ep, tr, vl in zip(s.epochs, s.train_acc, s.val_acc):
                w.writerow(["A", s.name, ep, f"{vl:.2f}", "",
                            f"train={tr:.2f}", str(s.log.name)])
        # Panel B: gaps
        for s in SFMS:
            gap = s.best_val - s.best_train
            w.writerow(["B", s.name, "best_epoch_gap",
                        f"{gap:.2f}", "",
                        f"best_ep={s.best_epoch} train={s.best_train:.2f} "
                        f"val={s.best_val:.2f}",
                        str(s.log.name)])
        # Panel C: OOD decay points
        for c in CURVES:
            for p in c.points:
                w.writerow(["C", c.name, p.threshold, f"{p.r1:.1f}",
                            "" if p.sd is None else f"{p.sd:.1f}",
                            f"LV={'PASS' if p.lv_pass else 'FAIL'}",
                            "audit bundle"])


def main() -> None:
    load_panel_A()
    write_csv(OUT / "fig3_combined_data.csv")
    build_figure()
    print("wrote:")
    for f in ("fig3_combined_data.csv",
              "fig3_combined.pdf",
              "fig3_combined.svg",
              "fig3_combined.png"):
        p = OUT / f
        print(f"  {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
