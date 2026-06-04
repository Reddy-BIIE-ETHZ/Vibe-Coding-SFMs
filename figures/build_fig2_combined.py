"""Combined Figure 2 — bidirectional retrieval + external validation
+ mirror-image comparison.

Layout:
   Panel A (full width)  : Agent → Target  pool-512 R@1, all 6 SFMs
   Panel B (full width)  : Target → Agent  pool-512 R@1, all 6 SFMs
   Panel C | D | E | F   : (lower row, 4 sub-panels)
     C — mir-SFM vs seed-matching, stratified by canonical / non-canonical
     D — mhcSFM cascade on Gurung 2023 (NetMHCpan EL alone vs cascade)
     E — crisprSFM cascade on CRISPRoffT (Hamming alone vs cascade)
     F — Mirror-image: crisprSFM (Watson-Crick) vs mir-SFM (complex rules)

Style: blue (#3667a6) for SFM/cascade bars, white for baseline/standalone,
thin black borders, light-grey y-grid, sans-serif, no panel titles
(direction tags or small italic descriptors under x-axis instead).

All numeric values are audit-verified — see fig2_combined_data.csv for
per-bar provenance.
"""

from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Palette — single blue accent for all SFM/cascade bars
COL_SFM   = "#3667a6"
COL_BASE  = "#ffffff"
EDGE      = "black"


# =============================================================================
# Panel A/B data (bidirectional pool-512 retrieval)
# =============================================================================

@dataclass
class RetrievalRow:
    sfm: str
    direction: str       # "agent_to_target" or "target_to_agent"
    threshold: str
    r1_mean: float
    r1_sd: float | None


SFM_ORDER = ["tSFM", "eSFM", "mhcSFM", "crisprSFM", "mir-SFM", "dtSFM"]

OOD_THRESHOLD = {
    "tSFM":      "mmseqs_060",
    "eSFM":      "mmseqs_080",
    "mhcSFM":    "zero_shot_rare_alleles",
    "crisprSFM": "hamming_080",
    "mir-SFM":   "by_mirna_holdout",
    "dtSFM":     "mmseqs_080",
}

DIRECTION_TAGS = {
    "agent_to_target": {
        "tSFM":      "TF → DNA",
        "eSFM":      "E → S",
        "mhcSFM":    "peptide → allele",
        "crisprSFM": "gRNA → OT",
        "mir-SFM":   "miRNA → target",
        "dtSFM":     "drug → target",
    },
    "target_to_agent": {
        "tSFM":      "DNA → TF",
        "eSFM":      "S → E",
        "mhcSFM":    "allele → peptide",
        "crisprSFM": "OT → gRNA",
        "mir-SFM":   "target → miRNA",
        "dtSFM":     "target → drug",
    },
}

RETRIEVAL: list[RetrievalRow] = [
    # tSFM — bundle CL-2/CL-3 (ID folds 0-3); mmseqs_060 5 folds
    RetrievalRow("tSFM", "agent_to_target", "identity_100", 83.1, 3.0),
    RetrievalRow("tSFM", "target_to_agent", "identity_100", 25.8, 2.5),
    RetrievalRow("tSFM", "agent_to_target", "mmseqs_060",   53.7, 4.9),
    RetrievalRow("tSFM", "target_to_agent", "mmseqs_060",    8.6, 2.0),
    # eSFM — bundle CL-5/6/7/8 (writeup table)
    RetrievalRow("eSFM", "agent_to_target", "identity_100", 86.3, 1.5),
    RetrievalRow("eSFM", "target_to_agent", "identity_100", 61.8, 0.4),
    RetrievalRow("eSFM", "agent_to_target", "mmseqs_080",   85.0, 1.5),
    RetrievalRow("eSFM", "target_to_agent", "mmseqs_080",   58.6, 2.2),
    # mhcSFM — 5-fold from archival log + 14-allele zero-shot holdout
    RetrievalRow("mhcSFM", "agent_to_target", "identity_100", 65.1, 1.9),
    RetrievalRow("mhcSFM", "target_to_agent", "identity_100", 93.3, 2.6),
    RetrievalRow("mhcSFM", "agent_to_target", "zero_shot_rare_alleles", 53.2, None),
    RetrievalRow("mhcSFM", "target_to_agent", "zero_shot_rare_alleles", 95.4, None),
    # crisprSFM — log summary 5-fold (ddof=0)
    RetrievalRow("crisprSFM", "agent_to_target", "identity_100", 95.4, 6.5),
    RetrievalRow("crisprSFM", "target_to_agent", "identity_100", 74.8, 20.2),
    RetrievalRow("crisprSFM", "agent_to_target", "hamming_080",  87.0, 6.3),
    RetrievalRow("crisprSFM", "target_to_agent", "hamming_080",  55.5, 18.6),
    # mir-SFM — identity_100 (ID) and mmseqs_080 (OOD by-miRNA holdout)
    RetrievalRow("mir-SFM", "agent_to_target", "identity_100", 98.0, 0.6),
    RetrievalRow("mir-SFM", "target_to_agent", "identity_100", 25.4, 4.7),
    RetrievalRow("mir-SFM", "agent_to_target", "by_mirna_holdout", 99.6, 0.4),
    RetrievalRow("mir-SFM", "target_to_agent", "by_mirna_holdout", 68.8, 4.4),
    # dtSFM v2 — bundle §3.1 (5-fold), §3.2 (single fold)
    RetrievalRow("dtSFM", "agent_to_target", "identity_100", 27.7, 3.3),
    RetrievalRow("dtSFM", "target_to_agent", "identity_100", 60.5, 8.7),
    RetrievalRow("dtSFM", "agent_to_target", "mmseqs_080",   46.0, None),
    RetrievalRow("dtSFM", "target_to_agent", "mmseqs_080",   95.7, None),
]


def lookup_retrieval(sfm: str, direction: str, threshold: str):
    for r in RETRIEVAL:
        if r.sfm == sfm and r.direction == direction and r.threshold == threshold:
            return r
    return None


def draw_retrieval_panel(ax, direction: str, panel_label: str) -> None:
    n = len(SFM_ORDER)
    bar_w = 0.36
    centres = np.arange(n)

    for i, sfm in enumerate(SFM_ORDER):
        id_row  = lookup_retrieval(sfm, direction, "identity_100")
        ood_row = lookup_retrieval(sfm, direction, OOD_THRESHOLD[sfm])
        xs   = [centres[i] - bar_w / 2, centres[i] + bar_w / 2]
        rows = [id_row, ood_row]
        cols = [COL_SFM, COL_BASE]
        for x, r, c in zip(xs, rows, cols):
            if r is None: continue
            ax.bar(x, r.r1_mean, width=bar_w * 0.92,
                   color=c, edgecolor=EDGE, linewidth=0.7, zorder=3)
            if r.r1_sd is not None:
                ax.errorbar(x, r.r1_mean, yerr=r.r1_sd,
                            fmt="none", ecolor=EDGE,
                            elinewidth=0.7, capsize=2.6, capthick=0.7, zorder=4)

    tags = DIRECTION_TAGS[direction]
    ax.set_xticks(centres)
    ax.set_xticklabels([""] * n)
    for i, sfm in enumerate(SFM_ORDER):
        ax.text(centres[i], -0.05, sfm,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9)
        ax.text(centres[i], -0.16, f"({tags[sfm]})",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, style="italic",
                color="#444444")

    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("R@1 (%)", fontsize=9.5)

    panel_label_box(ax, panel_label)
    plain_axes(ax)


# =============================================================================
# Panel C: mir-SFM vs seed-matching (stratified canonical / non-canonical)
# =============================================================================

PANEL_C_DATA = [
    # (group, bar_label, value, is_sfm)
    ("Canonical",     "mir-SFM",         98.0, True),
    ("Canonical",     "Seed matching",    9.0, False),
    ("Non-canonical", "mir-SFM",         98.0, True),
    ("Non-canonical", "Seed matching",    6.1, False),
]


def draw_panel_C(ax) -> None:
    groups = ["Canonical", "Non-canonical"]
    bar_labels = ["mir-SFM", "Seed matching"]
    bar_w = 0.36
    centres = np.arange(len(groups))

    for i, grp in enumerate(groups):
        for j, lab in enumerate(bar_labels):
            val = next(d[2] for d in PANEL_C_DATA if d[0] == grp and d[1] == lab)
            is_sfm = next(d[3] for d in PANEL_C_DATA if d[0] == grp and d[1] == lab)
            x = centres[i] + (j - 0.5) * bar_w
            ax.bar(x, val, width=bar_w * 0.92,
                   color=COL_SFM if is_sfm else COL_BASE,
                   edgecolor=EDGE, linewidth=0.7, zorder=3)

    ax.set_xticks(centres)
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("R@1 (%)", fontsize=9)

    # In-panel legend
    handles = [
        mpatches.Patch(facecolor=COL_SFM,  edgecolor=EDGE,
                       linewidth=0.7, label="mir-SFM"),
        mpatches.Patch(facecolor=COL_BASE, edgecolor=EDGE,
                       linewidth=0.7, label="Seed matching"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), fontsize=7,
              frameon=False, ncol=2, columnspacing=1.4,
              handlelength=1.2, handleheight=0.9)

    panel_label_box(ax, "C")
    plain_axes(ax)


# =============================================================================
# Panel D: mhcSFM cascade (Gurung MS-presentation)
# =============================================================================

PANEL_D_DATA = [
    ("NetMHCpan\nEL",        16.9, False),
    ("EL ∩ mhcSFM\ncascade", 30.0, True),
]


def draw_panel_D(ax) -> None:
    bar_w = 0.4
    xs = np.arange(len(PANEL_D_DATA))
    for x, (lab, val, is_sfm) in zip(xs, PANEL_D_DATA):
        ax.bar(x, val, width=bar_w,
               color=COL_SFM if is_sfm else COL_BASE,
               edgecolor=EDGE, linewidth=0.7, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([""] * len(PANEL_D_DATA))
    for x, (lab, _, _) in zip(xs, PANEL_D_DATA):
        ax.text(x, -0.04, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, linespacing=1.2)
    ax.set_xlim(-0.6, len(PANEL_D_DATA) - 0.4)
    ax.set_ylim(0, 50)
    ax.set_ylabel("MS-presentation\nprecision (%)", fontsize=9,
                  linespacing=1.2)

    panel_label_box(ax, "D")
    plain_axes(ax)


# =============================================================================
# Panel E: crisprSFM cascade (CRISPRoffT ≤4 mm)
# =============================================================================

PANEL_E_DATA = [
    ("Hamming\nalone",            33.2, False),
    ("Hamming →\ncrisprSFM",      94.0, True),
]


def draw_panel_E(ax) -> None:
    bar_w = 0.4
    xs = np.arange(len(PANEL_E_DATA))
    for x, (lab, val, is_sfm) in zip(xs, PANEL_E_DATA):
        ax.bar(x, val, width=bar_w,
               color=COL_SFM if is_sfm else COL_BASE,
               edgecolor=EDGE, linewidth=0.7, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([""] * len(PANEL_E_DATA))
    for x, (lab, _, _) in zip(xs, PANEL_E_DATA):
        ax.text(x, -0.04, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, linespacing=1.2)
    ax.set_xlim(-0.6, len(PANEL_E_DATA) - 0.4)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Precision (%)", fontsize=9)

    panel_label_box(ax, "E")
    plain_axes(ax)


# =============================================================================
# Panel F: Mirror-image (crisprSFM vs mir-SFM)
# =============================================================================

PANEL_F_DATA = [
    ("crisprSFM", "Watson–Crick\ncomplementarity",
        7.0, None, 45.8, None),           # SFM 7.0, baseline 45.8
    ("mir-SFM",   "complex, context-\ndependent rules",
        98.0, 0.6, 6.8, None),
]


def draw_panel_F(ax) -> None:
    bar_w = 0.36
    centres = np.arange(len(PANEL_F_DATA))
    for i, (grp, sub, sfm_val, sfm_sd, base_val, base_sd) in enumerate(PANEL_F_DATA):
        x_s = centres[i] - bar_w / 2
        x_b = centres[i] + bar_w / 2
        ax.bar(x_s, sfm_val, width=bar_w * 0.92,
               color=COL_SFM, edgecolor=EDGE, linewidth=0.7, zorder=3)
        if sfm_sd is not None:
            ax.errorbar(x_s, sfm_val, yerr=sfm_sd, fmt="none",
                        ecolor=EDGE, elinewidth=0.7, capsize=2.4,
                        capthick=0.7, zorder=4)
        ax.bar(x_b, base_val, width=bar_w * 0.92,
               color=COL_BASE, edgecolor=EDGE, linewidth=0.7, zorder=3)
        if base_sd is not None:
            ax.errorbar(x_b, base_val, yerr=base_sd, fmt="none",
                        ecolor=EDGE, elinewidth=0.7, capsize=2.4,
                        capthick=0.7, zorder=4)
        # Δ label above the taller bar
        gap = sfm_val - base_val
        sign = "+" if gap > 0 else "−"
        ax.text(centres[i], max(sfm_val, base_val) + 5,
                f"Δ = {sign}{abs(gap):.0f} pp",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(centres)
    ax.set_xticklabels([""] * len(PANEL_F_DATA))
    for i, (grp, sub, *_) in enumerate(PANEL_F_DATA):
        ax.text(centres[i], -0.05, grp,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5)
        ax.text(centres[i], -0.16, sub,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, style="italic",
                color="#444444", linespacing=1.2)

    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("R@1 (%)", fontsize=9)

    # In-panel legend
    handles = [
        mpatches.Patch(facecolor=COL_SFM,  edgecolor=EDGE,
                       linewidth=0.7, label="SFM"),
        mpatches.Patch(facecolor=COL_BASE, edgecolor=EDGE,
                       linewidth=0.7, label="Domain baseline"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.25), fontsize=7,
              frameon=False, ncol=2, columnspacing=1.4,
              handlelength=1.2, handleheight=0.9)

    panel_label_box(ax, "F")
    plain_axes(ax)


# =============================================================================
# Shared style helpers
# =============================================================================

def panel_label_box(ax, label: str) -> None:
    ax.text(-0.22, 1.14, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="left", va="top")


def plain_axes(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(width=0.5, labelsize=8.5)
    ax.yaxis.grid(True, color="#d8d8d8", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


# =============================================================================
# Master figure assembly
# =============================================================================

def build_figure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(10.0, 10.5))
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[1.0, 1.0, 1.05],
        hspace=0.75, wspace=0.70,
        left=0.075, right=0.985, top=0.94, bottom=0.07,
    )

    axA = fig.add_subplot(gs[0, :])    # full width
    axB = fig.add_subplot(gs[1, :])    # full width
    axC = fig.add_subplot(gs[2, 0])
    axD = fig.add_subplot(gs[2, 1])
    axE = fig.add_subplot(gs[2, 2])
    axF = fig.add_subplot(gs[2, 3])

    draw_retrieval_panel(axA, "agent_to_target", "A")
    draw_retrieval_panel(axB, "target_to_agent", "B")
    draw_panel_C(axC)
    draw_panel_D(axD)
    draw_panel_E(axE)
    draw_panel_F(axF)

    # Shared legend for A/B at the very top
    handles = [
        mpatches.Patch(facecolor=COL_SFM, edgecolor=EDGE,
                       linewidth=0.7, label="In-distribution"),
        mpatches.Patch(facecolor=COL_BASE, edgecolor=EDGE,
                       linewidth=0.7, label="Out-of-distribution"),
    ]
    fig.legend(handles=handles,
               loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=2, frameon=False, fontsize=9,
               handlelength=1.6, handleheight=1.0, columnspacing=2.4)

    fig.savefig(OUT / "fig2_combined.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_combined.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig2_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "sfm_or_group", "bar_label", "value",
                    "sd", "source"])
        for r in RETRIEVAL:
            panel = "A" if r.direction == "agent_to_target" else "B"
            w.writerow([panel, r.sfm,
                        "ID" if r.threshold == "identity_100" else "OOD",
                        f"{r.r1_mean:.1f}",
                        "" if r.r1_sd is None else f"{r.r1_sd:.1f}",
                        f"{r.threshold}"])
        for grp, lab, val, is_sfm in PANEL_C_DATA:
            w.writerow(["C", grp, lab, f"{val:.1f}", "",
                        "mirSFM bundle CL-6 / §3.1"])
        for lab, val, is_sfm in PANEL_D_DATA:
            w.writerow(["D", "Gurung 2023 MS", lab, f"{val:.1f}", "",
                        "mhcSFM bundle CL-6"])
        for lab, val, is_sfm in PANEL_E_DATA:
            w.writerow(["E", "CRISPRoffT ≤4 mm", lab, f"{val:.1f}", "",
                        "crisprSFM bundle CL-10 + cascade log"])
        for grp, sub, sv, ssd, bv, bsd in PANEL_F_DATA:
            w.writerow(["F", grp, "SFM", f"{sv:.1f}",
                        "" if ssd is None else f"{ssd:.1f}",
                        "audit CL"])
            w.writerow(["F", grp, "Baseline", f"{bv:.1f}",
                        "" if bsd is None else f"{bsd:.1f}",
                        "audit CL"])


def main() -> None:
    write_csv(OUT / "fig2_combined_data.csv")
    build_figure()
    print("wrote:")
    for f in ("fig2_combined_data.csv",
              "fig2_combined.pdf",
              "fig2_combined.svg",
              "fig2_combined.png"):
        p = OUT / f
        print(f"  {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
