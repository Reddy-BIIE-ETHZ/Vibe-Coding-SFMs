# V-SFM Figures — Shared Style Module

`biie_style.py` is the **single source of truth** for the typography, palette,
sizing, and PDF-export conventions used across every V-SFM paper figure.
Every chat producing figures (encoder, decoder, scaling, ablations,
supplementary) imports from here so the paper portfolio looks coherent.

Conventions locked **2026-05-10**. Do not change without coordinating across
all SFM chats.

---

## Quick start

```python
from calm.figures.biie_style import (
    apply_style, BIIE, save_figure, label_bars, direct_label,
    SIZE_2PANEL, SIZE_3PANEL,
)
import matplotlib.pyplot as plt

apply_style()
fig, axes = plt.subplots(1, 2, figsize=SIZE_2PANEL)

# ID bars: solid blue-purple
axes[0].bar(targets, r1_id, color=BIIE.ID_FILL)
# OOD bars: white fill, blue-purple edge
axes[1].bar(targets, r1_ood, color=BIIE.OOD_FILL,
            edgecolor=BIIE.OOD_EDGE, linewidth=1.5)

save_figure(fig, "fig2_retrieval_bidirectional")  # → dtSFM-Figures/*.pdf + *.png
```

---

## Palette (bold; locked 2026-05-10, replacing original BIIE-logo gradient)

| Token | Hex | Used for |
| --- | --- | --- |
| `BIIE.BLUE` | `#1565C0` | **Workhorse.** STRONG / decoder / Class A / ID / case study |
| `BIIE.GREEN` | `#2E7D32` | MODERATE / anchor / Class B |
| `BIIE.PURPLE` | `#6A1B9A` | WEAK / Class C / accent |
| `BIIE.GOLD` | `#F59E0B` | Extra series accent — use when blue / green / purple don't separate cleanly in a 3+-series chart |
| `BIIE.ALERT` | `#C62828` | Threshold lines, pass/fail cutoffs, warning callouts |
| `BIIE.GREY_MID` | `#888888` | OFF-TARGET RISK / negative controls |
| `BIIE.BLACK` | `#000000` | Axes, edges |
| `BIIE.GRADIENT` | (blue, green, red) | Heatmap colormap; call `biie_colormap()` |

Old names (`BIIE.TEAL`, `BIIE.BLUE_PURPLE`, `BIIE.MAGENTA`) are kept as
backward-compat aliases to the new bold tones — existing code keeps
working but new figures should prefer the explicit `BLUE` / `GREEN` /
`PURPLE` names.

Semantic shortcuts: `BIIE.STRONG`, `BIIE.MODERATE`, `BIIE.WEAK`, `BIIE.NEGATIVE`,
`BIIE.ID_FILL`, `BIIE.OOD_FILL`, `BIIE.OOD_EDGE`.

---

## Sizes (locked)

| Shortcut | Inches | When |
| --- | --- | --- |
| `SIZE_1PANEL` | 3.5 × 3.0 | single small panel |
| `SIZE_2PANEL` | 6.5 × 3.0 | two horizontal panels (Fig 2 idiom) |
| `SIZE_3PANEL` | 9.0 × 3.0 | three horizontal panels |
| `SIZE_4PANEL_2x2` | 6.5 × 6.0 | 2×2 grid |
| `SIZE_SQUARE` | 3.5 × 3.5 | heatmap / scatter |
| `SIZE_TALL` | 3.5 × 5.0 | tall single column |
| `SIZE_WIDE` | 9.0 × 4.5 | chemical-structure grid, AF3 overlay row |

Page width is 7.0–7.2 in for two-column journals; never exceed 7.0 in for a
"single-row" figure. Use 9.0-in widths only for figures that will stretch
across both columns.

---

## Bar idioms

**ID vs OOD (Vibe-coding Fig 2 style):**

```python
ax.bar(x_id,  v_id,  color=BIIE.ID_FILL)
ax.bar(x_ood, v_ood, color=BIIE.OOD_FILL,
       edgecolor=BIIE.OOD_EDGE, linewidth=1.5)
```

**Verdict stacks (decoder §F.5 / cofold panels):** stack STRONG (blue-purple)
on bottom, MODERATE (teal), WEAK (magenta), OFF-TARGET RISK (grey) on top.
Numerical totals via `label_bars(...)`.

**Numerical labels above bars:**

```python
bars = ax.bar(x, vals, color=BIIE.STRONG)
label_bars(ax, bars, vals, fmt="{:.0f} %", padding=1.0)
```

---

## No legends

**Avoid `ax.legend()` whenever possible.** Reasons:

- Legends eat ~15% of figure area for redundant info.
- Direct labels are easier to read at print scale.
- Captions can carry category definitions without extra ink.

**Preferred patterns:**

1. Color the panel title or axis label in the series color.
2. Inline colored text via `direct_label(ax, x, y, "STRONG", BIIE.STRONG)`.
3. Caption text. Define each color once in the figure caption.

Only fall back to `ax.legend(frameon=False)` when none of the above fit.

---

## Output

- **Default dir:** `dtSFM-Figures/` (renamed 2026-05-10 from `V-SFM-Figures/` to keep dtSFM-paper outputs separate from cross-SFM portfolio). Other SFM chats should override `output_dir` to their own folder (e.g. `eSFM-Figures/`, `tSFM-Figures/`) — change one keyword at the call site, not the template default.
- **Naming:** `fig<section>_<panel>_<slug>.pdf`, e.g. `fig5_1_workflow.pdf`,
  `fig2_retrieval_bidirectional.pdf`. Supplementary use `figS<num>_...`.
- `save_figure(fig, stem)` writes both PDF (embedded TrueType, `pdf.fonttype=42`)
  and PNG at 300 DPI with tight bbox.

---

## Verifying the install (any chat)

```bash
PYTHONPATH=src python3 -m calm.figures.biie_style
# → /tmp/biie_style_demo.{pdf,png}
```

If the demo renders a 3-panel-width §F.5.3-style stacked verdict bar chart
with blue-purple / teal / magenta / grey bars and **no legend** (categories
labeled inline at the right edge), the install is correct.

---

## Structural overlay rendering (AF3 / Boltz-2 cofold figures)

For figures that show protein–ligand overlay panels from AF3 or Boltz-2
cofold outputs (decoder §F.5.5, encoder §5.3 e5, and future
crystallography panels, etc.), see `calm.figures.af3_overlay`. Reference
implementations:

| Reference figure | Render script | Composition script |
|---|---|---|
| Decoder F5.5 (4 targets × 17 candidates + 4 multi-anchor refs) | `scripts/figures/render_fig5_5_overlays.py` | `scripts/figures/make_fig5_5_structural_overlays.py` |

**Key techniques** — copy these patterns when adapting for a new figure:

1. **`cmd.cealign(target_obj+" and polymer", source_obj+" and polymer")`** — align two CIFs by protein backbone before overlaying ligands. Both AF3 and Boltz-2 outputs use chain `A` for the protein and chain `L` (or `B`) for the small molecule; PyMOL's `polymer` selector matches the protein chain regardless of letter.
2. **Color carbons only** — `cmd.color("0x1565C0", obj+" and not polymer and elem C")`. Leaving heteroatoms in element colors (N blue, O red, F green, S yellow) preserves chemistry readability while making the candidate vs anchor identity unambiguous from the C-skeleton color.
3. **`cmd.orient(ligand_sel) → cmd.turn("x", -20) → cmd.zoom(ligand_sel, 3.0)`** — the camera recipe for tight pocket-focused views without whitespace. `orient` finds the ligand's principal axes, `turn` adds a slight tilt for 3D depth, `zoom` with buffer ≈ 3 Å keeps the pocket framed.
4. **Saved-camera reuse across cells** — `view = cmd.get_view()` after rendering the column-header (multi-anchor) cell; `cmd.set_view(view)` in each candidate cell of that column. Guarantees the candidate cells read in the same orientation as the header. Works because CEALIGN keeps the anchor protein at its native CIF coords across both renders.
5. **Per-cell extra-turn overrides** — when one cell needs a different angle because a foreground fold obstructs the ligand. Apply `cmd.turn(axis, deg)` after `cmd.set_view` and before `cmd.ray`. Keep these as a small per-cell dict so they're easy to tune iteratively.
6. **Standard render quality**:
   ```python
   cmd.bg_color("white")
   cmd.set("ray_shadow", 0)
   cmd.set("ray_opaque_background", 1)
   cmd.set("ambient", 0.35)
   cmd.set("specular", 0.2)
   cmd.set("antialias", 2)
   cmd.ray(1200, 900)
   cmd.png(out_png, dpi=300)
   ```
7. **Matplotlib composition** — load each PyMOL PNG with `mpimg.imread`, place into a `plt.subplots` grid with `hspace=0.05–0.1`, hide axes spines+ticks, and overlay text labels with white-halo `path_effects`:
   ```python
   import matplotlib.patheffects as pe
   HALO = [pe.withStroke(linewidth=2.0, foreground="white")]
   ax.text(0.03, 0.97, "d469", transform=ax.transAxes,
           color=BIIE.BLUE, weight="bold", path_effects=HALO)
   ```
   The halo keeps labels readable on top of protein cartoon.

**Color convention for SFM structural overlays:**
- Decoder candidate / Class A : `BIIE.BLUE` (`#1565C0`)
- Approved-drug primary anchor : `BIIE.GOLD` (`#F59E0B`)
- Protein cartoon : `0xC8C8C8` (light grey)
- Multi-anchor reference palette : `[GOLD, "#EA580C" (orange), ALERT, PURPLE, "#92400E" (brown)]`

**Three render modes** — pick one per cell:
- *close*: anchor + candidate both shown, `orient` on ligand cluster, tight zoom (the default).
- *wide*: anchor + candidate both shown, larger zoom buffer (~6 Å) — for "two-pocket" cases where the candidate docks at a non-canonical site distinct from the anchor's pocket.
- *candidate-only*: no anchor loaded; candidate ligand and its own protein cartoon only. Used when overlaying the anchor adds visual noise (e.g. very-different binding sites). Label the cell `(alt site)` instead of `vs <drug>`.

If a future SFM chat plans many AF3 overlay figures, factor the helpers above into `calm.figures.af3_overlay` as a proper module (currently the patterns live inline in `render_fig5_5_overlays.py` — extraction is a 30-min refactor whenever the second consumer arrives).
