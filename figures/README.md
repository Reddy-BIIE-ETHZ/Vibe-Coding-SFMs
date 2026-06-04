# Figure Build Scripts

All figures are reproducible from the build scripts and data CSVs in this directory.
Rendered outputs (PDF/PNG/SVG) are available in the Zenodo deposit.

## Figure 2 — Bidirectional retrieval + external validation

```bash
python build_fig2_combined.py
# Outputs: fig2_combined.{pdf,svg,png}
```

Source data: `fig2_combined_data.csv`  
Caption: `fig2_combined_caption.md`

## Figure 3 — Training dynamics and physics signatures

```bash
python build_fig3_combined.py
# Outputs: fig3_combined.{pdf,svg,png}
```

Requires archival training logs (see `../sfms/<SFM>/` for log paths).  
Source data (pre-extracted): `fig3_combined_data.csv`  
Caption: `fig3_combined_caption.md`

## Dependencies

```
matplotlib>=3.7
numpy
```
