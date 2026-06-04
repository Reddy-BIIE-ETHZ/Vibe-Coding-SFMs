# Supplementary Materials

## Supp Table S1 — Complete pool-512 retrieval results

`supp_table_s1_data.csv` — machine-readable source data (265 rows):
- All 6 SFMs × all thresholds × all folds × both directions
- R@1, R@5, R@10 (%)
- Per-fold values where archived; cross-fold mean ± SD for eSFM (per-fold logs not archived)

`SuppTable_S1_DetailedRetrieval.docx` — formatted Word document (Times New Roman 8pt,
portrait US Letter), suitable for journal submission.

## Rebuild from source

```bash
# Regenerate the DOCX from the CSV
cd supplementary/
python build_supp_s1_docx.py
```

Requires: `python-docx`
