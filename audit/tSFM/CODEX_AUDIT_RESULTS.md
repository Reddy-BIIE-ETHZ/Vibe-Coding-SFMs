# tSFM v2 Codex Audit Results (SA + CL)

Date: 2026-04-27  
Specification: `audit/tsfm_audit_v0.3.yaml`

## Summary Table

| Group | PASS | PARTIAL | FAIL | N/A |
|---|---:|---:|---:|---:|
| SA (12 items) | 10 | 2 | 0 | 0 |
| CL (8 items) | 8 | 0 | 0 | 0 |

## SA Items

### SA-1 — Training hyperparameters match RETRAIN_LOG and writeup
**Verdict: PASS**
- Config confirms `num_epochs=100`, `batch_size=64`, `learning_rate=0.001`, `weight_decay=0.2`, `optimizer=AdamW`, scheduler `CosineAnnealingWarmRestarts` with `T0=20`, `Tmult=1`.
- Training log confirms 100-epoch phase (`Epoch 99`) and dataset echo lines.
- RETRAIN_LOG pipeline text matches these values.

### SA-2 — Data scale-up: 102,700 training pairs from 50x PWM sampling
**Verdict: PASS**
- `jaspar.py` exposes `--samples_per_pwm` (default 10) and uses it in metadata build path.
- Training log shows `Total pairs: 102700`, `TF proteins: 999 unique`, `DNA sequences: 65461 unique`.
- RETRAIN_LOG states the same 50x scale-up result.

### SA-3 — OOD split methodology (MMseqs2 80/60/40 + identity_100)
**Verdict: PASS**
- `mmseqs_splits.py` uses MMseqs2 `easy-cluster` with thresholded `--min-seq-id` and generates `identity_100` via singleton clusters.
- `scripts/euler_train_tsfm_full.slurm` SPLIT_METHODS array is `identity_100`, `mmseqs_080`, `mmseqs_060`, `mmseqs_040`.
- RETRAIN_LOG documents the same split taxonomy.

### SA-4 — Cluster counts at OOD thresholds
**Verdict: PARTIAL**
- RETRAIN_LOG claims: 882/799/557 clusters (80/60/40) from 999 unique protein sequences.
- Local surrogate re-clustering (using committed split identities + `data/jaspar/metadata.csv` protein sequences, MMseqs2 `easy-cluster`) reproduced 999 unique protein sequences but gave 881/800/566.
- 80/60 are within +/-5 tolerance; 40% differs by +9. No committed canonical cluster-membership artifact is available for direct digit-exact replay.

### SA-5 — Embedding model identifiers
**Verdict: PASS**
- `jaspar.py` uses ESM-2 (`esm2_t33_650M_UR50D`) and DNABERT-2 (`zhihan1996/DNABERT-2-117M`) embedding functions.
- Training log shape echoes match expected dimensions: proteins `[N,512,1280]`, DNA `[M,12,768]`.

### SA-6 — Fold-4 exclusion documentation and method soundness
**Verdict: PARTIAL**
- In-repo evidence is consistent: RETRAIN_LOG documents empty-val fold pathology and exclusion; identity_100 CSV shows fold_4 anomaly (R@1 TF->DNA 27.77, DNA->TF 4.46).
- External check to `https://github.com/Reddy-BIIE-ETHZ/CALM-SFM/issues/6` returned 404 from both web fetch and GitHub API in this environment, so issue-page existence/content could not be independently confirmed.

### SA-7 — Preservation Rule 2: durable summary CSVs
**Verdict: PASS**
- Both `eval_pool512.py` and `eval_pool512_fast.py` implement `SFM_SUMMARY_DIR` + `--sfm_name` + `--split_name` pathing.
- All four split CSVs exist under `audit/tsfm/eval_summaries/` and include fold rows plus SUMMARY block.

### SA-8 — Preservation Rule 3: archival logs committed
**Verdict: PASS**
- Required logs are present and non-empty in `audit/tsfm/archival_logs/`.
- Repository includes five production-relevant logs (`tsfm_full`, `tsfm_eval`, `tsfm_pool512`, `tsfm_val`, `val_retrieval`), exceeding the minimum set.

### SA-9 — Source shape: SFM_SUMMARY_DIR logic parity
**Verdict: PASS**
- `eval_pool512.py` and `eval_pool512_fast.py` both implement the same three-tier output resolution:
  1) explicit `--output`, 2) durable `SFM_SUMMARY_DIR` path, 3) legacy `eval_dir` fallback.
- Backward-compatible fallback remains intact.

### SA-10 — Pseudo-prospective dataset construction
**Verdict: PASS**
- `jaspar_validation.py` uses JASPAR API `release=2022` to define temporal split.
- `validation_summary.json` confirms 1200 train profiles / 60000 train pairs and 854 held-out profiles / 42700 held-out pairs, with 178 truly novel TFs.
- `results_summary.json` confirms 28 skipped (unevaluable) and 787 evaluable held-out TFs; truly novel evaluable `n=150`.
- Arithmetic checks: 60000 + 42700 = 102700; 178 = 150 + 28.

### SA-11 — Scaling discipline via `--samples_per_pwm 50`
**Verdict: PASS**
- `jaspar.py` exposes single scaling flag `--samples_per_pwm` (default 10).
- `SCOPING.md` documents v0.5->v2 scale-up as a parameter-level change.
- No v2-specific architecture rewrites are required for this scale-up claim.

### SA-12 — Digit-exactness meta-audit of RETRAIN_LOG numerics
**Verdict: PASS**
- All v2 numeric claims in retrieval tables (R@1/R@5/R@10 per split, per-fold raw table, pseudo-prospective 44.7/45.9) match committed artifacts (eval CSVs, pool512 log, val retrieval log) within stated rounding tolerance.
- CL-8-style v0.5->v2 deltas are arithmetically correct on the v2 side; v0.5 terms remain documented as historical writeup values (not independently re-verifiable post-purge), consistent with spec note.

## CL Items

### CL-1 — Training pair count = 102,700
**Verdict: PASS**
- Training log: `Total pairs: 102700`.
- RETRAIN_LOG reports same value.

### CL-2 — Pool-512 R@1 TF->DNA mean +/- SD
**Verdict: PASS**
- Recomputed from per-fold CSVs (sample SD, fold_4 excluded for identity_100):
  - identity_100: **83.1 +/- 3.0**
  - mmseqs_080: **54.7 +/- 5.1**
  - mmseqs_060: **53.7 +/- 4.9**
  - mmseqs_040: **48.4 +/- 4.3**
- Matches RETRAIN_LOG; per-fold digits also match pool512 archival log summaries.

### CL-3 — Pool-512 R@1 DNA->TF mean +/- SD
**Verdict: PASS**
- Recomputed from CSVs:
  - identity_100: **25.8 +/- 2.5** (fold_4 excluded)
  - mmseqs_080: **10.2 +/- 2.7**
  - mmseqs_060: **8.6 +/- 2.0**
  - mmseqs_040: **7.5 +/- 1.6**
- Matches RETRAIN_LOG.

### CL-4 — Pool-512 R@5 and R@10 across splits/directions
**Verdict: PASS**
- All 16 cells (4 splits x {R@5,R@10} x 2 directions) recomputed from committed CSVs match RETRAIN_LOG to one decimal.

### CL-5 — Per-fold raw R@1 TF->DNA matrix
**Verdict: PASS**
- All 20 cells match across:
  1) RETRAIN_LOG per-fold table,
  2) `audit/tsfm/eval_summaries/*.csv`,
  3) `tsfm_pool512_64822444.out` per-fold summary blocks.

### CL-6 — Truly-novel top-1 family retrieval = 44.7% on n=150
**Verdict: PASS**
- `val_retrieval_64842658.out`: family top-1 = 44.7%.
- `validation_retrieval/results_summary.json`: truly_novel `n=150`, top_1_family_pct = 44.666...
- `validation/validation_summary.json`: truly_novel_tfs = 178.
- `results_summary.json`: skipped=28, evaluable=787, giving truly novel evaluable count 150 per claim context.

### CL-7 — Exact-TF top-1 retrieval = 45.9%
**Verdict: PASS**
- `val_retrieval_64842658.out`: exact top-1 = 45.9% (`n=634 with exact match in pool`).
- RETRAIN_LOG reports same.

### CL-8 — v0.5->v2 deltas
**Verdict: PASS**
- Delta arithmetic signs/magnitudes are correct from cited v0.5 baselines to v2 values.
- As specified, v0.5 source values are historical writeup numbers and not independently re-verifiable from committed artifacts after purge.

## Notes

- This audit executed against current `main` HEAD after confirming with `git log --oneline -5`.
- Leg 2 and Leg 3 results are documented separately in:
  - `audit/tsfm/LEAKAGE_VERIFICATION.md`
  - `audit/tsfm/FILTERED_R1_RESULTS.md`
