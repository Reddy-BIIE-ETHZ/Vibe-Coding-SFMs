# mir-SFM Codex Audit Results (Leg 1: SA + CL)

## Scope and method
Executed against `audit/mirsfm_audit_v0.3.yaml` with evidence from committed configs, source code, archival logs, and data files. Findings are classified PASS / PARTIAL / FAIL / N/A.

## SA findings (9)
- **SA-1 — PASS.** Training hyperparameters match configs and logs: epochs=100, batch=64, AdamW lr=0.001 wd=0.2, CAWR T0=20/Tmult=1, d_model=512, tau=0.07.
- **SA-2 — PASS.** Dual DNABERT-2 encoder usage and U→T preprocessing confirmed; archival preprocess log records DNABERT-2 and expected embedding shapes.
- **SA-3 — PASS.** Dataset counts (206,692 total; 135 unique miRNAs; 110,221 unique targets; 43,472 canonical; 163,220 non-canonical) confirmed in archival preprocess/training/mmseqs logs.
- **SA-4 — PASS.** Required source files are present in `data/mirsfm/`; sizes are consistent with claimed scale.
- **SA-5 — PASS.** MMseqs2 re-run confirmation: 135 clusters / 135 sequences at min-seq-id 0.4, 0.6, and 0.8 (singleton-only at all thresholds).
- **SA-6 — PASS.** By-miRNA-identity split logic confirmed (`cluster_id` assigned from `mirna_name`), and preprocess log reports 25 split files.
- **SA-7 — PASS.** Baseline implementation matches described seed-matching logic and hierarchy; baseline summary values are present in pool-512 archival log.
- **SA-8 — PASS.** Preservation status documented correctly as pre-discipline; archival-log audit mode is appropriate and sufficient for in-scope claims.
- **SA-9 — PASS.** Cross-SFM debug-label inheritance confirmed: mir-SFM logs print tSFM labels while showing mir-SFM-correct counts.

**SA summary:** 9 PASS / 0 PARTIAL / 0 FAIL.

## CL findings (6)
- **CL-1 — PASS.** Section 1 dataset counts match archival evidence.
- **CL-2 — PARTIAL (pre-acknowledged).** identity_100 summary means/SDs in archival pool-512 log match RETRAIN_LOG; writeup digit uses fold-0 `98.4%` for ag→ab R@1 while cross-fold mean is `98.0% ± 0.6%`.
- **CL-3 — PASS.** mmseqs_080 summary means/SDs in archival pool-512 log match RETRAIN_LOG values; folds 3/4 are present in logs and reconcile with the 5-fold summary.
- **CL-4 — PASS.** Batch-64 training accuracy summary is supported by archived `mirsfm_full_62523321_*.out` logs present in `audit/mirsfm/archival_logs/`.
- **CL-5 — PASS.** Overall seed baseline identity_100 and mmseqs_080 R@1/5/10 values match archival pool-512 baseline section; arithmetic deltas are consistent.
- **CL-6 — PASS.** Canonical/non-canonical baseline R@1 breakdown values (9.0/6.1 ID, 5.2/2.9 OOD) match archival pool-512 baseline section.

**CL summary:** 5 PASS / 1 PARTIAL / 0 FAIL.

## Overall Leg 1 verdict
Leg 1 closes with **PASS overall**, with one **pre-acknowledged PARTIAL** for writeup digit-exactness in Section 3.1 (fold-0 value used in prose vs cross-fold mean).
