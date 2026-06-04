# dtSFM v2 Audit Results (Codex, Leg 1 SA + CL)

Date: 2026-05-04 (UTC)

## Repository state verification

- Local HEAD at audit start: `1c88f65` (`dtsfm_v2 audit setup: SCOPING + RETRAIN_LOG + audit YAML + 31 archival logs`).
- No drift observed before executing audit legs.

## SA verdicts

| ID | Verdict | Notes |
|---|---|---|
| SA-1 | PARTIAL | Hyperparameters verified from archival training logs; v2-named config preservation gap confirmed (Finding 4). |
| SA-2 | PASS | Dual-encoder dims and architecture evidence consistent in logs/code. |
| SA-3 | PASS | 121,548 pairs / 10,338 drugs / 1,358 proteins verified across logs and metadata. |
| SA-4 | PASS | MMseqs cluster counts reproduced within ±5% tolerance. |
| SA-5 | PARTIAL | Eval script uses `effective_pool = min(pool_size, N)`; writeup phrasing ambiguity confirmed (Finding 2). |
| SA-6 | PARTIAL | Fold-4 degeneracy in mmseqs_060/mmseqs_040 verified; exclusion rationale valid (Finding 1). |
| SA-7 | PARTIAL | mmseqs_080 has 1-fold operational checkpoint in pool-512 run; framing is indicative/single-fold (Finding 3). |
| SA-8 | PASS | Split generation is cluster-based by protein sequence via MMseqs and cluster-aware partitioning logic. |
| SA-9 | PASS | Preservation status predates formal discipline; archival methodology appropriate and sufficient for in-scope verification. |
| SA-10 | PARTIAL | Cross-SFM debug-label inheritance confirmed (Finding 5). |

**SA tally:** 5 PASS / 5 PARTIAL / 0 FAIL.

## CL verdicts

| ID | Verdict | Notes |
|---|---|---|
| CL-1 | PASS | Dataset counts and source-dataset presence verified. |
| CL-2 | PASS | Training configuration and architecture claims reproduce from logs/code. |
| CL-3 | PARTIAL | mmseqs_080 results are single-fold; `±0.0` notation is mathematically true but should remain explicitly labeled as non-averaged. |
| CL-4 | PASS | mmseqs_060 fold-0-3 arithmetic verified: D→T R@1 = (49.0+33.3+39.9+44.8)/4 = 41.75% ≈ 41.8%. |
| CL-5 | PASS | mmseqs_040 fold-0-3 arithmetic verified: D→T R@1 = (31.1+47.9+18.9+67.4)/4 = 41.325% ≈ 41.3%. |
| CL-6 | PASS | In-distribution identity_100 summary metrics match archival pool-512 summary block. |

**CL tally:** 5 PASS / 1 PARTIAL / 0 FAIL.

## Mapping of pre-acknowledged findings (Path A)

1. Finding 1 (fold-4 degeneracy): SA-6 PARTIAL.
2. Finding 2 (pool-size ambiguity): SA-5 PARTIAL.
3. Finding 3 (mmseqs_080 single-fold): SA-7 PARTIAL and CL-3 PARTIAL.
4. Finding 4 (v2 config preservation gap): SA-1 PARTIAL.
5. Finding 5 (debug-label inheritance): SA-10 PARTIAL.

No additional FAIL-class findings were discovered in Leg 1.
