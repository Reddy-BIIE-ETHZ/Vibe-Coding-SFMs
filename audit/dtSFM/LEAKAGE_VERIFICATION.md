# dtSFM v2 Leakage Verification (Leg 2)

Date: 2026-05-04 (UTC)
Mode: CPU-only reproduction from committed metadata and code.

## Method

1. Loaded `data/dtsfm_v2/metadata.csv` and extracted 1,358 unique `protein_seq` entries.
2. Re-ran `mmseqs easy-cluster` at 40/60/80% identity with `-c 0.8 --cov-mode 0`.
3. Reconstructed cluster-CV fold assignment with `seed=42` and the same split logic used in `build_cluster_splits`.
4. For each outer fold, built test-protein and train+val-protein FASTA sets and ran MMseqs search at the matching threshold.
5. Labeled a test protein as leaked if it had any search hit in train+val at or above the split threshold.

Commands were executed via inline Python orchestration using MMseqs binaries available in the environment.

## Cluster count reproduction

- 40%: **731** clusters (archival log: 726; +0.69%, within ±5% tolerance).
- 60%: **878** clusters (archival log: 875; +0.34%, within ±5% tolerance).
- 80%: **1012** clusters (archival log: 1018; -0.59%, within ±5% tolerance).

## Leakage by threshold and fold

Leakage (%) = leaked test proteins / total test proteins.

| Split threshold | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| mmseqs_080 | 1.01 | 1.07 | 3.00 | 0.37 | 1.25 | 1.34 |
| mmseqs_060 | 3.92 | 2.39 | 0.95 | 6.46 | 2.23 | 3.19 |
| mmseqs_040 | 6.74 | 1.08 | 13.99 | 1.69 | 6.36 | 5.97 |

### Coverage aligned to writeup reporting

- mmseqs_080: only fold 0 has a checkpoint in the pool-512 archival run; operational leakage for the reported fold = **1.01%**.
- mmseqs_060: writeup uses folds 0–3 (fold 4 excluded for degeneracy). Mean leakage over folds 0–3 = **3.43%**.
- mmseqs_040: writeup uses folds 0–3 (fold 4 excluded for degeneracy). Mean leakage over folds 0–3 = **5.88%**.

## Interpretation for paper prep

Observed leakage is low across all three thresholds under this reconstruction protocol. On this evidence:

- Keep mmseqs_080 as strict-OOD evidence (single-fold caveat remains separate).
- Keep mmseqs_060 with caveat that fold-4 degeneracy is excluded in efficacy reporting.
- mmseqs_040 does **not** show high leakage in this reproduction, but it remains unstable from a performance-variance standpoint and fold-4 degeneracy; qualifiers should emphasize variance/degeneracy rather than leakage-driven invalidation.

## Notes

- This leg is true recomputation from metadata with current local MMseqs binary; counts differ slightly from archival values, consistent with previously documented MMseqs stochasticity/ordering sensitivity.
- No GPU jobs or retraining were performed.
