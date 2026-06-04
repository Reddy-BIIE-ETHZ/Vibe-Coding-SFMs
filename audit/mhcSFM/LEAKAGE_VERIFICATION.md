# mhcSFM Leakage Verification (Leg 2)

This leg follows the regenerated-split path defined in `audit/mhcsfm_audit_v0.3.yaml` because canonical split-index artifacts were not preserved.

## Method

1. Constructed a deterministic allele-level metadata table (105 HLA-A/B/C alleles) with:
   - `row_id` = allele name
   - `pseudo_seq` = 34-residue pseudo-sequence from `allele_sequences.csv`
2. Regenerated splits with:

```bash
PYTHONPATH=src python -m calm.preprocess.mmseqs_splits \
  --metadata /tmp/mhcsfm_allele_metadata.csv \
  --seq_col pseudo_seq \
  --hash_col row_id \
  --output_dir audit/mhcsfm/split_index_regenerated \
  --thresholds 0.6 0.8 \
  --n_folds 5 \
  --seed 42
```

3. Kept the 10 audit-target files (`outerfold_N_innerfold_N` for N=0..4 across mmseqs_060 and mmseqs_080).
4. For each fold, ran `mmseqs easy-search` of test alleles vs train+val alleles at the matching threshold (`-c 0.8 --cov-mode 0`), and marked a test allele as leaked if any hit met/exceeded threshold.

## Regenerated Cluster Counts (reference replay)

Independent replay on the same 105 pseudo-sequences:
- 40%: 2 clusters
- 60%: 3 clusters
- 80%: 25 clusters

These differ from writeup values (2/5/31) but remain within the ±5 stochastic tolerance in the audit spec.

## Leakage Results

### mmseqs_080

| Fold | Test alleles | Leaked | Leakage % |
|---:|---:|---:|---:|
| 0 | 20 | 4 | 20.00 |
| 1 | 20 | 7 | 35.00 |
| 2 | 26 | 8 | 30.77 |
| 3 | 17 | 9 | 52.94 |
| 4 | 9 | 5 | 55.56 |

**Mean leakage (5 folds): 38.85%**

### mmseqs_060

| Fold | Test alleles | Leaked | Leakage % | Note |
|---:|---:|---:|---:|---|
| 0 | 92 | 85 | 92.39 | — |
| 1 | 1 | 1 | 100.00 | — |
| 2 | 12 | 12 | 100.00 | — |
| 3 | 0 | 0 | N/A | empty test/train partition |
| 4 | 0 | 0 | N/A | empty test/train partition |

**Mean leakage (non-empty folds): 97.46%**

## Coverage Notes

- `identity_100`: skipped (LV=100% by definition).
- `mmseqs_040`: skipped (out of final-table scope in writeup).

## Caveat (required)

Regenerated splits are not guaranteed to match the original canonical training splits exactly, due to MMseqs2 clustering stochasticity and pre-discipline artifact loss. Reported leakage values are representative for this regenerated pathway and should be interpreted as such.
