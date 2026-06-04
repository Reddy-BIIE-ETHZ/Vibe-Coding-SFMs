# tSFM v2 Leakage Verification (Leg 2)

Date: 2026-04-27  
Specification: `audit/tsfm_audit_v0.3.yaml`

## Scope

Leakage verification was run for OOD split methods `mmseqs_080`, `mmseqs_060`, `mmseqs_040` across folds 0-4 using committed split-index JSONs in `audit/tsfm/split_index/`.

Per spec, `identity_100` is in-distribution by construction and is documented as 100% leakage by definition (not re-evaluated).

## Method

1. Read split file `split_hash_ids_outerfold_<N>_innerfold_<N>.json`.
2. Build partitions:
   - train+val identities
   - test identities
3. Identity key used for LV: **`<MATRIX_ID>_<UNIPROT>`** (sample suffix removed).
4. Map identities to protein sequences via committed `data/jaspar/metadata.csv` (`matrix_id`, `uniprot_id`, `protein_seq`), which covers all 2054 identities present in the split-index artifacts.
5. Write FASTA for train+val and test per fold.
6. Run MMseqs2:
   - `mmseqs easy-search test.fasta trainval.fasta hits.m8 tmp --min-seq-id <threshold> -c 0.8 --cov-mode 0 --format-output query,target,pident`
7. Mark a test identity as leaked if it has at least one hit meeting threshold.
8. Compute leakage % = leaked / total test identities.

## Results

### Per-threshold, per-fold leakage (% leaked test identities)

| Split | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| mmseqs_080 | 0.0% (0/425) | 0.0% (0/427) | 0.0% (0/413) | 0.7% (3/401) | 0.0% (0/383) | **0.15%** |
| mmseqs_060 | 0.8% (3/372) | 0.0% (0/422) | 0.7% (3/418) | 3.7% (17/459) | 1.3% (5/375) | **1.31%** |
| mmseqs_040 | 3.7% (16/428) | 11.3% (49/435) | 3.2% (11/339) | 5.1% (19/374) | 14.3% (63/442) | **7.52%** |

## Interpretation

- Leakage increases as threshold loosens (080 -> 060 -> 040), consistent with cross-SFM directionality, but at materially lower absolute rates than prior eSFM/crisprSFM audits.
- `mmseqs_080` is effectively strict-OOD under this direct check (near-zero leakage).
- `mmseqs_040` shows non-trivial leakage in multiple folds and should be treated with caution for strict-OOD interpretation.

## Reproducibility note

This Leg 2 run is fully executable from committed artifacts plus local MMseqs2 binary. No GPU is required.
