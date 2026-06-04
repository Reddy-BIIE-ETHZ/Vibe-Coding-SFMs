# mir-SFM Leakage Verification (Leg 2)

## Objective
Confirmation audit (not discovery): verify the writeup §3.2 acknowledgment that MMseqs2 clustering over the 135 mature miRNA sequences yields singleton-only output at thresholds 40%, 60%, and 80% identity.

## Procedure executed
1. Extracted unique `mirna_seq` values from `data/mirsfm/mirsfm_all_pairs.tsv` (135 unique sequences).
2. Built FASTA and ran:
   - `mmseqs easy-cluster ... --min-seq-id 0.4 -c 0.8 --cov-mode 0 --threads 4 --dbtype 2`
   - `mmseqs easy-cluster ... --min-seq-id 0.6 -c 0.8 --cov-mode 0 --threads 4 --dbtype 2`
   - `mmseqs easy-cluster ... --min-seq-id 0.8 -c 0.8 --cov-mode 0 --threads 4 --dbtype 2`
3. Counted `*_cluster.tsv` rows per threshold.

## Results
- 0.4 identity: **135 clusters / 135 sequences**
- 0.6 identity: **135 clusters / 135 sequences**
- 0.8 identity: **135 clusters / 135 sequences**

All three thresholds are singleton-only (±0 tolerance satisfied).

## Interpretation
This confirms (rather than newly reveals) the writeup’s pre-existing statement: similarity-based clustering is degenerate for this 135-sequence mature-miRNA set, so the training “OOD split” is functionally by-miRNA-identity holdout. This remains paper-positive because the limitation was already documented at training/writeup time.
