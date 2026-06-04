# mhcSFM Filtered R@1 Approximation (Leg 3)

Exact strict filtered R@1 requires per-query rank tensors (`cosine_sim.pt`), which are not available in the repository for Section 3.1 folds. Following the audit spec, this leg uses conservative lower-bound reweighting:

\[
R_{clean,lower} = \frac{R_{raw} - \lambda}{1-\lambda}
\]

where:
- \(R_{raw}\) is reported raw R@1,
- \(\lambda\) is leakage fraction from Leg 2.

Negative results are clipped to 0 as a lower bound.

## Inputs

- Raw Section 3.1 R@1 values (from archival pool-512 log means):
  - mmseqs_080: peptide→allele 67.9%, allele→peptide 96.1%
  - mmseqs_060: peptide→allele 69.2%, allele→peptide 95.4%
- Leg 2 mean leakage:
  - mmseqs_080: 38.85%
  - mmseqs_060: 97.46% (computed across non-empty folds)

## Lower-bound Results

### mmseqs_080 (\(\lambda=0.3885\))

- peptide→allele:
  - raw: 67.9%
  - clean lower bound: **47.5%**
- allele→peptide:
  - raw: 96.1%
  - clean lower bound: **93.6%**

### mmseqs_060 (\(\lambda=0.9746\))

- peptide→allele:
  - raw: 69.2%
  - clean lower bound (clipped): **0.0%**
- allele→peptide:
  - raw: 95.4%
  - clean lower bound (clipped): **0.0%**

## Interpretation

Under this conservative assumption, mmseqs_080 retains a non-trivial lower-bound clean signal while mmseqs_060 collapses due to very high measured leakage in regenerated folds.
