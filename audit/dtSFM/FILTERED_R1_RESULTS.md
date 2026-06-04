# dtSFM v2 Filtered R@1 Results (Leg 3)

Conservative lower-bound formula:

\[
R_{clean\_lower} = \max\left(0, \frac{R_{raw} - \lambda}{1-\lambda}\right)
\]

where \(R_{raw}\) and \(\lambda\) are in fraction units.

## Inputs

- mmseqs_080 R@1 D→T raw (writeup §3.2 reported fold): 46.0%
- mmseqs_060 R@1 D→T raw (writeup fold-0-3 mean): 41.8%
- mmseqs_040 R@1 D→T raw (writeup fold-0-3 mean): 41.3%

Leakage rates from Leg 2 (coverage-aligned):

- mmseqs_080 (fold 0): 1.01%
- mmseqs_060 (folds 0–3 mean): 3.43%
- mmseqs_040 (folds 0–3 mean): 5.88%

## Results

| Split | R@1 raw D→T | Leakage λ | R_clean_lower D→T | Absolute drop |
|---|---:|---:|---:|---:|
| mmseqs_080 | 46.0% | 1.01% | 45.45% | -0.55 pp |
| mmseqs_060 | 41.8% | 3.43% | 40.44% | -1.36 pp |
| mmseqs_040 | 41.3% | 5.88% | 38.66% | -2.64 pp |

## Interpretation

All three R@1 values remain materially non-zero under conservative leakage filtering, with limited degradation in this reproduction. Under these measurements, the main writeup direction-of-effect for dtSFM v2 OOD retrieval is preserved, while acknowledging separate structural issues (single-fold mmseqs_080 and fold-4 degeneracy in 060/040).
