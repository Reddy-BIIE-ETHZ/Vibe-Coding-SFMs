# tSFM v2 Filtered R@1 (Leg 3)

Date: 2026-04-27  
Specification: `audit/tsfm_audit_v0.3.yaml`

## Chosen approach

**Approach (a): Approximate filtered R@1 via re-weighting bounds**.

Reason: true filtered R@1 requires per-test-item rank tensors (`cosine_sim.pt`) to recompute retrieval on LV-clean subsets. Those tensors are not committed for all 20 folds (as expected under preservation discipline Rule 2), so exact strict filtered R@1 cannot be recovered locally without Euler re-execution.

## Inputs

- Raw per-fold R@1 TF->DNA from committed summary CSVs (`audit/tsfm/eval_summaries/*.csv`)
- Fold-level leakage fractions from Leg 2 (`audit/tsfm/LEAKAGE_VERIFICATION.md`)

## Approximation model

For each fold:

R_raw = (1 - lambda) * R_clean + lambda * R_leaked

where `lambda` is the leakage fraction.

We report a **conservative lower bound** by assuming leaked items are perfectly easy:

R_leaked = 100% => R_clean(lower) = (R_raw - lambda) / (1 - lambda)

(Percent scale normalized in computation.)

This is not a true filtered R@1; it is a sensitivity bound.

## Results (TF->DNA R@1)

| Split | Raw R@1 mean | Mean leakage | Approx. lower-bound clean R@1 |
|---|---:|---:|---:|
| mmseqs_080 | 54.7% | 0.15% | **54.6%** |
| mmseqs_060 | 53.7% | 1.31% | **53.1%** |
| mmseqs_040 | 48.4% | 7.52% | **43.9%** |

## Interpretation

- At `mmseqs_080` and `mmseqs_060`, leakage-correction impact is minimal under this conservative bound.
- At `mmseqs_040`, leakage can explain a larger portion of headline R@1.
- Directionality (080 >= 060 > 040) remains.

## Exact filtered R@1 requirements (for future rerun)

To obtain true filtered R@1, rerun pool-512 evaluation on Euler with:
1. Fold-level leakage masks from Leg 2.
2. Original per-fold similarity tensors/rank tensors (`cosine_sim.pt`) for each split/fold.

Given these tensors are ephemeral by design, exact filtered R@1 is structurally feasible but requires controlled re-execution.
