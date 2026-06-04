# Filtered R@1 (Leg 3)

Not applicable for mir-SFM in this audit context. The MMseqs2 step used for leakage accounting produced singleton-only clusters at all tested thresholds (40/60/80), so a cluster-overlap leakage fraction λ is undefined in the usual sense. Therefore, the standard filtered metric `R_filtered = (R_raw - λ)/(1 - λ)` is not computable/meaningful here, and Leg 3 is documented as intentionally skipped.
