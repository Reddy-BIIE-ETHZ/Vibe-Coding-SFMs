# eSFM — Known Caveats

**Audit specification**: `audit/esfm_audit_v0.1.yaml`
**Paper section**: §3.4
**Writeup source**: `V-SFM-manuscripts/eSFM_results_validation_VC.docx`
**Document version**: 0.1
**Document date**: 2026-04-20

This document enumerates the known, documented caveats affecting the eSFM training, evaluation, and reported results. Each caveat is listed with: (a) a description of what was observed, (b) the documented handling (what was done about it), and (c) the audit item that confirms the handling is internally consistent.

The audit verifies **consistency between reported numbers and documented handling**; it does not detect these caveats afresh. If the writeup reports a mean, the audit recomputes that mean over the documented fold subset and confirms the values match. Caveats that are not documented here are treated as audit findings and should not appear in an eSFM verification report unless they were introduced between this document's version and the audit run.

---

## Caveat 1 — Fold-4 degenerate validation set (CALM split-generator bug)

### Description

CALM's nested 5×5 cross-validation split generator assigns a degenerate validation set (size 2–5 entries) when `inner_fold == outer_fold == 4`. The resulting val-accuracy tracking is meaningless at these fold assignments, and the "best validation accuracy" checkpoint freezes at epoch 0 (the random initialization). This affects all four fold-4 runs in eSFM (one per OOD threshold: identity_100, mmseqs_080, mmseqs_060, mmseqs_040).

### Documented handling

The eSFM evaluation script (`V-SFM-manuscripts/eval_esfm_pool512.py`) automatically detects fold-4 runs by the condition `best_val_checkpoint_epoch == 0` and skips them when computing reported means. All pool-512 retrieval statistics reported in §3.4 and in `eSFM_results_validation_VC.docx` are means ± standard deviations computed over folds {0, 1, 2, 3} only.

The exclusion is documented in the validation writeup, §3.1: "Fold 4 was excluded from all reported means."

### Audit verification

Audit item **SA-6 (Fold-4 Exclusion Consistency)**. The verifier:
1. Reads the §3.1 exclusion statement from the writeup.
2. Reads the eSFM evaluation output files (per-fold R@1 values).
3. Computes the mean over folds {0, 1, 2, 3}.
4. Confirms the computed mean matches the writeup-reported mean within 0.1 points.

### Scope note

The bug affects every CALM-based SFM domain, not just eSFM. Analogous caveats apply to mhcSFM, miR-SFM, and dtSFM and will appear in their respective `known_caveats/` documents when their audits are built. The bug originates in CALM's split-generation utility (`src/calm/preprocess/split_utils.py`) and has not been fixed in the current codebase; a fix is deferred to a future release. The audit confirms the bug's effect is handled correctly in the reported numbers; it does not attempt to fix the underlying code.

### Reference

Documented in `eSFM_results_validation_VC.docx` §3.1 under "Fold-4 exclusion." Also noted in `physics_signatures_verification.md` §1.4 under the eSFM subsection.

---

## Caveat 2 — Cofactor filter heuristic (substrate extraction)

### Description

The raw ReactZyme dataset (`data/esfm/cleaned_uniprot_rhea.tsv`) contains reactions with multiple molecular participants: one primary substrate plus cofactors (ATP, NADH, NADP, water, CO2, O2, etc.). For eSFM training, each reaction is reduced to a single (substrate, enzyme) pair by extracting the "canonical substrate" — the largest non-cofactor molecule participating in the reaction.

The reduction is performed by a heuristic in the preprocessing script (`src/calm/preprocess/esfm.py`): (a) a hard-coded list of cofactor molecules is excluded, (b) among the remaining molecules, the one with the largest SMILES string length is selected as the canonical substrate.

### Documented handling

The heuristic is implemented as a pure function of the ReactZyme input and a static cofactor list — it produces the same output every time given the same input. A manual spot-check of ~50 reactions (performed during preprocessing development) confirmed >95% agreement between the heuristic's selected substrate and the biologically intended primary substrate, for the reactions checked.

The heuristic is a domain-expert decision, not a biologically general rule. Edge cases exist — for example, some reactions have two legitimate substrates (bi-substrate enzymes) and the heuristic picks only one. The writeup's §2 Training Configuration documents this as a known preprocessing limitation.

### Audit verification

Audit item **SA-3 (Preprocessing Reproducibility Including Cofactor Filter)**. The verifier:
1. Runs the preprocessing script on the raw ReactZyme data.
2. Compares the resulting `data/esfm/pairs.parquet` and `data/esfm/metadata.csv` against the committed versions bitwise.
3. Confirms SHA-256 hashes match exactly.

The audit verifies **reproducibility** (same input → same output every time), not **correctness** (whether the heuristic's choices are biologically optimal). Correctness is a domain-expert judgment documented in the writeup and not audited.

### Scope note

A future scaled eSFM (eSFM-BRENDA) will replace this heuristic with bi-substrate-aware reasoning and kinetic-parameter-weighted substrate ranking. The prototype version's heuristic is sufficient for the retrieval experiments reported in §3.4.

### Reference

Documented in `eSFM_results_validation_VC.docx` §2 under "Cofactor filtering" and "Training Configuration."

---

## Caveat 3 — Mean pooling at embedding time (memory optimization)

### Description

The ESM-2 (t33_650M_UR50D) encoder used for enzyme sequences produces token-level embeddings of shape `(sequence_length, 1280)` per enzyme. Storing token-level embeddings for all 177,389 unique enzymes at max sequence length 1024 would require approximately 900 GB of disk space (`177,389 × 1024 × 1280 × 4 bytes`), which exceeds the available Euler scratch allocation.

### Documented handling

Mean pooling is applied at embedding time rather than at projection time. Each enzyme is represented by a single vector of shape `(1, 1280)` computed as the mean over valid (non-padding) token positions. Storage is reduced to approximately 0.9 GB (1000× reduction).

This is mathematically equivalent to the CALM default, which applies mean pooling at projection time. In the CALM architecture, projection operates on the mean-pooled vector regardless of whether the mean was taken before or after the projection matrix, because mean is linear: `mean(Wx_i) = W × mean(x_i)`. The equivalence holds for the FFN projection head used in eSFM (`configs/model/encoder/esfm_ffn.yaml`).

The optimization is noted in the preprocessing script's docstring (`src/calm/preprocess/esfm.py`) and confirmed empirically by spot-check: recomputing the mean-pooled embedding from token-level embeddings for a sample enzyme produces the identical vector.

### Audit verification

Audit item **SA-4 (Embedding Cache Integrity)**. The verifier:
1. Samples 100 enzyme protein_seqs and 100 substrate SMILES.
2. Recomputes their embeddings using the pinned encoder checkpoints.
3. Loads the cached mean-pooled embeddings at the matching indices.
4. Confirms element-wise L-infinity distance ≤ 1e-5 (floating-point tolerance).
5. Verifies encoder checkpoint hashes in the cache manifest match the HuggingFace-pinned versions.

### Scope note

This optimization is eSFM-specific in its magnitude (largest enzyme dataset). Similar mean-pool-at-embed-time patterns may exist in other domains but with smaller storage impact and are not specifically flagged as caveats.

### Reference

Documented in `eSFM_results_validation_VC.docx` §2 under "Training Configuration" and in the preprocessing script docstring.

---

## What counts as a "new finding" versus a "documented caveat"

If the audit surfaces any of the following, they should be reported as **audit findings** (new items) rather than treated as covered by this document:

- Numerical inconsistency between the writeup and the reproduction that cannot be attributed to fold-4 exclusion or to floating-point tolerance.
- A preprocessing step that produces different outputs on re-execution (non-reproducible).
- An embedding cache mismatch at magnitudes larger than 1e-5.
- Data leakage (any test sequence sharing ≥ threshold identity with a training sequence, for splits at that threshold).
- Any caveat not listed in this document.

If any of the above appears in the audit report, it is a real finding that requires human attention and may require a paper revision or retraction-worthy disclosure depending on severity.

---

## Document lifecycle

This document is versioned with the audit YAML. When new caveats are discovered, they are added here with the version bumped. The audit YAML's `known_caveats:` block references specific caveats by name, so adding a caveat to this document does not automatically trigger re-audit; the YAML must also be updated.

Future domain-specific known-caveats documents (`mhcsfm.md`, `crispr.md`, `mirsfm.md`, `dtsfm.md`, `tsfm.md`) will follow the same structure. Caveats that apply across multiple domains (e.g., the fold-4 bug) will be cross-referenced.
