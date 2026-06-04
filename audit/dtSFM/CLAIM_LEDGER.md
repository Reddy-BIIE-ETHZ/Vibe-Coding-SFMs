# dtSFM v2 Numerical Claims (RETRAIN_LOG-equivalent)

**Source writeup:** `V-SFM-manuscripts/dtSFM_results_validation_VC.docx`
**Date generated:** 2026-05-03
**Purpose:** Comprehensive claim ledger for digit-exactness audit by Codex against committed source artifacts.
**Scope:** Sections 1, 2, 3.1, 3.2 only. Sections 3.3, 3.4, 4, 5, 6, 7, 8, 9 are explicitly out of scope per `audit/dtsfm_v2/SCOPING.md` §3.2.

This document mirrors the role of `audit/mhcsfm/RETRAIN_LOG.md` and `audit/mirsfm/RETRAIN_LOG.md`: every numeric claim within audit scope is enumerated here with its source artifact, so Codex can verify each claim from a single reference document.

---

## Section 1 — Dataset

### 1.1 Counts

| Property | Value | Source |
|---|---|---|
| Total positive binding pairs | 121,548 | `audit/dtsfm_v2/archival_logs/dtsfm2_full_63170670_10.out` ("Total pairs: 121548"); `dtsfm2_p512_63407083.out` ("Pairs: 121548") |
| Unique drugs (SMILES) | 10,338 | training log "TF proteins: 10338 unique" (debug-label inheritance — see SA-10); pool-512 log "Drugs: 10338 unique" |
| Unique protein targets | 1,358 | training log "DNA sequences: 1358 unique"; pool-512 log "Proteins: 1358 unique"; mmseqs job log "Loaded 121548 rows, 1358 unique sequences" |
| MMseqs2 clusters @ 80% identity | 1,018 | mmseqs job log "MMseqs2 at 80% identity: 1018 clusters from 1358 sequences" |
| MMseqs2 clusters @ 60% identity | 875 | mmseqs job log "MMseqs2 at 60% identity: 875 clusters from 1358 sequences" |
| MMseqs2 clusters @ 40% identity | 726 | mmseqs job log "MMseqs2 at 40% identity: 726 clusters from 1358 sequences" |

### 1.2 Data sources

Writeup §1 cites three source datasets composing v2:

- **DAVIS:** 68 inhibitors × 379 kinases (Kd values) — `data/dtsfm_v2/davis_raw.csv` (21 MB)
- **KIBA:** 2,068 compounds × 229 kinases (integrated KIBA score) — `data/dtsfm_v2/kiba_raw.csv` (95 MB)
- **BindingDB_Kd:** 10,661 drugs × 1,413 proteins (Kd values from Therapeutics Data Commons) — `data/dtsfm_v2/bindingdb_kd_raw.csv` (41 MB)

Combined v2 dataset: `data/dtsfm_v2/metadata.csv` (187 MB).

**Note on writeup precision:** Writeup §1 dataset table claims "BindingDB_Kd (10,661 drugs × 1,413 proteins, Kd)" — these are source-file counts, not post-deduplication counts in the merged v2 dataset. The merged dataset has 10,338 unique drugs and 1,358 unique proteins (after deduplication across all three sources and removal of pairs with incompatible identifiers).

### 1.3 Bilinearity classification

Writeup §1.2: "**Emergent (no analytical decomposition of drug-target binding energy)**." This is a methodological framing claim, not a numerical claim. The audit verifies this language is consistent with the architecture (no per-position decomposition encoded; pure InfoNCE contrastive learning).

dtSFM is **the first SFM with emergent bilinearity** in the program (tSFM/crisprSFM/mhcSFM/mir-SFM all have given bilinearity through PWMs, Watson-Crick, pseudo-sequences, or seed matching).

---

## Section 2 — Training Configuration

| Parameter | Value | Source |
|---|---|---|
| Architecture | FFN projection head | training log echoes; v1 `configs/model/encoder/dtsfm_ffn.yaml` (preserved); v2 configs not preserved as v2-named (see SA-9) |
| Projection dimension (d_model) | 512 | training log echoes |
| Drug input dim | 768 (MoLFormer-XL) | training log "[10338, 512, 768]" → 768-dim per token |
| Protein input dim | 1280 (ESM-2-650M) | training log "[1358, 1024, 1280]" → 1280-dim per residue |
| Pooling | Mean pooling (masked) | architecture inheritance from CALM |
| Temperature τ initialization | 0.07 (max_scale = 100) | architecture inheritance |
| Loss | Symmetric InfoNCE (bidirectional contrastive) | architecture inheritance |
| Optimizer | AdamW (lr = 0.001, weight_decay = 0.2) | training log echoes |
| Scheduler | CosineAnnealingWarmRestarts (T0 = 20, Tmult = 1) | training log echoes |
| Epochs | 100 | training log echoes |
| Batch size | 64 | training log echoes |
| OOD split | MMseqs2 on protein sequences at 40/60/80% identity | mmseqs job log; writeup §2 |
| Folds | 5-fold outer × 5-fold inner CV (25 splits) | preprocess; per-fold split JSONs |
| Compute | Euler ETH HPC, NVIDIA TITAN RTX | training log job submission |

### 2.1 Encoder architecture

- **Drug (agent) encoder:** MoLFormer-XL (IBM, 46.8M parameters), pre-trained on ~110M molecules from ZINC and PubChem. 768-dim hidden, 512 max tokens.
- **Protein (target) encoder:** ESM-2-650M (Meta, 650M parameters), 1280-dim, 1024 max tokens.
- **Code change:** Single-line addition of `'molformer': 768` to `_EMBED_DIMS` dictionary in `src/calm/encoder/model.py` to support the new molecular encoder.

### 2.2 v2 configs preservation status

Per writeup §2 the training used v2 configs. The repo contains `configs/train/encoder/dtsfm_full.yaml` and `configs/model/encoder/dtsfm_ffn.yaml`, which are v1-era files (no `dtsfm_v2_*.yaml` variant exists). The v2 training was launched after edit-in-place of these configs between v1 (Apr 9) and v2 (Apr 12) training dates.

**SA-9 evidence**: The audit verifies v2 hyperparameters from training-log echo (job 63170670, dtsfm2_full_63170670_*.out files) rather than from a v2-named config file. This is the crisprSFM-pattern preservation finding.

---

## Section 3 — Pool-512 Retrieval Results

Writeup §3 describes the evaluation as "pool-512 retrieval results on deduplicated unique (drug, protein) pairs. Pool size = min(512, unique targets in test fold)."

**Pool-size ambiguity (Finding 2 from SCOPING):** Writeup §3.1 says "Identity_100 has ~920 unique targets per fold (true pool-512 evaluation)." This is internally inconsistent: if the pool is min(512, unique targets), and there are 920 unique targets per fold, the pool should be 512, not 920. The audit reads the actual evaluation script to determine the true pool definition. This is a writeup-precision issue; the underlying numerical R@1/5/10 values verify against the pool-512 log regardless.

### 3.1 v2 In-Distribution (identity_100) — across 5 folds

Source: `audit/dtsfm_v2/archival_logs/dtsfm2_p512_63407083.out` "Split: identity_100" section.

**Per-fold values (from log):**

| Fold | Unique pairs | Drugs | Targets | R@1 D→T | R@1 T→D | R@5 D→T | R@5 T→D | R@10 D→T | R@10 T→D |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24,309 | 4,133 | 924 | 23.3 | 49.5 | 40.5 | 58.5 | 45.4 | 63.5 |
| 1 | 24,309 | 4,242 | 906 | 25.3 | 50.9 | 42.3 | 61.0 | 47.6 | 66.9 |
| 2 | 24,309 | 4,122 | 927 | 31.3 | 65.6 | 44.0 | 76.3 | 47.1 | 80.5 |
| 3 | 24,309 | 4,245 | 928 | 26.9 | 65.8 | 40.1 | 75.7 | 43.9 | 79.7 |
| 4 | 24,312 | 4,182 | 908 | 31.6 | 70.8 | 46.0 | 80.1 | 49.5 | 84.2 |

**Cross-fold summary (5 folds, from log "--- identity_100 summary (5 folds) ---"):**

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Drug → Target | **27.7% ± 3.3%** | **42.6% ± 2.2%** | **46.7% ± 1.9%** |
| Target → Drug | **60.5% ± 8.7%** | **70.3% ± 8.8%** | **74.9% ± 8.2%** |

These match writeup §3.1 table exactly.

**Best-epoch values per fold:** fold 0 = epoch 3, fold 1 = epoch 3, fold 2 = epoch 8, fold 3 = epoch 8, fold 4 = epoch 7. Range 3-8 (consistent with writeup §5 "fast convergence" hypothesis claim — out of audit scope).

### 3.2 v2 OOD: mmseqs_080 (single fold)

Source: `audit/dtsfm_v2/archival_logs/dtsfm2_p512_63407083.out` "Split: mmseqs_080" section.

**Per-fold values (from log):**

| Fold | Status | Unique pairs | Drugs | Targets | R@1 D→T | R@1 T→D | R@5 D→T | R@5 T→D | R@10 D→T | R@10 T→D |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Trained (epoch 62) | 22,993 | 4,697 | 264 | 46.0 | 95.7 | 79.9 | 95.8 | 91.2 | 95.9 |
| 1 | **No checkpoint** | — | — | — | — | — | — | — | — | — |
| 2 | **No checkpoint** | — | — | — | — | — | — | — | — | — |
| 3 | **No checkpoint** | — | — | — | — | — | — | — | — | — |
| 4 | **No checkpoint** | — | — | — | — | — | — | — | — | — |

**Single-fold "summary" (from log "--- mmseqs_080 summary (1 folds) ---"):**

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Drug → Target | **46.0% ± 0.0%** (single fold) | **79.9% ± 0.0%** | **91.2% ± 0.0%** |
| Target → Drug | **95.7% ± 0.0%** | **95.8% ± 0.0%** | **95.9% ± 0.0%** |

**Finding 3 from SCOPING:** mmseqs_080 has only 1 of 5 folds with a checkpoint. The reported values are single-fold (fold 0), not 5-fold means. The "± 0.0%" notation is honest (zero variance with one observation) but readable as "averaged with zero spread," which is misleading. Writeup §3.2 correctly labels these as "indicative but not averaged."

**Why folds 1-4 of mmseqs_080 lack checkpoints:** TBD via training-log inspection. Either (a) training jobs aborted early, (b) val-set was degenerate (cluster-CV split bug at high cluster count), or (c) checkpoints were purged but other folds' checkpoints survived. The audit's SA item investigates.

### 3.3 v2 OOD: mmseqs_060 (folds 0-3, fold 4 excluded)

Source: same log, "Split: mmseqs_060" section.

**Per-fold values (from log):**

| Fold | Status | Unique pairs | Drugs | Targets | R@1 D→T | R@1 T→D | R@5 D→T | R@5 T→D | R@10 D→T | R@10 T→D | Best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Trained | 17,972 | 3,714 | 249 | 49.0 | 96.4 | 66.6 | 97.3 | 92.3 | 97.6 | 68 |
| 1 | Trained | 26,119 | 4,278 | 286 | 33.3 | 90.8 | 37.0 | 92.4 | 49.9 | 93.7 | 36 |
| 2 | Trained | 27,059 | 4,077 | 244 | 39.9 | 91.1 | 50.3 | 92.1 | 70.5 | 92.9 | 38 |
| 3 | Trained | 25,482 | 4,466 | 282 | 44.8 | 91.5 | 54.7 | 92.5 | 64.1 | 92.8 | 46 |
| 4 | **Degenerate (epoch 0 ckpt)** | 24,916 | 3,485 | 297 | 18.3 | 2.1 | 27.7 | 9.8 | 33.7 | 16.0 | **0** |

**Cross-fold summary including all 5 folds (from log "--- mmseqs_060 summary (5 folds) ---"):**

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Drug → Target | 37.1% ± 10.7% (with fold 4) | 47.3% ± 13.6% | 62.1% ± 19.7% |
| Target → Drug | 74.4% ± 36.2% (with fold 4) | 76.8% ± 33.6% | 78.6% ± 31.3% |

**Writeup §3.2 uses fold-0-3 means (fold 4 excluded as degenerate):**

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Drug → Target | **41.8% ± 6.2%** | **52.2% ± 11.1%** | **69.2% ± 15.6%** |
| Target → Drug | **92.5% ± 2.5%** | **93.6% ± 2.5%** (?) | **94.3% ± 2.0%** |

**Verification math:**
- D→T R@1 (folds 0-3): (49.0 + 33.3 + 39.9 + 44.8) / 4 = **41.75%** ≈ 41.8% ✓
- T→D R@1 (folds 0-3): (96.4 + 90.8 + 91.1 + 91.5) / 4 = **92.45%** ≈ 92.5% ✓
- D→T R@5 (folds 0-3): (66.6 + 37.0 + 50.3 + 54.7) / 4 = **52.15%** ≈ 52.2% ✓
- T→D R@5 (folds 0-3): (97.3 + 92.4 + 92.1 + 92.5) / 4 = **93.575%** ≈ 93.6% ✓ (writeup currently shows R@10 T→D not R@5 T→D — verify table column)
- D→T R@10 (folds 0-3): (92.3 + 49.9 + 70.5 + 64.1) / 4 = **69.2%** ✓
- T→D R@10 (folds 0-3): (97.6 + 93.7 + 92.9 + 92.8) / 4 = **94.25%** ≈ 94.3% ✓

The fold-0-3 exclusion arithmetic verifies cleanly. Writeup §3.2 is internally consistent.

### 3.4 v2 OOD: mmseqs_040 (folds 0-3, fold 4 excluded)

Source: same log, "Split: mmseqs_040" section.

**Per-fold values:**

| Fold | Status | Unique pairs | Drugs | Targets | R@1 D→T | R@1 T→D | R@5 D→T | R@5 T→D | R@10 D→T | R@10 T→D | Best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Trained | 26,637 | 3,559 | 258 | 31.1 | 76.3 | 36.5 | 79.3 | 45.0 | 79.8 | 13 |
| 1 | Trained | 20,105 | 4,055 | 270 | 47.9 | 95.6 | 62.1 | 97.3 | 77.9 | 97.4 | 50 |
| 2 | Trained | 29,222 | 3,640 | 289 | 18.9 | 61.5 | 22.7 | 63.4 | 27.2 | 67.4 | 6 |
| 3 | Trained | 22,727 | 4,328 | 248 | 67.4 | 97.6 | 96.3 | 97.8 | 100.0 | 98.0 | 92 |
| 4 | **Degenerate (epoch 0 ckpt)** | 22,855 | 3,373 | 292 | 8.7 | 2.2 | 15.7 | 6.8 | 20.6 | 10.7 | **0** |

**Cross-fold summary including all 5 folds (from log "--- mmseqs_040 summary (5 folds) ---"):**

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Drug → Target | 34.8% ± 20.9% (with fold 4) | 46.7% ± 29.5% | 54.1% ± 30.3% |
| Target → Drug | 66.6% ± 34.8% (with fold 4) | 68.9% ± 33.6% | 70.7% ± 32.1% |

**Writeup §3.2 uses fold-0-3 means (fold 4 excluded):**

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Drug → Target | **41.3% ± 17.2%** | **54.4% ± 26.6%** | **62.5% ± 27.3%** |
| Target → Drug | **82.8% ± 14.1%** | **84.5% ± 13.9%** (?) | **85.7% ± 12.2%** |

**Verification math:**
- D→T R@1 (folds 0-3): (31.1 + 47.9 + 18.9 + 67.4) / 4 = **41.325%** ≈ 41.3% ✓
- T→D R@1 (folds 0-3): (76.3 + 95.6 + 61.5 + 97.6) / 4 = **82.75%** ≈ 82.8% ✓
- D→T R@5 (folds 0-3): (36.5 + 62.1 + 22.7 + 96.3) / 4 = **54.4%** ✓
- T→D R@5 (folds 0-3): (79.3 + 97.3 + 63.4 + 97.8) / 4 = **84.45%** ≈ 84.5% ✓ (verify writeup table — may show different column)
- D→T R@10 (folds 0-3): (45.0 + 77.9 + 27.2 + 100.0) / 4 = **62.525%** ≈ 62.5% ✓
- T→D R@10 (folds 0-3): (79.8 + 97.4 + 67.4 + 98.0) / 4 = **85.65%** ≈ 85.7% ✓

The fold-0-3 exclusion arithmetic verifies cleanly. Writeup §3.2 is internally consistent.

### 3.5 Summary table from log

The pool-512 log ends with a final summary table (writeup §3.2 references this as "Tables 21-22 (unique drug-target pairs)"):

```
Split            R@1(D→T)     R@1(T→D)     R@5(D→T)     R@5(T→D)     R@10(D→T)    R@10(T→D)
identity_100     27.7±3.3     60.5±8.7     42.6±2.2     70.3±8.8     46.7±1.9     74.9±8.2
mmseqs_080       46.0±0.0     95.7±0.0     79.9±0.0     95.8±0.0     91.2±0.0     95.9±0.0
mmseqs_060       37.1±10.7    74.4±36.2    47.3±13.6    76.8±33.6    62.1±19.7    78.6±31.3   ← all 5 folds, includes fold 4
mmseqs_040       34.8±20.9    66.6±34.8    46.7±29.5    68.9±33.6    54.1±30.3    70.7±32.1   ← all 5 folds, includes fold 4
```

**Important note:** This table includes fold 4 in mmseqs_060/040 means. The writeup §3.2 reports the fold-0-3 means (excluding fold 4), but readers consulting the pool-512 log directly will see fold-inclusive values. This is **not a writeup error** — the writeup explicitly says "Excluding fold 4 (degenerate validation set) from OOD averages" — but the pool-512 log itself only computed full-fold means.

The "Tables 21-22" reference suggests the writeup may have a separate explicit fold-0-3 computation. The audit verifies whether this computation is reproducible from the per-fold values in the log (which I have verified manually above; arithmetic is correct).

---

## Cross-references for audit YAML

This document is the source-of-truth for the audit's CL items. Every value above is recoverable from at least one of:

- `audit/dtsfm_v2/archival_logs/dtsfm2_p512_63407083.out` (primary: §3.1, §3.2, §3.3, §3.4, §3.5)
- `audit/dtsfm_v2/archival_logs/dtsfm2_full_63170670_*.out` and `dtsfm2_full_63170839_*.out` (training-log echoes for §1 dataset, §2 hyperparameters, fold-4 epoch-0 evidence)
- `audit/dtsfm_v2/archival_logs/dtsfm2_mmseqs_*.out` (§1 cluster counts) — note: not yet in archival_logs, must be pulled or referenced from session-time observation
- `data/dtsfm_v2/metadata.csv` (§1.2 source data verification)
- `data/dtsfm_v2/{davis,kiba,bindingdb_kd}_raw.csv` (§1.2 source data files)

The audit YAML enumerates each CL item with explicit pointers and the digit-exact values to verify.

---

## Anticipated PARTIAL findings (Path A pre-acknowledgments from SCOPING §3.4)

Five findings will surface during audit. The audit YAML's `expected_partial_finding` annotations document them in advance:

1. **Finding 1: Fold-4 degeneracy across mmseqs_060 and mmseqs_040.** Cluster-CV split bug, fifth program-level recurrence (GitHub issue #6). Documented and excluded from writeup §3.2 means. Audit verifies the exclusion arithmetic (above ✓).

2. **Finding 2: Pool-size definition ambiguity.** Writeup §3.1 says "Identity_100 has ~920 unique targets per fold (true pool-512 evaluation)" but min(512, 920) = 512. Audit reads `src/calm/eval/eval_pool512.py` to determine actual pool definition.

3. **Finding 3: mmseqs_080 single-fold framing.** Folds 1-4 of mmseqs_080 have no checkpoints (training did not produce usable models). Writeup correctly labels as "indicative but not averaged" but the "± 0.0%" notation can mislead. Recommend paper-prep clarification.

4. **Finding 4: v2 configs not preserved as v2-named YAML files.** v2 hyperparameters extracted from training-log echo. Same shape as crisprSFM preservation finding.

5. **Finding 5: Cross-SFM debug-label inheritance.** "Loading tSFM unique embeddings... TF proteins: 10338 unique" persists from tSFM data loader. Fifth occurrence in the program.
