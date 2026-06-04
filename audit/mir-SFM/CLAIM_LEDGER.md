# mir-SFM Numerical Claims (RETRAIN_LOG-equivalent)

**Source writeup:** `V-SFM-manuscripts/miRNAsfm_results_validation_VC.docx`
**Date generated:** 2026-05-02
**Purpose:** Comprehensive claim ledger for digit-exactness audit by Codex against committed source artifacts.

This document mirrors the role of `audit/tsfm/RETRAIN_LOG.md` and `audit/mhcsfm/RETRAIN_LOG.md`: every numeric claim in the writeup is enumerated here with its source artifact, so that Codex (Leg 1 SA-12 digit-exactness meta-audit) can verify each claim from a single reference document.

---

## Section 1 — Dataset

### 1.1 Counts

| Property | Value | Source |
|---|---|---|
| Total training pairs | 206,692 | `audit/mirsfm/archival_logs/mirsfm_full_62523321_0.out` ("Total pairs: 206692"); `audit/mirsfm/archival_logs/mirsfm_preprocess_62519445.out` final state |
| Unique miRNAs | 135 | training log "TF proteins: 135 unique" (debug-label inheritance — see SA-13); `mirsfm_mmseqs_62523236.out` "Loaded 206692 rows, 135 unique sequences" |
| Unique target sites | 110,221 | training log "DNA sequences: 110221 unique"; preprocess log final ag_embed.pt shape |
| Canonical pairs (seed match exists) | 43,472 (21%) | TargetScan flag classification per writeup §1.2; recoverable from `data/mirsfm/mirsfm_all_pairs.tsv` |
| Non-canonical pairs (no seed match) | 163,220 (79%) | Same source; arithmetic check 43,472 + 163,220 = 206,692 ✓ |

### 1.2 Data sources

- **miRTarBase 2025:** 380,257 experimentally validated miRNA-gene interaction pairs (`data/mirsfm/hsa_MTI.xlsx`, 42 MB)
- **ENCORI:** 582,405 AGO-CLIP-seq binding sites across 139 miRNAs (`data/mirsfm/encori_clip_sites.tsv`, 91 MB)
- **miRBase:** 2,656 mature human miRNA entries (`data/mirsfm/mature.fa`, 3.9 MB)
- **hg38:** human genome reference for sequence extraction (`data/mirsfm/hg38.2bit`, 835 MB)
- **TargetScan v8 conservation tables** (`data/mirsfm/Conserved_Family_Info.txt`, 535 MB; `Predicted_Targets_Context_Scores.default_predictions.txt`, 137 MB) for canonical/non-canonical classification

### 1.3 Validation holdout

Writeup §1: "Validation holdout: 40,805 non-canonical pairs (Option A)"

**Note:** This is described as a holdout but Section 5.2 clarifies the design: there is no separate holdout experiment; the validation is embedded in the pool-512 evaluation by stratifying results on canonical vs non-canonical target type. The 40,805 number is the count of non-canonical pairs in the test partition of fold 0.

**Source:** Recoverable from training log fold split sizes (test: 41,338 pairs at identity_100 fold 0; non-canonical fraction × 41,338 ≈ 40,805 if fraction is ~98.7%, but the canonical/non-canonical breakdown of the test set must be verified during audit).

---

## Section 2 — Training Configuration

| Parameter | Value | Source |
|---|---|---|
| Architecture | FFN projection head | `configs/model/encoder/mirsfm_ffn.yaml` |
| d_model (projection dim) | 512 | config |
| Pooling | Mean pooling | source code |
| Temperature τ initialization | 0.07 (max_scale = 100) | config |
| Loss | Symmetric InfoNCE (bidirectional contrastive) | source code |
| Optimizer | AdamW (lr=0.001, weight_decay=0.2) | `configs/train/encoder/mirsfm_full.yaml` |
| Scheduler | CosineAnnealingWarmRestarts (T0=20, Tmult=1) | config |
| Epochs | 100 | config |
| Batch size | 64 | config |
| Checkpoint | Best validation prediction accuracy | source code |
| OOD split | By miRNA identity (cluster_id = miRNA name) | `mirsfm_mmseqs_62523236.out` + writeup §2 |
| Folds | 5-fold outer × 5-fold inner CV (25 splits) | `mirsfm_preprocess_62519445.out` "split_index/: CV split index files (5×5 = 25)" |

### 2.1 Encoder architecture

- **Agent (miRNA) encoder:** DNABERT-2 (`zhihan1996/DNABERT-2-117M`, 768-dim, 12 tokens after tokenization)
- **Target encoder:** DNABERT-2 (same model)
- **U-to-T conversion:** Both miRNA and target sequences converted U→T before DNABERT-2 tokenization. Same protocol as crisprSFM.

### 2.2 MMseqs2 clustering outcome (pre-LV finding)

**Critical finding documented in `mirsfm_mmseqs_62523236.out`:**

| Threshold | Cluster count |
|---|---|
| 40% identity | 135 clusters from 135 sequences |
| 60% identity | 135 clusters from 135 sequences |
| 80% identity | 135 clusters from 135 sequences |

All three thresholds produced singleton-only output. The "OOD split" used in training is functionally **by-miRNA-identity holdout** — each fold tests on entirely-unseen miRNAs, partitioned by miRNA name rather than similarity-based clustering.

This is the writeup §3.2 finding: "MMseqs2 identity thresholds at 40%, 60%, and 80% produced identical splits for miRNA sequences (~22 nt). Short nucleotide sequences cannot be meaningfully distinguished at these thresholds."

---

## Section 3 — Pool-512 Retrieval Results

Pool-512 evaluation: pool size 512, 100 random trials per fold, 500 queries subsampled per trial. Chance: R@1 = 0.2%, R@5 = 1.0%, R@10 = 2.0%.

### 3.1 In-Distribution (identity_100) — across 5 folds

Writeup §3.1 reports approximate values. Cross-fold mean (SD across 5 folds) from `audit/mirsfm/archival_logs/mirsfm_pool512_62780089.out`:

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| miRNA → target | **98.0% ± 0.6%** | 100.0% ± 0.0% | 100.0% ± 0.0% |
| target → miRNA | **25.4% ± 4.2%** | 53.4% ± 7.4% | 67.5% ± 7.2% |

**Per-fold values (from log):**

| Fold | R@1 ag→ab | R@1 ab→ag | R@5 ag→ab | R@5 ab→ag | R@10 ag→ab | R@10 ab→ag |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 98.4 | 26.4 | 100.0 | 56.3 | 100.0 | 70.9 |
| 1 | 98.1 | 26.4 | 100.0 | 56.0 | 100.0 | 70.5 |
| 2 | 98.5 | 27.2 | 100.0 | 57.6 | 100.0 | 71.5 |
| 3 | 98.3 | 29.5 | 100.0 | 58.3 | 100.0 | 71.5 |
| 4 | 96.8 | 17.3 | 100.0 | 38.6 | 100.0 | 53.0 |

**Note (writeup imprecision):** Writeup §3.1 cites "98.4%" for miRNA→target R@1 — this matches fold 0 only, not the cross-fold mean (98.0%). The "~100%" notation in the writeup is consistent with the cross-fold means (R@5 and R@10 both 100.0% ± 0.0%). Fold 4 is a clear outlier on the ab→ag direction (17.3% vs 26.4-29.5% for other folds), driving the high cross-fold SD. Worth investigating during audit.

### 3.2 OOD by miRNA Identity (mmseqs_080) — across 5 folds

From log:

| Fold | Test pool | R@1 ag→ab | R@1 ab→ag | R@5 ag→ab | R@5 ab→ag | R@10 ag→ab | R@10 ab→ag |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 41,920 | 98.8 | 64.0 | 99.4 | 99.3 | 99.5 | 99.3 |
| 1 | 41,408 | 99.8 | 67.2 | 100.0 | 99.8 | 100.0 | 99.8 |
| 2 | 41,728 | 99.9 | 76.4 | 100.0 | 100.0 | 100.0 | 100.0 |
| 3 | 41,024 | 99.7 | 65.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| 4 | 40,384 | 99.8 | 70.7 | 100.0 | 100.0 | 100.0 | 100.0 |

**Cross-fold summary (from log):**

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| miRNA → target | **99.6% ± 0.4%** | 99.9% ± 0.3% | 99.9% ± 0.2% |
| target → miRNA | **68.8% ± 4.4%** | 99.8% ± 0.3% | 99.8% ± 0.3% |

**Note:** Writeup §3.2 only tabulates folds 0/1/2, omitting folds 3/4. Cross-fold means in the writeup must reconcile against the log values above.

### 3.3 Batch-64 Training Accuracy Summary

Writeup §3.3 claims:

| Split Method | Test Accuracy (mean ± s.d.) | Best Epoch |
|---|---|---|
| identity_100 | 45.9% ± 0.5% | 15-28 |
| mmseqs_080 | 33.0% ± 6.0% | 23-53 |

**Source:** Recoverable from training logs `mirsfm_full_62523321_*.out` (20 logs total, one per fold per split). Each log reports best validation prediction accuracy at training completion. Cross-fold mean and SD computed across 5 folds per split.

This is **batch-64 prediction accuracy** — a different metric from pool-512 retrieval R@1. It measures whether the contrastive model's argmax over the 64-batch matches the true positive pair (1 in 64 = 1.6% chance baseline).

---

## Section 4 — Seed-Matching Baseline Comparison

### 4.1 Overall Comparison

The seed-matching baseline computes the reverse complement of miRNA positions 2-8 (the seed region), checks each candidate target for the 7-mer, ranks candidates by site type (8mer > 7mer-m8 > 7mer-A1 > 6mer > no match). This is the algorithmic core of TargetScan.

From `mirsfm_pool512_62780089.out` baseline section:

| Method | R@1 (ID) | R@5 (ID) | R@10 (ID) | R@1 (OOD) | R@5 (OOD) | R@10 (OOD) |
|---|---:|---:|---:|---:|---:|---:|
| miR-SFM | 98.4% | ~100% | ~100% | ~99% | 100% | 100% |
| Seed matching | **6.8%** | **29.0%** | **47.7%** | **3.5%** | **16.4%** | **31.2%** |
| Advantage | +91.6 | +71 | +52 | +95.5 | +84 | +69 |

**Note:** The miR-SFM "98.4% ID" matches fold 0 (writeup imprecision; cross-fold mean is 98.0%). The seed-matching baseline values are from fold 0 of each split (only fold 0 baseline numbers are in the pool-512 log).

### 4.2 Breakdown by Target Type — THE HEADLINE RESULT

| Target Type | Seed Matching R@1 | miR-SFM R@1 | Advantage |
|---|---:|---:|---|
| Canonical (seed match exists, 21% of test) | **9.0%** | ~98% | +89 pts |
| Non-canonical (no seed match, 79% of test) | **6.1%** | ~98% | +92 pts |
| Overall | 6.8% | 98.4% | +91.6 pts |

**Source:** From pool-512 log baseline section (identity_100):
```
=== Baseline: identity_100 ===
  miRNA→target: R@1=6.8%, R@5=29.0%, R@10=47.7%
    Canonical:     R@1=9.0%
    Non-canonical: R@1=6.1%
```

Same structure for mmseqs_080:
```
=== Baseline: mmseqs_080 ===
  miRNA→target: R@1=3.5%, R@5=16.4%, R@10=31.2%
    Canonical:     R@1=5.2%
    Non-canonical: R@1=2.9%
```

**This is the single most important table in the paper for mir-SFM.** It demonstrates that seed-matching, even when the seed match exists, only achieves 9% R@1 — far below miR-SFM's ~98%. And critically, on non-canonical targets (79% of biology), miR-SFM achieves the same ~98% while seed-matching can only get 6.1%.

### 4.3 Cross-domain contrast with crisprSFM

Writeup §4.3 narrative claim:

| Domain | SFM R@1 | Baseline R@1 | Winner |
|---|---:|---:|---|
| crisprSFM | 10.2% | 45.8% (Hamming) | Baseline |
| miR-SFM | 98.4% | 6.8% (Seed match) | SFM (+91.6) |

**Source:** miR-SFM values from this audit; crisprSFM values from `audit/crisprSFM/` audit closure (cite the corresponding tag/commit).

---

## Section 5 — Validation Narrative (Option A: Non-Canonical Target Recovery)

§5 is descriptive narrative explaining the validation design rationale. No new numerical claims beyond those already covered in §4.2.

The 80% non-canonical fraction claim ("approximately 80% of experimentally validated miRNA-target interactions lack a canonical seed match") is verified via §1.1 dataset construction: 163,220 / 206,692 = 79.0%. ✓

---

## Sections 6, 7, 8 — Translational, Roadmap, Decision Log

§6 (translational impact: miRNA therapeutics, MRX34 reference), §7 (roadmap: RNA-FM upgrade, transcriptome-wide scoring, safety profiling), and §8 (decision log) are paper narrative without new numerical claims.

The single specific factual claim in §6.1 (MRX34 halted in Phase 1 in 2016 after four patient deaths) is an external claim not subject to internal-data verification; the audit may flag it as "external claim, requires literature cross-check at paper-prep" but this is not a CL item.

---

## Cross-references for audit YAML

This document is the source-of-truth for SA-12 (digit-exactness meta-audit). Every value above is recoverable from at least one of:
- `audit/mirsfm/archival_logs/mirsfm_pool512_62780089.out` (Sections 3, 4)
- `audit/mirsfm/archival_logs/mirsfm_full_62523321_0.out` and `_10.out` (Section 1, 2, 3.3)
- `audit/mirsfm/archival_logs/mirsfm_mmseqs_62523236.out` (Section 2.2 cluster counts)
- `audit/mirsfm/archival_logs/mirsfm_preprocess_62519445.out` (Section 1 final data state)
- `data/mirsfm/` source data files (Section 1.2 data source verification)
- `configs/train/encoder/mirsfm_full.yaml` and `configs/model/encoder/mirsfm_ffn.yaml` (Section 2 hyperparameters)

The audit YAML enumerates each CL item with explicit pointers.
