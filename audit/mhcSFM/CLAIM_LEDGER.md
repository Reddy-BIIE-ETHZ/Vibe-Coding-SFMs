# mhcSFM Numerical Claims (RETRAIN_LOG-equivalent)

**Source writeup:** `V-SFM-manuscripts/mhcSFM_results_validation_VC.docx`
**Date generated:** 2026-04-27
**Purpose:** Comprehensive claim ledger for digit-exactness audit by Codex against committed source artifacts.

This document mirrors the role of `audit/tsfm/RETRAIN_LOG.md` for tSFM v2: every numeric claim in the writeup is enumerated here with its source artifact, so that Codex (Leg 1 SA-12 digit-exactness meta-audit) can verify each claim from a single reference document.

Citation note: the Genentech monoallelic neoantigen dataset is cited as **Gurung et al., 2023, Nature Biotechnology** throughout this audit. The original session-time writeup labels it "Pyke 2024" — this is a first-author misattribution. See `SCOPING.md` §4. Internal SLURM/CSV/script filenames retain `pyke` prefixes as historical artifacts.

---

## Section 1 — Dataset

### 1.1 Counts

| Property | Value | Source |
|---|---|---|
| Total training pairs | 168,710 | `data/mhcsfm/metadata.csv` row count; `mhcsfm_full_62515934_0.out` "Total pairs:" line |
| Unique peptides | 38,837 | `metadata.csv` unique peptide column |
| Unique HLA alleles | 105 | `metadata.csv` unique allele column; `data/mhcsfm/NetMHCpan_train/MHC_pseudo.dat` |
| Binders (IC50 < 500 nM, 1-log50k ≥ 0.426) | 41,877 (24.8%) | `metadata.csv` binary_label==1 count |
| Non-binders (hard negatives) | 126,833 (75.2%) | `metadata.csv` binary_label==0 count |

### 1.2 Peptide length distribution

| Length | Fraction |
|---|---|
| 8-mer | 6.1% |
| 9-mer | 69.1% |
| 10-mer | 19.4% |
| 11-mer | 4.8% |
| 12-14 mer | 0.5% |

**Source:** Recoverable from `metadata.csv` peptide string lengths. Total adds to 99.9% (rounding).

### 1.3 Validation holdout

- **14 rare alleles selected:** HLA-A\*02:12, A\*24:03, A\*26:02, A\*80:01, B\*15:02, B\*15:03, B\*40:02, B\*51:01, C\*03:03, C\*06:02, C\*07:01, C\*12:03, C\*14:02, C\*15:02
- **Coverage:** 13 distinct supertypes + HLA-A\*80:01 (unclassified)
- **Binders removed from training:** 2,874 (5.5% of training binders)
- **Source:** `data/mhcsfm/find_holdout_candidates.py`, `analyze_holdout_representativeness.py`, `build_holdout_split.py`

### 1.4 Holdout representativeness

Hamming distance on 34-residue pseudo-sequences to the nearest training allele:

| Population | Median Hamming distance | Identity equivalent |
|---|---|---|
| 14-allele holdout | 3 / 34 mismatches | 91.2% identity |
| ~10,000 other HLA-A/B/C alleles in IMGT | 1 / 34 mismatches | 97.1% identity |

**Additional finding:** 38.3% of the ~10,000 other HLA alleles share an identical pseudo-sequence with some training allele.

**Source:** `analyze_holdout_representativeness.py`; reproducible from `data/mhcsfm/allele_sequences.csv` and `MHC_pseudo.dat`.

### 1.5 34-residue pseudo-mask positions

Mature protein numbering: 7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171.

**Verification claim:** masked extraction reproduces NetMHCpan MHC_pseudo.dat sequence exactly for **105 of 105** classical HLA-A/B/C alleles.

**Source:** `data/mhcsfm/pseudo_mask_positions.txt`, `embed_mhcsfm.py`, `data/mhcsfm/NetMHCpan_train/MHC_pseudo.dat`.

### 1.6 NetMHCpan 4.1 BA training data

- 5 cross-validation files yielded **208,093** peptide-HLA pairs across **170** alleles
- After filtering to classical HLA-A/B/C with verified IMGT sequences, excluding HLA-E: **168,710** pairs across **105** alleles
- Binding threshold: 1-log50k ≥ 0.426 (equivalent to IC50 < 500 nM)

**Source:** `data/mhcsfm/NetMHCpan_train.tar.gz` (91 MB) → `data/mhcsfm/NetMHCpan_train/c{000..004}_{ba,el}` files; `build_metadata.py`.

---

## Section 2 — Training Configuration

| Parameter | Value | Source |
|---|---|---|
| Architecture | FFN projection head | `configs/model/encoder/mhcsfm_ffn.yaml` |
| d_model (projection dim) | 512 | config |
| d_ff (hidden dim) | 2048 | config |
| Pooling | Mean over masked positions (HLA side) | source code |
| Temperature τ initialization | 0.07 | config; max_scale = 100 |
| Loss | Symmetric InfoNCE with multi-positive NNK masking | source code |
| Optimizer | AdamW (lr=0.001, weight_decay=0.2) | `configs/train/encoder/mhcsfm_full.yaml` |
| Scheduler | CosineAnnealingWarmRestarts (T0=20, Tmult=1) | config |
| Epochs | 100 | config |
| Batch size | 64 | config |
| Checkpoint policy | Best validation prediction accuracy | source code |

### 2.1 Cluster counts at OOD thresholds (pseudo-sequence clustering)

| MMseqs2 threshold | Clusters from 105 pseudo-sequences |
|---|---|
| 40% identity | 2 clusters (degenerate; mmseqs_040 excluded from final tables) |
| 60% identity | 5 clusters |
| 80% identity | 31 clusters |

**Note:** Clustering on full α1-α2 (182 aa) gives 2 clusters at 40% and 1 cluster at 60% — too coarse. Pseudo-sequence clustering produces meaningful splits. Documented in writeup §2.

**Source:** `mmseqs_splits.py` reproducible run on pseudo-sequences. SA-4-style stochasticity caveat: ±1 cluster count tolerance per tSFM SA-4 evidence.

---

## Section 3 — Pool-512 Retrieval Results

Retrieval evaluation: pool size 512, 100 random trials per fold, 200 queries subsampled per trial. Chance: R@1 = 0.2%, R@5 = 1.0%, R@10 = 2.0%.

### 3.1 Bidirectional 5-fold cross-validation (mean across 5 folds)

#### Peptide → MHC allele direction

| OOD threshold | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| identity_100 (in-distribution) | 65.1% | 78.4% | 81.8% |
| mmseqs_080 (31 clusters) | 67.9% | 92.4% | 92.9% |
| mmseqs_060 (5 clusters) | 69.2% | 93.8% | 94.0% |

#### MHC allele → peptide direction

| OOD threshold | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| identity_100 | 93.3% | 99.2% | 99.5% |
| mmseqs_080 | 96.1% | 98.4% | 99.2% |
| mmseqs_060 | 95.4% | 98.2% | 99.3% |

**Source:** `audit/mhcsfm/archival_logs/mhcsfm_pool512_62630321.out` — 15 `=== Summary ===` blocks (3 splits × 5 folds), per-fold R@1_ag2ab and R@1_ab2ag values verifiable inline.

### 3.2 Interpretation note (from writeup)

The MHC→peptide direction R@1 ≈ 95% reflects that mhcSFM has 168K pairs but only 105 unique alleles, so each allele has ~1,600 binder peptides on average. Pool-512 sampling means many test pairs share an allele, and multi-positive NNK masking counts any of these as correct. The peptide→MHC direction R@1 ≈ 65% is harder because a peptide typically binds only 1-3 alleles out of 105. **High numbers reflect dataset structure, not model strength claim.** This is documented honestly in the writeup as Section 3.2.

### 3.3 Rare allele zero-shot holdout

Single-fold pool-512 evaluation on the 91-allele-trained model evaluated on 14 unseen alleles (12,169 test pairs):

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Peptide → allele (harder) | 53.2% | 88.5% | 91.8% |
| Allele → peptide (easier) | 95.4% | 99.3% | 100.0% |

**Source:** `audit/mhcsfm/archival_logs/mhcsfm_holdout_62779013.out`.

**Interpretation (writeup):** The 95.4% R@1 on alleles with zero training data demonstrates that the pseudo-sequence masked embedding captures transferable groove geometry.

---

## Section 4 — Head-to-Head vs NetMHCpan 4.1

Validation set: Gurung et al. 2023 Nature Biotechnology. 25,260 peptide-HLA combinations across 47 cancer neoantigens × 15 prevalent HLA alleles. Two orthogonal readouts: TR-FRET biochemical binding (609 binders) and LC-MS/MS immunopeptidomics (86 MS-validated pairs).

### 4.1 AUROC head-to-head

| Model | vs TR-FRET (n=609 binders) | vs MS-confirmed (n=86 presented) |
|---|---:|---:|
| mhcSFM | AUROC 0.616 | AUROC 0.778 |
| NetMHCpan 4.1 BA | AUROC 0.756 | AUROC 0.968 |
| NetMHCpan 4.1 EL | AUROC 0.752 | AUROC 0.969 |
| Random baseline | 0.500 | 0.500 |

**Source:** `data/mhcsfm/compare_netmhcpan.py` re-run on `data/mhcsfm/pyke_3way_comparison.csv` (5 MB, contains all per-pair scores merged).

### 4.2 NetMHCpan false positive rates on TR-FRET ground truth

| NetMHCpan prediction | Predicted binders | TR-FRET confirmed | False positive rate |
|---|---:|---:|---:|
| BA binary call | 436 | 159 (36.5%) | 63.5% |
| EL binary call | 431 | 160 (37.1%) | 62.9% |

**Source:** `data/mhcsfm/analyze_false_positives.py` re-run on `pyke_3way_comparison.csv`.

### 4.3 Cascade filter (NetMHCpan ∩ mhcSFM top-50)

| Filter | N predictions | TR-FRET hits | TR Precision | MS Precision |
|---|---:|---:|---:|---:|
| NetMHCpan BA alone (baseline) | 436 | 159 | 36.5% | — |
| NetMHCpan BA ∩ mhcSFM top-50 | 50 | 24 | 48.0% | — |
| NetMHCpan EL alone (baseline) | 431 | 160 / 73 (MS) | 37.1% | 16.9% (MS) |
| **NetMHCpan EL ∩ mhcSFM top-50** | **50** | **26 / 15 (MS)** | **52.0%** | **30.0% (MS)** |

**Headline claim:** NetMHCpan EL ∩ mhcSFM top-50 nearly doubles MS presentation precision from 16.9% to 30.0%, halving the false positive rate.

**Source:** `data/mhcsfm/compare_netmhcpan.py` cascade-filter computation.

### 4.4 Top-K agreement table

| Top-K by mhcSFM | mhcSFM alone (TR prec.) | ∩ NetMHCpan BA | ∩ NetMHCpan EL |
|---|---:|---:|---:|
| K=100 | 11.0% | 60.0% | 50.0% |
| K=200 | 8.5% | 55.6% | 57.1% |
| K=500 | 6.6% | 51.3% | 55.3% |
| K=1000 | 5.9% | 49.2% | 53.1% |

**Source:** `compare_netmhcpan.py` agreement-set computation.

---

## Section 5 — Gurung Validation Narrative

### 5.1 ESCAPE-seq abandonment (historical decision document)

ESCAPE-seq (Shi et al., Nature Genetics 2025) was attempted as the first validation dataset:
- 75,900 peptide-HLA combinations across 50 HLA alleles with oncogenic mutations
- Mean per-allele Spearman correlation against mhcSFM scores: **0.06** (n.s., binomial p = 0.32)

**Diagnosed cause (writeup):** ESCAPE-seq measures presentation efficiency (single-chain expression + presentation), not binding affinity. mhcSFM was trained on binding affinity data, so the ground-truth signals are mismatched. **Not a model failure; a methodology mismatch.**

**Source:** `data/mhcsfm/escape_validation_results.csv`, `data/mhcsfm/escape_cosine_sim.csv`, `analyze_escape.py`. Audit verifies the 0.06 number is recoverable; ESCAPE not in scope as a paper claim.

### 5.2 Gurung 86 MS-validated retrieval

Task: for each of 86 MS-validated peptide-HLA pairs, rank the 12 test alleles by cosine similarity for the given peptide. Does the true allele appear in top-1 / top-3 / top-5?

| Metric | mhcSFM | Random (12 alleles) | Fold over random | Binomial p |
|---|---:|---:|---:|---|
| R@1 | 24.4% (21/86) | 8.3% | 2.9× | 5.9e-06 |
| R@3 | 64.0% (55/86) | 25.0% | 2.6× | 2.9e-14 |
| R@5 | 80.2% (69/86) | 41.7% | 1.9× | — |
| Mean rank | 3.52 / 12 | 6.5 / 12 | — | — |

**Zero-shot HLA-C\*03:04:** never seen in any mhcSFM training data, achieved **100% R@3** on its 2 validated pairs.

**Source:** `audit/mhcsfm/archival_logs/mhcsfm_pyke_62831745.out` (load + setup), `data/mhcsfm/pyke_mhcsfm_ranks.csv` (the rankings), `data/mhcsfm/analyze_pyke.py` (computation).

### 5.3 TR-FRET ground truth and MS false negative rate

| Measurement | Count | Percentage |
|---|---:|---:|
| TR-FRET biochemical binders (full dataset) | 609 | 2.4% |
| Also captured by MS | 67 | 11.0% of TR-FRET binders |
| Missed by MS (false negatives) | 542 | 89.0% of TR-FRET binders |
| MS-confirmed pairs (Gurung's 86 headline) | 86 | 0.3% |

**MS false negative rate:** 89% of biochemical binders are not captured by MS.

**AUROC contrast:**
- vs TR-FRET ground truth: 0.616
- vs MS ground truth: 0.778

**At K=50:** mhcSFM achieves 12% TR-FRET precision vs 2.4% random (5× enrichment).

**Source:** `audit/mhcsfm/archival_logs/mhcsfm_pykefull_62840008.out` (cosine sim computation), `data/mhcsfm/pyke_trfret_mhcsfm_merged.csv` (per-pair scores), `data/mhcsfm/analyze_pyke_trfret.py` (computation), `data/mhcsfm/score_pyke_full.py`.

### 5.4 TCGA 8,950-patient vaccine simulation

For each of 8,950 real TCGA patients (HLA typings from Gurung Supplementary Table 1), present mhcSFM with the 71 Gurung neoantigen peptides and the patient's 6 HLA alleles. Rank the 71×6 = 426 candidate pairs. Count MS-validated hits in top-K.

| K (vaccine size) | Mean hits | Precision | Recall | Random expected | Fold over random |
|---|---:|---:|---:|---:|---:|
| 10 | 4.56 | 45.6% | 29.9% | 0.97 | 4.72× |
| **20 (typical)** | **7.23** | **36.2%** | **44.0%** | **1.93** | **3.75×** |
| 30 | 9.09 | 30.3% | 55.8% | 2.90 | 3.14× |
| 50 | 11.32 | 22.6% | 67.9% | 4.83 | 2.35× |

**Aggregate AUROC across all patient-pair evaluations:** 0.795.

**Coverage statistics across 8,950 patients (top-20 vaccine):**
- 97% received at least one validated peptide
- 83% received at least four
- 47% received seven or more

**Cross-cancer-type stability:** Performance stable across all 33 TCGA cancer types.

**Source:** `data/mhcsfm/simulate_vaccine_design.py` re-run on `data/mhcsfm/vaccine_simulation_results.csv` (3 MB).

### 5.5 Per-allele AUROC variability

**Well-trained alleles:**
- HLA-A\*02:01: 0.839
- HLA-B\*35:01: 0.767
- HLA-A\*11:01: 0.742
- HLA-B\*07:02: 0.721

**HLA-C alleles (under-trained):**
- HLA-C\*04:01: 0.375 (below random)
- HLA-C\*05:01: 0.429
- HLA-C\*07:02: 0.521

**Diagnosis (writeup):** HLA-C is historically underrepresented in immunopeptidomics and binding affinity datasets. Clear improvement target for next version (per writeup §7.1, "Balance HLA-C representation").

**Source:** `data/mhcsfm/pyke_per_allele_3way.csv`, `pyke_per_allele_trfret.csv`.

---

## Section 6 — Translational Impact (writeup §6, paper-narrative — not numerical claims)

§6.1 (current standard limitations) and §6.2 (what mhcSFM adds today, four capabilities) and §6.3 (caveats) are narrative and not subject to numerical-claim verification, except where they reference Section 4-5 numbers (which are verified above).

§6.2 references:
- "16.9% to 30.0% MS precision" — verified via Section 4.3
- "95.4% R@1 (allele-to-peptide) on 14 completely unseen alleles" — verified via Section 3.3

---

## Section 7 — Roadmap (writeup §7, paper-narrative)

§7.1-7.4 (GPT-2/3/4 equivalents roadmap) are forward-looking, not subject to claim verification. Audit treats §7 as narrative scope, no verifiable numbers.

---

## Section 8 — Domain Expert Decision Log (writeup §8, paper-narrative)

8 decisions with rationales documented. Single verifiable claim:
- "**105/105 pseudo-sequences verified against NetMHCpan MHC_pseudo.dat**" — already covered by §1.5 of this document.

Audit treats §8 as narrative scope; no additional CL items derived from it.

---

## Cross-references for audit YAML

This document is the source-of-truth for SA-12 (digit-exactness meta-audit). Every value above is recoverable from at least one committed artifact in either `audit/mhcsfm/archival_logs/` or `data/mhcsfm/`. The audit YAML enumerates each CL item with explicit pointers.
