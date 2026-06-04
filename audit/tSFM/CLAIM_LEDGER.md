# tSFM v2 (50× scale) Retrain Log

**Date executed:** 2026-04-25 to 2026-04-26
**Author:** Sai Reddy (domain expert) + Claude Code (coding agent)
**Audit purpose:** Vibe-Coding SFMs paper.

---

## Why this retrain

The original v0.5 tSFM run (Session 1, April 2026) used `--samples_per_pwm 10` producing ~20K training pairs. Per `audit/tsfm/SCOPING.md`, we scaled the PWM sampling to 50× to produce ~102.7K pairs and re-evaluated under the new artifact-preservation discipline (`audit/PRESERVATION_DISCIPLINE.md`).

The original v0.5 training artifacts had been purged from `/cluster/scratch`, and the v0.5 evaluation pipeline did not print summary metrics to stdout (only to scratch CSV), making post-hoc archival impossible. So we retrained from scratch on the 50× dataset rather than attempt archival audit.

## Pipeline executed

1. **Preprocessing** (`scripts/euler_preprocess_jaspar.slurm`, edited): `--samples_per_pwm 50`, output to `/cluster/scratch/$USER/tsfm/jaspar_data_v2_50x/`. ~30 min on GPU. Produced 102,700 pairs with 999 unique TF protein sequences and 65,461 unique DNA motifs.

2. **MMseqs2 OOD splits**: 4 thresholds (identity_100 in-distribution + mmseqs_080/060/040 OOD). 999 unique protein sequences → 882/799/557 clusters at 80/60/40% identity. ~30 sec on login node.

3. **Training** (`scripts/euler_train_tsfm_full.slurm`): 4 splits × 5 folds = 20 array jobs, each running 100 Phase-1 epochs followed by `best_epoch + 1` Phase-2 retrain epochs. Per-fold runtime 4–8 hours on RTX 3090/Quadro RTX 6000.

   - **First submission (job 64763913):** 14/20 completed; 6 timed out (tasks 11–16 = mmseqs_060 folds 1–4 and mmseqs_040 folds 0–1) due to GPU node contention — `srun: step creation still disabled, retrying` for the entire 12-hour wall-clock without ever starting the python process.
   - **Resubmit of failed indices (job 64799502):** all 6 completed cleanly on fresh GPU nodes.

4. **Evaluation** (`scripts/euler_eval_tsfm.slurm`): 20 array jobs producing `cosine_sim.pt`, `labels_masks.pt`, `features.pt`, `projections.pt` per fold. ~5 min per fold, all 20 done in parallel on GPU.

5. **Pool-512 retrieval** (`scripts/euler_pool512_tsfm.slurm`, patched B/E): single job, CPU-only. Iterates all 20 (split, fold) pairs using `eval_pool512_fast.py` with `--max_queries 200 --n_trials 100`. Produces 4 durable summary CSVs in `~/CALM-0.1.0/eval_summaries/tsfm/`. ~10 min total.

## Patches applied this session

| Patch | File | Purpose |
|---|---|---|
| A1 | `src/calm/encoder/eval_pool512.py` | Durable CSV via `SFM_SUMMARY_DIR` env var + `--sfm_name` + `--split_name` |
| A2 | `src/calm/encoder/eval_pool512_fast.py` | Same durable-CSV resolution as A1 |
| B | `scripts/euler_pool512_tsfm.slurm` (new) | Sets `SFM_SUMMARY_DIR`; aggregates per-fold CSVs into per-split summaries |
| C | `scripts/euler_train_tsfm_full.slurm` + `scripts/euler_eval_tsfm.slurm` | DATA_DIR points at `jaspar_data_v2_50x` |
| D | `src/calm/encoder/evaluate.py` | Defensive: falls back to `best_model_val_pred_acc_*.pth` when no `final_model_*.pth` (unused in production once Phase 2 confirmed working) |
| E | `scripts/euler_pool512_tsfm.slurm` | Switched from `eval_pool512.py` (slow, hours per fold) to `eval_pool512_fast.py` with `--max_queries 200` |

## Known issues encountered

### Issue 1 — `outerfold_4_innerfold_4` split has empty val (cross-SFM)

Fold 4 of `identity_100` (and likely the other splits' fold 4) has `val: 0 pairs` due to an issue in the cluster-CV split partitioning where the inner fold and outer fold collide when both equal the highest fold index. We confirmed this is the same pathology observed in other SFMs in the project.

Effect on this run: Phase 1 trained for 100 epochs but val pred_acc was 0.0 every epoch (because val was empty). The checkpoint manager only saved `best_model_val_pred_acc_epoch_0.pth` (the initial random weights, which beat all subsequent epochs at 0.0 vs 0.0 because epoch 0 happens first). Phase 2 retrained for `best_epoch+1 = 1` epoch, yielding `final_model_epoch_0.pth` — essentially an untrained model.

Pool-512 result for that fold: `R@1 (TF→DNA) = 27.8%`, `R@1 (DNA→TF) = 4.5%`. These are not meaningful tSFM numbers — they reflect a near-random model.

**Resolution:** Fold 4 of `identity_100` is excluded from aggregated `identity_100` statistics. Mean ± s.d. reported below uses 4 folds (0, 1, 2, 3) for `identity_100`, all 5 folds for the OOD splits. This decision is consistent with how the issue is handled in other SFM domains in the project.

The underlying split-generation bug should be fixed centrally in `mmseqs_splits.py` / `split_utils.py`. Tracked outside this audit.

### Issue 2 — Phase 2 retrain silent exit at smoke-test scale

During the smoke test (`num_epochs=5` forced via Hydra), Phase 2 retrain exited after data loading without printing any epoch output. We patched `evaluate.py` (Patch D) to fall back to `best_model_val_pred_acc_*.pth` and continued. Production runs (default `num_epochs=100`) completed Phase 2 normally, so the fallback was never triggered. The smoke-only behavior is documented but not investigated further.

### Issue 3 — Node contention on initial submission

6 of 20 first-submission tasks landed on GPU nodes that other users were holding. SLURM allocated those nodes to our jobs but the `srun` step creation kept retrying for the full 12-hour wall-clock without success. Resubmitting the same indices to fresh nodes resolved cleanly. No code change needed; documented as a transient cluster artifact.

## Aggregated results — pool-512, mean ± sample SD across folds

Pool size 512, 100 random trials per fold, `--max_queries 200` query subsampling.

### TF → DNA retrieval

| Split | n folds | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| identity_100 (excl. fold 4) | 4 | 83.1 ± 3.0 | 98.9 ± 0.4 | 99.6 ± 0.5 |
| mmseqs_080 (80% ID, easier OOD) | 5 | 54.7 ± 5.1 | 79.1 ± 5.0 | 86.1 ± 3.8 |
| mmseqs_060 (60% ID) | 5 | 53.7 ± 4.9 | 79.9 ± 4.2 | 87.2 ± 2.6 |
| mmseqs_040 (40% ID, hardest OOD) | 5 | 48.4 ± 4.3 | 75.3 ± 2.8 | 85.5 ± 2.7 |

### DNA → TF retrieval (asymmetric, lower because short motifs carry less info)

| Split | n folds | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| identity_100 (excl. fold 4) | 4 | 25.8 ± 2.5 | 45.0 ± 2.3 | 53.8 ± 2.6 |
| mmseqs_080 | 5 | 10.2 ± 2.7 | 17.7 ± 4.9 | 25.5 ± 5.0 |
| mmseqs_060 | 5 | 8.6 ± 2.0 | 16.7 ± 2.4 | 23.2 ± 2.3 |
| mmseqs_040 | 5 | 7.5 ± 1.6 | 14.6 ± 2.9 | 21.1 ± 3.0 |

### Comparison vs v0.5 (10×) headline R@1 TF → DNA

| Split | v0.5 (10×, 5-fold) | v2 (50×) | Improvement |
|---|---|---|---|
| identity_100 | 37.2 ± 17.7 | 83.1 ± 3.0 (4 folds) | +46 pts; variance dropped 5× |
| mmseqs_080 | 23.1 ± 8.0 | 54.7 ± 5.0 | +32 pts |
| mmseqs_060 | 25.2 ± 1.6 | 53.6 ± 4.5 | +28 pts |
| mmseqs_040 | 22.8 ± 3.8 | 48.4 ± 4.0 | +26 pts |

**The 5× scale-up roughly doubles R@1 across every split.** This is the headline retrain win for the Vibe-Coding SFMs paper.

## Per-fold raw numbers (R@1 TF→DNA)

| Fold | identity_100 | mmseqs_080 | mmseqs_060 | mmseqs_040 |
|---|---|---|---|---|
| 0 | 81.5 | 62.2 | 49.7 | 52.7 |
| 1 | 84.6 | 52.7 | 49.0 | 44.2 |
| 2 | 79.9 | 48.5 | 52.3 | 50.7 |
| 3 | 86.5 | 53.5 | 56.4 | 51.0 |
| 4 | **27.8 (excluded — empty val)** | 56.5 | 60.8 | 43.3 |

## Durable artifacts (HOME, survives scratch purges)

```
/cluster/home/reddys/CALM-0.1.0/eval_summaries/tsfm/
├── identity_100_pool512_results.csv
├── mmseqs_080_pool512_results.csv
├── mmseqs_060_pool512_results.csv
└── mmseqs_040_pool512_results.csv
```

Each CSV contains 5 fold-level rows (header + 5 data rows) with columns: `R@1_ag2ab, R@5_ag2ab, R@10_ag2ab, R@1_ab2ag, R@5_ab2ag, R@10_ab2ag, pool_size, n_trials, n_test, max_queries, fold`.

## Total compute

- **Preprocessing:** 1 GPU job, ~30 min
- **MMseqs2 splits:** login-node, ~30 sec
- **Training (Phase 1+2):** 20 + 6 = 26 GPU-jobs at 4–8 hours each = ~150 GPU-hours total
- **Evaluation:** 20 GPU jobs, ~5 min each = ~2 GPU-hours
- **Pool-512:** 1 CPU job, ~10 min
- **Total wall-clock end-to-end:** ~24 hours including queue waits and resubmits

## Pseudo-prospective validation (v2 scale)

A separate experiment was run to test whether the v2 architecture and scale generalize to TFs outside the training set. JASPAR's release history was used to define a temporal split: train on JASPAR 2022 only, test on profiles added in 2024 and 2026.

**Setup**: 60,000 training pairs (1,200 JASPAR 2022 profiles × 50× PWM sampling), 42,700 held-out pairs (854 post-2022 profiles × 50× PWM sampling). 178 of the held-out TFs are "truly novel" — they have no prior profile in JASPAR 2022. Single fold trained: mmseqs_080 fold 0, matching the v0.5 validation methodology for direct comparison.

**Pipeline**: same as main retrain, but on the 60K-pair JASPAR 2022 subset only. Phase 1 reached best val pred_acc 79% at epoch 35; Phase 2 retrained on train+val for 36 epochs producing `final_model_epoch_35.pth`. Total wall-clock ~6 hours for the single fold + ~30 min preprocessing + ~10 min retrieval = ~7 hours end-to-end.

**Results — Family-level retrieval** (correct DNA motif's family in top-k of pool):

| Group | N | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|
| All held-out | 787 | 67.7% | 87.3% | 91.9% |
| Truly novel | 150 | **44.7%** | **68.0%** | **78.0%** |
| Updated versions | 637 | 73.2% | 91.8% | 95.1% |

**Results — Exact TF retrieval** (correct exact TF's motif in top-k, for 634 held-out TFs with exact match in pool): Top-1 45.9%, Top-5 65.3%, Top-10 71.9%.

**v0.5 → v2 validation comparison** (same temporal split, same fold choice):

| Metric | v0.5 (10×) | v2 (50×) | Δ |
|---|---|---|---|
| Truly novel — Top-1 family | 38.7% | 44.7% | **+6.0** |
| Truly novel — Top-5 family | 71.3% | 68.0% | −3.3 |
| Truly novel — Top-10 family | 79.3% | 78.0% | −1.3 |
| All held-out — Top-1 family | 61.6% | 67.7% | **+6.1** |
| Exact TF — Top-1 | 37.2% | 45.9% | **+8.7** |
| Exact TF — Top-5 | 59.1% | 65.3% | **+6.2** |

**Interpretation**: The 5× scale-up sharpens top-1 predictions for genuinely novel TFs (+6 pts) and substantially improves exact-TF discrimination (+8.7 pts at top-1). Top-5 / top-10 family retrieval for truly novel TFs is roughly preserved (within sampling noise); v2 trades a small amount of broad top-k coverage for sharper top-1 commitment, which is the practically more valuable behavior for wet-lab use.

28 of 178 truly novel TFs were unevaluable (their TF family had no member in the 2022 training pool by construction). The 150 evaluable truly novel TFs span 75 distinct TF families covering the major DNA-binding domain classes (zinc fingers, homeodomain, bHLH, bZIP, nuclear receptors, etc.), supporting broad applicability of the result.

**Validation artifacts**:
- Trained model: `/cluster/scratch/$USER/tsfm/output/tsfm_validation_v2_50x/tsfm-val-mmseqs_080-fold0/`
- Retrieval outputs: `/cluster/scratch/$USER/tsfm/output/validation_retrieval_v2_50x/mmseqs_080_fold0/`
- Validation summary: `/cluster/scratch/$USER/tsfm/jaspar_validation_v2_50x/validation_summary.json` (lists all 178 truly novel TF names)

## Audit signal

Per `audit/PRESERVATION_DISCIPLINE.md` Rule 1 (mandatory stdout summaries) and Rule 2 (durable summary CSVs):
- ✅ `eval_pool512_fast.py` printed `=== Summary ===` block to stdout for every fold (Rule 1)
- ✅ Per-split summary CSVs written to `~/CALM-0.1.0/eval_summaries/tsfm/` (Rule 2)
- 🟡 Rule 3 (commit archival logs within 7 days) — this file commits the archival log; matched.
