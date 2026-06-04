#!/bin/bash -l
# ==============================================================================
# SLURM job: miR-SFM pool-512 evaluation (fast version for large test sets)
#
# For miR-SFM's ~41K test pairs, we subsample 1000 queries per trial
# instead of iterating over all 41K. This gives stable metrics in minutes
# instead of hours.
#
# Also runs the seed-matching baseline for comparison.
#
# Usage:  sbatch scripts/euler_pool512_mirsfm_fast.slurm
# ==============================================================================

#SBATCH --job-name=mirsfm-p512
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mirsfm_pool512_%j.out
#SBATCH --error=logs/mirsfm_pool512_%j.err

CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/mirsfm/data"
ROOT_DIR="/cluster/scratch/${USER}/mirsfm/output"

module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CALM_DIR}/src:${PYTHONPATH}"

mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "miR-SFM Pool-512 Evaluation (Fast)"
echo "  Start time: $(date)"
echo "============================================"

cd ${CALM_DIR}

python3 << 'PYEOF'
import os, json, sys
import numpy as np
import torch
from pathlib import Path

def pool512_fast(cosine_sim, labels_masks, pool_size=512, n_trials=100,
                 n_queries_per_trial=500, seed=42):
    """Fast pool-512 eval: subsample queries for large test sets."""
    rng = np.random.RandomState(seed)
    sim = cosine_sim.numpy() if isinstance(cosine_sim, torch.Tensor) else cosine_sim
    N = sim.shape[0]
    effective_pool = min(pool_size, N)

    results = {}
    for direction, label_key in [("ag2ab", "ag"), ("ab2ag", "ab")]:
        if direction == "ag2ab":
            sim_matrix = sim
            pos_mask = labels_masks[label_key].numpy()
        else:
            sim_matrix = sim.T
            pos_mask = labels_masks[label_key].numpy().T

        n_total = sim_matrix.shape[0]
        n_q = min(n_queries_per_trial, n_total)

        trial_recalls = {1: [], 5: [], 10: []}
        for trial in range(n_trials):
            query_indices = rng.choice(n_total, size=n_q, replace=False)
            hits = {1: 0, 5: 0, 10: 0}
            valid = 0
            for qi in query_indices:
                pos_idx = np.where(pos_mask[qi])[0]
                if len(pos_idx) == 0:
                    continue
                neg_idx = np.where(~pos_mask[qi])[0]
                if len(neg_idx) == 0:
                    continue
                n_neg = effective_pool - len(pos_idx)
                if n_neg <= 0:
                    n_neg = 1
                if len(neg_idx) >= n_neg:
                    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
                else:
                    sampled_neg = neg_idx
                pool_idx = np.concatenate([pos_idx, sampled_neg])
                pool_sims = sim_matrix[qi, pool_idx]
                ranked = np.argsort(-pool_sims)
                is_pos = np.zeros(len(pool_idx), dtype=bool)
                is_pos[:len(pos_idx)] = True
                for k in [1, 5, 10]:
                    if np.any(is_pos[ranked[:k]]):
                        hits[k] += 1
                valid += 1
            if valid > 0:
                for k in [1, 5, 10]:
                    trial_recalls[k].append(hits[k] / valid * 100.0)
            if (trial + 1) % 25 == 0:
                r1 = np.mean(trial_recalls[1]) if trial_recalls[1] else 0
                print(f"    {direction} trial {trial+1}/{n_trials}: R@1={r1:.1f}%")

        for k in [1, 5, 10]:
            vals = trial_recalls[k]
            results[f"R@{k}_{direction}"] = float(np.mean(vals)) if vals else 0.0
            results[f"R@{k}_{direction}_std"] = float(np.std(vals)) if vals else 0.0

    results["pool_size"] = effective_pool
    results["n_trials"] = n_trials
    results["n_test"] = N
    return results


# Run for both splits
for split in ["identity_100", "mmseqs_080"]:
    eval_dir = Path(f"/cluster/scratch/{os.environ['USER']}/mirsfm/output/{split}/mirsfm-{split}/eval")
    print(f"\n{'='*60}")
    print(f"Pool-512: {split}")
    print(f"{'='*60}")

    all_fold_results = []
    fold_dirs = sorted(eval_dir.glob("fold_*"))
    for fold_dir in fold_dirs:
        cosine_path = fold_dir / "cosine_sim.pt"
        labels_path = fold_dir / "labels_masks.pt"
        if not cosine_path.exists():
            print(f"  Skipping {fold_dir.name}: no cosine_sim.pt")
            continue

        print(f"\n  {fold_dir.name}: loading cosine_sim.pt...")
        cosine_sim = torch.load(cosine_path, weights_only=True, map_location="cpu")
        labels_masks = torch.load(labels_path, weights_only=True, map_location="cpu")
        print(f"    Shape: {cosine_sim.shape}")

        r = pool512_fast(cosine_sim, labels_masks, pool_size=512,
                        n_trials=100, n_queries_per_trial=500, seed=42)
        r["fold"] = fold_dir.name
        all_fold_results.append(r)

        print(f"    R@1 ag→ab: {r['R@1_ag2ab']:.1f}%, R@1 ab→ag: {r['R@1_ab2ag']:.1f}%")
        print(f"    R@5 ag→ab: {r['R@5_ag2ab']:.1f}%, R@5 ab→ag: {r['R@5_ab2ag']:.1f}%")
        print(f"    R@10 ag→ab: {r['R@10_ag2ab']:.1f}%, R@10 ab→ag: {r['R@10_ab2ag']:.1f}%")

        # Free memory
        del cosine_sim, labels_masks

    if all_fold_results:
        print(f"\n  === {split} SUMMARY (mean ± s.d.) ===")
        for metric in ["R@1_ag2ab", "R@5_ag2ab", "R@10_ag2ab", "R@1_ab2ag", "R@5_ab2ag", "R@10_ab2ag"]:
            vals = [r[metric] for r in all_fold_results]
            print(f"    {metric}: {np.mean(vals):.1f} ± {np.std(vals):.1f}%")

        # Save
        out_path = f"/cluster/scratch/{os.environ['USER']}/mirsfm/output/{split}/pool512_results.json"
        with open(out_path, "w") as f:
            json.dump({"per_fold": all_fold_results,
                       "summary": {m: {"mean": float(np.mean([r[m] for r in all_fold_results])),
                                       "std": float(np.std([r[m] for r in all_fold_results]))}
                                   for m in ["R@1_ag2ab","R@5_ag2ab","R@10_ag2ab","R@1_ab2ag","R@5_ab2ag","R@10_ab2ag"]}
                       }, f, indent=2)
        print(f"  Saved: {out_path}")


# --- Seed-matching baseline ---
print(f"\n{'='*60}")
print("SEED-MATCHING BASELINE")
print(f"{'='*60}")

sys.path.insert(0, "src")
from calm.encoder.mirsfm_baseline import seed_baseline_retrieval
import pandas as pd

meta = pd.read_csv(f"/cluster/scratch/{os.environ['USER']}/mirsfm/data/metadata.csv")

for split in ["identity_100", "mmseqs_080"]:
    print(f"\n=== Baseline: {split} ===")
    for fold in range(5):
        split_file = f"/cluster/scratch/{os.environ['USER']}/mirsfm/data/split_index/{split}/split_hash_ids_outerfold_{fold}_innerfold_0.json"
        if not os.path.exists(split_file):
            continue
        with open(split_file) as f:
            sp = json.load(f)
        test_hashes = set(sp["test"])
        test_idx = meta.index[meta["Unique_ag_vh_vl_hash"].isin(test_hashes)].tolist()
        print(f"  fold {fold}: {len(test_idx)} test pairs")

        results = seed_baseline_retrieval(meta, test_idx, pool_size=512, n_trials=100, seed=42)
        print(f"    miRNA→target: R@1={results['mirna_to_target']['R@1']:.1f}%, "
              f"R@5={results['mirna_to_target']['R@5']:.1f}%, "
              f"R@10={results['mirna_to_target']['R@10']:.1f}%")
        print(f"    Canonical:     R@1={results['canonical_only']['R@1']:.1f}%")
        print(f"    Non-canonical: R@1={results['noncanonical_only']['R@1']:.1f}%")

        out_path = f"/cluster/scratch/{os.environ['USER']}/mirsfm/output/{split}/baseline_fold{fold}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        break  # Run baseline for fold 0 only (it's the same splits anyway)

print(f"\n{'='*60}")
print("DONE!")
print(f"{'='*60}")
PYEOF

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "============================================"
