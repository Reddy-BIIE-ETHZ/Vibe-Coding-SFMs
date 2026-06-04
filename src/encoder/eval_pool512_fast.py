"""Fast pool-512 evaluation for large test sets.

The original eval_pool512.py is O(n_queries * n_trials) with expensive
np.where calls per iteration. For crisprSFM with ~6,000 test samples,
this takes hours. This version:
  1. Precomputes positive/negative indices once per query
  2. Subsamples queries (default 200) for tractability
  3. Vectorizes the ranking step

Usage:
    python -m calm.encoder.eval_pool512_fast \
        --eval_dir /path/to/eval \
        --pool_size 512 \
        --n_trials 100 \
        --max_queries 200
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch


def evaluate_pool_fast(
    cosine_sim: torch.Tensor,
    labels_masks: dict[str, torch.Tensor],
    pool_size: int = 512,
    n_trials: int = 100,
    max_queries: int = 200,
    k_values: tuple[int, ...] = (1, 5, 10),
    seed: int = 42,
) -> dict[str, float]:
    """Fast pool-512 evaluation with query subsampling."""
    rng = np.random.RandomState(seed)
    sim = cosine_sim.numpy() if isinstance(cosine_sim, torch.Tensor) else cosine_sim
    N = sim.shape[0]
    effective_pool = min(pool_size, N)

    if effective_pool < pool_size:
        print(f"  Note: test set ({N}) < pool_size ({pool_size}), "
              f"using effective pool = {N}")

    results = {}

    for direction, label_key in [("ag2ab", "ag"), ("ab2ag", "ab")]:
        if direction == "ag2ab":
            sim_matrix = sim
            pos_mask = labels_masks[label_key].numpy()
        else:
            sim_matrix = sim.T
            pos_mask = labels_masks[label_key].numpy().T

        n_total_queries = sim_matrix.shape[0]

        # Precompute positive and negative indices for all queries
        print(f"    {direction}: precomputing indices for {n_total_queries} queries...")
        query_pos = {}
        query_neg = {}
        valid_queries = []
        for qi in range(n_total_queries):
            pos_idx = np.nonzero(pos_mask[qi])[0]
            if len(pos_idx) == 0:
                continue
            neg_idx = np.nonzero(~pos_mask[qi])[0]
            if len(neg_idx) == 0:
                continue
            query_pos[qi] = pos_idx
            query_neg[qi] = neg_idx
            valid_queries.append(qi)

        # Subsample queries if too many
        if len(valid_queries) > max_queries:
            valid_queries = sorted(rng.choice(valid_queries, size=max_queries, replace=False))
            print(f"    Subsampled to {max_queries} queries (from {n_total_queries})")
        else:
            print(f"    Using all {len(valid_queries)} valid queries")

        # Run trials
        trial_recalls = {k: [] for k in k_values}

        for trial in range(n_trials):
            hits_at_k = {k: 0 for k in k_values}

            for qi in valid_queries:
                pos_idx = query_pos[qi]
                neg_idx = query_neg[qi]

                n_neg_needed = effective_pool - len(pos_idx)
                if n_neg_needed <= 0:
                    n_neg_needed = 1

                if len(neg_idx) >= n_neg_needed:
                    sampled_neg = rng.choice(neg_idx, size=n_neg_needed, replace=False)
                else:
                    sampled_neg = neg_idx

                pool_indices = np.concatenate([pos_idx, sampled_neg])
                pool_sims = sim_matrix[qi, pool_indices]
                ranked = np.argsort(-pool_sims)

                # Positions 0..len(pos_idx)-1 are positives
                n_pos = len(pos_idx)
                for k in k_values:
                    if np.any(ranked[:k] < n_pos):
                        hits_at_k[k] += 1

            n_q = len(valid_queries)
            for k in k_values:
                trial_recalls[k].append(hits_at_k[k] / n_q * 100.0)

            if (trial + 1) % 25 == 0:
                print(f"    {direction}: trial {trial+1}/{n_trials}, "
                      f"R@1={trial_recalls[1][-1]:.1f}%")

        for k in k_values:
            vals = trial_recalls[k]
            results[f"R@{k}_{direction}"] = float(np.mean(vals))
            results[f"R@{k}_{direction}_std"] = float(np.std(vals))

    results["pool_size"] = effective_pool
    results["n_trials"] = n_trials
    results["n_test"] = N
    results["max_queries"] = max_queries

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast pool-512 evaluation for large test sets."
    )
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--pool_size", type=int, default=512)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--max_queries", type=int, default=200)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--sfm_name", default=None,
        help="SFM name (e.g. 'tsfm'). Used in durable summary path with SFM_SUMMARY_DIR env var."
    )
    parser.add_argument(
        "--split_name", default=None,
        help="Split method (e.g. 'identity_100'). Used in durable summary path."
    )
    # Domain-aware retrieval-direction labels (Phase 0.6, mhcSFM v2 audit cleanup).
    # Defaults preserve the v1 crisprSFM-origin gRNA<->OT labels so existing SFM
    # runs produce identical log output. Future SFMs override these, e.g.
    #   mhcSFM v2: --ag2ab_label "peptide->HLA" --ab2ag_label "HLA->peptide"
    #   tSFM:      --ag2ab_label "TF->DNA"      --ab2ag_label "DNA->TF"
    parser.add_argument(
        "--ag2ab_label", default="gRNA->OT",
        help="Label for the agent->target retrieval direction in printed summaries."
    )
    parser.add_argument(
        "--ab2ag_label", default="OT->gRNA",
        help="Label for the target->agent retrieval direction in printed summaries."
    )
    args = parser.parse_args()

    # Output path resolution (Rule 2 of audit/PRESERVATION_DISCIPLINE.md):
    # Priority 1: explicit --output flag
    # Priority 2: SFM_SUMMARY_DIR env + --sfm_name + --split_name (durable, in home)
    # Priority 3: default to eval_dir (legacy behavior, on scratch — gets purged)
    if args.output:
        output_path = args.output
    elif os.environ.get("SFM_SUMMARY_DIR") and args.sfm_name and args.split_name:
        summary_dir = Path(os.environ["SFM_SUMMARY_DIR"]) / args.sfm_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(summary_dir / f"{args.split_name}_pool512_results.csv")
        print(f"  Durable summary path: {output_path}")
    else:
        output_path = os.path.join(args.eval_dir, "pool512_results.csv")

    print(f"Fast Pool-512 evaluation")
    print(f"  eval_dir:    {args.eval_dir}")
    print(f"  pool_size:   {args.pool_size}")
    print(f"  n_trials:    {args.n_trials}")
    print(f"  max_queries: {args.max_queries}")
    print()

    eval_path = Path(args.eval_dir)
    fold_dirs = sorted(eval_path.glob("fold_*"))

    if not fold_dirs:
        if (eval_path / "cosine_sim.pt").exists():
            fold_dirs = [eval_path]
        else:
            raise FileNotFoundError(f"No fold dirs or cosine_sim.pt in {args.eval_dir}")

    all_results = []
    for fold_dir in fold_dirs:
        cosine_sim_path = fold_dir / "cosine_sim.pt"
        labels_path = fold_dir / "labels_masks.pt"

        if not cosine_sim_path.exists() or not labels_path.exists():
            print(f"  Skipping {fold_dir.name}: missing files")
            continue

        print(f"  === {fold_dir.name} ===")
        print(f"  Loading cosine_sim.pt ({cosine_sim_path.stat().st_size / 1e9:.1f} GB)...")
        cosine_sim = torch.load(cosine_sim_path, weights_only=True, map_location="cpu")
        labels_masks = torch.load(labels_path, weights_only=True, map_location="cpu")

        fold_results = evaluate_pool_fast(
            cosine_sim, labels_masks,
            pool_size=args.pool_size, n_trials=args.n_trials,
            max_queries=args.max_queries, k_values=(1, 5, 10),
            seed=args.seed,
        )
        fold_results["fold"] = fold_dir.name
        all_results.append(fold_results)

        # Print fold results immediately (labels configurable via CLI flags;
        # defaults match earlier behavior for back-compat).
        print(f"  R@1  {args.ag2ab_label}: {fold_results['R@1_ag2ab']:.1f}%  "
              f"{args.ab2ag_label}: {fold_results['R@1_ab2ag']:.1f}%")
        print(f"  R@5  {args.ag2ab_label}: {fold_results['R@5_ag2ab']:.1f}%  "
              f"{args.ab2ag_label}: {fold_results['R@5_ab2ag']:.1f}%")
        print(f"  R@10 {args.ag2ab_label}: {fold_results['R@10_ag2ab']:.1f}%  "
              f"{args.ab2ag_label}: {fold_results['R@10_ab2ag']:.1f}%")
        print()

        # Free memory
        del cosine_sim, labels_masks

    # Summary across folds
    if all_results:
        print("=== Summary (mean ± s.d. across folds) ===")
        for metric in ["R@1_ag2ab", "R@1_ab2ag", "R@5_ag2ab", "R@5_ab2ag",
                        "R@10_ag2ab", "R@10_ab2ag"]:
            vals = [r[metric] for r in all_results]
            print(f"  {metric}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")

        # Save CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            keys = [k for k in all_results[0] if not k.endswith("_std")]
            writer.writerow(keys)
            for r in all_results:
                writer.writerow([r.get(k, "") for k in keys])
            writer.writerow([])
            writer.writerow(["SUMMARY"])
            for metric in ["R@1_ag2ab", "R@1_ab2ag", "R@5_ag2ab", "R@5_ab2ag",
                            "R@10_ag2ab", "R@10_ab2ag"]:
                vals = [r[metric] for r in all_results]
                writer.writerow([metric, f"{np.mean(vals):.1f} ± {np.std(vals):.1f}"])
        print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
