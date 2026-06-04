"""Compute pool-invariant ranking metrics (MRR, median rank, fold-above-random)
from a pre-existing cosine_sim.pt + labels_masks.pt.

Pool-invariant metrics are necessary for fair cross-version comparison when
the eval set sizes differ (e.g., comparing mhcSFM v1 R@1 on 33K val pairs
vs v2 R@1 on 695K or 5K): R@K alone is misleading across pool sizes.

Metrics computed per direction (ag2ab, ab2ag):
  - MRR             mean(1 / rank_of_first_positive)
  - median_rank     median rank of first positive (lower = better)
  - mean_rank       mean rank of first positive
  - fold_above_random   median_rank / random_baseline
                          (random_baseline = (N+1) / (n_pos+1))

Usage:
    python -m calm.encoder.compute_mrr --eval_dir /path/to/eval/fold_0
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch


def compute_mrr_and_ranks(
    cosine_sim: torch.Tensor,
    labels_masks: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute pool-invariant ranking metrics for both retrieval directions."""
    sim = cosine_sim.numpy() if isinstance(cosine_sim, torch.Tensor) else cosine_sim
    N = sim.shape[0]

    results = {"n_test": N}

    for direction, label_key in [("ag2ab", "ag"), ("ab2ag", "ab")]:
        if direction == "ag2ab":
            sim_matrix = sim
            pos_mask = labels_masks[label_key].numpy()
        else:
            sim_matrix = sim.T
            pos_mask = labels_masks[label_key].numpy().T

        n_total_queries = sim_matrix.shape[0]

        first_pos_ranks = []  # 1-based rank of first positive per query
        n_pos_per_query = []
        random_baseline_ranks = []  # expected rank under random ordering

        for qi in range(n_total_queries):
            pos_idx = np.nonzero(pos_mask[qi])[0]
            if len(pos_idx) == 0:
                continue

            # Rank ALL candidates by sim (descending). 1-based ranks.
            order = np.argsort(-sim_matrix[qi])
            ranks = np.empty(n_total_queries, dtype=np.int64)
            ranks[order] = np.arange(1, n_total_queries + 1)

            first_pos_rank = int(ranks[pos_idx].min())
            first_pos_ranks.append(first_pos_rank)
            n_pos_per_query.append(len(pos_idx))

            # Expected first-positive rank under random ordering:
            #   E[min rank of n_pos uniform draws from N candidates] = (N+1) / (n_pos+1)
            n_pos = len(pos_idx)
            random_baseline_ranks.append((n_total_queries + 1) / (n_pos + 1))

        if not first_pos_ranks:
            print(f"  WARNING: no valid queries for {direction}")
            continue

        first_pos_ranks = np.array(first_pos_ranks)
        random_baseline_ranks = np.array(random_baseline_ranks)
        n_pos_per_query = np.array(n_pos_per_query)

        mrr = float(np.mean(1.0 / first_pos_ranks))
        median_rank = float(np.median(first_pos_ranks))
        mean_rank = float(np.mean(first_pos_ranks))
        median_random_baseline = float(np.median(random_baseline_ranks))
        fold_above_random = median_random_baseline / median_rank

        results[f"MRR_{direction}"] = mrr
        results[f"median_rank_{direction}"] = median_rank
        results[f"mean_rank_{direction}"] = mean_rank
        results[f"random_baseline_median_{direction}"] = median_random_baseline
        results[f"fold_above_random_{direction}"] = fold_above_random
        results[f"n_queries_{direction}"] = len(first_pos_ranks)
        results[f"mean_n_pos_{direction}"] = float(np.mean(n_pos_per_query))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool-invariant ranking metrics from cosine_sim.pt + labels_masks.pt"
    )
    parser.add_argument("--eval_dir", required=True,
                        help="Path containing cosine_sim.pt + labels_masks.pt")
    parser.add_argument("--label", default="",
                        help="Optional run label printed in output (e.g. 'v2_BA+EL_epoch8')")
    parser.add_argument("--output", default=None,
                        help="Optional CSV output path (defaults to eval_dir/mrr_results.csv)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    cosine_path = eval_dir / "cosine_sim.pt"
    labels_path = eval_dir / "labels_masks.pt"

    if not cosine_path.exists():
        raise FileNotFoundError(f"cosine_sim.pt not found in {eval_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels_masks.pt not found in {eval_dir}")

    print(f"=== Pool-invariant ranking metrics ===")
    if args.label:
        print(f"  Label:     {args.label}")
    print(f"  Eval dir:  {eval_dir}")
    print(f"  Loading cosine_sim.pt ({cosine_path.stat().st_size / 1e6:.1f} MB)...")

    cosine_sim = torch.load(cosine_path, weights_only=True, map_location="cpu")
    labels_masks = torch.load(labels_path, weights_only=True, map_location="cpu")

    print(f"  cosine_sim shape: {tuple(cosine_sim.shape)}")
    print()

    results = compute_mrr_and_ranks(cosine_sim, labels_masks)

    print(f"  N_test (eval pool size): {results['n_test']}")
    print()
    for direction, label in [("ag2ab", "peptide -> HLA"), ("ab2ag", "HLA -> peptide")]:
        if f"MRR_{direction}" not in results:
            continue
        print(f"  --- {direction} ({label}) ---")
        print(f"    n_queries:           {results[f'n_queries_{direction}']}")
        print(f"    mean_n_pos:          {results[f'mean_n_pos_{direction}']:.2f}")
        print(f"    MRR:                 {results[f'MRR_{direction}']:.4f}")
        print(f"    median_rank:         {results[f'median_rank_{direction}']:.1f}")
        print(f"    mean_rank:           {results[f'mean_rank_{direction}']:.1f}")
        print(f"    random_baseline:     {results[f'random_baseline_median_{direction}']:.1f}")
        print(f"    fold_above_random:   {results[f'fold_above_random_{direction}']:.1f}x")
        print()

    output_path = Path(args.output) if args.output else eval_dir / "mrr_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        if args.label:
            writer.writerow(["label", args.label])
        for k, v in results.items():
            writer.writerow([k, v])
    print(f"  Results saved: {output_path}")


if __name__ == "__main__":
    main()
