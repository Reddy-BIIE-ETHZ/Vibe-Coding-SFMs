"""Post-hoc evaluation at a fixed pool size (default 512).

All SFMs are evaluated with the same pool size (n=512) so results are
directly comparable across domains.

What this does in plain language:
    After training, the model produces a "similarity score" between every
    agent (e.g., transcription factor) and every target (e.g., DNA sequence).
    To test whether the model can find the correct partner, we give it a
    lineup of 512 candidates (1 correct + 511 wrong ones) and ask: "Which
    one is the correct binding partner?" We measure how often the correct
    answer appears in the top 1, top 5, or top 10 guesses. We do this in
    both directions (agent→target and target→agent).

    We repeat this 100 times with different random lineups to get stable
    numbers, then average across 5 cross-validation folds.

Usage (on Mac terminal or Euler):
    python -m calm.encoder.eval_pool512 \\
        --eval_dir output/tsfm/ffn/full/eval \\
        --pool_size 512 \\
        --n_trials 100

    This reads the saved similarity matrices from training and produces
    a CSV file with R@1, R@5, R@10 for the paper tables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def evaluate_pool(
    cosine_sim: torch.Tensor,
    labels_masks: dict[str, torch.Tensor],
    pool_size: int = 512,
    n_trials: int = 100,
    k_values: tuple[int, ...] = (1, 5, 10),
    seed: int = 42,
) -> dict[str, float]:
    """Compute retrieval metrics at a fixed pool size.

    For each query, we randomly sample (pool_size - 1) negatives plus
    the true positive(s), then check if the correct answer is in the
    top-k. We repeat this ``n_trials`` times for stability.

    Parameters
    ----------
    cosine_sim : torch.Tensor
        Full N×N similarity matrix from evaluate.py.
        Row i, column j = similarity between agent i and target j.
    labels_masks : dict
        Contains "ag" and "ab" keys, each a boolean N×N tensor.
        labels_masks["ag"][i,j] = True if target j is a correct
        match for agent i. "ab" is the transpose direction.
    pool_size : int
        Number of candidates in each retrieval pool.
    n_trials : int
        Number of random pool samplings per query.
    k_values : tuple of int
        Which top-k values to compute (e.g., 1, 5, 10).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        Keys like "R@1_ag2ab", "R@5_ab2ag", "cosine_sim_pos", etc.
    """
    rng = np.random.RandomState(seed)
    N = cosine_sim.shape[0]
    sim = cosine_sim.numpy() if isinstance(cosine_sim, torch.Tensor) else cosine_sim

    # If the test set is smaller than pool_size, use full test set
    effective_pool = min(pool_size, N)
    if effective_pool < pool_size:
        print(f"  Note: test set ({N}) < pool_size ({pool_size}), "
              f"using full test set as pool (effective pool = {N})")

    results = {}

    for direction, label_key in [("ag2ab", "ag"), ("ab2ag", "ab")]:
        # ag2ab: rows are queries (agents), columns are candidates (targets)
        # ab2ag: columns are queries (targets), rows are candidates (agents)
        if direction == "ag2ab":
            sim_matrix = sim  # (N_agents × N_targets)
            pos_mask = labels_masks[label_key].numpy()
        else:
            sim_matrix = sim.T  # transpose: now rows = targets querying agents
            pos_mask = labels_masks[label_key].numpy().T

        n_queries = sim_matrix.shape[0]
        n_candidates = sim_matrix.shape[1]

        # Collect per-trial recall values
        trial_recalls = {k: [] for k in k_values}
        trial_cosim_pos = []

        for _trial in range(n_trials):
            hits_at_k = {k: 0 for k in k_values}
            cosim_pos_sum = 0.0
            n_valid_queries = 0

            for qi in range(n_queries):
                # Find positive indices for this query
                pos_indices = np.where(pos_mask[qi])[0]
                if len(pos_indices) == 0:
                    continue

                # Find negative indices
                neg_indices = np.where(~pos_mask[qi])[0]
                if len(neg_indices) == 0:
                    continue

                # Sample negatives to fill pool
                n_neg_needed = effective_pool - len(pos_indices)
                if n_neg_needed <= 0:
                    # More positives than pool size — rare edge case
                    n_neg_needed = 1

                if len(neg_indices) >= n_neg_needed:
                    sampled_neg = rng.choice(neg_indices, size=n_neg_needed, replace=False)
                else:
                    # Not enough negatives — use all of them
                    sampled_neg = neg_indices

                # Build pool: positives + sampled negatives
                pool_indices = np.concatenate([pos_indices, sampled_neg])
                pool_sims = sim_matrix[qi, pool_indices]

                # Rank by similarity (descending)
                ranked = np.argsort(-pool_sims)

                # Check which ranked positions are positives
                is_positive = np.zeros(len(pool_indices), dtype=bool)
                is_positive[:len(pos_indices)] = True
                # After ranking, check if any positive is in top-k
                for k in k_values:
                    if np.any(is_positive[ranked[:k]]):
                        hits_at_k[k] += 1

                # Mean cosine similarity for positives
                cosim_pos_sum += sim_matrix[qi, pos_indices].mean()
                n_valid_queries += 1

            if n_valid_queries > 0:
                for k in k_values:
                    trial_recalls[k].append(hits_at_k[k] / n_valid_queries * 100.0)
                trial_cosim_pos.append(cosim_pos_sum / n_valid_queries)

        # Average across trials
        for k in k_values:
            vals = trial_recalls[k]
            results[f"R@{k}_{direction}"] = float(np.mean(vals)) if vals else 0.0
            results[f"R@{k}_{direction}_std"] = float(np.std(vals)) if vals else 0.0

        results[f"cosine_sim_pos_{direction}"] = float(np.mean(trial_cosim_pos)) if trial_cosim_pos else 0.0

    # Average cosine similarity (mean of both directions)
    results["cosine_sim_pos"] = (
        results.get("cosine_sim_pos_ag2ab", 0) + results.get("cosine_sim_pos_ab2ag", 0)
    ) / 2.0

    results["pool_size"] = effective_pool
    results["n_trials"] = n_trials
    results["n_test"] = N

    return results


def evaluate_all_folds(
    eval_dir: str,
    pool_size: int = 512,
    n_trials: int = 100,
    k_values: tuple[int, ...] = (1, 5, 10),
    seed: int = 42,
) -> list[dict[str, float]]:
    """Run pool-512 evaluation across all folds found in eval_dir.

    Looks for fold directories (fold_0, fold_1, ...) containing
    cosine_sim.pt and labels_masks.pt saved by evaluate.py.

    Parameters
    ----------
    eval_dir : str
        Path to the evaluation output directory (contains fold_* subdirs).
    pool_size : int
        Pool size for retrieval evaluation.
    n_trials : int
        Number of random samplings per query.
    k_values : tuple of int
        Top-k values to compute.
    seed : int
        Random seed.

    Returns
    -------
    list of dict
        One dict per fold with all metrics.
    """
    eval_path = Path(eval_dir)
    fold_dirs = sorted(eval_path.glob("fold_*"))

    if not fold_dirs:
        # Maybe the eval outputs are directly in eval_dir (single fold)
        if (eval_path / "cosine_sim.pt").exists():
            fold_dirs = [eval_path]
        else:
            raise FileNotFoundError(
                f"No fold directories or cosine_sim.pt found in {eval_dir}"
            )

    all_results = []
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        cosine_sim_path = fold_dir / "cosine_sim.pt"
        labels_path = fold_dir / "labels_masks.pt"

        if not cosine_sim_path.exists():
            print(f"  Skipping {fold_name}: cosine_sim.pt not found")
            continue
        if not labels_path.exists():
            print(f"  Skipping {fold_name}: labels_masks.pt not found")
            continue

        print(f"  Evaluating {fold_name}...")
        cosine_sim = torch.load(cosine_sim_path, weights_only=True)
        labels_masks = torch.load(labels_path, weights_only=True)

        fold_results = evaluate_pool(
            cosine_sim, labels_masks,
            pool_size=pool_size, n_trials=n_trials,
            k_values=k_values, seed=seed,
        )
        fold_results["fold"] = fold_name
        all_results.append(fold_results)

    return all_results


def summarize_folds(all_results: list[dict[str, float]]) -> dict[str, str]:
    """Compute mean ± s.d. across folds for the paper tables.

    Parameters
    ----------
    all_results : list of dict
        Output from ``evaluate_all_folds()``.

    Returns
    -------
    dict[str, str]
        Formatted strings like "34.9 ± 2.1" for each metric.
    """
    if not all_results:
        return {}

    # Collect numeric keys (skip 'fold')
    metric_keys = [k for k in all_results[0] if k != "fold" and not k.endswith("_std")]

    summary = {}
    for key in metric_keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            if key.startswith("R@"):
                summary[key] = f"{mean:.1f} ± {std:.1f}"
            elif "cosine" in key:
                summary[key] = f"{mean:.3f} ± {std:.3f}"
            else:
                summary[key] = f"{mean:.1f}"

    return summary


def save_results_csv(
    all_results: list[dict[str, float]],
    summary: dict[str, str],
    output_path: str,
) -> None:
    """Save per-fold results and summary to CSV.

    Parameters
    ----------
    all_results : list of dict
        Per-fold results.
    summary : dict
        Mean ± s.d. summary.
    output_path : str
        Where to save the CSV.
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        if all_results:
            keys = [k for k in all_results[0] if not k.endswith("_std")]
            writer.writerow(keys)

            # Per-fold rows
            for r in all_results:
                writer.writerow([r.get(k, "") for k in keys])

            # Summary row
            writer.writerow([])
            writer.writerow(["SUMMARY (mean ± s.d.)"])
            for key, val in summary.items():
                writer.writerow([key, val])

    print(f"  Results saved to {output_path}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval at fixed pool size (default 512)."
    )
    parser.add_argument(
        "--eval_dir", required=True,
        help="Path to evaluation output directory (contains fold_* subdirs "
             "with cosine_sim.pt and labels_masks.pt)."
    )
    parser.add_argument(
        "--pool_size", type=int, default=512,
        help="Number of candidates in retrieval pool (default: 512)."
    )
    parser.add_argument(
        "--n_trials", type=int, default=100,
        help="Number of random pool samplings per query (default: 100)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path. Default: {eval_dir}/pool512_results.csv"
    )
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
    args = parser.parse_args()

    # Output path resolution (Rule 2 of audit/PRESERVATION_DISCIPLINE.md):
    # Priority 1: explicit --output flag
    # Priority 2: SFM_SUMMARY_DIR env + --sfm_name + --split_name (durable, in home)
    # Priority 3: default to eval_dir (legacy, on scratch)
    if args.output:
        output_path = args.output
    elif os.environ.get("SFM_SUMMARY_DIR") and args.sfm_name and args.split_name:
        summary_dir = Path(os.environ["SFM_SUMMARY_DIR"]) / args.sfm_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(summary_dir / f"{args.split_name}_pool512_results.csv")
        print(f"  Durable summary path: {output_path}")
    else:
        output_path = os.path.join(args.eval_dir, "pool512_results.csv")

    print(f"Pool-512 evaluation")
    print(f"  eval_dir:  {args.eval_dir}")
    print(f"  pool_size: {args.pool_size}")
    print(f"  n_trials:  {args.n_trials}")
    print()

    all_results = evaluate_all_folds(
        args.eval_dir,
        pool_size=args.pool_size,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    summary = summarize_folds(all_results)

    print()
    print("=== Summary (mean ± s.d. across folds) ===")
    for key, val in summary.items():
        print(f"  {key}: {val}")

    save_results_csv(all_results, summary, output_path)


if __name__ == "__main__":
    main()
