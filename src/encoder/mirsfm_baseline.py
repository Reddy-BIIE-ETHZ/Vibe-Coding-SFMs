"""
Seed-matching baseline for miR-SFM evaluation.

Computes retrieval metrics using reverse-complement seed matching —
the naive baseline equivalent to Hamming distance for crisprSFM.

For each miRNA query, ranks all targets in the pool by:
  1. Whether the target contains the reverse complement of the miRNA seed (pos 2-8)
  2. Site type hierarchy: 8mer > 7mer-m8 > 7mer-A1 > 6mer > no match

This is essentially what TargetScan does. miR-SFM should beat this
baseline on non-canonical targets (where no seed match exists).

Usage:
    python -m calm.encoder.mirsfm_baseline \\
        --metadata /cluster/scratch/$USER/mirsfm/data/metadata.csv \\
        --split_file /cluster/scratch/$USER/mirsfm/data/split_index/identity_100/split_hash_ids_outerfold_0_innerfold_0.json \\
        --pool_size 512 \\
        --n_trials 100
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Seed matching logic
# ---------------------------------------------------------------------------

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    return "".join(COMPLEMENT.get(c, "N") for c in reversed(seq.upper()))


def get_seed(mirna_seq: str) -> str:
    """Extract the seed region (positions 2-8, 1-indexed) from a miRNA sequence.

    The seed is the primary determinant of target recognition.
    Returns 7 nucleotides (0-indexed positions 1-7).
    """
    if len(mirna_seq) < 8:
        return mirna_seq[1:] if len(mirna_seq) > 1 else mirna_seq
    return mirna_seq[1:8]


def classify_site(mirna_seq: str, target_seq: str) -> tuple[int, str]:
    """Classify the seed match type between miRNA and target site.

    Returns (score, site_type) where higher score = stronger match:
      4 = 8mer-A1   (seed match pos 2-8 + A opposite pos 1)
      3 = 7mer-m8   (seed match pos 2-8)
      2 = 7mer-A1   (seed match pos 2-7 + A opposite pos 1)
      1 = 6mer      (seed match pos 2-7)
      0 = no match
    """
    seed_7 = get_seed(mirna_seq)          # positions 2-8 (7nt)
    seed_6 = seed_7[:6]                    # positions 2-7 (6nt)
    rc_7 = reverse_complement(seed_7)      # target complement of full seed
    rc_6 = reverse_complement(seed_6)      # target complement of 6mer seed

    target_upper = target_seq.upper()

    # Check for 7mer seed match (positions 2-8)
    has_7mer = rc_7 in target_upper

    # Check for 6mer seed match (positions 2-7)
    has_6mer = rc_6 in target_upper

    if has_7mer:
        # Check for 8mer: 7mer + A opposite position 1 of miRNA
        # In the target, an "A" should appear just downstream of the seed match
        idx = target_upper.find(rc_7)
        if idx > 0 and target_upper[idx - 1] == "A":
            return 4, "8mer-A1"
        return 3, "7mer-m8"

    if has_6mer:
        # Check for 7mer-A1: 6mer + A
        idx = target_upper.find(rc_6)
        if idx > 0 and target_upper[idx - 1] == "A":
            return 2, "7mer-A1"
        return 1, "6mer"

    return 0, "no_match"


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


def seed_baseline_retrieval(
    metadata: pd.DataFrame,
    test_indices: list[int],
    pool_size: int = 512,
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    """Run seed-matching baseline retrieval.

    For each trial:
      1. Sample pool_size test pairs
      2. For each miRNA in the pool, score all targets by seed match type
      3. Rank targets by score
      4. Compute R@1, R@5, R@10 (does the true target rank in top k?)

    This is bidirectional: miRNA→target AND target→miRNA.

    Parameters
    ----------
    metadata : pd.DataFrame
        Full metadata with mirna_seq, target_seq columns.
    test_indices : list[int]
        Row indices for the test set.
    pool_size : int
        Number of pairs to sample per trial.
    n_trials : int
        Number of random trials.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Results with R@1, R@5, R@10 for both directions + breakdown by site type.
    """
    rng = random.Random(seed)
    test_data = metadata.iloc[test_indices].reset_index(drop=True)

    if len(test_data) < pool_size:
        print(f"  Warning: test set ({len(test_data)}) < pool_size ({pool_size}). "
              f"Using full test set.")
        pool_size = len(test_data)

    # Pre-compute all seed sequences for efficiency
    mirna_seqs = test_data["mirna_seq"].tolist()
    target_seqs = test_data["target_seq"].tolist()

    results_ag2ab = {1: [], 5: [], 10: []}  # miRNA → target
    results_ab2ag = {1: [], 5: [], 10: []}  # target → miRNA

    # Also track by canonical vs non-canonical
    canonical_flags = test_data["is_canonical"].astype(bool).tolist()
    results_canonical = {1: [], 5: [], 10: []}
    results_noncanonical = {1: [], 5: [], 10: []}

    for trial in range(n_trials):
        # Sample pool
        pool_idx = rng.sample(range(len(test_data)), pool_size)
        pool_mirna = [mirna_seqs[i] for i in pool_idx]
        pool_target = [target_seqs[i] for i in pool_idx]
        pool_canonical = [canonical_flags[i] for i in pool_idx]

        # --- Direction 1: miRNA → target ---
        for i in range(pool_size):
            query_mirna = pool_mirna[i]
            true_target_idx = i

            # Score all targets
            scores = []
            for j in range(pool_size):
                score, _ = classify_site(query_mirna, pool_target[j])
                scores.append((score, j))

            # Rank by score (descending), break ties randomly
            rng.shuffle(scores)  # randomize before stable sort for tie-breaking
            scores.sort(key=lambda x: -x[0])

            rank = next(k for k, (_, j) in enumerate(scores) if j == true_target_idx) + 1

            for k in [1, 5, 10]:
                hit = 1.0 if rank <= k else 0.0
                results_ag2ab[k].append(hit)
                if pool_canonical[i]:
                    results_canonical[k].append(hit)
                else:
                    results_noncanonical[k].append(hit)

        # --- Direction 2: target → miRNA ---
        for i in range(pool_size):
            query_target = pool_target[i]
            true_mirna_idx = i

            scores = []
            for j in range(pool_size):
                score, _ = classify_site(pool_mirna[j], query_target)
                scores.append((score, j))

            rng.shuffle(scores)
            scores.sort(key=lambda x: -x[0])

            rank = next(k for k, (_, j) in enumerate(scores) if j == true_mirna_idx) + 1

            for k in [1, 5, 10]:
                results_ab2ag[k].append(1.0 if rank <= k else 0.0)

        if (trial + 1) % 25 == 0:
            r1 = sum(results_ag2ab[1]) / len(results_ag2ab[1]) * 100
            print(f"  Trial {trial+1}/{n_trials}: miRNA→target R@1 = {r1:.1f}%")

    # Compute final metrics
    def mean_pct(vals):
        return sum(vals) / len(vals) * 100 if vals else 0.0

    output = {
        "pool_size": pool_size,
        "n_trials": n_trials,
        "n_test_pairs": len(test_data),
        "mirna_to_target": {
            "R@1": round(mean_pct(results_ag2ab[1]), 2),
            "R@5": round(mean_pct(results_ag2ab[5]), 2),
            "R@10": round(mean_pct(results_ag2ab[10]), 2),
        },
        "target_to_mirna": {
            "R@1": round(mean_pct(results_ab2ag[1]), 2),
            "R@5": round(mean_pct(results_ab2ag[5]), 2),
            "R@10": round(mean_pct(results_ab2ag[10]), 2),
        },
        "canonical_only": {
            "R@1": round(mean_pct(results_canonical[1]), 2),
            "R@5": round(mean_pct(results_canonical[5]), 2),
            "R@10": round(mean_pct(results_canonical[10]), 2),
            "n_queries": len(results_canonical[1]),
        },
        "noncanonical_only": {
            "R@1": round(mean_pct(results_noncanonical[1]), 2),
            "R@5": round(mean_pct(results_noncanonical[5]), 2),
            "R@10": round(mean_pct(results_noncanonical[10]), 2),
            "n_queries": len(results_noncanonical[1]),
        },
    }

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Seed-matching baseline for miR-SFM evaluation.",
    )
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--split_file", required=True, help="Path to split JSON file")
    parser.add_argument("--pool_size", type=int, default=512)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--output", default=None, help="Output JSON path (default: print to stdout)")
    args = parser.parse_args()

    print("=" * 60)
    print("Seed-Matching Baseline for miR-SFM")
    print("=" * 60)

    # Load metadata
    meta = pd.read_csv(args.metadata)
    print(f"Loaded {len(meta):,} pairs")

    # Load split
    with open(args.split_file) as f:
        split = json.load(f)

    test_hashes = set(split["test"])
    test_mask = meta["Unique_ag_vh_vl_hash"].isin(test_hashes)
    test_indices = meta.index[test_mask].tolist()
    print(f"Test set: {len(test_indices):,} pairs")

    # Count canonical vs non-canonical in test set
    test_data = meta.iloc[test_indices]
    n_can = test_data["is_canonical"].astype(bool).sum()
    n_nc = len(test_data) - n_can
    print(f"  Canonical: {n_can:,} ({n_can/len(test_data)*100:.1f}%)")
    print(f"  Non-canonical: {n_nc:,} ({n_nc/len(test_data)*100:.1f}%)")

    # Run baseline
    print(f"\nRunning seed-matching baseline (pool={args.pool_size}, trials={args.n_trials})...")
    results = seed_baseline_retrieval(
        meta, test_indices,
        pool_size=args.pool_size,
        n_trials=args.n_trials,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"SEED-MATCHING BASELINE RESULTS (pool={args.pool_size})")
    print(f"{'='*60}")
    print(f"\nmiRNA → target:")
    for k in ["R@1", "R@5", "R@10"]:
        print(f"  {k}: {results['mirna_to_target'][k]:.1f}%")
    print(f"\ntarget → miRNA:")
    for k in ["R@1", "R@5", "R@10"]:
        print(f"  {k}: {results['target_to_mirna'][k]:.1f}%")
    print(f"\nCanonical pairs only (miRNA → target):")
    for k in ["R@1", "R@5", "R@10"]:
        print(f"  {k}: {results['canonical_only'][k]:.1f}%")
    print(f"  ({results['canonical_only']['n_queries']:,} queries)")
    print(f"\nNon-canonical pairs only (miRNA → target):")
    for k in ["R@1", "R@5", "R@10"]:
        print(f"  {k}: {results['noncanonical_only'][k]:.1f}%")
    print(f"  ({results['noncanonical_only']['n_queries']:,} queries)")

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
