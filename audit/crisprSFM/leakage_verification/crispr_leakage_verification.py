#!/usr/bin/env python3
"""
crisprSFM Leakage Verification (LV)

Production-scale audit of whether the hamming_XXX splits actually deliver
the strict-OOD property implied by their thresholds.

For each (threshold, outer_fold), this script:
  1. Loads the regenerated split_index_lv JSON (inner fold 0 is used as
     canonical since folds 0-4 differ only in val set composition)
  2. Maps hashes back to unique guide sequences via metadata.csv
  3. Computes min Hamming distance from each unique test guide to every
     unique train+val guide
  4. Reports leakage: fraction of test guides that have a train+val guide
     within the threshold's Hamming bound (4/9/13 mismatches for 80/60/40%)

Leakage interpretation:
  - 0% leakage = strict-OOD achieved; every test guide is beyond threshold
  - Non-zero leakage = some test guides have training neighbors within
    threshold; algorithm did NOT deliver promised OOD separation

Usage:
    python crispr_leakage_verification.py \
        --metadata /cluster/home/reddys/CALM-0.1.0/data/crispr/metadata.csv \
        --split_dir /cluster/scratch/reddys/crispr/data/split_index_lv \
        --output_dir /cluster/home/reddys/CALM-0.1.0/audit/leakage_verification/crisprsfm

Environment: requires pandas and numpy. Tested with calm_env.
Runtime: ~1 minute total across all 12 (threshold, fold) combinations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Thresholds and their max-mismatch bounds for 22-23nt sequences.
# From src/calm/preprocess/hamming_splits.py: max_mismatches = int((1-threshold) * len)
# For 23nt: 80% -> 4, 60% -> 9, 40% -> 13
# For 22nt: 80% -> 4, 60% -> 8, 40% -> 13
# Using 23nt bounds since guides are predominantly 23nt.
THRESHOLD_BOUNDS = {
    "hamming_080": 4,   # max 4 mismatches between any test-train pair
    "hamming_060": 9,
    "hamming_040": 13,
}


def hamming_distance_length_penalty(s1: str, s2: str) -> int:
    """
    Reproduces the length-penalty semantics of
    src/calm/preprocess/hamming_splits.py:hamming_distance.
    For unequal lengths: mismatches in aligned prefix + |len(s1)-len(s2)|.
    """
    min_len = min(len(s1), len(s2))
    mismatches = sum(a != b for a, b in zip(s1[:min_len], s2[:min_len]))
    length_penalty = abs(len(s1) - len(s2))
    return mismatches + length_penalty


def min_hamming_to_set(query: str, targets: list) -> int:
    """
    Compute the minimum Hamming distance from `query` to any string in `targets`.
    Uses the length-penalty-inclusive Hamming used by the splits code.
    """
    return min(hamming_distance_length_penalty(query, t) for t in targets)


def load_hash_to_seq_mapping(metadata_path: str, hash_col: str, seq_col: str) -> dict:
    """Load metadata and build hash -> unique guide_seq mapping."""
    df = pd.read_csv(metadata_path)
    # metadata has one row per (guide, off-target) pair; we want unique guides
    unique_guides = df.drop_duplicates(subset=[hash_col])
    return dict(zip(unique_guides[hash_col], unique_guides[seq_col]))


def verify_split_leakage(split_path: str, hash_to_seq: dict, max_mismatches: int) -> dict:
    """
    For a single split JSON file, compute the leakage statistics.

    Returns dict with:
      - n_train_val_unique
      - n_test_unique
      - min_hammings: list of int, one per unique test guide
      - n_leaked: number of test guides with min_hamming <= max_mismatches
      - leakage_fraction: n_leaked / n_test_unique
    """
    with open(split_path) as f:
        split = json.load(f)

    # Hash -> sequence lookups for each partition
    train_val_hashes = split["train"] + split["val"]
    test_hashes = split["test"]

    # Map to unique sequences (not rows)
    train_val_seqs = sorted({hash_to_seq[h] for h in train_val_hashes if h in hash_to_seq})
    test_seqs = sorted({hash_to_seq[h] for h in test_hashes if h in hash_to_seq})

    # Compute min Hamming distance from each test guide to train+val set
    min_hammings = []
    n_leaked = 0
    for test_seq in test_seqs:
        min_h = min_hamming_to_set(test_seq, train_val_seqs)
        min_hammings.append(min_h)
        if min_h <= max_mismatches:
            n_leaked += 1

    return {
        "split_file": os.path.basename(split_path),
        "n_train_val_unique": len(train_val_seqs),
        "n_test_unique": len(test_seqs),
        "max_mismatches": max_mismatches,
        "min_hamming_overall": int(min(min_hammings)),
        "min_hamming_mean": round(float(np.mean(min_hammings)), 2),
        "min_hamming_median": int(np.median(min_hammings)),
        "n_leaked": n_leaked,
        "leakage_fraction": round(n_leaked / len(test_seqs), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split_dir", required=True,
                        help="Directory containing hamming_040, hamming_060, hamming_080 subdirectories")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hash_col", default="Unique_ag_vh_vl_hash")
    parser.add_argument("--seq_col", default="guide_seq")
    parser.add_argument("--inner_fold", type=int, default=0,
                        help="Which inner fold to audit (default 0; others yield similar results)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

    print(f"crisprSFM Leakage Verification")
    print(f"  metadata:   {args.metadata}")
    print(f"  split_dir:  {args.split_dir}")
    print(f"  output_dir: {args.output_dir}")
    print()

    # Build hash -> sequence mapping
    print("Loading metadata and building hash -> sequence mapping...")
    hash_to_seq = load_hash_to_seq_mapping(args.metadata, args.hash_col, args.seq_col)
    print(f"  {len(hash_to_seq)} unique hashes -> guide sequences")
    print()

    # Process each threshold
    all_results = {}
    for threshold_name, max_mm in THRESHOLD_BOUNDS.items():
        print(f"=== {threshold_name} (max {max_mm} mismatches) ===")
        threshold_results = []
        for outer_fold in range(5):
            split_filename = f"split_hash_ids_outerfold_{outer_fold}_innerfold_{args.inner_fold}.json"
            split_path = os.path.join(args.split_dir, threshold_name, split_filename)
            if not os.path.exists(split_path):
                print(f"  WARNING: fold {outer_fold} split file not found: {split_path}")
                continue
            result = verify_split_leakage(split_path, hash_to_seq, max_mm)
            result["outer_fold"] = outer_fold
            result["inner_fold"] = args.inner_fold
            threshold_results.append(result)
            print(f"  fold {outer_fold}: "
                  f"{result['n_leaked']}/{result['n_test_unique']} leaked "
                  f"({result['leakage_fraction']*100:.1f}%), "
                  f"min_ham_overall={result['min_hamming_overall']}, "
                  f"mean={result['min_hamming_mean']}")
        all_results[threshold_name] = threshold_results

        # Save per-threshold JSON
        out_path = os.path.join(args.output_dir, "results", f"{threshold_name}_fold0-4.json")
        with open(out_path, "w") as f:
            json.dump(threshold_results, f, indent=2)
        print(f"  Saved to {out_path}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = {}
    for threshold_name, results in all_results.items():
        if not results:
            continue
        leakage_fractions = [r["leakage_fraction"] for r in results]
        min_hammings_overall = [r["min_hamming_overall"] for r in results]
        summary[threshold_name] = {
            "max_mismatches": THRESHOLD_BOUNDS[threshold_name],
            "n_folds": len(results),
            "leakage_fraction_mean": round(float(np.mean(leakage_fractions)), 4),
            "leakage_fraction_max": round(float(np.max(leakage_fractions)), 4),
            "min_hamming_overall_min": int(np.min(min_hammings_overall)),
            "min_hamming_overall_max": int(np.max(min_hammings_overall)),
        }
        print(f"{threshold_name}:")
        print(f"  mean leakage across folds: {summary[threshold_name]['leakage_fraction_mean']*100:.1f}%")
        print(f"  max leakage across folds:  {summary[threshold_name]['leakage_fraction_max']*100:.1f}%")
        print(f"  min hamming observed:      {summary[threshold_name]['min_hamming_overall_min']} "
              f"(should be > {THRESHOLD_BOUNDS[threshold_name]} for strict-OOD)")
        print()

    summary_path = os.path.join(args.output_dir, "results", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
