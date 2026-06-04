"""Generate OOD splits using Hamming distance clustering for short sequences.

MMseqs2 struggles with very short sequences (< 30 nt). For CRISPR guide RNAs
(20-23 nt), direct Hamming distance clustering is faster and more appropriate.

What this does in plain language:
    We want to group similar guide RNA sequences so that during testing, the
    model never sees a guide that's very similar to one it trained on.

    Hamming distance = number of positions where two sequences differ.
    For 22 nt guides:
        80% identity = at most 4 mismatches  (22 * 0.20 = 4.4)
        60% identity = at most 8 mismatches  (22 * 0.40 = 8.8)
        40% identity = at most 13 mismatches (22 * 0.60 = 13.2)

    We use greedy single-linkage clustering: pick an unclustered sequence,
    find all unclustered sequences within the mismatch threshold, group them,
    repeat. This is the same idea as MMseqs2 but exact for short sequences.

Usage (Euler login node — no GPU needed):
    python -m calm.preprocess.hamming_splits \
        --metadata /cluster/scratch/reddys/crispr/data/metadata.csv \
        --seq_col guide_seq \
        --output_dir /cluster/scratch/reddys/crispr/data/split_index \
        --thresholds 0.4 0.6 0.8

Output structure (same as mmseqs_splits.py):
    split_index/
        hamming_040/         <- 40% identity (strictest OOD)
        hamming_060/         <- 60% identity
        hamming_080/         <- 80% identity
        identity_100/        <- in-distribution (random split)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from calm.preprocess.split_utils import build_cluster_splits


def hamming_distance(s1: str, s2: str) -> int:
    """Count mismatches between two sequences of equal or similar length."""
    s1, s2 = s1.upper(), s2.upper()
    minlen = min(len(s1), len(s2))
    return sum(c1 != c2 for c1, c2 in zip(s1[:minlen], s2[:minlen])) + abs(len(s1) - len(s2))


def cluster_by_hamming(
    sequences: list[str],
    identity_threshold: float,
) -> dict[str, int]:
    """Greedy single-linkage clustering by Hamming distance.

    Parameters
    ----------
    sequences : list of str
        Unique sequences to cluster.
    identity_threshold : float
        Minimum identity to merge (e.g., 0.8 = 80%).
        Two sequences merge if their Hamming distance <= (1 - threshold) * max(len).

    Returns
    -------
    dict[str, int]
        Mapping from sequence to cluster ID.
    """
    seq_list = list(set(sequences))  # deduplicate
    n = len(seq_list)
    assigned = [-1] * n
    cluster_id = 0

    for i in range(n):
        if assigned[i] >= 0:
            continue
        # Start a new cluster with sequence i
        assigned[i] = cluster_id
        max_mismatches = int((1 - identity_threshold) * len(seq_list[i]))

        # Find all unassigned sequences within threshold
        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            if hamming_distance(seq_list[i], seq_list[j]) <= max_mismatches:
                assigned[j] = cluster_id

        cluster_id += 1

    result = {seq: cid for seq, cid in zip(seq_list, assigned)}
    n_clusters = cluster_id
    print(f"  Hamming clustering at {identity_threshold:.0%} identity: "
          f"{n_clusters} clusters from {n} sequences "
          f"(max {int((1 - identity_threshold) * len(seq_list[0]))} mismatches)")
    return result


def generate_splits_at_threshold(
    df: pd.DataFrame,
    seq_col: str,
    hash_col: str,
    threshold: float,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Generate cluster-aware OOD splits at one identity threshold."""
    unique_seqs = df[seq_col].unique().tolist()

    # Cluster by Hamming distance
    seq_to_cluster = cluster_by_hamming(unique_seqs, threshold)

    # Map cluster IDs back to all rows
    df = df.copy()
    df["_hamming_cluster"] = df[seq_col].map(seq_to_cluster)

    # Create split directory (use "hamming_" prefix to distinguish from mmseqs_)
    threshold_dir = output_dir / f"hamming_{int(threshold * 100):03d}"
    threshold_dir.mkdir(parents=True, exist_ok=True)

    build_cluster_splits(
        df, threshold_dir,
        cluster_col="_hamming_cluster",
        hash_col=hash_col,
        n_folds=n_folds,
        seed=seed,
    )


def generate_id_splits(
    df: pd.DataFrame,
    hash_col: str,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Generate in-distribution splits (no clustering)."""
    df = df.copy()
    df["_singleton_cluster"] = range(len(df))

    id_dir = output_dir / "identity_100"
    id_dir.mkdir(parents=True, exist_ok=True)

    build_cluster_splits(
        df, id_dir,
        cluster_col="_singleton_cluster",
        hash_col=hash_col,
        n_folds=n_folds,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OOD splits using Hamming distance clustering (for short sequences)."
    )
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--seq_col", required=True, help="Column with sequences to cluster")
    parser.add_argument("--output_dir", required=True, help="Output directory for split files")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    parser.add_argument("--hash_col", default="Unique_ag_vh_vl_hash")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_id", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hamming distance OOD split generation")
    print(f"  metadata:   {args.metadata}")
    print(f"  seq_col:    {args.seq_col}")
    print(f"  thresholds: {args.thresholds}")
    print(f"  output_dir: {output_dir}")
    print()

    df = pd.read_csv(args.metadata)
    print(f"  Loaded {len(df)} rows, {df[args.seq_col].nunique()} unique sequences")

    for threshold in args.thresholds:
        print(f"\n--- Threshold: {threshold:.0%} identity ---")
        generate_splits_at_threshold(
            df, args.seq_col, args.hash_col, threshold,
            output_dir, n_folds=args.n_folds, seed=args.seed,
        )

    if args.include_id:
        print(f"\n--- In-distribution (no clustering) ---")
        generate_id_splits(
            df, args.hash_col, output_dir,
            n_folds=args.n_folds, seed=args.seed,
        )

    print(f"\nDone. Split directories created in {output_dir}")


if __name__ == "__main__":
    main()
