"""Shared utilities for cluster-based OOD train/val/test splitting.

This module extracts the cluster-to-fold splitting logic used across all
SFM domains (tSFM, crisprSFM, etc.) into a single reusable function.

What this does in plain language:
    Imagine you have 100 transcription factor families. You want to test
    whether the model works on families it has NEVER seen during training.
    So you divide the 100 families into 5 groups of 20. Each group takes
    a turn being the "test" group, while the rest are used for training
    and validation. This is called "cluster-aware cross-validation."
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


def build_cluster_splits(
    df: pd.DataFrame,
    output_dir: str | Path,
    cluster_col: str = "cluster_id",
    hash_col: str = "Unique_ag_vh_vl_hash",
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Create train/val/test split files based on cluster membership.

    All samples belonging to the same cluster stay together in the same
    split (train, val, or test). This prevents data leakage — the model
    never sees related examples in both training and testing.

    The output is a set of JSON files that the CALM training pipeline
    reads to know which data rows belong to train, val, or test.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata with at least ``cluster_col`` and ``hash_col`` columns.
    output_dir : str or Path
        Where to save the split JSON files.
    cluster_col : str
        Column in ``df`` that assigns each row to a cluster.
    hash_col : str
        Column in ``df`` with unique identifiers for each row.
        Must match what ``load_split_indices()`` looks for.
    n_folds : int
        Number of cross-validation folds (default 5).
    seed : int
        Random seed for reproducibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids = df[cluster_col].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(cluster_ids)

    n_test = max(1, len(cluster_ids) // n_folds)
    n_val = max(1, len(cluster_ids) // n_folds)

    for outer_fold in range(n_folds):
        # Pick test clusters for this outer fold
        test_start = outer_fold * n_test
        test_end = min(test_start + n_test, len(cluster_ids))
        test_clusters = set(cluster_ids[test_start:test_end])
        trainval_clusters = [c for c in cluster_ids if c not in test_clusters]

        rng.shuffle(trainval_clusters)

        for inner_fold in range(n_folds):
            # Pick validation clusters for this inner fold
            val_start = inner_fold * n_val
            val_end = min(val_start + n_val, len(trainval_clusters))
            val_clusters = set(trainval_clusters[val_start:val_end])
            train_clusters = set(trainval_clusters) - val_clusters

            split_hashes = {
                "train": df[df[cluster_col].isin(train_clusters)][hash_col].tolist(),
                "val": df[df[cluster_col].isin(val_clusters)][hash_col].tolist(),
                "test": df[df[cluster_col].isin(test_clusters)][hash_col].tolist(),
            }

            filename = output_dir / f"split_hash_ids_outerfold_{outer_fold}_innerfold_{inner_fold}.json"
            with open(filename, "w") as f:
                json.dump(split_hashes, f)

    total_files = n_folds * n_folds
    print(f"  Saved {total_files} split files to {output_dir}")
    print(f"  Clusters: {len(cluster_ids)} total, ~{n_test} per test fold")
