"""
Fold-level unique entity diagnostics for eSFM.

For each split (identity_100, mmseqs_080, mmseqs_060, mmseqs_040) and each
fold, counts:
    - raw train / val / test pair counts
    - unique substrate (ag_idx) counts in train / val / test
    - unique enzyme (ab_idx) counts in train / val / test
    - unique (ag_idx, ab_idx) pair counts

This catches the duplicate-embedding problem BEFORE running any training:
if a test fold has few unique substrates, "pool-512" retrieval there is
actually pool-n (n << 512) and results will be artificially inflated.

Usage:
    python -m calm.encoder.esfm_fold_diagnostics \\
        --data_dir /cluster/scratch/reddys/esfm \\
        --output fold_diagnostics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch


def analyze_fold(
    metadata: pd.DataFrame,
    ag_indices: torch.Tensor,
    ab_indices: torch.Tensor,
    hash_to_row: dict[str, int],
    split_file: Path,
) -> dict:
    """Return counts for one fold's train/val/test split."""
    with open(split_file) as f:
        split_data = json.load(f)

    row = {"split_file": split_file.name}
    for phase in ("train", "val", "test"):
        hashes = split_data.get(phase, [])
        rows = [hash_to_row[h] for h in hashes if h in hash_to_row]
        if not rows:
            row[f"{phase}_pairs"] = 0
            row[f"{phase}_unique_ag"] = 0
            row[f"{phase}_unique_ab"] = 0
            row[f"{phase}_unique_pairs"] = 0
            continue

        ag_ids = ag_indices[rows].tolist()
        ab_ids = ab_indices[rows].tolist()

        row[f"{phase}_pairs"] = len(rows)
        row[f"{phase}_unique_ag"] = len(set(ag_ids))
        row[f"{phase}_unique_ab"] = len(set(ab_ids))
        row[f"{phase}_unique_pairs"] = len({(a, b) for a, b in zip(ag_ids, ab_ids)})

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Count unique substrates/enzymes per eSFM fold"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="fold_diagnostics.csv")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["identity_100", "mmseqs_080", "mmseqs_060", "mmseqs_040"],
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    metadata = pd.read_csv(data_dir / "metadata.csv")
    ag_indices = torch.load(
        data_dir / "ag_indices.pt", map_location="cpu", weights_only=True
    )
    ab_indices = torch.load(
        data_dir / "ab_indices.pt", map_location="cpu", weights_only=True
    )

    n_total_ag = int(ag_indices.max().item()) + 1
    n_total_ab = int(ab_indices.max().item()) + 1
    print(f"Metadata: {len(metadata):,} pairs")
    print(f"  Total unique substrates (ag): {n_total_ag:,}")
    print(f"  Total unique enzymes (ab):    {n_total_ab:,}")
    print()

    hash_to_row = {h: i for i, h in enumerate(metadata["Unique_ag_vh_vl_hash"])}

    all_rows = []

    for split in args.splits:
        split_dir = data_dir / "split_index" / split
        if not split_dir.exists():
            print(f"[skip] {split}: directory not found")
            continue

        split_files = sorted(split_dir.glob("split_hash_ids_outerfold_*.json"))
        if not split_files:
            print(f"[skip] {split}: no split files")
            continue

        print(f"=== {split} ({len(split_files)} fold files) ===")
        header = (
            f"  {'file':<55} {'train_pairs':>11} {'test_pairs':>11} "
            f"{'test_uAg':>8} {'test_uAb':>8} {'test_uPair':>11}"
        )
        print(header)

        for sf in split_files:
            row = analyze_fold(metadata, ag_indices, ab_indices, hash_to_row, sf)
            row["split"] = split
            all_rows.append(row)
            print(
                f"  {sf.name:<55} "
                f"{row['train_pairs']:>11,} "
                f"{row['test_pairs']:>11,} "
                f"{row['test_unique_ag']:>8,} "
                f"{row['test_unique_ab']:>8,} "
                f"{row['test_unique_pairs']:>11,}"
            )

        # Summary over folds within this split
        test_uags = [r["test_unique_ag"] for r in all_rows if r["split"] == split]
        test_uabs = [r["test_unique_ab"] for r in all_rows if r["split"] == split]
        print(
            f"  -> test_unique_ag:  min={min(test_uags)}, median={sorted(test_uags)[len(test_uags)//2]}, max={max(test_uags)}"
        )
        print(
            f"  -> test_unique_ab:  min={min(test_uabs)}, median={sorted(test_uabs)[len(test_uabs)//2]}, max={max(test_uabs)}"
        )
        print()

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved {len(all_rows)} fold rows to {args.output}")

    # Final verdict
    print()
    print("=" * 72)
    print("POOL-512 VERDICT (substrate -> enzyme direction is usually harder)")
    print("=" * 72)
    for split in args.splits:
        split_rows = [r for r in all_rows if r["split"] == split]
        if not split_rows:
            continue
        uabs = [r["test_unique_ab"] for r in split_rows]
        uags = [r["test_unique_ag"] for r in split_rows]
        med_uab = sorted(uabs)[len(uabs) // 2]
        med_uag = sorted(uags)[len(uags) // 2]
        verdict_ab = "OK" if med_uab >= 512 else f"DEGRADED (pool={med_uab})"
        verdict_ag = "OK" if med_uag >= 512 else f"DEGRADED (pool={med_uag})"
        print(f"  {split:<15} enzyme pool: {verdict_ab:<25} substrate pool: {verdict_ag}")


if __name__ == "__main__":
    main()
