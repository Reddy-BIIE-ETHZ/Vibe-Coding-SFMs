"""Filtered pool-512 eval for eSFM strict-OOD reevaluation."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from calm.encoder.eval_esfm_pool512 import (
    project_unique_embeddings,
    pool_retrieval_unique,
)


def load_clean_hashes(filter_dir, split_method, fold):
    path = Path(filter_dir) / f"{split_method}_fold_{fold}_clean_test_hashes.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return set(data["clean_hashes"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--filter_dir", required=True)
    parser.add_argument("--pool_size", type=int, default=512)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_queries", type=int, default=500)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["mmseqs_040", "mmseqs_060", "mmseqs_080"],
    )
    parser.add_argument("--output_csv",
                        default="pool512_results_unique_filtered.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("Loading embeddings...")
    ag_embed = torch.load(data_dir / "ag_embed.pt", map_location="cpu", weights_only=True)
    ab_embed = torch.load(data_dir / "ab_embed.pt", map_location="cpu", weights_only=True)
    ag_mask = torch.load(data_dir / "ag_mask.pt", map_location="cpu", weights_only=True)
    ab_mask = torch.load(data_dir / "ab_mask.pt", map_location="cpu", weights_only=True)
    ag_indices = torch.load(data_dir / "ag_indices.pt", map_location="cpu", weights_only=True)
    ab_indices = torch.load(data_dir / "ab_indices.pt", map_location="cpu", weights_only=True)
    metadata = pd.read_csv(data_dir / "metadata.csv")

    hash_to_row = {h: i for i, h in enumerate(metadata["Unique_ag_vh_vl_hash"])}

    all_results = []
    proj_cache = {}

    for split in args.splits:
        print(f"\n{'='*60}\nSplit: {split} (FILTERED)\n{'='*60}")
        fold_results = []
        for fold in range(5):
            run_dir = output_dir / f"esfm-{split}-fold{fold}" / "train" / f"fold_{fold}"
            ckpt_files = list(run_dir.glob("best_model_val_pred_acc_epoch_*.pth"))
            if not ckpt_files:
                alt = run_dir / "best_model.pth"
                if alt.exists():
                    ckpt_files = [alt]
            if not ckpt_files:
                print(f"  fold {fold}: no checkpoint, skipping")
                continue

            def epoch_of(p):
                stem = p.stem
                if "_epoch_" in stem:
                    try:
                        return int(stem.split("_")[-1])
                    except ValueError:
                        return -1
                return -1

            best_ckpt = max(ckpt_files, key=epoch_of)

            split_dir = data_dir / "split_index" / split
            split_file = split_dir / f"split_hash_ids_outerfold_{fold}_innerfold_{fold}.json"
            if not split_file.exists():
                split_file = split_dir / f"split_hash_ids_outerfold_{fold}_innerfold_0.json"
            if not split_file.exists():
                print(f"  fold {fold}: split not found, skipping")
                continue

            with open(split_file) as f:
                split_data = json.load(f)

            if epoch_of(best_ckpt) == 0:
                print(f"  fold {fold}: SKIP (degenerate val)")
                continue

            test_hashes_all = split_data.get("test", [])

            clean_hashes = load_clean_hashes(args.filter_dir, split, fold)
            if clean_hashes is None:
                print(f"  fold {fold}: no filter file, skipping")
                continue
            test_hashes = [h for h in test_hashes_all if h in clean_hashes]
            n_filtered_out = len(test_hashes_all) - len(test_hashes)
            print(f"  fold {fold}: filtered {n_filtered_out}/{len(test_hashes_all)} "
                  f"leaked test pairs; {len(test_hashes)} remain")

            test_row_indices = [hash_to_row[h] for h in test_hashes if h in hash_to_row]
            if not test_row_indices:
                print(f"  fold {fold}: empty test set after filter, skipping")
                continue

            test_ag_rows = [ag_indices[r].item() for r in test_row_indices]
            test_ab_rows = [ab_indices[r].item() for r in test_row_indices]
            test_sub_ids = sorted(set(test_ag_rows))
            test_enz_ids = sorted(set(test_ab_rows))
            unique_pairs = list(set(zip(test_ag_rows, test_ab_rows)))

            if str(best_ckpt) not in proj_cache:
                proj_cache[str(best_ckpt)] = project_unique_embeddings(
                    str(best_ckpt), ag_embed, ab_embed, ag_mask, ab_mask,
                )
            all_ag_proj, all_ab_proj, temp = proj_cache[str(best_ckpt)]

            sub_local = {s: i for i, s in enumerate(test_sub_ids)}
            enz_local = {e: i for i, e in enumerate(test_enz_ids)}

            pos_s2e = {}
            pos_e2s = {}
            for s_id, e_id in unique_pairs:
                sl, el = sub_local[s_id], enz_local[e_id]
                pos_s2e.setdefault(sl, set()).add(el)
                pos_e2s.setdefault(el, set()).add(sl)

            sub_proj = all_ag_proj[test_sub_ids]
            enz_proj = all_ab_proj[test_enz_ids]

            n_s = len(test_sub_ids)
            n_e = len(test_enz_ids)
            print(f"    ckpt={best_ckpt.name}, {len(unique_pairs)} unique pairs, "
                  f"{n_s} substrates, {n_e} enzymes")

            fr = pool_retrieval_unique(
                sub_proj, enz_proj, pos_s2e, pos_e2s,
                pool_size=args.pool_size, n_trials=args.n_trials,
                seed=args.seed, max_queries=args.max_queries,
            )
            fr["split"] = split + "_filtered"
            fr["fold"] = fold
            fr["temperature"] = temp
            fr["checkpoint"] = best_ckpt.name
            fr["n_unique_pairs"] = len(unique_pairs)
            fr["n_test_hashes_pre_filter"] = len(test_hashes_all)
            fr["n_test_hashes_post_filter"] = len(test_hashes)

            print(f"    R@1 sub->enz: {fr['R@1_ag2ab']:5.1f}% | enz->sub: {fr['R@1_ab2ag']:5.1f}%")
            fold_results.append(fr)
            all_results.append(fr)

        if fold_results:
            print(f"\n  --- {split}_filtered summary ({len(fold_results)} folds) ---")
            for m in ["R@1_ag2ab", "R@1_ab2ag"]:
                vals = [r[m] for r in fold_results]
                print(f"    {m:<12}: {np.mean(vals):5.1f} +/- {np.std(vals):4.1f}%")

    if all_results:
        results_path = output_dir / args.output_csv
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
