"""
Pool-512 evaluation for eSFM from saved checkpoints.

Evaluates on UNIQUE (substrate, enzyme) pairs only (not raw data pairs which
have duplication due to many pairs sharing substrates). For each query
substrate, retrieves from a pool of unique enzymes. For each query enzyme,
retrieves from a pool of unique substrates.

CRITICAL: Follows the pattern from eval_dtsfm_pool512.py to avoid the
duplicate-embedding retrieval bug. Reports effective pool size honestly.

Architectural note: in eSFM the substrate is the "target" slot and the
enzyme is the "agent" slot in the biology, but CALM's code uses
  ag = MoLFormer (substrate, 768-dim)    [ag_embed.pt, ag_indices.pt]
  ab = ESM-2    (enzyme, 1280-dim)       [ab_embed.pt, ab_indices.pt]
to match the dtSFM pairing (drug in ag slot, protein in ab slot). Results
are reported in both directions:
  ag2ab = substrate -> enzyme
  ab2ag = enzyme    -> substrate

Usage:
    python -m calm.encoder.eval_esfm_pool512 \\
        --data_dir /cluster/scratch/reddys/esfm \\
        --output_dir /cluster/scratch/reddys/esfm/output/esfm_full \\
        --pool_size 512 --n_trials 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def project_unique_embeddings(
    checkpoint_path: str,
    ag_embed: torch.Tensor,
    ab_embed: torch.Tensor,
    ag_mask: torch.Tensor,
    ab_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Project ALL unique embeddings through the trained projection heads.

    For eSFM the preprocessed embeddings are already mean-pooled to shape
    (N, 1, D), so "mean-pool of length 1" is a no-op; the result is
    identical to token-level mean pooling.

    Returns
    -------
    all_ag_proj: (N_unique_substrates, 512) L2-normalized
    all_ab_proj: (N_unique_enzymes, 512)    L2-normalized
    temperature: learned temperature (scalar)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    enc_ag = nn.Sequential(
        nn.Linear(768, 2048, bias=False),
        nn.ReLU(),
        nn.Linear(2048, 512, bias=False),
    )
    enc_ab = nn.Sequential(
        nn.Linear(1280, 2048, bias=False),
        nn.ReLU(),
        nn.Linear(2048, 512, bias=False),
    )
    enc_ag.load_state_dict(
        {
            k.replace("encoder_ag.", ""): v
            for k, v in ckpt.items()
            if k.startswith("encoder_ag.")
        }
    )
    enc_ab.load_state_dict(
        {
            k.replace("encoder_ab.", ""): v
            for k, v in ckpt.items()
            if k.startswith("encoder_ab.")
        }
    )
    enc_ag.eval()
    enc_ab.eval()

    temperature = 1.0 / math.exp(ckpt["logit_scale"].item())

    with torch.no_grad():
        ag_mf = ag_mask.unsqueeze(-1).float()
        ag_pooled = (ag_embed * ag_mf).sum(1) / ag_mf.sum(1).clamp(min=1)
        all_ag = F.normalize(enc_ag(ag_pooled), dim=-1)

        ab_mf = ab_mask.unsqueeze(-1).float()
        ab_pooled = (ab_embed * ab_mf).sum(1) / ab_mf.sum(1).clamp(min=1)
        all_ab = F.normalize(enc_ab(ab_pooled), dim=-1)

    return all_ag, all_ab, temperature


def _retrieval_one_direction(
    sim: np.ndarray,
    pos_map: dict[int, set[int]],
    n_items: int,
    pool_size: int,
    n_trials: int,
    k_values: tuple[int, ...],
    rng: np.random.RandomState,
    max_queries: int | None = None,
) -> dict[str, float]:
    """Pool retrieval from one side: for each query, run n_trials draws of
    pool_size negatives + one positive, record rank.

    If max_queries is set and n_queries > max_queries, subsample queries
    uniformly. Random subsampling of queries preserves mean recall (each
    query is an i.i.d. estimator of the population recall).
    """
    eff_pool = min(pool_size, n_items)
    n_queries = sim.shape[0]

    # Pick which queries to use
    all_query_ids = [qi for qi in range(n_queries) if pos_map.get(qi)]
    if max_queries is not None and len(all_query_ids) > max_queries:
        chosen_idx = rng.choice(
            len(all_query_ids), size=max_queries, replace=False
        )
        query_ids = [all_query_ids[i] for i in chosen_idx]
    else:
        query_ids = all_query_ids

    per_trial_hits = {k: np.zeros(n_trials, dtype=np.int64) for k in k_values}
    per_trial_valid = np.zeros(n_trials, dtype=np.int64)

    for qi in query_ids:
        pos_items = pos_map[qi]

        is_pos = np.zeros(n_items, dtype=bool)
        is_pos[list(pos_items)] = True
        neg_pool = np.where(~is_pos)[0]

        n_neg = min(eff_pool - 1, neg_pool.shape[0])
        if n_neg <= 0:
            continue

        sim_row = sim[qi]
        pos_list = list(pos_items)
        n_pos = len(pos_list)

        for trial_idx in range(n_trials):
            q_pos = pos_list[rng.randint(n_pos)]
            sampled_neg = rng.choice(neg_pool, size=n_neg, replace=False)
            # Vectorised rank computation
            pos_sim = sim_row[q_pos]
            neg_sims = sim_row[sampled_neg]
            rank = int((neg_sims > pos_sim).sum())
            per_trial_valid[trial_idx] += 1
            for k in k_values:
                if rank < k:
                    per_trial_hits[k][trial_idx] += 1

    results = {}
    for k in k_values:
        with np.errstate(divide="ignore", invalid="ignore"):
            recalls = np.where(
                per_trial_valid > 0,
                per_trial_hits[k] / np.maximum(per_trial_valid, 1) * 100.0,
                0.0,
            )
        results[f"R@{k}"] = float(np.mean(recalls))
        results[f"R@{k}_std"] = float(np.std(recalls))
    results["n_queries_used"] = len(query_ids)
    return results


def pool_retrieval_unique(
    sub_proj: torch.Tensor,
    enz_proj: torch.Tensor,
    pos_sub_to_enz: dict[int, set[int]],
    pos_enz_to_sub: dict[int, set[int]],
    pool_size: int = 512,
    n_trials: int = 100,
    k_values: tuple[int, ...] = (1, 5, 10),
    seed: int = 42,
    max_queries: int | None = 500,
) -> dict[str, float]:
    """Pool retrieval on unique substrates × unique enzymes.

    Substrate→Enzyme: for each query substrate, sample pool_size-1 wrong
    enzymes + the correct one, compute cosine sim, rank.

    Enzyme→Substrate: same, in the reverse direction.

    max_queries: subsample query count per direction (default 500). With
    35K+ enzymes, running full query loops is slow; random subsampling
    gives identical mean recall with ~70× speedup.
    """
    rng = np.random.RandomState(seed)
    n_sub = sub_proj.shape[0]
    n_enz = enz_proj.shape[0]

    # All-vs-all cosine similarity (already L2-normalized)
    sim_s2e = (sub_proj @ enz_proj.T).numpy().astype(np.float32)
    sim_e2s = sim_s2e.T.copy()

    results = {}

    s2e = _retrieval_one_direction(
        sim_s2e, pos_sub_to_enz, n_enz,
        pool_size, n_trials, k_values, rng, max_queries=max_queries,
    )
    for k in k_values:
        results[f"R@{k}_ag2ab"] = s2e[f"R@{k}"]
        results[f"R@{k}_ag2ab_std"] = s2e[f"R@{k}_std"]
    results["n_queries_ag2ab"] = s2e["n_queries_used"]

    e2s = _retrieval_one_direction(
        sim_e2s, pos_enz_to_sub, n_sub,
        pool_size, n_trials, k_values, rng, max_queries=max_queries,
    )
    for k in k_values:
        results[f"R@{k}_ab2ag"] = e2s[f"R@{k}"]
        results[f"R@{k}_ab2ag_std"] = e2s[f"R@{k}_std"]
    results["n_queries_ab2ag"] = e2s["n_queries_used"]

    results["pool_size_ag2ab"] = min(pool_size, n_enz)
    results["pool_size_ab2ag"] = min(pool_size, n_sub)
    results["n_unique_substrates"] = n_sub
    results["n_unique_enzymes"] = n_enz

    return results


def main():
    parser = argparse.ArgumentParser(description="Pool-512 eval for eSFM")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with ag_embed.pt, ab_embed.pt, "
                             "ag_indices.pt, ab_indices.pt, metadata.csv, "
                             "and split_index/ subdirectories")
    parser.add_argument("--output_dir", required=True,
                        help="Directory containing esfm-*-fold* training "
                             "output subdirectories")
    parser.add_argument("--pool_size", type=int, default=512)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_queries",
        type=int,
        default=500,
        help="Subsample this many query entities per direction (default 500). "
             "With 35K+ enzymes running the full query loop is slow; "
             "random subsampling gives identical mean recall.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["identity_100", "mmseqs_080", "mmseqs_060", "mmseqs_040"],
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("Loading embeddings...")
    ag_embed = torch.load(
        data_dir / "ag_embed.pt", map_location="cpu", weights_only=True
    )
    ab_embed = torch.load(
        data_dir / "ab_embed.pt", map_location="cpu", weights_only=True
    )
    ag_mask = torch.load(
        data_dir / "ag_mask.pt", map_location="cpu", weights_only=True
    )
    ab_mask = torch.load(
        data_dir / "ab_mask.pt", map_location="cpu", weights_only=True
    )
    ag_indices = torch.load(
        data_dir / "ag_indices.pt", map_location="cpu", weights_only=True
    )
    ab_indices = torch.load(
        data_dir / "ab_indices.pt", map_location="cpu", weights_only=True
    )
    metadata = pd.read_csv(data_dir / "metadata.csv")
    print(f"  Substrates: {ag_embed.shape[0]} unique (ag)")
    print(f"  Enzymes:    {ab_embed.shape[0]} unique (ab)")
    print(f"  Pairs:      {len(metadata)}")

    hash_to_row = {
        h: i for i, h in enumerate(metadata["Unique_ag_vh_vl_hash"])
    }

    all_results = []

    # Cache projected embeddings per checkpoint to avoid recomputing
    proj_cache = {}

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")

        fold_results = []

        for fold in range(5):
            run_dir = (
                output_dir
                / f"esfm-{split}-fold{fold}"
                / "train"
                / f"fold_{fold}"
            )

            ckpt_files = list(
                run_dir.glob("best_model_val_pred_acc_epoch_*.pth")
            )
            if not ckpt_files:
                # Also try the plain best_model.pth naming
                alt = run_dir / "best_model.pth"
                if alt.exists():
                    ckpt_files = [alt]

            if not ckpt_files:
                print(f"  fold {fold}: no checkpoint, skipping")
                continue

            # Pick the latest (highest-epoch) checkpoint
            def epoch_of(p: Path) -> int:
                stem = p.stem
                if "_epoch_" in stem:
                    try:
                        return int(stem.split("_")[-1])
                    except ValueError:
                        return -1
                return -1

            best_ckpt = max(ckpt_files, key=epoch_of)

            # Load split
            split_dir = data_dir / "split_index" / split
            split_file = (
                split_dir
                / f"split_hash_ids_outerfold_{fold}_innerfold_{fold}.json"
            )
            if not split_file.exists():
                split_file = (
                    split_dir
                    / f"split_hash_ids_outerfold_{fold}_innerfold_0.json"
                )
            if not split_file.exists():
                print(f"  fold {fold}: split not found, skipping")
                continue

            with open(split_file) as f:
                split_data = json.load(f)

            # Degenerate-val-set bug: CALM's innerfold_N == outerfold_N
            # splits leave val sets of size ~2, which makes val pred_acc
            # swing chaotically between 0% and 50% per epoch. The "best
            # val" tracker never improves past epoch 0, so the saved
            # checkpoint is effectively untrained. Skip these folds.
            val_hashes = split_data.get("val", [])
            if epoch_of(best_ckpt) == 0:
                print(
                    f"  fold {fold}: SKIP — best val checkpoint at epoch 0 "
                    f"(val size = {len(val_hashes)}, likely degenerate val "
                    f"tracking)"
                )
                continue

            test_hashes = split_data.get("test", [])
            test_row_indices = [
                hash_to_row[h] for h in test_hashes if h in hash_to_row
            ]
            if not test_row_indices:
                print(f"  fold {fold}: empty test set, skipping")
                continue

            # Project embeddings (cache per checkpoint)
            ckpt_key = str(best_ckpt)
            if ckpt_key not in proj_cache:
                print(f"  Projecting embeddings for {best_ckpt.name}...")
                proj_cache[ckpt_key] = project_unique_embeddings(
                    ckpt_key, ag_embed, ab_embed, ag_mask, ab_mask
                )
            all_ag_proj, all_ab_proj, temp = proj_cache[ckpt_key]

            # Build unique (ag_idx, ab_idx) pairs from test rows
            test_ag = ag_indices[test_row_indices].numpy()
            test_ab = ab_indices[test_row_indices].numpy()
            unique_pairs = set()
            for i in range(len(test_row_indices)):
                unique_pairs.add((int(test_ag[i]), int(test_ab[i])))

            test_sub_ids = sorted(set(p[0] for p in unique_pairs))
            test_enz_ids = sorted(set(p[1] for p in unique_pairs))

            sub_local = {s: i for i, s in enumerate(test_sub_ids)}
            enz_local = {e: i for i, e in enumerate(test_enz_ids)}

            pos_s2e: dict[int, set[int]] = {}
            pos_e2s: dict[int, set[int]] = {}
            for s_id, e_id in unique_pairs:
                sl = sub_local[s_id]
                el = enz_local[e_id]
                pos_s2e.setdefault(sl, set()).add(el)
                pos_e2s.setdefault(el, set()).add(sl)

            sub_proj = all_ag_proj[test_sub_ids]
            enz_proj = all_ab_proj[test_enz_ids]

            n_s = len(test_sub_ids)
            n_e = len(test_enz_ids)
            print(
                f"  fold {fold}: ckpt={best_ckpt.name}, "
                f"{len(unique_pairs)} unique pairs, "
                f"{n_s} substrates, {n_e} enzymes"
            )

            fr = pool_retrieval_unique(
                sub_proj, enz_proj, pos_s2e, pos_e2s,
                pool_size=args.pool_size, n_trials=args.n_trials,
                seed=args.seed, max_queries=args.max_queries,
            )
            fr["split"] = split
            fr["fold"] = fold
            fr["temperature"] = temp
            fr["checkpoint"] = best_ckpt.name
            fr["n_unique_pairs"] = len(unique_pairs)

            print(
                f"    R@1  sub->enz: {fr['R@1_ag2ab']:5.1f}%  |  "
                f"enz->sub: {fr['R@1_ab2ag']:5.1f}%"
            )
            print(
                f"    R@5  sub->enz: {fr['R@5_ag2ab']:5.1f}%  |  "
                f"enz->sub: {fr['R@5_ab2ag']:5.1f}%"
            )
            print(
                f"    R@10 sub->enz: {fr['R@10_ag2ab']:5.1f}%  |  "
                f"enz->sub: {fr['R@10_ab2ag']:5.1f}%"
            )

            fold_results.append(fr)
            all_results.append(fr)

        if fold_results:
            print(f"\n  --- {split} summary ({len(fold_results)} folds) ---")
            for m in [
                "R@1_ag2ab", "R@1_ab2ag",
                "R@5_ag2ab", "R@5_ab2ag",
                "R@10_ag2ab", "R@10_ab2ag",
            ]:
                vals = [r[m] for r in fold_results]
                print(
                    f"    {m:<12}: {np.mean(vals):5.1f} ± {np.std(vals):4.1f}%"
                )

    # Save all fold-level results
    if all_results:
        results_path = output_dir / "pool512_results_unique.csv"
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {results_path}")

    # Final table
    print(f"\n{'='*72}")
    print("FINAL SUMMARY — Tables 15-16 (unique substrate-enzyme pairs)")
    print(f"{'='*72}")
    print(
        f"{'Split':<14} {'R@1(S→E)':>12} {'R@1(E→S)':>12} "
        f"{'R@5(S→E)':>12} {'R@5(E→S)':>12} "
        f"{'R@10(S→E)':>12} {'R@10(E→S)':>12}"
    )
    print("-" * 84)
    for split in args.splits:
        sr = [r for r in all_results if r["split"] == split]
        if sr:
            cells = []
            for m in [
                "R@1_ag2ab", "R@1_ab2ag",
                "R@5_ag2ab", "R@5_ab2ag",
                "R@10_ag2ab", "R@10_ab2ag",
            ]:
                vals = [r[m] for r in sr]
                cells.append(f"{np.mean(vals):5.1f}±{np.std(vals):4.1f}")
            print(f"{split:<14} " + " ".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    main()
