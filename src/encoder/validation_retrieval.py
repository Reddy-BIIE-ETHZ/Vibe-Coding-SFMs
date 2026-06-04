"""
Pseudo-prospective validation retrieval for tSFM.

Loads a model trained on JASPAR 2022 data, then tests whether it can
retrieve the correct DNA binding motif for TFs that were only characterized
after 2022 (held-out set).

The key question: given a held-out TF protein sequence, can the model
find its true DNA binding motif from the full pool of all DNA motifs?

Usage (on Euler, needs GPU):
    python -m calm.encoder.validation_retrieval \
        --model_dir /path/to/trained_model/train/fold_0 \
        --train_data_dir /path/to/jaspar_validation/train \
        --heldout_data_dir /path/to/jaspar_validation/heldout \
        --output_dir /path/to/validation_results \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SimpleEmbeddingDataset(Dataset):
    """Dataset that returns pre-computed embeddings via index lookup."""

    def __init__(
        self,
        ag_embed: torch.Tensor,
        ag_mask: torch.Tensor,
        ab_embed: torch.Tensor,
        ab_mask: torch.Tensor,
        ag_indices: torch.Tensor,
        ab_indices: torch.Tensor,
        row_indices: list[int],
    ):
        self.ag_embed = ag_embed
        self.ag_mask = ag_mask
        self.ab_embed = ab_embed
        self.ab_mask = ab_mask
        self.ag_indices = ag_indices
        self.ab_indices = ab_indices
        self.row_indices = torch.as_tensor(row_indices, dtype=torch.long)

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, idx):
        j = int(self.row_indices[idx])
        ag_idx = int(self.ag_indices[j])
        ab_idx = int(self.ab_indices[j])
        return (
            self.ag_embed[ag_idx],
            self.ab_embed[ab_idx],
            self.ag_mask[ag_idx],
            self.ab_mask[ab_idx],
            j,
        )


def load_model_from_dir(model_dir: str, device: str = "cuda"):
    """Load trained model state dict from a fold directory.

    Looks for final_model_epoch_*.pth in the directory.
    """
    # Find the final model checkpoint
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith("final_model")]
    if not checkpoints:
        # Fall back to best_model files
        checkpoints = [f for f in os.listdir(model_dir)
                       if f.startswith("best_model_val")]
        if checkpoints:
            # Get the one with highest epoch
            checkpoints.sort()
            checkpoints = [checkpoints[-1]]

    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoint found in {model_dir}")

    checkpoint_path = os.path.join(model_dir, checkpoints[0])
    print(f"Loading model from: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return state_dict


def build_model_from_config(device: str = "cuda"):
    """Build a CALMEncoder model using the tSFM FFN config."""
    from calm.encoder.model import CALMEncoder
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.create({
        "encoder_type": "ffn",
        "model_ag": "esm2",
        "model_ab": "dnabert2",
        "d_model": 512,
        "d_ff": 2048,
        "tau": 0.07,
        "max_scale": 100.0,
        "pooling": "mean",
        "max_length_ag": 512,
        "max_length_ab": 64,
        "include_linear_bias": False,
    })

    model = CALMEncoder(model_cfg, reduce_embeddings=False)
    return model


def embed_through_model(
    state_dict: dict,
    ag_embed: torch.Tensor,
    ag_mask: torch.Tensor,
    ab_embed: torch.Tensor,
    ab_mask: torch.Tensor,
    ag_indices: torch.Tensor,
    ab_indices: torch.Tensor,
    row_indices: list[int],
    device: str = "cuda",
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run embeddings through the trained projection heads.

    Returns normalized projected features for agents and targets.
    """
    model = build_model_from_config(device)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    # Create dataset
    dataset = SimpleEmbeddingDataset(
        ag_embed, ag_mask, ab_embed, ab_mask,
        ag_indices, ab_indices, row_indices,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_feat_ag = []
    all_feat_ab = []

    with torch.no_grad():
        for batch in dataloader:
            ag, ab, ag_m, ab_m, _ = batch
            ag = ag.to(device)
            ab = ab.to(device)
            ag_m = ag_m.to(device)
            ab_m = ab_m.to(device)

            _, _, _, outputs = model(ag, ab, ag_m, ab_m)
            all_feat_ag.append(outputs["features_ag"].cpu())
            all_feat_ab.append(outputs["features_ab"].cpu())

    feat_ag = torch.cat(all_feat_ag, dim=0)  # (N, d_model) normalized
    feat_ab = torch.cat(all_feat_ab, dim=0)  # (N, d_model) normalized

    return feat_ag, feat_ab


def run_validation_retrieval(
    model_dir: str,
    train_data_dir: str,
    heldout_data_dir: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    n_trials: int = 100,
    pool_size: int = 512,
) -> dict:
    """Run the full pseudo-prospective validation.

    1. Load trained model
    2. Project all training DNA motifs through the model
    3. Project all held-out TF proteins through the model
    4. For each held-out TF, retrieve from the DNA pool
    5. Check if correct motif family is in top-k

    Parameters
    ----------
    model_dir : str
        Path to trained model fold directory (contains final_model*.pth)
    train_data_dir : str
        Path to training data (JASPAR 2022)
    heldout_data_dir : str
        Path to held-out data (post-2022)
    output_dir : str
        Where to save results
    device : str
        cuda or cpu
    batch_size : int
        Batch size for projection
    n_trials : int
        Number of random pool samplings
    pool_size : int
        Retrieval pool size
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print("=== Loading trained model ===")
    state_dict = load_model_from_dir(model_dir, device)

    # --- Load training data (for DNA pool) ---
    print("\n=== Loading training DNA pool ===")
    train_ab_embed = torch.load(os.path.join(train_data_dir, "ab_embed.pt"), weights_only=True)
    train_ab_mask = torch.load(os.path.join(train_data_dir, "ab_mask.pt"), weights_only=True)
    train_ab_indices = torch.load(os.path.join(train_data_dir, "ab_indices.pt"), weights_only=True)
    train_ag_embed = torch.load(os.path.join(train_data_dir, "ag_embed.pt"), weights_only=True)
    train_ag_mask = torch.load(os.path.join(train_data_dir, "ag_mask.pt"), weights_only=True)
    train_ag_indices = torch.load(os.path.join(train_data_dir, "ag_indices.pt"), weights_only=True)
    train_meta = pd.read_csv(os.path.join(train_data_dir, "metadata.csv"))
    print(f"  Training: {len(train_meta)} pairs")

    # --- Load held-out data ---
    print("\n=== Loading held-out TFs ===")
    ho_ag_embed = torch.load(os.path.join(heldout_data_dir, "ag_embed.pt"), weights_only=True)
    ho_ag_mask = torch.load(os.path.join(heldout_data_dir, "ag_mask.pt"), weights_only=True)
    ho_ag_indices = torch.load(os.path.join(heldout_data_dir, "ag_indices.pt"), weights_only=True)
    ho_ab_embed = torch.load(os.path.join(heldout_data_dir, "ab_embed.pt"), weights_only=True)
    ho_ab_mask = torch.load(os.path.join(heldout_data_dir, "ab_mask.pt"), weights_only=True)
    ho_ab_indices = torch.load(os.path.join(heldout_data_dir, "ab_indices.pt"), weights_only=True)
    ho_meta = pd.read_csv(os.path.join(heldout_data_dir, "metadata.csv"))
    print(f"  Held-out: {len(ho_meta)} pairs")
    print(f"  Truly novel TFs: {ho_meta['is_truly_novel'].sum()} pairs")

    # --- Project training data through model ---
    print("\n=== Projecting training data ===")
    train_feat_ag, train_feat_ab = embed_through_model(
        state_dict, train_ag_embed, train_ag_mask, train_ab_embed, train_ab_mask,
        train_ag_indices, train_ab_indices,
        list(range(len(train_meta))), device, batch_size,
    )
    print(f"  Training features: AG {train_feat_ag.shape}, AB {train_feat_ab.shape}")

    # --- Project held-out data through model ---
    print("\n=== Projecting held-out data ===")
    ho_feat_ag, ho_feat_ab = embed_through_model(
        state_dict, ho_ag_embed, ho_ag_mask, ho_ab_embed, ho_ab_mask,
        ho_ag_indices, ho_ab_indices,
        list(range(len(ho_meta))), device, batch_size,
    )
    print(f"  Held-out features: AG {ho_feat_ag.shape}, AB {ho_feat_ab.shape}")

    # --- Build the DNA pool from training data ---
    # Get unique DNA motifs from training (one per consensus sequence)
    # We use the first occurrence of each unique DNA sequence
    train_meta_reset = train_meta.reset_index(drop=True)
    unique_dna_idx = train_meta_reset.drop_duplicates(subset=["consensus_dna"]).index.tolist()
    dna_pool_features = train_feat_ab[unique_dna_idx]  # (N_unique_dna, d_model)
    dna_pool_meta = train_meta_reset.iloc[unique_dna_idx].reset_index(drop=True)
    print(f"\n  DNA pool: {len(dna_pool_features)} unique motifs from training set")

    # --- Retrieval: for each held-out TF, rank DNA motifs ---
    print("\n=== Running validation retrieval ===")

    # Get unique held-out TFs (one per TF name, using consensus sequence row)
    ho_meta_reset = ho_meta.reset_index(drop=True)
    unique_ho_tfs = ho_meta_reset.drop_duplicates(subset=["tf_name"])
    unique_ho_idx = unique_ho_tfs.index.tolist()

    query_features = ho_feat_ag[unique_ho_idx]  # (N_query, d_model)
    query_meta = ho_meta_reset.iloc[unique_ho_idx].reset_index(drop=True)

    print(f"  Queries: {len(query_features)} unique held-out TFs")
    print(f"  Truly novel: {query_meta['is_truly_novel'].sum()}")

    # Compute similarity: query TFs x DNA pool
    sim_matrix = (query_features @ dna_pool_features.t()).numpy()  # (N_query, N_dna_pool)

    # For each query, check if the correct TF family's motif is retrieved
    results_per_tf = []
    rng = np.random.RandomState(42)

    for qi in range(len(query_meta)):
        tf_name = query_meta.iloc[qi]["tf_name"]
        tf_family = query_meta.iloc[qi]["tf_family"]
        is_novel = bool(query_meta.iloc[qi]["is_truly_novel"])

        # Find which DNA pool entries belong to the same TF family
        family_mask = (dna_pool_meta["tf_family"] == tf_family).values
        n_family_in_pool = family_mask.sum()

        # If no family members in training pool, skip (can't evaluate)
        if n_family_in_pool == 0:
            results_per_tf.append({
                "tf_name": tf_name,
                "tf_family": tf_family,
                "is_truly_novel": is_novel,
                "n_family_in_pool": 0,
                "rank_best_family": -1,
                "in_top_1": False,
                "in_top_5": False,
                "in_top_10": False,
                "skipped": True,
            })
            continue

        # Exact match: does the exact same TF name exist in training?
        exact_mask = (dna_pool_meta["tf_name"] == tf_name).values
        n_exact_in_pool = exact_mask.sum()

        # Rank all DNA motifs by similarity
        sims = sim_matrix[qi]
        ranked_indices = np.argsort(-sims)

        # Find rank of best family match
        family_ranks = np.where(family_mask[ranked_indices])[0]
        best_family_rank = int(family_ranks[0]) + 1 if len(family_ranks) > 0 else -1

        # Find rank of exact TF match (if exists)
        if n_exact_in_pool > 0:
            exact_ranks = np.where(exact_mask[ranked_indices])[0]
            best_exact_rank = int(exact_ranks[0]) + 1 if len(exact_ranks) > 0 else -1
        else:
            best_exact_rank = -1

        results_per_tf.append({
            "tf_name": tf_name,
            "tf_family": tf_family,
            "is_truly_novel": is_novel,
            "n_family_in_pool": int(n_family_in_pool),
            "n_exact_in_pool": int(n_exact_in_pool),
            "rank_best_family": best_family_rank,
            "rank_best_exact": best_exact_rank,
            "in_top_1_family": best_family_rank <= 1,
            "in_top_5_family": best_family_rank <= 5,
            "in_top_10_family": best_family_rank <= 10,
            "in_top_1_exact": best_exact_rank <= 1 if best_exact_rank > 0 else False,
            "in_top_5_exact": best_exact_rank <= 5 if best_exact_rank > 0 else False,
            "in_top_10_exact": best_exact_rank <= 10 if best_exact_rank > 0 else False,
            "skipped": False,
        })

    results_df = pd.DataFrame(results_per_tf)

    # --- Compute summary statistics ---
    valid = results_df[~results_df["skipped"]]
    novel = valid[valid["is_truly_novel"]]
    updated = valid[~valid["is_truly_novel"]]

    print(f"\n=== VALIDATION RESULTS ===")
    print(f"Total held-out TFs: {len(results_df)}")
    print(f"Evaluable (family in pool): {len(valid)}")
    print(f"Skipped (family not in pool): {results_df['skipped'].sum()}")
    print()

    for label, subset in [("All held-out", valid), ("Truly novel", novel), ("Updated versions", updated)]:
        if len(subset) == 0:
            continue
        print(f"--- {label} ({len(subset)} TFs) ---")
        print(f"  Family match in top-1:  {subset['in_top_1_family'].mean()*100:.1f}%")
        print(f"  Family match in top-5:  {subset['in_top_5_family'].mean()*100:.1f}%")
        print(f"  Family match in top-10: {subset['in_top_10_family'].mean()*100:.1f}%")
        print(f"  Median family rank:     {subset['rank_best_family'].median():.0f}")
        print(f"  Mean family rank:       {subset['rank_best_family'].mean():.1f}")
        if "in_top_1_exact" in subset.columns:
            has_exact = subset[subset["n_exact_in_pool"] > 0]
            if len(has_exact) > 0:
                print(f"  Exact match in top-1:   {has_exact['in_top_1_exact'].mean()*100:.1f}% ({len(has_exact)} with exact match in pool)")
                print(f"  Exact match in top-5:   {has_exact['in_top_5_exact'].mean()*100:.1f}%")
                print(f"  Exact match in top-10:  {has_exact['in_top_10_exact'].mean()*100:.1f}%")
        print()

    # Save results
    results_df.to_csv(out / "validation_results.csv", index=False)

    summary = {
        "total_heldout_tfs": len(results_df),
        "evaluable": len(valid),
        "skipped": int(results_df["skipped"].sum()),
        "all_heldout": {
            "n": len(valid),
            "top_1_family_pct": float(valid["in_top_1_family"].mean() * 100),
            "top_5_family_pct": float(valid["in_top_5_family"].mean() * 100),
            "top_10_family_pct": float(valid["in_top_10_family"].mean() * 100),
            "median_family_rank": float(valid["rank_best_family"].median()),
        },
        "truly_novel": {
            "n": len(novel),
            "top_1_family_pct": float(novel["in_top_1_family"].mean() * 100) if len(novel) > 0 else 0,
            "top_5_family_pct": float(novel["in_top_5_family"].mean() * 100) if len(novel) > 0 else 0,
            "top_10_family_pct": float(novel["in_top_10_family"].mean() * 100) if len(novel) > 0 else 0,
            "median_family_rank": float(novel["rank_best_family"].median()) if len(novel) > 0 else 0,
        },
        "dna_pool_size": len(dna_pool_features),
        "model_dir": model_dir,
    }

    with open(out / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {out}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tSFM pseudo-prospective validation retrieval")
    parser.add_argument("--model_dir", required=True, help="Trained model fold directory")
    parser.add_argument("--train_data_dir", required=True, help="Training data directory")
    parser.add_argument("--heldout_data_dir", required=True, help="Held-out data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    run_validation_retrieval(
        model_dir=args.model_dir,
        train_data_dir=args.train_data_dir,
        heldout_data_dir=args.heldout_data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
