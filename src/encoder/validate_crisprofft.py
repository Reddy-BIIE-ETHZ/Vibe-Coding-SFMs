"""CRISPRoffT pseudo-prospective validation for crisprSFM.

Embeds held-out CRISPRoffT guide+off-target sequences, projects them through
a trained crisprSFM model, and evaluates:

  Tier 1 — Standard retrieval: Can the model find TRUE off-targets among
           random negatives from other guides? (Hamming baseline is strong here.)

  Tier 2 — Hard-negative discrimination: Among sites with similar mismatch
           counts to the SAME guide, can the model rank TRUE above FALSE?
           (Hamming baseline is near-chance here. This is the headline result.)

Usage (Euler GPU node):
    python -m calm.encoder.validate_crisprofft \
        --validation_csv /path/to/crisprofft_validation_full.csv \
        --model_checkpoint /path/to/final_model_epoch_99.pth \
        --model_config_dir /path/to/CALM-0.1.0/configs \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def hamming_distance(s1: str, s2: str) -> int:
    s1, s2 = s1.upper(), s2.upper()
    minlen = min(len(s1), len(s2))
    return sum(c1 != c2 for c1, c2 in zip(s1[:minlen], s2[:minlen])) + abs(len(s1) - len(s2))


def embed_sequences_dnabert2(sequences: list[str], device: str = "cpu",
                              batch_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed DNA sequences with DNABERT-2, returning (embeddings, masks)."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)
    model.eval()

    all_embeddings = []
    all_masks = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        if isinstance(outputs, tuple):
            emb = outputs[0].cpu()
        else:
            emb = outputs.last_hidden_state.cpu()
        mask = inputs["attention_mask"].cpu().bool()
        all_embeddings.append(emb)
        all_masks.append(mask)

        if (i // batch_size + 1) % 10 == 0:
            print(f"    Embedded {i+len(batch)}/{len(sequences)} sequences")

    # Pad to same length
    max_len = max(e.shape[1] for e in all_embeddings)
    padded = torch.zeros(len(sequences), max_len, all_embeddings[0].shape[2])
    masks = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    idx = 0
    for emb, msk in zip(all_embeddings, all_masks):
        n, L = emb.shape[0], emb.shape[1]
        padded[idx:idx+n, :L] = emb
        masks[idx:idx+n, :L] = msk
        idx += n

    print(f"    Embeddings: {padded.shape}, Masks: {masks.shape}")
    return padded, masks


def load_model_and_project(
    checkpoint_path: str,
    ag_embed: torch.Tensor, ag_mask: torch.Tensor,
    ab_embed: torch.Tensor, ab_mask: torch.Tensor,
    config_dir: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load trained model and project embeddings to shared space.

    Returns L2-normalized feature vectors for agents and targets.
    """
    from calm.encoder.model import CALMEncoder

    # Load model config
    model_cfg = OmegaConf.load(Path(config_dir) / "model" / "encoder" / "crispr_ffn.yaml")

    model = CALMEncoder(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle checkpoint format (may be wrapped in 'model_state_dict')
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Project in batches to avoid OOM
    batch_size = 256
    ag_features_list = []
    ab_features_list = []

    with torch.no_grad():
        for i in range(0, len(ag_embed), batch_size):
            ag_batch = ag_embed[i:i+batch_size].to(device)
            ab_batch = ab_embed[i:i+batch_size].to(device)
            ag_m = ag_mask[i:i+batch_size].to(device)
            ab_m = ab_mask[i:i+batch_size].to(device)

            # FFN projection + mean pooling (same as model.forward)
            ag_proj = model.encoder_ag(ag_batch)
            ab_proj = model.encoder_ab(ab_batch)

            ag_proj = ag_proj * ag_m.unsqueeze(-1)
            ab_proj = ab_proj * ab_m.unsqueeze(-1)

            # Masked mean pooling
            ag_feat = (ag_proj * ag_m.unsqueeze(-1)).sum(1) / ag_m.sum(1, keepdim=True).clamp(min=1)
            ab_feat = (ab_proj * ab_m.unsqueeze(-1)).sum(1) / ab_m.sum(1, keepdim=True).clamp(min=1)

            # L2 normalize
            ag_feat = F.normalize(ag_feat, p=2, dim=-1)
            ab_feat = F.normalize(ab_feat, p=2, dim=-1)

            ag_features_list.append(ag_feat.cpu())
            ab_features_list.append(ab_feat.cpu())

    ag_features = torch.cat(ag_features_list, dim=0)
    ab_features = torch.cat(ab_features_list, dim=0)

    print(f"    Projected features: ag={ag_features.shape}, ab={ab_features.shape}")
    return ag_features, ab_features


def tier1_evaluation(
    df: pd.DataFrame,
    ag_features: torch.Tensor,
    ab_features: torch.Tensor,
    pool_size: int = 512,
    n_trials: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """Tier 1: Standard retrieval — TRUE off-targets vs random negatives."""
    rng = np.random.RandomState(seed)
    true_mask = df["validation"].values == "TRUE"
    true_indices = np.where(true_mask)[0]

    cosine_sim = (ag_features @ ab_features.t()).numpy()

    results = {}
    for direction in ["ag2ab", "ab2ag"]:
        sim = cosine_sim if direction == "ag2ab" else cosine_sim.T

        hits = {1: 0, 5: 0, 10: 0}
        n_queries = 0

        for trial in range(n_trials):
            for qi in true_indices:
                # True positive is at position qi
                all_others = [j for j in range(len(df)) if j != qi]
                if len(all_others) >= pool_size - 1:
                    neg = rng.choice(all_others, size=pool_size - 1, replace=False)
                else:
                    neg = np.array(all_others)
                pool = np.concatenate([[qi], neg])
                pool_sims = sim[qi, pool]
                ranked = np.argsort(-pool_sims)

                for k in [1, 5, 10]:
                    if ranked[0] < 1 and k >= 1:  # True pos is index 0 in pool
                        pass
                    # Check if position 0 (true positive) is in top k
                    if 0 in ranked[:k]:
                        hits[k] += 1
                n_queries += 1

        for k in [1, 5, 10]:
            results[f"R@{k}_{direction}"] = hits[k] / n_queries * 100.0

    return results


def tier2_evaluation(
    df: pd.DataFrame,
    ag_features: torch.Tensor,
    ab_features: torch.Tensor,
    n_trials: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """Tier 2: Hard-negative discrimination — TRUE vs FALSE for same guide.

    For each guide that has both TRUE and FALSE off-targets, rank them by
    SFM similarity. Can the model rank TRUE above FALSE?

    Also computes Hamming distance baseline for comparison.
    """
    rng = np.random.RandomState(seed)
    cosine_sim = (ag_features @ ab_features.t()).numpy()

    # Group by guide
    guides_with_both = []
    for guide, group in df.groupby("guide_seq"):
        has_true = (group["validation"] == "TRUE").any()
        has_false = (group["validation"] == "FALSE").any()
        if has_true and has_false:
            guides_with_both.append(guide)

    print(f"    Tier 2: {len(guides_with_both)} guides with both TRUE and FALSE")

    sfm_hits = {1: 0, 5: 0, 10: 0}
    ham_hits = {1: 0, 5: 0, 10: 0}
    n_queries = 0

    for guide in guides_with_both:
        group = df[df["guide_seq"] == guide]
        true_idx = group.index[group["validation"] == "TRUE"].tolist()
        false_idx = group.index[group["validation"] == "FALSE"].tolist()

        if not true_idx or not false_idx:
            continue

        for trial in range(n_trials):
            # Pick a random TRUE off-target as the query
            qi = rng.choice(true_idx)

            # Pool: this TRUE + all FALSE for this guide
            pool = [qi] + false_idx
            if len(pool) < 2:
                continue

            # SFM ranking (guide→OT direction)
            pool_sims = cosine_sim[qi, pool]
            sfm_ranked = np.argsort(-pool_sims)

            # Hamming ranking (lower distance = better)
            guide_seq = df.loc[qi, "guide_seq"]
            pool_dists = [hamming_distance(guide_seq, df.loc[p, "offtarget_seq"]) for p in pool]
            ham_ranked = np.argsort(pool_dists)

            for k in [1, 5, 10]:
                if 0 in sfm_ranked[:k]:
                    sfm_hits[k] += 1
                if 0 in ham_ranked[:k]:
                    ham_hits[k] += 1

            n_queries += 1

    results = {}
    for k in [1, 5, 10]:
        results[f"SFM_R@{k}"] = sfm_hits[k] / n_queries * 100.0 if n_queries > 0 else 0
        results[f"Hamming_R@{k}"] = ham_hits[k] / n_queries * 100.0 if n_queries > 0 else 0
        results[f"Delta_R@{k}"] = results[f"SFM_R@{k}"] - results[f"Hamming_R@{k}"]
    results["n_guides"] = len(guides_with_both)
    results["n_queries"] = n_queries

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CRISPRoffT pseudo-prospective validation for crisprSFM")
    parser.add_argument("--validation_csv", required=True)
    parser.add_argument("--model_checkpoint", required=True,
                        help="Path to trained model checkpoint (final_model_epoch_*.pth)")
    parser.add_argument("--model_config_dir", required=True,
                        help="Path to CALM configs directory (contains model/encoder/crispr_ffn.yaml)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.validation_csv, dtype=str)
    print(f"Loaded {len(df)} validation pairs")
    print(f"  TRUE: {(df['validation']=='TRUE').sum()}, FALSE: {(df['validation']=='FALSE').sum()}")
    print(f"  Guides: {df['guide_seq'].nunique()}")

    # Step 1: Embed sequences
    print("\nStep 1: Embedding guide sequences with DNABERT-2...")
    guide_seqs = df["guide_seq"].tolist()
    ag_embed, ag_mask = embed_sequences_dnabert2(guide_seqs, args.device, args.batch_size)

    print("\nStep 2: Embedding off-target sequences with DNABERT-2...")
    ot_seqs = df["offtarget_seq"].tolist()
    ab_embed, ab_mask = embed_sequences_dnabert2(ot_seqs, args.device, args.batch_size)

    # Step 2: Load model and project
    print(f"\nStep 3: Loading model from {args.model_checkpoint}...")
    ag_features, ab_features = load_model_and_project(
        args.model_checkpoint, ag_embed, ag_mask, ab_embed, ab_mask,
        args.model_config_dir, args.device)

    # Step 3: Tier 1 evaluation
    print("\nStep 4: Tier 1 — Standard retrieval (TRUE vs random negatives)...")
    tier1 = tier1_evaluation(df, ag_features, ab_features, n_trials=args.n_trials)
    print("  SFM Results:")
    for k, v in tier1.items():
        print(f"    {k}: {v:.1f}%")

    # Tier 1 Hamming baseline
    print("\n  Hamming baseline (Tier 1)...")
    rng = np.random.RandomState(42)
    true_idx = df.index[df["validation"] == "TRUE"].tolist()
    ham_t1_hits = {1: 0, 5: 0, 10: 0}
    n_q = 0
    for trial in range(args.n_trials):
        for qi in true_idx:
            guide = df.loc[qi, "guide_seq"]
            all_others = [j for j in range(len(df)) if j != qi]
            neg = rng.choice(all_others, size=min(511, len(all_others)), replace=False)
            pool = [qi] + neg.tolist()
            dists = [hamming_distance(guide, df.loc[p, "offtarget_seq"]) for p in pool]
            ranked = np.argsort(dists)
            for k in [1, 5, 10]:
                if 0 in ranked[:k]:
                    ham_t1_hits[k] += 1
            n_q += 1
    print("  Hamming Results:")
    tier1_hamming = {}
    for k in [1, 5, 10]:
        val = ham_t1_hits[k] / n_q * 100.0
        tier1_hamming[f"Hamming_R@{k}"] = val
        print(f"    R@{k}: {val:.1f}%")

    # Step 4: Tier 2 evaluation
    print("\nStep 5: Tier 2 — Hard-negative discrimination (TRUE vs FALSE, same guide)...")
    tier2 = tier2_evaluation(df, ag_features, ab_features, n_trials=args.n_trials)
    print("  Results:")
    print(f"    Guides tested: {tier2['n_guides']}")
    print(f"    Total queries: {tier2['n_queries']}")
    for k in [1, 5, 10]:
        print(f"    R@{k}:  SFM={tier2[f'SFM_R@{k}']:.1f}%  "
              f"Hamming={tier2[f'Hamming_R@{k}']:.1f}%  "
              f"Delta={tier2[f'Delta_R@{k}']:+.1f}%")

    # Save results
    output_path = args.output or str(Path(args.model_checkpoint).parent / "crisprofft_validation_results.json")
    results = {
        "tier1_sfm": tier1,
        "tier1_hamming": tier1_hamming,
        "tier2": tier2,
        "model": str(args.model_checkpoint),
        "validation_csv": str(args.validation_csv),
        "n_trials": args.n_trials,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("CRISPROFFT VALIDATION SUMMARY")
    print("=" * 60)
    print(f"\nTier 1 — Standard retrieval (pool ~{min(512, len(df))}):")
    print(f"  SFM   R@1: {tier1['R@1_ag2ab']:.1f}% (gRNA→OT)  {tier1['R@1_ab2ag']:.1f}% (OT→gRNA)")
    print(f"  Hamming R@1: {tier1_hamming['Hamming_R@1']:.1f}%")
    print(f"\nTier 2 — Hard-negative discrimination ({tier2['n_guides']} guides):")
    for k in [1, 5, 10]:
        print(f"  R@{k}:  SFM={tier2[f'SFM_R@{k}']:.1f}%  "
              f"Hamming={tier2[f'Hamming_R@{k}']:.1f}%  "
              f"Delta={tier2[f'Delta_R@{k}']:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
