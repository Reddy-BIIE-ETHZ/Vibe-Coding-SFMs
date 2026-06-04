"""
Validation Experiment Option C: Off-target prediction from DAVIS selectivity data.

For each drug in DAVIS, we know its full selectivity profile (Kd against 379 kinases).
This script:
1. Loads the trained dtSFM model
2. Ranks ALL unique kinases by cosine similarity for each drug
3. Compares the model's ranking to the experimental DAVIS selectivity data
4. Specifically highlights "off-target" predictions: cases where the model
   correctly identifies a kinase as a binder that is NOT the drug's primary target

This directly tests the drug repurposing / safety screening use case:
"Given a drug designed for kinase X, can the model predict it also binds kinase Y?"

Usage:
    python -m calm.encoder.validation_offtarget \
        --checkpoint /path/to/best_model.pth \
        --data_dir data/dtsfm \
        --davis_csv data/dtsfm/davis_raw.csv \
        --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_validation(
    checkpoint_path: str,
    data_dir: str,
    davis_csv: str,
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(data_dir)

    print("=" * 60)
    print("dtSFM Validation: Off-Target Prediction (Option C)")
    print("=" * 60)

    # --- Load and project unique embeddings ---
    print("\nStep 1: Loading embeddings and projecting...")
    ag_embed = torch.load(data / "ag_embed.pt", map_location="cpu", weights_only=True)
    ab_embed = torch.load(data / "ab_embed.pt", map_location="cpu", weights_only=True)
    ag_mask = torch.load(data / "ag_mask.pt", map_location="cpu", weights_only=True)
    ab_mask = torch.load(data / "ab_mask.pt", map_location="cpu", weights_only=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    enc_ag = nn.Sequential(nn.Linear(768, 2048, bias=False), nn.ReLU(), nn.Linear(2048, 512, bias=False))
    enc_ab = nn.Sequential(nn.Linear(1280, 2048, bias=False), nn.ReLU(), nn.Linear(2048, 512, bias=False))
    enc_ag.load_state_dict({k.replace("encoder_ag.", ""): v for k, v in ckpt.items() if k.startswith("encoder_ag.")})
    enc_ab.load_state_dict({k.replace("encoder_ab.", ""): v for k, v in ckpt.items() if k.startswith("encoder_ab.")})
    enc_ag.eval()
    enc_ab.eval()

    temperature = 1.0 / math.exp(ckpt["logit_scale"].item())
    print(f"  Learned temperature: {temperature:.4f}")

    with torch.no_grad():
        ag_mf = ag_mask.unsqueeze(-1).float()
        ag_pooled = (ag_embed * ag_mf).sum(1) / ag_mf.sum(1).clamp(min=1)
        all_ag = F.normalize(enc_ag(ag_pooled), dim=-1)

        ab_mf = ab_mask.unsqueeze(-1).float()
        ab_pooled = (ab_embed * ab_mf).sum(1) / ab_mf.sum(1).clamp(min=1)
        all_ab = F.normalize(enc_ab(ab_pooled), dim=-1)

    print(f"  Drug projections: {all_ag.shape}")
    print(f"  Protein projections: {all_ab.shape}")

    # Drug→Target similarity matrix (all unique drugs × all unique targets)
    sim_matrix = (all_ag @ all_ab.T).numpy()  # (2037, 429)
    print(f"  Similarity matrix: {sim_matrix.shape}")

    # --- Load DAVIS data ---
    print("\nStep 2: Loading DAVIS selectivity data...")
    davis = pd.read_csv(davis_csv)
    metadata = pd.read_csv(data / "metadata.csv")

    # Build mappings from SMILES/sequence to embedding index
    unique_drugs = metadata["drug_smiles"].drop_duplicates().tolist()
    unique_targets = metadata["target_seq"].drop_duplicates().tolist()
    drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
    target_to_idx = {s: i for i, s in enumerate(unique_targets)}

    # Build DAVIS drug profiles: for each drug, which kinases bind and at what Kd?
    drug_profiles = defaultdict(dict)
    drug_names = {}
    target_names = {}

    for _, row in davis.iterrows():
        smiles = row["Drug"]
        seq = row["Target"]
        kd = row["Y"]
        drug_id = str(row["Drug_ID"])
        target_id = row["Target_ID"]

        if smiles in drug_to_idx and seq in target_to_idx:
            di = drug_to_idx[smiles]
            ti = target_to_idx[seq]
            drug_profiles[di][ti] = kd
            drug_names[di] = drug_id
            target_names[ti] = target_id

    print(f"  DAVIS drugs mapped: {len(drug_profiles)}")

    # --- Identify drugs with multiple targets (off-target candidates) ---
    print("\nStep 3: Identifying multi-target drugs...")

    results_per_drug = []

    for di in sorted(drug_profiles.keys()):
        profile = drug_profiles[di]
        name = drug_names.get(di, str(di))

        # Classify targets by binding strength
        strong_binders = {ti: kd for ti, kd in profile.items() if kd < 100}     # Kd < 100 nM
        moderate_binders = {ti: kd for ti, kd in profile.items() if 100 <= kd < 1000}  # 100-1000 nM
        weak_binders = {ti: kd for ti, kd in profile.items() if 1000 <= kd < 10000}    # 1-10 µM
        non_binders = {ti: kd for ti, kd in profile.items() if kd >= 10000}

        all_binders = {**strong_binders, **moderate_binders, **weak_binders}

        if len(all_binders) < 2:
            continue  # Need at least 2 targets to test off-target prediction

        # Primary target = strongest binder
        primary_ti = min(all_binders, key=all_binders.get)
        primary_name = target_names.get(primary_ti, str(primary_ti))
        primary_kd = all_binders[primary_ti]

        # Off-targets = all other binders
        off_targets = {ti: kd for ti, kd in all_binders.items() if ti != primary_ti}

        # Get model's ranking of ALL targets for this drug
        drug_sims = sim_matrix[di]  # (429,)
        ranked_targets = np.argsort(-drug_sims)  # highest similarity first
        target_ranks = {ti: rank for rank, ti in enumerate(ranked_targets)}

        # Where does the primary target rank?
        primary_rank = target_ranks.get(primary_ti, -1) + 1  # 1-indexed

        # Where do off-targets rank?
        off_target_results = []
        for ti, kd in sorted(off_targets.items(), key=lambda x: x[1]):
            rank = target_ranks.get(ti, -1) + 1
            off_target_results.append({
                "target_name": target_names.get(ti, str(ti)),
                "kd_nm": kd,
                "model_rank": rank,
                "cos_sim": float(drug_sims[ti]) if ti < len(drug_sims) else 0.0,
                "in_top_10": rank <= 10,
                "in_top_50": rank <= 50,
            })

        results_per_drug.append({
            "drug_id": name,
            "drug_idx": di,
            "n_total_targets": len(all_binders),
            "n_strong": len(strong_binders),
            "n_moderate": len(moderate_binders),
            "n_weak": len(weak_binders),
            "n_non_binders": len(non_binders),
            "primary_target": primary_name,
            "primary_kd_nm": primary_kd,
            "primary_rank": primary_rank,
            "off_targets": off_target_results,
        })

    print(f"  Multi-target drugs: {len(results_per_drug)}")

    # --- Aggregate off-target prediction metrics ---
    print("\nStep 4: Computing off-target prediction metrics...")

    all_off_target_ranks = []
    all_primary_ranks = []
    n_off_in_top10 = 0
    n_off_in_top50 = 0
    n_off_total = 0

    for drug in results_per_drug:
        all_primary_ranks.append(drug["primary_rank"])
        for ot in drug["off_targets"]:
            all_off_target_ranks.append(ot["model_rank"])
            n_off_total += 1
            if ot["in_top_10"]:
                n_off_in_top10 += 1
            if ot["in_top_50"]:
                n_off_in_top50 += 1

    n_targets_total = sim_matrix.shape[1]

    print(f"\n  Off-target prediction results:")
    print(f"    Total off-target pairs evaluated: {n_off_total}")
    print(f"    Total unique targets in pool: {n_targets_total}")
    print(f"    Random chance top-10: {10/n_targets_total*100:.1f}%")
    print(f"    Random chance top-50: {50/n_targets_total*100:.1f}%")
    print(f"")
    print(f"    Primary target median rank: {np.median(all_primary_ranks):.0f} / {n_targets_total}")
    print(f"    Primary target mean rank:   {np.mean(all_primary_ranks):.1f} / {n_targets_total}")
    print(f"    Primary target in top-10:   {sum(1 for r in all_primary_ranks if r <= 10)}/{len(all_primary_ranks)} ({sum(1 for r in all_primary_ranks if r <= 10)/len(all_primary_ranks)*100:.1f}%)")
    print(f"")
    print(f"    Off-target median rank:     {np.median(all_off_target_ranks):.0f} / {n_targets_total}")
    print(f"    Off-target mean rank:       {np.mean(all_off_target_ranks):.1f} / {n_targets_total}")
    print(f"    Off-targets in top-10:      {n_off_in_top10}/{n_off_total} ({n_off_in_top10/n_off_total*100:.1f}%)")
    print(f"    Off-targets in top-50:      {n_off_in_top50}/{n_off_total} ({n_off_in_top50/n_off_total*100:.1f}%)")

    # --- By binding strength ---
    print(f"\n  Off-target prediction by binding strength:")
    for label, kd_lo, kd_hi in [("Strong (<100 nM)", 0, 100), ("Moderate (100-1000 nM)", 100, 1000), ("Weak (1-10 µM)", 1000, 10000)]:
        ranks = []
        for drug in results_per_drug:
            for ot in drug["off_targets"]:
                if kd_lo <= ot["kd_nm"] < kd_hi:
                    ranks.append(ot["model_rank"])
        if ranks:
            top10 = sum(1 for r in ranks if r <= 10)
            top50 = sum(1 for r in ranks if r <= 50)
            print(f"    {label}: n={len(ranks)}, median_rank={np.median(ranks):.0f}, "
                  f"top-10={top10}/{len(ranks)} ({top10/len(ranks)*100:.1f}%), "
                  f"top-50={top50}/{len(ranks)} ({top50/len(ranks)*100:.1f}%)")

    # --- Top examples: drugs with best off-target prediction ---
    print(f"\n  Top 10 drugs with best off-target prediction:")
    drugs_sorted = sorted(results_per_drug,
                         key=lambda d: np.median([ot["model_rank"] for ot in d["off_targets"]]))

    for i, drug in enumerate(drugs_sorted[:10]):
        ot_ranks = [ot["model_rank"] for ot in drug["off_targets"]]
        ot_names = [ot["target_name"] for ot in drug["off_targets"][:3]]
        print(f"    {i+1}. Drug {drug['drug_id']}: {drug['n_total_targets']} targets, "
              f"primary={drug['primary_target']} (rank {drug['primary_rank']}), "
              f"off-target median rank={np.median(ot_ranks):.0f}, "
              f"off-targets: {', '.join(ot_names)}")

    # --- Save results ---
    summary = {
        "n_multi_target_drugs": len(results_per_drug),
        "n_off_target_pairs": n_off_total,
        "n_unique_targets": n_targets_total,
        "primary_target": {
            "median_rank": float(np.median(all_primary_ranks)),
            "mean_rank": float(np.mean(all_primary_ranks)),
            "top_10_pct": sum(1 for r in all_primary_ranks if r <= 10) / len(all_primary_ranks) * 100,
        },
        "off_target": {
            "median_rank": float(np.median(all_off_target_ranks)),
            "mean_rank": float(np.mean(all_off_target_ranks)),
            "top_10_pct": n_off_in_top10 / n_off_total * 100,
            "top_50_pct": n_off_in_top50 / n_off_total * 100,
            "random_top_10_pct": 10 / n_targets_total * 100,
            "random_top_50_pct": 50 / n_targets_total * 100,
        },
        "temperature": float(temperature),
    }

    with open(out / "offtarget_validation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-drug results
    flat_results = []
    for drug in results_per_drug:
        for ot in drug["off_targets"]:
            flat_results.append({
                "drug_id": drug["drug_id"],
                "primary_target": drug["primary_target"],
                "primary_kd_nm": drug["primary_kd_nm"],
                "primary_rank": drug["primary_rank"],
                "off_target": ot["target_name"],
                "off_target_kd_nm": ot["kd_nm"],
                "off_target_rank": ot["model_rank"],
                "off_target_cos_sim": ot["cos_sim"],
                "in_top_10": ot["in_top_10"],
                "in_top_50": ot["in_top_50"],
            })
    pd.DataFrame(flat_results).to_csv(out / "offtarget_predictions.csv", index=False)

    print(f"\n  Results saved to {out}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dtSFM off-target validation (Option C)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--davis_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    run_validation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        davis_csv=args.davis_csv,
        output_dir=args.output_dir,
    )
