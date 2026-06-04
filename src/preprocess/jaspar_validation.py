"""
Preprocessing for tSFM VALIDATION experiment: temporal split.

Splits JASPAR profiles into:
  - TRAINING: profiles present in JASPAR 2022 release
  - HELD-OUT: profiles added after 2022 (in 2024 and 2026 releases)

The held-out set contains 177 truly novel TFs (no prior profile in 2022).
This enables a pseudo-prospective test: "Could a model trained on 2022 data
predict the binding specificities of TFs characterized after 2022?"

Uses the same preprocessing as jaspar.py (10x PWM sampling, ESM-2 + DNABERT-2
embedding) but saves two separate datasets.

Usage:
    # Step 1: Build metadata + tag profiles (Mac Terminal, no GPU)
    python -m calm.preprocess.jaspar_validation \
        --jaspar_dir data/jaspar \
        --output_dir data/jaspar_validation \
        --samples_per_pwm 10 \
        --skip_embedding

    # Step 2: Embed on Euler (GPU)
    python -m calm.preprocess.jaspar_validation \
        --jaspar_dir data/jaspar \
        --output_dir data/jaspar_validation \
        --samples_per_pwm 10 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

try:
    import torch
except ImportError:
    torch = None

from calm.preprocess.jaspar import (
    build_metadata,
    embed_dna_dnabert2,
    embed_proteins_esm2,
    pfm_to_consensus,
)


def fetch_jaspar_2022_matrix_ids() -> set[str]:
    """Fetch all matrix IDs from the JASPAR 2022 release via API.

    The JASPAR REST API supports a release= parameter that returns
    profiles as they existed in a specific release.

    Returns
    -------
    set[str]
        Set of matrix IDs (e.g., {'MA0634.1', 'MA0007.2', ...})
    """
    ids = set()
    page = 1
    while True:
        url = (
            f"https://jaspar.elixir.no/api/v1/matrix/?format=json"
            f"&collection=CORE&tax_group=vertebrates"
            f"&release=2022&page_size=100&page={page}"
        )
        print(f"  Fetching JASPAR 2022 page {page}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for entry in data["results"]:
            ids.add(entry["matrix_id"])
        if data["next"] is None:
            break
        page += 1
        time.sleep(0.2)
    print(f"  JASPAR 2022 release: {len(ids)} CORE vertebrate profiles")
    return ids


def run_validation_preprocessing(
    jaspar_dir: str,
    output_dir: str,
    samples_per_pwm: int = 10,
    device: str = "cpu",
    esm_batch_size: int = 2,
    dna_batch_size: int = 8,
    skip_embedding: bool = False,
) -> None:
    """Build validation dataset: 2022 training + post-2022 held-out.

    Parameters
    ----------
    jaspar_dir : str
        Directory with cached jaspar_profiles.json and profiles_with_sequences.json
        (from the original jaspar.py preprocessing run).
    output_dir : str
        Where to save the validation dataset.
    samples_per_pwm : int
        DNA sequences per PWM (10 = same as main training).
    device : str
        Device for embedding models.
    esm_batch_size : int
        Batch size for ESM-2.
    dna_batch_size : int
        Batch size for DNABERT-2.
    skip_embedding : bool
        If True, only build metadata (no GPU needed).
    """
    jaspar_path = Path(jaspar_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load cached profiles ---
    print("Step 1: Loading cached JASPAR profiles...")
    with open(jaspar_path / "profiles_with_sequences.json") as f:
        profiles = json.load(f)
    print(f"  Loaded {len(profiles)} profiles with protein sequences")

    # Compute consensus DNA for all
    for p in profiles:
        p["consensus_dna"] = pfm_to_consensus(p["pfm"])

    # --- Step 2: Fetch JASPAR 2022 IDs and tag profiles ---
    cache_file = out / "jaspar_2022_ids.json"
    if cache_file.exists():
        print("Step 2: Loading cached JASPAR 2022 IDs...")
        with open(cache_file) as f:
            ids_2022 = set(json.load(f))
    else:
        print("Step 2: Fetching JASPAR 2022 release IDs from API...")
        ids_2022 = fetch_jaspar_2022_matrix_ids()
        with open(cache_file, "w") as f:
            json.dump(sorted(ids_2022), f)

    # Tag each profile
    train_profiles = []
    heldout_profiles = []
    for p in profiles:
        if p["matrix_id"] in ids_2022:
            p["release_group"] = "train_2022"
            train_profiles.append(p)
        else:
            p["release_group"] = "heldout_post2022"
            heldout_profiles.append(p)

    # Stats
    train_tfs = {p["name"] for p in train_profiles}
    heldout_tfs = {p["name"] for p in heldout_profiles}
    truly_novel = heldout_tfs - train_tfs

    print(f"\n  === Temporal Split ===")
    print(f"  Training (JASPAR 2022):     {len(train_profiles)} profiles, {len(train_tfs)} unique TFs")
    print(f"  Held-out (post-2022):       {len(heldout_profiles)} profiles, {len(heldout_tfs)} unique TFs")
    print(f"  Truly novel TFs:            {len(truly_novel)} (not in 2022 at all)")
    print(f"  Updated TFs (new versions): {len(heldout_tfs) - len(truly_novel)}")
    print()

    # --- Step 3: Build metadata for BOTH sets ---
    # Training metadata (for actual training)
    print(f"Step 3a: Building training metadata ({samples_per_pwm} samples per PWM)...")
    df_train = build_metadata(train_profiles, samples_per_pwm=samples_per_pwm)
    df_train["release_group"] = "train_2022"

    # Held-out metadata (for validation retrieval)
    print(f"Step 3b: Building held-out metadata ({samples_per_pwm} samples per PWM)...")
    df_heldout = build_metadata(heldout_profiles, samples_per_pwm=samples_per_pwm)
    df_heldout["release_group"] = "heldout_post2022"

    # Tag truly novel TFs
    df_heldout["is_truly_novel"] = df_heldout["tf_name"].isin(truly_novel)

    # Save metadata
    train_dir = out / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(train_dir / "metadata.csv", index=False)

    heldout_dir = out / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    df_heldout.to_csv(heldout_dir / "metadata.csv", index=False)

    # Also save a combined metadata for reference
    df_combined = pd.concat([df_train, df_heldout], ignore_index=True)
    df_combined.to_csv(out / "metadata_combined.csv", index=False)

    # Save summary
    summary = {
        "train_profiles": len(train_profiles),
        "train_pairs": len(df_train),
        "train_unique_tfs": int(df_train["ag_id"].nunique()),
        "heldout_profiles": len(heldout_profiles),
        "heldout_pairs": len(df_heldout),
        "heldout_unique_tfs": int(df_heldout["ag_id"].nunique()),
        "truly_novel_tfs": len(truly_novel),
        "truly_novel_tf_names": sorted(truly_novel),
        "samples_per_pwm": samples_per_pwm,
    }
    with open(out / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Training:  {len(df_train)} pairs, {df_train['ag_id'].nunique()} TFs")
    print(f"  Held-out:  {len(df_heldout)} pairs, {df_heldout['ag_id'].nunique()} TFs")
    print(f"  Truly novel TFs in held-out: {df_heldout['is_truly_novel'].sum()} pairs "
          f"({len(truly_novel)} unique TFs)")

    # --- Step 4: Embed unique sequences ---
    if not skip_embedding:
        for label, df, save_dir in [("Training", df_train, train_dir),
                                     ("Held-out", df_heldout, heldout_dir)]:
            # Unique proteins
            unique_proteins = df["protein_seq"].drop_duplicates().tolist()
            prot_to_idx = {seq: i for i, seq in enumerate(unique_proteins)}
            ag_indices = torch.tensor([prot_to_idx[s] for s in df["protein_seq"]], dtype=torch.long)

            print(f"\nStep 4: Embedding {label} — {len(unique_proteins)} unique proteins...")
            ag_embed, ag_mask = embed_proteins_esm2(
                unique_proteins, device=device, batch_size=esm_batch_size
            )
            torch.save(ag_embed, save_dir / "ag_embed.pt")
            torch.save(ag_mask, save_dir / "ag_mask.pt")
            torch.save(ag_indices, save_dir / "ag_indices.pt")
            print(f"  Saved ag_embed.pt {ag_embed.shape}")

            # Unique DNA
            unique_dna = df["consensus_dna"].drop_duplicates().tolist()
            dna_to_idx = {seq: i for i, seq in enumerate(unique_dna)}
            ab_indices = torch.tensor([dna_to_idx[s] for s in df["consensus_dna"]], dtype=torch.long)

            print(f"  Embedding {label} — {len(unique_dna)} unique DNA sequences...")
            ab_embed, ab_mask = embed_dna_dnabert2(
                unique_dna, device=device, batch_size=dna_batch_size
            )
            torch.save(ab_embed, save_dir / "ab_embed.pt")
            torch.save(ab_mask, save_dir / "ab_mask.pt")
            torch.save(ab_indices, save_dir / "ab_indices.pt")
            print(f"  Saved ab_embed.pt {ab_embed.shape}")

            # Free memory between sets
            del ag_embed, ag_mask, ab_embed, ab_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("\nStep 4: Skipping embedding (--skip_embedding)")

    print(f"\nDone! Validation dataset saved to {out}")
    print(f"  {train_dir}/  — training data (JASPAR 2022)")
    print(f"  {heldout_dir}/ — held-out data (post-2022)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess JASPAR validation dataset (temporal split)"
    )
    parser.add_argument("--jaspar_dir", required=True,
                        help="Directory with cached jaspar_profiles.json")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for validation dataset")
    parser.add_argument("--samples_per_pwm", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--esm_batch_size", type=int, default=2)
    parser.add_argument("--dna_batch_size", type=int, default=8)
    parser.add_argument("--skip_embedding", action="store_true")
    args = parser.parse_args()

    run_validation_preprocessing(
        jaspar_dir=args.jaspar_dir,
        output_dir=args.output_dir,
        samples_per_pwm=args.samples_per_pwm,
        device=args.device,
        esm_batch_size=args.esm_batch_size,
        dna_batch_size=args.dna_batch_size,
        skip_embedding=args.skip_embedding,
    )
